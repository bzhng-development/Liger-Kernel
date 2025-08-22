import operator

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from liger_kernel.ops.utils import calculate_settings, compare_version, ensure_contiguous

if compare_version("triton", operator.ge, "3.0.0"):
    try:
        # typical import path with dispatch available
        from triton.language.extra.libdevice import tanh
    except ModuleNotFoundError:
        # for working with NGC containers
        from triton.language.extra.cuda.libdevice import tanh
else:
    from triton.language.math import tanh


@triton.jit
def _sparse_geglu_forward_kernel(
    gate_ptr,  # *[n_rows, n_cols]
    up_ptr,    # *[n_rows, n_cols]
    out_ptr,   # *[n_rows, n_cols]
    stride,    # row stride in elements
    n_cols: tl.constexpr,
    std_mult: tl.constexpr,  # scalar float32 compile-time constant
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)

    # Row pointers
    gate_row = gate_ptr + pid * stride
    up_row = up_ptr + pid * stride
    out_row = out_ptr + pid * stride

    # First pass: reduce mean and variance in fp32
    offs = tl.arange(0, BLOCK_SIZE)
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)
    for start in range(0, n_cols, BLOCK_SIZE):
        idx = start + offs
        mask = idx < n_cols
        g = tl.load(gate_row + idx, mask=mask, other=0).to(tl.float32)
        sum_val += tl.sum(g, axis=0)
        sum_sq += tl.sum(g * g, axis=0)

    n_cols_f = tl.full((), n_cols, dtype=tl.float32)
    mean = sum_val / n_cols_f
    var = sum_sq / n_cols_f - mean * mean
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var + 1e-12)

    cutoff = mean + std * std_mult

    # Second pass: apply relu(gate - cutoff), gelu(tanh), multiply with up
    sqrt_2_over_pi = 0.7978845608028654
    for start in range(0, n_cols, BLOCK_SIZE):
        idx = start + offs
        mask = idx < n_cols
        g = tl.load(gate_row + idx, mask=mask, other=0).to(tl.float32)
        u = tl.load(up_row + idx, mask=mask, other=0)

        x = g - cutoff
        x = tl.maximum(x, 0.0)
        x_cubed = x * x * x
        t = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
        t = tanh(t)
        act = 0.5 * x * (1.0 + t)

        h = act.to(u.dtype) * u
        tl.store(out_row + idx, h, mask=mask)


def _icdf_std_multiplier(sparsity: float, device: torch.device) -> torch.Tensor:
    # icdf(N(0,1), sparsity) as float32 scalar tensor on device
    return torch.distributions.normal.Normal(0, 1).icdf(
        torch.tensor(sparsity, dtype=torch.float32, device=device)
    )


class LigerGELUSparseMulFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor, sparsity: float, approximate: str = "tanh"):
        if approximate != "tanh":
            raise ValueError(f"Only approximate='tanh' is supported. Got: {approximate}")
        if not (0.0 < float(sparsity) < 1.0):
            raise ValueError("sparsity must be in (0.0, 1.0) for sparse GEGLU")

        ori_shape = gate.shape
        n_cols = ori_shape[-1]
        gate_2d = gate.view(-1, n_cols)
        up_2d = up.view(-1, n_cols)
        out_2d = torch.empty_like(gate_2d)

        n_rows = gate_2d.shape[0]
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)

        # Compute Gaussian cutoff multiplier on host and pass as a Python float
        # to avoid Triton treating it as a pointer when passed as a tensor.
        std_mult_t = _icdf_std_multiplier(float(sparsity), gate_2d.device).to(torch.float32)
        std_mult = float(std_mult_t.item())

        _sparse_geglu_forward_kernel[(n_rows,)](
            gate_2d,
            up_2d,
            out_2d,
            out_2d.stride(-2),
            n_cols=n_cols,
            std_mult=std_mult,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

        # Save for backward (Python recompute path)
        ctx.sparsity = float(sparsity)
        ctx.approximate = approximate
        ctx.save_for_backward(gate, up)
        return out_2d.view(*ori_shape)

    @staticmethod
    @ensure_contiguous
    def backward(ctx, grad_out: torch.Tensor):
        gate, up = ctx.saved_tensors
        sparsity = ctx.sparsity
        approximate = ctx.approximate

        # Recompute with PyTorch ops to get correct gradients
        gate_r = gate.detach().requires_grad_(True)
        up_r = up.detach().requires_grad_(True)
        mean = torch.mean(gate_r, dim=-1, keepdim=True)
        std = torch.std(gate_r, dim=-1, keepdim=True, unbiased=False)
        std_mult = _icdf_std_multiplier(sparsity, gate_r.device).to(gate_r.dtype)
        cutoff = mean + std * std_mult
        gate_sparse = F.relu(gate_r - cutoff)
        act = F.gelu(gate_sparse, approximate=approximate)
        out = act * up_r

        # Compute vector-Jacobian product
        (dgate, dup) = torch.autograd.grad(
            outputs=out,
            inputs=(gate_r, up_r),
            grad_outputs=grad_out,
            allow_unused=False,
            retain_graph=False,
            create_graph=False,
        )

        return dgate, dup, None, None


def geglu_sparse_forward(gate: torch.Tensor, up: torch.Tensor, sparsity: float, approximate: str = "tanh") -> torch.Tensor:
    return LigerGELUSparseMulFunction.apply(gate, up, sparsity, approximate)
