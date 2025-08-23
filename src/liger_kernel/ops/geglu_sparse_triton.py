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
        g_raw = tl.load(gate_row + idx, mask=mask, other=0)
        g = g_raw.to(tl.float32)
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

        # For non-tanh approximation or low-precision dtypes, fall back to
        # PyTorch recompute to ensure numerical parity with the reference.
        if approximate != "tanh" or gate.dtype != torch.float32:
            # Fallback to PyTorch recompute path under autograd-enabled context
            with torch.enable_grad():
                gate_r = gate.detach().requires_grad_(True)
                up_r = up.detach().requires_grad_(True)
                mean = torch.mean(gate_r, dim=-1, keepdim=True)
                std = torch.std(gate_r, dim=-1, keepdim=True, unbiased=False)
                std_mult = _icdf_std_multiplier(sparsity, gate_r.device).to(gate_r.dtype)
                cutoff = mean + std * std_mult
                gate_sparse = F.relu(gate_r - cutoff)
                act = F.gelu(gate_sparse, approximate=approximate)
                out = act * up_r
            dgate, dup = torch.autograd.grad(out, (gate_r, up_r), grad_out, retain_graph=False, create_graph=False)
            return dgate, dup, None, None

        # Triton backward for approximate='tanh'
        ori_shape = gate.shape
        n_cols = ori_shape[-1]
        gate_2d = gate.view(-1, n_cols)
        up_2d = up.view(-1, n_cols)
        grad_out_2d = grad_out.view(-1, n_cols)
        dgate_2d = torch.empty_like(gate_2d)
        dup_2d = torch.empty_like(up_2d)

        n_rows = gate_2d.shape[0]
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)

        std_mult_t = _icdf_std_multiplier(float(sparsity), gate_2d.device).to(torch.float32)
        std_mult = float(std_mult_t.item())

        _sparse_geglu_backward_kernel[(n_rows,)](
            gate_2d,
            up_2d,
            grad_out_2d,
            dgate_2d,
            dup_2d,
            gate_2d.stride(-2),
            n_cols=n_cols,
            std_mult=std_mult,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

        return dgate_2d.view(*ori_shape), dup_2d.view(*ori_shape), None, None


def geglu_sparse_forward(gate: torch.Tensor, up: torch.Tensor, sparsity: float, approximate: str = "tanh") -> torch.Tensor:
    return LigerGELUSparseMulFunction.apply(gate, up, sparsity, approximate)


@triton.jit
def _sparse_geglu_backward_kernel(
    gate_ptr,      # *[n_rows, n_cols]
    up_ptr,        # *[n_rows, n_cols]
    grad_out_ptr,  # *[n_rows, n_cols]
    dgate_ptr,     # *[n_rows, n_cols]
    dup_ptr,       # *[n_rows, n_cols]
    stride,        # row stride in elements
    n_cols: tl.constexpr,
    std_mult: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)

    gate_row = gate_ptr + pid * stride
    up_row = up_ptr + pid * stride
    gout_row = grad_out_ptr + pid * stride
    dgate_row = dgate_ptr + pid * stride
    dup_row = dup_ptr + pid * stride

    # First: compute mean and std in fp32
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

    # Constants for GELU tanh approximation
    sqrt_2_over_pi = 0.7978845608028654

    # Accumulate s = sum_j r_j * w_j
    s = tl.zeros((), dtype=tl.float32)
    for start in range(0, n_cols, BLOCK_SIZE):
        idx = start + offs
        mask = idx < n_cols
        g_raw = tl.load(gate_row + idx, mask=mask, other=0)
        u_raw = tl.load(up_row + idx, mask=mask, other=0)
        go_raw = tl.load(gout_row + idx, mask=mask, other=0)
        g = g_raw.to(tl.float32)
        u = u_raw.to(tl.float32)
        go = go_raw.to(tl.float32)

        x = g - cutoff
        rmask = x > 0
        x_pos = tl.where(rmask, x, 0.0)
        x2 = x_pos * x_pos
        x3 = x2 * x_pos
        t = sqrt_2_over_pi * (x_pos + 0.044715 * x3)
        u_tanh = tanh(t)
        gelu_prime = 0.5 * (1.0 + u_tanh) + 0.5 * x_pos * (1.0 - u_tanh * u_tanh) * sqrt_2_over_pi * (1.0 + 0.134145 * x2)
        w = go * u * gelu_prime
        s += tl.sum(tl.where(rmask, w, 0.0), axis=0)

    # Precompute factor for dcut/dg_i: (1/N) * (1 + std_mult * (g_i - mean)/std)
    invN = 1.0 / n_cols_f

    # Second pass: compute dup and dgate
    for start in range(0, n_cols, BLOCK_SIZE):
        idx = start + offs
        mask = idx < n_cols
        g_raw = tl.load(gate_row + idx, mask=mask, other=0)
        u_raw = tl.load(up_row + idx, mask=mask, other=0)
        go_raw = tl.load(gout_row + idx, mask=mask, other=0)
        g = g_raw.to(tl.float32)
        u = u_raw.to(tl.float32)
        go = go_raw.to(tl.float32)

        x = g - cutoff
        rmask = x > 0
        x_pos = tl.where(rmask, x, 0.0)

        # GELU(x_pos) and its derivative
        x2 = x_pos * x_pos
        x3 = x2 * x_pos
        t = sqrt_2_over_pi * (x_pos + 0.044715 * x3)
        u_tanh = tanh(t)
        gelu = 0.5 * x_pos * (1.0 + u_tanh)
        gelu_prime = 0.5 * (1.0 + u_tanh) + 0.5 * x_pos * (1.0 - u_tanh * u_tanh) * sqrt_2_over_pi * (1.0 + 0.134145 * x2)

        # Grad wrt up: dL/dup = grad_out * GELU(x)
        dup_val = go * gelu

        # w = dL/dx
        w = go * u * gelu_prime
        w_masked = tl.where(rmask, w, 0.0)

        # dcut/dg_i factor
        dcut_dg = invN * (1.0 + std_mult * (g - mean) / (std + 1e-12))

        # dgate_i = r_i * w_i - dcut_dg_i * s
        dgate_val = w_masked - dcut_dg * s

        # Store
        tl.store(dgate_row + idx, dgate_val.to(g_raw.dtype), mask=mask)
        tl.store(dup_row + idx, dup_val.to(u_raw.dtype), mask=mask)
