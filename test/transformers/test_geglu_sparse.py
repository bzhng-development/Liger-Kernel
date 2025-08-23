import math
import os
import random

import pytest
import torch

from liger_kernel.ops.geglu_sparse import gelu_and_mul_sparse as ref_gelu_and_mul_sparse
from liger_kernel.ops.geglu_sparse_triton import geglu_sparse_forward as triton_geglu_sparse_forward
from test.utils import supports_bfloat16


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _dtype_tolerances(dtype: torch.dtype):
    if dtype == torch.float32:
        return 5e-4, 5e-4
    if dtype == torch.bfloat16:
        return 2e-2, 3e-2
    if dtype == torch.float16:
        return 2e-2, 3e-2
    return 1e-3, 1e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GEGLU-sparse Triton tests require CUDA")
@pytest.mark.parametrize(
    "shape",
    [
        (2, 4, 64),
        (1, 32, 128),
        (4, 1, 256),
    ],
)
@pytest.mark.parametrize("sparsity", [0.1, 0.5, 0.95])
@pytest.mark.parametrize("dtype", [torch.float32, pytest.param(torch.bfloat16, marks=pytest.mark.skipif(not supports_bfloat16(), reason="no bf16")), torch.float16])
def test_geglu_sparse_forward_backward_parity(shape, sparsity, dtype):
    set_seed(123)
    device = torch.device("cuda")

    B, T, H = shape
    gate = torch.randn(B, T, H, device=device, dtype=dtype, requires_grad=True)
    up = torch.randn(B, T, H, device=device, dtype=dtype, requires_grad=True)

    # Upstream gradient
    upstream = torch.randn(B, T, H, device=device, dtype=torch.float32).to(dtype)

    # Triton path
    out_triton = triton_geglu_sparse_forward(gate, up, sparsity, approximate="tanh")
    loss_triton = (out_triton * upstream).sum()
    loss_triton.backward()
    dgate_triton = gate.grad.detach().clone()
    dup_triton = up.grad.detach().clone()

    # Reference PyTorch path
    gate_ref = gate.detach().clone().requires_grad_(True)
    up_ref = up.detach().clone().requires_grad_(True)
    x_ref = torch.cat([gate_ref, up_ref], dim=-1)
    out_ref = ref_gelu_and_mul_sparse(x_ref, activation_sparsity=sparsity, approximate="tanh")
    loss_ref = (out_ref * upstream).sum()
    loss_ref.backward()
    dgate_ref, dup_ref = gate_ref.grad.detach(), up_ref.grad.detach()

    atol, rtol = _dtype_tolerances(dtype)

    # Forward parity
    torch.testing.assert_close(out_triton.to(torch.float32), out_ref.to(torch.float32), atol=atol, rtol=rtol)
    # Grad parity
    torch.testing.assert_close(dgate_triton.to(torch.float32), dgate_ref.to(torch.float32), atol=atol, rtol=rtol)
    torch.testing.assert_close(dup_triton.to(torch.float32), dup_ref.to(torch.float32), atol=atol, rtol=rtol)

