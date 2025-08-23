import torch
import torch.nn as nn

from liger_kernel.ops.geglu import LigerGELUMulFunction
from liger_kernel.ops.geglu_sparse import gelu_and_mul_sparse
try:
    from liger_kernel.ops.geglu_sparse_triton import geglu_sparse_forward as _triton_geglu_sparse_forward
    _HAS_TRITON_SPARSE = True
except ImportError:
    _HAS_TRITON_SPARSE = False


class LigerGEGLUMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        # TODO: support exact GELU
        # Right now Gemma 1, 1.1 and 2 models are all using `gelu_pytorch_tanh`
        # https://github.com/huggingface/transformers/blob/v4.40.1/src/transformers/models/gemma/modeling_gemma.py#L175
        # https://github.com/huggingface/transformers/blob/v4.40.1/src/transformers/activations.py#L46
        # So we can safely assume we use tanh approximation form all the time

    def forward(self, x):
        return self.down_proj(LigerGELUMulFunction.apply(self.gate_proj(x), self.up_proj(x)))


def liger_geglu_sparse_forward(self, x):
    """
    Sparse GEGLU forward that respects `self.activation_sparsity` per-layer.

    Semantics mirror Gemma3n's GeluAndMulSparse:
      gate = gate_proj(x)
      up   = up_proj(x)
      gate = gaussian_topk(gate)  # sparsity via Gaussian percentile cutoff per token
      act  = GELU(gate, approximate='tanh')
      out  = down_proj(act * up)

    Assumptions:
    - `self` exposes Linear layers: `gate_proj`, `up_proj`, `down_proj`.
    - `self.activation_sparsity` is a float in [0, 1). For example, 0.95 zeros ~95% of gate features.
    """
    sparsity = float(getattr(self, "activation_sparsity", 0.0))
    if sparsity <= 0.0:
        # Fallback to dense path using fused tanh-GELU*Mul
        return self.down_proj(LigerGELUMulFunction.apply(self.gate_proj(x), self.up_proj(x)))

    gate = self.gate_proj(x)
    up = self.up_proj(x)
    # Concatenate [gate, up] along feature dim for the op API
    if _HAS_TRITON_SPARSE:
        h = _triton_geglu_sparse_forward(gate, up, sparsity, approximate="tanh")
    else:
        # Fallback to pure PyTorch reference
        concat = torch.cat([gate, up], dim=-1)
        h = gelu_and_mul_sparse(concat, activation_sparsity=sparsity, approximate="tanh")
    return self.down_proj(h)
