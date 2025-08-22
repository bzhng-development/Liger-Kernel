import torch
import torch.nn as nn
import torch.nn.functional as F


def gelu_and_mul_sparse(x: torch.Tensor, activation_sparsity: float, approximate: str = "none") -> torch.Tensor:
    """
    Sparse GEGLU (GeluAndMulSparse) as used by Gemma3n.

    Computes:
      - Split x into gate and up halves along the last dim: x = [gate, up]
      - gate <- gaussian_topk(gate) with target sparsity
      - act <- GELU(gate, approximate)
      - out <- act * up

    Shapes:
      - x: (..., 2 * d)
      - return: (..., d)

    Args:
      x: Input tensor containing concatenated [gate, up] features along last dim.
      activation_sparsity: Fraction in [0, 1). When 0.95, about 95% of gate features become zeroed via
        a Gaussian percentile cutoff computed per token.
      approximate: GELU mode, one of {"none", "tanh"}.

    Notes:
      - This is a pure-PyTorch implementation intended for correctness. It is differentiable and uses
        standard autograd. A fused kernel can replace internals later for performance.
    """
    if approximate not in ("none", "tanh"):
        raise ValueError(f"Unknown approximate mode: {approximate}")
    if not (0.0 <= activation_sparsity < 1.0):
        raise ValueError("activation_sparsity must be in [0.0, 1.0)")
    if activation_sparsity == 0.0:
        raise ValueError("activation_sparsity is 0.0. Please use dense GeluAndMul instead.")

    d2 = x.shape[-1]
    if d2 % 2 != 0:
        raise ValueError("Last dimension of input must be even (concat of gate and up)")
    d = d2 // 2

    gate = x[..., :d]
    up = x[..., d:]

    # Gaussian percentile threshold per token: cutoff = mean + std * icdf(sparsity)
    mean = torch.mean(gate, dim=-1, keepdim=True)
    std = torch.std(gate, dim=-1, keepdim=True, unbiased=False)
    std_multiplier = torch.distributions.normal.Normal(0, 1).icdf(
        torch.tensor(activation_sparsity, dtype=torch.float32, device=x.device)
    )
    cutoff = mean + std * std_multiplier.to(gate.dtype)

    # Keep values above cutoff (relu on shifted gate)
    gate_sparse = F.relu(gate - cutoff)

    # GELU activation (approximate="tanh" or exact)
    act = F.gelu(gate_sparse, approximate=approximate)

    # Elementwise product with up half
    return act * up


class LigerGELUMulSparse(nn.Module):
    """
    Module wrapper around gelu_and_mul_sparse for convenience.

    This module expects input of shape (..., 2 * d) with concatenated [gate, up].
    """

    def __init__(self, activation_sparsity: float, approximate: str = "none") -> None:
        super().__init__()
        self.activation_sparsity = float(activation_sparsity)
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return gelu_and_mul_sparse(x, self.activation_sparsity, self.approximate)

