from __future__ import annotations

import torch
from torch import nn

try:
    from transformers.models.gemma3n.configuration_gemma3n import Gemma3nTextConfig  # type: ignore
except Exception:  # pragma: no cover
    Gemma3nTextConfig = object

from liger_kernel.transformers.rms_norm import LigerRMSNormForGemma3n


class LigerGemma3nLaurelBlock(nn.Module):
    """Learned Augmented Residual Layer (Laurel) for Gemma3n text.

    Matches HF Gemma3nTextLaurelBlock semantics but uses Liger's RMSNorm.
    Construction mirrors HF: accepts a Gemma3nTextConfig and reads sizes.
    """

    def __init__(self, config: Gemma3nTextConfig) -> None:
        super().__init__()
        self.config = config

        hidden_size = int(config.hidden_size)
        laurel_rank = int(config.laurel_rank)
        eps = float(config.rms_norm_eps)

        # Linear projections (no bias). These map H -> r -> H.
        self.linear_left = nn.Linear(hidden_size, laurel_rank, bias=False)
        self.linear_right = nn.Linear(laurel_rank, hidden_size, bias=False)

        # Post-laurel normalization uses Liger RMSNorm variant for Gemma3n
        self.post_laurel_norm = LigerRMSNormForGemma3n(dim=hidden_size, eps=eps, with_scale=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        laurel_x = self.linear_left(x)
        laurel_x = self.linear_right(laurel_x)
        normed_laurel_x = self.post_laurel_norm(laurel_x)
        return x + normed_laurel_x

