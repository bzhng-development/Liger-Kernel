from __future__ import annotations

from typing import Optional

import torch
from torch import nn

try:
    # Import only for type hints and runtime access when transformers is present
    from transformers.models.gemma3n.configuration_gemma3n import Gemma3nTextConfig  # type: ignore
except Exception:  # pragma: no cover - optional at import time
    Gemma3nTextConfig = object  # fallback for type checking without transformers

from liger_kernel.transformers.rms_norm import LigerRMSNormForGemma3n


class LigerGemma3nAltUp(nn.Module):
    """
    Alternating updates (AltUp) wrapper for Gemma3n (text-only).

    Provides the predict/correct steps and the router path using Liger's RMSNorm.
    This mirrors the semantics of HF's Gemma3nTextAltUp and vLLM's Gemma3nAltUp,
    with shape conventions compatible with HF:
      - hidden_states shape: [num_altup_inputs, batch, seq_len, hidden_size]
      - returns tensors of matching shapes.

    The module exposes:
      - predict(hidden_states): apply prediction mixing across AltUp inputs
      - correct(predictions, activated): propagate corrections from the active branch
      - scale_corrected_output(x): learned per-channel scaling for corrected output
    """

    def __init__(self, config: Gemma3nTextConfig) -> None:
        super().__init__()
        # Read required attributes from the HF config
        self.config = config
        hidden_size = int(config.hidden_size)
        altup_num_inputs = int(config.altup_num_inputs)
        rms_norm_eps = float(config.rms_norm_eps)

        # Coefficients and router projections
        self.correction_coefs = nn.Linear(altup_num_inputs, altup_num_inputs, bias=False)
        self.prediction_coefs = nn.Linear(altup_num_inputs, altup_num_inputs**2, bias=False)
        self.modality_router = nn.Linear(hidden_size, altup_num_inputs, bias=False)

        # Router norm uses Liger RMSNorm for Gemma3n
        self.router_norm = LigerRMSNormForGemma3n(dim=hidden_size, eps=rms_norm_eps, with_scale=True)
        # Match HF's buffer dtype behavior; scale is applied prior to router linear
        self.register_buffer(
            "router_input_scale",
            torch.tensor(hidden_size ** -1.0, dtype=torch.float32),
            persistent=False,
        )

        # Learned output scaling after correction (matches HF/vLLM semantics)
        self.correct_output_scale = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))

    # Keep naming aligned with HF for easier monkey-patching
    def compute_router_modalities(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, hidden_size]
        router_inputs = self.router_norm(x) * self.router_input_scale.type_as(x)
        routed = self.modality_router(router_inputs)
        # tanh on fp32 for numerical stability, then cast back
        return torch.tanh(routed.float()).type_as(x)

    def predict(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Predicts the output of a layer using a trainable mixing map.

        Args:
          hidden_states: [num_altup_inputs, batch, seq_len, hidden_size]

        Returns:
          predictions: [num_altup_inputs, batch, seq_len, hidden_size]
        """
        assert hidden_states.dim() == 4, "Expected hidden_states of shape [num_altup_inputs, batch, seq_len, hidden_size]"
        n_inputs = int(self.config.altup_num_inputs)
        assert hidden_states.size(0) == n_inputs, "Mismatched first dim vs altup_num_inputs"

        # Router on the active branch
        active_idx = int(self.config.altup_active_idx)
        modalities = self.compute_router_modalities(hidden_states[active_idx])  # [B, T, n_inputs]

        # Optional coefficient clipping during training
        coef_clip = getattr(self.config, "altup_coef_clip", None)
        if self.training and coef_clip is not None:
            self.prediction_coefs.weight.data.clamp_(-coef_clip, coef_clip)

        # Compute all coefficients and reshape to [B, T, n_inputs, n_inputs], then transpose last 2 dims
        all_coefs = self.prediction_coefs(modalities)
        all_coefs = all_coefs.reshape(*modalities.shape[:-1], n_inputs, n_inputs).permute(0, 1, 3, 2)

        # Bring hidden_states to [B, T, H, n_inputs] to right-multiply by all_coefs
        predictions = torch.matmul(hidden_states.permute(1, 2, 3, 0), all_coefs)
        predictions = predictions.permute(3, 0, 1, 2)  # back to [n_inputs, B, T, H]
        predictions = predictions + hidden_states
        return predictions.contiguous().type_as(hidden_states)

    def correct(self, predictions: torch.Tensor, activated: torch.Tensor) -> torch.Tensor:
        """
        Corrects the predictions relative to the activated branch.

        Args:
          predictions: [num_altup_inputs, batch, seq_len, hidden_size]
          activated:   [batch, seq_len, hidden_size]

        Returns:
          corrected:   [num_altup_inputs, batch, seq_len, hidden_size]
        """
        assert predictions.dim() == 4 and activated.dim() == 3, "Unexpected tensor ranks for AltUp.correct"
        n_inputs = int(self.config.altup_num_inputs)
        active_idx = int(self.config.altup_active_idx)

        modalities = self.compute_router_modalities(activated)  # [B, T, n_inputs]
        innovation = activated - predictions[active_idx]  # [B, T, H]
        # Repeat along the AltUp dimension to broadcast per-branch scalars
        innovation = innovation.repeat(n_inputs, 1, 1, 1)

        coef_clip = getattr(self.config, "altup_coef_clip", None)
        if coef_clip is not None:
            self.correction_coefs.weight.data.clamp_(-coef_clip, coef_clip)

        # all_coefs: [B, T, n_inputs]; then to [n_inputs, B, T, 1] for broadcast over H
        all_coefs = self.correction_coefs(modalities) + 1.0
        all_coefs = all_coefs.permute(2, 0, 1).unsqueeze(-1)

        corrected = innovation * all_coefs
        corrected = corrected + predictions
        return corrected.contiguous().type_as(activated)

    # Define forward solely so offloading hooks can find parameters/buffers
    def forward(self, corrected: torch.Tensor) -> torch.Tensor:
        return (corrected.type_as(self.correct_output_scale) * self.correct_output_scale).type_as(corrected)

    def scale_corrected_output(self, corrected: torch.Tensor) -> torch.Tensor:
        return self.forward(corrected)

