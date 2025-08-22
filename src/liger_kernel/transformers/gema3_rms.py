import torch

from .rms_norm import LigerRMSNorm


class LigerRMSNormForGemma3(LigerRMSNorm):
    """
    Gemma3/Gemma3n RMSNorm shim:
    - Accepts `dim` instead of `hidden_size`.
    - Supports `with_scale` flag used by q_norm/k_norm in Gemma3n to disable learnable scaling.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        offset: float = 1.0,
        casting_mode: str = "gemma",
        init_fn: str = "zeros",
        in_place: bool = False,
        with_scale: bool = True,
    ):
        # Initialize as Gemma-style RMSNorm: weight initialized to zeros with offset=1.0
        # so the effective scale is (1.0 + weight).
        super().__init__(dim, eps, offset, casting_mode, init_fn, in_place)

        # If Gemma3n requests no scale (with_scale=False), ensure the scale stays constant 1.0.
        # Convert the learnable parameter to a non-trainable buffer of zeros so (offset + weight) == 1.0.
        if not with_scale:
            # Remove parameter and register a buffer with the same name
            try:
                delattr(self, "weight")
            except Exception:
                # Fallback in case attribute is not set yet
                pass
            self.register_buffer("weight", torch.zeros(dim), persistent=False)


class LigerRMSNormForGemma3n(LigerRMSNorm):
    """
    Gemma3n RMSNorm shim:
    - Uses standard RMSNorm (offset = 0.0) when learnable scale is present.
    - When `with_scale=False` (e.g., v-norm), keeps unit scale by using a zero
      buffer as weight and sets offset = 1.0 so gamma == 1.0 without training.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        casting_mode: str = "gemma",
        in_place: bool = False,
        with_scale: bool = True,
    ):
        # Default to standard RMSNorm behavior when scale is learnable:
        # gamma is initialized to ones and offset is 0.0.
        if with_scale:
            super().__init__(
                hidden_size=dim,
                eps=eps,
                offset=0.0,
                casting_mode=casting_mode,
                init_fn="ones",
                in_place=in_place,
            )
        else:
            # No learnable scale: keep effective gamma == 1.0
            super().__init__(
                hidden_size=dim,
                eps=eps,
                offset=1.0,
                casting_mode=casting_mode,
                init_fn="zeros",
                in_place=in_place,
            )
            # Replace weight parameter with a non-trainable zero buffer.
            try:
                delattr(self, "weight")
            except Exception:
                pass
            self.register_buffer("weight", torch.zeros(dim), persistent=False)
