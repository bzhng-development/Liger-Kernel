from liger_kernel.ops.rope import LigerRopeFunction
import torch


def liger_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    Applies Rotary Positional Embedding (RoPE) operation to query and key states.

    Args:
        q (torch.Tensor): The query tensor of shape (bsz, n_q_head, seq_len, head_dim).
        k (torch.Tensor): The key tensor of shape (bsz, n_kv_head, seq_len, head_dim).
        cos (torch.Tensor): The cosine tensor of shape (1, seq_len, head_dim) or (bsz, seq_len, head_dim).
        sin (torch.Tensor): The sine tensor of shape (1, seq_len, head_dim) or (bsz, seq_len, head_dim).
        position_ids (torch.Tensor, optional): The position ids tensor. Defaults to None.
        unsqueeze_dim (int, optional): The dimension to unsqueeze. Defaults to 1.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The query and key tensors after applying the RoPE operation.
    """

    return LigerRopeFunction.apply(q, k, cos, sin, position_ids, unsqueeze_dim)


def liger_gemma3n_apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    unsqueeze_dim: int = 1,
):
    """
    Gemma3n-compatible RoPE wrapper that leverages Liger's fused kernel.

    HF Gemma3n applies RoPE via a single-tensor function with signature
    `apply_rotary_pos_emb(x, cos, sin, position_ids=None, unsqueeze_dim=...)`
    and calls it separately for q and k. Liger's kernel expects (q, k) and
    returns both. This wrapper adapts HF's call site by:

    - Reordering `x` to (batch, heads, seq, head_dim) if needed
    - Invoking the fused kernel with (x, x) and returning the rotated `q`
    - Restoring the original layout

    This preserves Gemma3n's broadcasting semantics via `unsqueeze_dim` while
    enabling the fused RoPE kernel.
    """
    original_layout = x
    ro_unsqueeze_dim = unsqueeze_dim

    # HF Gemma3n calls with x shaped [B, T, H, D] and unsqueeze_dim=2.
    # Liger expects [B, H, T, D] with unsqueeze along the head axis (dim=1).
    if x.dim() == 4 and unsqueeze_dim == 2:
        x = x.permute(0, 2, 1, 3).contiguous()
        ro_unsqueeze_dim = 1

    # Call fused kernel; we pass x for both q and k, and return the first.
    q_rot, _ = LigerRopeFunction.apply(x, x, cos, sin, position_ids, ro_unsqueeze_dim)

    # Restore original layout if we permuted.
    if original_layout is not x:
        q_rot = q_rot.permute(0, 2, 1, 3).contiguous()

    return q_rot
