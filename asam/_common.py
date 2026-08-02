"""
Shared utility functions used across ASAM attention layer implementations.

These are extracted from asam_layer.py and asam_layer_optimized.py to
eliminate code duplication.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def normalize_attention_mask(
    mask: torch.Tensor,
    batch: int,
    heads: int,
    seq_len: int,
) -> torch.Tensor:
    """Normalize attention mask to 4D [batch, heads, seq_len, seq_len].

    Accepts masks of shape [seq_len, seq_len], [batch, seq_len, seq_len],
    or [batch, heads, seq_len, seq_len] and expands them to a consistent
    4D boolean tensor. This is used by both the original ASAMLayer and
    OptimizedASAMLayer to handle user-provided masks.

    Args:
        mask: Input attention mask.
        batch: Target batch size.
        heads: Target number of attention heads.
        seq_len: Target sequence length.

    Returns:
        Boolean mask of shape [batch, heads, seq_len, seq_len].

    Raises:
        ValueError: If mask dimensions or sizes are incompatible.
    """
    if mask.dtype != torch.bool:
        mask = mask.to(torch.bool)

    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)
    elif mask.dim() != 4:
        raise ValueError("attention mask must have 2, 3, or 4 dimensions")

    if mask.size(-2) != seq_len or mask.size(-1) != seq_len:
        raise ValueError("attention mask must match sequence length")

    if mask.size(0) == 1 and batch != 1:
        mask = mask.expand(batch, -1, -1, -1)
    elif mask.size(0) != batch:
        raise ValueError("attention mask batch dimension is not broadcastable")

    if mask.size(1) == 1 and heads != 1:
        mask = mask.expand(-1, heads, -1, -1)
    elif mask.size(1) != heads:
        raise ValueError("attention mask head dimension is not broadcastable")

    return mask


def gather_values_by_positions(
    tensor: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Gather values from a tensor at specified positions along dim=2.

    Given a tensor of shape [batch, heads, seq_len, dim_head] and positions
    of shape [seq_len, context_size] or [heads, seq_len, context_size], returns a tensor of shape
    [batch, heads, seq_len, context_size, dim_head] where position (i, j)
    selects tensor[:, :, positions[i, j], :] for broadcast positions, or
    tensor[:, h, positions[h, i, j], :] for head-specific positions.

    Used in sparse attention to gather only the keys/values that are
    included in the sparse pattern, avoiding O(n^2) memory.

    Args:
        tensor: Source tensor [batch, heads, seq_len, dim_head].
        positions: Index tensor with values in [0, seq_len).

    Returns:
        Gathered tensor [batch, heads, seq_len, context_size, dim_head].
    """
    batch, heads, seq_len, dim_head = tensor.shape
    context_size = positions.size(-1)

    if positions.dim() == 2:
        if positions.size(0) != seq_len:
            raise ValueError("broadcast positions must match sequence length")
        head_positions = positions.view(1, seq_len, context_size).expand(heads, -1, -1)
    elif positions.dim() == 3:
        if positions.size(1) != seq_len:
            raise ValueError("head-specific positions must match sequence length")
        if positions.size(0) == 1 and heads != 1:
            head_positions = positions.expand(heads, -1, -1)
        elif positions.size(0) == heads:
            head_positions = positions
        else:
            raise ValueError("head-specific positions must match attention head count")
    else:
        raise ValueError(
            "positions must have shape [seq_len, context] or [heads, seq_len, context]"
        )

    expanded_tensor = tensor.unsqueeze(3).expand(-1, -1, -1, context_size, -1)
    gather_index = head_positions.view(1, heads, seq_len, context_size, 1).expand(
        batch, heads, -1, -1, dim_head
    )
    return torch.gather(expanded_tensor, 2, gather_index)


def expand_pattern_mask(
    pattern_mask: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    """Expand 2D pattern mask to 3D with per-head dimension.

    Args:
        pattern_mask: Mask of shape [seq_len, seq_len] or [heads, seq_len, seq_len].
        num_heads: Target number of attention heads.

    Returns:
        Expanded mask [num_heads, seq_len, seq_len].

    Raises:
        ValueError: If pattern_mask has unexpected dimensions.
    """
    if pattern_mask.dim() == 2:
        return pattern_mask.unsqueeze(0).expand(num_heads, -1, -1)
    if pattern_mask.dim() == 3:
        return pattern_mask
    raise ValueError("pattern mask must have 2 or 3 dimensions")


def pattern_mask_to_indices(
    pattern_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a boolean pattern mask to index tensors for gather-based attention.

    For each query position, sorts attendable key positions by presence
    (True before False) and returns the top-k indices along with a validity mask.
    This enables efficient sparse attention via torch.gather without building
    full O(n^2) intermediate tensors.

    Args:
        pattern_mask: Boolean mask [num_heads, seq_len, seq_len] where True
            indicates that query i may attend to key j.

    Returns:
        positions: LongTensor [num_heads, seq_len, max_connections] with
            key indices for each query position.
        valid_mask: BoolTensor [num_heads, seq_len, max_connections] where
            True indicates a valid (not padding) connection.
    """
    num_connections = pattern_mask.sum(dim=-1)
    max_connections = max(1, int(num_connections.max().item()))

    sorted_indices = torch.argsort(pattern_mask.to(torch.int64), dim=-1, descending=True)
    positions = sorted_indices[..., :max_connections]
    valid_mask = torch.arange(max_connections, device=pattern_mask.device).view(
        1, 1, -1
    ) < num_connections.unsqueeze(-1)

    return positions, valid_mask
