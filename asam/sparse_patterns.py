"""
Sparse Attention Patterns
=========================

This module implements various sparse attention patterns that reduce
the O(n²) complexity of standard attention to O(n log n) or O(n).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparsePattern(ABC, nn.Module):
    """Base class for sparse attention patterns."""

    def __init__(self, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self._device_pattern_cache = {}
        # Use register_buffer with persistent=False to avoid saving in state_dict
        # but we can't use persistent in older PyTorch, so use None check instead

    @abstractmethod
    def build_pattern(self) -> torch.Tensor:
        """
        Build the sparse attention mask pattern.

        Returns:
            Boolean tensor of shape [seq_len, seq_len] where True
            indicates positions that should be attended to.
        """
        pass

    def get_pattern(self, device: torch.device) -> torch.Tensor:
        """Get pattern, caching if necessary."""
        if not hasattr(self, "_cached_pattern"):
            pattern = self.build_pattern()
            self.register_buffer("_cached_pattern", pattern)

        if self._cached_pattern.device == device:
            return self._cached_pattern

        cache_key = (device.type, device.index)
        cached_pattern = self._device_pattern_cache.get(cache_key)
        if cached_pattern is None:
            cached_pattern = self._cached_pattern.to(device)
            self._device_pattern_cache[cache_key] = cached_pattern

        return cached_pattern


class LocalSparsePattern(SparsePattern):
    """
    Local (sliding window) sparse pattern.
    Each position only attends to its local neighborhood.

    Complexity: O(n * window_size)
    """

    def __init__(self, seq_len: int, window_size: int = 128):
        self.window_size = window_size
        super().__init__(seq_len)

    def build_pattern(self) -> torch.Tensor:
        positions = torch.arange(self.seq_len)
        return (positions.view(-1, 1) - positions.view(1, -1)).abs() <= (self.window_size // 2)


class StridedSparsePattern(SparsePattern):
    """
    Strided sparse pattern with fixed stride intervals.
    Useful for capturing periodic patterns.

    Complexity: O(n * n / stride)
    """

    def __init__(self, seq_len: int, stride: int = 32, local_window: int = 16):
        self.stride = stride
        self.local_window = local_window
        super().__init__(seq_len)

    def build_pattern(self) -> torch.Tensor:
        positions = torch.arange(self.seq_len)
        pattern = (positions.view(-1, 1) - positions.view(1, -1)).abs() <= self.local_window
        pattern[:, torch.arange(0, self.seq_len, self.stride)] = True
        return pattern


class RandomSparsePattern(SparsePattern):
    """
    Random sparse pattern for each head.
    Based on "Random Feature Attention" research.

    Complexity: O(n * num_random)
    """

    def __init__(self, seq_len: int, num_random: int = 128, num_heads: int = 8, seed: int = 42):
        self.num_random = num_random
        self.num_heads = num_heads
        self.seed = seed
        super().__init__(seq_len)

    def build_pattern(self) -> torch.Tensor:
        pattern = torch.zeros(self.num_heads, self.seq_len, self.seq_len, dtype=torch.bool)
        num_random = min(self.num_random, self.seq_len)

        for h in range(self.num_heads):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed + h)
            random_scores = torch.rand(self.seq_len, self.seq_len, generator=generator)
            random_indices = random_scores.topk(k=num_random, dim=-1).indices
            pattern[h].scatter_(1, random_indices, True)

        return pattern


class ClusteredSparsePattern(SparsePattern):
    """
    Clustered sparse pattern using learnable cluster centroids.
    Assigns tokens to clusters and allows intra-cluster attention.

    This is an original contribution: dynamic clustering-based sparsity.
    """

    def __init__(
        self, seq_len: int, num_clusters: int = 32, num_heads: int = 8, dim_head: int = 64
    ):
        self.num_clusters = num_clusters
        self.num_heads = num_heads
        self.dim_head = dim_head
        super().__init__(seq_len)

        # Learnable cluster centroids for each head
        self.centroids = nn.Parameter(torch.randn(num_heads, num_clusters, dim_head) * 0.02)

        # Temperature for soft assignment
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)

    def build_pattern(self) -> torch.Tensor:
        """Initial pattern (will be updated dynamically during forward pass)."""
        pattern = torch.ones(self.seq_len, self.seq_len, dtype=torch.bool)
        return pattern

    def compute_cluster_assignment(
        self, queries: torch.Tensor, keys: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute soft cluster assignments for queries and keys.

        Args:
            queries: [batch, heads, seq_len, dim_head]
            keys: [batch, heads, seq_len, dim_head]

        Returns:
            q_assign: [batch, heads, seq_len, num_clusters]
            k_assign: [batch, heads, seq_len, num_clusters]
        """
        batch, heads, seq_len, dim_head = queries.shape

        q_norm = F.normalize(queries, dim=-1)
        k_norm = F.normalize(keys, dim=-1)
        centroids_norm = F.normalize(self.centroids, dim=-1).transpose(-1, -2).contiguous()

        q_flat = q_norm.reshape(batch * heads, seq_len, dim_head)
        k_flat = k_norm.reshape(batch * heads, seq_len, dim_head)
        centroids_flat = (
            centroids_norm.unsqueeze(0)
            .expand(batch, -1, -1, -1)
            .reshape(
                batch * heads,
                dim_head,
                self.num_clusters,
            )
        )

        q_sim = torch.bmm(q_flat, centroids_flat).reshape(batch, heads, seq_len, self.num_clusters)
        k_sim = torch.bmm(k_flat, centroids_flat).reshape(batch, heads, seq_len, self.num_clusters)

        temperature_scale = self.temperature.abs().clamp_min(1e-6).reciprocal()
        q_assign = F.softmax(q_sim * temperature_scale, dim=-1)
        k_assign = F.softmax(k_sim * temperature_scale, dim=-1)

        return q_assign, k_assign

    def apply_cluster_mask(
        self, attn_scores: torch.Tensor, q_assign: torch.Tensor, k_assign: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply cluster-based sparse mask to attention scores.

        Args:
            attn_scores: [batch, heads, seq_len, seq_len]
            q_assign: [batch, heads, seq_len, num_clusters]
            k_assign: [batch, heads, seq_len, num_clusters]

        Returns:
            Masked attention scores
        """
        cluster_affinity = torch.einsum("b h q c, b h k c -> b h q k", q_assign, k_assign)
        mask = cluster_affinity > 0.1  # Threshold for sparsity
        return attn_scores.masked_fill(~mask, torch.finfo(attn_scores.dtype).min)


class HierarchicalSparsePattern(SparsePattern):
    """
    Hierarchical sparse pattern combining multiple granularities.
    Original contribution: multi-scale attention mechanism.
    """

    def __init__(
        self,
        seq_len: int,
        scales: Optional[List[int]] = None,
        num_heads: int = 8,
    ) -> None:
        self.scales = scales or [4, 16, 64]
        self.num_heads = num_heads
        super().__init__(seq_len)

        # Create sub-patterns for different scales
        self.patterns = nn.ModuleList(
            [StridedSparsePattern(seq_len, stride=s, local_window=s // 4) for s in self.scales]
        )
        self._pattern_stack_cache = {}

        # Learnable scale weights
        self.scale_weights = nn.Parameter(torch.ones(len(self.scales), num_heads))

    def build_pattern(self) -> torch.Tensor:
        return self.patterns[0].build_pattern()

    def _get_pattern_stack(self, device: torch.device) -> torch.Tensor:
        cache_key = (device.type, device.index)
        cached_stack = self._pattern_stack_cache.get(cache_key)
        if cached_stack is not None:
            return cached_stack

        stacked_patterns = []
        for pattern_module in self.patterns:
            pattern_tensor = pattern_module.get_pattern(device)
            if pattern_tensor.dim() == 2:
                pattern_tensor = pattern_tensor.unsqueeze(0).expand(self.num_heads, -1, -1)
            stacked_patterns.append(pattern_tensor)

        cached_stack = torch.stack(stacked_patterns, dim=0).to(dtype=torch.float32)
        self._pattern_stack_cache[cache_key] = cached_stack
        return cached_stack

    def combine_patterns(self, device: torch.device) -> torch.Tensor:
        """Combine patterns from all scales with learned weights."""
        pattern_stack = self._get_pattern_stack(device)
        weights = F.softmax(self.scale_weights, dim=0).to(device=device, dtype=pattern_stack.dtype)
        combined = torch.einsum("sh,shij->hij", weights, pattern_stack)
        return combined > 0.5
