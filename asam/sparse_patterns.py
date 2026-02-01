"""
Sparse Attention Patterns
=========================

This module implements various sparse attention patterns that reduce
the O(n²) complexity of standard attention to O(n log n) or O(n).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from abc import ABC, abstractmethod


class SparsePattern(ABC, nn.Module):
    """Base class for sparse attention patterns."""
    
    def __init__(self, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
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
        # Check if we already have a cached pattern
        if hasattr(self, '_cached_pattern'):
            return self._cached_pattern.to(device)
        
        # Build and cache the pattern
        pattern = self.build_pattern()
        self.register_buffer('_cached_pattern', pattern)
        return self._cached_pattern.to(device)


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
        pattern = torch.zeros(self.seq_len, self.seq_len, dtype=torch.bool)
        for i in range(self.seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(self.seq_len, i + self.window_size // 2 + 1)
            pattern[i, start:end] = True
        return pattern


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
        pattern = torch.zeros(self.seq_len, self.seq_len, dtype=torch.bool)
        
        # Local window for each position
        for i in range(self.seq_len):
            start = max(0, i - self.local_window)
            end = min(self.seq_len, i + self.local_window + 1)
            pattern[i, start:end] = True
        
        # Strided global attention
        for i in range(self.seq_len):
            strided_indices = torch.arange(0, self.seq_len, self.stride)
            pattern[i, strided_indices] = True
        
        return pattern


class RandomSparsePattern(SparsePattern):
    """
    Random sparse pattern for each head.
    Based on "Random Feature Attention" research.
    
    Complexity: O(n * num_random)
    """
    
    def __init__(self, seq_len: int, num_random: int = 128, num_heads: int = 8):
        self.num_random = num_random
        self.num_heads = num_heads
        super().__init__(seq_len)
    
    def build_pattern(self) -> torch.Tensor:
        # Each head gets a different random pattern
        torch.manual_seed(42)
        pattern = torch.zeros(self.num_heads, self.seq_len, self.seq_len, dtype=torch.bool)
        
        for h in range(self.num_heads):
            torch.manual_seed(42 + h)
            for i in range(self.seq_len):
                random_indices = torch.randperm(self.seq_len)[:self.num_random]
                pattern[h, i, random_indices] = True
        
        return pattern


class ClusteredSparsePattern(SparsePattern):
    """
    Clustered sparse pattern using learnable cluster centroids.
    Assigns tokens to clusters and allows intra-cluster attention.
    
    This is an original contribution: dynamic clustering-based sparsity.
    """
    
    def __init__(
        self, 
        seq_len: int, 
        num_clusters: int = 32,
        num_heads: int = 8,
        dim_head: int = 64
    ):
        self.num_clusters = num_clusters
        self.num_heads = num_heads
        self.dim_head = dim_head
        super().__init__(seq_len)
        
        # Learnable cluster centroids for each head
        self.centroids = nn.Parameter(
            torch.randn(num_heads, num_clusters, dim_head) * 0.02
        )
        
        # Temperature for soft assignment
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)
    
    def build_pattern(self) -> torch.Tensor:
        """Initial pattern (will be updated dynamically during forward pass)."""
        pattern = torch.ones(self.seq_len, self.seq_len, dtype=torch.bool)
        return pattern
    
    def compute_cluster_assignment(
        self, 
        queries: torch.Tensor,
        keys: torch.Tensor
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
        # Normalize for cosine similarity
        q_norm = F.normalize(queries, dim=-1)
        k_norm = F.normalize(keys, dim=-1)
        centroids_norm = F.normalize(self.centroids, dim=-1)
        
        # Compute similarities
        q_sim = torch.einsum('b h s d, h c d -> b h s c', q_norm, centroids_norm)
        k_sim = torch.einsum('b h s d, h c d -> b h s c', k_norm, centroids_norm)
        
        # Soft assignment with temperature
        q_assign = F.softmax(q_sim / self.temperature.abs(), dim=-1)
        k_assign = F.softmax(k_sim / self.temperature.abs(), dim=-1)
        
        return q_assign, k_assign
    
    def apply_cluster_mask(
        self,
        attn_scores: torch.Tensor,
        q_assign: torch.Tensor,
        k_assign: torch.Tensor
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
        # Compute cluster affinity matrix
        cluster_affinity = torch.einsum('b h q c, b h k c -> b h q k', q_assign, k_assign)
        
        # Only attend if query and key are in similar clusters
        mask = cluster_affinity > 0.1  # Threshold for sparsity
        
        # Apply mask (set non-matching to very negative)
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        
        return attn_scores


class HierarchicalSparsePattern(SparsePattern):
    """
    Hierarchical sparse pattern combining multiple granularities.
    Original contribution: multi-scale attention mechanism.
    """
    
    def __init__(
        self,
        seq_len: int,
        scales: List[int] = None,
        num_heads: int = 8
    ):
        self.scales = scales or [4, 16, 64]
        self.num_heads = num_heads
        super().__init__(seq_len)
        
        # Create sub-patterns for different scales
        self.patterns = nn.ModuleList([
            StridedSparsePattern(seq_len, stride=s, local_window=s//4)
            for s in self.scales
        ])
        
        # Learnable scale weights
        self.scale_weights = nn.Parameter(torch.ones(len(self.scales), num_heads))
    
    def build_pattern(self) -> torch.Tensor:
        return self.patterns[0].build_pattern()
    
    def combine_patterns(self, device: torch.device) -> torch.Tensor:
        """Combine patterns from all scales with learned weights."""
        combined = []
        
        for pattern_module in self.patterns:
            p = pattern_module.get_pattern(device)
            if p.dim() == 2:
                p = p.unsqueeze(0).expand(self.num_heads, -1, -1)
            combined.append(p.float())
        
        # Stack and weight
        combined = torch.stack(combined, dim=0)  # [num_scales, heads, seq, seq]
        weights = F.softmax(self.scale_weights, dim=0).view(-1, self.num_heads, 1, 1).to(device)
        
        # Weighted combination
        result = (combined * weights).sum(dim=0) > 0.5
        
        return result
