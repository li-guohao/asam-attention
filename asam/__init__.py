"""
Adaptive Sparse Attention Mechanism (ASAM)
============================================

A novel attention mechanism that combines sparse attention patterns with
adaptive gating to efficiently process long sequences.

Author: Guohao Li
License: MIT
"""

from .adaptive_gate import AdaptiveGate, DynamicSparseDenseAttention
from .asam_layer import ASAMConfig, ASAMEncoder, ASAMLayer
from .asam_layer_optimized import OptimizedASAMLayer
from .continual_asam import ContinualASAMConfig, ContinualASAMLayer, PrototypeContinualASAMLayer
from .efficient_attention import EfficientASAMLayer, FlashASAMLayer
from .flash_asam import FlashAttnASAMLayer, HybridASAM
from .sparse_patterns import (
    ClusteredSparsePattern,
    HierarchicalSparsePattern,
    LocalSparsePattern,
    RandomSparsePattern,
    StridedSparsePattern,
)

__version__ = "1.2.0"
__all__ = [
    # Core layer
    "ASAMLayer",
    "ASAMConfig",
    "ASAMEncoder",
    # Efficient variants
    "FlashASAMLayer",  # from efficient_attention (SDPA-based)
    "FlashAttnASAMLayer",  # from flash_asam (flash-attn library)
    "EfficientASAMLayer",  # from efficient_attention
    "OptimizedASAMLayer",  # from asam_layer_optimized
    "HybridASAM",  # from flash_asam
    # Continual learning
    "ContinualASAMLayer",
    "ContinualASAMConfig",
    "PrototypeContinualASAMLayer",
    # Sparse patterns
    "LocalSparsePattern",
    "StridedSparsePattern",
    "RandomSparsePattern",
    "ClusteredSparsePattern",
    "HierarchicalSparsePattern",
    # Gating
    "AdaptiveGate",
    "DynamicSparseDenseAttention",
]
