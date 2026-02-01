"""
Adaptive Sparse Attention Mechanism (ASAM)
============================================

A novel attention mechanism that combines sparse attention patterns with 
adaptive gating to efficiently process long sequences.

Author: Guohao Li
License: MIT
"""

from .asam_layer import ASAMLayer, ASAMConfig
from .sparse_patterns import (
    LocalSparsePattern, 
    StridedSparsePattern, 
    RandomSparsePattern,
    ClusteredSparsePattern
)
from .adaptive_gate import AdaptiveGate

__version__ = "0.1.0"
__all__ = [
    "ASAMLayer",
    "ASAMConfig", 
    "LocalSparsePattern",
    "StridedSparsePattern",
    "RandomSparsePattern",
    "ClusteredSparsePattern",
    "AdaptiveGate",
]
