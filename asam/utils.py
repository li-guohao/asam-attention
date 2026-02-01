"""
Utility functions for ASAM.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import time
import functools


def compute_attention_sparsity(attention_mask: torch.Tensor) -> float:
    """
    Compute the sparsity ratio of an attention mask.
    
    Args:
        attention_mask: Boolean tensor [seq_len, seq_len] or [heads, seq_len, seq_len]
        
    Returns:
        Sparsity ratio (0.0 = dense, 1.0 = completely sparse)
    """
    total_elements = attention_mask.numel()
    zero_elements = (~attention_mask).sum().item()
    return zero_elements / total_elements


def measure_attention_efficiency(
    attn_fn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_warmup: int = 3,
    num_runs: int = 10
) -> Tuple[float, float, float]:
    """
    Measure time and memory efficiency of attention computation.
    
    Args:
        attn_fn: Attention function to benchmark
        q, k, v: Input tensors
        num_warmup: Number of warmup iterations
        num_runs: Number of benchmark runs
        
    Returns:
        avg_time_ms: Average time in milliseconds
        peak_memory_mb: Peak memory usage in MB
        flops: Estimated FLOPs
    """
    device = q.device
    
    # Warmup
    for _ in range(num_warmup):
        _ = attn_fn(q, k, v)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        start = time.time()
        _ = attn_fn(q, k, v)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.time()
        
        times.append((end - start) * 1000)  # Convert to ms
    
    avg_time = sum(times) / len(times)
    
    # Memory
    peak_memory = 0
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    
    # FLOPs estimation
    batch, heads, seq_len, dim_head = q.shape
    flops = 2 * batch * heads * seq_len * seq_len * dim_head  # QK^T + softmax + @V
    
    return avg_time, peak_memory, flops


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create causal (lower triangular) attention mask."""
    return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))


def create_local_mask(
    seq_len: int, 
    window_size: int, 
    device: torch.device
) -> torch.Tensor:
    """Create local (sliding window) attention mask."""
    mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)
    for i in range(seq_len):
        start = max(0, i - window_size // 2)
        end = min(seq_len, i + window_size // 2 + 1)
        mask[i, start:end] = True
    return mask


def memory_efficient_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int = 1024,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Memory-efficient attention that processes in chunks.
    
    Useful for very long sequences that don't fit in memory.
    """
    batch, heads, seq_len, dim_head = q.shape
    
    if seq_len <= chunk_size:
        # Standard attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (dim_head ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v)
    
    # Chunked attention
    outputs = []
    for i in range(0, seq_len, chunk_size):
        end_i = min(i + chunk_size, seq_len)
        q_chunk = q[:, :, i:end_i, :]
        
        scores = torch.matmul(q_chunk, k.transpose(-2, -1)) / (dim_head ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(~mask[i:end_i], float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        outputs.append(out)
    
    return torch.cat(outputs, dim=2)


class GradientCheckpointingWrapper(nn.Module):
    """Wrapper to enable gradient checkpointing for any module."""
    
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
    
    def forward(self, *args, **kwargs):
        if self.training:
            return torch.utils.checkpoint.checkpoint(
                self.module, *args, **kwargs, use_reentrant=False
            )
        return self.module(*args, **kwargs)


def init_weights(module: nn.Module, std: float = 0.02):
    """Initialize weights with truncated normal distribution."""
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        torch.nn.init.ones_(module.weight)
        torch.nn.init.zeros_(module.bias)


class PerformanceMonitor:
    """Monitor and log performance metrics."""
    
    def __init__(self):
        self.metrics = {}
    
    def record(self, name: str, value: float):
        """Record a metric."""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def get_average(self, name: str) -> float:
        """Get average of a metric."""
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return 0.0
        return sum(self.metrics[name]) / len(self.metrics[name])
    
    def get_summary(self) -> dict:
        """Get summary of all metrics."""
        return {
            name: {
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'count': len(values)
            }
            for name, values in self.metrics.items()
        }
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = {}
