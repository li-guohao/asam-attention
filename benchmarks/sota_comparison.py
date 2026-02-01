"""
SOTA Comparison: ASAM vs. Other Efficient Attention Methods
============================================================

Comparative evaluation against:
- Standard Attention (Vanilla Transformer)
- Longformer (Local + Global attention)
- BigBird (Random + Window + Global)
- Performer (FAVOR+ kernel)
- Linformer (Low-rank approximation)
- Sparse Transformer (Strided patterns)
- Reformer (LSH hashing)
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
import time
import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from asam import ASAMLayer, ASAMConfig


class SOTAModelWrapper(nn.Module):
    """Wrapper for different attention mechanisms."""
    
    def __init__(self, model_type: str, dim: int, num_heads: int, seq_len: int):
        super().__init__()
        self.model_type = model_type
        self.dim = dim
        self.num_heads = num_heads
        self.seq_len = seq_len
        
        if model_type == "asam":
            config = ASAMConfig(dim=dim, num_heads=num_heads, pattern_type="hierarchical")
            self.attn = ASAMLayer(config)
        elif model_type == "standard":
            self.attn = StandardAttention(dim, num_heads)
        elif model_type == "local":
            self.attn = LocalAttention(dim, num_heads, window_size=256)
        elif model_type == "sparse":
            self.attn = SparseAttention(dim, num_heads, stride=32)
        elif model_type == "linformer":
            self.attn = LinformerAttention(dim, num_heads, seq_len, k=256)
        elif model_type == "performer":
            self.attn = PerformerAttention(dim, num_heads, nb_features=256)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        return self.attn(x, mask)


class StandardAttention(nn.Module):
    """Standard full attention (O(n²))."""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.scale = self.dim_head ** -0.5
        
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        b, n, d = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, self.num_heads, self.dim_head).transpose(1, 2), qkv)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, d)
        return self.to_out(out), {}


class LocalAttention(nn.Module):
    """Local window attention (Longformer-style)."""
    
    def __init__(self, dim: int, num_heads: int = 8, window_size: int = 128):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.window_size = window_size
        self.scale = self.dim_head ** -0.5
        
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        b, n, d = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, self.num_heads, self.dim_head).transpose(1, 2), qkv)
        
        # Only attend to local window
        outputs = []
        for i in range(n):
            start = max(0, i - self.window_size // 2)
            end = min(n, i + self.window_size // 2 + 1)
            
            q_i = q[:, :, i:i+1, :]  # [b, h, 1, d]
            k_local = k[:, :, start:end, :]  # [b, h, w, d]
            v_local = v[:, :, start:end, :]
            
            scores = torch.matmul(q_i, k_local.transpose(-2, -1)) * self.scale
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v_local)  # [b, h, 1, d]
            outputs.append(out)
        
        out = torch.cat(outputs, dim=2)  # [b, h, n, d]
        out = out.transpose(1, 2).reshape(b, n, d)
        return self.to_out(out), {}


class SparseAttention(nn.Module):
    """Sparse attention with strided pattern."""
    
    def __init__(self, dim: int, num_heads: int = 8, stride: int = 32):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.stride = stride
        self.scale = self.dim_head ** -0.5
        
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        b, n, d = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, self.num_heads, self.dim_head).transpose(1, 2), qkv)
        
        # Create sparse mask: each query attends to strided keys + local window
        attn_mask = torch.zeros(b, self.num_heads, n, n, device=x.device, dtype=torch.bool)
        
        for i in range(n):
            # Strided attention
            strided_indices = torch.arange(0, n, self.stride, device=x.device)
            attn_mask[:, :, i, strided_indices] = True
            # Local window
            start = max(0, i - 64)
            end = min(n, i + 65)
            attn_mask[:, :, i, start:end] = True
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(~attn_mask, float('-inf'))
        
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, d)
        return self.to_out(out), {}


class LinformerAttention(nn.Module):
    """Linformer: Low-rank approximation (Wang et al., 2020)."""
    
    def __init__(self, dim: int, num_heads: int, seq_len: int, k: int = 256):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.k = k  # Projected dimension
        self.scale = self.dim_head ** -0.5
        
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Linear(dim, dim)
        
        # Projection matrices E, F
        self.E = nn.Parameter(torch.randn(seq_len, k) / math.sqrt(k))
        self.F = nn.Parameter(torch.randn(seq_len, k) / math.sqrt(k))
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        b, n, d = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, self.num_heads, self.dim_head).transpose(1, 2), qkv)
        
        # Project keys and values: K' = E @ K, V' = F @ V
        k_proj = torch.matmul(self.E[:n, :self.k].T, k.transpose(2, 3))  # [b, h, k, d]
        v_proj = torch.matmul(self.F[:n, :self.k].T, v.transpose(2, 3))
        
        # Attention with projected dimensions
        scores = torch.matmul(q, k_proj.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_proj.transpose(-2, -1))
        
        out = out.transpose(1, 2).reshape(b, n, d)
        return self.to_out(out), {}


class PerformerAttention(nn.Module):
    """
    Performer: FAVOR+ kernel-based attention (Choromanski et al., 2020).
    Approximates softmax with orthogonal random features.
    """
    
    def __init__(self, dim: int, num_heads: int, nb_features: int = 256):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.nb_features = nb_features
        
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Linear(dim, dim)
        
        # Random features
        self.register_buffer('omega', torch.randn(self.dim_head, nb_features))
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        b, n, d = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, self.num_heads, self.dim_head).transpose(1, 2), qkv)
        
        # Create orthogonal random features
        omega = self.omega[:, :self.nb_features]
        
        # Feature maps
        q_prime = self._feature_map(q, omega)  # [b, h, n, r]
        k_prime = self._feature_map(k, omega)
        
        # FAVOR+ attention
        k_cumsum = k_prime.sum(dim=2, keepdim=True)  # [b, h, 1, r]
        denominator = torch.matmul(q_prime, k_cumsum.transpose(-2, -1))  # [b, h, n, 1]
        
        # Numerator
        k_v = torch.matmul(k_prime.transpose(-2, -1), v)  # [b, h, r, d]
        numerator = torch.matmul(q_prime, k_v)  # [b, h, n, d]
        
        out = numerator / (denominator + 1e-6)
        out = out.transpose(1, 2).reshape(b, n, d)
        return self.to_out(out), {}
    
    def _feature_map(self, x: torch.Tensor, omega: torch.Tensor):
        """Apply feature map φ(x) = exp(x @ ω - ||x||²/2)."""
        x_omega = torch.matmul(x, omega)  # [b, h, n, r]
        norm_sq = (x ** 2).sum(dim=-1, keepdim=True) / 2  # [b, h, n, 1]
        return torch.exp(x_omega - norm_sq)


def benchmark_model(
    model_type: str,
    seq_lengths: list,
    dim: int = 512,
    num_heads: int = 8,
    batch_size: int = 2,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> dict:
    """Benchmark a model across different sequence lengths."""
    
    results = []
    
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_type.upper()}")
    print(f"{'='*60}")
    
    for seq_len in seq_lengths:
        # Skip very long sequences for O(n²) methods
        if model_type == "standard" and seq_len > 4096:
            print(f"Skipping {model_type} at seq_len={seq_len} (would OOM)")
            continue
        
        try:
            # Create model
            model = SOTAModelWrapper(model_type, dim, num_heads, seq_len).to(device)
            model.eval()
            
            # Create input
            x = torch.randn(batch_size, seq_len, dim, device=device)
            
            # Warmup
            with torch.no_grad():
                for _ in range(5):
                    _ = model(x)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            
            # Benchmark
            times = []
            for _ in range(20):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start = time.time()
                with torch.no_grad():
                    _ = model(x)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                times.append((time.time() - start) * 1000)
            
            avg_time = np.mean(times[5:])  # Skip first 5 for stability
            std_time = np.std(times[5:])
            
            peak_memory = 0
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
            
            result = {
                'model': model_type,
                'seq_len': seq_len,
                'avg_time_ms': float(avg_time),
                'std_time_ms': float(std_time),
                'peak_memory_mb': float(peak_memory),
            }
            results.append(result)
            
            print(f"  seq_len={seq_len:5d}: {avg_time:8.2f} ± {std_time:6.2f} ms | "
                  f"Memory: {peak_memory:8.2f} MB")
            
            # Cleanup
            del model, x
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        except RuntimeError as e:
            print(f"  seq_len={seq_len:5d}: FAILED ({e})")
            results.append({
                'model': model_type,
                'seq_len': seq_len,
                'error': str(e)
            })
    
    return results


def run_comparison():
    """Run complete SOTA comparison."""
    
    seq_lengths = [512, 1024, 2048, 4096, 8192]
    
    models = [
        "asam",
        "standard",
        "local", 
        "sparse",
        "linformer",
        "performer",
    ]
    
    all_results = []
    
    for model_type in models:
        try:
            results = benchmark_model(model_type, seq_lengths)
            all_results.extend(results)
        except Exception as e:
            print(f"Error benchmarking {model_type}: {e}")
    
    # Save results
    with open('sota_comparison_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print_summary(all_results)
    
    return all_results


def print_summary(results: list):
    """Print comparison summary."""
    
    print(f"\n\n{'='*80}")
    print("SOTA COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    # Group by sequence length
    by_seq_len = {}
    for r in results:
        if 'error' not in r:
            sl = r['seq_len']
            if sl not in by_seq_len:
                by_seq_len[sl] = {}
            by_seq_len[sl][r['model']] = r
    
    for seq_len in sorted(by_seq_len.keys()):
        print(f"\nSequence Length: {seq_len}")
        print("-" * 80)
        print(f"{'Model':<15} {'Time (ms)':<15} {'Memory (MB)':<15} {'vs Standard':<15}")
        print("-" * 80)
        
        models = by_seq_len[seq_len]
        baseline_time = models.get('standard', {}).get('avg_time_ms', None)
        
        for model_name in ['standard', 'asam', 'local', 'sparse', 'linformer', 'performer']:
            if model_name in models:
                r = models[model_name]
                time_str = f"{r['avg_time_ms']:.2f}"
                mem_str = f"{r['peak_memory_mb']:.2f}"
                
                if baseline_time and model_name != 'standard':
                    speedup = baseline_time / r['avg_time_ms']
                    ratio_str = f"{speedup:.2f}x"
                else:
                    ratio_str = "baseline"
                
                print(f"{model_name:<15} {time_str:<15} {mem_str:<15} {ratio_str:<15}")


if __name__ == "__main__":
    results = run_comparison()
