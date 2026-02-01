"""
ASAM Performance Benchmark
==========================

Benchmark ASAM against standard attention for various sequence lengths.
"""

import torch
import torch.nn as nn
import time
from typing import Dict, List
import json

from asam import ASAMLayer, ASAMConfig


class StandardAttention(nn.Module):
    """Standard full attention for comparison."""
    
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.scale = self.dim_head ** -0.5
        
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        batch, seq_len, dim = x.shape
        
        # Attention
        residual = x
        x = self.norm(x)
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(batch, seq_len, self.num_heads, self.dim_head).transpose(1, 2), qkv)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch, seq_len, dim)
        out = self.to_out(out)
        
        x = residual + out
        
        # FFN
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = residual + x
        
        return x


def benchmark_model(model, x, num_warmup=3, num_runs=10):
    """Benchmark a model."""
    model.eval()
    device = x.device
    
    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    # Measure memory
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.time()
        with torch.no_grad():
            _ = model(x)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        times.append((time.time() - start) * 1000)  # ms
    
    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
    
    memory_mb = 0
    if torch.cuda.is_available():
        memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    return {
        'avg_time_ms': avg_time,
        'std_time_ms': std_time,
        'memory_mb': memory_mb,
    }


def run_benchmarks():
    """Run comprehensive benchmarks."""
    print("=" * 80)
    print("ASAM Performance Benchmark")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Configurations to test
    configs = [
        {"seq_len": 512, "dim": 512},
        {"seq_len": 1024, "dim": 512},
        {"seq_len": 2048, "dim": 512},
        {"seq_len": 4096, "dim": 512},
    ]
    
    results = []
    
    for cfg in configs:
        seq_len = cfg['seq_len']
        dim = cfg['dim']
        batch_size = 2
        
        print(f"\n{'-' * 60}")
        print(f"Sequence Length: {seq_len}, Dim: {dim}, Batch: {batch_size}")
        print(f"{'-' * 60}")
        
        x = torch.randn(batch_size, seq_len, dim, device=device)
        
        # Standard Attention
        try:
            std_attn = StandardAttention(dim).to(device)
            std_result = benchmark_model(std_attn, x, num_runs=5)
            print(f"Standard Attention:")
            print(f"  Time: {std_result['avg_time_ms']:.2f} ± {std_result['std_time_ms']:.2f} ms")
            print(f"  Memory: {std_result['memory_mb']:.2f} MB")
        except RuntimeError as e:
            print(f"Standard Attention: FAILED ({e})")
            std_result = None
        
        # ASAM with different patterns
        pattern_results = {}
        for pattern in ["local", "strided", "hierarchical"]:
            config = ASAMConfig(
                dim=dim,
                num_heads=8,
                dim_head=dim // 8,
                pattern_type=pattern,
                use_adaptive_gate=True,
            )
            
            try:
                asam = ASAMLayer(config).to(device)
                asam_result = benchmark_model(asam, x, num_runs=5)
                
                print(f"\nASAM ({pattern}):")
                print(f"  Time: {asam_result['avg_time_ms']:.2f} ± {asam_result['std_time_ms']:.2f} ms")
                print(f"  Memory: {asam_result['memory_mb']:.2f} MB")
                
                if std_result:
                    speedup = std_result['avg_time_ms'] / asam_result['avg_time_ms']
                    mem_reduction = std_result['memory_mb'] / max(asam_result['memory_mb'], 1)
                    print(f"  Speedup: {speedup:.2f}x")
                    print(f"  Memory reduction: {mem_reduction:.2f}x")
                
                pattern_results[pattern] = asam_result
                
            except RuntimeError as e:
                print(f"ASAM ({pattern}): FAILED ({e})")
        
        results.append({
            'config': cfg,
            'standard': std_result,
            'asam': pattern_results,
        })
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    print(f"\n{'Seq Len':<10} {'Pattern':<15} {'Time (ms)':<15} {'Memory (MB)':<15} {'Speedup':<10}")
    print("-" * 80)
    
    for r in results:
        cfg = r['config']
        seq_len = cfg['seq_len']
        
        if r['standard']:
            print(f"{seq_len:<10} {'Standard':<15} {r['standard']['avg_time_ms']:<15.2f} {r['standard']['memory_mb']:<15.2f} {'1.00x':<10}")
        
        for pattern, result in r['asam'].items():
            speedup = r['standard']['avg_time_ms'] / result['avg_time_ms'] if r['standard'] else 0
            print(f"{seq_len:<10} {pattern:<15} {result['avg_time_ms']:<15.2f} {result['memory_mb']:<15.2f} {speedup:<10.2f}x")
    
    return results


if __name__ == "__main__":
    results = run_benchmarks()
    
    # Save results
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to benchmark_results.json")
