#!/usr/bin/env python3
"""
Benchmark: Original vs Optimized ASAM
======================================

Compare performance between the original and optimized implementations.
"""

import torch
import torch.nn as nn
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from asam import ASAMLayer, ASAMConfig
from asam.asam_layer_optimized import OptimizedASAMLayer


def benchmark_model(model, x, num_iters=10, warmup=3):
    """Benchmark a model."""
    model.eval()
    device = x.device
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_iters):
            torch.cuda.synchronize()
            start = time.time()
            _ = model(x)
            torch.cuda.synchronize()
            times.append((time.time() - start) * 1000)
    
    return sum(times) / len(times)


def main():
    print("="*70)
    print("ASAM Optimization Benchmark")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return
    
    device = torch.device('cuda')
    print(f"\nDevice: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print()
    
    # Test configurations
    configs = [
        (128, 256, 4),
        (256, 256, 4),
        (512, 256, 4),
        (1024, 256, 4),
    ]
    
    print(f"{'Seq Len':<10} {'Dim':<8} {'Original (ms)':<15} {'Optimized (ms)':<15} {'Speedup':<10}")
    print("-"*70)
    
    for seq_len, dim, num_heads in configs:
        x = torch.randn(2, seq_len, dim, device=device)
        
        # Original ASAM
        try:
            torch.cuda.empty_cache()
            config = ASAMConfig(dim=dim, num_heads=num_heads, pattern_type='local')
            model_orig = ASAMLayer(config).to(device)
            time_orig = benchmark_model(model_orig, x)
            del model_orig
            torch.cuda.empty_cache()
        except Exception as e:
            time_orig = float('inf')
        
        # Optimized ASAM
        try:
            torch.cuda.empty_cache()
            model_opt = OptimizedASAMLayer(
                dim=dim, 
                num_heads=num_heads,
                window_size=128,
                pattern_type='local'
            ).to(device)
            time_opt = benchmark_model(model_opt, x)
            del model_opt
            torch.cuda.empty_cache()
        except Exception as e:
            time_opt = float('inf')
        
        speedup = time_orig / time_opt if time_opt > 0 else 0
        print(f"{seq_len:<10} {dim:<8} {time_orig:>13.2f}  {time_opt:>13.2f}  {speedup:>8.2f}x")
    
    print("\n" + "="*70)
    print("Memory Efficiency Test")
    print("="*70)
    
    # Test max sequence length
    dim = 256
    num_heads = 4
    batch_size = 2
    
    for seq_len in [512, 1024, 2048, 4096]:
        x = torch.randn(batch_size, seq_len, dim, device=device)
        
        try:
            torch.cuda.reset_peak_memory_stats()
            model = OptimizedASAMLayer(dim=dim, num_heads=num_heads, window_size=128).to(device)
            with torch.no_grad():
                _ = model(x)
            mem = torch.cuda.max_memory_allocated() / (1024**2)
            print(f"  Seq {seq_len:>4}: {mem:>8.1f} MB - OK")
            del model
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  Seq {seq_len:>4}: OOM")
            else:
                print(f"  Seq {seq_len:>4}: Error - {e}")


if __name__ == "__main__":
    main()
