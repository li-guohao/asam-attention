#!/usr/bin/env python3
"""
GTX 3060 12GB Baseline Experiments
===================================

Conservative experiments designed for GTX 3060 12GB:
- Automatic memory monitoring
- Graceful OOM handling
- Progressive scaling
- Result saving and visualization

Expected runtime: 2-6 hours depending on settings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import json
import time
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from asam import ASAMLayer, ASAMConfig

# Create results directory
RESULTS_DIR = Path("experiments/results_3060")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_gpu_memory():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0


def print_gpu_status():
    """Print current GPU status."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        print(f"  GPU Memory: {allocated:.1f}MB / {total:.1f}MB "
              f"(Reserved: {reserved:.1f}MB)")


def safe_run(func, *args, **kwargs):
    """Run function with OOM protection."""
    try:
        return func(*args, **kwargs)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"  [WARNING] OOM Error! Clearing cache...")
            torch.cuda.empty_cache()
            return None
        raise


class SimpleBenchmarkModel(nn.Module):
    """Simple model for baseline testing."""
    
    def __init__(self, config, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(1000, config.dim)
        self.layers = nn.ModuleList([
            ASAMLayer(config) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(config.dim)
        self.head = nn.Linear(config.dim, 10)
    
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x, _ = layer(x)
        x = self.norm(x)
        return self.head(x.mean(dim=1))


class BaselineTransformer(nn.Module):
    """Standard transformer for comparison."""
    
    def __init__(self, dim, num_heads, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(1000, dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, 
            dim_feedforward=dim*4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 10)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.norm(x)
        return self.head(x.mean(dim=1))


def benchmark_forward_speed(model, seq_lengths, batch_size=2, device='cuda'):
    """Benchmark forward pass speed."""
    results = []
    
    model = model.to(device)
    model.eval()
    
    print(f"\n{'='*60}")
    print(f"Benchmarking Forward Speed (Batch={batch_size})")
    print(f"{'='*60}")
    print(f"{'Seq Len':<10} {'Time (ms)':<12} {'Memory (MB)':<15} {'Status':<10}")
    print("-"*60)
    
    for seq_len in seq_lengths:
        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        try:
            # Create input
            x = torch.randint(0, 1000, (batch_size, seq_len), device=device)
            
            # Warmup
            for _ in range(3):
                with torch.no_grad():
                    _ = model(x)
            
            torch.cuda.synchronize()
            
            # Benchmark
            times = []
            for _ in range(10):
                torch.cuda.synchronize()
                start = time.time()
                
                with torch.no_grad():
                    _ = model(x)
                
                torch.cuda.synchronize()
                times.append((time.time() - start) * 1000)
            
            avg_time = np.mean(times)
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
            
            print(f"{seq_len:<10} {avg_time:>10.2f}  {peak_mem:>13.1f}  {'OK':<10}")
            
            results.append({
                'seq_len': seq_len,
                'time_ms': float(avg_time),
                'memory_mb': float(peak_mem),
                'success': True
            })
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{seq_len:<10} {'OOM':>10}  {'-':>13}  {'FAIL':<10}")
                results.append({
                    'seq_len': seq_len,
                    'time_ms': None,
                    'memory_mb': None,
                    'success': False,
                    'error': 'OOM'
                })
                torch.cuda.empty_cache()
            else:
                raise
    
    return results


def benchmark_training(model, seq_len=512, batch_size=2, num_steps=50, device='cuda'):
    """Benchmark training speed."""
    print(f"\n{'='*60}")
    print(f"Benchmarking Training (Seq={seq_len}, Batch={batch_size})")
    print(f"{'='*60}")
    
    model = model.to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    times = []
    losses = []
    
    for step in range(num_steps):
        x = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        y = torch.randint(0, 10, (batch_size,), device=device)
        
        torch.cuda.synchronize()
        start = time.time()
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        torch.cuda.synchronize()
        elapsed = (time.time() - start) * 1000
        
        times.append(elapsed)
        losses.append(loss.item())
        
        if (step + 1) % 10 == 0:
            print(f"  Step {step+1}/{num_steps}: {np.mean(times[-10:]):.2f}ms/step, "
                  f"Loss: {np.mean(losses[-10:]):.4f}")
    
    return {
        'avg_time_ms': float(np.mean(times)),
        'std_time_ms': float(np.std(times)),
        'final_loss': float(losses[-1]),
        'loss_curve': losses
    }


def test_sparse_patterns(seq_len=256, batch_size=2, device='cuda'):
    """Test different sparse patterns."""
    from asam.sparse_patterns import (
        LocalSparsePattern, StridedSparsePattern, 
        HierarchicalSparsePattern
    )
    
    print(f"\n{'='*60}")
    print(f"Testing Sparse Patterns (Seq={seq_len})")
    print(f"{'='*60}")
    
    patterns = [
        ('local', ASAMConfig(dim=256, num_heads=4, pattern_type='local')),
        ('strided', ASAMConfig(dim=256, num_heads=4, pattern_type='strided')),
        ('hierarchical', ASAMConfig(dim=256, num_heads=4, pattern_type='hierarchical')),
    ]
    
    results = []
    x = torch.randn(batch_size, seq_len, 256, device=device)
    
    for name, config in patterns:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        try:
            model = ASAMLayer(config).to(device)
            model.eval()
            
            # Warmup
            with torch.no_grad():
                for _ in range(3):
                    _ = model(x)
            
            # Benchmark
            torch.cuda.synchronize()
            start = time.time()
            
            with torch.no_grad():
                for _ in range(20):
                    _, info = model(x, return_info=True)
            
            torch.cuda.synchronize()
            elapsed = (time.time() - start) / 20 * 1000
            
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
            sparse_ratio = info.get('sparse_ratio', 0) if info else 0
            
            print(f"  {name:15s}: {elapsed:8.2f}ms | {peak_mem:8.1f}MB | "
                  f"Sparse: {sparse_ratio:.1%}")
            
            results.append({
                'pattern': name,
                'time_ms': float(elapsed),
                'memory_mb': float(peak_mem),
                'sparse_ratio': float(sparse_ratio),
                'success': True
            })
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  {name:15s}: OOM")
                results.append({
                    'pattern': name,
                    'success': False,
                    'error': 'OOM'
                })
                torch.cuda.empty_cache()
    
    return results


def test_adaptive_gate_behavior(seq_len=512, device='cuda'):
    """Test adaptive gate behavior with different inputs."""
    print(f"\n{'='*60}")
    print(f"Testing Adaptive Gate Behavior (Seq={seq_len})")
    print(f"{'='*60}")
    
    config = ASAMConfig(
        dim=256, num_heads=4, 
        pattern_type='hierarchical',
        use_adaptive_gate=True
    )
    model = ASAMLayer(config).to(device)
    model.eval()
    
    # Different input types
    test_cases = [
        ('Random (high entropy)', torch.randn(2, seq_len, 256, device=device) * 2),
        ('Structured (low entropy)', torch.randn(2, seq_len, 256, device=device).cumsum(dim=1)),
        ('Sparse pattern', torch.cat([
            torch.zeros(2, seq_len//2, 256, device=device),
            torch.randn(2, seq_len//2, 256, device=device)
        ], dim=1)),
    ]
    
    results = []
    
    for name, x in test_cases:
        with torch.no_grad():
            _, info = model(x, return_info=True)
        
        if info:
            gate_mean = info['gate_values'].mean().item()
            confidence = info['confidence'].mean().item()
            sparse_ratio = info['sparse_ratio']
            
            print(f"  {name:25s}: Gate={gate_mean:.3f}, "
                  f"Conf={confidence:.3f}, Sparse={sparse_ratio:.1%}")
            
            results.append({
                'input_type': name,
                'gate_mean': float(gate_mean),
                'confidence': float(confidence),
                'sparse_ratio': float(sparse_ratio)
            })
    
    return results


def save_results(all_results, timestamp):
    """Save all results to JSON and generate plots."""
    
    # Save JSON
    result_file = RESULTS_DIR / f"results_{timestamp}.json"
    with open(result_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {result_file}")
    
    # Generate plots
    generate_plots(all_results, timestamp)


def generate_plots(results, timestamp):
    """Generate visualization plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Speed comparison
    ax = axes[0, 0]
    if 'forward_speed' in results:
        for model_name, data in results['forward_speed'].items():
            seq_lens = [r['seq_len'] for r in data if r['success']]
            times = [r['time_ms'] for r in data if r['success']]
            if seq_lens:
                ax.plot(seq_lens, times, 'o-', label=model_name, linewidth=2, markersize=8)
        
        ax.set_xlabel('Sequence Length')
        ax.set_ylabel('Time (ms)')
        ax.set_title('Forward Pass Speed')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 2: Memory usage
    ax = axes[0, 1]
    if 'forward_speed' in results:
        for model_name, data in results['forward_speed'].items():
            seq_lens = [r['seq_len'] for r in data if r['success']]
            mems = [r['memory_mb'] for r in data if r['success']]
            if seq_lens:
                ax.plot(seq_lens, mems, 's-', label=model_name, linewidth=2, markersize=8)
        
        ax.set_xlabel('Sequence Length')
        ax.set_ylabel('Memory (MB)')
        ax.set_title('Peak Memory Usage')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Pattern comparison
    ax = axes[1, 0]
    if 'sparse_patterns' in results:
        patterns = [r['pattern'] for r in results['sparse_patterns'] if r.get('success')]
        times = [r['time_ms'] for r in results['sparse_patterns'] if r.get('success')]
        if patterns:
            ax.bar(patterns, times, color=['#3498DB', '#E74C3C', '#2ECC71'], alpha=0.8)
            ax.set_xlabel('Pattern Type')
            ax.set_ylabel('Time (ms)')
            ax.set_title('Sparse Pattern Comparison')
            ax.grid(axis='y', alpha=0.3)
    
    # Plot 4: Gate behavior
    ax = axes[1, 1]
    if 'gate_behavior' in results:
        types = [r['input_type'] for r in results['gate_behavior']]
        gates = [r['gate_mean'] for r in results['gate_behavior']]
        if types:
            colors = ['#2ECC71', '#F39C12', '#E74C3C']
            ax.barh(types, gates, color=colors, alpha=0.8)
            ax.set_xlabel('Gate Value (Sparse Attention Weight)')
            ax.set_title('Adaptive Gate Behavior')
            ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
            ax.grid(axis='x', alpha=0.3)
    
    plt.suptitle(f'ASAM 3060 Baseline Results ({timestamp})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_file = RESULTS_DIR / f"plots_{timestamp}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Plots saved to: {plot_file}")
    plt.close()


def main():
    """Main experiment runner."""
    
    print("="*70)
    print("ASAM Baseline Experiments for GTX 3060 12GB")
    print("="*70)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: No CUDA device available!")
        return
    
    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    print(f"\nDevice: {gpu_name}")
    print(f"Total Memory: {total_mem:.1f} GB")
    print_gpu_status()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {
        'timestamp': timestamp,
        'device': gpu_name,
        'total_memory_gb': total_mem
    }
    
    # Test 1: Forward Speed (progressive seq lengths)
    print("\n" + "="*70)
    print("TEST 1: Forward Pass Speed")
    print("="*70)
    
    seq_lengths = [128, 256, 512, 1024, 1536, 2048]
    
    asam_config = ASAMConfig(dim=256, num_heads=4, pattern_type='hierarchical')
    asam_model = SimpleBenchmarkModel(asam_config, num_layers=2)
    
    standard_model = BaselineTransformer(dim=256, num_heads=4, num_layers=2)
    
    all_results['forward_speed'] = {
        'ASAM': benchmark_forward_speed(asam_model, seq_lengths, batch_size=2, device=device),
        'Standard': benchmark_forward_speed(standard_model, seq_lengths[:4], batch_size=2, device=device)
    }
    
    # Test 2: Training Speed
    print("\n" + "="*70)
    print("TEST 2: Training Speed")
    print("="*70)
    
    all_results['training'] = {
        'ASAM': benchmark_training(asam_model, seq_len=512, batch_size=2, num_steps=50, device=device)
    }
    
    # Test 3: Sparse Patterns
    print("\n" + "="*70)
    print("TEST 3: Sparse Pattern Comparison")
    print("="*70)
    
    all_results['sparse_patterns'] = test_sparse_patterns(seq_len=256, batch_size=2, device=device)
    
    # Test 4: Adaptive Gate
    print("\n" + "="*70)
    print("TEST 4: Adaptive Gate Behavior")
    print("="*70)
    
    all_results['gate_behavior'] = test_adaptive_gate_behavior(seq_len=512, device=device)
    
    # Save results
    save_results(all_results, timestamp)
    
    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETED!")
    print("="*70)
    print(f"\nResults saved in: {RESULTS_DIR}")
    print(f"Next steps:")
    print(f"  1. Review results in: results_{timestamp}.json")
    print(f"  2. Check plots in: plots_{timestamp}.png")
    print(f"  3. If successful, scale up to larger experiments")


if __name__ == "__main__":
    main()
