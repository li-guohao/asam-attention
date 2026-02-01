"""
ASAM Tutorial 1: Getting Started
=================================

This script introduces the basics of Adaptive Sparse Attention Mechanism (ASAM).

Run this script to see ASAM in action!
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from asam import ASAMLayer, ASAMConfig


def tutorial_1_basic_usage():
    """Tutorial 1: Basic ASAM usage."""
    print("="*60)
    print("Tutorial 1: Basic ASAM Usage")
    print("="*60)
    
    # Configuration
    config = ASAMConfig(
        dim=256,
        num_heads=4,
        pattern_type="hierarchical",
        use_adaptive_gate=True,
    )
    
    # Create layer
    layer = ASAMLayer(config)
    layer.eval()
    
    print(f"\n✓ ASAM layer created")
    print(f"  Parameters: {sum(p.numel() for p in layer.parameters()):,}")
    
    # Forward pass
    x = torch.randn(2, 512, config.dim)
    
    with torch.no_grad():
        output, info = layer(x, return_info=True)
    
    print(f"\nForward pass:")
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Sparse ratio: {info['sparse_ratio']:.2%}")
    print(f"  Confidence:   {info['confidence'].mean():.4f}")


def tutorial_2_pattern_comparison():
    """Tutorial 2: Compare different sparse patterns."""
    print("\n" + "="*60)
    print("Tutorial 2: Sparse Pattern Comparison")
    print("="*60)
    
    from asam.sparse_patterns import (
        LocalSparsePattern,
        StridedSparsePattern,
        HierarchicalSparsePattern
    )
    
    seq_len = 128
    
    patterns = [
        ("Local (window=32)", LocalSparsePattern(seq_len, window_size=32)),
        ("Strided (stride=16)", StridedSparsePattern(seq_len, stride=16)),
        ("Hierarchical", HierarchicalSparsePattern(seq_len, scales=[4, 16])),
    ]
    
    print(f"\nSparse patterns for sequence length {seq_len}:")
    print("-" * 60)
    
    for name, pattern in patterns:
        mask = pattern.build_pattern()
        if mask.dim() == 3:
            mask = mask[0]  # Take first head
        
        sparsity = 1.0 - mask.float().mean().item()
        active = mask.sum().item()
        total = mask.numel()
        
        print(f"{name:25s} | Sparsity: {sparsity:6.2%} | Active: {active:6d}/{total:6d}")


def tutorial_3_sequence_scaling():
    """Tutorial 3: Test with different sequence lengths."""
    print("\n" + "="*60)
    print("Tutorial 3: Sequence Length Scaling")
    print("="*60)
    
    import time
    
    config = ASAMConfig(dim=256, num_heads=4, pattern_type="hierarchical")
    layer = ASAMLayer(config)
    layer.eval()
    
    seq_lengths = [128, 256, 512, 1024, 2048]
    
    print(f"\n{'Seq Length':<12} {'Time (ms)':<12} {'Sparse Ratio':<15}")
    print("-" * 45)
    
    for seq_len in seq_lengths:
        x = torch.randn(2, seq_len, config.dim)
        
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = layer(x)
        
        # Measure
        start = time.time()
        for _ in range(10):
            with torch.no_grad():
                output, info = layer(x)
        elapsed = (time.time() - start) / 10 * 1000
        
        print(f"{seq_len:<12} {elapsed:>10.2f}  {info['sparse_ratio']:>13.2%}")


def tutorial_4_adaptive_behavior():
    """Tutorial 4: Demonstrate adaptive gating."""
    print("\n" + "="*60)
    print("Tutorial 4: Adaptive Gating Behavior")
    print("="*60)
    
    config = ASAMConfig(dim=256, num_heads=4, use_adaptive_gate=True)
    layer = ASAMLayer(config)
    layer.eval()
    
    # Create inputs with different characteristics
    test_cases = [
        ("Random (high entropy)", torch.randn(1, 256, config.dim)),
        ("Smooth (low entropy)", torch.randn(1, 256, config.dim).cumsum(dim=1)),
        ("Sparse pattern", torch.cat([torch.zeros(1, 200, config.dim), 
                                      torch.randn(1, 56, config.dim)], dim=1)),
    ]
    
    print("\nAdaptive gating for different input types:")
    print("-" * 60)
    print(f"{'Input Type':<25} {'Sparse Ratio':<15} {'Confidence':<12}")
    print("-" * 60)
    
    for name, x in test_cases:
        with torch.no_grad():
            _, info = layer(x, return_info=True)
        
        print(f"{name:<25} {info['sparse_ratio']:>13.2%}  {info['confidence'].mean():>10.4f}")


def main():
    """Run all tutorials."""
    print("\n" + "="*60)
    print("ASAM Interactive Tutorials")
    print("="*60)
    
    tutorial_1_basic_usage()
    tutorial_2_pattern_comparison()
    tutorial_3_sequence_scaling()
    tutorial_4_adaptive_behavior()
    
    print("\n" + "="*60)
    print("Tutorials completed!")
    print("="*60)
    print("\nNext steps:")
    print("  - See examples/ for more complex usage")
    print("  - Run benchmarks/ for performance evaluation")
    print("  - Read docs/ for detailed documentation")


if __name__ == "__main__":
    main()
