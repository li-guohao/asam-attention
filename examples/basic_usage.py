#!/usr/bin/env python3
"""
ASAM Basic Usage Example
=========================

Demonstrates basic usage of ASAM layer.
"""

import torch
from asam import ASAMLayer, ASAMConfig


def main():
    print("="*60)
    print("ASAM Basic Usage Example")
    print("="*60)
    
    # Configuration
    dim = 256
    num_heads = 4
    seq_len = 512
    batch_size = 2
    
    # Create ASAM config
    config = ASAMConfig(
        dim=dim,
        num_heads=num_heads,
        pattern_type='local',
        use_adaptive_gate=True,
    )
    
    # Create layer
    layer = ASAMLayer(config)
    
    # Create input
    x = torch.randn(batch_size, seq_len, dim)
    
    # Forward pass
    output, info = layer(x, return_info=True)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Sparse ratio: {info['sparse_ratio']:.1%}")
    print(f"Gate mean: {info['gate_values'].mean().item():.3f}")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
