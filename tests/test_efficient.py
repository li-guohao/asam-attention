#!/usr/bin/env python3
"""
Tests for Efficient Attention implementations
"""

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from asam.efficient_attention import FlashASAMLayer, EfficientASAMLayer


def test_flash_asam_layer():
    """Test FlashASAMLayer basic functionality."""
    layer = FlashASAMLayer(dim=256, num_heads=4)
    x = torch.randn(2, 128, 256)
    
    output, info = layer(x, return_info=True)
    
    assert output.shape == x.shape
    assert info is not None


def test_efficient_asam_layer():
    """Test EfficientASAMLayer basic functionality."""
    layer = EfficientASAMLayer(dim=256, num_heads=4)
    x = torch.randn(2, 128, 256)
    
    output, info = layer(x, return_info=True)
    
    assert output.shape == x.shape
    assert info is not None


def test_different_sequence_lengths():
    """Test with various sequence lengths."""
    layer = FlashASAMLayer(dim=64, num_heads=2)
    
    for seq_len in [64, 128, 256, 512]:
        x = torch.randn(2, seq_len, 64)
        output, _ = layer(x)
        assert output.shape == x.shape


def test_gradient_flow():
    """Test that gradients flow properly."""
    layer = FlashASAMLayer(dim=64, num_heads=2)
    x = torch.randn(2, 32, 64, requires_grad=True)
    
    output, _ = layer(x)
    loss = output.sum()
    loss.backward()
    
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


if __name__ == "__main__":
    test_flash_asam_layer()
    print("test_flash_asam_layer passed")
    
    test_efficient_asam_layer()
    print("test_efficient_asam_layer passed")
    
    test_different_sequence_lengths()
    print("test_different_sequence_lengths passed")
    
    test_gradient_flow()
    print("test_gradient_flow passed")
    
    print("\nAll tests passed!")
