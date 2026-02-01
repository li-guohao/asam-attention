#!/usr/bin/env python3
"""
Basic tests for ASAM layer
"""

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from asam import ASAMLayer, ASAMConfig


def test_asam_layer_creation():
    """Test ASAM layer can be created."""
    config = ASAMConfig(dim=256, num_heads=4)
    layer = ASAMLayer(config)
    assert layer is not None


def test_asam_forward():
    """Test forward pass."""
    config = ASAMConfig(dim=256, num_heads=4)
    layer = ASAMLayer(config)
    
    x = torch.randn(2, 128, 256)
    output, info = layer(x, return_info=True)
    
    assert output.shape == x.shape
    assert info is not None


def test_different_patterns():
    """Test different sparse patterns."""
    for pattern_type in ['local', 'strided', 'hierarchical']:
        config = ASAMConfig(dim=256, num_heads=4, pattern_type=pattern_type)
        layer = ASAMLayer(config)
        
        x = torch.randn(2, 256, 256)
        output, _ = layer(x)
        
        assert output.shape == x.shape


def test_without_adaptive_gate():
    """Test without adaptive gate."""
    config = ASAMConfig(dim=256, num_heads=4, use_adaptive_gate=False)
    layer = ASAMLayer(config)
    
    x = torch.randn(2, 128, 256)
    output, _ = layer(x)
    
    assert output.shape == x.shape


if __name__ == "__main__":
    test_asam_layer_creation()
    print("test_asam_layer_creation passed")
    
    test_asam_forward()
    print("test_asam_forward passed")
    
    test_different_patterns()
    print("test_different_patterns passed")
    
    test_without_adaptive_gate()
    print("test_without_adaptive_gate passed")
    
    print("\nAll basic tests passed!")
