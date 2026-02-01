"""
Unit tests for ASAM.
"""

import torch
import pytest
from asam import ASAMLayer, ASAMConfig, AdaptiveGate
from asam.sparse_patterns import (
    LocalSparsePattern,
    StridedSparsePattern,
    ClusteredSparsePattern,
    HierarchicalSparsePattern,
)


class TestSparsePatterns:
    """Test sparse attention patterns."""
    
    def test_local_pattern_shape(self):
        seq_len = 128
        pattern = LocalSparsePattern(seq_len, window_size=32)
        mask = pattern.build_pattern()
        assert mask.shape == (seq_len, seq_len)
        assert mask.dtype == torch.bool
    
    def test_local_pattern_sparsity(self):
        seq_len = 128
        window_size = 32
        pattern = LocalSparsePattern(seq_len, window_size)
        mask = pattern.build_pattern()
        
        # Check that pattern is sparse
        sparsity = (~mask).sum().item() / mask.numel()
        assert sparsity > 0.5  # Should be more than 50% sparse
    
    def test_strided_pattern(self):
        seq_len = 256
        pattern = StridedSparsePattern(seq_len, stride=32)
        mask = pattern.build_pattern()
        assert mask.shape == (seq_len, seq_len)
    
    def test_clustered_pattern_assignment(self):
        seq_len = 64
        batch = 2
        heads = 4
        dim_head = 32
        
        pattern = ClusteredSparsePattern(seq_len, num_clusters=8, num_heads=heads, dim_head=dim_head)
        
        # Create dummy Q, K
        q = torch.randn(batch, heads, seq_len, dim_head)
        k = torch.randn(batch, heads, seq_len, dim_head)
        
        q_assign, k_assign = pattern.compute_cluster_assignment(q, k)
        
        assert q_assign.shape == (batch, heads, seq_len, 8)
        assert k_assign.shape == (batch, heads, seq_len, 8)
        
        # Check probabilities sum to 1
        assert torch.allclose(q_assign.sum(dim=-1), torch.ones(batch, heads, seq_len), atol=1e-5)
    
    def test_hierarchical_pattern(self):
        seq_len = 128
        pattern = HierarchicalSparsePattern(seq_len, scales=[4, 16, 64], num_heads=4)
        
        device = torch.device('cpu')
        combined = pattern.combine_patterns(device)
        
        assert combined.shape[0] == 4  # num_heads
        assert combined.shape[1] == seq_len
        assert combined.shape[2] == seq_len


class TestAdaptiveGate:
    """Test adaptive gating mechanism."""
    
    def test_gate_output_shape(self):
        batch = 2
        seq_len = 64
        dim = 256
        num_heads = 8
        
        gate = AdaptiveGate(dim, num_heads)
        x = torch.randn(batch, seq_len, dim)
        
        gate_values, confidence, pattern_logits = gate(x)
        
        assert gate_values.shape == (batch, num_heads, seq_len)
        assert confidence.shape == (batch, num_heads)
        assert pattern_logits.shape == (batch, 4)
        
        # Check ranges
        assert (gate_values >= 0).all() and (gate_values <= 1).all()
        assert (confidence >= 0).all() and (confidence <= 1).all()
    
    def test_gate_gradient_flow(self):
        batch = 2
        seq_len = 32
        dim = 128
        num_heads = 4
        
        gate = AdaptiveGate(dim, num_heads)
        x = torch.randn(batch, seq_len, dim, requires_grad=True)
        
        gate_values, confidence, _ = gate(x)
        loss = gate_values.sum() + confidence.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestASAMLayer:
    """Test ASAM layer."""
    
    def test_forward_pass(self):
        config = ASAMConfig(
            dim=256,
            num_heads=4,
            dim_head=64,
            pattern_type="local",
            use_adaptive_gate=False,
        )
        
        layer = ASAMLayer(config)
        layer.eval()
        
        batch = 2
        seq_len = 64
        x = torch.randn(batch, seq_len, config.dim)
        
        with torch.no_grad():
            output, _ = layer(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_forward_with_adaptive_gate(self):
        config = ASAMConfig(
            dim=256,
            num_heads=4,
            dim_head=64,
            pattern_type="hierarchical",
            use_adaptive_gate=True,
        )
        
        layer = ASAMLayer(config)
        layer.eval()
        
        batch = 2
        seq_len = 64
        x = torch.randn(batch, seq_len, config.dim)
        
        with torch.no_grad():
            output, info = layer(x, return_info=True)
        
        assert output.shape == x.shape
        assert info is not None
        assert 'gate_values' in info
        assert 'confidence' in info
    
    def test_backward_pass(self):
        config = ASAMConfig(
            dim=128,
            num_heads=4,
            dim_head=32,
            pattern_type="local",
            use_adaptive_gate=True,
        )
        
        layer = ASAMLayer(config)
        layer.train()
        
        batch = 2
        seq_len = 32
        x = torch.randn(batch, seq_len, config.dim, requires_grad=True)
        
        output, _ = layer(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
    
    def test_different_patterns(self):
        """Test that all pattern types work."""
        patterns = ["local", "strided", "random", "clustered", "hierarchical"]
        
        for pattern_type in patterns:
            config = ASAMConfig(
                dim=128,
                num_heads=4,
                dim_head=32,
                pattern_type=pattern_type,
                use_adaptive_gate=False,
            )
            
            layer = ASAMLayer(config)
            layer.eval()
            
            x = torch.randn(2, 64, 128)
            
            with torch.no_grad():
                output, _ = layer(x)
            
            assert output.shape == x.shape, f"Failed for pattern: {pattern_type}"
    
    def test_variable_sequence_length(self):
        """Test that layer handles variable sequence lengths."""
        config = ASAMConfig(
            dim=128,
            num_heads=4,
            dim_head=32,
            pattern_type="local",
        )
        
        layer = ASAMLayer(config)
        layer.eval()
        
        # Test different sequence lengths
        for seq_len in [32, 64, 128, 256]:
            x = torch.randn(2, seq_len, 128)
            with torch.no_grad():
                output, _ = layer(x)
            assert output.shape == x.shape


class TestIntegration:
    """Integration tests."""
    
    def test_encoder_stack(self):
        from asam.asam_layer import ASAMEncoder
        
        config = ASAMConfig(dim=128, num_heads=4, dim_head=32)
        encoder = ASAMEncoder(config, num_layers=3)
        encoder.eval()
        
        x = torch.randn(2, 64, 128)
        
        with torch.no_grad():
            output = encoder(x)
        
        assert output.shape == x.shape
    
    def test_cuda_compatibility(self):
        """Test CUDA compatibility if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        config = ASAMConfig(
            dim=128,
            num_heads=4,
            dim_head=32,
            pattern_type="hierarchical",
            use_adaptive_gate=True,
        )
        
        layer = ASAMLayer(config).cuda()
        x = torch.randn(2, 64, 128).cuda()
        
        output, info = layer(x, return_info=True)
        
        assert output.is_cuda
        assert output.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
