"""
Unit tests for ASAM.
"""

import torch
import pytest
import torch.nn.functional as F
from asam import ASAMLayer, ASAMConfig, AdaptiveGate
from asam.sparse_patterns import (
    LocalSparsePattern,
    StridedSparsePattern,
    RandomSparsePattern,
    ClusteredSparsePattern,
    HierarchicalSparsePattern,
)


def _dense_reference_sparse_attention(q, k, v, pattern_mask, mask=None):
    scores = torch.matmul(q, k.transpose(-2, -1))

    if pattern_mask.dim() == 2:
        pattern_mask = pattern_mask.unsqueeze(0).unsqueeze(0)
    elif pattern_mask.dim() == 3:
        pattern_mask = pattern_mask.unsqueeze(0)

    scores = scores.masked_fill(~pattern_mask, float('-inf'))
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))

    attn = torch.softmax(scores, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)
    return torch.matmul(attn, v)


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

        strided_indices = torch.arange(0, seq_len, 32)
        assert mask[:, strided_indices].all()

    def test_random_pattern_is_deterministic(self):
        pattern_a = RandomSparsePattern(seq_len=64, num_random=8, num_heads=4, seed=123)
        pattern_b = RandomSparsePattern(seq_len=64, num_random=8, num_heads=4, seed=123)

        mask_a = pattern_a.build_pattern()
        mask_b = pattern_b.build_pattern()

        assert torch.equal(mask_a, mask_b)

    def test_random_pattern_does_not_mutate_global_rng(self):
        torch.manual_seed(999)
        expected = torch.rand(4)

        torch.manual_seed(999)
        _ = RandomSparsePattern(seq_len=32, num_random=4, num_heads=2, seed=123).build_pattern()
        actual = torch.rand(4)

        assert torch.allclose(actual, expected)
    
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

    def test_clustered_pattern_matches_reference_ops(self):
        seq_len = 16
        batch = 2
        heads = 2
        dim_head = 8
        num_clusters = 4

        pattern = ClusteredSparsePattern(
            seq_len,
            num_clusters=num_clusters,
            num_heads=heads,
            dim_head=dim_head,
        )

        q = torch.randn(batch, heads, seq_len, dim_head)
        k = torch.randn(batch, heads, seq_len, dim_head)

        q_assign, k_assign = pattern.compute_cluster_assignment(q, k)

        q_norm = F.normalize(q, dim=-1)
        k_norm = F.normalize(k, dim=-1)
        centroids_norm = F.normalize(pattern.centroids, dim=-1)
        temp = pattern.temperature.abs().clamp_min(1e-6)
        q_ref = F.softmax(torch.einsum('b h s d, h c d -> b h s c', q_norm, centroids_norm) / temp, dim=-1)
        k_ref = F.softmax(torch.einsum('b h s d, h c d -> b h s c', k_norm, centroids_norm) / temp, dim=-1)

        assert torch.allclose(q_assign, q_ref, atol=1e-6)
        assert torch.allclose(k_assign, k_ref, atol=1e-6)

        attn_scores = torch.randn(batch, heads, seq_len, seq_len)
        masked = pattern.apply_cluster_mask(attn_scores, q_assign, k_assign)
        affinity_ref = torch.einsum('b h q c, b h k c -> b h q k', q_assign, k_assign)
        masked_ref = attn_scores.masked_fill(~(affinity_ref > 0.1), torch.finfo(attn_scores.dtype).min)

        assert torch.equal(masked, masked_ref)
    
    def test_hierarchical_pattern(self):
        seq_len = 128
        pattern = HierarchicalSparsePattern(seq_len, scales=[4, 16, 64], num_heads=4)
        
        device = torch.device('cpu')
        combined = pattern.combine_patterns(device)
        
        assert combined.shape[0] == 4  # num_heads
        assert combined.shape[1] == seq_len
        assert combined.shape[2] == seq_len

        cache_key = (device.type, device.index)
        assert cache_key in pattern._pattern_stack_cache

    def test_hierarchical_pattern_respects_updated_weights(self):
        seq_len = 64
        num_heads = 2
        pattern = HierarchicalSparsePattern(seq_len, scales=[4, 16], num_heads=num_heads)
        device = torch.device('cpu')

        with torch.no_grad():
            pattern.scale_weights.fill_(-20.0)
            pattern.scale_weights[0].fill_(20.0)

        combined = pattern.combine_patterns(device)
        expected = pattern.patterns[0].get_pattern(device).unsqueeze(0).expand(num_heads, -1, -1)

        assert torch.equal(combined, expected)

    def test_pattern_cache_reuses_cpu_tensor(self):
        pattern = LocalSparsePattern(seq_len=128, window_size=32)

        mask_a = pattern.get_pattern(torch.device('cpu'))
        mask_b = pattern.get_pattern(torch.device('cpu'))

        assert mask_a.data_ptr() == mask_b.data_ptr()


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

    def test_sparse_attention_matches_dense_reference_for_local_pattern(self):
        config = ASAMConfig(
            dim=64,
            num_heads=2,
            dim_head=32,
            pattern_type="local",
            window_size=8,
            use_adaptive_gate=False,
        )
        layer = ASAMLayer(config)

        q = torch.randn(1, 2, 10, 32) * layer.scale
        k = torch.randn(1, 2, 10, 32)
        v = torch.randn(1, 2, 10, 32)
        extra_mask = torch.ones(1, 1, 10, 10, dtype=torch.bool)
        extra_mask[..., 0, 3] = False
        extra_mask[..., 5, 4] = False

        pattern = layer._get_pattern(seq_len=10, device=torch.device("cpu"))
        actual = layer._compute_sparse_attention(q, k, v, pattern, mask=extra_mask)
        expected = _dense_reference_sparse_attention(q, k, v, pattern.get_pattern(torch.device("cpu")), mask=extra_mask)

        assert torch.allclose(actual, expected, atol=1e-5)

    def test_sparse_attention_matches_dense_reference_for_hierarchical_pattern(self):
        config = ASAMConfig(
            dim=64,
            num_heads=2,
            dim_head=32,
            pattern_type="hierarchical",
            use_adaptive_gate=False,
        )
        layer = ASAMLayer(config)

        q = torch.randn(1, 2, 12, 32) * layer.scale
        k = torch.randn(1, 2, 12, 32)
        v = torch.randn(1, 2, 12, 32)
        extra_mask = torch.ones(1, 2, 12, 12, dtype=torch.bool)
        extra_mask[:, 0, 1, 8] = False
        extra_mask[:, 1, 9, 2] = False

        pattern = layer._get_pattern(seq_len=12, device=torch.device("cpu"))
        pattern_mask = pattern.combine_patterns(torch.device("cpu"))
        actual = layer._compute_sparse_attention(q, k, v, pattern, mask=extra_mask)
        expected = _dense_reference_sparse_attention(q, k, v, pattern_mask, mask=extra_mask)

        assert torch.allclose(actual, expected, atol=1e-5)


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
