"""
Tests for optimized ASAM execution paths.
"""

import math

import torch

from asam.asam_layer_optimized import OptimizedASAMLayer
from asam.efficient_attention import EfficientASAMLayer
from asam.sparse_patterns import StridedSparsePattern


def _dense_reference_attention(q, k, v, mask):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    scores = scores.masked_fill(~mask, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)
    return torch.matmul(attn, v)


def test_optimized_gate_shapes_follow_num_heads():
    layer = OptimizedASAMLayer(dim=256, num_heads=4, window_size=64, use_adaptive_gate=True)
    x = torch.randn(2, 96, 256)

    output, info = layer(x, return_info=True)

    assert output.shape == x.shape
    assert info["gate_values"].shape == (2, 4, 96)
    assert info["confidence"].shape == (2, 4)
    assert info["pattern_logits"].shape == (2, 4)


def test_optimized_gate_is_skipped_without_info(monkeypatch):
    layer = OptimizedASAMLayer(dim=128, num_heads=4, window_size=32, use_adaptive_gate=True)
    x = torch.randn(2, 48, 128)

    def fail_if_called(_):
        raise AssertionError("adaptive gate should not run when return_info is False")

    monkeypatch.setattr(layer.adaptive_gate, "forward", fail_if_called)

    output, info = layer(x, return_info=False)

    assert output.shape == x.shape
    assert info is None


def test_optimized_local_attention_handles_boundary_tokens():
    layer = OptimizedASAMLayer(dim=64, num_heads=2, window_size=16, use_adaptive_gate=False)
    x = torch.randn(2, 8, 64)

    output, _ = layer(x)

    assert output.shape == x.shape
    assert not torch.isnan(output).any()


def test_efficient_local_mask_cache_reuses_tensor():
    layer = EfficientASAMLayer(dim=64, num_heads=2, window_size=16)

    mask_a = layer._create_local_mask(128, torch.device("cpu"))
    mask_b = layer._create_local_mask(128, torch.device("cpu"))

    assert mask_a.data_ptr() == mask_b.data_ptr()


def test_optimized_layer_supports_attention_mask():
    layer = OptimizedASAMLayer(dim=64, num_heads=2, window_size=16, use_adaptive_gate=False)
    x = torch.randn(2, 12, 64)
    mask = torch.eye(12, dtype=torch.bool).unsqueeze(0).unsqueeze(0)

    output, _ = layer(x, mask=mask)

    assert output.shape == x.shape
    assert not torch.isnan(output).any()


def test_optimized_local_attention_matches_dense_reference_with_mask():
    layer = OptimizedASAMLayer(dim=32, num_heads=2, window_size=4, dropout=0.0, use_adaptive_gate=False)
    q = torch.randn(1, 2, 7, 16)
    k = torch.randn(1, 2, 7, 16)
    v = torch.randn(1, 2, 7, 16)

    extra_mask = torch.ones(1, 1, 7, 7, dtype=torch.bool)
    extra_mask[..., 0, 2] = False
    extra_mask[..., 3, 4] = False
    extra_mask[..., 6, 5] = False

    local_mask = layer._get_dense_local_mask(7, torch.device("cpu")).unsqueeze(0).unsqueeze(0)
    reference_mask = local_mask & extra_mask

    actual = layer._compute_local_attention(q, k, v, window_size=4, mask=extra_mask)
    expected = _dense_reference_attention(q, k, v, reference_mask.expand(-1, q.size(1), -1, -1))

    assert torch.allclose(actual, expected, atol=1e-5)


def test_optimized_strided_attention_matches_dense_reference():
    layer = OptimizedASAMLayer(
        dim=32,
        num_heads=2,
        window_size=8,
        stride=4,
        dropout=0.0,
        use_adaptive_gate=False,
        pattern_type='strided',
    )
    q = torch.randn(1, 2, 9, 16)
    k = torch.randn(1, 2, 9, 16)
    v = torch.randn(1, 2, 9, 16)

    local_window = layer._get_strided_local_window()
    pattern = StridedSparsePattern(seq_len=9, stride=layer.stride, local_window=local_window)
    reference_mask = pattern.build_pattern().unsqueeze(0).unsqueeze(0).expand(-1, q.size(1), -1, -1)

    actual = layer._compute_strided_attention(q, k, v, stride=layer.stride, local_window=local_window)
    expected = _dense_reference_attention(q, k, v, reference_mask)

    assert torch.allclose(actual, expected, atol=1e-5)


def test_optimized_local_attention_with_mask_keeps_sparse_path(monkeypatch):
    layer = OptimizedASAMLayer(dim=64, num_heads=2, window_size=16, use_adaptive_gate=False)
    x = torch.randn(2, 12, 64)
    mask = torch.eye(12, dtype=torch.bool).unsqueeze(0).unsqueeze(0)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("dense fallback should not be used for local masked attention")

    monkeypatch.setattr(layer, "_fallback_attention_with_mask", fail_if_called)

    output, _ = layer(x, mask=mask)

    assert output.shape == x.shape
