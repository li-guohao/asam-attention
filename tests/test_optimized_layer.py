"""
Tests for optimized ASAM execution paths.
"""

import torch

from asam.asam_layer_optimized import OptimizedASAMLayer
from asam.efficient_attention import EfficientASAMLayer


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
