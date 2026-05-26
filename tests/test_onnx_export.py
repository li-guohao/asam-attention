"""Tests for ONNX export utilities."""
import torch
import tempfile
import os
import pytest


def _onnx_available():
    """Check if torch.onnx.export is functional (requires onnxscript on PyTorch 2.7+)."""
    try:
        import torch.onnx
        # Verify the internal exporter can be imported
        from torch.onnx._internal.exporter import _compat
        return True
    except (ImportError, ModuleNotFoundError):
        return False


onnx_skip = pytest.mark.skipif(
    not _onnx_available(),
    reason="torch.onnx.export not available (install onnxscript for PyTorch 2.7+)",
)


@onnx_skip
def test_export_to_onnx_creates_file():
    """export_to_onnx creates a valid .onnx file."""
    from asam import ASAMConfig, ASAMLayer
    from asam.export import export_to_onnx

    config = ASAMConfig(dim=64, num_heads=2, use_adaptive_gate=False)
    model = ASAMLayer(config)
    model.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "test.onnx")

        with torch.no_grad():
            onnx_out = export_to_onnx(model, onnx_path, seq_len=128)

        assert os.path.exists(onnx_path)
        # Verify file is non-empty
        assert os.path.getsize(onnx_path) > 1000


@onnx_skip
def test_export_to_onnx_with_dynamic_batch():
    """export_to_onnx works with dynamic batch size."""
    from asam import ASAMConfig, ASAMLayer
    from asam.export import export_to_onnx

    config = ASAMConfig(dim=64, num_heads=2, use_adaptive_gate=False)
    model = ASAMLayer(config)
    model.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "test_dynamic.onnx")

        with torch.no_grad():
            onnx_out = export_to_onnx(
                model,
                onnx_path,
                seq_len=128,
                dynamic_batch=True,
                dynamic_seq_len=False,
            )

        assert os.path.exists(onnx_path)
        assert os.path.getsize(onnx_path) > 1000


@onnx_skip
def test_export_to_onnx_return_path():
    """export_to_onnx returns the output path."""
    from asam import ASAMConfig, ASAMLayer
    from asam.export import export_to_onnx

    config = ASAMConfig(dim=64, num_heads=2, use_adaptive_gate=False)
    model = ASAMLayer(config)
    model.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "test.onnx")

        with torch.no_grad():
            result = export_to_onnx(model, onnx_path, seq_len=128)

        assert result == onnx_path
