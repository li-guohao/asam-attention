"""ONNX export utilities for ASAM models."""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    batch_size: int = 1,
    seq_len: int = 512,
    dim: Optional[int] = None,
    dynamic_batch: bool = True,
    dynamic_seq_len: bool = False,
    opset_version: int = 17,
) -> str:
    """Export an ASAM model to ONNX format.

    Args:
        model: ASAM model in eval mode.
        output_path: Path for the output .onnx file.
        batch_size: Sample batch size for tracing.
        seq_len: Fixed sequence length for tracing.
        dim: Model dimension (inferred from model if None).
        dynamic_batch: Allow variable batch size.
        dynamic_seq_len: Allow variable sequence length (may not work
            with all pattern types -- local window only).
        opset_version: ONNX opset version.

    Returns:
        Path to the exported ONNX file.
    """
    if dim is None:
        # Try to infer from model
        for name, param in model.named_parameters():
            if "weight" in name:
                dim = param.shape[-1]
                break
        if dim is None:
            dim = 512

    sample_input = torch.randn(batch_size, seq_len, dim)
    device = next(model.parameters()).device
    sample_input = sample_input.to(device)

    dynamic_axes = {}
    if dynamic_batch:
        dynamic_axes["x"] = {0: "batch_size"}
        dynamic_axes["output"] = {0: "batch_size"}
    if dynamic_seq_len:
        dynamic_axes["x"] = {**dynamic_axes.get("x", {}), 1: "seq_len"}
        dynamic_axes["output"] = {**dynamic_axes.get("output", {}), 1: "seq_len"}

    input_names = ["x"]
    output_names = ["output"]

    with torch.no_grad():
        torch.onnx.export(
            model,
            sample_input,
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes or None,
            opset_version=opset_version,
            do_constant_folding=True,
        )

    return output_path


def verify_onnx_export(
    onnx_path: str,
    pytorch_model: nn.Module,
    sample_input: torch.Tensor,
    atol: float = 1e-5,
) -> bool:
    """Verify ONNX export matches PyTorch model output.

    Args:
        onnx_path: Path to the exported .onnx file.
        pytorch_model: The original PyTorch model.
        sample_input: Input tensor for comparison.
        atol: Absolute tolerance for output comparison.

    Returns:
        True if outputs match within tolerance.

    Raises:
        ImportError: If onnxruntime is not installed.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError(
            "onnxruntime is required for verification. "
            "Install with: pip install onnxruntime"
        )

    # PyTorch output
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(sample_input)
        if isinstance(pytorch_output, tuple):
            pytorch_output = pytorch_output[0]

    # ONNX output
    session = ort.InferenceSession(onnx_path)
    onnx_output = session.run(
        None, {"x": sample_input.cpu().numpy()}
    )[0]

    pytorch_np = pytorch_output.cpu().numpy()
    match = (
        abs(pytorch_np - onnx_output).max() < atol
    )

    if not match:
        max_diff = abs(pytorch_np - onnx_output).max()
        print(f"Max difference: {max_diff:.2e} (tolerance: {atol})")

    return match
