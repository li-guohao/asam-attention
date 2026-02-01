"""
Quantization Support for ASAM
==============================

INT8 and FP16 quantization for efficient inference.
"""

import torch
import torch.nn as nn
import torch.quantization
from typing import Dict, Optional


class QuantizedASAMLayer(nn.Module):
    """
    ASAM layer with INT8 quantization support.
    """
    
    def __init__(self, asam_layer, quantization_config='fbgemm'):
        super().__init__()
        self.original_layer = asam_layer
        
        # Prepare for quantization
        self.quantized_layer = torch.quantization.quantize_dynamic(
            asam_layer,
            {nn.Linear},
            dtype=torch.qint8
        )
    
    def forward(self, x):
        return self.quantized_layer(x)


def quantize_asam_model(model, dtype=torch.qint8):
    """
    Quantize ASAM model for efficient inference.
    
    Args:
        model: ASAM model
        dtype: Quantization dtype (torch.qint8 or torch.float16)
        
    Returns:
        Quantized model
    """
    if dtype == torch.qint8:
        # Dynamic quantization (INT8)
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.MultiheadAttention},
            dtype=torch.qint8
        )
    elif dtype == torch.float16:
        # FP16 quantization
        quantized_model = model.half()
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    return quantized_model


class MixedPrecisionASAM(nn.Module):
    """
    ASAM with automatic mixed precision training.
    """
    
    def __init__(self, asam_layer):
        super().__init__()
        self.layer = asam_layer
    
    def forward(self, x):
        # Use autocast for mixed precision
        with torch.cuda.amp.autocast():
            return self.layer(x)


def calibrate_quantization(model, dataloader, num_batches=10):
    """
    Calibrate quantization using sample data.
    
    Args:
        model: Model to calibrate
        dataloader: Data loader for calibration
        num_batches: Number of batches to use
    """
    model.eval()
    
    with torch.no_grad():
        for i, (x, _) in enumerate(dataloader):
            if i >= num_batches:
                break
            _ = model(x)
    
    print(f"Calibration completed using {num_batches} batches")
