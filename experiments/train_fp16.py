#!/usr/bin/env python3
"""
ASAM Training with Mixed Precision
===================================

Uses torch.cuda.amp for automatic mixed precision training.
Can be 2-3x faster on RTX 3060.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from asam.asam_layer_optimized import OptimizedASAMLayer


def train_with_mixed_precision():
    """Demo training with FP16."""
    device = torch.device('cuda')
    
    # Model
    model = OptimizedASAMLayer(dim=512, num_heads=8, window_size=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Dummy data
    x = torch.randn(4, 1024, 512, device=device)
    target = torch.randn(4, 1024, 512, device=device)
    criterion = nn.MSELoss()
    
    model.train()
    
    # Training step with mixed precision
    optimizer.zero_grad()
    
    with autocast():  # FP16 forward pass
        output, _ = model(x)
        loss = criterion(output, target)
    
    # Scaled backward
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    print(f"Loss: {loss.item():.4f}")
    print("Mixed precision training successful!")


if __name__ == "__main__":
    train_with_mixed_precision()
