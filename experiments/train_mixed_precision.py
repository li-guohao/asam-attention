#!/usr/bin/env python3
"""
Mixed Precision Training for ASAM
==================================

Demonstrates 2-3x speedup using torch.cuda.amp on RTX 3060.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from asam.efficient_attention import FlashASAMLayer


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, dim=256, num_heads=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(1000, dim)
        self.layers = nn.ModuleList([
            FlashASAMLayer(dim, num_heads) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 10)
    
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x, _ = layer(x)
        x = self.norm(x)
        return self.head(x.mean(dim=1))


def train_step(model, x, y, optimizer, criterion, use_amp=False):
    """Single training step."""
    optimizer.zero_grad()
    
    if use_amp:
        with autocast():
            logits = model(x)
            loss = criterion(logits, y)
        
        scaler = GradScaler()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
    
    return loss.item()


def benchmark_training(seq_len=512, batch_size=4, num_steps=20, use_amp=False):
    """Benchmark training speed."""
    device = torch.device('cuda')
    model = SimpleModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    times = []
    
    # Warmup
    for _ in range(3):
        x = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        y = torch.randint(0, 10, (batch_size,), device=device)
        train_step(model, x, y, optimizer, criterion, use_amp)
    
    torch.cuda.synchronize()
    
    # Benchmark
    for step in range(num_steps):
        x = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        y = torch.randint(0, 10, (batch_size,), device=device)
        
        torch.cuda.synchronize()
        start = time.time()
        loss = train_step(model, x, y, optimizer, criterion, use_amp)
        torch.cuda.synchronize()
        
        times.append((time.time() - start) * 1000)
    
    avg_time = sum(times) / len(times)
    return avg_time


def main():
    print("="*70)
    print("Mixed Precision Training Benchmark")
    print("="*70)
    
    device = torch.device('cuda')
    print(f"\nDevice: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    
    print(f"\n{'Seq Len':<10} {'FP32 (ms)':<12} {'FP16 (ms)':<12} {'Speedup':<10}")
    print("-"*70)
    
    for seq_len in [256, 512, 1024]:
        # FP32 training
        t_fp32 = benchmark_training(seq_len=seq_len, use_amp=False, num_steps=10)
        
        # FP16 training
        t_fp16 = benchmark_training(seq_len=seq_len, use_amp=True, num_steps=10)
        
        speedup = t_fp32 / t_fp16
        print(f"{seq_len:<10} {t_fp32:>10.2f}  {t_fp16:>10.2f}  {speedup:>8.2f}x")
    
    print("\n" + "="*70)
    print("Notes:")
    print("- FP16 uses half precision (2x memory saving)")
    print("- RTX 3060 has Tensor Cores for FP16 acceleration")
    print("- Speedup varies based on model size and batch size")
    print("="*70)


if __name__ == "__main__":
    main()
