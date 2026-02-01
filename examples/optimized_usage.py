#!/usr/bin/env python3
"""
ASAM Optimized Usage Example
============================

Demonstrates Flash Attention and mixed precision training.
Requires GPU with CUDA support.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import time

from asam.efficient_attention import FlashASAMLayer


class SimpleModel(nn.Module):
    """Simple model with Flash ASAM layers."""
    
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


def benchmark_inference(model, x, num_iters=50):
    """Benchmark inference speed."""
    model.eval()
    device = x.device
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_iters):
            torch.cuda.synchronize()
            start = time.time()
            _ = model(x)
            torch.cuda.synchronize()
            times.append((time.time() - start) * 1000)
    
    return sum(times) / len(times)


def train_with_mixed_precision(model, x, y, num_steps=20):
    """Train with automatic mixed precision."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    
    times = []
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        torch.cuda.synchronize()
        start = time.time()
        
        # Mixed precision forward
        with autocast():
            logits = model(x)
            loss = criterion(logits, y)
        
        # Scaled backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        torch.cuda.synchronize()
        times.append((time.time() - start) * 1000)
        
        if (step + 1) % 5 == 0:
            print(f"  Step {step+1}: Loss={loss.item():.4f}, Time={sum(times[-5:])/5:.2f}ms")
    
    return sum(times) / len(times)


def main():
    print("="*70)
    print("ASAM Optimized Usage Example")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("\nERROR: This example requires CUDA GPU.")
        print("Please run on a machine with NVIDIA GPU.")
        return
    
    device = torch.device('cuda')
    print(f"\nDevice: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    
    # Create model
    dim = 256
    num_heads = 4
    model = SimpleModel(dim, num_heads).to(device)
    
    print(f"\nModel config: dim={dim}, heads={num_heads}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test different sequence lengths
    print("\n" + "="*70)
    print("INFERENCE BENCHMARK")
    print("="*70)
    print(f"\n{'Seq Len':<10} {'Time (ms)':<12} {'Throughput':<15}")
    print("-"*70)
    
    for seq_len in [256, 512, 1024]:
        x = torch.randint(0, 1000, (4, seq_len), device=device)
        
        avg_time = benchmark_inference(model, x)
        throughput = x.size(0) / (avg_time / 1000)
        
        print(f"{seq_len:<10} {avg_time:>10.2f}  {throughput:>10.1f} seq/s")
    
    # Training benchmark
    print("\n" + "="*70)
    print("TRAINING WITH MIXED PRECISION")
    print("="*70)
    
    seq_len = 512
    x = torch.randint(0, 1000, (4, seq_len), device=device)
    y = torch.randint(0, 10, (4,), device=device)
    
    print(f"\nTraining with seq_len={seq_len}, batch_size=4")
    avg_time = train_with_mixed_precision(model, x, y, num_steps=20)
    print(f"\nAverage training time: {avg_time:.2f}ms/step")
    
    print("\n" + "="*70)
    print("Key Optimizations Used:")
    print("  1. Flash Attention - Memory-efficient attention computation")
    print("  2. Mixed Precision - FP16 forward/backward with Tensor Cores")
    print("  3. Local Attention - O(n*window) complexity for long sequences")
    print("="*70)


if __name__ == "__main__":
    main()
