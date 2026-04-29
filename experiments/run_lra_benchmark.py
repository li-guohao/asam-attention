#!/usr/bin/env python3
"""Run complete LRA benchmark across all 5 tasks + baseline models.

Usage:
    python experiments/run_lra_benchmark.py --task all
    python experiments/run_lra_benchmark.py --task listops --model asam
    python experiments/run_lra_benchmark.py --task all --output results.json

Output:
    experiments/lra_results.json — complete benchmark data
"""

import argparse
import json
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from asam.datasets.lra_dataset import create_lra_dataloaders, LRA_TASKS
from asam.modeling_asam import ASAMHFConfig, ASAMHFForSequenceClassification


class StandardTransformer(nn.Module):
    """Baseline: standard full-attention transformer."""

    def __init__(self, dim: int, num_heads: int, num_layers: int, vocab_size: int,
                 num_classes: int, max_len: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(max_len, dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, batch_first=True, dim_feedforward=dim * 4
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x, mask=None):
        b, n = x.shape
        pos = torch.arange(n, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos(pos)
        h = self.encoder(h)
        return self.classifier(h.mean(dim=1))


class LocalTransformer(nn.Module):
    """Baseline: local window attention (Longformer-style)."""

    def __init__(self, dim: int, num_heads: int, num_layers: int, vocab_size: int,
                 num_classes: int, max_len: int, window_size: int = 256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(max_len, dim)

        # Build local mask (triangular band)
        self.register_buffer(
            "local_mask",
            torch.ones(max_len, max_len, dtype=torch.bool).tril(
                diagonal=window_size // 2
            ).triu(diagonal=-window_size // 2),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, batch_first=True, dim_feedforward=dim * 4
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x, mask=None):
        b, n = x.shape
        pos = torch.arange(n, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos(pos)
        local_mask = self.local_mask[:n, :n].to(x.device)
        # Invert for PyTorch's attn_mask convention (True = masked out)
        h = self.encoder(h, mask=~local_mask)
        return self.classifier(h.mean(dim=1))


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        if isinstance(batch[0], tuple):
            # Retrieval: ((x1, x2), y)
            (x1, x2), y = batch
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            logits = model(x1)  # simplified — uses only doc1
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)

        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    start_time = time.time()

    for batch in loader:
        if isinstance(batch[0], tuple):
            (x1, x2), y = batch
            x1, y = x1.to(device), y.to(device)
            logits = model(x1)
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)

        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    elapsed = (time.time() - start_time) * 1000 / len(loader)
    return correct / total, elapsed


def bench_task(task_name, model_name, device="cuda"):
    task_info = LRA_TASKS[task_name]
    batch_size = 8
    epochs = 3  # Quick benchmark; increase to 5-10 for final paper

    loaders = create_lra_dataloaders(task_name, batch_size=batch_size, train_samples=2000)

    # Create model
    common_kwargs = dict(
        dim=64, num_heads=2, num_layers=2,
        vocab_size=task_info["vocab_size"] + 1,
        num_classes=task_info["num_classes"],
        max_len=task_info["seq_len"],
    )

    if model_name == "asam":
        config = ASAMHFConfig(
            dim=64, num_heads=2, num_layers=2,
            vocab_size=task_info["vocab_size"] + 1,
            num_labels=task_info["num_classes"],
            max_position_embeddings=task_info["seq_len"],
            pattern_type="hierarchical",
        )
        model = ASAMHFForSequenceClassification(config)
    elif model_name == "transformer":
        model = StandardTransformer(**common_kwargs)
    elif model_name == "local":
        model = LocalTransformer(**common_kwargs, window_size=256)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Train
    acc = 0.0
    speed = 0.0
    for epoch in range(epochs):
        loss = train_epoch(model, loaders["train"], optimizer, device)
        if epoch == epochs - 1:
            acc, speed = evaluate(model, loaders["test"], device)

    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2) if device == "cuda" else 0

    return {
        "task": task_name,
        "model": model_name,
        "accuracy": round(acc * 100, 1),
        "speed_ms": round(speed, 2),
        "memory_mb": round(peak_memory, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="LRA Benchmark Runner")
    parser.add_argument("--task", default="all",
                        help="Task name or 'all' (default: all)")
    parser.add_argument("--model", default="all",
                        help="Model name or 'all' (default: all)")
    parser.add_argument("--output", default="experiments/lra_results.json",
                        help="Output JSON path (default: experiments/lra_results.json)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on (default: auto-detect)")
    parser.add_argument("--train-samples", type=int, default=2000,
                        help="Number of training samples (default: 2000)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs (default: 3)")
    args = parser.parse_args()

    tasks = list(LRA_TASKS) if args.task == "all" else [args.task]
    models = ["asam", "transformer", "local"] if args.model == "all" else [args.model]

    results = []
    for task_name in tasks:
        for model_name in models:
            print(f"\n{'='*50}")
            print(f"Benchmark: {task_name} / {model_name}")
            print(f"{'='*50}")
            try:
                result = bench_task(task_name, model_name, args.device)
                results.append(result)
                print(f"  Accuracy: {result['accuracy']}%")
                print(f"  Speed: {result['speed_ms']}ms/batch")
                print(f"  Memory: {result['memory_mb']}MB")
            except Exception as e:
                print(f"  FAILED: {e}")
                results.append({"task": task_name, "model": model_name, "error": str(e)})

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")
    print(f"Total results: {len(results)}")


if __name__ == "__main__":
    main()
