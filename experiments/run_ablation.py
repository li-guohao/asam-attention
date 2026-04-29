#!/usr/bin/env python3
"""Ablation study: measure contribution of each ASAM component.

Tests 4 configurations:
1. Full ASAM (hierarchical + adaptive gate)
2. Without AdaptiveGate (hierarchical only)
3. Without Hierarchical (local window only)
4. Standard Attention baseline

Usage:
    python experiments/run_ablation.py --output experiments/ablation_results.json
"""

import argparse
import json
import os
import time
import torch
import torch.nn as nn

from asam.datasets.lra_dataset import create_lra_dataloaders, LRA_TASKS
from asam.modeling_asam import ASAMHFConfig, ASAMHFForSequenceClassification


def train_epoch(model, loader, optimizer, device):
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    count = 0
    for batch in loader:
        if isinstance(batch[0], tuple):
            (x1, x2), y = batch
            x1, y = x1.to(device), y.to(device)
            logits = model(x1)
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)

        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        count += 1
    return total_loss / count


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model on a dataloader and return accuracy + speed (ms/batch)."""
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

    elapsed_ms = (time.time() - start_time) * 1000 / len(loader)
    return correct / total, elapsed_ms


def run_ablation(task_name, config_name, config_kwargs, device, train_samples=1000, epochs=3):
    """Run a single ablation configuration and return results."""
    task_info = LRA_TASKS[task_name]
    batch_size = 8

    loaders = create_lra_dataloaders(
        task_name, batch_size=batch_size, train_samples=train_samples
    )

    if config_kwargs is not None:
        # ASAM variant
        cfg = ASAMHFConfig(
            dim=64, num_heads=2, num_layers=2,
            vocab_size=task_info["vocab_size"] + 1,
            num_labels=task_info["num_classes"],
            max_position_embeddings=task_info["seq_len"],
            **config_kwargs,
        )
        model = ASAMHFForSequenceClassification(cfg)
    else:
        # Standard Transformer baseline
        from experiments.run_lra_benchmark import StandardTransformer
        model = StandardTransformer(
            dim=64, num_heads=2, num_layers=2,
            vocab_size=task_info["vocab_size"] + 1,
            num_classes=task_info["num_classes"],
            max_len=task_info["seq_len"],
        )

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Train for specified epochs
    acc, speed = 0.0, 0.0
    for epoch in range(epochs):
        loss = train_epoch(model, loaders["train"], optimizer, device)
        if epoch == epochs - 1:
            acc, speed = evaluate(model, loaders["test"], device)

    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2) if device == "cuda" else 0

    return {
        "task": task_name,
        "config": config_name,
        "accuracy": round(acc * 100, 1),
        "speed_ms": round(speed, 2),
        "memory_mb": round(peak_memory, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="ASAM Ablation Study")
    parser.add_argument("--tasks", nargs="+", default=["listops", "text"],
                        help="Tasks to run (default: listops text)")
    parser.add_argument("--output", default="experiments/ablation_results.json",
                        help="Output JSON path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on")
    parser.add_argument("--train-samples", type=int, default=1000,
                        help="Training samples per task (default: 1000)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Training epochs (default: 3)")
    args = parser.parse_args()

    # Four ablation configurations
    configs = [
        ("full", {"pattern_type": "hierarchical", "use_adaptive_gate": True}),
        ("no_gate", {"pattern_type": "hierarchical", "use_adaptive_gate": False}),
        ("no_hierarchical", {"pattern_type": "local", "use_adaptive_gate": True}),
        ("standard", None),  # Transformer baseline
    ]

    results = []
    for task_name in args.tasks:
        for config_name, config_kwargs in configs:
            print(f"\nAblation: {task_name} / {config_name}")
            try:
                result = run_ablation(
                    task_name, config_name, config_kwargs, args.device,
                    train_samples=args.train_samples, epochs=args.epochs,
                )
                results.append(result)
                print(f"  Accuracy: {result['accuracy']}%")
                print(f"  Speed: {result['speed_ms']}ms/batch")
                print(f"  Memory: {result['memory_mb']}MB")
            except Exception as e:
                print(f"  FAILED: {e}")
                results.append({"task": task_name, "config": config_name, "error": str(e)})

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")
    print(f"Total ablation results: {len(results)}")


if __name__ == "__main__":
    main()
