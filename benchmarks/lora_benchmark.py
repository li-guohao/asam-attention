"""
Long Range Arena (LRA) Benchmark for ASAM
==========================================

Standardized benchmark for evaluating long sequence models.
Includes: ListOps, Text Classification, Retrieval, Image Classification, Pathfinder

Reference: "Long Range Arena: A Benchmark for Efficient Transformers" (Tay et al., 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from asam import ASAMLayer, ASAMConfig


class LRABenchmarkTask:
    """Base class for LRA benchmark tasks."""
    
    def __init__(self, task_name: str, seq_len: int, vocab_size: int, num_classes: int):
        self.task_name = task_name
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_classes = num_classes
    
    def generate_dummy_data(self, batch_size: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate dummy data for testing."""
        x = torch.randint(0, self.vocab_size, (batch_size, self.seq_len))
        y = torch.randint(0, self.num_classes, (batch_size,))
        return x, y


class ListOpsTask(LRABenchmarkTask):
    """ListOps task: Hierarchical structure understanding."""
    
    def __init__(self):
        super().__init__("ListOps", seq_len=2048, vocab_size=20, num_classes=10)
    
    def generate_dummy_data(self, batch_size: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
        # Simulate nested list operations
        x = torch.randint(0, self.vocab_size, (batch_size, self.seq_len))
        y = torch.randint(0, self.num_classes, (batch_size,))
        return x, y


class TextClassificationTask(LRABenchmarkTask):
    """Text classification on long documents."""
    
    def __init__(self):
        super().__init__("Text", seq_len=4096, vocab_size=256, num_classes=2)


class RetrievalTask(LRABenchmarkTask):
    """Document retrieval with dual encoder."""
    
    def __init__(self):
        super().__init__("Retrieval", seq_len=4096, vocab_size=256, num_classes=2)


class ImageTask(LRABenchmarkTask):
    """Sequential image classification (flattened CIFAR-10)."""
    
    def __init__(self):
        super().__init__("Image", seq_len=1024, vocab_size=256, num_classes=10)
    
    def generate_dummy_data(self, batch_size: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
        # Flattened 32x32 images
        x = torch.randn(batch_size, self.seq_len) * 255
        x = x.long().clamp(0, self.vocab_size - 1)
        y = torch.randint(0, self.num_classes, (batch_size,))
        return x, y


class PathfinderTask(LRABenchmarkTask):
    """Pathfinder: Long-range spatial dependency."""
    
    def __init__(self):
        super().__init__("Pathfinder", seq_len=1024, vocab_size=256, num_classes=2)


class LRAModel(nn.Module):
    """Model for LRA benchmark."""
    
    def __init__(
        self,
        task: LRABenchmarkTask,
        dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        pattern_type: str = "hierarchical",
        use_adaptive_gate: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.task = task
        
        # Embedding
        self.embedding = nn.Embedding(task.vocab_size, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, task.seq_len, dim) * 0.02)
        
        # Encoder layers
        config = ASAMConfig(
            dim=dim,
            num_heads=num_heads,
            dim_head=dim // num_heads,
            dropout=dropout,
            pattern_type=pattern_type,
            use_adaptive_gate=use_adaptive_gate,
        )
        
        self.layers = nn.ModuleList([
            ASAMLayer(config) for _ in range(num_layers)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, task.num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # Embedding
        x = self.embedding(x)
        x = x + self.pos_embedding[:, :x.size(1), :]
        
        # Encoder
        info_list = []
        for layer in self.layers:
            x, info = layer(x, return_info=True)
            if info:
                info_list.append(info)
        
        # Classification
        x = x.mean(dim=1)  # Global average pooling
        x = self.norm(x)
        logits = self.classifier(x)
        
        # Aggregate info
        aggregated_info = self._aggregate_info(info_list)
        
        return logits, aggregated_info
    
    def _aggregate_info(self, info_list: List[Dict]) -> Dict:
        """Aggregate info from all layers."""
        if not info_list:
            return {}
        
        avg_sparse_ratio = sum(info['sparse_ratio'] for info in info_list) / len(info_list)
        avg_confidence = sum(info['confidence'].mean().item() for info in info_list) / len(info_list)
        
        return {
            'avg_sparse_ratio': avg_sparse_ratio,
            'avg_confidence': avg_confidence,
        }


class BaselineTransformer(nn.Module):
    """Standard Transformer for comparison."""
    
    def __init__(self, task: LRABenchmarkTask, dim: int = 128, num_layers: int = 4, num_heads: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(task.vocab_size, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, task.seq_len, dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, task.num_classes)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        x = self.embedding(x)
        x = x + self.pos_embedding[:, :x.size(1), :]
        
        # Causal mask for autoregressive tasks
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)
        x = self.encoder(x, mask=mask, is_causal=True)
        
        x = x.mean(dim=1)
        x = self.norm(x)
        logits = self.classifier(x)
        
        return logits, {}


def run_benchmark(
    task: LRABenchmarkTask,
    model_type: str = "asam",
    batch_size: int = 8,
    num_steps: int = 100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict:
    """Run benchmark for a specific task and model."""
    
    print(f"\n{'='*60}")
    print(f"Task: {task.task_name} | Model: {model_type.upper()}")
    print(f"Sequence Length: {task.seq_len} | Batch Size: {batch_size}")
    print(f"{'='*60}")
    
    # Create model
    if model_type == "asam":
        model = LRAModel(
            task,
            dim=128,
            num_layers=4,
            num_heads=4,
            pattern_type="hierarchical",
            use_adaptive_gate=True
        )
    else:
        model = BaselineTransformer(task, dim=128, num_layers=4, num_heads=4)
    
    model = model.to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Warmup
    print("Warming up...")
    for _ in range(10):
        x, y = task.generate_dummy_data(batch_size)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    # Benchmark
    print(f"Running {num_steps} steps...")
    times = []
    losses = []
    accuracies = []
    
    for step in range(num_steps):
        x, y = task.generate_dummy_data(batch_size)
        x, y = x.to(device), y.to(device)
        
        start = time.time()
        
        optimizer.zero_grad()
        logits, info = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed = time.time() - start
        times.append(elapsed * 1000)  # ms
        losses.append(loss.item())
        
        pred = logits.argmax(dim=-1)
        acc = (pred == y).float().mean().item()
        accuracies.append(acc)
    
    # Statistics
    avg_time = np.mean(times)
    std_time = np.std(times)
    avg_loss = np.mean(losses)
    avg_acc = np.mean(accuracies)
    
    peak_memory = 0
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    
    # Throughput
    throughput = (batch_size * num_steps) / (sum(times) / 1000)  # samples/sec
    
    results = {
        'task': task.task_name,
        'model': model_type,
        'seq_len': task.seq_len,
        'avg_time_ms': float(avg_time),
        'std_time_ms': float(std_time),
        'avg_loss': float(avg_loss),
        'avg_accuracy': float(avg_acc),
        'peak_memory_mb': float(peak_memory),
        'throughput': float(throughput),
        'asam_info': info if model_type == "asam" and info else {}
    }
    
    print(f"\nResults:")
    print(f"  Average time: {avg_time:.2f} ± {std_time:.2f} ms/step")
    print(f"  Average loss: {avg_loss:.4f}")
    print(f"  Average accuracy: {avg_acc:.4f}")
    print(f"  Peak memory: {peak_memory:.2f} MB")
    print(f"  Throughput: {throughput:.2f} samples/sec")
    
    if model_type == "asam" and info:
        print(f"  ASAM sparse ratio: {info.get('avg_sparse_ratio', 0):.2%}")
        print(f"  ASAM confidence: {info.get('avg_confidence', 0):.4f}")
    
    return results


def run_full_benchmark():
    """Run complete LRA benchmark."""
    
    tasks = [
        ListOpsTask(),
        TextClassificationTask(),
        RetrievalTask(),
        ImageTask(),
        PathfinderTask(),
    ]
    
    all_results = []
    
    for task in tasks:
        print(f"\n\n{'#'*60}")
        print(f"# TASK: {task.task_name}")
        print(f"{'#'*60}")
        
        # Run ASAM
        try:
            asam_results = run_benchmark(task, model_type="asam", num_steps=50)
            all_results.append(asam_results)
        except Exception as e:
            print(f"ASAM failed: {e}")
        
        # Run Baseline (skip for very long sequences to avoid OOM)
        if task.seq_len <= 4096:
            try:
                baseline_results = run_benchmark(task, model_type="baseline", num_steps=50)
                all_results.append(baseline_results)
            except Exception as e:
                print(f"Baseline failed: {e}")
        else:
            print(f"\nSkipping baseline for {task.task_name} (seq_len={task.seq_len} > 4096)")
    
    # Save results
    with open('lra_benchmark_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print_summary(all_results)
    
    return all_results


def print_summary(results: List[Dict]):
    """Print benchmark summary table."""
    
    print(f"\n\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n{'Task':<15} {'Model':<10} {'SeqLen':<8} {'Time(ms)':<12} {'Memory(MB)':<12} {'Accuracy':<10}")
    print("-"*80)
    
    for r in results:
        print(f"{r['task']:<15} {r['model']:<10} {r['seq_len']:<8} "
              f"{r['avg_time_ms']:>10.2f}  {r['peak_memory_mb']:>10.2f}  {r['avg_accuracy']:>8.4f}")
    
    # Calculate speedups
    print(f"\n\nSpeedup Analysis (ASAM vs Baseline):")
    print("-"*80)
    
    task_comparison = {}
    for r in results:
        task = r['task']
        if task not in task_comparison:
            task_comparison[task] = {}
        task_comparison[task][r['model']] = r
    
    for task, models in task_comparison.items():
        if 'asam' in models and 'baseline' in models:
            asam = models['asam']
            baseline = models['baseline']
            speedup = baseline['avg_time_ms'] / asam['avg_time_ms']
            mem_reduction = baseline['peak_memory_mb'] / max(asam['peak_memory_mb'], 1)
            acc_diff = asam['avg_accuracy'] - baseline['avg_accuracy']
            
            print(f"{task:<15} Speedup: {speedup:>6.2f}x  Memory: {mem_reduction:>6.2f}x  "
                  f"Acc Δ: {acc_diff:>+.4f}")


if __name__ == "__main__":
    results = run_full_benchmark()
