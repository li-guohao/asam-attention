#!/usr/bin/env python3
"""Compare Continual ASAM against standard continual learning baselines.

Runs fine_tune, EWC, SI, MAS, Continual ASAM (task routing), and Continual
ASAM (prototype routing) on the same task-incremental multi-head Split AG
News setup (char/byte-level tokenization, per-task classifier heads, oracle
task ids) so that every row shares the data pipeline, backbone scale, and
metric definitions.

Usage:
    python experiments/run_baseline_comparison.py --num-seeds 2
    python experiments/run_baseline_comparison.py --num-seeds 3 --epochs-per-task 1
"""

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from asam.continual_baselines import EWC, MAS, SI
from experiments.run_continual_text_benchmark import (
    RealBenchmarkArgs,
    get_continual_dataloaders,
    run_benchmark,
)
from experiments.train_continual_asam import compute_continual_metrics, set_seed


@dataclass
class BaselineComparisonArgs:
    dataset_name: str = "split_ag_news"
    classes_per_task: int = 2
    label_mode: str = "local"
    max_length: int = 128
    batch_size: int = 8
    max_train_samples: int = 64
    max_val_samples: int = 32
    vocab_size: int = 128
    num_workers: int = 0
    dim: int = 64
    num_heads: int = 4
    num_layers: int = 1
    learning_rate: float = 3e-4
    epochs_per_task: int = 1
    num_seeds: int = 2
    replay_batch_size: int = 4
    device: str = "cpu"
    output_json: str = "experiments/paper_suite/r2_baseline_comparison.json"


class TaskHeadTransformer(nn.Module):
    """Vanilla transformer backbone with per-task classifier heads.

    forward(x, task_ids=None) selects the per-task head; when task_ids is
    None, the head stored in ``self.task_id`` is used (per-task data loaders
    are task-homogeneous, which keeps EWC/SI/MAS wrappers working with a
    single-tensor forward call).
    """

    def __init__(
        self,
        vocab_size: int = 128,
        dim: int = 64,
        num_layers: int = 1,
        num_heads: int = 4,
        num_tasks: int = 2,
        classes_per_task: int = 2,
        max_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_tasks = num_tasks
        self.classes_per_task = classes_per_task
        self.task_id = 0
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(max_len, dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            batch_first=True,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(dim)
        self.heads = nn.ModuleList(
            [nn.Linear(dim, classes_per_task) for _ in range(num_tasks)]
        )

    def forward(
        self,
        x: torch.Tensor,
        task_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seq_len = x.shape
        device = x.device
        hidden = self.embed(x) + self.pos(torch.arange(seq_len, device=device))
        hidden = self.encoder(hidden)
        pooled = self.norm(hidden.mean(dim=1))
        per_head_logits = torch.stack([head(pooled) for head in self.heads], dim=1)
        if task_ids is None:
            task_ids = torch.full((batch,), self.task_id, dtype=torch.long, device=device)
        task_ids = task_ids.clamp(min=0, max=self.num_tasks - 1)
        index = task_ids.view(batch, 1, 1).expand(batch, 1, self.classes_per_task)
        selected = per_head_logits.gather(1, index)
        return selected.squeeze(1)


def run_vanilla_baseline(
    method_name: str,
    args: BaselineComparisonArgs,
    seed: int,
    train_loaders,
    val_loaders,
) -> Dict[str, float]:
    """Train fine_tune/EWC/SI/MAS on the shared task-incremental protocol."""
    device = torch.device(args.device)
    set_seed(seed)
    model = TaskHeadTransformer(
        vocab_size=args.vocab_size,
        dim=args.dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_tasks=len(train_loaders),
        classes_per_task=args.classes_per_task,
        max_len=args.max_length,
    ).to(device)

    cl_info = None
    if method_name == "ewc":
        cl_info = EWC(model, importance=1000.0)
    elif method_name == "si":
        cl_info = SI(model, importance=1.0)
    elif method_name == "mas":
        cl_info = MAS(model, importance=1.0)

    accuracy_matrix: List[List[float]] = []
    num_tasks = len(train_loaders)
    for task_id in range(num_tasks):
        model.task_id = task_id
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        if isinstance(cl_info, SI):
            cl_info.reset_trajectory()

        model.train()
        for _ in range(args.epochs_per_task):
            for inputs, labels, _task_ids in train_loaders[task_id]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                logits = model(inputs)
                loss = F.cross_entropy(logits, labels)
                if cl_info is not None:
                    loss = loss + cl_info.penalty()
                loss.backward()
                optimizer.step()
                if isinstance(cl_info, SI):
                    cl_info.update_trajectory()

        if cl_info is not None:
            if method_name == "ewc":
                cl_info.update_fisher(train_loaders[task_id], device=args.device)
            elif method_name == "mas":
                cl_info.update_importance(train_loaders[task_id], device=args.device)
            cl_info.consolidate()

        model.eval()
        row: List[float] = []
        with torch.no_grad():
            for seen_task in range(task_id + 1):
                model.task_id = seen_task
                correct = 0
                total = 0
                for inputs, labels, _task_ids in val_loaders[seen_task]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    logits = model(inputs)
                    correct += (logits.argmax(dim=-1) == labels).sum().item()
                    total += labels.size(0)
                row.append(correct / max(1, total))
        accuracy_matrix.append(row)

    return compute_continual_metrics(accuracy_matrix, num_tasks)


def run_asam_row(
    routing_mode: str,
    args: BaselineComparisonArgs,
    seed: int,
    output_json: Optional[str],
) -> Dict[str, float]:
    """Run the Continual ASAM benchmark for task or prototype routing."""
    benchmark_args = RealBenchmarkArgs(
        protocol="task_incremental_multihead",
        dataset_name=args.dataset_name,
        classes_per_task=args.classes_per_task,
        label_mode=args.label_mode,
        head_mode="multi",
        train_task_id_mode="oracle",
        eval_task_id_mode="oracle",
        max_length=args.max_length,
        batch_size=args.batch_size,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        num_workers=args.num_workers,
        vocab_size=args.vocab_size,
        dim=args.dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        top_k_patterns=2,
        routing_mode=routing_mode,
        prototype_routing_strategy="sinkhorn_topk",
        num_prototypes=0,
        prototype_slots_per_task=2,
        prototype_top_k=2,
        learning_rate=args.learning_rate,
        epochs_per_task=args.epochs_per_task,
        replay_batch_size=args.replay_batch_size,
        adaptive_hyperparameters=False,
        adaptation_strategy="correlation",
        device=args.device,
        seed=seed,
        output_json=output_json,
    )
    result = run_benchmark(benchmark_args)
    return {
        "avg_accuracy": float(result["avg_accuracy"]),
        "avg_forgetting": float(result["avg_forgetting"]),
        "backward_transfer": float(result["backward_transfer"]),
    }


def build_markdown_table(methods: Dict[str, Dict[str, float]]) -> str:
    lines = [
        "| Method | Accuracy (mean?std) | Forgetting (mean?std) | BWT (mean?std) | Runs |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in ["fine_tune", "ewc", "si", "mas", "task_routing", "prototype"]:
        row = methods[name]
        lines.append(
            "| {name} | {acc:.4f}?{acc_s:.4f} | {forget:.4f}?{forget_s:.4f} | {bwt:.4f}?{bwt_s:.4f} | {runs} |".format(
                name=name,
                acc=row["accuracy_mean"],
                acc_s=row["accuracy_std"],
                forget=row["forgetting_mean"],
                forget_s=row["forgetting_std"],
                bwt=row["backward_transfer_mean"],
                bwt_s=row["backward_transfer_std"],
                runs=int(row["num_runs"]),
            )
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-seeds", type=int, default=2)
    parser.add_argument("--epochs-per-task", type=int, default=1)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--max-train-samples", type=int, default=64)
    parser.add_argument("--max-val-samples", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-json",
        default="experiments/paper_suite/r2_baseline_comparison.json",
    )
    args = parser.parse_args()

    config = BaselineComparisonArgs(
        num_seeds=args.num_seeds,
        epochs_per_task=args.epochs_per_task,
        dim=args.dim,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        vocab_size=args.vocab_size,
        device=args.device,
        output_json=args.output_json,
    )

    set_seed(config.num_seeds)
    train_loaders, val_loaders = get_continual_dataloaders(
        dataset_name=config.dataset_name,
        batch_size=config.batch_size,
        max_length=config.max_length,
        classes_per_task=config.classes_per_task,
        num_workers=config.num_workers,
        max_train_samples=config.max_train_samples,
        max_val_samples=config.max_val_samples,
        tokenizer_vocab_size=config.vocab_size,
        use_char_tokenizer=config.vocab_size <= 256,
        label_mode=config.label_mode,
    )
    num_tasks = len(train_loaders)

    methods = ["fine_tune", "ewc", "si", "mas", "task_routing", "prototype"]
    per_seed: Dict[str, List[Dict[str, object]]] = {m: [] for m in methods}

    output_base = Path(config.output_json)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    for seed_offset in range(config.num_seeds):
        seed = 42 + seed_offset
        print(f"Seed {seed}:")
        for method in methods:
            if method in ("fine_tune", "ewc", "si", "mas"):
                metrics = run_vanilla_baseline(method, config, seed, train_loaders, val_loaders)
                seed_output = None
            else:
                seed_output = str(
                    output_base.with_name(
                        f"{output_base.stem}_{method}_seed{seed}.json"
                    )
                )
                routing_mode = "task" if method == "task_routing" else "prototype"
                metrics = run_asam_row(routing_mode, config, seed, seed_output)
            entry = {"seed": seed, **metrics}
            per_seed[method].append(entry)
            print(
                f"  {method:14s} acc={metrics['avg_accuracy']:.4f} "
                f"forget={metrics['avg_forgetting']:.4f} "
                f"bwt={metrics['backward_transfer']:.4f}"
            )

    methods_summary: Dict[str, Dict[str, float]] = {}
    for method in methods:
        rows = per_seed[method]
        methods_summary[method] = {
            "num_runs": float(len(rows)),
            "accuracy_mean": float(np.mean([r["avg_accuracy"] for r in rows])),
            "accuracy_std": float(np.std([r["avg_accuracy"] for r in rows])),
            "forgetting_mean": float(np.mean([r["avg_forgetting"] for r in rows])),
            "forgetting_std": float(np.std([r["avg_forgetting"] for r in rows])),
            "backward_transfer_mean": float(
                np.mean([r["backward_transfer"] for r in rows])
            ),
            "backward_transfer_std": float(
                np.std([r["backward_transfer"] for r in rows])
            ),
        }

    summary = {
        "config": {
            **asdict(config),
            "protocol": "task_incremental_multihead",
            "label_mode": "local",
            "head_mode": "multi",
            "train_task_id_mode": "oracle",
            "eval_task_id_mode": "oracle",
            "tokenizer": "char" if config.vocab_size <= 256 else "bpe",
            "num_tasks": num_tasks,
            "asam_rows_replay_batch_size": config.replay_batch_size,
            "vanilla_rows_replay": "none",
        },
        "methods": methods_summary,
        "per_seed": per_seed,
    }
    with open(config.output_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    table_path = output_base.with_suffix(".table.md")
    with open(table_path, "w", encoding="utf-8") as handle:
        handle.write(build_markdown_table(methods_summary) + "\n")

    print(f"Results saved to {config.output_json}")


if __name__ == "__main__":
    main()
