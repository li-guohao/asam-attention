#!/usr/bin/env python3
"""Compare Continual ASAM against standard continual learning baselines.

Runs fine_tune, EWC, SI, MAS, ER, A-GEM, Continual ASAM (task routing), and
Continual ASAM (prototype routing) on the same task-incremental multi-head
Split AG News setup (char/byte-level tokenization, per-task classifier heads,
oracle task ids).

Fairness conventions (recorded in the output config):
- Method rows (fine_tune / EWC / SI / MAS / Continual ASAM) run WITHOUT replay
  so that the comparison isolates the CL mechanism itself.
- ER and A-GEM are explicit replay-based reference rows.
- EWC / SI / MAS use a small regularizer-strength (lambda) grid; the lambda
  with the best average accuracy is selected and reported.

Usage:
    python experiments/run_baseline_comparison.py --num-seeds 2
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    output_json: str = "experiments/paper_suite/r3_baseline_comparison.json"


# Regularizer-strength grids for the standard baselines.
LAMBDA_GRID: Dict[str, List[float]] = {
    "ewc": [100.0, 1000.0, 5000.0],
    "si": [0.1, 1.0, 10.0],
    "mas": [0.1, 1.0, 10.0],
}


class TaskHeadTransformer(nn.Module):
    """Vanilla transformer backbone with per-task classifier heads."""

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


class SampleReplayMemory:
    """Sample-level replay memory used by ER and A-GEM."""

    def __init__(self, max_samples: int = 512):
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.max_samples = max_samples

    def add_batch(self, inputs: torch.Tensor, labels: torch.Tensor):
        for index in range(inputs.size(0)):
            self.samples.append((inputs[index].detach().cpu(), labels[index].detach().cpu()))
        if len(self.samples) > self.max_samples:
            self.samples = self.samples[-self.max_samples:]

    def sample(self, device: torch.device, k: int = 4) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if not self.samples:
            return None
        count = min(k, len(self.samples))
        indices = torch.randint(0, len(self.samples), (count,))
        inputs = torch.stack([self.samples[i][0] for i in indices]).to(device)
        labels = torch.stack([self.samples[i][1] for i in indices]).to(device)
        return inputs, labels


def project_gradients(main_grads: List[Optional[torch.Tensor]], ref_grads: List[Optional[torch.Tensor]]):
    """A-GEM projection: make the main gradient non-increasing on the memory loss."""
    flat_g = torch.cat([g.flatten() for g in main_grads if g is not None])
    flat_r = torch.cat([r.flatten() for r in ref_grads if r is not None])
    dot = torch.dot(flat_g, flat_r)
    if dot < 0:
        coefficient = dot / (flat_r.dot(flat_r) + 1e-8)
        for g, r in zip(main_grads, ref_grads):
            if g is not None and r is not None:
                g.add_(r, alpha=-float(coefficient))


def build_model(args: BaselineComparisonArgs, num_tasks: int) -> TaskHeadTransformer:
    return TaskHeadTransformer(
        vocab_size=args.vocab_size,
        dim=args.dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_tasks=num_tasks,
        classes_per_task=args.classes_per_task,
        max_len=args.max_length,
    )


def run_vanilla_baseline(
    method_name: str,
    args: BaselineComparisonArgs,
    seed: int,
    train_loaders,
    val_loaders,
    regularizer_lambda: Optional[float] = None,
) -> Dict[str, float]:
    """Train fine_tune/EWC/SI/MAS/ER/A-GEM on the shared protocol."""
    device = torch.device(args.device)
    set_seed(seed)
    num_tasks = len(train_loaders)
    model = build_model(args, num_tasks).to(device)

    cl_info = None
    if method_name == "ewc":
        cl_info = EWC(model, importance=regularizer_lambda if regularizer_lambda is not None else 1000.0)
    elif method_name == "si":
        cl_info = SI(model, importance=regularizer_lambda if regularizer_lambda is not None else 1.0)
    elif method_name == "mas":
        cl_info = MAS(model, importance=regularizer_lambda if regularizer_lambda is not None else 1.0)

    replay = SampleReplayMemory() if method_name == "er" else None
    agem = SampleReplayMemory() if method_name == "agem" else None

    accuracy_matrix: List[List[float]] = []
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

                if replay is not None:
                    memory_batch = replay.sample(device, k=args.replay_batch_size)
                    if memory_batch is not None:
                        mem_inputs, mem_labels = memory_batch
                        inputs = torch.cat([inputs, mem_inputs], dim=0)
                        labels = torch.cat([labels, mem_labels], dim=0)

                optimizer.zero_grad()
                logits = model(inputs)
                loss = F.cross_entropy(logits, labels)
                if cl_info is not None:
                    loss = loss + cl_info.penalty()
                loss.backward()

                if agem is not None:
                    memory_batch = agem.sample(device, k=args.replay_batch_size)
                    if memory_batch is not None:
                        mem_inputs, mem_labels = memory_batch
                        main_grads = [
                            param.grad.clone() if param.grad is not None else None
                            for param in model.parameters()
                        ]
                        optimizer.zero_grad()
                        ref_loss = F.cross_entropy(model(mem_inputs), mem_labels)
                        ref_loss.backward()
                        ref_grads = [param.grad for param in model.parameters()]
                        project_gradients(main_grads, ref_grads)
                        for param, projected in zip(model.parameters(), main_grads):
                            if projected is not None:
                                param.grad.copy_(projected)

                optimizer.step()
                if isinstance(cl_info, SI):
                    cl_info.update_trajectory()

        if cl_info is not None:
            if method_name == "ewc":
                cl_info.update_fisher(train_loaders[task_id], device=args.device)
            elif method_name == "mas":
                cl_info.update_importance(train_loaders[task_id], device=args.device)
            cl_info.consolidate()

        if replay is not None or agem is not None:
            memory = replay if replay is not None else agem
            with torch.no_grad():
                for inputs, labels, _task_ids in train_loaders[task_id]:
                    memory.add_batch(inputs, labels)

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
    """Run the Continual ASAM benchmark for task or prototype routing.

    For the baseline comparison, ASAM rows run WITHOUT replay so the comparison
    isolates the routing mechanism; ER/A-GEM provide the replay-based reference.
    """
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
        replay_batch_size=0,
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


def aggregate(rows: List[Dict[str, object]]) -> Dict[str, float]:
    return {
        "num_runs": float(len(rows)),
        "accuracy_mean": float(np.mean([r["avg_accuracy"] for r in rows])),
        "accuracy_std": float(np.std([r["avg_accuracy"] for r in rows])),
        "forgetting_mean": float(np.mean([r["avg_forgetting"] for r in rows])),
        "forgetting_std": float(np.std([r["avg_forgetting"] for r in rows])),
        "backward_transfer_mean": float(np.mean([r["backward_transfer"] for r in rows])),
        "backward_transfer_std": float(np.std([r["backward_transfer"] for r in rows])),
    }


def build_markdown_table(methods: Dict[str, Dict[str, object]]) -> str:
    lines = [
        "| Method | Accuracy (mean?std) | Forgetting (mean?std) | BWT (mean?std) | Runs | Lambda |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ["fine_tune", "ewc", "si", "mas", "er", "agem", "task_routing", "prototype"]:
        row = methods[name]
        lam = row.get("lambda_chosen", "")
        lines.append(
            "| {name} | {acc:.4f}?{acc_s:.4f} | {forget:.4f}?{forget_s:.4f} | {bwt:.4f}?{bwt_s:.4f} | {runs} | {lam} |".format(
                name=name,
                acc=row["accuracy_mean"],
                acc_s=row["accuracy_std"],
                forget=row["forgetting_mean"],
                forget_s=row["forgetting_std"],
                bwt=row["backward_transfer_mean"],
                bwt_s=row["backward_transfer_std"],
                runs=int(row["num_runs"]),
                lam=lam,
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
        default="experiments/paper_suite/r3_baseline_comparison.json",
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

    methods = ["fine_tune", "ewc", "si", "mas", "er", "agem", "task_routing", "prototype"]
    per_seed: Dict[str, List[Dict[str, object]]] = {m: [] for m in methods}
    lambda_grid_results: Dict[str, Dict[str, object]] = {}

    output_base = Path(config.output_json)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    for method in methods:
        if method in LAMBDA_GRID:
            # Regularizer-strength grid: pick the lambda with the best accuracy.
            grid_summary = {}
            best_lambda = None
            best_accuracy = -1.0
            for lam in LAMBDA_GRID[method]:
                rows = []
                for seed_offset in range(config.num_seeds):
                    seed = 42 + seed_offset
                    metrics = run_vanilla_baseline(
                        method,
                        config,
                        seed,
                        train_loaders,
                        val_loaders,
                        regularizer_lambda=lam,
                    )
                    rows.append({"seed": seed, "lambda": lam, **metrics})
                agg = aggregate(rows)
                grid_summary[str(lam)] = {
                    "accuracy_mean": agg["accuracy_mean"],
                    "forgetting_mean": agg["forgetting_mean"],
                }
                if agg["accuracy_mean"] > best_accuracy:
                    best_accuracy = agg["accuracy_mean"]
                    best_lambda = lam
                per_seed[method].extend(rows)
            lambda_grid_results[method] = {
                "grid": LAMBDA_GRID[method],
                "chosen_lambda": best_lambda,
                "per_lambda": grid_summary,
            }
            chosen_rows = [row for row in per_seed[method] if row["lambda"] == best_lambda]
            methods_summary_row = aggregate(chosen_rows)
            methods_summary_row["lambda_chosen"] = best_lambda
            methods_summary_row["num_runs"] = float(len(chosen_rows))
        else:
            for seed_offset in range(config.num_seeds):
                seed = 42 + seed_offset
                if method in ("fine_tune", "er", "agem"):
                    metrics = run_vanilla_baseline(method, config, seed, train_loaders, val_loaders)
                    seed_output = None
                else:
                    seed_output = str(
                        output_base.with_name(f"{output_base.stem}_{method}_seed{seed}.json")
                    )
                    routing_mode = "task" if method == "task_routing" else "prototype"
                    metrics = run_asam_row(routing_mode, config, seed, seed_output)
                per_seed[method].append({"seed": seed, **metrics})
            methods_summary_row = aggregate(per_seed[method])

        print(
            f"{method:14s} acc={methods_summary_row['accuracy_mean']:.4f} "
            f"forget={methods_summary_row['forgetting_mean']:.4f} "
            f"lambda={methods_summary_row.get('lambda_chosen', '-')}"
        )

    methods_summary: Dict[str, Dict[str, object]] = {m: {} for m in methods}
    # Recompute summary for grid methods from their chosen rows.
    for method in methods:
        if method in LAMBDA_GRID:
            chosen = lambda_grid_results[method]["chosen_lambda"]
            rows = [row for row in per_seed[method] if row["lambda"] == chosen]
            methods_summary[method] = aggregate(rows)
            methods_summary[method]["lambda_chosen"] = chosen
            methods_summary[method]["lambda_grid"] = lambda_grid_results[method]
        else:
            methods_summary[method] = aggregate(per_seed[method])

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
            "method_rows_replay_batch_size": 0,
            "er_agem_replay_batch_size": config.replay_batch_size,
            "lambda_grid": LAMBDA_GRID,
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
