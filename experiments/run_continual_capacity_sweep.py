"""
Capacity sweep runner for continual ASAM text benchmarks.
"""

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.run_continual_text_benchmark import RealBenchmarkArgs, run_benchmark


@dataclass
class CapacitySweepArgs:
    dataset_name: str = "split_ag_news"
    classes_per_task: int = 2
    max_length: int = 128
    batch_size: int = 8
    max_train_samples: Optional[int] = 256
    max_val_samples: Optional[int] = 128
    num_workers: int = 0
    dim: int = 64
    num_heads: int = 4
    num_layers: int = 1
    top_k_patterns: int = 2
    prototype_routing_strategy: str = "sinkhorn_topk"
    learning_rate: float = 3e-4
    epochs_per_task: int = 1
    overlap_weight: float = 0.1
    stability_weight: float = 0.1
    balance_weight: float = 0.05
    diversity_weight: float = 0.05
    transport_weight: float = 0.05
    replay_batch_size: int = 4
    prototype_reset_threshold: float = 0.01
    prototype_split_threshold: float = 0.20
    prototype_noise_scale: float = 0.05
    prototype_merge_threshold: float = 0.9
    prototype_merge_usage_threshold: float = 0.1
    prototype_prior_strength: float = 1.0
    prototype_capacity_blend: float = 0.5
    prototype_relocation_strength: float = 0.75
    adaptive_hyperparameters: bool = False
    adaptation_strategy: str = "meta_secant"
    num_seeds: int = 3
    seed: int = 42
    device: str = "cpu"
    prototype_slots_grid: str = "2,4"
    prototype_topk_grid: str = "1,2"
    include_task_baseline: bool = True
    output_json: Optional[str] = None


def parse_int_grid(spec: str) -> List[int]:
    values = []
    for item in spec.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        values.append(int(stripped))
    if not values:
        raise ValueError("grid specification must contain at least one integer")
    return sorted(set(values))


def _mean_std(rows: Sequence[Dict[str, object]], key: str) -> Tuple[float, float]:
    values = [float(row.get(key, 0.0)) for row in rows]
    return float(np.mean(values)), float(np.std(values))


def build_task_baseline_args(args: CapacitySweepArgs, seed: int, output_json: Optional[str]) -> RealBenchmarkArgs:
    return RealBenchmarkArgs(
        dataset_name=args.dataset_name,
        classes_per_task=args.classes_per_task,
        max_length=args.max_length,
        batch_size=args.batch_size,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        num_workers=args.num_workers,
        dim=args.dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        top_k_patterns=args.top_k_patterns,
        routing_mode="task",
        prototype_routing_strategy=args.prototype_routing_strategy,
        num_prototypes=0,
        prototype_slots_per_task=1,
        prototype_top_k=1,
        learning_rate=args.learning_rate,
        epochs_per_task=args.epochs_per_task,
        overlap_weight=args.overlap_weight,
        stability_weight=args.stability_weight,
        balance_weight=args.balance_weight,
        diversity_weight=args.diversity_weight,
        transport_weight=args.transport_weight,
        replay_batch_size=args.replay_batch_size,
        prototype_reset_threshold=args.prototype_reset_threshold,
        prototype_split_threshold=args.prototype_split_threshold,
        prototype_noise_scale=args.prototype_noise_scale,
        prototype_merge_threshold=args.prototype_merge_threshold,
        prototype_merge_usage_threshold=args.prototype_merge_usage_threshold,
        prototype_prior_strength=args.prototype_prior_strength,
        prototype_capacity_blend=args.prototype_capacity_blend,
        prototype_relocation_strength=args.prototype_relocation_strength,
        adaptive_hyperparameters=False,
        adaptation_strategy="correlation",
        device=args.device,
        seed=seed,
        output_json=output_json,
    )


def build_capacity_args(
    args: CapacitySweepArgs,
    seed: int,
    prototype_slots_per_task: int,
    prototype_top_k: int,
    output_json: Optional[str],
) -> RealBenchmarkArgs:
    return RealBenchmarkArgs(
        dataset_name=args.dataset_name,
        classes_per_task=args.classes_per_task,
        max_length=args.max_length,
        batch_size=args.batch_size,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        num_workers=args.num_workers,
        dim=args.dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        top_k_patterns=args.top_k_patterns,
        routing_mode="prototype",
        prototype_routing_strategy=args.prototype_routing_strategy,
        num_prototypes=0,
        prototype_slots_per_task=prototype_slots_per_task,
        prototype_top_k=prototype_top_k,
        learning_rate=args.learning_rate,
        epochs_per_task=args.epochs_per_task,
        overlap_weight=args.overlap_weight,
        stability_weight=args.stability_weight,
        balance_weight=args.balance_weight,
        diversity_weight=args.diversity_weight,
        transport_weight=args.transport_weight,
        replay_batch_size=args.replay_batch_size,
        prototype_reset_threshold=args.prototype_reset_threshold,
        prototype_split_threshold=args.prototype_split_threshold,
        prototype_noise_scale=args.prototype_noise_scale,
        prototype_merge_threshold=args.prototype_merge_threshold,
        prototype_merge_usage_threshold=args.prototype_merge_usage_threshold,
        prototype_prior_strength=args.prototype_prior_strength,
        prototype_capacity_blend=args.prototype_capacity_blend,
        prototype_relocation_strength=args.prototype_relocation_strength,
        adaptive_hyperparameters=args.adaptive_hyperparameters,
        adaptation_strategy=args.adaptation_strategy,
        device=args.device,
        seed=seed,
        output_json=output_json,
    )


def summarize_result(
    config_name: str,
    result: Dict[str, object],
    prototype_slots_per_task: Optional[int],
    prototype_top_k: Optional[int],
) -> Dict[str, object]:
    theory = result.get("theory_diagnostics", {})
    resolved = result.get("resolved_prototype_layout", {})
    return {
        "config_name": config_name,
        "routing_mode": str(result.get("config", {}).get("routing_mode", "prototype")),
        "prototype_slots_per_task": prototype_slots_per_task,
        "prototype_top_k": prototype_top_k,
        "resolved_num_prototypes": int(resolved.get("num_prototypes", 0) or 0),
        "resolved_prototype_top_k": int(resolved.get("prototype_top_k", 0) or 0),
        "avg_accuracy": float(result.get("avg_accuracy", 0.0)),
        "avg_forgetting": float(result.get("avg_forgetting", 0.0)),
        "backward_transfer": float(result.get("backward_transfer", 0.0)),
        "final_transport_gap": float((theory.get("stage_transport_gap", [0.0]) or [0.0])[-1]),
        "final_routing_stability": float((theory.get("stage_routing_stability_loss", [0.0]) or [0.0])[-1]),
    }


def aggregate_config_runs(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    sample = rows[0]
    avg_accuracy_mean, avg_accuracy_std = _mean_std(rows, "avg_accuracy")
    avg_forgetting_mean, avg_forgetting_std = _mean_std(rows, "avg_forgetting")
    backward_transfer_mean, backward_transfer_std = _mean_std(rows, "backward_transfer")
    final_transport_gap_mean, final_transport_gap_std = _mean_std(rows, "final_transport_gap")
    final_routing_stability_mean, final_routing_stability_std = _mean_std(rows, "final_routing_stability")
    return {
        "config_name": sample["config_name"],
        "routing_mode": sample["routing_mode"],
        "prototype_slots_per_task": sample["prototype_slots_per_task"],
        "prototype_top_k": sample["prototype_top_k"],
        "resolved_num_prototypes": sample["resolved_num_prototypes"],
        "resolved_prototype_top_k": sample["resolved_prototype_top_k"],
        "num_runs": len(rows),
        "avg_accuracy_mean": avg_accuracy_mean,
        "avg_accuracy_std": avg_accuracy_std,
        "avg_forgetting_mean": avg_forgetting_mean,
        "avg_forgetting_std": avg_forgetting_std,
        "backward_transfer_mean": backward_transfer_mean,
        "backward_transfer_std": backward_transfer_std,
        "final_transport_gap_mean": final_transport_gap_mean,
        "final_transport_gap_std": final_transport_gap_std,
        "final_routing_stability_mean": final_routing_stability_mean,
        "final_routing_stability_std": final_routing_stability_std,
    }


def build_markdown_table(rows: Sequence[Dict[str, object]]) -> str:
    header = [
        "| Config | Mode | Slots/Task | Proto Top-k | Resolved Protos | Accuracy (mean +/- std) | Forgetting (mean +/- std) | BWT (mean +/- std) | Gap (mean +/- std) | Acc Gain vs Task | Forgetting Gain vs Task | Runs |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines = []
    for row in rows:
        slots = row["prototype_slots_per_task"] if row["prototype_slots_per_task"] is not None else "-"
        top_k = row["prototype_top_k"] if row["prototype_top_k"] is not None else "-"
        lines.append(
            "| {config_name} | {routing_mode} | {slots} | {top_k} | {resolved_num_prototypes} | "
            "{avg_accuracy_mean:.4f} +/- {avg_accuracy_std:.4f} | {avg_forgetting_mean:.4f} +/- {avg_forgetting_std:.4f} | "
            "{backward_transfer_mean:.4f} +/- {backward_transfer_std:.4f} | {final_transport_gap_mean:.4f} +/- {final_transport_gap_std:.4f} | "
            "{accuracy_gain_vs_task:+.4f} | {forgetting_gain_vs_task:+.4f} | {num_runs} |".format(
                **row,
                slots=slots,
                top_k=top_k,
            )
        )
    return "\n".join(header + lines)


def save_csv(rows: Sequence[Dict[str, object]], path: Path) -> None:
    fieldnames = [
        "config_name",
        "routing_mode",
        "prototype_slots_per_task",
        "prototype_top_k",
        "resolved_num_prototypes",
        "resolved_prototype_top_k",
        "avg_accuracy_mean",
        "avg_accuracy_std",
        "avg_forgetting_mean",
        "avg_forgetting_std",
        "backward_transfer_mean",
        "backward_transfer_std",
        "final_transport_gap_mean",
        "final_transport_gap_std",
        "final_routing_stability_mean",
        "final_routing_stability_std",
        "accuracy_gain_vs_task",
        "forgetting_gain_vs_task",
        "num_runs",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def save_plot(
    rows: Sequence[Dict[str, object]],
    slots_grid: Sequence[int],
    topk_grid: Sequence[int],
    path: Path,
) -> None:
    prototype_rows = [row for row in rows if row["routing_mode"] == "prototype"]
    accuracy_grid = np.full((len(slots_grid), len(topk_grid)), np.nan, dtype=float)
    forgetting_grid = np.full((len(slots_grid), len(topk_grid)), np.nan, dtype=float)
    gain_grid = np.full((len(slots_grid), len(topk_grid)), np.nan, dtype=float)

    for row in prototype_rows:
        slot_index = slots_grid.index(int(row["prototype_slots_per_task"]))
        topk_index = topk_grid.index(int(row["prototype_top_k"]))
        accuracy_grid[slot_index, topk_index] = float(row["avg_accuracy_mean"])
        forgetting_grid[slot_index, topk_index] = float(row["avg_forgetting_mean"])
        gain_grid[slot_index, topk_index] = float(row["accuracy_gain_vs_task"])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    panels = [
        (accuracy_grid, "Avg Accuracy", "viridis"),
        (forgetting_grid, "Avg Forgetting", "magma_r"),
        (gain_grid, "Accuracy Gain vs Task", "coolwarm"),
    ]

    for axis, (matrix, title, cmap) in zip(axes, panels):
        image = axis.imshow(matrix, aspect="auto", cmap=cmap)
        axis.set_title(title)
        axis.set_xlabel("prototype_top_k")
        axis.set_ylabel("slots_per_task")
        axis.set_xticks(range(len(topk_grid)), [str(value) for value in topk_grid])
        axis.set_yticks(range(len(slots_grid)), [str(value) for value in slots_grid])
        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                value = matrix[row_index, col_index]
                if np.isfinite(value):
                    axis.text(col_index, row_index, f"{value:.3f}", ha="center", va="center", color="white")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_report(summary: Dict[str, object]) -> str:
    rows = summary["aggregated_configs"]
    best_accuracy = summary["best_prototype_accuracy"]
    best_forgetting = summary["best_prototype_forgetting"]
    baseline = summary.get("task_baseline")
    lines = [
        "# Continual Capacity Sweep",
        "",
        f"- Dataset: `{summary['config']['dataset_name']}`",
        f"- Slots/task grid: `{summary['config']['prototype_slots_grid']}`",
        f"- Prototype top-k grid: `{summary['config']['prototype_topk_grid']}`",
        f"- Adaptive hyperparameters: `{summary['config']['adaptive_hyperparameters']}`",
        f"- Adaptation strategy: `{summary['config']['adaptation_strategy']}`",
        f"- Seeds per config: `{summary['config']['num_seeds']}`",
        "",
        f"- Best prototype avg accuracy: `{best_accuracy['config_name']} ({best_accuracy['value']:.4f})`",
        f"- Best prototype avg forgetting: `{best_forgetting['config_name']} ({best_forgetting['value']:.4f})`",
    ]
    if baseline is not None:
        lines.extend(
            [
                f"- Task baseline avg accuracy: `{baseline['avg_accuracy_mean']:.4f}`",
                f"- Task baseline avg forgetting: `{baseline['avg_forgetting_mean']:.4f}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Aggregated Table",
            "",
            build_markdown_table(rows),
            "",
            "## Notes",
            "",
            "- `slots_per_task` expands prototype capacity beyond the number of tasks.",
            "- `prototype_top_k` controls how hard the routing sparsity is per sample.",
            "- Positive `accuracy_gain_vs_task` means the prototype configuration beats task routing on average accuracy.",
            "- Positive `forgetting_gain_vs_task` means the prototype configuration forgets less than task routing.",
        ]
    )
    return "\n".join(lines)


def run_capacity_sweep(args: CapacitySweepArgs) -> Dict[str, object]:
    slots_grid = parse_int_grid(args.prototype_slots_grid)
    topk_grid = parse_int_grid(args.prototype_topk_grid)
    output_base = Path(args.output_json) if args.output_json else None
    per_run_configs = []
    raw_results = {}

    if args.include_task_baseline:
        for seed_offset in range(args.num_seeds):
            seed = args.seed + seed_offset
            baseline_output = None
            if output_base is not None:
                output_base.parent.mkdir(parents=True, exist_ok=True)
                baseline_output = str(output_base.with_name(f"{output_base.stem}_task_routing_seed{seed}.json"))
            result = run_benchmark(build_task_baseline_args(args, seed, baseline_output))
            raw_results[f"task_routing_seed{seed}"] = result
            per_run_configs.append(summarize_result("task_routing", result, None, None))

    for slots_per_task in slots_grid:
        for prototype_top_k in topk_grid:
            config_name = f"slots{slots_per_task}_topk{prototype_top_k}"
            for seed_offset in range(args.num_seeds):
                seed = args.seed + seed_offset
                config_output = None
                if output_base is not None:
                    output_base.parent.mkdir(parents=True, exist_ok=True)
                    config_output = str(
                        output_base.with_name(
                            f"{output_base.stem}_{config_name}_seed{seed}.json"
                        )
                    )
                result = run_benchmark(
                    build_capacity_args(
                        args,
                        seed=seed,
                        prototype_slots_per_task=slots_per_task,
                        prototype_top_k=prototype_top_k,
                        output_json=config_output,
                    )
                )
                raw_results[f"{config_name}_seed{seed}"] = result
                per_run_configs.append(
                    summarize_result(config_name, result, slots_per_task, prototype_top_k)
                )

    aggregated = []
    config_names = []
    if args.include_task_baseline:
        config_names.append("task_routing")
    config_names.extend(
        f"slots{slots}_topk{topk}"
        for slots in slots_grid
        for topk in topk_grid
    )

    for config_name in config_names:
        rows = [row for row in per_run_configs if row["config_name"] == config_name]
        aggregated.append(aggregate_config_runs(rows))

    baseline_row = next((row for row in aggregated if row["config_name"] == "task_routing"), None)
    for row in aggregated:
        if baseline_row is None or row["config_name"] == "task_routing":
            row["accuracy_gain_vs_task"] = 0.0
            row["forgetting_gain_vs_task"] = 0.0
            continue
        row["accuracy_gain_vs_task"] = float(row["avg_accuracy_mean"] - baseline_row["avg_accuracy_mean"])
        row["forgetting_gain_vs_task"] = float(
            baseline_row["avg_forgetting_mean"] - row["avg_forgetting_mean"]
        )

    prototype_rows = [row for row in aggregated if row["routing_mode"] == "prototype"]
    best_prototype_accuracy = max(prototype_rows, key=lambda item: item["avg_accuracy_mean"])
    best_prototype_forgetting = min(prototype_rows, key=lambda item: item["avg_forgetting_mean"])

    summary = {
        "config": asdict(args),
        "aggregated_configs": aggregated,
        "per_run_configs": per_run_configs,
        "task_baseline": baseline_row,
        "best_prototype_accuracy": {
            "config_name": best_prototype_accuracy["config_name"],
            "value": best_prototype_accuracy["avg_accuracy_mean"],
        },
        "best_prototype_forgetting": {
            "config_name": best_prototype_forgetting["config_name"],
            "value": best_prototype_forgetting["avg_forgetting_mean"],
        },
    }

    if output_base is not None:
        markdown_path = output_base.with_name(f"{output_base.stem}_table.md")
        csv_path = output_base.with_name(f"{output_base.stem}.csv")
        plot_path = output_base.with_name(f"{output_base.stem}.png")
        report_path = output_base.with_name(f"{output_base.stem}_report.md")
        summary.update(
            {
                "table_path": str(markdown_path),
                "csv_path": str(csv_path),
                "plot_path": str(plot_path),
                "report_path": str(report_path),
            }
        )
        output_base.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        markdown_path.write_text(build_markdown_table(aggregated), encoding="utf-8")
        save_csv(aggregated, csv_path)
        save_plot(aggregated, slots_grid, topk_grid, plot_path)
        report_path.write_text(build_report(summary), encoding="utf-8")

    summary["raw_results"] = raw_results
    return summary


def parse_args() -> CapacitySweepArgs:
    parser = argparse.ArgumentParser(description="Run a capacity sweep for continual ASAM prototype routing")
    for field_name, field_def in CapacitySweepArgs.__dataclass_fields__.items():
        arg_name = f"--{field_name.replace('_', '-')}"
        default_value = field_def.default
        arg_type = type(default_value) if default_value is not None else str
        parser.add_argument(arg_name, type=arg_type, default=default_value)
    namespace = parser.parse_args()
    return CapacitySweepArgs(**vars(namespace))


def main() -> None:
    args = parse_args()
    results = run_capacity_sweep(args)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
