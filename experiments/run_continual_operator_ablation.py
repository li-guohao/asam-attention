"""
Operator-level ablation runner for continual ASAM text benchmarks.
"""

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.run_continual_text_benchmark import RealBenchmarkArgs, run_benchmark


@dataclass
class OperatorAblationArgs:
    dataset_name: str = "split_ag_news"
    classes_per_task: int = 2
    max_length: int = 128
    batch_size: int = 8
    max_train_samples: Optional[int] = 64
    max_val_samples: Optional[int] = 32
    num_workers: int = 0
    dim: int = 64
    num_heads: int = 4
    num_layers: int = 1
    top_k_patterns: int = 2
    num_prototypes: int = 0
    prototype_slots_per_task: int = 2
    prototype_top_k: int = 2
    learning_rate: float = 3e-4
    epochs_per_task: int = 1
    overlap_weight: float = 0.1
    stability_weight: float = 0.1
    balance_weight: float = 0.05
    diversity_weight: float = 0.05
    transport_weight: float = 0.05
    replay_batch_size: int = 4
    prototype_routing_strategy: str = "sinkhorn_topk"
    prototype_reset_threshold: float = 0.01
    prototype_split_threshold: float = 0.20
    prototype_noise_scale: float = 0.05
    prototype_merge_threshold: float = 0.9
    prototype_merge_usage_threshold: float = 0.1
    prototype_masked_sinkhorn_candidate_k: int = 0
    prototype_masked_sinkhorn_capacity_bias: float = 0.0
    prototype_prior_strength: float = 1.0
    prototype_capacity_blend: float = 0.5
    prototype_relocation_strength: float = 0.75
    seed: int = 42
    num_seeds: int = 2
    device: str = "cpu"
    output_json: Optional[str] = None


OPERATOR_STRATEGIES = [
    {
        "name": "sinkhorn_topk",
        "prototype_routing_strategy": "sinkhorn_topk",
        "transport_weight": 0.05,
        "prototype_merge_usage_threshold": 0.1,
        "prototype_relocation_strength": 0.75,
    },
    {
        "name": "kl_topk",
        "prototype_routing_strategy": "kl_topk",
        "transport_weight": 0.05,
        "prototype_merge_usage_threshold": 0.1,
        "prototype_relocation_strength": 0.75,
    },
    {
        "name": "masked_sinkhorn_topk",
        "prototype_routing_strategy": "masked_sinkhorn_topk",
        "transport_weight": 0.05,
        "prototype_merge_usage_threshold": 0.1,
        "prototype_relocation_strength": 0.75,
    },
    {
        "name": "no_transport",
        "prototype_routing_strategy": "sinkhorn_topk",
        "transport_weight": 0.0,
        "prototype_merge_usage_threshold": 0.1,
        "prototype_relocation_strength": 0.75,
    },
    {
        "name": "no_merge",
        "prototype_routing_strategy": "sinkhorn_topk",
        "transport_weight": 0.05,
        "prototype_merge_usage_threshold": 0.0,
        "prototype_relocation_strength": 0.75,
    },
    {
        "name": "no_relocation",
        "prototype_routing_strategy": "sinkhorn_topk",
        "transport_weight": 0.05,
        "prototype_merge_usage_threshold": 0.1,
        "prototype_relocation_strength": 0.0,
    },
]


def build_benchmark_args(args: OperatorAblationArgs, strategy: Dict[str, object], seed: int, output_json: Optional[str]) -> RealBenchmarkArgs:
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
        prototype_routing_strategy=str(strategy.get("prototype_routing_strategy", args.prototype_routing_strategy)),
        num_prototypes=int(strategy.get("num_prototypes", args.num_prototypes)),
        prototype_slots_per_task=int(strategy.get("prototype_slots_per_task", args.prototype_slots_per_task)),
        prototype_top_k=int(strategy.get("prototype_top_k", args.prototype_top_k)),
        learning_rate=args.learning_rate,
        epochs_per_task=args.epochs_per_task,
        overlap_weight=args.overlap_weight,
        stability_weight=args.stability_weight,
        balance_weight=args.balance_weight,
        diversity_weight=args.diversity_weight,
        transport_weight=float(strategy.get("transport_weight", args.transport_weight)),
        replay_batch_size=args.replay_batch_size,
        prototype_reset_threshold=args.prototype_reset_threshold,
        prototype_split_threshold=args.prototype_split_threshold,
        prototype_noise_scale=args.prototype_noise_scale,
        prototype_merge_threshold=args.prototype_merge_threshold,
        prototype_merge_usage_threshold=float(strategy.get("prototype_merge_usage_threshold", args.prototype_merge_usage_threshold)),
        prototype_masked_sinkhorn_candidate_k=int(
            strategy.get("prototype_masked_sinkhorn_candidate_k", args.prototype_masked_sinkhorn_candidate_k)
        ),
        prototype_masked_sinkhorn_capacity_bias=float(
            strategy.get("prototype_masked_sinkhorn_capacity_bias", args.prototype_masked_sinkhorn_capacity_bias)
        ),
        prototype_prior_strength=args.prototype_prior_strength,
        prototype_capacity_blend=args.prototype_capacity_blend,
        prototype_relocation_strength=float(strategy.get("prototype_relocation_strength", args.prototype_relocation_strength)),
        adaptive_hyperparameters=False,
        adaptation_strategy="correlation",
        device=args.device,
        seed=seed,
        output_json=output_json,
    )


def summarize_result(name: str, strategy: Dict[str, object], result: Dict[str, object]) -> Dict[str, object]:
    theory = result.get("theory_diagnostics", {})
    lifecycle = result.get("prototype_lifecycle", [])
    def last(series_name: str) -> float:
        values = theory.get(series_name, [0.0]) or [0.0]
        return float(values[-1])
    return {
        "strategy": name,
        "routing_strategy": str(result.get("config", {}).get("prototype_routing_strategy", strategy.get("prototype_routing_strategy", "sinkhorn_topk"))),
        "transport_weight": float(result.get("config", {}).get("transport_weight", strategy.get("transport_weight", 0.0))),
        "merge_usage_threshold": float(result.get("config", {}).get("prototype_merge_usage_threshold", strategy.get("prototype_merge_usage_threshold", 0.0))),
        "relocation_strength": float(result.get("config", {}).get("prototype_relocation_strength", strategy.get("prototype_relocation_strength", 0.0))),
        "avg_accuracy": float(result.get("avg_accuracy", 0.0)),
        "avg_forgetting": float(result.get("avg_forgetting", 0.0)),
        "backward_transfer": float(result.get("backward_transfer", 0.0)),
        "final_transport_gap": float((theory.get("stage_transport_gap", [0.0]) or [0.0])[-1]),
        "final_transport_loss": float((theory.get("stage_transport_loss", [0.0]) or [0.0])[-1]),
        "final_routing_stability": float((theory.get("stage_routing_stability_loss", [0.0]) or [0.0])[-1]),
        "final_candidate_support_residual": last("stage_candidate_support_residual"),
        "final_support_projection_residual": last("stage_support_projection_residual"),
        "final_support_residual_delta": last("stage_support_residual_delta"),
        "final_effective_capacity_residual": last("stage_effective_capacity_residual"),
        "final_support_density": last("stage_support_density"),
        "final_support_size": last("stage_support_size"),
        "final_support_active_prototypes": last("stage_support_active_prototypes"),
        "final_support_weight_leakage": last("stage_support_weight_leakage"),
        "final_capacity_bias_selection_rate": last("stage_capacity_bias_selection_rate"),
        "total_resets": sum(int(item.get("reset_count", 0)) for item in lifecycle),
        "total_splits": sum(int(item.get("split_count", 0)) for item in lifecycle),
        "total_merges": sum(int(item.get("merge_count", 0)) for item in lifecycle),
    }


def aggregate_strategy_runs(strategy: Dict[str, object], rows: List[Dict[str, object]]) -> Dict[str, object]:
    def mean_std(key: str) -> tuple[float, float]:
        values = np.array([float(row[key]) for row in rows], dtype=float)
        return float(values.mean()), float(values.std(ddof=0))

    avg_accuracy_mean, avg_accuracy_std = mean_std("avg_accuracy")
    avg_forgetting_mean, avg_forgetting_std = mean_std("avg_forgetting")
    bwt_mean, bwt_std = mean_std("backward_transfer")
    transport_gap_mean, transport_gap_std = mean_std("final_transport_gap")
    transport_loss_mean, transport_loss_std = mean_std("final_transport_loss")
    routing_stability_mean, routing_stability_std = mean_std("final_routing_stability")
    candidate_support_mean, candidate_support_std = mean_std("final_candidate_support_residual")
    support_projection_mean, support_projection_std = mean_std("final_support_projection_residual")
    support_residual_delta_mean, support_residual_delta_std = mean_std("final_support_residual_delta")
    effective_capacity_mean, effective_capacity_std = mean_std("final_effective_capacity_residual")
    support_density_mean, support_density_std = mean_std("final_support_density")
    support_size_mean, support_size_std = mean_std("final_support_size")
    support_active_prototypes_mean, support_active_prototypes_std = mean_std("final_support_active_prototypes")
    support_weight_leakage_mean, support_weight_leakage_std = mean_std("final_support_weight_leakage")
    capacity_bias_selection_rate_mean, capacity_bias_selection_rate_std = mean_std("final_capacity_bias_selection_rate")

    return {
        "strategy": str(strategy["name"]),
        "routing_strategy": str(strategy.get("prototype_routing_strategy", "sinkhorn_topk")),
        "transport_weight": float(strategy.get("transport_weight", 0.0)),
        "merge_usage_threshold": float(strategy.get("prototype_merge_usage_threshold", 0.0)),
        "relocation_strength": float(strategy.get("prototype_relocation_strength", 0.0)),
        "num_runs": len(rows),
        "avg_accuracy_mean": avg_accuracy_mean,
        "avg_accuracy_std": avg_accuracy_std,
        "avg_forgetting_mean": avg_forgetting_mean,
        "avg_forgetting_std": avg_forgetting_std,
        "backward_transfer_mean": bwt_mean,
        "backward_transfer_std": bwt_std,
        "final_transport_gap_mean": transport_gap_mean,
        "final_transport_gap_std": transport_gap_std,
        "final_transport_loss_mean": transport_loss_mean,
        "final_transport_loss_std": transport_loss_std,
        "final_routing_stability_mean": routing_stability_mean,
        "final_routing_stability_std": routing_stability_std,
        "final_candidate_support_residual_mean": candidate_support_mean,
        "final_candidate_support_residual_std": candidate_support_std,
        "final_support_projection_residual_mean": support_projection_mean,
        "final_support_projection_residual_std": support_projection_std,
        "final_support_residual_delta_mean": support_residual_delta_mean,
        "final_support_residual_delta_std": support_residual_delta_std,
        "final_effective_capacity_residual_mean": effective_capacity_mean,
        "final_effective_capacity_residual_std": effective_capacity_std,
        "final_support_density_mean": support_density_mean,
        "final_support_density_std": support_density_std,
        "final_support_size_mean": support_size_mean,
        "final_support_size_std": support_size_std,
        "final_support_active_prototypes_mean": support_active_prototypes_mean,
        "final_support_active_prototypes_std": support_active_prototypes_std,
        "final_support_weight_leakage_mean": support_weight_leakage_mean,
        "final_support_weight_leakage_std": support_weight_leakage_std,
        "final_capacity_bias_selection_rate_mean": capacity_bias_selection_rate_mean,
        "final_capacity_bias_selection_rate_std": capacity_bias_selection_rate_std,
    }


def build_markdown_table(rows: List[Dict[str, object]]) -> str:
    lines = [
        "| Strategy | Routing | Transport W | Merge Usage | Relocation | Accuracy (mean±std) | Forgetting (mean±std) | BWT (mean±std) | Final Gap (mean±std) | Final Transport (mean±std) | Final Candidate Residual (mean±std) | Final Support Residual (mean±std) | Final Delta (mean±std) | Final Density (mean±std) | Runs |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {strategy} | {routing_strategy} | {transport_weight:.2f} | {merge_usage_threshold:.2f} | {relocation_strength:.2f} | {avg_accuracy_mean:.4f}±{avg_accuracy_std:.4f} | {avg_forgetting_mean:.4f}±{avg_forgetting_std:.4f} | {backward_transfer_mean:.4f}±{backward_transfer_std:.4f} | {final_transport_gap_mean:.4f}±{final_transport_gap_std:.4f} | {final_transport_loss_mean:.4f}±{final_transport_loss_std:.4f} | {final_candidate_support_residual_mean:.4f}±{final_candidate_support_residual_std:.4f} | {final_support_projection_residual_mean:.4f}±{final_support_projection_residual_std:.4f} | {final_support_residual_delta_mean:.4f}±{final_support_residual_delta_std:.4f} | {final_support_density_mean:.4f}±{final_support_density_std:.4f} | {num_runs} |".format(**row)
        )
    return "\n".join(lines)


def save_csv(rows: List[Dict[str, object]], path: Path):
    fields = [
        "strategy",
        "routing_strategy",
        "transport_weight",
        "merge_usage_threshold",
        "relocation_strength",
        "num_runs",
        "avg_accuracy_mean",
        "avg_accuracy_std",
        "avg_forgetting_mean",
        "avg_forgetting_std",
        "backward_transfer_mean",
        "backward_transfer_std",
        "final_transport_gap_mean",
        "final_transport_gap_std",
        "final_transport_loss_mean",
        "final_transport_loss_std",
        "final_routing_stability_mean",
        "final_routing_stability_std",
        "final_candidate_support_residual_mean",
        "final_candidate_support_residual_std",
        "final_support_projection_residual_mean",
        "final_support_projection_residual_std",
        "final_support_residual_delta_mean",
        "final_support_residual_delta_std",
        "final_effective_capacity_residual_mean",
        "final_effective_capacity_residual_std",
        "final_support_density_mean",
        "final_support_density_std",
        "final_support_size_mean",
        "final_support_size_std",
        "final_support_active_prototypes_mean",
        "final_support_active_prototypes_std",
        "final_support_weight_leakage_mean",
        "final_support_weight_leakage_std",
        "final_capacity_bias_selection_rate_mean",
        "final_capacity_bias_selection_rate_std",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def save_plot(rows: List[Dict[str, object]], path: Path):
    labels = [row["strategy"] for row in rows]
    accuracy = [row["avg_accuracy_mean"] for row in rows]
    accuracy_err = [row["avg_accuracy_std"] for row in rows]
    forgetting = [row["avg_forgetting_mean"] for row in rows]
    forgetting_err = [row["avg_forgetting_std"] for row in rows]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(x, accuracy, yerr=accuracy_err, color="#4C78A8", capsize=4)
    axes[0].set_title("Average Accuracy")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20)
    axes[0].set_ylim(0.0, max(1.0, max(accuracy) + max(accuracy_err + [0.0]) + 0.05))

    axes[1].bar(x, forgetting, yerr=forgetting_err, color="#E45756", capsize=4)
    axes[1].set_title("Average Forgetting")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20)
    axes[1].set_ylim(0.0, max(0.1, max(forgetting) + max(forgetting_err + [0.0]) + 0.05))

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_report(summary: Dict[str, object]) -> str:
    rows = summary["aggregated_strategies"]
    best_accuracy = summary["best_avg_accuracy"]
    best_forgetting = summary["lowest_avg_forgetting"]
    lines = [
        "# Continual Operator Ablation Summary",
        "",
        f"- Dataset: `{summary['config']['dataset_name']}`",
        f"- Number of operator settings: `{len(rows)}`",
        f"- Seeds per setting: `{summary['config']['num_seeds']}`",
        f"- Best average accuracy: `{best_accuracy['strategy']} ({best_accuracy['value']:.4f})`",
        f"- Lowest average forgetting: `{best_forgetting['strategy']} ({best_forgetting['value']:.4f})`",
        "",
        "## Aggregated Table",
        "",
        build_markdown_table(rows),
        "",
        "## Notes",
        "",
        "- `sinkhorn_topk` is the full capacity-aware routing baseline.",
        "- `kl_topk` removes Sinkhorn balancing while keeping sparse prototype routing.",
        "- `masked_sinkhorn_topk` runs Sinkhorn directly on the sparse top-k support.",
        "- `no_transport` disables the transport-loss training term.",
        "- `no_merge` disables merge events through `prototype_merge_usage_threshold=0`.",
        "- `no_relocation` disables relocation updates through `prototype_relocation_strength=0`.",
    ]
    return "\n".join(lines)


def run_operator_ablation(args: OperatorAblationArgs) -> Dict[str, object]:
    output_base = Path(args.output_json) if args.output_json else None
    per_run_results = []
    raw_results = {}

    for strategy in OPERATOR_STRATEGIES:
        for seed_offset in range(args.num_seeds):
            seed = args.seed + seed_offset
            strategy_output = None
            if output_base is not None:
                output_base.parent.mkdir(parents=True, exist_ok=True)
                strategy_output = str(output_base.with_name(f"{output_base.stem}_{strategy['name']}_seed{seed}.json"))

            benchmark_args = build_benchmark_args(args, strategy, seed, strategy_output)
            result = run_benchmark(benchmark_args)
            raw_results[f"{strategy['name']}_seed{seed}"] = result
            summarized = summarize_result(strategy["name"], strategy, result)
            summarized["seed"] = seed
            per_run_results.append(summarized)

    aggregated = []
    for strategy in OPERATOR_STRATEGIES:
        rows = [row for row in per_run_results if row["strategy"] == strategy["name"]]
        aggregated.append(aggregate_strategy_runs(strategy, rows))

    best_accuracy = max(aggregated, key=lambda item: item["avg_accuracy_mean"])
    best_forgetting = min(aggregated, key=lambda item: item["avg_forgetting_mean"])
    summary = {
        "config": asdict(args),
        "aggregated_strategies": aggregated,
        "per_run_strategies": per_run_results,
        "best_avg_accuracy": {
            "strategy": best_accuracy["strategy"],
            "value": best_accuracy["avg_accuracy_mean"],
        },
        "lowest_avg_forgetting": {
            "strategy": best_forgetting["strategy"],
            "value": best_forgetting["avg_forgetting_mean"],
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
        save_plot(aggregated, plot_path)
        report_path.write_text(build_report(summary), encoding="utf-8")

    summary["raw_results"] = raw_results
    return summary


def parse_args() -> OperatorAblationArgs:
    parser = argparse.ArgumentParser(description="Run operator-level continual ASAM ablations on the text benchmark")
    for field_name, field_def in OperatorAblationArgs.__dataclass_fields__.items():
        arg_name = f"--{field_name.replace('_', '-')}"
        default_value = field_def.default
        arg_type = type(default_value) if default_value is not None else str
        parser.add_argument(arg_name, type=arg_type, default=default_value)
    namespace = parser.parse_args()
    return OperatorAblationArgs(**vars(namespace))


def main():
    args = parse_args()
    results = run_operator_ablation(args)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
