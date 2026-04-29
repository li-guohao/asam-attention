"""
One-command paper-ready pipeline for continual ASAM experiments.
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.run_continual_operator_ablation import OperatorAblationArgs, run_operator_ablation
from experiments.run_continual_text_ablation import AblationArgs, run_ablation
from experiments.run_continual_text_benchmark import RealBenchmarkArgs, run_benchmark
from scripts.sync_continual_appendix import build_continual_appendix, sync_paper_appendix


@dataclass
class PipelineArgs:
    output_dir: str = "experiments/paper_suite"
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
    prototype_routing_strategy: str = "sinkhorn_topk"
    num_prototypes: int = 0
    prototype_slots_per_task: int = 2
    prototype_top_k: int = 2
    candidate_profile: str = "default"
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
    seed: int = 42
    num_seeds: int = 2
    device: str = "cpu"
    paper_tex: Optional[str] = None
    paper_output_tex: Optional[str] = None
    appendix_only_tex: Optional[str] = None


CANDIDATE_PROFILES: Dict[str, Dict[str, object]] = {
    "default": {
        "description": "Use the explicit prototype layout values passed on the command line.",
    },
    "accuracy": {
        "description": "Accuracy-oriented prototype routing preset from the capacity sweep.",
        "num_prototypes": 0,
        "prototype_slots_per_task": 2,
        "prototype_top_k": 2,
    },
    "retention": {
        "description": "Retention-oriented prototype routing preset from the capacity sweep.",
        "num_prototypes": 0,
        "prototype_slots_per_task": 2,
        "prototype_top_k": 1,
    },
    "retention_no_transport": {
        "description": "Retention-oriented sparse routing with the transport loss disabled; this is the strongest pilot combination found so far.",
        "num_prototypes": 0,
        "prototype_slots_per_task": 2,
        "prototype_top_k": 1,
        "transport_weight": 0.0,
    },
}


def resolve_candidate_profile(args: PipelineArgs) -> Tuple[PipelineArgs, Dict[str, object]]:
    profile_name = str(args.candidate_profile).strip().lower()
    profile = CANDIDATE_PROFILES.get(profile_name)
    if profile is None:
        available = ", ".join(sorted(CANDIDATE_PROFILES))
        raise ValueError(f"Unknown candidate_profile '{args.candidate_profile}'. Expected one of: {available}.")

    overrides = {"candidate_profile": profile_name}
    valid_fields = PipelineArgs.__dataclass_fields__
    for key, value in profile.items():
        if key == "description":
            continue
        if key in valid_fields:
            overrides[key] = value

    return replace(args, **overrides), profile


def build_benchmark_args(args: PipelineArgs, output_json: str) -> RealBenchmarkArgs:
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
        num_prototypes=args.num_prototypes,
        prototype_slots_per_task=args.prototype_slots_per_task,
        prototype_top_k=args.prototype_top_k,
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
        adaptive_hyperparameters=True,
        adaptation_strategy="meta_secant",
        device=args.device,
        seed=args.seed,
        output_json=output_json,
    )


def build_ablation_args(args: PipelineArgs, output_json: str) -> AblationArgs:
    return AblationArgs(
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
        prototype_routing_strategy=args.prototype_routing_strategy,
        num_prototypes=args.num_prototypes,
        prototype_slots_per_task=args.prototype_slots_per_task,
        prototype_top_k=args.prototype_top_k,
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
        prototype_prior_strength=args.prototype_prior_strength,
        prototype_capacity_blend=args.prototype_capacity_blend,
        prototype_relocation_strength=args.prototype_relocation_strength,
        seed=args.seed,
        num_seeds=args.num_seeds,
        device=args.device,
        output_json=output_json,
    )


def build_operator_ablation_args(args: PipelineArgs, output_json: str) -> OperatorAblationArgs:
    return OperatorAblationArgs(
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
        prototype_routing_strategy=args.prototype_routing_strategy,
        num_prototypes=args.num_prototypes,
        prototype_slots_per_task=args.prototype_slots_per_task,
        prototype_top_k=args.prototype_top_k,
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
        seed=args.seed,
        num_seeds=args.num_seeds,
        device=args.device,
        output_json=output_json,
    )


def build_pipeline_report(
    args: PipelineArgs,
    benchmark_results: Dict[str, object],
    ablation_results: Dict[str, object],
    operator_ablation_results: Dict[str, object],
    manifest: Dict[str, object],
) -> str:
    best_accuracy = ablation_results.get("best_avg_accuracy", {})
    best_forgetting = ablation_results.get("lowest_avg_forgetting", {})
    operator_best_accuracy = operator_ablation_results.get("best_avg_accuracy", {})
    operator_best_forgetting = operator_ablation_results.get("lowest_avg_forgetting", {})
    resolved_config = manifest.get("resolved_config", {})
    candidate_profile = manifest.get("candidate_profile", args.candidate_profile)
    profile_description = manifest.get("candidate_profile_description")
    resolved_num_prototypes = resolved_config.get("num_prototypes", args.num_prototypes)
    resolved_slots_per_task = resolved_config.get("prototype_slots_per_task", args.prototype_slots_per_task)
    resolved_top_k = resolved_config.get("prototype_top_k", args.prototype_top_k)
    resolved_transport_weight = resolved_config.get("transport_weight", args.transport_weight)
    prototype_layout = (
        f"num_prototypes={resolved_num_prototypes}, "
        f"slots_per_task={resolved_slots_per_task}, top_k={resolved_top_k}"
    )
    lines = [
        "# Continual ASAM Paper Suite",
        "",
        "## Run Config",
        "",
        f"- Dataset: `{args.dataset_name}`",
        f"- Output directory: `{manifest['output_dir']}`",
        f"- Device: `{args.device}`",
        f"- Seeds for ablation: `{args.num_seeds}`",
        f"- Candidate profile: `{candidate_profile}`",
        f"- Prototype layout: `{prototype_layout}`",
        f"- Transport weight: `{resolved_transport_weight}`",
        "",
        "## Benchmark",
        "",
        f"- Meta-secant avg accuracy: `{benchmark_results.get('avg_accuracy', 0.0):.4f}`",
        f"- Meta-secant avg forgetting: `{benchmark_results.get('avg_forgetting', 0.0):.4f}`",
        f"- Meta-secant backward transfer: `{benchmark_results.get('backward_transfer', 0.0):.4f}`",
        f"- Benchmark JSON: `{Path(manifest['benchmark_json']).name}`",
        f"- Benchmark report: `{Path(manifest['benchmark_report']).name}`",
        "",
        "## Ablation",
        "",
        f"- Best avg accuracy: `{best_accuracy.get('strategy')} ({best_accuracy.get('value', 0.0):.4f})`",
        f"- Lowest avg forgetting: `{best_forgetting.get('strategy')} ({best_forgetting.get('value', 0.0):.4f})`",
        f"- Ablation JSON: `{Path(manifest['ablation_json']).name}`",
        f"- Ablation report: `{Path(manifest['ablation_report']).name}`",
        f"- Ablation table: `{Path(manifest['ablation_table']).name}`",
        f"- Ablation CSV: `{Path(manifest['ablation_csv']).name}`",
        f"- Ablation plot: `{Path(manifest['ablation_plot']).name}`",
        "",
        "## Operator Ablation",
        "",
        f"- Best operator avg accuracy: `{operator_best_accuracy.get('strategy')} ({operator_best_accuracy.get('value', 0.0):.4f})`",
        f"- Lowest operator avg forgetting: `{operator_best_forgetting.get('strategy')} ({operator_best_forgetting.get('value', 0.0):.4f})`",
        f"- Operator Ablation JSON: `{Path(manifest['operator_ablation_json']).name}`",
        f"- Operator Ablation report: `{Path(manifest['operator_ablation_report']).name}`",
        f"- Operator Ablation table: `{Path(manifest['operator_ablation_table']).name}`",
        f"- Operator Ablation CSV: `{Path(manifest['operator_ablation_csv']).name}`",
        f"- Operator Ablation plot: `{Path(manifest['operator_ablation_plot']).name}`",
    ]

    if profile_description:
        lines.append(f"- Profile note: {profile_description}")

    if manifest.get("synced_paper_tex") or manifest.get("appendix_only_tex"):
        lines.extend(
            [
                "",
                "## Paper Sync",
                "",
            ]
        )
        if manifest.get("paper_tex"):
            lines.append(f"- Source paper TeX: `{Path(manifest['paper_tex']).name}`")
        if manifest.get("synced_paper_tex"):
            lines.append(f"- Synced paper TeX: `{Path(manifest['synced_paper_tex']).name}`")
        if manifest.get("appendix_only_tex"):
            lines.append(f"- Standalone appendix TeX: `{Path(manifest['appendix_only_tex']).name}`")

    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "- Use the ablation report and CSV as the main strategy-level paper tables.",
            "- Use the operator ablation report and CSV for the mechanistic appendix and operator study.",
            "- Use the benchmark report as the detailed continual-learning diagnostics appendix.",
            "- Use the plot artifacts directly in slides and drafts.",
        ]
    )
    return "\n".join(lines)


def export_appendix_artifacts(
    args: PipelineArgs,
    benchmark_results: Dict[str, object],
    ablation_results: Dict[str, object],
    operator_ablation_results: Dict[str, object],
    manifest: Dict[str, object],
) -> Dict[str, str]:
    if args.paper_output_tex is not None and args.paper_tex is None:
        raise ValueError("paper_output_tex requires paper_tex to be set.")

    appendix_artifacts: Dict[str, str] = {}
    appendix_text: Optional[str] = None

    if args.appendix_only_tex is not None:
        appendix_text = build_continual_appendix(
            benchmark_results,
            ablation_results,
            operator_ablation_results,
        )
        appendix_only_path = Path(args.appendix_only_tex)
        appendix_only_path.parent.mkdir(parents=True, exist_ok=True)
        appendix_only_path.write_text(appendix_text, encoding="utf-8")
        appendix_artifacts["appendix_only_tex"] = str(appendix_only_path)

    if args.paper_tex is not None:
        paper_tex = Path(args.paper_tex)
        output_tex = Path(args.paper_output_tex) if args.paper_output_tex is not None else None
        if output_tex is not None:
            output_tex.parent.mkdir(parents=True, exist_ok=True)
        sync_paper_appendix(
            Path(manifest["benchmark_json"]),
            Path(manifest["ablation_json"]),
            Path(manifest["operator_ablation_json"]),
            paper_tex,
            output_tex,
        )
        appendix_artifacts["paper_tex"] = str(paper_tex)
        appendix_artifacts["synced_paper_tex"] = str(output_tex or paper_tex)

    return appendix_artifacts


def run_pipeline(args: PipelineArgs) -> Dict[str, object]:
    resolved_args, profile = resolve_candidate_profile(args)
    output_dir = Path(resolved_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_json = output_dir / "continual_benchmark.json"
    ablation_json = output_dir / "continual_ablation.json"
    operator_ablation_json = output_dir / "continual_operator_ablation.json"
    suite_manifest = output_dir / "paper_suite_manifest.json"
    suite_report = output_dir / "paper_suite_report.md"

    benchmark_results = run_benchmark(build_benchmark_args(resolved_args, str(benchmark_json)))
    ablation_results = run_ablation(build_ablation_args(resolved_args, str(ablation_json)))
    operator_ablation_results = run_operator_ablation(build_operator_ablation_args(resolved_args, str(operator_ablation_json)))

    manifest = {
        "config": asdict(args),
        "resolved_config": asdict(resolved_args),
        "candidate_profile": resolved_args.candidate_profile,
        "candidate_profile_description": profile.get("description"),
        "output_dir": str(output_dir),
        "benchmark_json": str(benchmark_json),
        "benchmark_plot": benchmark_results.get("plot_path"),
        "benchmark_report": benchmark_results.get("report_path"),
        "ablation_json": str(ablation_json),
        "ablation_table": ablation_results.get("table_path"),
        "ablation_csv": ablation_results.get("csv_path"),
        "ablation_plot": ablation_results.get("plot_path"),
        "ablation_report": ablation_results.get("report_path"),
        "operator_ablation_json": str(operator_ablation_json),
        "operator_ablation_table": operator_ablation_results.get("table_path"),
        "operator_ablation_csv": operator_ablation_results.get("csv_path"),
        "operator_ablation_plot": operator_ablation_results.get("plot_path"),
        "operator_ablation_report": operator_ablation_results.get("report_path"),
    }
    manifest.update(
        export_appendix_artifacts(
            resolved_args,
            benchmark_results,
            ablation_results,
            operator_ablation_results,
            manifest,
        )
    )
    suite_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    suite_report.write_text(
        build_pipeline_report(resolved_args, benchmark_results, ablation_results, operator_ablation_results, manifest),
        encoding="utf-8",
    )

    return {
        "manifest_path": str(suite_manifest),
        "report_path": str(suite_report),
        "benchmark_results": benchmark_results,
        "ablation_results": ablation_results,
        "operator_ablation_results": operator_ablation_results,
        "manifest": manifest,
    }


def parse_args() -> PipelineArgs:
    parser = argparse.ArgumentParser(description="Run the one-command paper-ready continual ASAM suite")
    for field_name, field_def in PipelineArgs.__dataclass_fields__.items():
        arg_name = f"--{field_name.replace('_', '-')}"
        default_value = field_def.default
        arg_type = type(default_value) if default_value is not None else str
        parser.add_argument(arg_name, type=arg_type, default=default_value)
    namespace = parser.parse_args()
    return PipelineArgs(**vars(namespace))


def main():
    args = parse_args()
    results = run_pipeline(args)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
