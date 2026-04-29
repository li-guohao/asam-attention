"""
Sync the continual-learning appendix in the paper from exported experiment artifacts.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

BEGIN_MARKER = "% BEGIN AUTO-GENERATED CONTINUAL APPENDIX"
END_MARKER = "% END AUTO-GENERATED CONTINUAL APPENDIX"


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_scalar(value: float) -> str:
    return f"{value:.4f}"


def _format_pm(mean: float, std: float) -> str:
    return f"${mean:.4f} \\pm {std:.4f}$"


def _latex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def _pretty_dataset_name(name: str) -> str:
    if name == "split_ag_news":
        return "Split AG News"
    return name.replace("_", " ").title()


def _pretty_controller_name(name: str) -> str:
    if name == "meta_secant":
        return "meta-secant"
    if name == "task_routing":
        return "task-routing"
    return name.replace("_", "-")


def _format_learning_rate(value: float) -> str:
    if value <= 0:
        return _format_scalar(value)
    mantissa, exponent = f"{value:.1e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    exponent_int = int(exponent)
    return f"{mantissa} \\times 10^{{{exponent_int}}}"


def _find_strategy(rows: List[Dict], name: str) -> Optional[Dict]:
    for row in rows:
        if row.get("strategy") == name:
            return row
    return None


def _build_strategy_rows(rows: List[Dict]) -> str:
    rendered_rows = []
    for row in rows:
        rendered_rows.append(
            f"\\texttt{{{_latex_escape(row['strategy'])}}} & "
            f"{_format_pm(row['avg_accuracy_mean'], row['avg_accuracy_std'])} & "
            f"{_format_pm(row['avg_forgetting_mean'], row['avg_forgetting_std'])} & "
            f"{_format_pm(row['backward_transfer_mean'], row['backward_transfer_std'])} & "
            f"{_format_pm(row['final_transport_gap_mean'], row['final_transport_gap_std'])} "
            + r"\\"
        )
    return "\n".join(rendered_rows)


def _build_operator_rows(rows: List[Dict]) -> str:
    rendered_rows = []
    for row in rows:
        routing_key = row.get("routing_strategy", row["strategy"])
        if routing_key == "sinkhorn_topk":
            routing_label = "Sinkhorn Top-$k$"
        elif routing_key == "kl_topk":
            routing_label = "KL Top-$k$"
        else:
            routing_label = routing_key.replace("_", " ").title()
        rendered_rows.append(
            f"\\texttt{{{_latex_escape(row['strategy'])}}} & {routing_label} & "
            f"{_format_pm(row['avg_accuracy_mean'], row['avg_accuracy_std'])} & "
            f"{_format_pm(row['avg_forgetting_mean'], row['avg_forgetting_std'])} & "
            f"{_format_pm(row['final_transport_gap_mean'], row['final_transport_gap_std'])} & "
            f"{_format_pm(row['final_transport_loss_mean'], row['final_transport_loss_std'])} "
            + r"\\"
        )
    return "\n".join(rendered_rows)


def build_continual_appendix(
    benchmark: Dict,
    ablation: Dict,
    operator_ablation: Dict,
) -> str:
    benchmark_config = benchmark["config"]
    strategies = ablation["aggregated_strategies"]
    operators = operator_ablation["aggregated_strategies"]

    best_strategy = max(strategies, key=lambda row: row["avg_accuracy_mean"])
    task_routing_row = _find_strategy(strategies, "task_routing")
    sinkhorn_row = _find_strategy(operators, "sinkhorn_topk")
    kl_row = _find_strategy(operators, "kl_topk")
    no_transport_row = _find_strategy(operators, "no_transport")

    benchmark_strategy = benchmark_config.get("adaptation_strategy", "meta_secant")
    benchmark_strategy_display = _pretty_controller_name(benchmark_strategy)
    dataset_name = benchmark_config.get("dataset_name", "split_ag_news")
    dataset_display_name = _pretty_dataset_name(dataset_name)
    task_count = benchmark.get("num_tasks", 0)
    classes_per_task = benchmark_config.get("classes_per_task", "?")
    max_length = benchmark_config.get("max_length", "?")
    batch_size = benchmark_config.get("batch_size", "?")
    train_samples = benchmark_config.get("max_train_samples", "?")
    val_samples = benchmark_config.get("max_val_samples", "?")
    dim = benchmark_config.get("dim", "?")
    num_heads = benchmark_config.get("num_heads", "?")
    num_layers = benchmark_config.get("num_layers", "?")
    top_k_patterns = benchmark_config.get("top_k_patterns", "?")
    learning_rate = benchmark_config.get("learning_rate", 0.0)
    epochs_per_task = benchmark_config.get("epochs_per_task", "?")
    num_seeds = ablation["config"].get("num_seeds", "?")

    if best_strategy["strategy"] == "task_routing":
        strategy_take = (
            "The strongest strategy in the present setup is still the explicit task-conditioned baseline, "
            "not a prototype-routed variant. This is an important negative result: the current continual "
            "ASAM implementation is measurable and executable, but it does not yet outperform task-ID routing "
            "on retention at this scale."
        )
        takeaway_text = (
            "The current continual-learning results support a narrow but honest conclusion. The continual ASAM "
            "framework already provides executable transport-aware diagnostics, prototype lifecycle statistics, "
            "and operator-level ablations. However, under the present small-scale CPU configuration, these "
            "mechanisms do not yet translate into a retention advantage over an explicit task-conditioned baseline. "
            "The most defensible claim at this stage is therefore a framework-level one: continual sparse routing "
            "is measurable and partly interpretable, but stronger data scale, longer training, and richer ablations "
            "are still needed before claiming a clear empirical advantage."
        )
    else:
        strategy_take = (
            f"The strongest strategy in the present setup is {_latex_escape(best_strategy['strategy'])}, "
            "which indicates that the prototype-routed continual variant can already produce measurable differences "
            "under the current pilot benchmark."
        )
        if task_routing_row is not None:
            avg_accuracy_gap = best_strategy["avg_accuracy_mean"] - task_routing_row["avg_accuracy_mean"]
            avg_forgetting_gain = task_routing_row["avg_forgetting_mean"] - best_strategy["avg_forgetting_mean"]
            takeaway_text = (
                "The current continual-learning results support a cautious positive pilot conclusion. The continual "
                "ASAM framework already provides executable transport-aware diagnostics, prototype lifecycle statistics, "
                "and operator-level ablations. In this specific small-scale CPU run, the best prototype-routed variant "
                "improves average accuracy over " + r"\texttt{task\_routing}" + f" by {_format_scalar(avg_accuracy_gap)} and "
                f"reduces average forgetting by {_format_scalar(avg_forgetting_gain)}. At the same time, the margin is "
                f"small and the benchmark remains low-budget ({num_seeds} seeds with modest sample counts), so this "
                "should be treated as an encouraging pilot signal rather than a decisive retention claim. Stronger data "
                "scale, longer training, and richer ablations are still needed before claiming a robust empirical advantage."
            )
        else:
            takeaway_text = (
                "The current continual-learning results support a cautious positive pilot conclusion. The continual ASAM "
                "framework already provides executable transport-aware diagnostics, prototype lifecycle statistics, and "
                "operator-level ablations, and the best prototype-routed variant is measurably competitive in the present "
                "small-scale setup. However, the benchmark budget is still modest, so this should be treated as a pilot "
                "signal rather than a decisive retention claim."
            )

    operator_take_parts = [
        "Table~\\ref{tab:continual_operator_ablation} isolates several operator choices inside prototype-routed continual ASAM."
    ]
    if sinkhorn_row and kl_row:
        operator_take_parts.append(
            "Task-level differences are currently modest, but the transport-facing diagnostics already reveal a "
            f"meaningful pattern: \\texttt{{sinkhorn\\_topk}} and \\texttt{{kl\\_topk}} achieve the same mean accuracy, "
            f"while the Sinkhorn version preserves a final transport gap of {_format_scalar(sinkhorn_row['final_transport_gap_mean'])} "
            f"and the KL-based variant shows {_format_scalar(kl_row['final_transport_gap_mean'])}."
        )
    if no_transport_row:
        operator_take_parts.append(
            f"Disabling the transport term slightly lowers accuracy to {_format_scalar(no_transport_row['avg_accuracy_mean'])} "
            f"and raises the final transport-loss trace to {_format_scalar(no_transport_row['final_transport_loss_mean'])}."
        )
    operator_take_parts.append(
        "We therefore interpret the present operator study as an early mechanistic signal rather than a decisive empirical win."
    )
    operator_take = " ".join(operator_take_parts)

    strategy_rows_text = _build_strategy_rows(strategies)
    operator_rows_text = _build_operator_rows(operators)
    strategy_header = (
        "\\textbf{Strategy} & \\textbf{Avg Acc.} & \\textbf{Avg Forgetting} & \\textbf{BWT} & \\textbf{Final Gap} "
        + r"\\"
    )
    operator_header = (
        "\\textbf{Operator Setting} & \\textbf{Routing} & \\textbf{Avg Acc.} & \\textbf{Avg Forgetting} & \\textbf{Final Gap} & \\textbf{Final Transport} "
        + r"\\"
    )

    return f"""{BEGIN_MARKER}
\\appendix
\\section{{Continual ASAM Pilot Study}}
\\label{{app:continual_asam}}

To assess whether the adaptive sparse-routing ideas behind ASAM can be extended to continual learning, we implemented a pilot continual-learning variant in the repository. This appendix intentionally presents the continual results as a conservative addendum rather than a replacement for the main long-sequence claims of the paper. The goal is to document the current executable continual-learning evidence, including both positive diagnostic signals and negative comparative findings.

\\subsection{{Setup}}

We evaluate the continual extension on class-incremental {dataset_display_name} with {task_count} tasks of {classes_per_task} classes each and a compact text classifier built from continual ASAM layers. The current paper-oriented run uses sequence length ${max_length}$, batch size ${batch_size}$, ${train_samples}$ training examples and ${val_samples}$ validation examples per split, {num_layers} layer with model width ${dim}$, {num_heads} heads, top-$k={top_k_patterns}$ sparse pattern selection, AdamW with learning rate ${_format_learning_rate(learning_rate)}$, and {epochs_per_task} training epoch per task. Strategy-level and operator-level ablations are aggregated over {num_seeds} seeds. This setting is deliberately modest and CPU-runnable, so it should be interpreted as a controlled pilot benchmark rather than a final large-scale continual-learning claim.

\\subsection{{Diagnostic Benchmark}}

The single-run prototype-routing benchmark with the {benchmark_strategy_display} controller reaches average accuracy {_format_scalar(benchmark['avg_accuracy'])}, average forgetting {_format_scalar(benchmark['avg_forgetting'])}, and backward transfer {_format_scalar(benchmark['backward_transfer'])}. Because this benchmark is single-seed, we use it primarily as a diagnostic case study: it is useful for inspecting stage-wise transport loss, routing stability, entropy, and prototype occupancy, but it should not be treated as the main comparative result.

\\subsection{{Strategy-Level Ablation}}

Table~\\ref{{tab:continual_strategy_ablation}} summarizes the current multi-seed routing/controller comparison. {strategy_take}

\\begin{{table}}[htbp]
\\centering
\\caption{{Strategy-level continual ablation on {dataset_display_name} ({num_seeds} seeds). Higher accuracy and backward transfer are better; lower forgetting is better.}}
\\label{{tab:continual_strategy_ablation}}
\\begin{{tabular}}{{@{{}}lcccc@{{}}}}
\\toprule
{strategy_header}
\\midrule
{strategy_rows_text}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\subsection{{Operator-Level Ablation}}

{operator_take}

\\begin{{table}}[htbp]
\\centering
\\caption{{Operator-level continual ablation on {dataset_display_name} ({num_seeds} seeds).}}
\\label{{tab:continual_operator_ablation}}
\\begin{{tabular}}{{@{{}}lccccc@{{}}}}
\\toprule
{operator_header}
\\midrule
{operator_rows_text}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\subsection{{Takeaway}}

{takeaway_text}
{END_MARKER}
"""


def sync_paper_appendix(
    benchmark_json: Path,
    ablation_json: Path,
    operator_json: Path,
    paper_tex: Path,
    output_tex: Optional[Path] = None,
) -> str:
    appendix = build_continual_appendix(
        _load_json(benchmark_json),
        _load_json(ablation_json),
        _load_json(operator_json),
    )
    target_path = output_tex or paper_tex
    paper_text = paper_tex.read_text(encoding="utf-8")
    pattern = re.compile(
        re.escape(BEGIN_MARKER) + r".*?" + re.escape(END_MARKER),
        re.DOTALL,
    )
    if not pattern.search(paper_text):
        raise ValueError("Could not find continual appendix markers in paper TeX.")
    updated_text = pattern.sub(lambda _: appendix, paper_text, count=1)
    target_path.write_text(updated_text, encoding="utf-8")
    return appendix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync continual appendix into paper/asam_paper.tex")
    parser.add_argument(
        "--benchmark-json",
        type=Path,
        default=Path("experiments/paper_suite_paperish/continual_benchmark.json"),
    )
    parser.add_argument(
        "--ablation-json",
        type=Path,
        default=Path("experiments/paper_suite_paperish/continual_ablation.json"),
    )
    parser.add_argument(
        "--operator-json",
        type=Path,
        default=Path("experiments/paper_suite_paperish/continual_operator_ablation.json"),
    )
    parser.add_argument(
        "--paper-tex",
        type=Path,
        default=Path("paper/asam_paper.tex"),
    )
    parser.add_argument(
        "--output-tex",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--appendix-only",
        type=Path,
        default=None,
        help="Write only the generated appendix block to a standalone .tex file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    appendix = build_continual_appendix(
        _load_json(args.benchmark_json),
        _load_json(args.ablation_json),
        _load_json(args.operator_json),
    )
    if args.appendix_only is not None:
        args.appendix_only.write_text(appendix, encoding="utf-8")
    sync_paper_appendix(
        args.benchmark_json,
        args.ablation_json,
        args.operator_json,
        args.paper_tex,
        args.output_tex,
    )


if __name__ == "__main__":
    main()
