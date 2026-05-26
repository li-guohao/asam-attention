"""
Smoke tests for continual ablation runner.
"""

import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from experiments.run_continual_text_ablation import AblationArgs, build_benchmark_args, run_ablation


def test_build_benchmark_args_forwards_transport_weight_override():
    args = AblationArgs(transport_weight=0.2, prototype_top_k=3)
    benchmark_args = build_benchmark_args(
        args,
        {
            "name": "no_transport",
            "routing_mode": "prototype",
            "adaptive_hyperparameters": False,
            "adaptation_strategy": "correlation",
            "transport_weight": 0.0,
            "prototype_top_k": 1,
        },
        seed=42,
        output_json=None,
    )

    assert benchmark_args.transport_weight == 0.0
    assert benchmark_args.prototype_top_k == 1



def test_continual_text_ablation_runner_exports_summary_table(tmp_path):
    output_json = tmp_path / "ablation_summary.json"
    args = AblationArgs(
        max_length=48,
        batch_size=4,
        max_train_samples=8,
        max_val_samples=4,
        dim=32,
        num_heads=2,
        num_layers=1,
        epochs_per_task=1,
        num_seeds=2,
        output_json=str(output_json),
    )

    results = run_ablation(args)

    assert output_json.exists()
    assert len(results["aggregated_strategies"]) == 5
    assert len(results["per_run_strategies"]) == 10
    assert {row["strategy"] for row in results["aggregated_strategies"]} == {
        "task_routing",
        "no_adaptation",
        "correlation",
        "dual_transport",
        "meta_secant",
    }
    assert "best_avg_accuracy" in results
    assert "lowest_avg_forgetting" in results

    persisted = json.loads(output_json.read_text(encoding="utf-8"))
    assert len(persisted["aggregated_strategies"]) == 5
    assert len(persisted["per_run_strategies"]) == 10
    table_path = Path(persisted["table_path"])
    csv_path = Path(persisted["csv_path"])
    plot_path = Path(persisted["plot_path"])
    report_path = Path(persisted["report_path"])
    assert table_path.exists()
    assert csv_path.exists()
    assert plot_path.exists()
    assert report_path.exists()
    table_text = table_path.read_text(encoding="utf-8")
    report_text = report_path.read_text(encoding="utf-8")
    assert "| Strategy | Routing | Accuracy (mean?std) |" in table_text
    assert "meta_secant" in table_text
    assert "dual_transport" in table_text
    assert "task_routing" in table_text
    assert "Seeds per strategy" in report_text
    assert "# Continual Ablation Summary" in report_text
