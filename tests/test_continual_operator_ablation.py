"""
Smoke tests for continual operator ablation runner.
"""

import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from experiments.run_continual_operator_ablation import OperatorAblationArgs, run_operator_ablation



def test_continual_operator_ablation_runner_exports_summary_table(tmp_path):
    output_json = tmp_path / "operator_ablation_summary.json"
    args = OperatorAblationArgs(
        max_length=48,
        batch_size=4,
        max_train_samples=8,
        max_val_samples=4,
        dim=32,
        num_heads=2,
        num_layers=1,
        epochs_per_task=1,
        num_seeds=1,
        output_json=str(output_json),
    )

    results = run_operator_ablation(args)

    assert output_json.exists()
    assert len(results["aggregated_strategies"]) == 5
    assert len(results["per_run_strategies"]) == 5
    assert {row["strategy"] for row in results["aggregated_strategies"]} == {
        "sinkhorn_topk",
        "kl_topk",
        "no_transport",
        "no_merge",
        "no_relocation",
    }
    assert "best_avg_accuracy" in results
    assert "lowest_avg_forgetting" in results

    persisted = json.loads(output_json.read_text(encoding="utf-8"))
    assert len(persisted["aggregated_strategies"]) == 5
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
    assert "| Strategy | Routing | Transport W |" in table_text
    assert "sinkhorn_topk" in table_text
    assert "kl_topk" in table_text
    assert "# Continual Operator Ablation Summary" in report_text
    assert "no_transport" in report_text
