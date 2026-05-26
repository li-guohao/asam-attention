"""
Smoke tests for continual operator ablation runner.
"""

import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import experiments.run_continual_operator_ablation as operator_ablation_module
from experiments.run_continual_operator_ablation import OperatorAblationArgs, run_operator_ablation


def test_continual_operator_ablation_runner_exports_summary_table(tmp_path, monkeypatch):
    strategy_names = {strategy["name"] for strategy in operator_ablation_module.OPERATOR_STRATEGIES}
    assert "masked_sinkhorn_topk" in strategy_names
    assert "sinkhorn_support_masked" in strategy_names

    benchmark_calls = []

    def fake_run_benchmark(benchmark_args):
        benchmark_calls.append(benchmark_args)
        if benchmark_args.output_json is not None:
            Path(benchmark_args.output_json).write_text("{}", encoding="utf-8")
        assert benchmark_args.prototype_masked_sinkhorn_candidate_k in {0, 5, 7}
        assert benchmark_args.prototype_masked_sinkhorn_capacity_bias in {0.0, 0.25, 0.75}
        strategy_scores = {
            "sinkhorn_topk": 0.60,
            "sinkhorn_support_masked": 0.66,
            "masked_sinkhorn_topk": 0.65,
            "kl_topk": 0.55,
        }
        score = strategy_scores[benchmark_args.prototype_routing_strategy]
        return {
            "config": {
                "prototype_routing_strategy": benchmark_args.prototype_routing_strategy,
                "transport_weight": benchmark_args.transport_weight,
                "prototype_merge_usage_threshold": benchmark_args.prototype_merge_usage_threshold,
                "prototype_relocation_strength": benchmark_args.prototype_relocation_strength,
            },
            "avg_accuracy": score,
            "avg_forgetting": 1.0 - score,
            "backward_transfer": score / 10.0,
            "theory_diagnostics": {
                "stage_transport_gap": [0.1],
                "stage_transport_loss": [0.2],
                "stage_routing_stability_loss": [0.3],
                "stage_support_projection_residual": [0.06],
                "stage_effective_capacity_residual": [0.02],
                "stage_support_density": [0.5],
            },
            "prototype_lifecycle": [
                {
                    "reset_count": 1,
                    "split_count": 2,
                    "merge_count": 3,
                    "mean_transport_gap": 0.1,
                    "max_transport_gap": 0.2,
                }
            ],
        }

    monkeypatch.setattr(operator_ablation_module, "run_benchmark", fake_run_benchmark)
    monkeypatch.setattr(
        operator_ablation_module,
        "OPERATOR_STRATEGIES",
        [
            {
                "name": "sinkhorn_topk",
                "prototype_routing_strategy": "sinkhorn_topk",
                "transport_weight": 0.05,
                "prototype_merge_usage_threshold": 0.1,
                "prototype_relocation_strength": 0.75,
                "prototype_masked_sinkhorn_candidate_k": 5,
                "prototype_masked_sinkhorn_capacity_bias": 0.25,
            },
            {
                "name": "masked_sinkhorn_topk",
                "prototype_routing_strategy": "masked_sinkhorn_topk",
                "transport_weight": 0.05,
                "prototype_merge_usage_threshold": 0.1,
                "prototype_relocation_strength": 0.75,
                "prototype_masked_sinkhorn_candidate_k": 7,
                "prototype_masked_sinkhorn_capacity_bias": 0.75,
            },
            {
                "name": "sinkhorn_support_masked",
                "prototype_routing_strategy": "sinkhorn_support_masked",
                "transport_weight": 0.05,
                "prototype_merge_usage_threshold": 0.1,
                "prototype_relocation_strength": 0.75,
                "prototype_masked_sinkhorn_candidate_k": 0,
                "prototype_masked_sinkhorn_capacity_bias": 0.0,
            },
            {
                "name": "kl_topk",
                "prototype_routing_strategy": "kl_topk",
                "transport_weight": 0.05,
                "prototype_merge_usage_threshold": 0.1,
                "prototype_relocation_strength": 0.75,
            },
        ],
    )
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
        prototype_masked_sinkhorn_candidate_k=5,
        prototype_masked_sinkhorn_capacity_bias=0.25,
    )

    results = run_operator_ablation(args)

    assert output_json.exists()
    assert len(results["aggregated_strategies"]) == 4
    assert len(results["per_run_strategies"]) == 4
    assert [call.prototype_routing_strategy for call in benchmark_calls] == [
        "sinkhorn_topk",
        "masked_sinkhorn_topk",
        "sinkhorn_support_masked",
        "kl_topk",
    ]
    assert [call.prototype_masked_sinkhorn_candidate_k for call in benchmark_calls] == [
        5,
        7,
        0,
        5,
    ]
    assert [call.prototype_masked_sinkhorn_capacity_bias for call in benchmark_calls] == [
        0.25,
        0.75,
        0.0,
        0.25,
    ]
    assert {row["strategy"] for row in results["aggregated_strategies"]} == {
        "sinkhorn_topk",
        "kl_topk",
        "masked_sinkhorn_topk",
        "sinkhorn_support_masked",
    }
    assert "best_avg_accuracy" in results
    assert "lowest_avg_forgetting" in results

    persisted = json.loads(output_json.read_text(encoding="utf-8"))
    assert len(persisted["aggregated_strategies"]) == 4
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
    assert "sinkhorn_support_masked" in table_text
    assert "kl_topk" in table_text
    assert "masked_sinkhorn_topk" in table_text
    assert "Final Gap" in table_text
    assert "Final Transport" in table_text
    assert "# Continual Operator Ablation Summary" in report_text
    assert "masked_sinkhorn_topk" in report_text
    assert "no_transport" in report_text
