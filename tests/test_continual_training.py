"""
Smoke tests for the continual ASAM training scaffold.
"""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from experiments.train_continual_asam import ExperimentArgs, resolve_prototype_layout, run_experiment


def test_continual_training_scaffold_runs_and_reports_metrics(tmp_path):
    output_json = tmp_path / "continual_results.json"
    args = ExperimentArgs(
        num_tasks=3,
        train_samples=12,
        val_samples=8,
        seq_len=24,
        batch_size=4,
        replay_batch_size=2,
        replay_samples_per_task=4,
        epochs_per_task=1,
        dim=32,
        num_heads=2,
        num_layers=1,
        top_k_patterns=2,
        output_json=str(output_json),
    )

    results = run_experiment(args)

    assert output_json.exists()
    assert len(results["accuracy_matrix"]) == 3
    assert len(results["accuracy_matrix"][-1]) == 3
    assert 0.0 <= results["avg_accuracy"] <= 1.0
    assert "avg_forgetting" in results
    assert "backward_transfer" in results
    assert len(results["stage_training_metrics"]) == 3
    assert "transport_loss" in results["stage_training_metrics"][0]


def test_continual_training_scaffold_supports_prototype_routing(tmp_path):
    output_json = tmp_path / "prototype_results.json"
    args = ExperimentArgs(
        num_tasks=3,
        train_samples=12,
        val_samples=8,
        seq_len=24,
        batch_size=4,
        replay_batch_size=2,
        replay_samples_per_task=4,
        epochs_per_task=1,
        dim=32,
        num_heads=2,
        num_layers=1,
        top_k_patterns=2,
        routing_mode="prototype",
        num_prototypes=0,
        prototype_slots_per_task=2,
        prototype_top_k=2,
        balance_weight=0.05,
        diversity_weight=0.05,
        prototype_reset_threshold=0.01,
        prototype_split_threshold=0.2,
        output_json=str(output_json),
    )

    results = run_experiment(args)

    assert output_json.exists()
    assert results["routing_mode"] == "prototype"
    assert len(results["accuracy_matrix"]) == 3
    assert 0.0 <= results["avg_accuracy"] <= 1.0
    assert len(results["prototype_lifecycle"]) == 3
    assert len(results["prototype_diagnostics"]) == 3
    assert results["resolved_prototype_layout"]["num_prototypes"] == 6
    assert results["resolved_prototype_layout"]["prototype_top_k"] == 2
    assert "task_prototype_heatmap" in results["prototype_diagnostics"][-1]
    assert "task_routing_entropy" in results["prototype_diagnostics"][-1]
    assert "layer_capacity_ema" in results["prototype_diagnostics"][-1]
    assert "layer_latent_ema" in results["prototype_diagnostics"][-1]
    assert all(key in results["prototype_lifecycle"][-1] for key in ["reset_count", "split_count", "merge_count"])
    assert "transport_loss" in results["stage_training_metrics"][-1]
    assert "support_projection_residual" in results["stage_training_metrics"][-1]
    assert "effective_capacity_residual" in results["stage_training_metrics"][-1]
    assert "support_density" in results["stage_training_metrics"][-1]


def test_resolve_prototype_layout_enforces_true_sparse_support():
    auto_layout = resolve_prototype_layout(
        num_tasks=2,
        num_prototypes=0,
        prototype_slots_per_task=1,
        prototype_top_k=2,
    )
    explicit_layout = resolve_prototype_layout(
        num_tasks=2,
        num_prototypes=2,
        prototype_slots_per_task=1,
        prototype_top_k=2,
    )

    assert auto_layout["num_prototypes"] == 3
    assert auto_layout["prototype_top_k"] == 2
    assert explicit_layout["num_prototypes"] == 2
    assert explicit_layout["prototype_top_k"] == 1
