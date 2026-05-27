"""
Smoke tests for the real-text continual benchmark.
"""

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from experiments.run_continual_text_benchmark import (
    RealBenchmarkArgs,
    adapt_hyperparameters_from_diagnostics,
    compute_task_conditioned_transport_penalty,
    compute_theory_diagnostics,
    run_benchmark,
)
from experiments.train_continual_asam import ContinualTextClassifier


class _TinyContinualTaskDataset(Dataset):
    def __init__(self, base_dataset, task_id: int, max_length: int):
        self.base_dataset = base_dataset
        self.task_id = task_id
        self.max_length = max_length

    def __len__(self):
        return 4

    def __getitem__(self, idx):
        tokens = torch.full((self.max_length,), idx + self.task_id + 1, dtype=torch.long)
        label = torch.tensor(idx % 2, dtype=torch.long)
        task_id = torch.tensor(self.task_id, dtype=torch.long)
        return tokens, label, task_id


class _TinyBaseDataset:
    def __init__(self, provenance):
        self.dataset_provenance = provenance


class _DummyPrototypeModel:
    def __init__(self):
        self.state = {
            "prototype_prior_strength": 1.0,
            "prototype_capacity_blend": 0.5,
            "prototype_masked_sinkhorn_capacity_bias": 0.0,
            "prototype_relocation_strength": 0.75,
            "prototype_merge_threshold": 0.9,
            "prototype_merge_usage_threshold": 0.1,
            "prototype_top_k": 2,
        }

    def get_prototype_hyperparameters(self):
        return dict(self.state)

    def set_prototype_hyperparameters(
        self,
        prototype_prior_strength=None,
        prototype_capacity_blend=None,
        prototype_masked_sinkhorn_capacity_bias=None,
        prototype_relocation_strength=None,
        prototype_merge_threshold=None,
        prototype_merge_usage_threshold=None,
        prototype_top_k=None,
    ):
        if prototype_prior_strength is not None:
            self.state["prototype_prior_strength"] = float(prototype_prior_strength)
        if prototype_capacity_blend is not None:
            self.state["prototype_capacity_blend"] = float(prototype_capacity_blend)
        if prototype_masked_sinkhorn_capacity_bias is not None:
            self.state["prototype_masked_sinkhorn_capacity_bias"] = float(
                prototype_masked_sinkhorn_capacity_bias
            )
        if prototype_relocation_strength is not None:
            self.state["prototype_relocation_strength"] = float(prototype_relocation_strength)
        if prototype_merge_threshold is not None:
            self.state["prototype_merge_threshold"] = float(prototype_merge_threshold)
        if prototype_merge_usage_threshold is not None:
            self.state["prototype_merge_usage_threshold"] = float(prototype_merge_usage_threshold)
        if prototype_top_k is not None:
            self.state["prototype_top_k"] = int(prototype_top_k)


def test_meta_secant_controller_avoids_premature_sparse_collapse():
    theory = {
        "stage_forgetting": [0.0],
        "stage_transport_gap": [0.12],
        "stage_transport_loss": [0.08],
        "stage_mean_abs_excess": [0.10],
        "stage_routing_stability_loss": [0.14],
        "stage_routing_entropy": [0.0],
        "stage_task_forgetting": [[0.0]],
        "forgetting_correlations": {
            "routing_stability": None,
            "transport_gap": None,
            "transport_loss": None,
            "max_transport_gap": None,
            "mean_abs_excess": None,
            "merge_count": None,
            "routing_entropy": None,
        },
    }
    model = _DummyPrototypeModel()
    model.num_prototypes = 4
    meta_args = RealBenchmarkArgs(adaptation_strategy="meta_secant", transport_weight=0.05)

    meta_record = adapt_hyperparameters_from_diagnostics(model, theory, meta_args, stage_index=0)

    assert meta_record["after"]["transport_weight"] == 0.05
    assert meta_record["after"]["prototype_top_k"] == 2
    assert meta_args.transport_weight == 0.05
    assert "transport_weight" in meta_record["hypergradients"]
    assert "prototype_top_k" in meta_record["hypergradients"]

    corr_model = _DummyPrototypeModel()
    corr_model.num_prototypes = 4
    corr_args = RealBenchmarkArgs(adaptation_strategy="correlation", transport_weight=0.05)
    corr_record = adapt_hyperparameters_from_diagnostics(
        corr_model, theory, corr_args, stage_index=0
    )

    assert corr_record["after"]["transport_weight"] == 0.05
    assert corr_record["after"]["prototype_top_k"] == 2
    assert corr_args.transport_weight == 0.05


def test_meta_secant_controller_increases_transport_weight_under_forgetting():
    theory = {
        "stage_forgetting": [0.0, 0.16, 0.22],
        "stage_transport_gap": [0.05, 0.16, 0.22],
        "stage_transport_loss": [0.02, 0.12, 0.14],
        "stage_mean_abs_excess": [0.03, 0.12, 0.16],
        "stage_routing_stability_loss": [0.05, 0.18, 0.24],
        "stage_routing_entropy": [0.0, 0.0, 0.0],
        "forgetting_correlations": {
            "routing_stability": 0.5,
            "transport_gap": 0.6,
            "transport_loss": 0.4,
            "max_transport_gap": 0.5,
            "mean_abs_excess": 0.3,
            "merge_count": None,
            "routing_entropy": None,
        },
    }
    model = _DummyPrototypeModel()
    model.num_prototypes = 4
    meta_args = RealBenchmarkArgs(adaptation_strategy="meta_secant", transport_weight=0.05)

    meta_record = adapt_hyperparameters_from_diagnostics(model, theory, meta_args, stage_index=2)

    assert meta_record["after"]["transport_weight"] > 0.05
    assert meta_record["after"]["prototype_top_k"] == 1
    assert meta_args.transport_weight == meta_record["after"]["transport_weight"]


def test_dual_transport_uses_exact_task_conditioned_transport_penalty():
    args = RealBenchmarkArgs(adaptation_strategy="dual_transport", transport_weight=0.05)
    args.task_transport_weights = [0.1, 0.3, 0.7]
    task_ids = torch.tensor([0, 1, 1, 2], dtype=torch.long)
    info = {
        "overlap_loss": torch.tensor(0.0),
        "transport_loss": torch.tensor(3.75),
        "transport_loss_per_sample": torch.tensor([1.0, 2.0, 4.0, 8.0]),
    }

    penalty, effective_weight = compute_task_conditioned_transport_penalty(args, info, task_ids)

    assert torch.allclose(penalty, torch.tensor(1.8750), atol=1e-6)
    assert abs(effective_weight - 0.35) < 1e-6


def test_masked_sinkhorn_candidate_k_reaches_classifier_config():
    args = RealBenchmarkArgs(
        routing_mode="prototype",
        prototype_routing_strategy="masked_sinkhorn_topk",
        prototype_masked_sinkhorn_candidate_k=6,
        prototype_masked_sinkhorn_capacity_bias=0.5,
    )

    model = ContinualTextClassifier(
        vocab_size=64,
        num_tasks=2,
        num_classes=2,
        dim=32,
        num_heads=2,
        num_layers=1,
        seq_len=16,
        top_k_patterns=2,
        routing_mode=args.routing_mode,
        prototype_routing_strategy=args.prototype_routing_strategy,
        prototype_masked_sinkhorn_candidate_k=args.prototype_masked_sinkhorn_candidate_k,
        prototype_masked_sinkhorn_capacity_bias=args.prototype_masked_sinkhorn_capacity_bias,
    )

    layer = model.layers[0]
    assert layer.continual_config.prototype_masked_sinkhorn_candidate_k == 6
    assert layer.continual_config.prototype_masked_sinkhorn_capacity_bias == 0.5
    assert layer.prototype_gate.masked_sinkhorn_candidate_k == 6
    assert layer.prototype_gate.masked_sinkhorn_capacity_bias == 0.5


def test_theory_diagnostics_track_task_transport_statistics():
    theory = compute_theory_diagnostics(
        accuracy_matrix=[[0.8, 0.0], [0.6, 0.7]],
        stage_training_metrics=[
            {
                "routing_stability_loss": 0.10,
                "stability_loss": 0.12,
                "overlap_loss": 0.20,
                "transport_loss": 0.30,
                "weighted_transport_loss": 0.03,
            },
            {
                "routing_stability_loss": 0.14,
                "stability_loss": 0.16,
                "overlap_loss": 0.18,
                "transport_loss": 0.22,
                "weighted_transport_loss": 0.05,
            },
        ],
        prototype_lifecycle=[
            {"mean_transport_gap": 0.08, "max_transport_gap": 0.16, "merge_count": 0},
            {"mean_transport_gap": 0.12, "max_transport_gap": 0.20, "merge_count": 1},
        ],
        prototype_diagnostics=[
            {
                "layer_excess_ema": [[0.1, -0.1]],
                "task_routing_entropy": [0.2],
                "task_transport_gap": [0.09],
                "task_max_transport_gap": [0.18],
                "task_transport_loss": [0.31],
            },
            {
                "layer_excess_ema": [[0.2, -0.1]],
                "task_routing_entropy": [0.3, 0.4],
                "task_transport_gap": [0.15, 0.04],
                "task_max_transport_gap": [0.24, 0.08],
                "task_transport_loss": [0.28, 0.07],
            },
        ],
    )

    assert theory["stage_task_transport_gap"] == [[0.09], [0.15, 0.04]]
    assert theory["stage_task_max_transport_gap"] == [[0.18], [0.24, 0.08]]
    assert theory["stage_task_transport_loss"] == [[0.31], [0.28, 0.07]]
    assert theory["stage_weighted_transport_loss"] == [0.03, 0.05]
    assert theory["stage_support_projection_residual"] == [0.0, 0.0]
    assert theory["stage_effective_capacity_residual"] == [0.0, 0.0]
    assert theory["stage_support_density"] == [0.0, 0.0]
    assert "support_projection_residual" in theory["forgetting_correlations"]
    assert "effective_capacity_residual" in theory["forgetting_correlations"]
    assert "support_density" in theory["forgetting_correlations"]
    assert "support_residual_delta" in theory["forgetting_correlations"]


def test_dual_transport_controller_uses_task_level_gap_and_loss_signals():
    theory = {
        "stage_forgetting": [0.0, 0.18],
        "stage_task_forgetting": [
            [0.0],
            [0.18, 0.18],
        ],
        "stage_transport_gap": [0.02, 0.10],
        "stage_transport_loss": [0.01, 0.09],
        "stage_task_transport_gap": [
            [0.02],
            [0.25, 0.02],
        ],
        "stage_task_transport_loss": [
            [0.01],
            [0.30, 0.03],
        ],
        "stage_mean_abs_excess": [0.02, 0.08],
        "stage_routing_stability_loss": [0.03, 0.12],
        "stage_routing_entropy": [0.0, 0.0],
        "forgetting_correlations": {
            "routing_stability": 0.4,
            "transport_gap": 0.6,
            "transport_loss": 0.4,
            "weighted_transport_loss": 0.4,
            "max_transport_gap": 0.5,
            "mean_abs_excess": 0.2,
            "merge_count": None,
            "routing_entropy": None,
        },
    }
    model = _DummyPrototypeModel()
    model.num_prototypes = 4
    args = RealBenchmarkArgs(adaptation_strategy="dual_transport", transport_weight=0.05)

    record = adapt_hyperparameters_from_diagnostics(model, theory, args, stage_index=1)

    weights = record["after"]["task_transport_weights"]
    assert len(weights) == 2
    assert weights[0] > weights[1] > 0.05
    assert (
        record["signals"]["task_transport_signals"][0]
        > record["signals"]["task_transport_signals"][1]
        > 0.0
    )
    assert record["signals"]["task_gap_signals"] == [0.25, 0.02]
    assert record["signals"]["task_loss_signals"] == [0.3, 0.03]


def test_dual_transport_relaxes_weights_without_forgetting_signal():
    theory = {
        "stage_forgetting": [0.0, 0.0],
        "stage_task_forgetting": [
            [0.0],
            [0.0, 0.0],
        ],
        "stage_transport_gap": [0.02, 0.02],
        "stage_transport_loss": [0.01, 0.01],
        "stage_task_transport_gap": [
            [0.02],
            [0.03, 0.02],
        ],
        "stage_task_transport_loss": [
            [0.01],
            [0.03, 0.02],
        ],
        "stage_mean_abs_excess": [0.01, 0.01],
        "stage_routing_stability_loss": [0.02, 0.02],
        "stage_routing_entropy": [0.0, 0.0],
        "forgetting_correlations": {
            "routing_stability": None,
            "transport_gap": None,
            "transport_loss": None,
            "weighted_transport_loss": None,
            "max_transport_gap": None,
            "mean_abs_excess": None,
            "merge_count": None,
            "routing_entropy": None,
        },
    }
    model = _DummyPrototypeModel()
    model.num_prototypes = 4
    model.num_tasks = 2
    args = RealBenchmarkArgs(adaptation_strategy="dual_transport", transport_weight=0.05)
    args.task_transport_weights = [0.08, 0.05]

    record = adapt_hyperparameters_from_diagnostics(model, theory, args, stage_index=1)

    assert record["signals"]["task_transport_signals"] == [0.0, 0.0]
    assert record["after"]["task_transport_weights"][0] < 0.08
    assert record["after"]["task_transport_weights"][0] > 0.05
    assert abs(record["after"]["task_transport_weights"][1] - 0.05) < 1e-6


def test_dual_transport_waits_for_observed_forgetting():
    theory = {
        "stage_forgetting": [0.0],
        "stage_transport_gap": [0.12],
        "stage_transport_loss": [0.08],
        "stage_mean_abs_excess": [0.10],
        "stage_routing_stability_loss": [0.14],
        "stage_routing_entropy": [0.0],
        "stage_task_forgetting": [[0.0]],
        "forgetting_correlations": {
            "routing_stability": None,
            "transport_gap": None,
            "transport_loss": None,
            "max_transport_gap": None,
            "mean_abs_excess": None,
            "merge_count": None,
            "routing_entropy": None,
        },
    }
    model = _DummyPrototypeModel()
    model.num_prototypes = 4
    args = RealBenchmarkArgs(adaptation_strategy="dual_transport", transport_weight=0.05)

    record = adapt_hyperparameters_from_diagnostics(model, theory, args, stage_index=0)

    assert record["after"]["transport_weight"] == 0.05
    assert record["after"]["prototype_top_k"] == 2
    assert record["after"]["task_transport_weights"] == [0.05]
    assert record["signals"]["controller_transport_signal"] == 0.0
    assert record["signals"]["controller_topk_signal"] == 0.0


def test_dual_transport_increases_regularization_under_sustained_forgetting():
    theory = {
        "stage_forgetting": [0.0, 0.20, 0.24, 0.28],
        "stage_task_forgetting": [
            [0.0],
            [0.20, 0.0],
            [0.24, 0.10, 0.0],
            [0.28, 0.14, 0.08, 0.0],
        ],
        "stage_transport_gap": [0.03, 0.16, 0.20, 0.24],
        "stage_transport_loss": [0.02, 0.10, 0.12, 0.16],
        "stage_mean_abs_excess": [0.02, 0.08, 0.10, 0.12],
        "stage_routing_stability_loss": [0.03, 0.12, 0.15, 0.18],
        "stage_routing_entropy": [0.0, 0.0, 0.0, 0.0],
        "forgetting_correlations": {
            "routing_stability": 0.4,
            "transport_gap": 0.6,
            "transport_loss": 0.4,
            "max_transport_gap": 0.5,
            "mean_abs_excess": 0.2,
            "merge_count": None,
            "routing_entropy": None,
        },
    }
    model = _DummyPrototypeModel()
    model.num_prototypes = 4
    args = RealBenchmarkArgs(adaptation_strategy="dual_transport", transport_weight=0.05)

    record = adapt_hyperparameters_from_diagnostics(model, theory, args, stage_index=3)

    assert record["after"]["transport_weight"] > 0.05
    assert record["after"]["prototype_top_k"] == 2
    assert len(record["after"]["task_transport_weights"]) == 4
    assert record["after"]["task_transport_weights"][0] > 0.05
    assert record["after"]["task_transport_weights"][1] > 0.05
    assert record["signals"]["controller_transport_signal"] > 0.0
    assert len(record["signals"]["task_transport_signals"]) == 4
    assert record["signals"]["task_transport_signals"][-1] == 0.0


def test_real_continual_benchmark_runs_with_split_ag_news(tmp_path, monkeypatch):
    output_json = tmp_path / "real_benchmark.json"
    train_provenance = {
        "source_kind": "fallback_synthetic",
        "split": "train",
        "sample_count": 16,
        "max_samples": 16,
        "reason": "UnitTestFixture",
    }
    val_provenance = {
        "source_kind": "fallback_synthetic",
        "split": "test",
        "sample_count": 8,
        "max_samples": 8,
        "reason": "UnitTestFixture",
    }

    def fake_get_continual_dataloaders(**kwargs):
        max_length = kwargs["max_length"]
        train_base = _TinyBaseDataset(train_provenance)
        val_base = _TinyBaseDataset(val_provenance)
        train_loaders = [
            DataLoader(_TinyContinualTaskDataset(train_base, task_id, max_length), batch_size=4)
            for task_id in range(2)
        ]
        val_loaders = [
            DataLoader(_TinyContinualTaskDataset(val_base, task_id, max_length), batch_size=4)
            for task_id in range(2)
        ]
        return train_loaders, val_loaders

    monkeypatch.setattr(
        "experiments.run_continual_text_benchmark.get_continual_dataloaders",
        fake_get_continual_dataloaders,
    )
    args = RealBenchmarkArgs(
        max_length=64,
        batch_size=4,
        max_train_samples=16,
        max_val_samples=8,
        dim=32,
        num_heads=2,
        num_layers=1,
        epochs_per_task=1,
        output_json=str(output_json),
        routing_mode="prototype",
        num_prototypes=0,
        prototype_slots_per_task=2,
        prototype_top_k=2,
    )

    results = run_benchmark(args)

    assert output_json.exists()
    assert results["num_tasks"] == 2
    assert results["dataset_provenance"]["train"] == train_provenance
    assert results["dataset_provenance"]["val"] == val_provenance
    assert len(results["accuracy_matrix"]) == 2
    assert 0.0 <= results["avg_accuracy"] <= 1.0
    assert len(results["prototype_diagnostics"]) == results["num_tasks"]
    assert results["resolved_prototype_layout"]["num_prototypes"] == 4
    assert results["resolved_prototype_layout"]["prototype_top_k"] == 2

    plot_path = Path(results["plot_path"])
    report_path = Path(results["report_path"])
    assert plot_path.exists()
    assert report_path.exists()

    assert "theory_diagnostics" in results
    assert len(results["theory_diagnostics"]["stage_forgetting"]) == results["num_tasks"]
    assert "routing_stability" in results["theory_diagnostics"]["forgetting_correlations"]
    assert "hyperparameter_history" in results
    assert "final_hyperparameters" in results
    assert "transport_loss" in results["theory_diagnostics"]["forgetting_correlations"]
    assert "prototype_merge_threshold" in results["final_hyperparameters"]
    assert "transport_weight" in results["final_hyperparameters"]
    assert "prototype_top_k" in results["final_hyperparameters"]
    assert results["config"]["adaptation_strategy"] == "meta_secant"
    assert len(results["hyperparameter_history"]) >= 1
    assert "meta_objective" in results["hyperparameter_history"][0]
    assert "hypergradients" in results["hyperparameter_history"][0]
    assert "transport_weight" in results["hyperparameter_history"][0]["before"]

    persisted = json.loads(output_json.read_text(encoding="utf-8"))
    assert persisted["plot_path"] == str(plot_path)
    assert persisted["report_path"] == str(report_path)
    assert len(persisted["prototype_diagnostics"]) == persisted["num_tasks"]
    assert "theory_diagnostics" in persisted
    assert "hyperparameter_history" in persisted
    assert "final_hyperparameters" in persisted

    report_text = report_path.read_text(encoding="utf-8")
    assert "## Theory Diagnostics" in report_text
    assert "routing stability correlation" in report_text
    assert "transport loss correlation" in report_text
    assert "## Hyperparameter Adaptation" in report_text
    assert "Adaptation strategy" in report_text


def test_benchmark_artifacts_record_dataset_provenance(tmp_path, monkeypatch):
    output_json = tmp_path / "real_benchmark.json"
    train_provenance = {
        "source_kind": "fallback_synthetic",
        "split": "train",
        "sample_count": 8,
        "max_samples": 8,
        "reason": "ImportError",
    }
    val_provenance = {
        "source_kind": "huggingface",
        "dataset_name": "ag_news",
        "dataset_config": None,
        "split": "test",
        "sample_count": 4,
        "max_samples": 4,
    }

    def fake_get_continual_dataloaders(**kwargs):
        max_length = kwargs["max_length"]
        train_base = _TinyBaseDataset(train_provenance)
        val_base = _TinyBaseDataset(val_provenance)
        train_loaders = [
            DataLoader(_TinyContinualTaskDataset(train_base, task_id, max_length), batch_size=2)
            for task_id in range(2)
        ]
        val_loaders = [
            DataLoader(_TinyContinualTaskDataset(val_base, task_id, max_length), batch_size=2)
            for task_id in range(2)
        ]
        return train_loaders, val_loaders

    monkeypatch.setattr(
        "experiments.run_continual_text_benchmark.get_continual_dataloaders",
        fake_get_continual_dataloaders,
    )
    args = RealBenchmarkArgs(
        max_length=16,
        batch_size=2,
        max_train_samples=8,
        max_val_samples=4,
        dim=16,
        num_heads=2,
        num_layers=1,
        epochs_per_task=1,
        output_json=str(output_json),
        routing_mode="prototype",
        num_prototypes=0,
        prototype_slots_per_task=2,
        prototype_top_k=2,
        adaptive_hyperparameters=False,
    )

    results = run_benchmark(args)

    assert results["dataset_provenance"]["train"] == train_provenance
    assert results["dataset_provenance"]["val"] == val_provenance
    persisted = json.loads(output_json.read_text(encoding="utf-8"))
    assert persisted["dataset_provenance"] == results["dataset_provenance"]
    report_text = Path(results["report_path"]).read_text(encoding="utf-8")
    assert "Dataset source (train): `fallback_synthetic`" in report_text
    assert "Dataset source (val): `huggingface`" in report_text


def test_real_continual_benchmark_runs_with_kl_topk_routing(tmp_path, monkeypatch):
    output_json = tmp_path / "real_benchmark_kl.json"
    train_provenance = {
        "source_kind": "fallback_synthetic",
        "split": "train",
        "sample_count": 16,
        "max_samples": 16,
        "reason": "UnitTestFixture",
    }
    val_provenance = {
        "source_kind": "fallback_synthetic",
        "split": "test",
        "sample_count": 8,
        "max_samples": 8,
        "reason": "UnitTestFixture",
    }

    def fake_get_continual_dataloaders(**kwargs):
        max_length = kwargs["max_length"]
        train_base = _TinyBaseDataset(train_provenance)
        val_base = _TinyBaseDataset(val_provenance)
        train_loaders = [
            DataLoader(_TinyContinualTaskDataset(train_base, task_id, max_length), batch_size=4)
            for task_id in range(2)
        ]
        val_loaders = [
            DataLoader(_TinyContinualTaskDataset(val_base, task_id, max_length), batch_size=4)
            for task_id in range(2)
        ]
        return train_loaders, val_loaders

    monkeypatch.setattr(
        "experiments.run_continual_text_benchmark.get_continual_dataloaders",
        fake_get_continual_dataloaders,
    )
    args = RealBenchmarkArgs(
        max_length=64,
        batch_size=4,
        max_train_samples=16,
        max_val_samples=8,
        dim=32,
        num_heads=2,
        num_layers=1,
        epochs_per_task=1,
        output_json=str(output_json),
        routing_mode="prototype",
        prototype_routing_strategy="kl_topk",
        adaptive_hyperparameters=False,
    )

    results = run_benchmark(args)

    assert output_json.exists()
    assert results["config"]["prototype_routing_strategy"] == "kl_topk"
    assert results["num_tasks"] == 2
    assert results["dataset_provenance"]["train"] == train_provenance
    assert results["dataset_provenance"]["val"] == val_provenance
    assert len(results["prototype_diagnostics"]) == results["num_tasks"]
    assert 0.0 <= results["avg_accuracy"] <= 1.0
    assert Path(results["plot_path"]).exists()
    assert Path(results["report_path"]).exists()
