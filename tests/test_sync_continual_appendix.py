"""
Tests for syncing the continual appendix into the paper.
"""

import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from scripts.sync_continual_appendix import build_continual_appendix, sync_paper_appendix


def test_paper_continual_objective_keeps_regularizer_plus_signs():
    paper_text = (repo_root / "paper" / "asam_paper.tex").read_text(encoding="utf-8")

    assert "\\mathcal{L}=\\mathcal{L}_{\\mathrm{task}}\n+" in paper_text
    assert "\\lambda_s\\mathcal{L}_{\\mathrm{stability}}\n+" in paper_text
    assert "\\lambda_o\\mathcal{L}_{\\mathrm{overlap}}\n+" in paper_text
    assert "\\lambda_b\\mathcal{L}_{\\mathrm{balance}}\n+" in paper_text
    assert "\\lambda_d\\mathcal{L}_{\\mathrm{diversity}}\n+" in paper_text


def test_paper_avoids_overstated_claim_phrases():
    paper_text = (repo_root / "paper" / "asam_paper.tex").read_text(encoding="utf-8")

    assert "official benchmark results" not in paper_text
    assert "does not yet outperform" not in paper_text
    assert "1 seeds" not in paper_text


def test_build_continual_appendix_renders_expected_tables(tmp_path):
    benchmark = {
        "config": {
            "dataset_name": "split_ag_news",
            "classes_per_task": 2,
            "max_length": 128,
            "batch_size": 8,
            "max_train_samples": 256,
            "max_val_samples": 128,
            "dim": 64,
            "num_heads": 4,
            "num_layers": 1,
            "top_k_patterns": 2,
            "learning_rate": 3e-4,
            "epochs_per_task": 1,
            "adaptation_strategy": "meta_secant",
        },
        "num_tasks": 2,
        "avg_accuracy": 0.5390625,
        "avg_forgetting": -0.0625,
        "backward_transfer": 0.0625,
    }
    ablation = {
        "config": {"num_seeds": 3},
        "aggregated_strategies": [
            {
                "strategy": "task_routing",
                "avg_accuracy_mean": 0.5234375,
                "avg_accuracy_std": 0.0446521567,
                "avg_forgetting_mean": -0.046875,
                "avg_forgetting_std": 0.0776024188,
                "backward_transfer_mean": 0.046875,
                "backward_transfer_std": 0.0776024188,
                "final_transport_gap_mean": 0.0,
                "final_transport_gap_std": 0.0,
            },
            {
                "strategy": "meta_secant",
                "avg_accuracy_mean": 0.4973958333,
                "avg_accuracy_std": 0.0294627825,
                "avg_forgetting_mean": 0.046875,
                "avg_forgetting_std": 0.0836582208,
                "backward_transfer_mean": -0.046875,
                "backward_transfer_std": 0.0836582208,
                "final_transport_gap_mean": 0.0,
                "final_transport_gap_std": 0.0,
            },
        ],
    }
    operator = {
        "aggregated_strategies": [
            {
                "strategy": "sinkhorn_topk",
                "routing_strategy": "sinkhorn_topk",
                "avg_accuracy_mean": 0.4973958333,
                "avg_accuracy_std": 0.0294627825,
                "avg_forgetting_mean": 0.046875,
                "avg_forgetting_std": 0.0836582208,
                "final_transport_gap_mean": 0.0,
                "final_transport_gap_std": 0.0,
                "final_transport_loss_mean": 0.0380610903,
                "final_transport_loss_std": 0.0075927120,
            },
            {
                "strategy": "kl_topk",
                "routing_strategy": "kl_topk",
                "avg_accuracy_mean": 0.4973958333,
                "avg_accuracy_std": 0.0294627825,
                "avg_forgetting_mean": 0.046875,
                "avg_forgetting_std": 0.0836582208,
                "final_transport_gap_mean": 0.0007743835,
                "final_transport_gap_std": 0.0003043876,
                "final_transport_loss_mean": 0.0369797295,
                "final_transport_loss_std": 0.0070665292,
            },
            {
                "strategy": "no_transport",
                "routing_strategy": "sinkhorn_topk",
                "avg_accuracy_mean": 0.4947916666,
                "avg_accuracy_std": 0.0257799347,
                "avg_forgetting_mean": 0.046875,
                "avg_forgetting_std": 0.0836582208,
                "final_transport_gap_mean": 0.0,
                "final_transport_gap_std": 0.0,
                "final_transport_loss_mean": 0.0418019112,
                "final_transport_loss_std": 0.0098388577,
            },
        ]
    }

    appendix = build_continual_appendix(benchmark, ablation, operator)
    lines = appendix.splitlines()

    assert "\\section{Continual ASAM Pilot Study}" in appendix
    assert "Split AG News" in appendix
    assert "meta-secant controller" in appendix
    assert "\\texttt{task\\_routing}" in appendix
    assert "Sinkhorn Top-$k$" in appendix
    assert "0.5234 \\pm 0.0447" in appendix
    assert any("Final Gap" in line and line.endswith("\\\\") for line in lines)
    assert any("task\\_routing" in line and line.endswith("\\\\") for line in lines)


def test_build_continual_appendix_uses_singular_seed_language_for_one_seed(tmp_path):
    benchmark = {
        "config": {
            "dataset_name": "split_ag_news",
            "classes_per_task": 2,
            "max_length": 64,
            "batch_size": 4,
            "max_train_samples": 16,
            "max_val_samples": 8,
            "dim": 32,
            "num_heads": 2,
            "num_layers": 1,
            "top_k_patterns": 2,
            "learning_rate": 3e-4,
            "epochs_per_task": 1,
            "adaptation_strategy": "meta_secant",
        },
        "num_tasks": 2,
        "avg_accuracy": 0.5,
        "avg_forgetting": 0.25,
        "backward_transfer": -0.25,
    }
    ablation = {
        "config": {"num_seeds": 1},
        "aggregated_strategies": [
            {
                "strategy": "task_routing",
                "avg_accuracy_mean": 0.625,
                "avg_accuracy_std": 0.0,
                "avg_forgetting_mean": -0.25,
                "avg_forgetting_std": 0.0,
                "backward_transfer_mean": 0.25,
                "backward_transfer_std": 0.0,
                "final_transport_gap_mean": 0.0,
                "final_transport_gap_std": 0.0,
            },
            {
                "strategy": "meta_secant",
                "avg_accuracy_mean": 0.5,
                "avg_accuracy_std": 0.0,
                "avg_forgetting_mean": 0.25,
                "avg_forgetting_std": 0.0,
                "backward_transfer_mean": -0.25,
                "backward_transfer_std": 0.0,
                "final_transport_gap_mean": 0.024,
                "final_transport_gap_std": 0.0,
            },
        ],
    }
    operator = {
        "aggregated_strategies": [
            {
                "strategy": "sinkhorn_topk",
                "routing_strategy": "sinkhorn_topk",
                "avg_accuracy_mean": 0.5,
                "avg_accuracy_std": 0.0,
                "avg_forgetting_mean": 0.25,
                "avg_forgetting_std": 0.0,
                "final_transport_gap_mean": 0.024,
                "final_transport_gap_std": 0.0,
                "final_transport_loss_mean": 0.074,
                "final_transport_loss_std": 0.0,
            },
            {
                "strategy": "no_transport",
                "routing_strategy": "sinkhorn_topk",
                "avg_accuracy_mean": 0.5,
                "avg_accuracy_std": 0.0,
                "avg_forgetting_mean": 0.25,
                "avg_forgetting_std": 0.0,
                "final_transport_gap_mean": 0.024,
                "final_transport_gap_std": 0.0,
                "final_transport_loss_mean": 0.070,
                "final_transport_loss_std": 0.0,
            }
        ]
    }

    appendix = build_continual_appendix(benchmark, ablation, operator)

    assert "aggregated over 1 seed." in appendix
    assert "single-seed routing/controller comparison" in appendix
    assert "(1 seed)" in appendix
    assert "1 seeds" not in appendix
    assert "multi-seed routing/controller comparison" not in appendix
    assert "does not yet improve over task-ID routing" in appendix
    assert "does not yet outperform" not in appendix
    assert "operator variants have identical accuracy" in appendix
    assert "meaningful pattern" not in appendix
    assert "slightly lowers accuracy" not in appendix



def test_sync_paper_appendix_replaces_marker_block(tmp_path):
    benchmark_path = tmp_path / "benchmark.json"
    ablation_path = tmp_path / "ablation.json"
    operator_path = tmp_path / "operator.json"
    paper_path = tmp_path / "paper.tex"

    benchmark = {
        "config": {
            "dataset_name": "split_ag_news",
            "classes_per_task": 2,
            "max_length": 128,
            "batch_size": 8,
            "max_train_samples": 256,
            "max_val_samples": 128,
            "dim": 64,
            "num_heads": 4,
            "num_layers": 1,
            "top_k_patterns": 2,
            "learning_rate": 3e-4,
            "epochs_per_task": 1,
            "adaptation_strategy": "meta_secant",
        },
        "num_tasks": 2,
        "avg_accuracy": 0.5,
        "avg_forgetting": 0.1,
        "backward_transfer": -0.1,
    }
    ablation = {
        "config": {"num_seeds": 2},
        "aggregated_strategies": [
            {
                "strategy": "task_routing",
                "avg_accuracy_mean": 0.6,
                "avg_accuracy_std": 0.01,
                "avg_forgetting_mean": 0.02,
                "avg_forgetting_std": 0.01,
                "backward_transfer_mean": -0.02,
                "backward_transfer_std": 0.01,
                "final_transport_gap_mean": 0.0,
                "final_transport_gap_std": 0.0,
            }
        ],
    }
    operator = {
        "aggregated_strategies": [
            {
                "strategy": "sinkhorn_topk",
                "routing_strategy": "sinkhorn_topk",
                "avg_accuracy_mean": 0.55,
                "avg_accuracy_std": 0.02,
                "avg_forgetting_mean": 0.03,
                "avg_forgetting_std": 0.01,
                "final_transport_gap_mean": 0.0,
                "final_transport_gap_std": 0.0,
                "final_transport_loss_mean": 0.04,
                "final_transport_loss_std": 0.01,
            }
        ]
    }

    benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")
    ablation_path.write_text(json.dumps(ablation), encoding="utf-8")
    operator_path.write_text(json.dumps(operator), encoding="utf-8")
    paper_path.write_text(
        "prefix\n% BEGIN AUTO-GENERATED CONTINUAL APPENDIX\nold body\n% END AUTO-GENERATED CONTINUAL APPENDIX\nsuffix\n",
        encoding="utf-8",
    )

    sync_paper_appendix(benchmark_path, ablation_path, operator_path, paper_path)

    updated = paper_path.read_text(encoding="utf-8")
    assert "old body" not in updated
    assert updated.startswith("prefix\n% BEGIN AUTO-GENERATED CONTINUAL APPENDIX")
    assert updated.rstrip().endswith("suffix")
    assert "\\section{Continual ASAM Pilot Study}" in updated
    assert "0.6000 \\pm 0.0100" in updated
