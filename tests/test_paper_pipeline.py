"""
Tests for the paper-ready continual suite pipeline.
"""

import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import scripts.run_continual_paper_suite as paper_suite
from scripts.run_continual_paper_suite import (
    PipelineArgs,
    build_pipeline_report,
    resolve_candidate_profile,
)


def test_pipeline_report_mentions_key_artifacts():
    args = PipelineArgs(output_dir="experiments/paper_suite_test", num_seeds=2)
    benchmark_results = {
        "avg_accuracy": 0.71,
        "avg_forgetting": 0.08,
        "backward_transfer": 0.03,
        "dataset_provenance": {
            "train": {"source_kind": "fallback_synthetic"},
            "val": {"source_kind": "huggingface"},
        },
    }
    ablation_results = {
        "best_avg_accuracy": {"strategy": "meta_secant", "value": 0.74},
        "lowest_avg_forgetting": {"strategy": "meta_secant", "value": 0.06},
    }
    operator_ablation_results = {
        "best_avg_accuracy": {"strategy": "sinkhorn_topk", "value": 0.73},
        "lowest_avg_forgetting": {"strategy": "no_transport", "value": 0.07},
    }
    manifest = {
        "output_dir": args.output_dir,
        "candidate_profile": "accuracy",
        "candidate_profile_description": "Accuracy-oriented prototype routing preset from the capacity sweep.",
        "resolved_config": {
            "num_prototypes": 0,
            "prototype_slots_per_task": 2,
            "prototype_top_k": 2,
            "transport_weight": 0.05,
        },
        "benchmark_json": "experiments/paper_suite_test/continual_benchmark.json",
        "benchmark_report": "experiments/paper_suite_test/continual_benchmark_report.md",
        "ablation_json": "experiments/paper_suite_test/continual_ablation.json",
        "ablation_table": "experiments/paper_suite_test/continual_ablation_table.md",
        "ablation_csv": "experiments/paper_suite_test/continual_ablation.csv",
        "ablation_plot": "experiments/paper_suite_test/continual_ablation.png",
        "ablation_report": "experiments/paper_suite_test/continual_ablation_report.md",
        "operator_ablation_json": "experiments/paper_suite_test/continual_operator_ablation.json",
        "operator_ablation_table": "experiments/paper_suite_test/continual_operator_ablation_table.md",
        "operator_ablation_csv": "experiments/paper_suite_test/continual_operator_ablation.csv",
        "operator_ablation_plot": "experiments/paper_suite_test/continual_operator_ablation.png",
        "operator_ablation_report": "experiments/paper_suite_test/continual_operator_ablation_report.md",
        "paper_tex": "paper/asam_paper.tex",
        "synced_paper_tex": "paper/asam_paper_synced.tex",
        "appendix_only_tex": "paper/continual_appendix_only.tex",
    }

    report = build_pipeline_report(
        args, benchmark_results, ablation_results, operator_ablation_results, manifest
    )

    assert "# Continual ASAM Paper Suite" in report
    assert "Meta-secant avg accuracy" in report
    assert "Ablation CSV" in report
    assert "Operator Ablation CSV" in report
    assert "Operator Ablation report" in report
    assert "Seeds for ablation" in report
    assert "Candidate profile" in report
    assert "Prototype layout" in report
    assert "Transport weight" in report
    assert "Dataset source (train): `fallback_synthetic`" in report
    assert "Dataset source (val): `huggingface`" in report
    assert "Profile note" in report
    assert "## Paper Sync" in report
    assert "Synced paper TeX" in report
    assert "Standalone appendix TeX" in report


def test_resolve_candidate_profile_overrides_layout():
    raw_args = PipelineArgs(
        candidate_profile="retention_no_transport",
        num_prototypes=13,
        prototype_slots_per_task=7,
        prototype_top_k=5,
        transport_weight=0.25,
    )

    resolved_args, profile = resolve_candidate_profile(raw_args)

    assert raw_args.num_prototypes == 13
    assert raw_args.transport_weight == 0.25
    assert resolved_args.candidate_profile == "retention_no_transport"
    assert resolved_args.num_prototypes == 0
    assert resolved_args.prototype_slots_per_task == 2
    assert resolved_args.prototype_top_k == 1
    assert resolved_args.transport_weight == 0.0
    assert "transport loss disabled" in profile["description"]


def test_manifest_provenance_redacts_sensitive_argv(monkeypatch, tmp_path):
    monkeypatch.setattr(
        paper_suite.sys,
        "argv",
        [
            "scripts/run_continual_paper_suite.py",
            "--hf-token",
            "hf_secret_value",
            "--github-token=ghp_secret_value",
            "--output-dir",
            str(tmp_path),
        ],
    )

    provenance = paper_suite.build_manifest_provenance(
        PipelineArgs(output_dir=str(tmp_path)),
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:01:00Z",
        [],
    )

    joined_argv = " ".join(provenance["argv"])
    assert "hf_secret_value" not in joined_argv
    assert "ghp_secret_value" not in joined_argv
    assert "<redacted>" in joined_argv
    assert str(tmp_path) in joined_argv


def test_run_pipeline_uses_resolved_candidate_profile(tmp_path, monkeypatch):
    output_dir = tmp_path / "paper_suite_profile"
    captured = {}
    call_order = []

    def _write(path: Path, content: str) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return str(path)

    def fake_run_benchmark(args):
        call_order.append("benchmark")
        captured["benchmark"] = args
        payload = {
            "avg_accuracy": 0.55,
            "avg_forgetting": -0.04,
            "backward_transfer": 0.02,
            "dataset_provenance": {
                "train": {
                    "source_kind": "fallback_synthetic",
                    "split": "train",
                    "sample_count": 64,
                    "max_samples": 64,
                    "reason": "ImportError",
                },
                "val": {
                    "source_kind": "fallback_synthetic",
                    "split": "test",
                    "sample_count": 32,
                    "max_samples": 32,
                    "reason": "ImportError",
                },
            },
            "plot_path": _write(output_dir / "continual_benchmark.png", "plot"),
            "report_path": _write(output_dir / "continual_benchmark_report.md", "report"),
        }
        Path(args.output_json).write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def fake_run_ablation(args):
        call_order.append("ablation")
        captured["ablation"] = args
        _write(output_dir / "continual_ablation_task_routing_seed42.json", "{}")
        payload = {
            "best_avg_accuracy": {"strategy": "prototype", "value": 0.56},
            "lowest_avg_forgetting": {"strategy": "prototype", "value": -0.05},
            "table_path": _write(output_dir / "continual_ablation_table.md", "table"),
            "csv_path": _write(output_dir / "continual_ablation.csv", "csv"),
            "plot_path": _write(output_dir / "continual_ablation.png", "plot"),
            "report_path": _write(output_dir / "continual_ablation_report.md", "report"),
        }
        Path(args.output_json).write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def fake_run_operator_ablation(args):
        call_order.append("operator_ablation")
        captured["operator_ablation"] = args
        payload = {
            "best_avg_accuracy": {"strategy": "sinkhorn_topk", "value": 0.54},
            "lowest_avg_forgetting": {"strategy": "sinkhorn_topk", "value": -0.03},
            "table_path": _write(output_dir / "continual_operator_ablation_table.md", "table"),
            "csv_path": _write(output_dir / "continual_operator_ablation.csv", "csv"),
            "plot_path": _write(output_dir / "continual_operator_ablation.png", "plot"),
            "report_path": _write(output_dir / "continual_operator_ablation_report.md", "report"),
        }
        Path(args.output_json).write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr(paper_suite, "run_benchmark", fake_run_benchmark)
    monkeypatch.setattr(paper_suite, "run_ablation", fake_run_ablation)
    monkeypatch.setattr(paper_suite, "run_operator_ablation", fake_run_operator_ablation)

    def fake_collect_git_provenance():
        call_order.append("git")
        return {"commit": "abc123", "dirty": False, "status_porcelain": ""}

    monkeypatch.setattr(paper_suite, "collect_git_provenance", fake_collect_git_provenance)

    args = PipelineArgs(
        output_dir=str(output_dir),
        candidate_profile="retention_no_transport",
        num_prototypes=9,
        prototype_slots_per_task=6,
        prototype_top_k=4,
        transport_weight=0.2,
    )
    results = paper_suite.run_pipeline(args)

    assert call_order[:2] == ["git", "benchmark"]

    for stage_name in ("benchmark", "ablation", "operator_ablation"):
        stage_args = captured[stage_name]
        assert stage_args.num_prototypes == 0
        assert stage_args.prototype_slots_per_task == 2
        assert stage_args.prototype_top_k == 1
        assert stage_args.transport_weight == 0.0

    manifest = results["manifest"]
    assert manifest["config"]["prototype_top_k"] == 4
    assert manifest["config"]["transport_weight"] == 0.2
    assert manifest["resolved_config"]["prototype_top_k"] == 1
    assert manifest["resolved_config"]["transport_weight"] == 0.0
    assert manifest["candidate_profile"] == "retention_no_transport"
    provenance = manifest["provenance"]
    assert provenance["argv"]
    assert provenance["python_version"]
    assert provenance["torch_version"]
    assert provenance["started_at_utc"]
    assert provenance["finished_at_utc"]
    assert provenance["git"]["commit"]
    assert isinstance(provenance["git"]["dirty"], bool)
    assert provenance["git"]["dirty"] is False
    assert provenance["dataset"] == {
        "name": "split_ag_news",
        "classes_per_task": 2,
        "max_train_samples": 64,
        "max_val_samples": 32,
        "num_seeds": 2,
        "seed": 42,
        "benchmark_provenance": {
            "train": {
                "source_kind": "fallback_synthetic",
                "split": "train",
                "sample_count": 64,
                "max_samples": 64,
                "reason": "ImportError",
            },
            "val": {
                "source_kind": "fallback_synthetic",
                "split": "test",
                "sample_count": 32,
                "max_samples": 32,
                "reason": "ImportError",
            },
        },
    }
    assert provenance["output_hashes"]["continual_benchmark.json"]
    assert provenance["output_hashes"]["continual_ablation.json"]
    assert provenance["output_hashes"]["continual_ablation_task_routing_seed42.json"]
    assert provenance["output_hashes"]["continual_operator_ablation.json"]

    report_text = Path(results["report_path"]).read_text(encoding="utf-8")
    assert "Candidate profile: `retention_no_transport`" in report_text
    assert "Prototype layout: `num_prototypes=0, slots_per_task=2, top_k=1`" in report_text
    assert "Transport weight: `0.0`" in report_text
    assert "Dataset source (train): `fallback_synthetic`" in report_text


def test_run_pipeline_syncs_appendix_outputs(tmp_path, monkeypatch):
    output_dir = tmp_path / "paper_suite"
    paper_tex = tmp_path / "asam_paper.tex"
    appendix_only_tex = tmp_path / "continual_appendix_only.tex"
    paper_tex.write_text(
        "prefix\n% BEGIN AUTO-GENERATED CONTINUAL APPENDIX\nold body\n% END AUTO-GENERATED CONTINUAL APPENDIX\nsuffix\n",
        encoding="utf-8",
    )

    def _write(path: Path, content: str) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return str(path)

    def fake_run_benchmark(args):
        payload = {
            "config": {
                "dataset_name": args.dataset_name,
                "classes_per_task": args.classes_per_task,
                "max_length": args.max_length,
                "batch_size": args.batch_size,
                "max_train_samples": args.max_train_samples,
                "max_val_samples": args.max_val_samples,
                "dim": args.dim,
                "num_heads": args.num_heads,
                "num_layers": args.num_layers,
                "top_k_patterns": args.top_k_patterns,
                "learning_rate": args.learning_rate,
                "epochs_per_task": args.epochs_per_task,
                "adaptation_strategy": "meta_secant",
            },
            "num_tasks": 2,
            "avg_accuracy": 0.5390625,
            "avg_forgetting": -0.0625,
            "backward_transfer": 0.0625,
        }
        output_json = Path(args.output_json)
        output_json.write_text(json.dumps(payload), encoding="utf-8")
        return {
            **payload,
            "plot_path": _write(output_dir / "continual_benchmark_plots.png", "plot"),
            "report_path": _write(output_dir / "continual_benchmark_report.md", "report"),
        }

    def fake_run_ablation(args):
        payload = {
            "config": {"num_seeds": args.num_seeds},
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
            "best_avg_accuracy": {"strategy": "task_routing", "value": 0.5234375},
            "lowest_avg_forgetting": {"strategy": "task_routing", "value": -0.046875},
            "table_path": _write(output_dir / "continual_ablation_table.md", "table"),
            "csv_path": _write(output_dir / "continual_ablation.csv", "csv"),
            "plot_path": _write(output_dir / "continual_ablation.png", "plot"),
            "report_path": _write(output_dir / "continual_ablation_report.md", "report"),
        }
        Path(args.output_json).write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def fake_run_operator_ablation(args):
        payload = {
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
            ],
            "best_avg_accuracy": {"strategy": "sinkhorn_topk", "value": 0.4973958333},
            "lowest_avg_forgetting": {"strategy": "sinkhorn_topk", "value": 0.046875},
            "table_path": _write(output_dir / "continual_operator_ablation_table.md", "table"),
            "csv_path": _write(output_dir / "continual_operator_ablation.csv", "csv"),
            "plot_path": _write(output_dir / "continual_operator_ablation.png", "plot"),
            "report_path": _write(output_dir / "continual_operator_ablation_report.md", "report"),
        }
        Path(args.output_json).write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr(paper_suite, "run_benchmark", fake_run_benchmark)
    monkeypatch.setattr(paper_suite, "run_ablation", fake_run_ablation)
    monkeypatch.setattr(paper_suite, "run_operator_ablation", fake_run_operator_ablation)

    args = PipelineArgs(
        output_dir=str(output_dir),
        num_seeds=3,
        paper_tex=str(paper_tex),
        appendix_only_tex=str(appendix_only_tex),
    )
    results = paper_suite.run_pipeline(args)

    manifest = results["manifest"]
    assert manifest["paper_tex"] == str(paper_tex)
    assert manifest["synced_paper_tex"] == str(paper_tex)
    assert manifest["appendix_only_tex"] == str(appendix_only_tex)
    assert Path(results["manifest_path"]).exists()
    assert Path(results["report_path"]).exists()

    synced_text = paper_tex.read_text(encoding="utf-8")
    assert "\\section{Continual ASAM Pilot Study}" in synced_text
    assert "task\\_routing" in synced_text

    appendix_only_text = appendix_only_tex.read_text(encoding="utf-8")
    assert appendix_only_text.startswith("% BEGIN AUTO-GENERATED CONTINUAL APPENDIX")
    assert "Sinkhorn Top-$k$" in appendix_only_text

    report_text = Path(results["report_path"]).read_text(encoding="utf-8")
    assert "## Paper Sync" in report_text
    assert "asam_paper.tex" in report_text
    assert "continual_appendix_only.tex" in report_text
