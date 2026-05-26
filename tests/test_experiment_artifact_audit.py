import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from scripts.audit_experiment_artifacts import audit_paths, semantic_fingerprint


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _current_manifest():
    return {
        "resolved_config": {"candidate_profile": "default", "prototype_top_k": 2},
        "candidate_profile": "default",
        "candidate_profile_description": "Default profile.",
        "provenance": {
            "argv": ["scripts/run_continual_paper_suite.py"],
            "python_version": "3.10.0",
            "torch_version": "2.0.0",
            "started_at_utc": "2026-01-01T00:00:00Z",
            "finished_at_utc": "2026-01-01T00:01:00Z",
            "git": {"commit": "abcdef0", "dirty": False},
            "dataset": {
                "name": "split_ag_news",
                "classes_per_task": 2,
                "max_train_samples": 64,
                "max_val_samples": 32,
                "seed": 42,
                "num_seeds": 2,
            },
            "output_hashes": {
                "continual_benchmark.json": "a" * 64,
                "continual_ablation.json": "b" * 64,
                "continual_operator_ablation.json": "c" * 64,
            },
        },
        "benchmark_json": "old/path/benchmark.json",
        "ablation_json": "old/path/ablation.json",
        "operator_ablation_json": "old/path/operator.json",
    }


def test_counts_raw_duplicate_json_files(tmp_path):
    suite = tmp_path / "experiments" / "paper_suite_raw"
    payload = {
        "resolved_config": {"candidate_profile": "default"},
        "candidate_profile": "default",
        "candidate_profile_description": "Default profile.",
    }
    _write_json(suite / "a.json", payload)
    _write_json(suite / "b.json", payload)

    summary = audit_paths([suite])

    assert summary["json_file_count"] == 2
    assert summary["raw_duplicate_count"] == 1
    assert summary["suspicious_issue_count"] >= 1


def test_semantic_duplicates_ignore_output_paths(tmp_path):
    suite = tmp_path / "experiments" / "paper_suite_semantic"
    first = {
        "accuracy_matrix": [[0.9, 0.0], [0.8, 0.85]],
        "avg_accuracy": 0.875,
        "avg_forgetting": 0.1,
        "backward_transfer": -0.1,
        "stage_training_metrics": [{"stage": 0, "loss": 0.5}],
        "prototype_lifecycle": [{"stage": 0, "active": 2}],
        "resolved_config": {"candidate_profile": "default", "prototype_top_k": 2},
        "output_json": "run-a/result.json",
        "plot_path": "run-a/plot.png",
    }
    second = dict(first)
    second["output_json"] = "run-b/result.json"
    second["plot_path"] = "run-b/plot.png"
    _write_json(suite / "first.json", first)
    _write_json(suite / "second.json", second)

    summary = audit_paths([suite])

    assert summary["raw_duplicate_count"] == 0
    assert len(summary["semantic_duplicate_groups"]) == 1
    assert sorted(
        grouped["path"] for grouped in summary["semantic_duplicate_groups"][0]["files"]
    ) == [
        str(suite / "first.json"),
        str(suite / "second.json"),
    ]


def test_aggregate_strategy_metrics_change_semantic_fingerprint():
    base = {
        "config": {"dataset_name": "split_ag_news", "seed": 42, "output_json": "run-a.json"},
        "aggregated_strategies": [
            {"strategy": "sinkhorn_topk", "avg_accuracy_mean": 0.5},
        ],
        "best_avg_accuracy": {"strategy": "sinkhorn_topk", "value": 0.5},
        "csv_path": "run-a.csv",
    }
    changed = {
        **base,
        "config": {"dataset_name": "split_ag_news", "seed": 42, "output_json": "run-b.json"},
        "aggregated_strategies": [
            {"strategy": "masked_sinkhorn_topk", "avg_accuracy_mean": 0.6},
        ],
        "best_avg_accuracy": {"strategy": "masked_sinkhorn_topk", "value": 0.6},
        "csv_path": "run-b.csv",
    }

    assert semantic_fingerprint(base) != semantic_fingerprint(changed)


def test_schema_provenance_ratings(tmp_path):
    current = tmp_path / "experiments" / "paper_suite_current"
    _write_json(current / "paper_suite_manifest.json", _current_manifest())
    _write_json(
        current / "continual_ablation.json", {"strategies": [{"strategy": "dual_transport"}]}
    )
    _write_json(
        current / "continual_operator_ablation.json",
        {"strategies": [{"name": "masked_sinkhorn_topk"}]},
    )

    mixed = tmp_path / "experiments" / "paper_suite_mixed"
    _write_json(mixed / "paper_suite_manifest.json", _current_manifest())
    _write_json(mixed / "continual_ablation.json", {"strategies": [{"strategy": "meta_secant"}]})
    _write_json(
        mixed / "continual_operator_ablation.json",
        {"strategies": [{"name": "masked_sinkhorn_topk"}]},
    )

    outdated = tmp_path / "experiments" / "paper_suite_outdated"
    _write_json(
        outdated / "paper_suite_manifest.json", {"config": {"candidate_profile": "default"}}
    )

    summary = audit_paths([current, mixed, outdated])
    ratings = {suite["path"]: suite["schema_provenance_rating"] for suite in summary["suites"]}

    assert ratings[str(current)] == "CURRENT"
    assert ratings[str(mixed)] == "MIXED"
    assert ratings[str(outdated)] == "OUTDATED"
    assert summary["schema_provenance_rating"] == "OUTDATED"
    assert summary["blocking_issue_count"] >= 1
    assert summary["suspicious_issue_count"] >= 1


def test_manifest_without_strict_provenance_is_outdated(tmp_path):
    suite = tmp_path / "experiments" / "paper_suite_missing_provenance"
    manifest = _current_manifest()
    manifest.pop("provenance")
    _write_json(suite / "paper_suite_manifest.json", manifest)
    _write_json(suite / "continual_ablation.json", {"strategies": [{"strategy": "dual_transport"}]})
    _write_json(
        suite / "continual_operator_ablation.json",
        {"strategies": [{"name": "masked_sinkhorn_topk"}]},
    )

    summary = audit_paths([suite])

    assert summary["suites"][0]["schema_provenance_rating"] == "OUTDATED"
    assert any("strict provenance" in issue["message"] for issue in summary["blocking_issues"])


def test_manifest_with_unknown_git_commit_is_outdated(tmp_path):
    suite = tmp_path / "experiments" / "paper_suite_unknown_git"
    manifest = _current_manifest()
    manifest["provenance"]["git"]["commit"] = "unknown"
    _write_json(suite / "paper_suite_manifest.json", manifest)
    _write_json(suite / "continual_ablation.json", {"strategies": [{"strategy": "dual_transport"}]})
    _write_json(
        suite / "continual_operator_ablation.json",
        {"strategies": [{"name": "masked_sinkhorn_topk"}]},
    )

    summary = audit_paths([suite])

    assert summary["suites"][0]["schema_provenance_rating"] == "OUTDATED"
    assert any("provenance.git.commit" in issue["message"] for issue in summary["blocking_issues"])
