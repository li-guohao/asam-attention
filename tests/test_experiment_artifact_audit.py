import hashlib
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from scripts.audit_experiment_artifacts import audit_paths, semantic_fingerprint

TEXT_ARTIFACT_SUFFIXES = {".csv", ".json", ".md", ".tex", ".txt"}


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _file_sha256(path):
    content = path.read_bytes()
    if path.suffix.lower() in TEXT_ARTIFACT_SUFFIXES:
        content = content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def _sha256_bytes(content):
    return hashlib.sha256(content).hexdigest()


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
                "benchmark_provenance": {
                    "train": {
                        "source_kind": "fallback_synthetic",
                        "split": "train",
                        "sample_count": 64,
                        "max_samples": 64,
                    },
                    "val": {
                        "source_kind": "fallback_synthetic",
                        "split": "test",
                        "sample_count": 32,
                        "max_samples": 32,
                    },
                },
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


def _current_benchmark_payload():
    return {
        "accuracy_matrix": [[0.5]],
        "avg_accuracy": 0.5,
        "dataset_provenance": _current_manifest()["provenance"]["dataset"]["benchmark_provenance"],
    }


def _write_current_suite(suite):
    benchmark = _current_benchmark_payload()
    ablation = {"strategies": [{"strategy": "dual_transport"}]}
    operator = {"strategies": [{"name": "masked_sinkhorn_topk"}]}
    _write_json(suite / "continual_benchmark.json", benchmark)
    _write_json(suite / "continual_ablation.json", ablation)
    _write_json(suite / "continual_operator_ablation.json", operator)

    manifest = _current_manifest()
    manifest["provenance"]["output_hashes"] = {
        "continual_benchmark.json": _file_sha256(suite / "continual_benchmark.json"),
        "continual_ablation.json": _file_sha256(suite / "continual_ablation.json"),
        "continual_operator_ablation.json": _file_sha256(
            suite / "continual_operator_ablation.json"
        ),
    }
    _write_json(suite / "paper_suite_manifest.json", manifest)
    return manifest


def _long_context_manifest():
    return {
        "suite_type": "long_context",
        "resolved_config": {
            "sequence_lengths": [32, 64, 128],
            "models": ["asam", "transformer", "local", "longformer_style"],
        },
        "candidate_profile": "long_context_smoke",
        "candidate_profile_description": "CPU-runnable long-context diagnostic smoke suite.",
        "provenance": {
            "argv": ["scripts/run_long_context_paper_suite.py"],
            "python_version": "3.10.0",
            "torch_version": "2.0.0",
            "started_at_utc": "2026-01-01T00:00:00Z",
            "finished_at_utc": "2026-01-01T00:01:00Z",
            "git": {"commit": "abcdef0", "dirty": False},
            "benchmark": {
                "name": "lra_style_synthetic_diagnostic",
                "source_kind": "synthetic",
                "claim_scope": "diagnostic_only",
                "sequence_lengths": [32, 64, 128],
                "models": ["asam", "transformer", "local", "longformer_style"],
                "metric_names": ["latency_ms_mean", "peak_memory_mb", "finite_output_rate"],
            },
            "output_hashes": {
                "long_context_benchmark.json": "a" * 64,
                "long_context_benchmark.csv": "b" * 64,
                "long_context_benchmark_report.md": "c" * 64,
            },
        },
        "benchmark_json": "old/path/long_context_benchmark.json",
        "benchmark_csv": "old/path/long_context_benchmark.csv",
        "benchmark_report": "old/path/long_context_benchmark_report.md",
    }


def _long_context_payload():
    return {
        "suite_type": "long_context",
        "claim_scope": "diagnostic_only",
        "sequence_lengths": [32, 64, 128],
        "models": ["asam", "transformer", "local", "longformer_style"],
        "results": [
            {
                "model": "asam",
                "sequence_length": 32,
                "latency_ms_mean": 1.0,
                "peak_memory_mb": 0.0,
                "finite_output_rate": 1.0,
            },
            {
                "model": "transformer",
                "sequence_length": 32,
                "latency_ms_mean": 1.1,
                "peak_memory_mb": 0.0,
                "finite_output_rate": 1.0,
            },
            {
                "model": "local",
                "sequence_length": 32,
                "latency_ms_mean": 1.2,
                "peak_memory_mb": 0.0,
                "finite_output_rate": 1.0,
            },
            {
                "model": "longformer_style",
                "sequence_length": 32,
                "latency_ms_mean": 1.3,
                "peak_memory_mb": 0.0,
                "finite_output_rate": 1.0,
            },
        ],
    }


def _write_current_long_context_suite(suite):
    benchmark = _long_context_payload()
    csv = (
        "model,sequence_length,latency_ms_mean,peak_memory_mb,finite_output_rate\n"
        "asam,32,1.0,0.0,1.0\n"
    )
    report = "# Long-Context ASAM Paper Suite\n\nDiagnostic only.\n"
    _write_json(suite / "long_context_benchmark.json", benchmark)
    (suite / "long_context_benchmark.csv").write_text(csv, encoding="utf-8")
    (suite / "long_context_benchmark_report.md").write_text(report, encoding="utf-8")

    manifest = _long_context_manifest()
    manifest["provenance"]["output_hashes"] = {
        "long_context_benchmark.json": _file_sha256(suite / "long_context_benchmark.json"),
        "long_context_benchmark.csv": _file_sha256(suite / "long_context_benchmark.csv"),
        "long_context_benchmark_report.md": _file_sha256(
            suite / "long_context_benchmark_report.md"
        ),
    }
    _write_json(suite / "paper_suite_manifest.json", manifest)
    return manifest


def test_long_context_manifest_with_strict_provenance_is_current(tmp_path):
    suite = tmp_path / "experiments" / "paper_suite_long_context_current"
    _write_current_long_context_suite(suite)

    summary = audit_paths([suite])

    assert summary["suites"][0]["schema_provenance_rating"] == "CURRENT"
    assert summary["blocking_issue_count"] == 0


def test_long_context_manifest_requires_three_lengths_and_core_models(tmp_path):
    suite = tmp_path / "experiments" / "paper_suite_long_context_bad"
    _write_current_long_context_suite(suite)
    manifest = json.loads((suite / "paper_suite_manifest.json").read_text(encoding="utf-8"))
    manifest["provenance"]["benchmark"]["sequence_lengths"] = [32, 64]
    manifest["provenance"]["benchmark"]["models"] = ["asam", "transformer", "local"]
    _write_json(suite / "paper_suite_manifest.json", manifest)

    summary = audit_paths([suite])

    assert summary["suites"][0]["schema_provenance_rating"] == "OUTDATED"
    assert any("long-context benchmark" in issue["message"] for issue in summary["blocking_issues"])


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
    _write_current_suite(current)

    mixed = tmp_path / "experiments" / "paper_suite_mixed"
    _write_current_suite(mixed)
    _write_json(mixed / "continual_ablation.json", {"strategies": [{"strategy": "meta_secant"}]})
    manifest = _current_manifest()
    manifest["provenance"]["output_hashes"] = {
        "continual_benchmark.json": _file_sha256(mixed / "continual_benchmark.json"),
        "continual_ablation.json": _file_sha256(mixed / "continual_ablation.json"),
        "continual_operator_ablation.json": _file_sha256(
            mixed / "continual_operator_ablation.json"
        ),
    }
    _write_json(mixed / "paper_suite_manifest.json", manifest)

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


def test_manifest_without_dataset_source_metadata_is_outdated(tmp_path):
    suite = tmp_path / "experiments" / "paper_suite_missing_dataset_source"
    manifest = _current_manifest()
    manifest["provenance"]["dataset"].pop("benchmark_provenance")
    _write_json(suite / "paper_suite_manifest.json", manifest)
    _write_json(suite / "continual_ablation.json", {"strategies": [{"strategy": "dual_transport"}]})
    _write_json(
        suite / "continual_operator_ablation.json",
        {"strategies": [{"name": "masked_sinkhorn_topk"}]},
    )

    summary = audit_paths([suite])

    assert summary["suites"][0]["schema_provenance_rating"] == "OUTDATED"
    assert any(
        "provenance.dataset.benchmark_provenance" in issue["message"]
        for issue in summary["blocking_issues"]
    )


def test_benchmark_json_without_dataset_source_metadata_is_outdated(tmp_path):
    suite = tmp_path / "experiments" / "paper_suite_missing_benchmark_dataset_source"
    _write_current_suite(suite)
    _write_json(suite / "continual_benchmark.json", {"accuracy_matrix": [[0.5]]})
    manifest = json.loads((suite / "paper_suite_manifest.json").read_text(encoding="utf-8"))
    manifest["provenance"]["output_hashes"]["continual_benchmark.json"] = _file_sha256(
        suite / "continual_benchmark.json"
    )
    _write_json(suite / "paper_suite_manifest.json", manifest)

    summary = audit_paths([suite])

    assert summary["suites"][0]["schema_provenance_rating"] == "OUTDATED"
    messages = [issue["message"] for issue in summary["blocking_issues"]]
    assert messages.count("continual_benchmark.json.dataset_provenance is missing") == 1


def test_benchmark_json_dataset_source_must_match_manifest(tmp_path):
    suite = tmp_path / "experiments" / "paper_suite_mismatched_benchmark_dataset_source"
    _write_current_suite(suite)
    benchmark = _current_benchmark_payload()
    benchmark["dataset_provenance"]["train"]["source_kind"] = "huggingface"
    _write_json(suite / "continual_benchmark.json", benchmark)
    manifest = json.loads((suite / "paper_suite_manifest.json").read_text(encoding="utf-8"))
    manifest["provenance"]["output_hashes"]["continual_benchmark.json"] = _file_sha256(
        suite / "continual_benchmark.json"
    )
    _write_json(suite / "paper_suite_manifest.json", manifest)

    summary = audit_paths([suite])

    assert summary["suites"][0]["schema_provenance_rating"] == "OUTDATED"
    assert any(
        "does not match provenance.dataset.benchmark_provenance" in issue["message"]
        for issue in summary["blocking_issues"]
    )


def test_dataset_source_metadata_requires_strict_split_schema(tmp_path):
    suite = tmp_path / "experiments" / "paper_suite_bad_dataset_source_schema"
    _write_current_suite(suite)
    manifest = json.loads((suite / "paper_suite_manifest.json").read_text(encoding="utf-8"))
    benchmark = _current_benchmark_payload()
    for payload in (
        manifest["provenance"]["dataset"]["benchmark_provenance"],
        benchmark["dataset_provenance"],
    ):
        payload["train"].pop("split")
        payload["val"]["source_kind"] = "unknown_source"

    _write_json(suite / "continual_benchmark.json", benchmark)
    manifest["provenance"]["output_hashes"]["continual_benchmark.json"] = _file_sha256(
        suite / "continual_benchmark.json"
    )
    _write_json(suite / "paper_suite_manifest.json", manifest)

    summary = audit_paths([suite])

    assert summary["suites"][0]["schema_provenance_rating"] == "OUTDATED"
    assert any(
        "provenance.dataset.benchmark_provenance.train.split" in issue["message"]
        for issue in summary["blocking_issues"]
    )
    assert any(
        "provenance.dataset.benchmark_provenance.val.source_kind" in issue["message"]
        for issue in summary["blocking_issues"]
    )


def test_manifest_hash_mismatch_is_blocking(tmp_path):
    suite = tmp_path / "experiments" / "paper_suite_hash_mismatch"
    _write_current_suite(suite)
    _write_json(suite / "continual_benchmark.json", {"accuracy_matrix": [[0.9]]})

    summary = audit_paths([suite])

    assert summary["blocking_issue_count"] >= 1
    assert any("hash mismatch" in issue["message"] for issue in summary["blocking_issues"])


def test_manifest_missing_hashed_artifact_is_blocking(tmp_path):
    suite = tmp_path / "experiments" / "paper_suite_missing_artifact"
    _write_current_suite(suite)
    (suite / "continual_operator_ablation.json").unlink()

    summary = audit_paths([suite])

    assert summary["blocking_issue_count"] >= 1
    assert any(
        "hashed artifact is missing" in issue["message"] for issue in summary["blocking_issues"]
    )


def test_unhashed_json_artifact_is_blocking(tmp_path):
    suite = tmp_path / "experiments" / "paper_suite_unhashed_json"
    _write_current_suite(suite)
    _write_json(
        suite / "continual_operator_ablation_sinkhorn_topk_seed42.json", {"avg_accuracy": 0.5}
    )

    summary = audit_paths([suite])

    assert summary["blocking_issue_count"] >= 1
    assert any(
        "JSON artifact is not covered" in issue["message"] for issue in summary["blocking_issues"]
    )


def test_manifest_hash_key_must_stay_within_suite(tmp_path):
    suite = tmp_path / "experiments" / "paper_suite_bad_hash_key"
    bad_keys = [
        "../escape.json",
        "/escape.json",
        "C:/escape.json",
        "C:escape.json",
        "//server/share/escape.json",
    ]

    for bad_key in bad_keys:
        _write_current_suite(suite)
        manifest = json.loads((suite / "paper_suite_manifest.json").read_text(encoding="utf-8"))
        manifest["provenance"]["output_hashes"][bad_key] = "d" * 64
        _write_json(suite / "paper_suite_manifest.json", manifest)

        summary = audit_paths([suite])

        assert summary["blocking_issue_count"] >= 1
        assert any(
            "not a relative suite path" in issue["message"] for issue in summary["blocking_issues"]
        )


def test_text_artifact_hashes_are_line_ending_stable(tmp_path):
    suite = tmp_path / "experiments" / "paper_suite_line_endings"
    _write_current_suite(suite)
    crlf_payload = (
        json.dumps(_current_benchmark_payload(), separators=(",", ":")).encode("utf-8") + b"\r\n"
    )
    lf_payload = crlf_payload.replace(b"\r\n", b"\n")
    (suite / "continual_benchmark.json").write_bytes(crlf_payload)
    manifest = json.loads((suite / "paper_suite_manifest.json").read_text(encoding="utf-8"))
    manifest["provenance"]["output_hashes"]["continual_benchmark.json"] = _sha256_bytes(lf_payload)
    _write_json(suite / "paper_suite_manifest.json", manifest)

    summary = audit_paths([suite])

    assert summary["blocking_issue_count"] == 0
