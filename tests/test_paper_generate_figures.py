import hashlib
import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_generate_figures_module():
    module_path = REPO_ROOT / "paper" / "generate_figures.py"
    spec = importlib.util.spec_from_file_location("paper_generate_figures", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _sha256_text(path):
    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def _diagnostic_payload():
    sequence_lengths = [32, 64, 128]
    models = ["asam", "transformer", "local", "longformer_style"]
    results = []
    for model_index, model in enumerate(models):
        for length_index, sequence_length in enumerate(sequence_lengths):
            results.append(
                {
                    "model": model,
                    "sequence_length": sequence_length,
                    "latency_ms_mean": 1.0 + model_index + length_index / 10,
                    "latency_ms_std": 0.0,
                    "peak_memory_mb": 0.0,
                    "finite_output_rate": 1.0,
                    "success": True,
                }
            )

    return {
        "suite_type": "long_context",
        "benchmark_name": "lra_style_synthetic_diagnostic",
        "claim_scope": "diagnostic_only",
        "sequence_lengths": sequence_lengths,
        "models": models,
        "results": results,
    }


def _manifest_payload(benchmark_path):
    benchmark = _diagnostic_payload()
    return {
        "suite_type": "long_context",
        "resolved_config": {
            "sequence_lengths": benchmark["sequence_lengths"],
            "models": benchmark["models"],
        },
        "candidate_profile": "long_context_smoke",
        "candidate_profile_description": "CPU-runnable LRA-style synthetic diagnostic suite.",
        "provenance": {
            "argv": ["scripts/run_long_context_paper_suite.py"],
            "python_version": "3.10.0",
            "torch_version": "2.0.0",
            "started_at_utc": "2026-01-01T00:00:00Z",
            "finished_at_utc": "2026-01-01T00:01:00Z",
            "git": {"commit": "abc123", "dirty": False},
            "benchmark": {
                "name": benchmark["benchmark_name"],
                "source_kind": "synthetic",
                "claim_scope": benchmark["claim_scope"],
                "sequence_lengths": benchmark["sequence_lengths"],
                "models": benchmark["models"],
                "metric_names": [
                    "latency_ms_mean",
                    "latency_ms_std",
                    "peak_memory_mb",
                    "finite_output_rate",
                ],
            },
            "output_hashes": {
                "long_context_benchmark.json": _sha256_text(benchmark_path),
            },
        },
    }


def test_load_long_context_diagnostic_requires_manifest_hash(tmp_path):
    generate_figures = _load_generate_figures_module()
    suite = tmp_path / "paper_suite_long_context"
    suite.mkdir()
    benchmark_path = suite / "long_context_benchmark.json"
    _write_json(benchmark_path, _diagnostic_payload())

    assert generate_figures.load_long_context_diagnostic(benchmark_path) is None

    _write_json(suite / "paper_suite_manifest.json", _manifest_payload(benchmark_path))
    loaded = generate_figures.load_long_context_diagnostic(benchmark_path)

    assert loaded is not None
    assert loaded["claim_scope"] == "diagnostic_only"


def test_load_long_context_diagnostic_rejects_stale_manifest_hash(tmp_path):
    generate_figures = _load_generate_figures_module()
    suite = tmp_path / "paper_suite_long_context"
    suite.mkdir()
    benchmark_path = suite / "long_context_benchmark.json"
    payload = _diagnostic_payload()
    _write_json(benchmark_path, payload)
    _write_json(suite / "paper_suite_manifest.json", _manifest_payload(benchmark_path))

    payload["results"][0]["latency_ms_mean"] = 99.0
    _write_json(benchmark_path, payload)

    assert generate_figures.load_long_context_diagnostic(benchmark_path) is None


def test_load_long_context_diagnostic_rejects_non_diagnostic_name(tmp_path):
    generate_figures = _load_generate_figures_module()
    suite = tmp_path / "paper_suite_long_context"
    suite.mkdir()
    benchmark_path = suite / "long_context_benchmark.json"
    payload = _diagnostic_payload()
    payload["benchmark_name"] = "official_lra"
    _write_json(benchmark_path, payload)
    manifest = _manifest_payload(benchmark_path)
    manifest["provenance"]["benchmark"]["name"] = "official_lra"
    _write_json(suite / "paper_suite_manifest.json", manifest)

    assert generate_figures.load_long_context_diagnostic(benchmark_path) is None


def test_load_long_context_diagnostic_rejects_incomplete_result_grid(tmp_path):
    generate_figures = _load_generate_figures_module()
    suite = tmp_path / "paper_suite_long_context"
    suite.mkdir()
    benchmark_path = suite / "long_context_benchmark.json"
    payload = _diagnostic_payload()
    payload["results"] = [row for row in payload["results"] if row["model"] != "local"]
    _write_json(benchmark_path, payload)
    _write_json(suite / "paper_suite_manifest.json", _manifest_payload(benchmark_path))

    assert generate_figures.load_long_context_diagnostic(benchmark_path) is None


def test_generate_figures_has_no_unaudited_lra_or_ablation_fallbacks():
    source = (REPO_ROOT / "paper" / "generate_figures.py").read_text(encoding="utf-8").lower()

    forbidden_claim_fragments = [
        "fallback simulated data",
        "simulated data",
        "diagnostic placeholder data",
        "long range arena results",
        "figure1_lra_results",
        "figure2_efficiency",
        "ablation study results",
        "accuracy (%)",
        "rtx 3060",
        "sota",
    ]

    for fragment in forbidden_claim_fragments:
        assert fragment not in source
