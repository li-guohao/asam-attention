import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import scripts.run_long_context_paper_suite as long_context_suite
from scripts.run_long_context_paper_suite import LongContextArgs, run_pipeline


def test_run_pipeline_writes_long_context_artifacts(tmp_path, monkeypatch):
    output_dir = tmp_path / "paper_suite_long_context"

    def fake_run_benchmark(args):
        return {
            "suite_type": "long_context",
            "claim_scope": "diagnostic_only",
            "sequence_lengths": args.sequence_lengths,
            "models": args.models,
            "results": [
                {
                    "model": "asam",
                    "sequence_length": args.sequence_lengths[0],
                    "latency_ms_mean": 1.0,
                    "latency_ms_std": 0.0,
                    "peak_memory_mb": 0.0,
                    "finite_output_rate": 1.0,
                    "diagnostic_score": 1.0,
                    "success": True,
                },
                {
                    "model": "transformer",
                    "sequence_length": args.sequence_lengths[0],
                    "latency_ms_mean": 1.2,
                    "latency_ms_std": 0.0,
                    "peak_memory_mb": 0.0,
                    "finite_output_rate": 1.0,
                    "diagnostic_score": 1.0,
                    "success": True,
                },
            ],
        }

    monkeypatch.setattr(long_context_suite, "run_benchmark", fake_run_benchmark)
    monkeypatch.setattr(
        long_context_suite,
        "collect_git_provenance",
        lambda: {"commit": "abc123", "dirty": False, "status_porcelain": ""},
    )

    args = LongContextArgs(
        output_dir=str(output_dir),
        sequence_lengths=[32, 64, 128],
        models=["asam", "transformer", "local", "longformer_style"],
        repeats=1,
        warmup=0,
        device="cpu",
    )

    results = run_pipeline(args)

    manifest_path = Path(results["manifest_path"])
    benchmark_json = output_dir / "long_context_benchmark.json"
    benchmark_csv = output_dir / "long_context_benchmark.csv"
    report = output_dir / "long_context_benchmark_report.md"

    assert benchmark_json.exists()
    assert benchmark_csv.exists()
    assert report.exists()
    assert manifest_path.exists()

    benchmark = json.loads(benchmark_json.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert benchmark["claim_scope"] == "diagnostic_only"
    assert manifest["suite_type"] == "long_context"
    assert manifest["provenance"]["benchmark"]["claim_scope"] == "diagnostic_only"
    assert manifest["provenance"]["benchmark"]["source_kind"] == "synthetic"
    assert manifest["provenance"]["output_hashes"]["long_context_benchmark.json"]
    assert manifest["provenance"]["output_hashes"]["long_context_benchmark.csv"]
    assert manifest["provenance"]["output_hashes"]["long_context_benchmark_report.md"]

    report_text = report.read_text(encoding="utf-8")
    assert "# Long-Context ASAM Paper Suite" in report_text
    assert "Diagnostic only" in report_text


def test_parse_sequence_lengths_and_models():
    args = long_context_suite.parse_args(
        [
            "--sequence-lengths",
            "32,64,128",
            "--models",
            "asam,transformer,local,longformer_style",
        ]
    )

    assert args.sequence_lengths == [32, 64, 128]
    assert args.models == ["asam", "transformer", "local", "longformer_style"]
