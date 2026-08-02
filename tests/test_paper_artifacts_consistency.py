"""Guard tests: every paper table number must match the canonical artifact.

If a table is edited without regenerating the artifact (or vice versa), one of
these tests fails. The canonical artifacts are:

- Table 1 (strategy ablation, BPE): experiments/paper_suite/r2_agnews_bpe_3ep.json
- Table 2 (operator ablation):       experiments/paper_suite/continual_operator_ablation.json
- Table 3 (baseline comparison):     experiments/paper_suite/r2_baseline_comparison.json
- Single-run diagnostics:            experiments/paper_suite/continual_benchmark.json
"""

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_TEX = REPO_ROOT / "paper" / "continual_asam.tex"
ARTIFACT_DIR = REPO_ROOT / "experiments" / "paper_suite"


def _load(name):
    with open(ARTIFACT_DIR / name, encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_tex(text):
    """Strip LaTeX formatting so values can be matched verbatim."""
    text = text.replace("+", "")
    text = re.sub(r"\\mathbf\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\texttt\{([^{}]*)\}", r"\1", text)
    text = text.replace("\\_", "_")
    return text


def _fmt(value):
    return f"{value:.4f}"


def _assert_pairs(tex_clean, pairs):
    for mean, std in pairs:
        needle = f"{_fmt(mean)} \\pm {_fmt(std)}"
        assert needle in tex_clean, f"tex is missing expected value pair: {needle!r}"


def test_canonical_artifacts_exist():
    for name in [
        "r2_agnews_bpe_3ep.json",
        "continual_operator_ablation.json",
        "r3_baseline_comparison.json",
        "continual_benchmark.json",
        "bootstrap_ci.json",
        "r4_dbpedia_longstream.json",
        "r4_cifar10_longstream.json",
        "r4_dbpedia_baselines.json",
        "r4_analysis.json",
    ]:
        assert (ARTIFACT_DIR / name).exists(), f"missing canonical artifact: {name}"


def test_table1_matches_bpe_rerun():
    data = _load("r2_agnews_bpe_3ep.json")
    tex_clean = _normalize_tex(PAPER_TEX.read_text(encoding="utf-8"))
    by_name = {row["strategy"]: row for row in data["aggregated_strategies"]}
    task = by_name["task_routing"]
    proto = by_name["no_adaptation"]
    _assert_pairs(
        tex_clean,
        [
            (task["avg_accuracy_mean"], task["avg_accuracy_std"]),
            (task["avg_forgetting_mean"], task["avg_forgetting_std"]),
            (proto["avg_accuracy_mean"], proto["avg_accuracy_std"]),
            (proto["avg_forgetting_mean"], proto["avg_forgetting_std"]),
            (proto["final_transport_gap_mean"], proto["final_transport_gap_std"]),
        ],
    )


def test_table2_matches_operator_ablation():
    data = _load("continual_operator_ablation.json")
    tex_clean = _normalize_tex(PAPER_TEX.read_text(encoding="utf-8"))
    for row in data["aggregated_strategies"]:
        _assert_pairs(
            tex_clean,
            [
                (row["avg_accuracy_mean"], row["avg_accuracy_std"]),
                (row["avg_forgetting_mean"], row["avg_forgetting_std"]),
                (row["final_transport_gap_mean"], row["final_transport_gap_std"]),
                (row["final_transport_loss_mean"], row["final_transport_loss_std"]),
            ],
        )


def test_table3_matches_baseline_comparison():
    data = _load("r3_baseline_comparison.json")
    tex_clean = _normalize_tex(PAPER_TEX.read_text(encoding="utf-8"))
    for row in data["methods"].values():
        _assert_pairs(
            tex_clean,
            [
                (row["accuracy_mean"], row["accuracy_std"]),
                (row["forgetting_mean"], row["forgetting_std"]),
                (row["backward_transfer_mean"], row["backward_transfer_std"]),
            ],
        )


def test_single_run_diagnostics_match_benchmark():
    data = _load("continual_benchmark.json")
    tex_clean = _normalize_tex(PAPER_TEX.read_text(encoding="utf-8"))
    for value in [data["avg_accuracy"], data["avg_forgetting"], data["backward_transfer"]]:
        assert _fmt(value) in tex_clean, f"tex is missing diagnostic value: {_fmt(value)}"


def test_abstract_matches_canonical_numbers():
    tex = PAPER_TEX.read_text(encoding="utf-8")
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    assert abstract is not None, "abstract environment not found"
    abstract_text = abstract.group(1)
    # Canonical Table 1 numbers that the abstract may cite.
    for needle in ["0.5104", "0.5312", "0.0000", "0.0417"]:
        assert needle in abstract_text, f"abstract is missing canonical value: {needle}"
    # The stitched two-experiment chain and the old overclaim must be gone.
    for stale in ["0.0625", "0.4844", "0.5156", "connect routing geometry diagnostics to forgetting"]:
        assert stale not in abstract_text, f"stale/stitched claim still in abstract: {stale}"
    assert (
        "Sec.~\\ref{sec:longstream}" in abstract_text
    ), "abstract should reference the long-stream correlation evidence"


def test_table4_matches_longstream_analysis():
    data = _load("r4_analysis.json")
    tex_clean = _normalize_tex(PAPER_TEX.read_text(encoding="utf-8"))
    datasets = {item["artifact"]: item for item in data["datasets"]}

    def agg(artifact, strategy):
        return datasets[artifact]["aggregated"][strategy]

    rows = [
        ("r4_dbpedia_baselines.json", "fine_tune"),
        ("r4_dbpedia_baselines.json", "ewc"),
        ("r4_dbpedia_baselines.json", "er"),
        ("r4_dbpedia_longstream.json", "no_adaptation"),
        ("r4_dbpedia_notransport", "no_transport"),
        ("r4_cifar10_longstream.json", "no_adaptation"),
        ("r4_cifar10_notransport", "no_transport"),
    ]
    for artifact, strategy in rows:
        row = agg(artifact, strategy)
        _assert_pairs(
            tex_clean,
            [
                (row["accuracy_mean"], row["accuracy_std"]),
                (row["forgetting_mean"], row["forgetting_std"]),
            ],
        )


def test_stale_unbacked_numbers_removed():
    tex = PAPER_TEX.read_text(encoding="utf-8")
    for stale in ["0.5521", "0.5469", "0.3203", "0.6250"]:
        assert stale not in tex, f"stale/unbacked number still present in tex: {stale}"
