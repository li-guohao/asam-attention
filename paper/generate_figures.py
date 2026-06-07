"""
Generate artifact-backed figures for the ASAM paper draft.

Paper-facing numerical plots must be traceable to saved JSON/CSV artifacts.
When an audited artifact is unavailable, this script skips the corresponding
plot instead of fabricating placeholder values.
"""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.dpi"] = 150

FIGURES_DIR = Path("figures")
LONG_CONTEXT_JSON = Path("experiments/paper_suite_long_context_smoke/long_context_benchmark.json")
LONG_CONTEXT_BENCHMARK_NAME = "lra_style_synthetic_diagnostic"
TEXT_ARTIFACT_SUFFIXES = {".csv", ".json", ".md", ".tex", ".txt"}
REQUIRED_LONG_CONTEXT_MODELS = {"asam", "transformer", "local", "longformer_style"}
REQUIRED_LONG_CONTEXT_METRICS = {
    "latency_ms_mean",
    "latency_ms_std",
    "peak_memory_mb",
    "finite_output_rate",
}
REQUIRED_RESULT_FIELDS = {
    "model",
    "sequence_length",
    "latency_ms_mean",
    "finite_output_rate",
}
MIN_LONG_CONTEXT_SEQUENCE_LENGTHS = 3


def _load_json(path):
    try:
        with Path(path).open(encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        print(f"Skipped: {path} is unavailable.")
        return None


def _file_sha256(path):
    content = Path(path).read_bytes()
    if Path(path).suffix.lower() in TEXT_ARTIFACT_SUFFIXES:
        content = content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def _load_manifest_for_benchmark(json_path):
    manifest_path = Path(json_path).parent / "paper_suite_manifest.json"
    manifest = _load_json(manifest_path)
    if manifest is None:
        return None
    if manifest.get("suite_type") != "long_context":
        print(f"Skipped: {manifest_path} is not a long-context suite manifest.")
        return None
    return manifest


def _validate_manifest_hash(json_path, manifest):
    output_hashes = manifest.get("provenance", {}).get("output_hashes", {})
    if not isinstance(output_hashes, dict):
        print("Skipped: manifest has no provenance.output_hashes mapping.")
        return False

    relative_name = Path(json_path).name
    expected_hash = output_hashes.get(relative_name)
    if not expected_hash:
        print(f"Skipped: manifest does not cover {relative_name}.")
        return False

    actual_hash = _file_sha256(json_path)
    if actual_hash != expected_hash:
        print(f"Skipped: manifest hash mismatch for {relative_name}.")
        return False
    return True


def _validate_manifest_claim_scope(data, manifest):
    benchmark = manifest.get("provenance", {}).get("benchmark")
    if not isinstance(benchmark, dict):
        print("Skipped: manifest has no provenance.benchmark section.")
        return False

    expected = {
        "suite_type": "long_context",
        "benchmark_name": benchmark.get("name"),
        "claim_scope": "diagnostic_only",
    }
    mismatches = {
        key: (data.get(key), value) for key, value in expected.items() if data.get(key) != value
    }
    if benchmark.get("source_kind") != "synthetic":
        mismatches["source_kind"] = (benchmark.get("source_kind"), "synthetic")
    if benchmark.get("name") != LONG_CONTEXT_BENCHMARK_NAME:
        mismatches["benchmark_name"] = (benchmark.get("name"), LONG_CONTEXT_BENCHMARK_NAME)
    if mismatches:
        print(f"Skipped: benchmark artifact is outside diagnostic scope: {mismatches}")
        return False

    for key in ["sequence_lengths", "models"]:
        if data.get(key) != benchmark.get(key):
            print(f"Skipped: benchmark {key} does not match manifest provenance.")
            return False

    models = data.get("models")
    metric_names = benchmark.get("metric_names")
    if not isinstance(models, list) or not REQUIRED_LONG_CONTEXT_MODELS.issubset(set(models)):
        print("Skipped: benchmark does not include all required long-context models.")
        return False
    if not isinstance(metric_names, list) or not REQUIRED_LONG_CONTEXT_METRICS.issubset(
        set(metric_names)
    ):
        print("Skipped: manifest does not include required long-context metrics.")
        return False

    return True


def _successful_rows(data):
    return [row for row in data.get("results", []) if row.get("success", True)]


def _validate_result_rows(data):
    rows = _successful_rows(data)
    if not rows:
        print("Skipped: benchmark artifact has no successful result rows.")
        return False

    sequence_lengths = data.get("sequence_lengths")
    models = data.get("models")
    if not isinstance(sequence_lengths, list) or len(sequence_lengths) < MIN_LONG_CONTEXT_SEQUENCE_LENGTHS:
        print("Skipped: benchmark does not cover the required sequence-length grid.")
        return False
    if not isinstance(models, list):
        print("Skipped: benchmark does not declare compared models.")
        return False

    expected_pairs = {(model, length) for model in models for length in sequence_lengths}
    observed_pairs = set()
    for index, row in enumerate(rows):
        missing = REQUIRED_RESULT_FIELDS - set(row)
        if missing:
            print(f"Skipped: result row {index} is missing fields: {sorted(missing)}")
            return False
        observed_pairs.add((row["model"], row["sequence_length"]))

    missing_pairs = expected_pairs - observed_pairs
    if missing_pairs:
        print(f"Skipped: benchmark is missing result rows: {sorted(missing_pairs)}")
        return False

    unexpected_pairs = observed_pairs - expected_pairs
    if unexpected_pairs:
        print(f"Skipped: benchmark has undeclared result rows: {sorted(unexpected_pairs)}")
        return False

    return True


def load_long_context_diagnostic(json_path=LONG_CONTEXT_JSON):
    """Load the manifest-audited long-context diagnostic artifact."""

    data = _load_json(json_path)
    if data is None:
        return None

    manifest = _load_manifest_for_benchmark(json_path)
    if manifest is None:
        return None

    if not _validate_manifest_hash(json_path, manifest):
        return None

    if not _validate_manifest_claim_scope(data, manifest):
        return None

    if not _validate_result_rows(data):
        return None

    return data


def _rows_by_model(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)
    for model_rows in grouped.values():
        model_rows.sort(key=lambda item: item["sequence_length"])
    return grouped


def _save(fig, stem):
    FIGURES_DIR.mkdir(exist_ok=True)
    pdf_path = FIGURES_DIR / f"{stem}.pdf"
    png_path = FIGURES_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {pdf_path}")
    return [pdf_path, png_path]


def generate_long_context_latency_figure():
    """Plot CPU smoke latency from the LRA-style synthetic diagnostic artifact."""

    data = load_long_context_diagnostic()
    if data is None:
        return []

    result_rows = _successful_rows(data)
    grouped = _rows_by_model(result_rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, model_rows in grouped.items():
        seq_lengths = [row["sequence_length"] for row in model_rows]
        latency = [row["latency_ms_mean"] for row in model_rows]
        latency_std = [row.get("latency_ms_std", 0.0) for row in model_rows]
        ax.errorbar(seq_lengths, latency, yerr=latency_std, marker="o", label=model)

    ax.set_xlabel("Sequence Length", fontweight="bold")
    ax.set_ylabel("Latency (ms)", fontweight="bold")
    ax.set_title("LRA-Style Synthetic Latency Diagnostic", fontweight="bold")
    ax.set_xscale("log", base=2)
    if any(row["latency_ms_mean"] > 0 for row in result_rows):
        ax.set_yscale("log")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)
    ax.text(
        0.01,
        -0.22,
        "Diagnostic only: not an official LRA result or hardware speedup claim.",
        transform=ax.transAxes,
        fontsize=9,
    )
    fig.tight_layout()
    return _save(fig, "figure1_long_context_latency_diagnostic")


def generate_long_context_quality_figure():
    """Plot finite-output and memory diagnostics from the saved artifact."""

    data = load_long_context_diagnostic()
    if data is None:
        return []

    result_rows = _successful_rows(data)
    grouped = _rows_by_model(result_rows)
    fig, (output_ax, memory_ax) = plt.subplots(1, 2, figsize=(12, 5))

    for model, model_rows in grouped.items():
        seq_lengths = [row["sequence_length"] for row in model_rows]
        finite_rate = [row["finite_output_rate"] for row in model_rows]
        memory = [row.get("peak_memory_mb", 0.0) for row in model_rows]
        output_ax.plot(seq_lengths, finite_rate, marker="o", label=model)
        memory_ax.plot(seq_lengths, memory, marker="o", label=model)

    output_ax.set_xlabel("Sequence Length", fontweight="bold")
    output_ax.set_ylabel("Finite Output Rate", fontweight="bold")
    output_ax.set_title("Output Validity Diagnostic", fontweight="bold")
    output_ax.set_xscale("log", base=2)
    output_ax.set_ylim(0, 1.05)
    output_ax.grid(True, alpha=0.3)

    memory_ax.set_xlabel("Sequence Length", fontweight="bold")
    memory_ax.set_ylabel("Peak Memory (MB)", fontweight="bold")
    memory_ax.set_title("Recorded Memory Diagnostic", fontweight="bold")
    memory_ax.set_xscale("log", base=2)
    if any(row.get("peak_memory_mb", 0.0) > 0 for row in result_rows):
        memory_ax.set_yscale("log")
    else:
        memory_ax.text(
            0.5,
            0.5,
            "CPU smoke run records 0 MB\nfor all compared operators.",
            ha="center",
            va="center",
            transform=memory_ax.transAxes,
        )
    memory_ax.grid(True, alpha=0.3)
    memory_ax.legend(frameon=True)

    fig.tight_layout()
    return _save(fig, "figure2_long_context_quality_diagnostic")


def generate_sparse_pattern_schematic():
    """Generate a non-result schematic of sparse support patterns."""

    seq_len = 64
    patterns = {}

    local = np.zeros((seq_len, seq_len))
    window = 8
    for query_idx in range(seq_len):
        start = max(0, query_idx - window)
        end = min(seq_len, query_idx + window + 1)
        local[query_idx, start:end] = 1
    patterns["Local Support"] = local

    strided = np.zeros((seq_len, seq_len))
    stride = 4
    for query_idx in range(seq_len):
        strided[query_idx, range(0, seq_len, stride)] = 1
        start = max(0, query_idx - 4)
        end = min(seq_len, query_idx + 5)
        strided[query_idx, start:end] = 1
    patterns["Strided + Local Support"] = strided

    random_generator = np.random.default_rng(42)
    random_support = random_generator.random((seq_len, seq_len)) < 0.1
    for query_idx in range(seq_len):
        start = max(0, query_idx - 2)
        end = min(seq_len, query_idx + 3)
        random_support[query_idx, start:end] = True
    patterns["Seeded Random Support"] = random_support.astype(float)

    hierarchical = np.zeros((seq_len, seq_len))
    for query_idx in range(seq_len):
        start = max(0, query_idx - 4)
        end = min(seq_len, query_idx + 5)
        hierarchical[query_idx, start:end] = 0.5
        hierarchical[query_idx, range(0, seq_len, 8)] = 1.0
    patterns["Hierarchical Support"] = hierarchical

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, (name, pattern) in zip(axes.flatten(), patterns.items()):
        image = ax.imshow(pattern, cmap="Blues", interpolation="nearest")
        sparsity = 1 - pattern.mean()
        ax.set_title(f"{name}\nSparsity: {sparsity:.1%}", fontweight="bold")
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")
        ax.set_xticks(np.arange(0, seq_len, 16))
        ax.set_yticks(np.arange(0, seq_len, 16))
        ax.grid(True, alpha=0.3, color="gray", linewidth=0.5)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Sparse Support Pattern Schematics", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return _save(fig, "figure3_sparse_pattern_schematic")


def generate_long_context_table():
    """Print a LaTeX table directly from the diagnostic JSON."""

    data = load_long_context_diagnostic()
    print("\n% Long-context diagnostic table")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{LRA-style synthetic diagnostic; not official LRA.}")
    print("\\begin{tabular}{lrrr}")
    print("\\toprule")
    print("Model & Sequence length & Latency (ms) & Finite output \\\\")
    print("\\midrule")

    if data is None:
        print("\\multicolumn{4}{c}{No audited diagnostic artifact available} \\\\")
    else:
        rows = _successful_rows(data)
        for row in sorted(rows, key=lambda item: (item["model"], item["sequence_length"])):
            print(
                f"{row['model']} & {row['sequence_length']} & "
                f"{row['latency_ms_mean']:.4f} & {row['finite_output_rate']:.3f} \\\\"
            )

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def main():
    """Generate figures and tables from audited artifacts."""

    print("=" * 60)
    print("ASAM Paper Artifact-Backed Figure Generation")
    print("=" * 60)

    generated = []
    generated.extend(generate_long_context_latency_figure())
    generated.extend(generate_long_context_quality_figure())
    generated.extend(generate_sparse_pattern_schematic())
    generate_long_context_table()

    print()
    print("=" * 60)
    print(f"Generated {len(generated)} artifact-backed or schematic files.")
    print("=" * 60)
    for path in generated:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
