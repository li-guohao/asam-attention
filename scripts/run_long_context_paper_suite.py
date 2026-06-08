"""
Paper-facing long-context diagnostic suite for ASAM.

The suite is intentionally scoped as an LRA-style synthetic diagnostic. It
measures latency, memory, and finite-output sanity across sequence lengths and
attention operators, but it does not claim official LRA accuracy or SOTA status.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from asam import ASAMConfig, ASAMLayer

REPO_ROOT = Path(__file__).resolve().parent.parent
TEXT_ARTIFACT_SUFFIXES = {".csv", ".json", ".md", ".tex", ".txt"}
SENSITIVE_ARG_MARKERS = ("token", "password", "secret", "key")
DEFAULT_MODELS = ["asam", "transformer", "local", "longformer_style"]
DEFAULT_SEQUENCE_LENGTHS = [64, 128, 256]
METRIC_NAMES = ["latency_ms_mean", "latency_ms_std", "peak_memory_mb", "finite_output_rate"]


@dataclass
class LongContextArgs:
    output_dir: str = "experiments/paper_suite_long_context_smoke"
    sequence_lengths: list[int] = field(default_factory=lambda: list(DEFAULT_SEQUENCE_LENGTHS))
    models: list[str] = field(default_factory=lambda: list(DEFAULT_MODELS))
    batch_size: int = 2
    dim: int = 32
    num_heads: int = 2
    window_size: int = 16
    global_tokens: int = 1
    warmup: int = 1
    repeats: int = 3
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class DenseAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor, allowed_mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        hidden = self.norm(x)
        attn_mask = None
        if allowed_mask is not None:
            attn_mask = ~allowed_mask.to(dtype=torch.bool, device=x.device)
        attn_out, _weights = self.attn(hidden, hidden, hidden, attn_mask=attn_mask)
        hidden = residual + attn_out
        return hidden + self.ffn(self.ffn_norm(hidden))


class ASAMBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int):
        super().__init__()
        dim_head = max(1, dim // num_heads)
        self.layer = ASAMLayer(
            ASAMConfig(
                dim=dim,
                num_heads=num_heads,
                dim_head=dim_head,
                dropout=0.0,
                pattern_type="hierarchical",
                window_size=window_size,
                use_adaptive_gate=False,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _info = self.layer(x, return_info=True)
        return output


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _run_git_command(args: Iterable[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def collect_git_provenance() -> dict[str, object]:
    status = _run_git_command(["status", "--porcelain"])
    return {
        "commit": _run_git_command(["rev-parse", "HEAD"]),
        "dirty": bool(status and status != "unknown"),
        "status_porcelain": status if status != "unknown" else "",
    }


def _file_sha256(path: Path) -> str:
    content = path.read_bytes()
    if path.suffix.lower() in TEXT_ARTIFACT_SUFFIXES:
        content = content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def collect_output_hashes(paths: Iterable[Path], output_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in paths:
        if path.exists() and path.is_file():
            hashes[path.resolve().relative_to(output_root.resolve()).as_posix()] = _file_sha256(
                path
            )
    return hashes


def _is_sensitive_arg(arg: str) -> bool:
    lowered = arg.lower().lstrip("-")
    return any(marker in lowered for marker in SENSITIVE_ARG_MARKERS)


def redact_argv(argv: Iterable[str]) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    for raw_arg in argv:
        arg = str(raw_arg)
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue
        if "=" in arg:
            key, _value = arg.split("=", 1)
            if _is_sensitive_arg(key):
                redacted.append(f"{key}=<redacted>")
                continue
        if _is_sensitive_arg(arg):
            redacted.append(arg)
            redact_next = True
            continue
        redacted.append(arg)
    return redacted


def _local_allowed_mask(seq_len: int, window_size: int) -> torch.Tensor:
    positions = torch.arange(seq_len)
    distance = (positions[:, None] - positions[None, :]).abs()
    return distance <= max(0, window_size // 2)


def _longformer_style_allowed_mask(
    seq_len: int, window_size: int, global_tokens: int
) -> torch.Tensor:
    allowed = _local_allowed_mask(seq_len, window_size)
    if global_tokens > 0:
        count = min(global_tokens, seq_len)
        allowed[:, :count] = True
        allowed[:count, :] = True
    return allowed


def _build_model(model_name: str, args: LongContextArgs) -> nn.Module:
    if model_name == "asam":
        return ASAMBlock(args.dim, args.num_heads, args.window_size)
    if model_name in {"transformer", "local", "longformer_style"}:
        return DenseAttentionBlock(args.dim, args.num_heads)
    raise ValueError(f"Unknown long-context model '{model_name}'.")


def _allowed_mask_for_model(
    model_name: str, seq_len: int, args: LongContextArgs
) -> torch.Tensor | None:
    if model_name == "local":
        return _local_allowed_mask(seq_len, args.window_size)
    if model_name == "longformer_style":
        return _longformer_style_allowed_mask(seq_len, args.window_size, args.global_tokens)
    return None


def _peak_memory_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return float(torch.cuda.max_memory_allocated(device) / (1024**2))
    return 0.0


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _benchmark_one(model_name: str, seq_len: int, args: LongContextArgs) -> dict[str, object]:
    device = torch.device(args.device)
    torch.manual_seed(args.seed + seq_len + len(model_name))
    model = _build_model(model_name, args).to(device)
    model.eval()
    inputs = torch.randn(args.batch_size, seq_len, args.dim, device=device)
    allowed_mask = _allowed_mask_for_model(model_name, seq_len, args)
    if allowed_mask is not None:
        allowed_mask = allowed_mask.to(device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    try:
        with torch.no_grad():
            for _ in range(args.warmup):
                if allowed_mask is None:
                    output = model(inputs)
                else:
                    output = model(inputs, allowed_mask)

            times: list[float] = []
            for _ in range(args.repeats):
                _synchronize(device)
                started = time.perf_counter()
                if allowed_mask is None:
                    output = model(inputs)
                else:
                    output = model(inputs, allowed_mask)
                _synchronize(device)
                times.append((time.perf_counter() - started) * 1000)

        finite_output_rate = float(torch.isfinite(output).float().mean().item())
        return {
            "model": model_name,
            "sequence_length": seq_len,
            "batch_size": args.batch_size,
            "dim": args.dim,
            "num_heads": args.num_heads,
            "latency_ms_mean": float(sum(times) / len(times)),
            "latency_ms_std": float(torch.tensor(times).std(unbiased=False).item()),
            "peak_memory_mb": _peak_memory_mb(device),
            "finite_output_rate": finite_output_rate,
            "diagnostic_score": finite_output_rate,
            "success": True,
        }
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() and device.type == "cuda":
            torch.cuda.empty_cache()
        return {
            "model": model_name,
            "sequence_length": seq_len,
            "batch_size": args.batch_size,
            "dim": args.dim,
            "num_heads": args.num_heads,
            "success": False,
            "error": str(exc),
        }


def run_benchmark(args: LongContextArgs) -> dict[str, object]:
    results = [
        _benchmark_one(model_name, seq_len, args)
        for seq_len in args.sequence_lengths
        for model_name in args.models
    ]
    return {
        "suite_type": "long_context",
        "benchmark_name": "lra_style_synthetic_diagnostic",
        "claim_scope": "diagnostic_only",
        "sequence_lengths": list(args.sequence_lengths),
        "models": list(args.models),
        "metric_names": METRIC_NAMES,
        "results": results,
    }


def write_benchmark_csv(path: Path, benchmark: dict[str, object]) -> None:
    rows = list(benchmark.get("results", []))
    fieldnames = [
        "model",
        "sequence_length",
        "success",
        "latency_ms_mean",
        "latency_ms_std",
        "peak_memory_mb",
        "finite_output_rate",
        "diagnostic_score",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_report(args: LongContextArgs, benchmark: dict[str, object]) -> str:
    rows = list(benchmark.get("results", []))
    successful = [row for row in rows if row.get("success")]
    lines = [
        "# Long-Context ASAM Paper Suite",
        "",
        "Diagnostic only: this is an LRA-style synthetic operator/runtime sweep, not an official LRA result.",
        "",
        "## Config",
        "",
        f"- Sequence lengths: `{args.sequence_lengths}`",
        f"- Models: `{args.models}`",
        f"- Device: `{args.device}`",
        f"- Batch size: `{args.batch_size}`",
        f"- Width / heads: `{args.dim}` / `{args.num_heads}`",
        "",
        "## Results",
        "",
        "| Model | Seq Len | Success | Latency ms | Memory MB | Finite Rate |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        latency = row.get("latency_ms_mean", "")
        memory = row.get("peak_memory_mb", "")
        finite = row.get("finite_output_rate", "")
        lines.append(
            f"| `{row.get('model')}` | {row.get('sequence_length')} | {row.get('success')} | "
            f"{latency if latency == '' else f'{latency:.4f}'} | "
            f"{memory if memory == '' else f'{memory:.4f}'} | "
            f"{finite if finite == '' else f'{finite:.4f}'} |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- These numbers validate the benchmark harness, artifact provenance, and operator behavior.",
            "- They must not be reported as official Long Range Arena results or hardware speedup claims.",
            f"- Successful rows: `{len(successful)}/{len(rows)}`.",
        ]
    )
    return "\n".join(lines)


def build_manifest_provenance(
    args: LongContextArgs,
    started_at_utc: str,
    finished_at_utc: str,
    output_paths: Iterable[Path],
    git_provenance: dict[str, object] | None = None,
    output_root: Path | None = None,
) -> dict[str, object]:
    output_dir = output_root or Path(args.output_dir)
    return {
        "argv": redact_argv(sys.argv),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "git": git_provenance or collect_git_provenance(),
        "device": args.device,
        "benchmark": {
            "name": "lra_style_synthetic_diagnostic",
            "source_kind": "synthetic",
            "claim_scope": "diagnostic_only",
            "sequence_lengths": list(args.sequence_lengths),
            "models": list(args.models),
            "metric_names": METRIC_NAMES,
        },
        "output_hashes": collect_output_hashes(output_paths, output_dir),
    }


def run_pipeline(args: LongContextArgs) -> dict[str, object]:
    started_at_utc = _utc_timestamp()
    git_provenance = collect_git_provenance()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_json = output_dir / "long_context_benchmark.json"
    benchmark_csv = output_dir / "long_context_benchmark.csv"
    benchmark_report = output_dir / "long_context_benchmark_report.md"
    suite_manifest = output_dir / "paper_suite_manifest.json"

    benchmark = run_benchmark(args)
    benchmark_json.write_text(json.dumps(benchmark, indent=2), encoding="utf-8")
    write_benchmark_csv(benchmark_csv, benchmark)
    benchmark_report.write_text(build_report(args, benchmark), encoding="utf-8")

    finished_at_utc = _utc_timestamp()
    output_paths = [benchmark_json, benchmark_csv, benchmark_report]
    manifest = {
        "suite_type": "long_context",
        "config": asdict(args),
        "resolved_config": asdict(args),
        "candidate_profile": "long_context_smoke",
        "candidate_profile_description": "CPU-runnable LRA-style synthetic diagnostic suite.",
        "output_dir": str(output_dir),
        "benchmark_json": str(benchmark_json),
        "benchmark_csv": str(benchmark_csv),
        "benchmark_report": str(benchmark_report),
        "provenance": build_manifest_provenance(
            args,
            started_at_utc,
            finished_at_utc,
            output_paths,
            git_provenance,
            output_dir,
        ),
    }
    suite_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "manifest_path": str(suite_manifest),
        "benchmark_json": str(benchmark_json),
        "benchmark_csv": str(benchmark_csv),
        "benchmark_report": str(benchmark_report),
        "benchmark_results": benchmark,
        "manifest": manifest,
    }


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args(argv: list[str] | None = None) -> LongContextArgs:
    parser = argparse.ArgumentParser(
        description="Run the paper-facing long-context ASAM diagnostic suite"
    )
    parser.add_argument("--output-dir", default=LongContextArgs.output_dir)
    parser.add_argument(
        "--sequence-lengths",
        type=_parse_int_list,
        default=list(DEFAULT_SEQUENCE_LENGTHS),
    )
    parser.add_argument("--models", type=_parse_str_list, default=list(DEFAULT_MODELS))
    parser.add_argument("--batch-size", type=int, default=LongContextArgs.batch_size)
    parser.add_argument("--dim", type=int, default=LongContextArgs.dim)
    parser.add_argument("--num-heads", type=int, default=LongContextArgs.num_heads)
    parser.add_argument("--window-size", type=int, default=LongContextArgs.window_size)
    parser.add_argument("--global-tokens", type=int, default=LongContextArgs.global_tokens)
    parser.add_argument("--warmup", type=int, default=LongContextArgs.warmup)
    parser.add_argument("--repeats", type=int, default=LongContextArgs.repeats)
    parser.add_argument("--seed", type=int, default=LongContextArgs.seed)
    parser.add_argument("--device", default=LongContextArgs.device)
    namespace = parser.parse_args(argv)
    return LongContextArgs(**vars(namespace))


def main() -> None:
    results = run_pipeline(parse_args())
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
