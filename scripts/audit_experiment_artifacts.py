"""Audit paper-suite experiment artifacts for reproducibility risks.

The script is intentionally read-only: it scans JSON artifacts, emits a JSON
summary, and labels findings as blocking or suspicious risks without making
claims about intent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable

CURRENT = "CURRENT"
MIXED = "MIXED"
OUTDATED = "OUTDATED"

CORE_RESULT_KEYS = {
    "accuracy_matrix",
    "aggregated_strategies",
    "best_avg_accuracy",
    "stage_training_metrics",
    "prototype_lifecycle",
    "per_run_strategies",
    "avg_accuracy",
    "avg_forgetting",
    "backward_transfer",
    "lowest_avg_forgetting",
}

CONFIG_KEYS = {
    "config",
    "resolved_config",
    "candidate_profile",
    "candidate_profile_description",
    "adaptation_strategy",
    "prototype_routing_strategy",
    "transport_weight",
    "num_prototypes",
    "prototype_slots_per_task",
    "prototype_top_k",
    "seed",
    "seeds",
}

PATH_KEY_PARTS = {
    "path",
    "paths",
    "dir",
    "directory",
    "json",
    "plot",
    "report",
    "csv",
    "table",
    "tex",
    "png",
    "md",
}

TEXT_ARTIFACT_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".tex",
    ".txt",
}


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _read_json(path: Path) -> tuple[Any | None, str | None]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, json.JSONDecodeError) as exc:
        return None, str(exc)


def _is_path_key(key: str) -> bool:
    lowered = key.lower()
    if lowered in CORE_RESULT_KEYS or lowered in CONFIG_KEYS:
        return False
    parts = set(lowered.replace("-", "_").split("_"))
    return bool(parts & PATH_KEY_PARTS)


def _strip_path_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_path_fields(item)
            for key, item in sorted(value.items())
            if not _is_path_key(str(key))
        }
    if isinstance(value, list):
        return [_strip_path_fields(item) for item in value]
    return value


def semantic_fingerprint(payload: Any) -> str:
    if isinstance(payload, dict):
        focused = {
            key: _strip_path_fields(payload[key])
            for key in sorted(CORE_RESULT_KEYS | CONFIG_KEYS)
            if key in payload
        }
        if not focused:
            focused = _strip_path_fields(payload)
    else:
        focused = payload
    return hashlib.sha256(_json_dumps(focused).encode("utf-8")).hexdigest()


def _raw_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_hash_bytes(path: Path) -> bytes:
    content = path.read_bytes()
    if path.suffix.lower() in TEXT_ARTIFACT_SUFFIXES:
        return content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return content


def _artifact_fingerprint(path: Path) -> str:
    return hashlib.sha256(_artifact_hash_bytes(path)).hexdigest()


def _normalize_manifest_hash_key(key: str) -> str | None:
    normalized = str(key).replace("\\", "/").strip()
    posix_path = PurePosixPath(normalized)
    windows_path = PureWindowsPath(normalized)
    if (
        not normalized
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or ".." in posix_path.parts
    ):
        return None
    return posix_path.as_posix()


def _manifest_relative_path(suite_path: Path, normalized_key: str) -> Path:
    return suite_path.joinpath(*PurePosixPath(normalized_key).parts)


def _is_within_directory(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def _collect_values(value: Any, keys: Iterable[str]) -> set[str]:
    wanted = set(keys)
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in wanted and isinstance(item, (str, int, float, bool)):
                found.add(str(item))
            found.update(_collect_values(item, wanted))
    elif isinstance(value, list):
        for item in value:
            found.update(_collect_values(item, wanted))
    elif isinstance(value, (str, int, float, bool)):
        found.add(str(value))
    return found


def _load_named_json(suite_path: Path, name: str) -> Any | None:
    path = suite_path / name
    if not path.exists():
        return None
    payload, _error = _read_json(path)
    return payload


VALID_DATASET_SOURCE_KINDS = {"huggingface", "fallback_synthetic"}


def _dataset_provenance_schema_issues(provenance: Any, prefix: str) -> list[str]:
    missing: list[str] = []
    if not isinstance(provenance, dict):
        return [prefix]

    for split_name in ["train", "val"]:
        split_provenance = provenance.get(split_name)
        split_prefix = f"{prefix}.{split_name}"
        if not isinstance(split_provenance, dict):
            missing.append(split_prefix)
            continue

        source_kind = split_provenance.get("source_kind")
        if source_kind not in VALID_DATASET_SOURCE_KINDS:
            missing.append(f"{split_prefix}.source_kind")

        split = split_provenance.get("split")
        if not isinstance(split, str) or not split:
            missing.append(f"{split_prefix}.split")

        sample_count = split_provenance.get("sample_count")
        if not isinstance(sample_count, int) or sample_count < 0:
            missing.append(f"{split_prefix}.sample_count")

        if "max_samples" not in split_provenance:
            missing.append(f"{split_prefix}.max_samples")

    return missing


def _has_strict_provenance(manifest: dict[str, Any]) -> tuple[bool, list[str]]:
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        return False, ["provenance"]

    missing: list[str] = []
    for key in [
        "argv",
        "python_version",
        "torch_version",
        "started_at_utc",
        "finished_at_utc",
        "git",
        "dataset",
        "output_hashes",
    ]:
        if not provenance.get(key):
            missing.append(f"provenance.{key}")

    git = provenance.get("git")
    if not isinstance(git, dict):
        missing.append("provenance.git")
    else:
        if not git.get("commit") or git.get("commit") == "unknown":
            missing.append("provenance.git.commit")
        if not isinstance(git.get("dirty"), bool):
            missing.append("provenance.git.dirty")

    dataset = provenance.get("dataset")
    if not isinstance(dataset, dict):
        missing.append("provenance.dataset")
    else:
        for key in ["name", "classes_per_task", "seed", "num_seeds"]:
            if dataset.get(key) is None:
                missing.append(f"provenance.dataset.{key}")
        benchmark_provenance = dataset.get("benchmark_provenance")
        missing.extend(
            _dataset_provenance_schema_issues(
                benchmark_provenance,
                "provenance.dataset.benchmark_provenance",
            )
        )

    output_hashes = provenance.get("output_hashes")
    if not isinstance(output_hashes, dict):
        missing.append("provenance.output_hashes")
    else:
        for key in [
            "continual_benchmark.json",
            "continual_ablation.json",
            "continual_operator_ablation.json",
        ]:
            value = output_hashes.get(key)
            if not isinstance(value, str) or len(value) != 64:
                missing.append(f"provenance.output_hashes.{key}")

    return not missing, sorted(set(missing))


def _audit_manifest_output_hashes(
    suite_path: Path, manifest: dict[str, Any]
) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        return issues

    output_hashes = provenance.get("output_hashes")
    if not isinstance(output_hashes, dict):
        return issues

    covered_json_paths: set[str] = set()
    for raw_key, expected_hash in sorted(output_hashes.items()):
        normalized_key = _normalize_manifest_hash_key(str(raw_key))
        if normalized_key is None:
            issues.append(
                {
                    "severity": "blocking",
                    "suite": str(suite_path),
                    "path": str(raw_key),
                    "message": "manifest hash key is not a relative suite path",
                }
            )
            continue

        artifact_path = _manifest_relative_path(suite_path, normalized_key)
        if not _is_within_directory(artifact_path, suite_path):
            issues.append(
                {
                    "severity": "blocking",
                    "suite": str(suite_path),
                    "path": str(raw_key),
                    "message": "manifest hash key is not a relative suite path",
                }
            )
            continue

        if artifact_path.suffix.lower() == ".json":
            covered_json_paths.add(normalized_key)
        if not artifact_path.is_file():
            issues.append(
                {
                    "severity": "blocking",
                    "suite": str(suite_path),
                    "path": str(artifact_path),
                    "message": f"hashed artifact is missing: {normalized_key}",
                }
            )
            continue

        actual_hash = _artifact_fingerprint(artifact_path)
        if actual_hash != expected_hash:
            issues.append(
                {
                    "severity": "blocking",
                    "suite": str(suite_path),
                    "path": str(artifact_path),
                    "message": f"hash mismatch for {normalized_key}",
                }
            )

    for json_path in sorted(suite_path.rglob("*.json")):
        relative_json = json_path.relative_to(suite_path).as_posix()
        if relative_json == "paper_suite_manifest.json":
            continue
        if relative_json not in covered_json_paths:
            issues.append(
                {
                    "severity": "blocking",
                    "suite": str(suite_path),
                    "path": str(json_path),
                    "message": f"JSON artifact is not covered by manifest output_hashes: {relative_json}",
                }
            )

    return issues


def _audit_benchmark_dataset_provenance(
    suite_path: Path, manifest: dict[str, Any]
) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    benchmark_path = suite_path / "continual_benchmark.json"
    if not benchmark_path.exists():
        return issues

    benchmark = _load_named_json(suite_path, "continual_benchmark.json")
    if not isinstance(benchmark, dict):
        issues.append(
            {
                "severity": "blocking",
                "suite": str(suite_path),
                "path": str(benchmark_path),
                "message": "continual_benchmark.json is missing or not a JSON object",
            }
        )
        return issues

    dataset = manifest.get("provenance", {}).get("dataset", {})
    manifest_provenance = dataset.get("benchmark_provenance") if isinstance(dataset, dict) else None
    benchmark_provenance = benchmark.get("dataset_provenance")
    if not isinstance(benchmark_provenance, dict):
        issues.append(
            {
                "severity": "blocking",
                "suite": str(suite_path),
                "path": str(benchmark_path),
                "message": "continual_benchmark.json.dataset_provenance is missing",
            }
        )
        return issues

    benchmark_schema_issues = _dataset_provenance_schema_issues(
        benchmark_provenance,
        "continual_benchmark.json.dataset_provenance",
    )
    if benchmark_schema_issues:
        issues.append(
            {
                "severity": "blocking",
                "suite": str(suite_path),
                "path": str(benchmark_path),
                "message": "benchmark dataset provenance schema is invalid: "
                + ", ".join(benchmark_schema_issues),
            }
        )
    if benchmark_provenance != manifest_provenance:
        issues.append(
            {
                "severity": "blocking",
                "suite": str(suite_path),
                "path": str(benchmark_path),
                "message": (
                    "continual_benchmark.json.dataset_provenance does not match "
                    "provenance.dataset.benchmark_provenance"
                ),
            }
        )
    return issues


def _rate_suite_schema(suite_path: Path) -> tuple[str, list[dict[str, str]], list[dict[str, str]]]:
    blocking: list[dict[str, str]] = []
    suspicious: list[dict[str, str]] = []
    manifest = _load_named_json(suite_path, "paper_suite_manifest.json")

    if not isinstance(manifest, dict):
        blocking.append(
            {
                "severity": "blocking",
                "suite": str(suite_path),
                "message": "paper_suite_manifest.json is missing or not a JSON object",
            }
        )
        return OUTDATED, blocking, suspicious

    required_manifest_keys = [
        "resolved_config",
        "candidate_profile",
        "candidate_profile_description",
    ]
    missing = [key for key in required_manifest_keys if not manifest.get(key)]
    if missing:
        blocking.append(
            {
                "severity": "blocking",
                "suite": str(suite_path),
                "message": "manifest is missing current provenance fields: " + ", ".join(missing),
            }
        )
        return OUTDATED, blocking, suspicious

    has_provenance, missing_provenance = _has_strict_provenance(manifest)
    if not has_provenance:
        blocking.append(
            {
                "severity": "blocking",
                "suite": str(suite_path),
                "message": "manifest is missing strict provenance fields: "
                + ", ".join(missing_provenance),
            }
        )
        return OUTDATED, blocking, suspicious

    blocking.extend(_audit_manifest_output_hashes(suite_path, manifest))
    blocking.extend(_audit_benchmark_dataset_provenance(suite_path, manifest))

    ablation = _load_named_json(suite_path, "continual_ablation.json")
    operator = _load_named_json(suite_path, "continual_operator_ablation.json")
    ablation_strategies = _collect_values(ablation, ["strategy", "name", "adaptation_strategy"])
    operator_strategies = _collect_values(
        operator, ["strategy", "name", "prototype_routing_strategy"]
    )

    if "dual_transport" not in ablation_strategies:
        suspicious.append(
            {
                "severity": "suspicious",
                "suite": str(suite_path),
                "message": "ablation artifact lacks current strategy dual_transport",
            }
        )
    if "masked_sinkhorn_topk" not in operator_strategies:
        suspicious.append(
            {
                "severity": "suspicious",
                "suite": str(suite_path),
                "message": "operator artifact lacks current strategy masked_sinkhorn_topk",
            }
        )

    if blocking:
        return OUTDATED, blocking, suspicious
    if suspicious:
        return MIXED, blocking, suspicious
    return CURRENT, blocking, suspicious


def _discover_default_suites(root: Path) -> list[Path]:
    experiments = root / "experiments"
    return sorted(path for path in experiments.glob("paper_suite*") if path.is_dir())


def _normalize_paths(paths: Iterable[str | Path] | None) -> list[Path]:
    if paths:
        return [Path(path) for path in paths]
    return _discover_default_suites(Path.cwd())


def _overall_rating(suites: list[dict[str, Any]]) -> str:
    ratings = {suite["schema_provenance_rating"] for suite in suites}
    if OUTDATED in ratings:
        return OUTDATED
    if MIXED in ratings:
        return MIXED
    return CURRENT


def audit_paths(paths: Iterable[str | Path] | None = None) -> dict[str, Any]:
    suite_paths = _normalize_paths(paths)
    json_records: list[dict[str, Any]] = []
    blocking_issues: list[dict[str, str]] = []
    suspicious_issues: list[dict[str, str]] = []
    suite_summaries: list[dict[str, Any]] = []

    for suite_path in suite_paths:
        json_paths = sorted(suite_path.rglob("*.json")) if suite_path.exists() else []
        rating, suite_blocking, suite_suspicious = _rate_suite_schema(suite_path)
        blocking_issues.extend(suite_blocking)
        suspicious_issues.extend(suite_suspicious)
        suite_summaries.append(
            {
                "path": str(suite_path),
                "json_file_count": len(json_paths),
                "schema_provenance_rating": rating,
            }
        )
        for json_path in json_paths:
            payload, error = _read_json(json_path)
            record = {
                "path": str(json_path),
                "suite": str(suite_path),
                "raw_fingerprint": _raw_fingerprint(json_path),
                "semantic_fingerprint": None,
            }
            if error is not None:
                blocking_issues.append(
                    {
                        "severity": "blocking",
                        "suite": str(suite_path),
                        "path": str(json_path),
                        "message": "JSON artifact could not be parsed: " + error,
                    }
                )
            else:
                record["semantic_fingerprint"] = semantic_fingerprint(payload)
            json_records.append(record)

    raw_groups = defaultdict(list)
    semantic_groups = defaultdict(list)
    for record in json_records:
        raw_groups[record["raw_fingerprint"]].append(record)
        if record["semantic_fingerprint"]:
            semantic_groups[record["semantic_fingerprint"]].append(record)

    raw_duplicate_count = sum(len(group) - 1 for group in raw_groups.values() if len(group) > 1)
    if raw_duplicate_count:
        suspicious_issues.append(
            {
                "severity": "suspicious",
                "message": f"{raw_duplicate_count} raw duplicate JSON artifact(s) found",
            }
        )

    semantic_duplicate_groups = [
        {
            "fingerprint": fingerprint,
            "files": [{"path": record["path"], "suite": record["suite"]} for record in records],
        }
        for fingerprint, records in sorted(semantic_groups.items())
        if len(records) > 1
    ]
    if semantic_duplicate_groups:
        suspicious_issues.append(
            {
                "severity": "suspicious",
                "message": f"{len(semantic_duplicate_groups)} semantic duplicate group(s) found",
            }
        )

    return {
        "suites": suite_summaries,
        "json_file_count": len(json_records),
        "raw_duplicate_count": raw_duplicate_count,
        "semantic_duplicate_groups": semantic_duplicate_groups,
        "schema_provenance_rating": _overall_rating(suite_summaries),
        "blocking_issue_count": len(blocking_issues),
        "suspicious_issue_count": len(suspicious_issues),
        "blocking_issues": blocking_issues,
        "suspicious_issues": suspicious_issues,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only audit of experiment JSON artifacts under experiments/paper_suite*."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Suite directories to scan. Defaults to experiments/paper_suite* under the current directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = audit_paths(args.paths)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 1 if summary["blocking_issue_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
