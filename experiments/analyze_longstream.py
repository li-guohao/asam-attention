#!/usr/bin/env python3
"""Analyze the long-task-stream artifacts (r4_*).

Computes (a) aggregate strategy tables and (b) stage-wise correlations between
the exported diagnostics (transport gap, mean abs excess, routing stability)
and stage-wise forgetting for the sinkhorn prototype runs.

Usage:
    python experiments/analyze_longstream.py
"""

import json
from pathlib import Path

import numpy as np

ARTIFACT_DIR = Path(__file__).parent.parent / "experiments" / "paper_suite"


def load(name):
    with open(ARTIFACT_DIR / name, encoding="utf-8") as handle:
        return json.load(handle)


def stage_correlations(run):
    """Pearson correlations between stage diagnostics and stage forgetting."""
    diag = run.get("theory_diagnostics", {})
    forgetting = diag.get("stage_forgetting", [])
    if not forgetting or len(forgetting) < 2:
        return None
    f = np.asarray(forgetting, dtype=float)
    result = {}
    for key, label in [
        ("stage_transport_gap", "transport_gap"),
        ("stage_mean_abs_excess", "mean_abs_excess"),
        ("stage_routing_stability_loss", "routing_stability"),
    ]:
        values = diag.get(key, [])
        if not values or len(values) != len(f):
            result[label] = None
            continue
        v = np.asarray(values, dtype=float)
        if np.std(v) == 0.0 or np.std(f) == 0.0:
            result[label] = None
            continue
        result[label] = float(np.corrcoef(v, f)[0, 1])
    return result


def summarize_artifact(name, strategies=None):
    data = load(name)
    rows = []
    correlations = []
    stem = Path(name).stem
    for run in data.get("per_run_strategies", []):
        if strategies and run["strategy"] not in strategies:
            continue
        rows.append(
            {
                "strategy": run["strategy"],
                "seed": run["seed"],
                "avg_accuracy": run["avg_accuracy"],
                "avg_forgetting": run["avg_forgetting"],
                "final_transport_gap": run.get("final_transport_gap"),
            }
        )
        if run["strategy"] == "no_adaptation":
            seed_file = ARTIFACT_DIR / f"{stem}_{run['strategy']}_seed{run['seed']}.json"
            run_full = load(seed_file.name) if seed_file.exists() else run
            corr = stage_correlations(run_full)
            if corr:
                corr["seed"] = run["seed"]
                correlations.append(corr)

    agg = {}
    for row in rows:
        key = row["strategy"]
        agg.setdefault(key, []).append(row)
    aggregated = {}
    for key, group in agg.items():
        acc = [r["avg_accuracy"] for r in group]
        forget = [r["avg_forgetting"] for r in group]
        aggregated[key] = {
            "num_runs": len(group),
            "accuracy_mean": float(np.mean(acc)),
            "accuracy_std": float(np.std(acc)),
            "forgetting_mean": float(np.mean(forget)),
            "forgetting_std": float(np.std(forget)),
        }

    pooled = {}
    for label in ["transport_gap", "mean_abs_excess", "routing_stability"]:
        values = [c[label] for c in correlations if c.get(label) is not None]
        pooled[label] = (
            {"n": len(values), "mean": float(np.mean(values)), "std": float(np.std(values))}
            if values
            else None
        )
    return {"artifact": name, "aggregated": aggregated, "stage_correlations": correlations, "pooled": pooled}


def main():
    results = [
        summarize_artifact("r4_dbpedia_longstream.json"),
        summarize_artifact("r4_cifar10_longstream.json"),
    ]
    # no_transport single-seed runs (aggregate manually).
    for base in ["r4_dbpedia_notransport", "r4_cifar10_notransport"]:
        runs = []
        for seed in (42, 43, 44):
            data = load(f"{base}_seed{seed}.json")
            runs.append(
                {
                    "seed": seed,
                    "avg_accuracy": data["avg_accuracy"],
                    "avg_forgetting": data["avg_forgetting"],
                }
            )
        acc = [r["avg_accuracy"] for r in runs]
        forget = [r["avg_forgetting"] for r in runs]
        results.append(
            {
                "artifact": base,
                "aggregated": {
                    "no_transport": {
                        "num_runs": len(runs),
                        "accuracy_mean": float(np.mean(acc)),
                        "accuracy_std": float(np.std(acc)),
                        "forgetting_mean": float(np.mean(forget)),
                        "forgetting_std": float(np.std(forget)),
                    }
                },
                "stage_correlations": [],
                "pooled": {},
            }
        )
    baselines = load("r4_dbpedia_baselines.json")
    bl = {
        "artifact": "r4_dbpedia_baselines.json",
        "aggregated": {
            key: {
                "num_runs": int(val["num_runs"]),
                "accuracy_mean": val["accuracy_mean"],
                "accuracy_std": val["accuracy_std"],
                "forgetting_mean": val["forgetting_mean"],
                "forgetting_std": val["forgetting_std"],
            }
            for key, val in baselines["methods"].items()
        },
        "stage_correlations": [],
        "pooled": {},
    }
    results.append(bl)

    out = ARTIFACT_DIR / "r4_analysis.json"
    with open(out, "w", encoding="utf-8") as handle:
        json.dump({"datasets": results, "config": {"stage_corr_method": "pearson"}}, handle, indent=2)

    for item in results:
        print(f"\n=== {item['artifact']} ===")
        for key, agg in item["aggregated"].items():
            print(
                f"  {key:14s} acc={agg['accuracy_mean']:.4f}+-{agg['accuracy_std']:.4f} "
                f"forget={agg['forgetting_mean']:.4f}+-{agg['forgetting_std']:.4f} n={agg['num_runs']}"
            )
        if item["pooled"]:
            for label, val in item["pooled"].items():
                print(f"  corr({label}, forgetting): {val}")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
