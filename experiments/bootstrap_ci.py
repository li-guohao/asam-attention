#!/usr/bin/env python3
"""Bootstrap 95% confidence intervals for key paper differences.

Reads per-seed values from the canonical artifacts and reports percentile
intervals for (a) Table 1: prototype vs task-conditioned accuracy/forgetting,
and (b) Table 3: prototype vs fine_tune / ER accuracy/forgetting.

Usage:
    python experiments/bootstrap_ci.py
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
ARTIFACT_DIR = REPO_ROOT / "experiments" / "paper_suite"


def load(name):
    with open(ARTIFACT_DIR / name, encoding="utf-8") as handle:
        return json.load(handle)


def mean_ci(values, n_boot=2000, seed=0):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = [
        float(np.mean(rng.choice(values, size=values.size, replace=True)))
        for _ in range(n_boot)
    ]
    return float(np.mean(values)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def diff_ci(values_a, values_b, n_boot=2000, seed=0):
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    rng = np.random.default_rng(seed)
    boot = [
        float(
            np.mean(rng.choice(a, size=a.size, replace=True))
            - np.mean(rng.choice(b, size=b.size, replace=True))
        )
        for _ in range(n_boot)
    ]
    return float(np.mean(boot)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main():
    table1 = load("r2_agnews_bpe_3ep.json")
    table3 = load("r3_baseline_comparison.json")

    t1_by_name = {row["strategy"]: row for row in table1["aggregated_strategies"]}
    task1 = [r["avg_forgetting"] for r in table1["per_run_strategies"] if r["strategy"] == "task_routing"]
    proto1 = [r["avg_forgetting"] for r in table1["per_run_strategies"] if r["strategy"] == "no_adaptation"]
    task1_acc = [r["avg_accuracy"] for r in table1["per_run_strategies"] if r["strategy"] == "task_routing"]
    proto1_acc = [r["avg_accuracy"] for r in table1["per_run_strategies"] if r["strategy"] == "no_adaptation"]

    t3 = table3["per_seed"]
    proto3_forget = [r["avg_forgetting"] for r in t3["prototype"]]
    proto3_acc = [r["avg_accuracy"] for r in t3["prototype"]]
    fine3_forget = [r["avg_forgetting"] for r in t3["fine_tune"]]
    fine3_acc = [r["avg_accuracy"] for r in t3["fine_tune"]]
    er3_forget = [r["avg_forgetting"] for r in t3["er"]]
    er3_acc = [r["avg_accuracy"] for r in t3["er"]]

    def entry(name, a, b=None, metric="diff"):
        if b is None:
            mean, low, high = mean_ci(a)
            return {"comparison": name, "metric": metric, "mean": mean, "ci_low": low, "ci_high": high, "n": len(a)}
        mean, low, high = diff_ci(a, b)
        return {
            "comparison": name,
            "metric": metric,
            "mean": mean,
            "ci_low": low,
            "ci_high": high,
            "n_a": len(a),
            "n_b": len(b),
        }

    results = {
        "config": {
            "n_boot": 2000,
            "seed": 0,
            "table1_artifact": "r2_agnews_bpe_3ep.json",
            "table3_artifact": "r3_baseline_comparison.json",
        },
        "entries": [
            entry("table1 forgetting prototype - task", proto1, task1),
            entry("table1 accuracy prototype - task", proto1_acc, task1_acc),
            entry("table3 forgetting prototype - fine_tune", proto3_forget, fine3_forget),
            entry("table3 accuracy prototype - fine_tune", proto3_acc, fine3_acc),
            entry("table3 forgetting prototype - ER", proto3_forget, er3_forget),
            entry("table3 accuracy prototype - ER", proto3_acc, er3_acc),
            entry("table3 forgetting fine_tune (CI only)", fine3_forget),
            entry("table3 accuracy prototype (CI only)", proto3_acc),
        ],
    }

    output_path = ARTIFACT_DIR / "bootstrap_ci.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    for item in results["entries"]:
        print(
            f"{item['comparison']:42s} mean={item['mean']:+.4f} "
            f"95% CI=[{item['ci_low']:+.4f}, {item['ci_high']:+.4f}]"
        )
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    sys.exit(main())
