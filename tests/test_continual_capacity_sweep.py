"""
Tests for continual capacity sweep runner.
"""

import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import experiments.run_continual_capacity_sweep as capacity_sweep
from experiments.run_continual_capacity_sweep import CapacitySweepArgs, parse_int_grid, run_capacity_sweep



def test_parse_int_grid_removes_duplicates_and_sorts():
    assert parse_int_grid("4,2,2,1") == [1, 2, 4]



def test_capacity_sweep_runner_exports_summary_artifacts(tmp_path, monkeypatch):
    output_json = tmp_path / "capacity_sweep.json"

    def fake_run_benchmark(args):
        if args.routing_mode == "task":
            avg_accuracy = 0.52
            avg_forgetting = -0.04
            resolved_num_prototypes = 0
            resolved_top_k = 0
        else:
            resolved_num_prototypes = max(1, 2 * args.prototype_slots_per_task)
            resolved_top_k = min(args.prototype_top_k, max(1, resolved_num_prototypes - 1))
            avg_accuracy = 0.50 + 0.01 * args.prototype_slots_per_task - 0.005 * args.prototype_top_k
            avg_forgetting = -0.03 - 0.01 * (2 - args.prototype_top_k)

        payload = {
            "config": {
                "routing_mode": args.routing_mode,
                "dataset_name": args.dataset_name,
                "prototype_routing_strategy": args.prototype_routing_strategy,
            },
            "resolved_prototype_layout": {
                "num_prototypes": resolved_num_prototypes,
                "prototype_top_k": resolved_top_k,
                "prototype_slots_per_task": args.prototype_slots_per_task,
            },
            "avg_accuracy": avg_accuracy,
            "avg_forgetting": avg_forgetting,
            "backward_transfer": -avg_forgetting,
            "theory_diagnostics": {
                "stage_transport_gap": [0.0, max(0.0, 0.02 * args.prototype_top_k if args.routing_mode == "prototype" else 0.0)],
                "stage_routing_stability_loss": [0.0, max(0.0, 0.1 * args.prototype_slots_per_task if args.routing_mode == "prototype" else 0.0)],
            },
        }
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr(capacity_sweep, "run_benchmark", fake_run_benchmark)

    args = CapacitySweepArgs(
        prototype_slots_grid="2,4",
        prototype_topk_grid="1,2",
        num_seeds=2,
        output_json=str(output_json),
    )
    results = run_capacity_sweep(args)

    assert output_json.exists()
    assert len(results["aggregated_configs"]) == 5
    assert results["best_prototype_accuracy"]["config_name"] == "slots4_topk1"
    assert results["best_prototype_forgetting"]["config_name"] == "slots2_topk1"

    persisted = json.loads(output_json.read_text(encoding="utf-8"))
    assert persisted["best_prototype_accuracy"]["config_name"] == "slots4_topk1"
    assert Path(persisted["table_path"]).exists()
    assert Path(persisted["csv_path"]).exists()
    assert Path(persisted["plot_path"]).exists()
    assert Path(persisted["report_path"]).exists()

    report_text = Path(persisted["report_path"]).read_text(encoding="utf-8")
    table_text = Path(persisted["table_path"]).read_text(encoding="utf-8")
    assert "# Continual Capacity Sweep" in report_text
    assert "Task baseline avg accuracy" in report_text
    assert "| Config | Mode | Slots/Task |" in table_text
    assert "slots4_topk1" in table_text
    assert "task_routing" in table_text
