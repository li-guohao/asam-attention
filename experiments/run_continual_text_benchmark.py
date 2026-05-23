"""
Real-text continual benchmark for Continual ASAM.
"""

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.train_continual_asam import (
    ContinualTextClassifier,
    collect_prototype_diagnostics,
    compute_continual_metrics,
    set_seed,
)


def _load_local_module(module_name: str, relative_path: str):
    module_path = Path(__file__).parent.parent / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_text_dataset_module = _load_local_module("asam_text_dataset_module", "datasets/text_dataset.py")
get_continual_dataloaders = _text_dataset_module.get_continual_dataloaders


@dataclass
class RealBenchmarkArgs:
    dataset_name: str = "split_ag_news"
    classes_per_task: int = 2
    max_length: int = 256
    batch_size: int = 8
    max_train_samples: Optional[int] = 128
    max_val_samples: Optional[int] = 64
    num_workers: int = 0
    dim: int = 64
    num_heads: int = 4
    num_layers: int = 1
    top_k_patterns: int = 2
    routing_mode: str = "prototype"
    prototype_routing_strategy: str = "sinkhorn_topk"
    num_prototypes: int = 0
    prototype_slots_per_task: int = 2
    prototype_top_k: int = 2
    learning_rate: float = 3e-4
    epochs_per_task: int = 1
    overlap_weight: float = 0.1
    stability_weight: float = 0.1
    balance_weight: float = 0.05
    diversity_weight: float = 0.05
    transport_weight: float = 0.05
    replay_batch_size: int = 4
    prototype_reset_threshold: float = 0.01
    prototype_split_threshold: float = 0.20
    prototype_noise_scale: float = 0.05
    prototype_merge_threshold: float = 0.9
    prototype_merge_usage_threshold: float = 0.1
    prototype_masked_sinkhorn_candidate_k: int = 0
    prototype_birkhoff_transport_strength: float = 0.02
    prototype_birkhoff_adaptive_gate: bool = True
    prototype_birkhoff_gap_target: float = 0.03
    prototype_birkhoff_max_applied_offdiag_mass: float = 0.006
    prototype_birkhoff_gap_tolerance: float = 0.0
    prototype_birkhoff_min_effective_strength: float = 1e-4
    prototype_prior_strength: float = 1.0
    prototype_capacity_blend: float = 0.5
    prototype_masked_sinkhorn_capacity_bias: float = 0.0
    prototype_relocation_strength: float = 0.75
    adaptive_hyperparameters: bool = True
    adaptation_strategy: str = "meta_secant"
    adaptation_warmup_stages: int = 1
    target_stage_forgetting: float = 0.05
    prior_strength_step: float = 0.25
    capacity_blend_step: float = 0.10
    relocation_strength_step: float = 0.10
    meta_transport_weight: float = 0.5
    meta_transport_loss_weight: float = 0.25
    meta_excess_weight: float = 0.25
    meta_routing_weight: float = 0.25
    meta_secant_mix: float = 0.25
    meta_secant_eps: float = 1e-3
    transport_weight_step: float = 0.5
    prototype_topk_step: float = 2.5
    dual_transport_trend_weight: float = 0.5
    dual_transport_gap_weight: float = 0.25
    dual_transport_loss_weight: float = 0.25
    dual_topk_min_stages: int = 3
    dual_topk_threshold: float = 0.35
    device: str = "cpu"
    seed: int = 42
    output_json: Optional[str] = None


class ReplayBuffer:
    def __init__(self):
        self.storage: List[tuple] = []

    def add_batch(self, inputs: torch.Tensor, labels: torch.Tensor, task_ids: torch.Tensor):
        self.storage.append((inputs.cpu(), labels.cpu(), task_ids.cpu()))

    def sample(self, batch_size: int, device: torch.device):
        if batch_size <= 0 or not self.storage:
            return None
        flat_inputs = torch.cat([item[0] for item in self.storage], dim=0)
        flat_labels = torch.cat([item[1] for item in self.storage], dim=0)
        flat_task_ids = torch.cat([item[2] for item in self.storage], dim=0)
        sample_count = min(batch_size, flat_inputs.size(0))
        indices = torch.randperm(flat_inputs.size(0))[:sample_count]
        return (
            flat_inputs[indices].to(device),
            flat_labels[indices].to(device),
            flat_task_ids[indices].to(device),
        )


def initialize_task_transport_weights(args: RealBenchmarkArgs, num_tasks: int):
    base_weight = float(args.transport_weight)
    args.task_transport_weights = [base_weight for _ in range(num_tasks)]


def get_task_transport_weights(args: RealBenchmarkArgs, num_tasks: Optional[int] = None) -> List[float]:
    task_weights = list(getattr(args, "task_transport_weights", []))
    if num_tasks is None:
        return task_weights
    if len(task_weights) < num_tasks:
        task_weights.extend([float(args.transport_weight)] * (num_tasks - len(task_weights)))
    return task_weights[:num_tasks]


def compute_effective_transport_weight(args: RealBenchmarkArgs, task_ids: Optional[torch.Tensor] = None) -> float:
    task_weights = getattr(args, "task_transport_weights", None)
    if not task_weights or task_ids is None or args.adaptation_strategy != "dual_transport":
        return float(args.transport_weight)
    task_weight_tensor = torch.tensor(task_weights, dtype=torch.float32, device=task_ids.device)
    sampled_weights = task_weight_tensor[task_ids.long().clamp(min=0, max=task_weight_tensor.numel() - 1)]
    return float(sampled_weights.mean().item())


def compute_task_conditioned_transport_penalty(
    args: RealBenchmarkArgs,
    info: Dict[str, torch.Tensor],
    task_ids: Optional[torch.Tensor] = None,
):
    transport_loss = info.get("transport_loss")
    if transport_loss is None:
        transport_loss = info["overlap_loss"].new_zeros(())
    effective_transport_weight = compute_effective_transport_weight(args, task_ids)
    transport_loss_per_sample = info.get("transport_loss_per_sample")
    task_weights = getattr(args, "task_transport_weights", None)
    if (
        args.adaptation_strategy == "dual_transport"
        and task_ids is not None
        and task_weights
        and transport_loss_per_sample is not None
    ):
        task_weight_tensor = torch.tensor(
            task_weights,
            dtype=transport_loss_per_sample.dtype,
            device=transport_loss_per_sample.device,
        )
        lambda_per_sample = task_weight_tensor[
            task_ids.long().clamp(min=0, max=task_weight_tensor.numel() - 1)
        ]
        weighted_transport_loss = (lambda_per_sample * transport_loss_per_sample).mean()
        effective_transport_weight = float(lambda_per_sample.mean().item())
        return weighted_transport_loss, effective_transport_weight
    return effective_transport_weight * transport_loss, effective_transport_weight


@torch.no_grad()
def evaluate_task(model: ContinualTextClassifier, data_loader, device: torch.device) -> float:
    model.eval()
    total_correct = 0
    total_samples = 0
    for inputs, labels, task_ids in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        task_ids = task_ids.to(device)
        logits, _ = model(inputs, task_ids=task_ids, return_info=False)
        predictions = logits.argmax(dim=-1)
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)
    return total_correct / max(1, total_samples)


def _safe_float(value: object) -> float:
    return float(value) if value is not None else 0.0


def compute_stage_forgetting_series(accuracy_matrix: List[List[float]], num_tasks: int) -> List[float]:
    stage_forgetting = []
    for stage in range(num_tasks):
        forgetting_terms = []
        for task_id in range(stage):
            seen_history = [accuracy_matrix[past_stage][task_id] for past_stage in range(task_id, stage + 1)]
            forgetting_terms.append(max(seen_history) - accuracy_matrix[stage][task_id])
        stage_forgetting.append(float(sum(forgetting_terms) / max(1, len(forgetting_terms))))
    return stage_forgetting


def compute_stage_task_forgetting(accuracy_matrix: List[List[float]], num_tasks: int) -> List[List[float]]:
    stage_task_forgetting = []
    for stage in range(num_tasks):
        task_forgetting = []
        for task_id in range(stage + 1):
            seen_history = [accuracy_matrix[past_stage][task_id] for past_stage in range(task_id, stage + 1)]
            task_forgetting.append(float(max(seen_history) - accuracy_matrix[stage][task_id]))
        stage_task_forgetting.append(task_forgetting)
    return stage_task_forgetting


def compute_theory_diagnostics(
    accuracy_matrix: List[List[float]],
    stage_training_metrics: List[Dict[str, float]],
    prototype_lifecycle: List[Dict[str, object]],
    prototype_diagnostics: List[Dict[str, object]],
) -> Dict[str, object]:
    stage_forgetting = compute_stage_forgetting_series(accuracy_matrix, len(accuracy_matrix))
    stage_task_forgetting = compute_stage_task_forgetting(accuracy_matrix, len(accuracy_matrix))
    stage_avg_accuracy = [float(sum(row[: stage + 1]) / max(1, stage + 1)) for stage, row in enumerate(accuracy_matrix)]
    stage_transport_gap = [_safe_float(item.get("mean_transport_gap")) for item in prototype_lifecycle]
    stage_max_transport_gap = [_safe_float(item.get("max_transport_gap")) for item in prototype_lifecycle]
    stage_routing_stability = [_safe_float(item.get("routing_stability_loss")) for item in stage_training_metrics]
    stage_stability_loss = [_safe_float(item.get("stability_loss")) for item in stage_training_metrics]
    stage_overlap_loss = [_safe_float(item.get("overlap_loss")) for item in stage_training_metrics]
    stage_transport_loss = [_safe_float(item.get("transport_loss")) for item in stage_training_metrics]
    stage_weighted_transport_loss = [
        _safe_float(item.get("weighted_transport_loss", item.get("transport_loss")))
        for item in stage_training_metrics
    ]
    stage_candidate_support_residual = [
        _safe_float(item.get("candidate_support_residual"))
        for item in stage_training_metrics
    ]
    stage_support_projection_residual = [
        _safe_float(item.get("support_projection_residual"))
        for item in stage_training_metrics
    ]
    stage_support_residual_delta = [
        _safe_float(item.get("support_residual_delta"))
        for item in stage_training_metrics
    ]
    stage_target_capacity_residual = [
        _safe_float(item.get("target_capacity_residual"))
        for item in stage_training_metrics
    ]
    stage_effective_capacity_residual = [
        _safe_float(item.get("effective_capacity_residual"))
        for item in stage_training_metrics
    ]
    stage_support_density = [_safe_float(item.get("support_density")) for item in stage_training_metrics]
    stage_support_size = [_safe_float(item.get("support_size")) for item in stage_training_metrics]
    stage_support_active_prototypes = [
        _safe_float(item.get("support_active_prototypes"))
        for item in stage_training_metrics
    ]
    stage_support_weight_leakage = [
        _safe_float(item.get("support_weight_leakage"))
        for item in stage_training_metrics
    ]
    stage_capacity_bias_selection_rate = [
        _safe_float(item.get("capacity_bias_selection_rate"))
        for item in stage_training_metrics
    ]
    stage_merge_count = [_safe_float(item.get("merge_count")) for item in prototype_lifecycle]
    stage_birkhoff_base_strength = [_safe_float(item.get("birkhoff_base_strength")) for item in prototype_lifecycle]
    stage_birkhoff_strength = [_safe_float(item.get("birkhoff_strength")) for item in prototype_lifecycle]
    stage_birkhoff_gate_factor = [_safe_float(item.get("birkhoff_gate_factor")) for item in prototype_lifecycle]
    stage_birkhoff_offdiag_mass = [_safe_float(item.get("birkhoff_offdiag_mass")) for item in prototype_lifecycle]
    stage_birkhoff_applied_offdiag_mass = [
        _safe_float(item.get("birkhoff_applied_offdiag_mass"))
        for item in prototype_lifecycle
    ]
    stage_birkhoff_row_error = [_safe_float(item.get("birkhoff_row_error")) for item in prototype_lifecycle]
    stage_birkhoff_col_error = [_safe_float(item.get("birkhoff_col_error")) for item in prototype_lifecycle]
    stage_birkhoff_gap_delta = [_safe_float(item.get("birkhoff_gap_delta")) for item in prototype_lifecycle]
    stage_mean_abs_excess = []
    stage_routing_entropy = []
    stage_task_transport_gap = []
    stage_task_max_transport_gap = []
    stage_task_transport_loss = []
    for stage in prototype_diagnostics:
        excess_layers = stage.get("layer_excess_ema", [])
        flattened_excess = [abs(float(value)) for layer in excess_layers for value in layer]
        stage_mean_abs_excess.append(float(np.mean(flattened_excess)) if flattened_excess else 0.0)
        entropies = stage.get("task_routing_entropy", [])
        stage_routing_entropy.append(float(np.mean(entropies)) if entropies else 0.0)
        stage_task_transport_gap.append([float(value) for value in stage.get("task_transport_gap", [])])
        stage_task_max_transport_gap.append([float(value) for value in stage.get("task_max_transport_gap", [])])
        stage_task_transport_loss.append([float(value) for value in stage.get("task_transport_loss", [])])

    def correlation(series_a: List[float], series_b: List[float]) -> Optional[float]:
        if len(series_a) != len(series_b) or len(series_a) < 2:
            return None
        array_a = np.array(series_a, dtype=float)
        array_b = np.array(series_b, dtype=float)
        if np.allclose(array_a, array_a[0]) or np.allclose(array_b, array_b[0]):
            return None
        return float(np.corrcoef(array_a, array_b)[0, 1])

    return {
        "stage_forgetting": stage_forgetting,
        "stage_avg_accuracy": stage_avg_accuracy,
        "stage_task_forgetting": stage_task_forgetting,
        "stage_transport_gap": stage_transport_gap,
        "stage_max_transport_gap": stage_max_transport_gap,
        "stage_routing_stability_loss": stage_routing_stability,
        "stage_stability_loss": stage_stability_loss,
        "stage_overlap_loss": stage_overlap_loss,
        "stage_transport_loss": stage_transport_loss,
        "stage_weighted_transport_loss": stage_weighted_transport_loss,
        "stage_candidate_support_residual": stage_candidate_support_residual,
        "stage_support_projection_residual": stage_support_projection_residual,
        "stage_support_residual_delta": stage_support_residual_delta,
        "stage_target_capacity_residual": stage_target_capacity_residual,
        "stage_effective_capacity_residual": stage_effective_capacity_residual,
        "stage_support_density": stage_support_density,
        "stage_support_size": stage_support_size,
        "stage_support_active_prototypes": stage_support_active_prototypes,
        "stage_support_weight_leakage": stage_support_weight_leakage,
        "stage_capacity_bias_selection_rate": stage_capacity_bias_selection_rate,
        "stage_task_transport_gap": stage_task_transport_gap,
        "stage_task_max_transport_gap": stage_task_max_transport_gap,
        "stage_task_transport_loss": stage_task_transport_loss,
        "stage_merge_count": stage_merge_count,
        "stage_birkhoff_base_strength": stage_birkhoff_base_strength,
        "stage_birkhoff_strength": stage_birkhoff_strength,
        "stage_birkhoff_gate_factor": stage_birkhoff_gate_factor,
        "stage_birkhoff_offdiag_mass": stage_birkhoff_offdiag_mass,
        "stage_birkhoff_applied_offdiag_mass": stage_birkhoff_applied_offdiag_mass,
        "stage_birkhoff_row_error": stage_birkhoff_row_error,
        "stage_birkhoff_col_error": stage_birkhoff_col_error,
        "stage_birkhoff_gap_delta": stage_birkhoff_gap_delta,
        "stage_mean_abs_excess": stage_mean_abs_excess,
        "stage_routing_entropy": stage_routing_entropy,
        "forgetting_correlations": {
            "routing_stability": correlation(stage_forgetting, stage_routing_stability),
            "stability_loss": correlation(stage_forgetting, stage_stability_loss),
            "transport_gap": correlation(stage_forgetting, stage_transport_gap),
            "transport_loss": correlation(stage_forgetting, stage_transport_loss),
            "weighted_transport_loss": correlation(stage_forgetting, stage_weighted_transport_loss),
            "candidate_support_residual": correlation(stage_forgetting, stage_candidate_support_residual),
            "support_projection_residual": correlation(stage_forgetting, stage_support_projection_residual),
            "support_residual_delta": correlation(stage_forgetting, stage_support_residual_delta),
            "target_capacity_residual": correlation(stage_forgetting, stage_target_capacity_residual),
            "effective_capacity_residual": correlation(stage_forgetting, stage_effective_capacity_residual),
            "support_density": correlation(stage_forgetting, stage_support_density),
            "support_size": correlation(stage_forgetting, stage_support_size),
            "support_active_prototypes": correlation(stage_forgetting, stage_support_active_prototypes),
            "support_weight_leakage": correlation(stage_forgetting, stage_support_weight_leakage),
            "capacity_bias_selection_rate": correlation(stage_forgetting, stage_capacity_bias_selection_rate),
            "max_transport_gap": correlation(stage_forgetting, stage_max_transport_gap),
            "mean_abs_excess": correlation(stage_forgetting, stage_mean_abs_excess),
            "merge_count": correlation(stage_forgetting, stage_merge_count),
            "routing_entropy": correlation(stage_forgetting, stage_routing_entropy),
        },
    }


def compute_meta_objective(
    theory_diagnostics: Dict[str, object],
    args: RealBenchmarkArgs,
) -> Dict[str, float]:
    stage_forgetting = theory_diagnostics.get("stage_forgetting", [0.0])
    stage_transport_gap = theory_diagnostics.get("stage_transport_gap", [0.0])
    stage_transport_loss = theory_diagnostics.get("stage_transport_loss", [0.0])
    stage_mean_abs_excess = theory_diagnostics.get("stage_mean_abs_excess", [0.0])
    stage_routing_stability = theory_diagnostics.get("stage_routing_stability_loss", [0.0])
    stage_routing_entropy = theory_diagnostics.get("stage_routing_entropy", [0.0])

    forgetting = float(stage_forgetting[-1]) if stage_forgetting else 0.0
    transport_gap = float(stage_transport_gap[-1]) if stage_transport_gap else 0.0
    transport_loss = float(stage_transport_loss[-1]) if stage_transport_loss else 0.0
    mean_abs_excess = float(stage_mean_abs_excess[-1]) if stage_mean_abs_excess else 0.0
    routing_stability = float(stage_routing_stability[-1]) if stage_routing_stability else 0.0
    routing_entropy = float(stage_routing_entropy[-1]) if stage_routing_entropy else 0.0
    forgetting_delta = (
        float(stage_forgetting[-1]) - float(stage_forgetting[-2])
        if len(stage_forgetting) >= 2
        else 0.0
    )
    transport_gap_delta = (
        float(stage_transport_gap[-1]) - float(stage_transport_gap[-2])
        if len(stage_transport_gap) >= 2
        else 0.0
    )
    forgetting_gap = forgetting - args.target_stage_forgetting
    routing_sharpness = 1.0 / (1.0 + max(0.0, routing_entropy))
    objective = (
        forgetting_gap
        + args.meta_transport_weight * (1.0 + 0.5 * routing_sharpness) * transport_gap
        + args.meta_transport_loss_weight * (1.0 + routing_sharpness) * transport_loss
        + args.meta_excess_weight * (1.0 + 0.5 * routing_sharpness) * mean_abs_excess
        + args.meta_routing_weight * routing_stability
        + 0.5 * max(0.0, forgetting_delta)
        + 0.25 * max(0.0, transport_gap_delta)
    )
    return {
        "objective": float(objective),
        "forgetting": forgetting,
        "forgetting_gap": float(forgetting_gap),
        "forgetting_delta": float(forgetting_delta),
        "transport_gap": transport_gap,
        "transport_gap_delta": float(transport_gap_delta),
        "transport_loss": transport_loss,
        "mean_abs_excess": mean_abs_excess,
        "routing_stability": routing_stability,
        "routing_entropy": routing_entropy,
        "routing_sharpness": float(routing_sharpness),
    }


def build_heuristic_hypergradients(
    theory_diagnostics: Dict[str, object],
    args: RealBenchmarkArgs,
) -> Dict[str, float]:
    correlations = theory_diagnostics.get("forgetting_correlations", {})
    meta_terms = compute_meta_objective(theory_diagnostics, args)
    forgetting_pressure = max(0.0, meta_terms["forgetting"] - args.target_stage_forgetting)
    routing_corr = max(0.0, float(correlations.get("routing_stability") or 0.0))
    transport_corr = max(0.0, float(correlations.get("transport_gap") or 0.0))
    transport_loss_corr = max(0.0, float(correlations.get("transport_loss") or 0.0))
    excess_corr = max(0.0, float(correlations.get("mean_abs_excess") or 0.0))

    return {
        "prototype_prior_strength": -(
            forgetting_pressure + 0.5 * routing_corr + 0.25 * meta_terms["routing_stability"]
        ),
        "prototype_capacity_blend": (
            forgetting_pressure
            + 0.5 * transport_corr
            + 0.25 * transport_loss_corr
            + 0.5 * excess_corr
            + 0.25 * meta_terms["transport_loss"]
            + 0.25 * meta_terms["mean_abs_excess"]
        ),
        "prototype_relocation_strength": -(
            forgetting_pressure
            + 0.5 * meta_terms["transport_gap"]
            + 0.25 * meta_terms["transport_loss"]
            + 0.25 * transport_loss_corr
            + 0.5 * excess_corr
        ),
    }


def compute_forgetting_activation(meta_terms: Dict[str, float], args: RealBenchmarkArgs) -> float:
    target = max(float(args.target_stage_forgetting), 1e-6)
    sustained_forgetting = max(0.0, float(meta_terms.get("forgetting", 0.0)))
    rising_forgetting = max(0.0, float(meta_terms.get("forgetting_delta", 0.0)))
    return float(min(1.0, (sustained_forgetting + 0.5 * rising_forgetting) / target))


def build_dual_transport_gradients(
    theory_diagnostics: Dict[str, object],
    args: RealBenchmarkArgs,
) -> Dict[str, object]:
    meta_terms = compute_meta_objective(theory_diagnostics, args)
    target = max(float(args.target_stage_forgetting), 1e-6)
    transport_gap = max(0.0, meta_terms["transport_gap"])
    transport_loss = max(0.0, meta_terms["transport_loss"])
    stage_task_forgetting = theory_diagnostics.get("stage_task_forgetting", [])
    stage_task_transport_gap = theory_diagnostics.get("stage_task_transport_gap", [])
    stage_task_transport_loss = theory_diagnostics.get("stage_task_transport_loss", [])
    latest_task_forgetting = [float(value) for value in (stage_task_forgetting[-1] if stage_task_forgetting else [])]
    previous_task_forgetting = []
    if len(stage_task_forgetting) >= 2:
        previous_task_forgetting = [float(value) for value in stage_task_forgetting[-2]]
    latest_task_transport_gap = [
        float(value) for value in (stage_task_transport_gap[-1] if stage_task_transport_gap else [])
    ]
    latest_task_transport_loss = [
        float(value) for value in (stage_task_transport_loss[-1] if stage_task_transport_loss else [])
    ]

    current_task_weights = get_task_transport_weights(args, len(latest_task_forgetting))
    updated_task_weights = list(current_task_weights)
    task_transport_signals = [0.0 for _ in latest_task_forgetting]
    task_gap_signals = [0.0 for _ in latest_task_forgetting]
    task_loss_signals = [0.0 for _ in latest_task_forgetting]
    base_weight = max(float(args.transport_weight), 1e-4)
    relaxation = float(np.clip(0.5 * args.transport_weight_step, 0.0, 0.5))

    for task_id, forgetting in enumerate(latest_task_forgetting):
        previous_forgetting = previous_task_forgetting[task_id] if task_id < len(previous_task_forgetting) else 0.0
        forgetting_pressure = max(0.0, forgetting - target)
        forgetting_trend = max(0.0, forgetting - previous_forgetting)
        activation = min(1.0, (max(0.0, forgetting) + 0.5 * forgetting_trend) / target)
        task_gap = latest_task_transport_gap[task_id] if task_id < len(latest_task_transport_gap) else transport_gap
        task_loss = latest_task_transport_loss[task_id] if task_id < len(latest_task_transport_loss) else transport_loss
        task_gap_signals[task_id] = float(max(0.0, task_gap))
        task_loss_signals[task_id] = float(max(0.0, task_loss))
        task_signal = activation * (
            forgetting_pressure
            + args.dual_transport_trend_weight * forgetting_trend
            + args.dual_transport_gap_weight * max(0.0, task_gap)
            + args.dual_transport_loss_weight * max(0.0, task_loss)
        )
        task_transport_signals[task_id] = float(task_signal)
        current_weight = max(1e-4, float(current_task_weights[task_id]))
        if task_signal > 0.0:
            updated_weight = current_weight * np.exp(args.transport_weight_step * task_signal)
        else:
            updated_weight = base_weight + (current_weight - base_weight) * (1.0 - relaxation)
        updated_task_weights[task_id] = float(
            min(1.0, max(1e-4, updated_weight))
        )

    effective_transport_weight = (
        float(np.mean(updated_task_weights[: len(latest_task_forgetting)]))
        if latest_task_forgetting
        else float(args.transport_weight)
    )
    return {
        "task_transport_weights": updated_task_weights,
        "task_transport_signals": task_transport_signals,
        "task_gap_signals": task_gap_signals,
        "task_loss_signals": task_loss_signals,
        "effective_transport_weight": effective_transport_weight,
        "meta_objective": float(meta_terms["objective"]),
    }

def build_meta_secant_bootstrap_gradients(
    theory_diagnostics: Dict[str, object],
    args: RealBenchmarkArgs,
) -> Dict[str, float]:
    correlations = theory_diagnostics.get("forgetting_correlations", {})
    meta_terms = compute_meta_objective(theory_diagnostics, args)
    forgetting_pressure = max(0.0, meta_terms["forgetting_gap"])
    gap_pressure = max(0.0, meta_terms["transport_gap"]) + max(0.0, meta_terms["transport_gap_delta"])
    transport_pressure = max(0.0, meta_terms["transport_loss"])
    excess_pressure = max(0.0, meta_terms["mean_abs_excess"])
    stability_pressure = max(0.0, meta_terms["routing_stability"])
    sharpness = max(0.0, min(1.0, meta_terms["routing_sharpness"]))
    transport_corr = max(0.0, float(correlations.get("transport_gap") or 0.0))
    excess_corr = max(0.0, float(correlations.get("mean_abs_excess") or 0.0))
    stage_forgetting = [float(value) for value in theory_diagnostics.get("stage_forgetting", [])]
    target_forgetting = max(float(args.target_stage_forgetting), 1e-6)
    recent_forgetting = stage_forgetting[-2:]
    sustained_forgetting_pressure = 0.0
    if len(recent_forgetting) == 2:
        sustained_forgetting_pressure = min(
            max(0.0, recent_forgetting[0] - target_forgetting),
            max(0.0, recent_forgetting[1] - target_forgetting),
        ) / target_forgetting
        sustained_forgetting_pressure = min(1.0, sustained_forgetting_pressure)
    forgetting_activation = compute_forgetting_activation(meta_terms, args)
    sparsity_activation = forgetting_activation * (0.5 + 0.5 * sharpness)
    transport_regularization_need = (
        forgetting_activation
        * (
            forgetting_pressure
            + 0.5 * gap_pressure
            + 0.5 * transport_pressure
            + 0.25 * excess_pressure
            + 0.25 * transport_corr
        )
    )
    specialization_need = (
        sustained_forgetting_pressure
        * sparsity_activation
        * (
            forgetting_pressure
            + 0.5 * stability_pressure
            + 0.5 * gap_pressure
            + 0.25 * transport_pressure
        )
    )

    return {
        "prototype_prior_strength": -(
            forgetting_pressure + 0.5 * stability_pressure + 0.25 * sharpness * gap_pressure
        ),
        "prototype_capacity_blend": (
            forgetting_pressure
            + 0.75 * gap_pressure
            + 0.25 * transport_pressure
            + 0.5 * excess_pressure
            + 0.25 * transport_corr
        ),
        "prototype_relocation_strength": -(
            forgetting_pressure
            + 0.75 * gap_pressure
            + 0.5 * excess_pressure
            + 0.25 * stability_pressure
            + 0.25 * excess_corr
        ),
        "transport_weight": -transport_regularization_need,
        "prototype_top_k": specialization_need,
    }

def merge_hypergradients(
    heuristic_gradients: Dict[str, float],
    secant_gradients: Dict[str, Optional[float]],
    mix: float,
) -> Dict[str, float]:
    mix = min(max(mix, 0.0), 1.0)
    merged = {}
    for key, heuristic_value in heuristic_gradients.items():
        secant_value = secant_gradients.get(key)
        if secant_value is None:
            merged[key] = float(heuristic_value)
        else:
            merged[key] = float(mix * secant_value + (1.0 - mix) * heuristic_value)
    return merged


def apply_meta_hyperparameter_update(
    current: Dict[str, float],
    gradients: Dict[str, float],
    args: RealBenchmarkArgs,
    max_prototype_top_k: Optional[int] = None,
) -> Dict[str, float]:
    updated_prior = current["prototype_prior_strength"] * np.exp(
        -args.prior_strength_step * gradients.get("prototype_prior_strength", 0.0)
    )
    updated_capacity = current["prototype_capacity_blend"] - (
        args.capacity_blend_step * gradients.get("prototype_capacity_blend", 0.0)
    )
    updated_relocation = current["prototype_relocation_strength"] - (
        args.relocation_strength_step * gradients.get("prototype_relocation_strength", 0.0)
    )
    updated_transport = current.get("transport_weight", 0.0) - (
        args.transport_weight_step * gradients.get("transport_weight", 0.0)
    )
    max_prototype_top_k = max(1, int(max_prototype_top_k or round(float(current.get("prototype_top_k", 1)))))
    updated_top_k = current.get("prototype_top_k", 1.0) - (
        args.prototype_topk_step * gradients.get("prototype_top_k", 0.0)
    )
    return {
        "prototype_prior_strength": float(min(4.0, max(0.25, updated_prior))),
        "prototype_capacity_blend": float(min(1.0, max(0.0, updated_capacity))),
        "prototype_relocation_strength": float(min(1.0, max(0.0, updated_relocation))),
        "transport_weight": float(min(1.0, max(0.0, updated_transport))),
        "prototype_top_k": int(min(max_prototype_top_k, max(1, round(float(updated_top_k))))),
    }


def estimate_secant_hypergradients(
    current_objective: float,
    previous_adaptation: Optional[Dict[str, object]],
    eps: float,
) -> Dict[str, Optional[float]]:
    secant = {
        "prototype_prior_strength": None,
        "prototype_capacity_blend": None,
        "prototype_relocation_strength": None,
    }
    if not previous_adaptation:
        return secant

    previous_before = previous_adaptation.get("before", {})
    previous_after = previous_adaptation.get("after", {})
    previous_objective = previous_adaptation.get("meta_objective")
    if previous_objective is None:
        return secant

    objective_delta = float(current_objective) - float(previous_objective)
    for key in secant:
        delta_theta = float(previous_after.get(key, 0.0)) - float(previous_before.get(key, 0.0))
        if abs(delta_theta) >= eps:
            secant[key] = float(np.clip(objective_delta / delta_theta, -2.0, 2.0))
    return secant


def adapt_hyperparameters_from_diagnostics(
    model: ContinualTextClassifier,
    theory_diagnostics: Dict[str, object],
    args: RealBenchmarkArgs,
    stage_index: int,
    previous_adaptation: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    current = model.get_prototype_hyperparameters()
    current["transport_weight"] = float(args.transport_weight)
    current_task_transport_weights = get_task_transport_weights(args)
    if current_task_transport_weights:
        current["task_transport_weights"] = list(current_task_transport_weights)
    correlations = theory_diagnostics.get("forgetting_correlations", {})
    meta_terms = compute_meta_objective(theory_diagnostics, args)
    heuristic_gradients = build_heuristic_hypergradients(theory_diagnostics, args)

    dual_update = None
    if args.adaptation_strategy == "meta_secant":
        bootstrap_gradients = build_meta_secant_bootstrap_gradients(theory_diagnostics, args)
        secant_gradients = estimate_secant_hypergradients(
            current_objective=meta_terms["objective"],
            previous_adaptation=previous_adaptation,
            eps=args.meta_secant_eps,
        )
        gradients = merge_hypergradients(
            heuristic_gradients=bootstrap_gradients,
            secant_gradients=secant_gradients,
            mix=args.meta_secant_mix,
        )
    elif args.adaptation_strategy == "dual_transport":
        secant_gradients = {
            "prototype_prior_strength": None,
            "prototype_capacity_blend": None,
            "prototype_relocation_strength": None,
        }
        dual_update = build_dual_transport_gradients(theory_diagnostics, args)
        gradients = {
            "prototype_prior_strength": 0.0,
            "prototype_capacity_blend": 0.0,
            "prototype_relocation_strength": 0.0,
            "transport_weight": -float(np.mean(dual_update.get("task_transport_signals", [0.0]))),
            "prototype_top_k": 0.0,
        }
    elif args.adaptation_strategy == "correlation":
        secant_gradients = {
            "prototype_prior_strength": None,
            "prototype_capacity_blend": None,
            "prototype_relocation_strength": None,
        }
        gradients = heuristic_gradients
    else:
        raise ValueError(f"Unsupported adaptation_strategy: {args.adaptation_strategy}")

    if args.adaptation_strategy == "dual_transport":
        updated = dict(current)
        updated["transport_weight"] = float(dual_update.get("effective_transport_weight", args.transport_weight))
        updated["task_transport_weights"] = list(dual_update.get("task_transport_weights", current.get("task_transport_weights", [])))
        updated["prototype_top_k"] = int(current.get("prototype_top_k", 1))
        args.task_transport_weights = list(updated["task_transport_weights"])
    else:
        updated = apply_meta_hyperparameter_update(
            current,
            gradients,
            args,
            max_prototype_top_k=max(1, model.num_prototypes - 1) if model.num_prototypes > 1 else 1,
        )
    model.set_prototype_hyperparameters(
        prototype_prior_strength=updated.get("prototype_prior_strength"),
        prototype_capacity_blend=updated.get("prototype_capacity_blend"),
        prototype_relocation_strength=updated.get("prototype_relocation_strength"),
        prototype_top_k=updated.get("prototype_top_k"),
    )
    args.transport_weight = float(updated.get("transport_weight", args.transport_weight))
    if args.adaptation_strategy == "dual_transport" and hasattr(model, "set_task_transport_weights"):
        model.set_task_transport_weights(
            get_task_transport_weights(args, model.num_tasks),
            base_weight=args.transport_weight,
        )
    return {
        "stage_index": stage_index,
        "strategy": args.adaptation_strategy,
        "before": current,
        "after": updated,
        "meta_objective": meta_terms["objective"],
        "hypergradients": gradients,
        "secant_hypergradients": secant_gradients,
        "signals": {
            "forgetting": meta_terms["forgetting"],
            "forgetting_gap": meta_terms["forgetting_gap"],
            "forgetting_delta": meta_terms["forgetting_delta"],
            "routing_stability": meta_terms["routing_stability"],
            "transport_gap": meta_terms["transport_gap"],
            "transport_gap_delta": meta_terms["transport_gap_delta"],
            "transport_loss": meta_terms["transport_loss"],
            "mean_abs_excess": meta_terms["mean_abs_excess"],
            "routing_entropy": meta_terms["routing_entropy"],
            "routing_sharpness": meta_terms["routing_sharpness"],
            "routing_correlation": correlations.get("routing_stability"),
            "transport_correlation": correlations.get("transport_gap"),
            "excess_correlation": correlations.get("mean_abs_excess"),
            "controller_transport_signal": float(-gradients.get("transport_weight", 0.0)),
            "controller_topk_signal": float(gradients.get("prototype_top_k", 0.0)),
            "task_transport_signals": list(dual_update.get("task_transport_signals", [])) if dual_update is not None else [],
            "task_gap_signals": list(dual_update.get("task_gap_signals", [])) if dual_update is not None else [],
            "task_loss_signals": list(dual_update.get("task_loss_signals", [])) if dual_update is not None else [],
        },
    }


def train_task(model: ContinualTextClassifier, data_loader, optimizer, replay_buffer: ReplayBuffer, args: RealBenchmarkArgs, device: torch.device):
    criterion = nn.CrossEntropyLoss()
    model.train()
    metric_sums = {
        "train_loss": 0.0,
        "overlap_loss": 0.0,
        "stability_loss": 0.0,
        "balance_loss": 0.0,
        "diversity_loss": 0.0,
        "transport_loss": 0.0,
        "weighted_transport_loss": 0.0,
        "effective_transport_weight": 0.0,
        "routing_stability_loss": 0.0,
    }
    step_count = 0

    for _ in range(args.epochs_per_task):
        for inputs, labels, task_ids in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            task_ids = task_ids.to(device)

            replay_batch = replay_buffer.sample(args.replay_batch_size, device)
            if replay_batch is not None:
                replay_inputs, replay_labels, replay_task_ids = replay_batch
                inputs = torch.cat([inputs, replay_inputs], dim=0)
                labels = torch.cat([labels, replay_labels], dim=0)
                task_ids = torch.cat([task_ids, replay_task_ids], dim=0)

            optimizer.zero_grad()
            logits, info = model(inputs, task_ids=task_ids, return_info=True)
            weighted_transport_loss, effective_transport_weight = compute_task_conditioned_transport_penalty(
                args,
                info,
                task_ids,
            )
            loss = criterion(logits, labels)
            loss = loss + args.overlap_weight * info["overlap_loss"]
            loss = loss + args.stability_weight * info["stability_loss"]
            loss = loss + args.balance_weight * info["balance_loss"]
            loss = loss + args.diversity_weight * info["diversity_loss"]
            loss = loss + weighted_transport_loss
            loss.backward()
            optimizer.step()
            model.update_memory(task_ids, info["layer_infos"])

            metric_sums["train_loss"] += float(loss.item())
            metric_sums["overlap_loss"] += float(info["overlap_loss"].item())
            metric_sums["stability_loss"] += float(info["stability_loss"].item())
            metric_sums["balance_loss"] += float(info["balance_loss"].item())
            metric_sums["diversity_loss"] += float(info["diversity_loss"].item())
            metric_sums["transport_loss"] += float(info.get("transport_loss", info["overlap_loss"]).item())
            metric_sums["weighted_transport_loss"] += float(weighted_transport_loss.item())
            metric_sums["effective_transport_weight"] += float(effective_transport_weight)
            metric_sums["routing_stability_loss"] += float(info.get("routing_stability_loss", info["stability_loss"]).item())
            step_count += 1

            replay_buffer.add_batch(inputs.detach().cpu(), labels.detach().cpu(), task_ids.detach().cpu())

    return {key: value / max(1, step_count) for key, value in metric_sums.items()}


def run_benchmark(args: RealBenchmarkArgs) -> Dict[str, object]:
    set_seed(args.seed)
    initial_config = asdict(args)
    device = torch.device(args.device)

    train_loaders, val_loaders = get_continual_dataloaders(
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        classes_per_task=args.classes_per_task,
        num_workers=args.num_workers,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )

    num_tasks = len(train_loaders)
    if args.adaptation_strategy == "dual_transport":
        initialize_task_transport_weights(args, num_tasks)
    model = ContinualTextClassifier(
        vocab_size=256,
        num_tasks=num_tasks,
        num_classes=args.classes_per_task,
        dim=args.dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        seq_len=args.max_length,
        top_k_patterns=args.top_k_patterns,
        routing_mode=args.routing_mode,
        prototype_routing_strategy=args.prototype_routing_strategy,
        num_prototypes=args.num_prototypes,
        prototype_slots_per_task=args.prototype_slots_per_task,
        prototype_top_k=args.prototype_top_k,
        prototype_reset_threshold=args.prototype_reset_threshold,
        prototype_split_threshold=args.prototype_split_threshold,
        prototype_noise_scale=args.prototype_noise_scale,
        prototype_merge_threshold=args.prototype_merge_threshold,
        prototype_merge_usage_threshold=args.prototype_merge_usage_threshold,
        prototype_masked_sinkhorn_candidate_k=args.prototype_masked_sinkhorn_candidate_k,
        prototype_masked_sinkhorn_capacity_bias=args.prototype_masked_sinkhorn_capacity_bias,
        prototype_birkhoff_transport_strength=args.prototype_birkhoff_transport_strength,
        prototype_birkhoff_adaptive_gate=args.prototype_birkhoff_adaptive_gate,
        prototype_birkhoff_gap_target=args.prototype_birkhoff_gap_target,
        prototype_birkhoff_max_applied_offdiag_mass=args.prototype_birkhoff_max_applied_offdiag_mass,
        prototype_birkhoff_gap_tolerance=args.prototype_birkhoff_gap_tolerance,
        prototype_birkhoff_min_effective_strength=args.prototype_birkhoff_min_effective_strength,
    ).to(device)
    model.set_prototype_hyperparameters(
        prototype_prior_strength=args.prototype_prior_strength,
        prototype_capacity_blend=args.prototype_capacity_blend,
        prototype_masked_sinkhorn_capacity_bias=args.prototype_masked_sinkhorn_capacity_bias,
        prototype_relocation_strength=args.prototype_relocation_strength,
    )
    if args.adaptation_strategy == "dual_transport" and hasattr(model, "set_task_transport_weights"):
        model.set_task_transport_weights(
            get_task_transport_weights(args, num_tasks),
            base_weight=args.transport_weight,
        )
    initial_resolved_layout = {
        "num_prototypes": model.num_prototypes,
        "prototype_top_k": model.prototype_top_k,
        "prototype_slots_per_task": model.prototype_slots_per_task,
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    replay_buffer = ReplayBuffer()

    accuracy_matrix: List[List[float]] = []
    prototype_lifecycle = []
    prototype_diagnostics = []
    stage_training_metrics = []
    hyperparameter_history = []
    for task_id in range(num_tasks):
        stage_training_metrics.append(
            train_task(model, train_loaders[task_id], optimizer, replay_buffer, args, device)
        )
        if args.routing_mode == "prototype":
            prototype_lifecycle.append(model.refresh_prototypes())

        row = [0.0 for _ in range(num_tasks)]
        for seen_task in range(task_id + 1):
            row[seen_task] = evaluate_task(model, val_loaders[seen_task], device)
        accuracy_matrix.append(row)
        if args.routing_mode == "prototype":
            prototype_diagnostics.append(
                collect_prototype_diagnostics(model, val_loaders, task_id + 1, device)
            )
            if (
                args.adaptive_hyperparameters
                and task_id + 1 >= args.adaptation_warmup_stages
                and task_id + 1 < num_tasks
            ):
                stage_theory = compute_theory_diagnostics(
                    accuracy_matrix=accuracy_matrix,
                    stage_training_metrics=stage_training_metrics,
                    prototype_lifecycle=prototype_lifecycle,
                    prototype_diagnostics=prototype_diagnostics,
                )
                previous_adaptation = hyperparameter_history[-1] if hyperparameter_history else None
                hyperparameter_history.append(
                    adapt_hyperparameters_from_diagnostics(
                        model,
                        stage_theory,
                        args,
                        task_id,
                        previous_adaptation=previous_adaptation,
                    )
                )

    metrics = compute_continual_metrics(accuracy_matrix, num_tasks)
    theory_diagnostics = compute_theory_diagnostics(
        accuracy_matrix=accuracy_matrix,
        stage_training_metrics=stage_training_metrics,
        prototype_lifecycle=prototype_lifecycle,
        prototype_diagnostics=prototype_diagnostics,
    )
    results = {
        "config": initial_config,
        "resolved_prototype_layout": initial_resolved_layout,
        "num_tasks": num_tasks,
        "accuracy_matrix": accuracy_matrix,
        "stage_training_metrics": stage_training_metrics,
        "prototype_lifecycle": prototype_lifecycle,
        "prototype_diagnostics": prototype_diagnostics,
        "theory_diagnostics": theory_diagnostics,
        "hyperparameter_history": hyperparameter_history,
        "final_hyperparameters": {
            **model.get_prototype_hyperparameters(),
            "transport_weight": float(args.transport_weight),
            "task_transport_weights": get_task_transport_weights(args, num_tasks),
        },
        **metrics,
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        artifact_paths = save_benchmark_artifacts(results, output_path)
        results.update(artifact_paths)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    return results


def save_benchmark_artifacts(results: Dict[str, object], output_json_path: Path) -> Dict[str, str]:
    figure_path = output_json_path.with_name(f"{output_json_path.stem}_plots.png")
    report_path = output_json_path.with_name(f"{output_json_path.stem}_report.md")

    accuracy_matrix = np.array(results["accuracy_matrix"], dtype=float)
    prototype_diagnostics = results.get("prototype_diagnostics", [])
    prototype_lifecycle = results.get("prototype_lifecycle", [])
    theory_diagnostics = results.get("theory_diagnostics", {})
    hyperparameter_history = results.get("hyperparameter_history", [])

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    im = axes[0, 0].imshow(accuracy_matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    axes[0, 0].set_title("Accuracy Matrix")
    axes[0, 0].set_xlabel("Evaluated Task")
    axes[0, 0].set_ylabel("Training Stage")
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

    final_row = accuracy_matrix[-1]
    axes[0, 1].plot(np.arange(1, len(final_row) + 1), final_row, marker="o")
    axes[0, 1].set_title("Final Task Accuracy")
    axes[0, 1].set_xlabel("Task")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_ylim(0.0, 1.0)

    if prototype_diagnostics:
        entropies = theory_diagnostics.get("stage_routing_entropy", []) or [
            float(np.mean(stage.get("task_routing_entropy", [0.0])))
            for stage in prototype_diagnostics
        ]
        forgetting = theory_diagnostics.get("stage_forgetting", [])
        transport_gap = theory_diagnostics.get("stage_transport_gap", [])
        transport_loss = theory_diagnostics.get("stage_transport_loss", [])
        stages = np.arange(1, len(entropies) + 1)
        axes[1, 0].plot(stages, entropies, marker="o", label="entropy")
        if forgetting:
            axes[1, 0].plot(stages[: len(forgetting)], forgetting, marker="s", label="forgetting")
        if transport_gap:
            axes[1, 0].plot(stages[: len(transport_gap)], transport_gap, marker="^", label="transport gap")
        if transport_loss:
            axes[1, 0].plot(stages[: len(transport_loss)], transport_loss, marker="x", label="transport loss")
        axes[1, 0].set_title("Theory Diagnostics by Stage")
        axes[1, 0].set_xlabel("Training Stage")
        axes[1, 0].legend()

        last_heatmap = prototype_diagnostics[-1].get("task_prototype_heatmap", [])
        if last_heatmap:
            heatmap_array = np.array(last_heatmap, dtype=float)
            axes[1, 1].imshow(heatmap_array, aspect="auto", cmap="magma")
            axes[1, 1].set_title("Task-Prototype Heatmap")
            axes[1, 1].set_xlabel("Prototype")
            axes[1, 1].set_ylabel("Seen Task")
        else:
            axes[1, 1].axis("off")
    else:
        reset_counts = [entry.get("reset_count", 0) for entry in prototype_lifecycle]
        split_counts = [entry.get("split_count", 0) for entry in prototype_lifecycle]
        merge_counts = [entry.get("merge_count", 0) for entry in prototype_lifecycle]
        stages = np.arange(1, len(reset_counts) + 1)
        axes[1, 0].bar(stages - 0.25, reset_counts, width=0.25, label="reset")
        axes[1, 0].bar(stages, split_counts, width=0.25, label="split")
        axes[1, 0].bar(stages + 0.25, merge_counts, width=0.25, label="merge")
        axes[1, 0].set_title("Prototype Lifecycle")
        axes[1, 0].set_xlabel("Training Stage")
        axes[1, 0].legend()
        axes[1, 1].axis("off")

    fig.tight_layout()
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)

    report_lines = [
        "# Continual Text Benchmark Report",
        "",
        f"- Dataset: `{results['config']['dataset_name']}`",
        f"- Routing mode: `{results['config']['routing_mode']}`",
        f"- Routing strategy: `{results['config'].get('prototype_routing_strategy', 'n/a')}`",
        f"- Tasks: `{results['num_tasks']}`",
        f"- Avg accuracy: `{results['avg_accuracy']:.4f}`",
        f"- Avg forgetting: `{results['avg_forgetting']:.4f}`",
        f"- Backward transfer: `{results['backward_transfer']:.4f}`",
        "",
        "## Artifacts",
        "",
        f"- Plot image: `{figure_path.name}`",
        f"- Raw JSON: `{output_json_path.name}`",
    ]
    resolved_layout = results.get("resolved_prototype_layout", {})
    if results["config"].get("routing_mode") == "prototype":
        report_lines.extend(
            [
                f"- Resolved prototypes: `{resolved_layout.get('num_prototypes', 'n/a')}`",
                f"- Prototype top-k: `{resolved_layout.get('prototype_top_k', 'n/a')}`",
                f"- Prototype slots/task: `{resolved_layout.get('prototype_slots_per_task', 'n/a')}`",
            ]
        )
    if prototype_diagnostics:
        last_stage = prototype_diagnostics[-1]
        forgetting_correlations = theory_diagnostics.get("forgetting_correlations", {})
        report_lines.extend(
            [
                "",
                "## Prototype Diagnostics",
                "",
                f"- Final-stage mean routing entropy: `{float(np.mean(last_stage.get('task_routing_entropy', [0.0]))):.4f}`",
                f"- Prototype heatmap rows: `{len(last_stage.get('task_prototype_heatmap', []))}`",
                "",
                "## Theory Diagnostics",
                "",
                f"- Stage forgetting trace: `{theory_diagnostics.get('stage_forgetting', [])}`",
                f"- Stage transport gap trace: `{theory_diagnostics.get('stage_transport_gap', [])}`",
                f"- Stage transport loss trace: `{theory_diagnostics.get('stage_transport_loss', [])}`",
                f"- Stage merge-count trace: `{theory_diagnostics.get('stage_merge_count', [])}`",
                f"- Stage Birkhoff base-strength trace: `{theory_diagnostics.get('stage_birkhoff_base_strength', [])}`",
                f"- Stage Birkhoff effective-strength trace: `{theory_diagnostics.get('stage_birkhoff_strength', [])}`",
                f"- Stage Birkhoff gate-factor trace: `{theory_diagnostics.get('stage_birkhoff_gate_factor', [])}`",
                f"- Stage Birkhoff offdiag-mass trace: `{theory_diagnostics.get('stage_birkhoff_offdiag_mass', [])}`",
                f"- Stage Birkhoff applied-offdiag trace: `{theory_diagnostics.get('stage_birkhoff_applied_offdiag_mass', [])}`",
                f"- Stage Birkhoff gap-delta trace: `{theory_diagnostics.get('stage_birkhoff_gap_delta', [])}`",
                f"- Stage Birkhoff row-error trace: `{theory_diagnostics.get('stage_birkhoff_row_error', [])}`",
                f"- Stage Birkhoff col-error trace: `{theory_diagnostics.get('stage_birkhoff_col_error', [])}`",
                f"- Stage routing stability trace: `{theory_diagnostics.get('stage_routing_stability_loss', [])}`",
                f"- Forgetting vs routing stability correlation: `{forgetting_correlations.get('routing_stability')}`",
                f"- Forgetting vs transport gap correlation: `{forgetting_correlations.get('transport_gap')}`",
                f"- Forgetting vs transport loss correlation: `{forgetting_correlations.get('transport_loss')}`",
                f"- Forgetting vs mean abs excess correlation: `{forgetting_correlations.get('mean_abs_excess')}`",
                f"- Forgetting vs merge-count correlation: `{forgetting_correlations.get('merge_count')}`",
            ]
        )
        if hyperparameter_history:
            report_lines.extend(
                [
                    "",
                    "## Hyperparameter Adaptation",
                    "",
                    f"- Adaptation steps: `{len(hyperparameter_history)}`",
                    f"- Adaptation strategy: `{results['config'].get('adaptation_strategy')}`",
                    f"- Final hyperparameters: `{results.get('final_hyperparameters', {})}`",
                ]
            )

    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return {
        "plot_path": str(figure_path),
        "report_path": str(report_path),
    }


def parse_args() -> RealBenchmarkArgs:
    parser = argparse.ArgumentParser(description="Run a real-text continual benchmark for ASAM")
    for field_name, field_def in RealBenchmarkArgs.__dataclass_fields__.items():
        arg_name = f"--{field_name.replace('_', '-')}"
        default_value = field_def.default
        arg_type = type(default_value) if default_value is not None else str
        parser.add_argument(arg_name, type=arg_type, default=default_value)
    namespace = parser.parse_args()
    return RealBenchmarkArgs(**vars(namespace))


def main():
    args = parse_args()
    results = run_benchmark(args)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
