"""
Minimal continual-learning training scaffold for Continual ASAM.
"""

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from asam import ContinualASAMConfig, ContinualASAMLayer, PrototypeContinualASAMLayer


class SyntheticContinualDataset(Dataset):
    """Simple task-structured token dataset for continual-learning smoke tests."""

    def __init__(
        self,
        task_id: int,
        num_samples: int,
        seq_len: int,
        vocab_size: int,
        num_classes: int,
        task_vocab_span: int,
        seed: int,
    ):
        generator = torch.Generator().manual_seed(seed + task_id)
        task_offset = 1 + task_id * task_vocab_span
        usable_span = min(task_vocab_span, vocab_size - task_offset - 1)
        if usable_span <= num_classes + 2:
            raise ValueError("vocab_size is too small for the requested task layout")

        inputs = torch.randint(
            low=task_offset,
            high=task_offset + usable_span,
            size=(num_samples, seq_len),
            generator=generator,
        )
        labels = torch.randint(0, num_classes, size=(num_samples,), generator=generator)

        class_anchor_base = task_offset
        position_anchor_base = task_offset + num_classes + 1
        for index in range(num_samples):
            label = int(labels[index].item())
            inputs[index, 0] = class_anchor_base + label
            inputs[index, 1] = position_anchor_base + (index % max(1, usable_span - num_classes - 1))

        self.inputs = inputs
        self.labels = labels
        self.task_ids = torch.full((num_samples,), task_id, dtype=torch.long)

    def __len__(self) -> int:
        return self.inputs.size(0)

    def __getitem__(self, index: int):
        return self.inputs[index], self.labels[index], self.task_ids[index]


class ReplayBuffer:
    """Per-task episodic replay buffer."""

    def __init__(self, samples_per_task: int = 0):
        self.samples_per_task = samples_per_task
        self.storage: Dict[int, Dict[str, torch.Tensor]] = {}

    def __len__(self) -> int:
        return sum(item["inputs"].size(0) for item in self.storage.values())

    def add_dataset(self, dataset: SyntheticContinualDataset, seed: int):
        if self.samples_per_task <= 0:
            return

        generator = torch.Generator().manual_seed(seed + int(dataset.task_ids[0].item()))
        count = min(self.samples_per_task, len(dataset))
        indices = torch.randperm(len(dataset), generator=generator)[:count]
        self.storage[int(dataset.task_ids[0].item())] = {
            "inputs": dataset.inputs[indices].clone(),
            "labels": dataset.labels[indices].clone(),
            "task_ids": dataset.task_ids[indices].clone(),
        }

    def sample(self, batch_size: int, device: torch.device):
        if batch_size <= 0 or not self.storage:
            return None

        all_inputs = torch.cat([entry["inputs"] for entry in self.storage.values()], dim=0)
        all_labels = torch.cat([entry["labels"] for entry in self.storage.values()], dim=0)
        all_task_ids = torch.cat([entry["task_ids"] for entry in self.storage.values()], dim=0)

        sample_count = min(batch_size, all_inputs.size(0))
        indices = torch.randperm(all_inputs.size(0))[:sample_count]
        return (
            all_inputs[indices].to(device),
            all_labels[indices].to(device),
            all_task_ids[indices].to(device),
        )


def resolve_prototype_layout(
    num_tasks: int,
    num_prototypes: int = 0,
    prototype_slots_per_task: int = 2,
    prototype_top_k: int = 2,
) -> Dict[str, int]:
    resolved_num_prototypes = int(num_prototypes)
    if resolved_num_prototypes <= 0:
        resolved_num_prototypes = max(1, int(num_tasks) * max(1, int(prototype_slots_per_task)))

    resolved_top_k = max(1, int(prototype_top_k))
    if resolved_num_prototypes > 1 and resolved_top_k >= resolved_num_prototypes:
        if int(num_prototypes) > 0:
            resolved_top_k = resolved_num_prototypes - 1
        else:
            resolved_num_prototypes = resolved_top_k + 1

    resolved_top_k = min(resolved_top_k, resolved_num_prototypes)
    return {
        "num_prototypes": resolved_num_prototypes,
        "prototype_top_k": resolved_top_k,
        "prototype_slots_per_task": max(1, int(prototype_slots_per_task)),
    }


class ContinualTextClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_tasks: int,
        num_classes: int,
        dim: int,
        num_heads: int,
        num_layers: int,
        seq_len: int,
        top_k_patterns: int,
        routing_mode: str = "task",
        prototype_routing_strategy: str = "sinkhorn_topk",
        num_prototypes: int = 0,
        prototype_slots_per_task: int = 2,
        prototype_top_k: int = 2,
        prototype_reset_threshold: float = 0.0,
        prototype_split_threshold: float = 0.0,
        prototype_noise_scale: float = 0.05,
        prototype_merge_threshold: float = 0.9,
        prototype_merge_usage_threshold: float = 0.1,
        prototype_birkhoff_transport_strength: float = 0.02,
        prototype_birkhoff_adaptive_gate: bool = True,
        prototype_birkhoff_gap_target: float = 0.03,
        prototype_birkhoff_max_applied_offdiag_mass: float = 0.006,
        prototype_birkhoff_gap_tolerance: float = 0.0,
        prototype_birkhoff_min_effective_strength: float = 1e-4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_classes = num_classes
        self.routing_mode = routing_mode

        resolved_layout = resolve_prototype_layout(
            num_tasks=num_tasks,
            num_prototypes=num_prototypes,
            prototype_slots_per_task=prototype_slots_per_task,
            prototype_top_k=prototype_top_k,
        )
        self.num_prototypes = resolved_layout["num_prototypes"]
        self.prototype_top_k = resolved_layout["prototype_top_k"]
        self.prototype_slots_per_task = resolved_layout["prototype_slots_per_task"]

        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Parameter(torch.randn(1, seq_len, dim) * 0.02)
        layer_config = ContinualASAMConfig(
            dim=dim,
            num_heads=num_heads,
            dim_head=dim // num_heads,
            dropout=dropout,
            num_tasks=num_tasks,
            num_prototypes=self.num_prototypes,
            top_k_patterns=top_k_patterns,
            prototype_top_k=self.prototype_top_k,
            prototype_routing_strategy=prototype_routing_strategy,
            prototype_reset_threshold=prototype_reset_threshold,
            prototype_split_threshold=prototype_split_threshold,
            prototype_noise_scale=prototype_noise_scale,
            prototype_merge_threshold=prototype_merge_threshold,
            prototype_merge_usage_threshold=prototype_merge_usage_threshold,
            prototype_birkhoff_transport_strength=prototype_birkhoff_transport_strength,
            prototype_birkhoff_adaptive_gate=prototype_birkhoff_adaptive_gate,
            prototype_birkhoff_gap_target=prototype_birkhoff_gap_target,
            prototype_birkhoff_max_applied_offdiag_mass=prototype_birkhoff_max_applied_offdiag_mass,
            prototype_birkhoff_gap_tolerance=prototype_birkhoff_gap_tolerance,
            prototype_birkhoff_min_effective_strength=prototype_birkhoff_min_effective_strength,
        )
        if routing_mode == "task":
            layer_cls = ContinualASAMLayer
        elif routing_mode == "prototype":
            layer_cls = PrototypeContinualASAMLayer
        else:
            raise ValueError(f"Unsupported routing_mode: {routing_mode}")
        self.layers = nn.ModuleList([layer_cls(layer_config) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)
        self.heads = nn.ModuleList([nn.Linear(dim, num_classes) for _ in range(num_tasks)])

    def forward(self, inputs: torch.Tensor, task_ids: torch.Tensor, return_info: bool = False):
        seq_len = inputs.size(1)
        hidden = self.token_embedding(inputs) + self.position_embedding[:, :seq_len, :]

        layer_infos: List[Dict[str, torch.Tensor]] = []
        for layer in self.layers:
            if isinstance(layer, PrototypeContinualASAMLayer):
                hidden, layer_info = layer(hidden, task_ids=task_ids, return_info=return_info)
            else:
                hidden, layer_info = layer(hidden, task_ids=task_ids, return_info=return_info)
            if return_info:
                layer_infos.append(layer_info)

        pooled = self.norm(hidden).mean(dim=1)
        all_logits = torch.stack([head(pooled) for head in self.heads], dim=1)
        batch_indices = torch.arange(inputs.size(0), device=inputs.device)
        logits = all_logits[batch_indices, task_ids]

        if not return_info:
            return logits, None

        overlap_loss = torch.stack([info["overlap_loss"] for info in layer_infos]).mean()
        stability_loss = torch.stack([info["stability_loss"] for info in layer_infos]).mean()
        balance_terms = [info["balance_loss"] for info in layer_infos if "balance_loss" in info]
        diversity_terms = [info["diversity_loss"] for info in layer_infos if "diversity_loss" in info]
        routing_stability_terms = [
            info["routing_stability_loss"]
            for info in layer_infos
            if "routing_stability_loss" in info
        ]
        transport_terms = [info["transport_loss"] for info in layer_infos if "transport_loss" in info]
        transport_per_sample_terms = [
            info["transport_loss_per_sample"]
            for info in layer_infos
            if "transport_loss_per_sample" in info
        ]
        info = {
            "layer_infos": layer_infos,
            "overlap_loss": overlap_loss,
            "stability_loss": stability_loss,
            "balance_loss": torch.stack(balance_terms).mean() if balance_terms else overlap_loss.new_zeros(()),
            "diversity_loss": torch.stack(diversity_terms).mean() if diversity_terms else overlap_loss.new_zeros(()),
            "routing_stability_loss": (
                torch.stack(routing_stability_terms).mean()
                if routing_stability_terms
                else overlap_loss.new_zeros(())
            ),
            "transport_loss": torch.stack(transport_terms).mean() if transport_terms else overlap_loss.new_zeros(()),
            "transport_loss_per_sample": (
                torch.stack(transport_per_sample_terms, dim=0).mean(dim=0)
                if transport_per_sample_terms
                else overlap_loss.new_zeros((inputs.size(0),))
            ),
        }
        return logits, info

    @torch.no_grad()
    def update_memory(self, task_ids: torch.Tensor, layer_infos: List[Dict[str, torch.Tensor]]):
        for layer, layer_info in zip(self.layers, layer_infos):
            if isinstance(layer, PrototypeContinualASAMLayer):
                layer.update_prototype_memory(
                    head_importance=layer_info["head_importance"],
                    pattern_weights=layer_info["pattern_weights"],
                    prototype_weights=layer_info["prototype_weights"],
                    prototype_capacity=layer_info.get("prototype_capacity"),
                    prototype_support=layer_info.get("prototype_support"),
                    prototype_latents=layer_info.get("prototype_latents"),
                    task_ids=task_ids,
                )
            else:
                layer.update_task_memory(
                    task_ids,
                    layer_info["head_importance"],
                    pattern_weights=layer_info["pattern_weights"],
                )

    @torch.no_grad()
    def set_task_transport_weights(
        self,
        task_transport_weights: List[float],
        base_weight: float,
    ):
        if self.num_tasks <= 0:
            return
        weights = list(task_transport_weights)
        if len(weights) < self.num_tasks:
            weights.extend([float(base_weight)] * (self.num_tasks - len(weights)))
        weight_tensor = torch.tensor(weights[: self.num_tasks], dtype=torch.float32)
        for layer in self.layers:
            if isinstance(layer, PrototypeContinualASAMLayer):
                layer.set_task_transport_weights(weight_tensor, base_weight=base_weight)

    @torch.no_grad()
    def refresh_prototypes(self) -> Dict[str, float]:
        reset_count = 0
        split_count = 0
        merge_count = 0
        transport_gaps = []
        max_transport_gaps = []
        excess_terms = []
        merge_similarities = []
        birkhoff_base_strengths = []
        birkhoff_strengths = []
        birkhoff_gate_factors = []
        birkhoff_offdiag_masses = []
        birkhoff_applied_offdiag_masses = []
        birkhoff_row_errors = []
        birkhoff_col_errors = []
        birkhoff_pre_gaps = []
        birkhoff_post_gaps = []
        birkhoff_gap_deltas = []
        for layer in self.layers:
            if isinstance(layer, PrototypeContinualASAMLayer):
                stats = layer.refresh_prototypes()
                reset_count += int(stats.get("reset_count", 0))
                split_count += int(stats.get("split_count", 0))
                merge_count += int(stats.get("merge_count", 0))
                transport_gaps.append(float(stats.get("mean_transport_gap", 0.0)))
                max_transport_gaps.append(float(stats.get("max_transport_gap", 0.0)))
                excess_terms.append(float(stats.get("mean_excess", 0.0)))
                birkhoff_base_strengths.append(float(stats.get("birkhoff_base_strength", 0.0)))
                birkhoff_strengths.append(float(stats.get("birkhoff_strength", 0.0)))
                birkhoff_gate_factors.append(float(stats.get("birkhoff_gate_factor", 0.0)))
                birkhoff_offdiag_masses.append(float(stats.get("birkhoff_offdiag_mass", 0.0)))
                birkhoff_applied_offdiag_masses.append(float(stats.get("birkhoff_applied_offdiag_mass", 0.0)))
                birkhoff_row_errors.append(float(stats.get("birkhoff_row_error", 0.0)))
                birkhoff_col_errors.append(float(stats.get("birkhoff_col_error", 0.0)))
                birkhoff_pre_gaps.append(float(stats.get("birkhoff_pre_gap", 0.0)))
                birkhoff_post_gaps.append(float(stats.get("birkhoff_post_gap", 0.0)))
                birkhoff_gap_deltas.append(float(stats.get("birkhoff_gap_delta", 0.0)))
                if int(stats.get("merge_count", 0)) > 0:
                    merge_similarities.append(float(stats.get("mean_merge_similarity", 0.0)))
        return {
            "reset_count": reset_count,
            "split_count": split_count,
            "merge_count": merge_count,
            "mean_transport_gap": float(sum(transport_gaps) / max(1, len(transport_gaps))),
            "max_transport_gap": float(max(max_transport_gaps) if max_transport_gaps else 0.0),
            "mean_excess": float(sum(excess_terms) / max(1, len(excess_terms))),
            "mean_merge_similarity": float(sum(merge_similarities) / max(1, len(merge_similarities))),
            "birkhoff_base_strength": float(sum(birkhoff_base_strengths) / max(1, len(birkhoff_base_strengths))),
            "birkhoff_strength": float(sum(birkhoff_strengths) / max(1, len(birkhoff_strengths))),
            "birkhoff_gate_factor": float(sum(birkhoff_gate_factors) / max(1, len(birkhoff_gate_factors))),
            "birkhoff_offdiag_mass": float(sum(birkhoff_offdiag_masses) / max(1, len(birkhoff_offdiag_masses))),
            "birkhoff_applied_offdiag_mass": float(sum(birkhoff_applied_offdiag_masses) / max(1, len(birkhoff_applied_offdiag_masses))),
            "birkhoff_row_error": float(max(birkhoff_row_errors) if birkhoff_row_errors else 0.0),
            "birkhoff_col_error": float(max(birkhoff_col_errors) if birkhoff_col_errors else 0.0),
            "birkhoff_pre_gap": float(sum(birkhoff_pre_gaps) / max(1, len(birkhoff_pre_gaps))),
            "birkhoff_post_gap": float(sum(birkhoff_post_gaps) / max(1, len(birkhoff_post_gaps))),
            "birkhoff_gap_delta": float(sum(birkhoff_gap_deltas) / max(1, len(birkhoff_gap_deltas))),
        }

    @torch.no_grad()
    def get_prototype_hyperparameters(self) -> Dict[str, float]:
        prototype_layers = [layer for layer in self.layers if isinstance(layer, PrototypeContinualASAMLayer)]
        if not prototype_layers:
            return {
                "prototype_prior_strength": 0.0,
                "prototype_capacity_blend": 0.0,
                "prototype_relocation_strength": 0.0,
                "prototype_merge_threshold": 0.0,
                "prototype_merge_usage_threshold": 0.0,
            }

        return {
            "prototype_prior_strength": float(
                sum(layer.prototype_gate.prior_strength for layer in prototype_layers) / len(prototype_layers)
            ),
            "prototype_capacity_blend": float(
                sum(layer.prototype_gate.capacity_blend for layer in prototype_layers) / len(prototype_layers)
            ),
            "prototype_relocation_strength": float(
                sum(layer.continual_config.prototype_relocation_strength for layer in prototype_layers)
                / len(prototype_layers)
            ),
            "prototype_merge_threshold": float(
                sum(layer.continual_config.prototype_merge_threshold for layer in prototype_layers)
                / len(prototype_layers)
            ),
            "prototype_merge_usage_threshold": float(
                sum(layer.continual_config.prototype_merge_usage_threshold for layer in prototype_layers)
                / len(prototype_layers)
            ),
            "prototype_top_k": int(
                round(sum(layer.prototype_gate.top_k for layer in prototype_layers) / len(prototype_layers))
            ),
        }

    @torch.no_grad()
    def set_prototype_hyperparameters(
        self,
        prototype_prior_strength: Optional[float] = None,
        prototype_capacity_blend: Optional[float] = None,
        prototype_relocation_strength: Optional[float] = None,
        prototype_merge_threshold: Optional[float] = None,
        prototype_merge_usage_threshold: Optional[float] = None,
        prototype_top_k: Optional[float] = None,
    ):
        for layer in self.layers:
            if not isinstance(layer, PrototypeContinualASAMLayer):
                continue
            if prototype_prior_strength is not None:
                layer.prototype_gate.prior_strength = float(prototype_prior_strength)
                layer.continual_config.prototype_prior_strength = float(prototype_prior_strength)
            if prototype_capacity_blend is not None:
                clipped_blend = float(min(max(prototype_capacity_blend, 0.0), 1.0))
                layer.prototype_gate.capacity_blend = clipped_blend
                layer.continual_config.prototype_capacity_blend = clipped_blend
            if prototype_relocation_strength is not None:
                clipped_relocation = float(min(max(prototype_relocation_strength, 0.0), 1.0))
                layer.continual_config.prototype_relocation_strength = clipped_relocation
            if prototype_merge_threshold is not None:
                clipped_merge = float(min(max(prototype_merge_threshold, -1.0), 1.0))
                layer.continual_config.prototype_merge_threshold = clipped_merge
            if prototype_merge_usage_threshold is not None:
                clipped_usage = float(max(prototype_merge_usage_threshold, 0.0))
                layer.continual_config.prototype_merge_usage_threshold = clipped_usage
            if prototype_top_k is not None:
                max_supported_top_k = max(1, self.num_prototypes - 1) if self.num_prototypes > 1 else 1
                clipped_top_k = int(min(max_supported_top_k, max(1, round(float(prototype_top_k)))))
                layer.prototype_gate.top_k = clipped_top_k
                layer.continual_config.prototype_top_k = clipped_top_k
                self.prototype_top_k = clipped_top_k


@dataclass
class ExperimentArgs:
    num_tasks: int = 3
    num_classes_per_task: int = 2
    train_samples: int = 48
    val_samples: int = 24
    seq_len: int = 48
    vocab_size: int = 512
    task_vocab_span: int = 96
    batch_size: int = 8
    replay_batch_size: int = 4
    replay_samples_per_task: int = 8
    epochs_per_task: int = 1
    dim: int = 64
    num_heads: int = 4
    num_layers: int = 1
    top_k_patterns: int = 2
    routing_mode: str = "task"
    prototype_routing_strategy: str = "sinkhorn_topk"
    num_prototypes: int = 0
    prototype_slots_per_task: int = 2
    prototype_top_k: int = 2
    learning_rate: float = 3e-4
    overlap_weight: float = 0.1
    stability_weight: float = 0.1
    balance_weight: float = 0.0
    diversity_weight: float = 0.0
    transport_weight: float = 0.05
    prototype_reset_threshold: float = 0.0
    prototype_split_threshold: float = 0.0
    prototype_noise_scale: float = 0.05
    prototype_merge_threshold: float = 0.9
    prototype_merge_usage_threshold: float = 0.1
    prototype_birkhoff_transport_strength: float = 0.02
    prototype_birkhoff_adaptive_gate: bool = True
    prototype_birkhoff_gap_target: float = 0.03
    prototype_birkhoff_max_applied_offdiag_mass: float = 0.006
    prototype_birkhoff_gap_tolerance: float = 0.0
    prototype_birkhoff_min_effective_strength: float = 1e-4
    prototype_prior_strength: float = 1.0
    prototype_capacity_blend: float = 0.5
    prototype_relocation_strength: float = 0.75
    dropout: float = 0.1
    seed: int = 42
    output_json: Optional[str] = None
    device: str = "cpu"


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_task_loaders(args: ExperimentArgs):
    train_loaders = []
    val_loaders = []
    train_datasets = []
    for task_id in range(args.num_tasks):
        train_dataset = SyntheticContinualDataset(
            task_id=task_id,
            num_samples=args.train_samples,
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
            num_classes=args.num_classes_per_task,
            task_vocab_span=args.task_vocab_span,
            seed=args.seed,
        )
        val_dataset = SyntheticContinualDataset(
            task_id=task_id,
            num_samples=args.val_samples,
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
            num_classes=args.num_classes_per_task,
            task_vocab_span=args.task_vocab_span,
            seed=args.seed + 1000,
        )
        train_datasets.append(train_dataset)
        train_loaders.append(DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True))
        val_loaders.append(DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False))
    return train_datasets, train_loaders, val_loaders


@torch.no_grad()
def evaluate_task(model: ContinualTextClassifier, data_loader: DataLoader, device: torch.device) -> float:
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


@torch.no_grad()
def collect_prototype_diagnostics(
    model: ContinualTextClassifier,
    val_loaders: List[DataLoader],
    num_seen_tasks: int,
    device: torch.device,
) -> Dict[str, object]:
    model.eval()

    capacity_references = []
    for layer in model.layers:
        if not isinstance(layer, PrototypeContinualASAMLayer):
            continue
        layer_capacity = getattr(layer, "prototype_capacity_ema", None)
        if layer_capacity is None:
            continue
        capacity = layer_capacity.detach().to(device=device, dtype=torch.float32)
        if capacity.numel() == 0:
            continue
        if capacity.sum().item() <= 0:
            capacity = torch.full_like(capacity, 1.0 / max(1, capacity.numel()))
        else:
            capacity = capacity / capacity.sum().clamp_min(1e-6)
        capacity_references.append(capacity)

    reference_capacity = None
    if capacity_references:
        reference_capacity = torch.stack(capacity_references, dim=0).mean(dim=0)
        reference_capacity = reference_capacity / reference_capacity.sum().clamp_min(1e-6)

    task_prototype_heatmap = []
    task_routing_entropy = []
    task_transport_gap = []
    task_max_transport_gap = []
    task_transport_loss = []
    for task_id in range(num_seen_tasks):
        prototype_sum = None
        entropy_sum = 0.0
        transport_loss_sum = 0.0
        sample_count = 0

        for inputs, _labels, task_ids in val_loaders[task_id]:
            inputs = inputs.to(device)
            task_ids = task_ids.to(device)
            _, info = model(inputs, task_ids=task_ids, return_info=True)

            prototype_layers = [
                layer_info["prototype_weights"]
                for layer_info in info["layer_infos"]
                if "prototype_weights" in layer_info
            ]
            if not prototype_layers:
                continue

            average_weights = torch.stack(prototype_layers, dim=0).mean(dim=0)
            if prototype_sum is None:
                prototype_sum = average_weights.sum(dim=0)
            else:
                prototype_sum = prototype_sum + average_weights.sum(dim=0)

            entropy = -(average_weights.clamp_min(1e-6) * average_weights.clamp_min(1e-6).log()).sum(dim=-1)
            entropy_sum += entropy.sum().item()
            transport_loss_per_sample = info.get("transport_loss_per_sample")
            if transport_loss_per_sample is not None:
                transport_loss_sum += transport_loss_per_sample.sum().item()
            sample_count += average_weights.size(0)

        if prototype_sum is None:
            task_prototype_heatmap.append([])
            task_routing_entropy.append(0.0)
            task_transport_gap.append(0.0)
            task_max_transport_gap.append(0.0)
            task_transport_loss.append(0.0)
        else:
            mean_weights = prototype_sum / max(1, sample_count)
            task_prototype_heatmap.append(mean_weights.cpu().tolist())
            task_routing_entropy.append(entropy_sum / max(1, sample_count))
            if reference_capacity is None or reference_capacity.numel() != mean_weights.numel():
                target_capacity = torch.full_like(mean_weights, 1.0 / max(1, mean_weights.numel()))
            else:
                target_capacity = reference_capacity.to(mean_weights)
            gap = (mean_weights - target_capacity).abs()
            task_transport_gap.append(float(gap.mean().item()))
            task_max_transport_gap.append(float(gap.max().item()))
            task_transport_loss.append(float(transport_loss_sum / max(1, sample_count)))

    layer_similarity = []
    layer_usage = []
    layer_capacity_ema = []
    layer_support_ema = []
    layer_excess_ema = []
    layer_latent_ema = []
    for layer in model.layers:
        if not isinstance(layer, PrototypeContinualASAMLayer):
            continue
        normalized_prototypes = F.normalize(layer.prototype_gate.prototype_embeddings.detach(), dim=-1)
        similarity = torch.matmul(normalized_prototypes, normalized_prototypes.transpose(0, 1))
        layer_similarity.append(similarity.cpu().tolist())
        layer_usage.append(layer.prototype_usage_ema.detach().cpu().tolist())
        layer_capacity = getattr(layer, "prototype_capacity_ema", None)
        layer_support = getattr(layer, "prototype_support_ema", None)
        layer_excess = getattr(layer, "prototype_excess_ema", None)
        layer_latent = getattr(layer, "prototype_latent_ema", None)
        layer_capacity_ema.append(layer_capacity.detach().cpu().tolist() if layer_capacity is not None else [])
        layer_support_ema.append(layer_support.detach().cpu().tolist() if layer_support is not None else [])
        layer_excess_ema.append(layer_excess.detach().cpu().tolist() if layer_excess is not None else [])
        layer_latent_ema.append(layer_latent.detach().cpu().tolist() if layer_latent is not None else [])

    return {
        "task_prototype_heatmap": task_prototype_heatmap,
        "task_routing_entropy": task_routing_entropy,
        "task_transport_gap": task_transport_gap,
        "task_max_transport_gap": task_max_transport_gap,
        "task_transport_loss": task_transport_loss,
        "layer_similarity": layer_similarity,
        "layer_usage_ema": layer_usage,
        "layer_capacity_ema": layer_capacity_ema,
        "layer_support_ema": layer_support_ema,
        "layer_excess_ema": layer_excess_ema,
        "layer_latent_ema": layer_latent_ema,
    }


def train_task(
    model: ContinualTextClassifier,
    data_loader: DataLoader,
    replay_buffer: ReplayBuffer,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: ExperimentArgs,
) -> Dict[str, float]:
    criterion = nn.CrossEntropyLoss()
    model.train()
    metric_sums = {
        "train_loss": 0.0,
        "overlap_loss": 0.0,
        "stability_loss": 0.0,
        "balance_loss": 0.0,
        "diversity_loss": 0.0,
        "transport_loss": 0.0,
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
            loss = criterion(logits, labels)
            loss = loss + args.overlap_weight * info["overlap_loss"]
            loss = loss + args.stability_weight * info["stability_loss"]
            loss = loss + args.balance_weight * info["balance_loss"]
            loss = loss + args.diversity_weight * info["diversity_loss"]
            loss = loss + args.transport_weight * info["transport_loss"]
            loss.backward()
            optimizer.step()

            model.update_memory(task_ids, info["layer_infos"])

            metric_sums["train_loss"] += float(loss.item())
            metric_sums["overlap_loss"] += float(info["overlap_loss"].item())
            metric_sums["stability_loss"] += float(info["stability_loss"].item())
            metric_sums["balance_loss"] += float(info["balance_loss"].item())
            metric_sums["diversity_loss"] += float(info["diversity_loss"].item())
            metric_sums["transport_loss"] += float(info["transport_loss"].item())
            metric_sums["routing_stability_loss"] += float(info.get("routing_stability_loss", info["stability_loss"]).item())
            step_count += 1

    return {key: value / max(1, step_count) for key, value in metric_sums.items()}


def compute_continual_metrics(accuracy_matrix: List[List[float]], num_tasks: int) -> Dict[str, float]:
    final_row = accuracy_matrix[-1]
    avg_accuracy = sum(final_row) / max(1, num_tasks)

    forgetting_terms = []
    for task_id in range(num_tasks - 1):
        best_before_end = max(row[task_id] for row in accuracy_matrix[:-1])
        forgetting_terms.append(best_before_end - final_row[task_id])
    avg_forgetting = sum(forgetting_terms) / max(1, len(forgetting_terms)) if forgetting_terms else 0.0

    backward_transfer_terms = []
    for task_id in range(num_tasks - 1):
        backward_transfer_terms.append(final_row[task_id] - accuracy_matrix[task_id][task_id])
    backward_transfer = (
        sum(backward_transfer_terms) / max(1, len(backward_transfer_terms)) if backward_transfer_terms else 0.0
    )

    return {
        "avg_accuracy": avg_accuracy,
        "avg_forgetting": avg_forgetting,
        "backward_transfer": backward_transfer,
    }


def run_experiment(args: ExperimentArgs) -> Dict[str, object]:
    set_seed(args.seed)
    device = torch.device(args.device)

    train_datasets, train_loaders, val_loaders = build_task_loaders(args)
    model = ContinualTextClassifier(
        vocab_size=args.vocab_size,
        num_tasks=args.num_tasks,
        num_classes=args.num_classes_per_task,
        dim=args.dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        seq_len=args.seq_len,
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
        prototype_birkhoff_transport_strength=args.prototype_birkhoff_transport_strength,
        prototype_birkhoff_adaptive_gate=args.prototype_birkhoff_adaptive_gate,
        prototype_birkhoff_gap_target=args.prototype_birkhoff_gap_target,
        prototype_birkhoff_max_applied_offdiag_mass=args.prototype_birkhoff_max_applied_offdiag_mass,
        prototype_birkhoff_gap_tolerance=args.prototype_birkhoff_gap_tolerance,
        prototype_birkhoff_min_effective_strength=args.prototype_birkhoff_min_effective_strength,
        dropout=args.dropout,
    ).to(device)
    if args.routing_mode == "prototype":
        model.set_prototype_hyperparameters(
            prototype_prior_strength=args.prototype_prior_strength,
            prototype_capacity_blend=args.prototype_capacity_blend,
            prototype_relocation_strength=args.prototype_relocation_strength,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    replay_buffer = ReplayBuffer(samples_per_task=args.replay_samples_per_task)
    lifecycle_stats = []
    prototype_diagnostics = []
    stage_training_metrics = []

    accuracy_matrix: List[List[float]] = []
    for task_id in range(args.num_tasks):
        stage_training_metrics.append(
            train_task(model, train_loaders[task_id], replay_buffer, optimizer, device, args)
        )
        if args.routing_mode == "prototype":
            lifecycle_stats.append(model.refresh_prototypes())
        replay_buffer.add_dataset(train_datasets[task_id], seed=args.seed + 5000)

        row = [0.0 for _ in range(args.num_tasks)]
        for seen_task in range(task_id + 1):
            row[seen_task] = evaluate_task(model, val_loaders[seen_task], device)
        accuracy_matrix.append(row)
        if args.routing_mode == "prototype":
            prototype_diagnostics.append(
                collect_prototype_diagnostics(model, val_loaders, task_id + 1, device)
            )

    metrics = compute_continual_metrics(accuracy_matrix, args.num_tasks)
    results = {
        "config": asdict(args),
        "resolved_prototype_layout": {
            "num_prototypes": model.num_prototypes,
            "prototype_top_k": model.prototype_top_k,
            "prototype_slots_per_task": model.prototype_slots_per_task,
        },
        "accuracy_matrix": accuracy_matrix,
        "routing_mode": args.routing_mode,
        "prototype_lifecycle": lifecycle_stats,
        "prototype_diagnostics": prototype_diagnostics,
        "stage_training_metrics": stage_training_metrics,
        **metrics,
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    return results


def parse_args() -> ExperimentArgs:
    parser = argparse.ArgumentParser(description="Train Continual ASAM on synthetic continual tasks")
    for field_name, field_def in ExperimentArgs.__dataclass_fields__.items():
        arg_name = f"--{field_name.replace('_', '-')}"
        default_value = field_def.default
        arg_type = type(default_value) if default_value is not None else str
        parser.add_argument(arg_name, type=arg_type, default=default_value)
    namespace = parser.parse_args()
    return ExperimentArgs(**vars(namespace))


def main():
    args = parse_args()
    results = run_experiment(args)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
