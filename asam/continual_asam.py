"""
Continual ASAM
==============

Task-aware sparse attention for continual learning.
"""

from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._common import expand_pattern_mask, normalize_attention_mask, pattern_mask_to_indices
from .asam_layer import ASAMConfig, ASAMLayer
from .sparse_patterns import (
    HierarchicalSparsePattern,
    LocalSparsePattern,
    RandomSparsePattern,
    StridedSparsePattern,
)


@dataclass
class ContinualASAMConfig(ASAMConfig):
    use_adaptive_gate: bool = False
    num_tasks: int = 8
    num_prototypes: int = 8
    task_embed_dim: int = 64
    prototype_embed_dim: int = 64
    task_hidden_dim: int = 128
    top_k_patterns: int = 2
    prototype_top_k: int = 2
    pattern_bank: Tuple[str, ...] = field(
        default_factory=lambda: ("local", "strided", "random", "hierarchical")
    )
    random_seed: int = 42
    memory_momentum: float = 0.9
    routing_temperature: float = 1.0
    prototype_routing_strategy: str = "sinkhorn_topk"
    prototype_prior_strength: float = 1.0
    prototype_prior_floor: float = 1e-3
    prototype_sinkhorn_epsilon: float = 0.1
    prototype_sinkhorn_iters: int = 20
    prototype_capacity_blend: float = 0.5
    prototype_masked_sinkhorn_candidate_k: int = 0
    prototype_masked_sinkhorn_capacity_bias: float = 0.0
    prototype_relocation_strength: float = 0.75
    prototype_balance_weight: float = 0.0
    prototype_diversity_weight: float = 0.0
    prototype_usage_momentum: float = 0.9
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
    prototype_birkhoff_sinkhorn_iters: int = 32
    prototype_birkhoff_epsilon: float = 0.25
    prototype_birkhoff_diag_bias: float = 4.0
    prototype_birkhoff_gap_weight: float = 4.0
    grouped_indexed_attention: bool = True
    grouped_indexed_attention_max_group_size: int = 1


class TaskAwareSparseGate(nn.Module):
    """Task-conditioned gate that routes attention heads to sparse patterns."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_tasks: int,
        num_patterns: int,
        task_embed_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_patterns = num_patterns

        self.task_embedding = nn.Embedding(num_tasks, task_embed_dim)
        self.feature_proj = nn.Sequential(
            nn.Linear(dim + task_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.pattern_proj = nn.Linear(hidden_dim, num_heads * num_patterns)
        self.head_proj = nn.Linear(hidden_dim, num_heads)

    def forward(self, x: torch.Tensor, task_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        pooled = x.mean(dim=1)
        task_features = self.task_embedding(task_ids)
        features = self.feature_proj(torch.cat([pooled, task_features], dim=-1))

        pattern_logits = self.pattern_proj(features).view(-1, self.num_heads, self.num_patterns)
        pattern_weights = F.softmax(pattern_logits, dim=-1)
        head_importance = torch.sigmoid(self.head_proj(features))

        return {
            "pattern_logits": pattern_logits,
            "pattern_weights": pattern_weights,
            "head_importance": head_importance,
            "task_features": task_features,
        }


class PrototypeSparseGate(nn.Module):
    """Task-agnostic sparse gate driven by KL-prox sparse prototype routing."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_patterns: int,
        num_prototypes: int,
        prototype_embed_dim: int = 64,
        hidden_dim: int = 128,
        routing_temperature: float = 1.0,
        top_k: int = 2,
        prior_strength: float = 1.0,
        prior_floor: float = 1e-3,
        routing_strategy: str = "sinkhorn_topk",
        sinkhorn_epsilon: float = 0.1,
        sinkhorn_iters: int = 20,
        capacity_blend: float = 0.5,
        masked_sinkhorn_candidate_k: int = 0,
        masked_sinkhorn_capacity_bias: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_patterns = num_patterns
        self.num_prototypes = num_prototypes
        self.routing_temperature = routing_temperature
        self.top_k = top_k
        self.prior_strength = prior_strength
        self.prior_floor = prior_floor
        self.routing_strategy = routing_strategy
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.sinkhorn_iters = sinkhorn_iters
        self.capacity_blend = capacity_blend
        self.masked_sinkhorn_candidate_k = masked_sinkhorn_candidate_k
        self.masked_sinkhorn_capacity_bias = masked_sinkhorn_capacity_bias

        self.input_proj = nn.Linear(dim, prototype_embed_dim)
        self.prototype_embeddings = nn.Parameter(
            torch.randn(num_prototypes, prototype_embed_dim) * 0.02
        )
        self.feature_proj = nn.Sequential(
            nn.Linear(dim + prototype_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.pattern_proj = nn.Linear(hidden_dim, num_heads * num_patterns)
        self.head_proj = nn.Linear(hidden_dim, num_heads)

    def _normalize_prior(
        self,
        routing_prior: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if routing_prior is None:
            routing_prior = torch.full(
                (batch_size, self.num_prototypes),
                1.0 / self.num_prototypes,
                device=device,
                dtype=dtype,
            )
        elif routing_prior.dim() == 1:
            routing_prior = routing_prior.unsqueeze(0).expand(batch_size, -1)
        elif routing_prior.dim() != 2 or routing_prior.size(0) != batch_size:
            raise ValueError("routing_prior must have shape [batch, num_prototypes]")

        routing_prior = routing_prior.to(device=device, dtype=dtype)
        routing_prior = routing_prior.clamp_min(self.prior_floor)
        return routing_prior / routing_prior.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def _topk_sparse_softmax(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        top_k = min(max(1, self.top_k), logits.size(-1))
        if top_k >= logits.size(-1):
            support = torch.ones_like(logits, dtype=torch.bool)
            return F.softmax(logits, dim=-1), support

        top_values, top_indices = logits.topk(k=top_k, dim=-1)
        sparse_logits = torch.full_like(logits, float("-inf"))
        sparse_logits.scatter_(dim=-1, index=top_indices, src=top_values)

        support = torch.zeros_like(logits, dtype=torch.bool)
        support.scatter_(
            dim=-1,
            index=top_indices,
            src=torch.ones(top_indices.shape, device=logits.device, dtype=torch.bool),
        )
        return F.softmax(sparse_logits, dim=-1), support

    def _topk_project_probabilities(
        self,
        probabilities: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        top_k = min(max(1, self.top_k), probabilities.size(-1))
        if top_k >= probabilities.size(-1):
            support = torch.ones_like(probabilities, dtype=torch.bool)
            normalized = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            return normalized, support

        top_values, top_indices = probabilities.topk(k=top_k, dim=-1)
        projected = torch.zeros_like(probabilities)
        projected.scatter_(dim=-1, index=top_indices, src=top_values)
        projected = projected / projected.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        support = torch.zeros_like(probabilities, dtype=torch.bool)
        support.scatter_(
            dim=-1,
            index=top_indices,
            src=torch.ones(top_indices.shape, device=probabilities.device, dtype=torch.bool),
        )
        return projected, support

    def _build_capacity_target(self, routing_prior: torch.Tensor) -> torch.Tensor:
        average_prior = routing_prior.mean(dim=0)
        uniform = torch.full_like(average_prior, 1.0 / average_prior.numel())
        blend = min(max(self.capacity_blend, 0.0), 1.0)
        target = blend * average_prior + (1.0 - blend) * uniform
        return target / target.sum().clamp_min(1e-6)

    def _project_capacity_to_support(
        self,
        target_capacity: torch.Tensor,
        support: torch.Tensor,
    ) -> torch.Tensor:
        support_weights = support.to(dtype=target_capacity.dtype)
        target_on_support = support_weights * target_capacity.unsqueeze(0)
        row_target_mass = target_on_support.sum(dim=-1, keepdim=True)
        fallback = support_weights / support_weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        row_distribution = torch.where(
            row_target_mass > 1e-6,
            target_on_support / row_target_mass.clamp_min(1e-6),
            fallback,
        )
        capacity = row_distribution.mean(dim=0)
        return capacity / capacity.sum().clamp_min(1e-6)

    def _sinkhorn_transport_weights(
        self,
        logits: torch.Tensor,
        target_capacity: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = logits.size(0)
        row_mass = torch.full(
            (batch_size,),
            1.0 / batch_size,
            device=logits.device,
            dtype=logits.dtype,
        )
        col_mass = target_capacity.to(device=logits.device, dtype=logits.dtype)
        col_mass = col_mass / col_mass.sum().clamp_min(1e-6)

        scaled_logits = logits / max(self.sinkhorn_epsilon, 1e-6)
        row_max = scaled_logits.max(dim=-1, keepdim=True).values
        row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
        stabilized_logits = torch.nan_to_num(
            scaled_logits - row_max,
            nan=0.0,
            posinf=0.0,
            neginf=-80.0,
        ).clamp(min=-80.0, max=0.0)
        kernel = torch.exp(stabilized_logits).clamp_min(1e-9)
        u = torch.ones_like(row_mass)
        v = torch.ones_like(col_mass)
        for _ in range(max(1, self.sinkhorn_iters)):
            u = row_mass / torch.matmul(kernel, v).clamp_min(1e-9)
            v = col_mass / torch.matmul(kernel.transpose(0, 1), u).clamp_min(1e-9)

        transport = u.unsqueeze(-1) * kernel * v.unsqueeze(0)
        return transport / transport.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def _route_with_sinkhorn(
        self,
        logits: torch.Tensor,
        routing_prior: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_capacity = self._build_capacity_target(routing_prior)
        dense_weights = self._sinkhorn_transport_weights(logits, target_capacity)
        sparse_weights, support = self._topk_project_probabilities(dense_weights)
        return sparse_weights, support, target_capacity

    def _route_with_sinkhorn_support_masked(
        self,
        logits: torch.Tensor,
        routing_prior: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        target_capacity = self._build_capacity_target(routing_prior)
        with torch.no_grad():
            proposal_weights = self._sinkhorn_transport_weights(
                logits.detach(),
                target_capacity.detach(),
            )
            _, support = self._topk_project_probabilities(proposal_weights)
        effective_capacity = self._project_capacity_to_support(target_capacity, support)
        weights = self._masked_sinkhorn_transport_weights(
            logits=logits,
            target_capacity=effective_capacity,
            support=support,
        )
        if not return_diagnostics:
            return weights, support, effective_capacity
        diagnostics = self._build_support_diagnostics(
            logits=logits,
            target_capacity=target_capacity,
            prototype_weights=weights,
            prototype_support=support,
            prototype_capacity=effective_capacity,
            candidate_support=support,
            proposal_weights=proposal_weights,
            biased_support_selected=False,
        )
        return weights, support, effective_capacity, diagnostics

    def _topk_support_from_scores(
        self,
        scores: torch.Tensor,
    ) -> torch.Tensor:
        candidate_k = int(self.masked_sinkhorn_candidate_k)
        if candidate_k <= 0:
            candidate_k = self.top_k
        candidate_k = min(max(1, candidate_k), scores.size(-1))
        if candidate_k >= scores.size(-1):
            return torch.ones_like(scores, dtype=torch.bool)

        _, top_indices = scores.topk(k=candidate_k, dim=-1)
        support = torch.zeros_like(scores, dtype=torch.bool)
        support.scatter_(
            dim=-1,
            index=top_indices,
            src=torch.ones(top_indices.shape, device=scores.device, dtype=torch.bool),
        )
        return support

    def _build_masked_sinkhorn_support(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        return self._topk_support_from_scores(logits)

    def _select_masked_sinkhorn_support(
        self,
        logits: torch.Tensor,
        target_capacity: torch.Tensor,
        return_selection: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        baseline_support = self._build_masked_sinkhorn_support(logits)
        baseline_capacity = self._project_capacity_to_support(target_capacity, baseline_support)
        beta = float(self.masked_sinkhorn_capacity_bias)
        if beta <= 0.0:
            result = (baseline_support, baseline_capacity, False)
            return result if return_selection else result[:2]

        capacity_scores = logits + beta * torch.log(
            target_capacity.to(device=logits.device, dtype=logits.dtype).clamp_min(self.prior_floor)
        ).unsqueeze(0)
        biased_support = self._topk_support_from_scores(capacity_scores)
        biased_capacity = self._project_capacity_to_support(target_capacity, biased_support)

        baseline_residual = (baseline_capacity - target_capacity).abs().sum()
        biased_residual = (biased_capacity - target_capacity).abs().sum()
        if biased_residual.item() < baseline_residual.item():
            result = (biased_support, biased_capacity, True)
            return result if return_selection else result[:2]
        result = (baseline_support, baseline_capacity, False)
        return result if return_selection else result[:2]

    def _build_support_diagnostics(
        self,
        logits: torch.Tensor,
        target_capacity: torch.Tensor,
        prototype_weights: torch.Tensor,
        prototype_support: torch.Tensor,
        prototype_capacity: torch.Tensor,
        candidate_support: torch.Tensor,
        proposal_weights: Optional[torch.Tensor] = None,
        biased_support_selected: bool = False,
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            diagnostic_weights = prototype_weights.detach()
            target_capacity = target_capacity.detach().to(
                device=diagnostic_weights.device, dtype=diagnostic_weights.dtype
            )
            prototype_capacity = prototype_capacity.detach().to(
                device=diagnostic_weights.device,
                dtype=diagnostic_weights.dtype,
            )
            candidate_support = candidate_support.detach().to(device=diagnostic_weights.device)
            prototype_support = prototype_support.detach().to(device=diagnostic_weights.device)
            candidate_capacity = self._project_capacity_to_support(
                target_capacity, candidate_support
            )
            support_capacity = self._project_capacity_to_support(target_capacity, prototype_support)
            candidate_residual = (candidate_capacity - target_capacity).abs().mean()
            support_residual = (support_capacity - target_capacity).abs().mean()
            mean_weights = diagnostic_weights.mean(dim=0)
            leakage_weights = (
                diagnostic_weights
                if proposal_weights is None
                else proposal_weights.detach().to(
                    device=diagnostic_weights.device,
                    dtype=diagnostic_weights.dtype,
                )
            )
            return {
                "prototype_target_capacity": target_capacity.detach(),
                "candidate_support_residual": candidate_residual.detach(),
                "support_projection_residual": support_residual.detach(),
                "support_residual_delta": (candidate_residual - support_residual).detach(),
                "target_capacity_residual": (mean_weights - target_capacity).abs().mean().detach(),
                "effective_capacity_residual": (mean_weights - prototype_capacity)
                .abs()
                .mean()
                .detach(),
                "support_density": prototype_support.float().mean().detach(),
                "support_size": prototype_support.float().sum(dim=-1).mean().detach(),
                "support_active_prototypes": prototype_support.any(dim=0).float().sum().detach(),
                "support_weight_leakage": leakage_weights.masked_select(~prototype_support)
                .abs()
                .sum()
                .detach(),
                "capacity_bias_selection_rate": diagnostic_weights.new_tensor(
                    1.0 if biased_support_selected else 0.0
                ).detach(),
            }

    def _masked_sinkhorn_transport_weights(
        self,
        logits: torch.Tensor,
        target_capacity: torch.Tensor,
        support: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = logits.size(0)
        row_mass = torch.full(
            (batch_size,),
            1.0 / batch_size,
            device=logits.device,
            dtype=logits.dtype,
        )
        col_mass = target_capacity.to(device=logits.device, dtype=logits.dtype)
        col_mass = col_mass / col_mass.sum().clamp_min(1e-6)

        scaled_logits = logits / max(self.sinkhorn_epsilon, 1e-6)
        scaled_logits = scaled_logits.masked_fill(~support, float("-inf"))
        row_max = scaled_logits.max(dim=-1, keepdim=True).values
        row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
        kernel = torch.exp(scaled_logits - row_max).masked_fill(~support, 0.0)
        kernel = kernel.clamp_min(0.0)

        u = torch.ones_like(row_mass)
        v = torch.ones_like(col_mass)
        for _ in range(max(1, self.sinkhorn_iters)):
            u = row_mass / torch.matmul(kernel, v).clamp_min(1e-9)
            v = col_mass / torch.matmul(kernel.transpose(0, 1), u).clamp_min(1e-9)

        transport = u.unsqueeze(-1) * kernel * v.unsqueeze(0)
        weights = transport / transport.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        weights = weights.masked_fill(~support, 0.0)
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)

    def _route_with_masked_sinkhorn(
        self,
        logits: torch.Tensor,
        routing_prior: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        target_capacity = self._build_capacity_target(routing_prior)
        candidate_support = self._build_masked_sinkhorn_support(logits)
        candidate_k = int(self.masked_sinkhorn_candidate_k)
        if candidate_k <= 0:
            candidate_k = self.top_k
        if min(max(1, candidate_k), logits.size(-1)) >= logits.size(-1):
            weights, support, effective_capacity = self._route_with_sinkhorn(logits, routing_prior)
            if not return_diagnostics:
                return weights, support, effective_capacity
            proposal_weights = self._sinkhorn_transport_weights(
                logits.detach(), target_capacity.detach()
            )
            diagnostics = self._build_support_diagnostics(
                logits=logits,
                target_capacity=target_capacity,
                prototype_weights=weights,
                prototype_support=support,
                prototype_capacity=effective_capacity,
                candidate_support=candidate_support,
                proposal_weights=proposal_weights,
                biased_support_selected=False,
            )
            return weights, support, effective_capacity, diagnostics

        support, effective_capacity, biased_support_selected = self._select_masked_sinkhorn_support(
            logits,
            target_capacity,
            return_selection=True,
        )

        weights = self._masked_sinkhorn_transport_weights(
            logits=logits,
            target_capacity=effective_capacity,
            support=support,
        )
        if not return_diagnostics:
            return weights, support, effective_capacity
        proposal_weights = self._sinkhorn_transport_weights(
            logits.detach(), target_capacity.detach()
        )
        diagnostics = self._build_support_diagnostics(
            logits=logits,
            target_capacity=target_capacity,
            prototype_weights=weights,
            prototype_support=support,
            prototype_capacity=effective_capacity,
            candidate_support=candidate_support,
            proposal_weights=proposal_weights,
            biased_support_selected=biased_support_selected,
        )
        return weights, support, effective_capacity, diagnostics

    def forward(
        self,
        x: torch.Tensor,
        routing_prior: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        pooled = x.mean(dim=1)
        routed_input = F.normalize(self.input_proj(pooled), dim=-1)
        normalized_prototypes = F.normalize(self.prototype_embeddings, dim=-1)
        prototype_logits = torch.matmul(routed_input, normalized_prototypes.transpose(0, 1))
        prototype_logits = prototype_logits / max(self.routing_temperature, 1e-6)

        routing_prior = self._normalize_prior(
            routing_prior,
            batch_size=pooled.size(0),
            device=prototype_logits.device,
            dtype=prototype_logits.dtype,
        )
        proximal_logits = prototype_logits + self.prior_strength * torch.log(
            routing_prior.clamp_min(self.prior_floor)
        )
        if self.routing_strategy == "sinkhorn_topk":
            prototype_weights, prototype_support, prototype_capacity = self._route_with_sinkhorn(
                proximal_logits,
                routing_prior,
            )
            proposal_weights = self._sinkhorn_transport_weights(
                proximal_logits.detach(),
                self._build_capacity_target(routing_prior).detach(),
            )
            support_diagnostics = self._build_support_diagnostics(
                logits=proximal_logits,
                target_capacity=self._build_capacity_target(routing_prior),
                prototype_weights=prototype_weights,
                prototype_support=prototype_support,
                prototype_capacity=prototype_capacity,
                candidate_support=self._build_masked_sinkhorn_support(proximal_logits),
                proposal_weights=proposal_weights,
                biased_support_selected=False,
            )
        elif self.routing_strategy == "masked_sinkhorn_topk":
            prototype_weights, prototype_support, prototype_capacity, support_diagnostics = (
                self._route_with_masked_sinkhorn(
                    proximal_logits,
                    routing_prior,
                    return_diagnostics=True,
                )
            )
        elif self.routing_strategy == "sinkhorn_support_masked":
            prototype_weights, prototype_support, prototype_capacity, support_diagnostics = (
                self._route_with_sinkhorn_support_masked(
                    proximal_logits,
                    routing_prior,
                    return_diagnostics=True,
                )
            )
        elif self.routing_strategy == "kl_topk":
            prototype_weights, prototype_support = self._topk_sparse_softmax(proximal_logits)
            prototype_capacity = routing_prior.mean(dim=0)
            proposal_weights = F.softmax(proximal_logits, dim=-1)
            support_diagnostics = self._build_support_diagnostics(
                logits=proximal_logits,
                target_capacity=self._build_capacity_target(routing_prior),
                prototype_weights=prototype_weights,
                prototype_support=prototype_support,
                prototype_capacity=prototype_capacity,
                candidate_support=self._build_masked_sinkhorn_support(proximal_logits),
                proposal_weights=proposal_weights,
                biased_support_selected=False,
            )
        else:
            raise ValueError(f"Unsupported routing strategy: {self.routing_strategy}")

        prototype_features = torch.matmul(prototype_weights, self.prototype_embeddings)
        features = self.feature_proj(torch.cat([pooled, prototype_features], dim=-1))
        pattern_logits = self.pattern_proj(features).view(-1, self.num_heads, self.num_patterns)
        pattern_weights = F.softmax(pattern_logits, dim=-1)
        head_importance = torch.sigmoid(self.head_proj(features))

        return {
            "prototype_logits": prototype_logits,
            "proximal_logits": proximal_logits,
            "prototype_prior": routing_prior,
            "prototype_capacity": prototype_capacity,
            "prototype_support": prototype_support,
            "prototype_weights": prototype_weights,
            "prototype_features": prototype_features,
            "prototype_latents": routed_input,
            "pattern_logits": pattern_logits,
            "pattern_weights": pattern_weights,
            "head_importance": head_importance,
            **support_diagnostics,
        }


class ContinualASAMLayer(ASAMLayer):
    """
    ASAM variant for continual learning.

    The layer conditions sparse supports on task identity and exposes two
    continual-learning regularizers:
    1. support overlap regularization across tasks,
    2. head-importance stability regularization across revisited tasks.
    """

    def __init__(self, config: ContinualASAMConfig):
        base_config = replace(config, use_adaptive_gate=False)
        super().__init__(base_config)
        self.continual_config = config
        self.pattern_bank = tuple(config.pattern_bank)
        self.task_gate = TaskAwareSparseGate(
            dim=config.dim,
            num_heads=config.num_heads,
            num_tasks=config.num_tasks,
            num_patterns=len(self.pattern_bank),
            task_embed_dim=config.task_embed_dim,
            hidden_dim=config.task_hidden_dim,
        )
        self._base_pattern_mask_cache = {}

        self.register_buffer("task_head_memory", torch.zeros(config.num_tasks, config.num_heads))
        self.register_buffer(
            "task_pattern_memory",
            torch.zeros(config.num_tasks, config.num_heads, len(self.pattern_bank)),
        )
        self.register_buffer("task_memory_seen", torch.zeros(config.num_tasks, dtype=torch.bool))

    def _create_pattern_module(self, pattern_name: str, seq_len: int):
        if pattern_name == "local":
            return LocalSparsePattern(seq_len, self.continual_config.window_size)
        if pattern_name == "strided":
            return StridedSparsePattern(seq_len, self.continual_config.stride)
        if pattern_name == "random":
            return RandomSparsePattern(
                seq_len,
                num_heads=self.num_heads,
                seed=self.continual_config.random_seed,
            )
        if pattern_name == "hierarchical":
            return HierarchicalSparsePattern(seq_len, num_heads=self.num_heads)
        raise ValueError(f"Unsupported continual pattern: {pattern_name}")

    def _get_base_pattern_masks(self, seq_len: int, device: torch.device) -> torch.Tensor:
        cache_key = (seq_len, device.type, device.index)
        cached_masks = self._base_pattern_mask_cache.get(cache_key)
        if cached_masks is not None:
            return cached_masks

        pattern_masks = []
        for pattern_name in self.pattern_bank:
            pattern_module = self._create_pattern_module(pattern_name, seq_len)
            if isinstance(pattern_module, HierarchicalSparsePattern):
                mask = pattern_module.combine_patterns(device)
            else:
                mask = pattern_module.get_pattern(device)
            mask = expand_pattern_mask(mask, self.num_heads)
            pattern_masks.append(mask)

        cached_masks = torch.stack(pattern_masks, dim=0)
        self._base_pattern_mask_cache[cache_key] = cached_masks
        return cached_masks

    def _build_task_pattern_mask(
        self,
        pattern_logits: torch.Tensor,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        top_k = min(self.continual_config.top_k_patterns, len(self.pattern_bank))
        top_indices = pattern_logits.topk(k=top_k, dim=-1).indices
        return self._build_pattern_mask_from_indices(top_indices, seq_len, device)

    def _build_pattern_mask_from_indices(
        self,
        top_indices: torch.Tensor,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        base_masks = self._get_base_pattern_masks(seq_len, device)
        selected_masks = []
        for head_index in range(self.num_heads):
            head_masks = base_masks[top_indices[head_index], head_index]
            selected_masks.append(head_masks.any(dim=0))
        return torch.stack(selected_masks, dim=0)

    def _compute_legacy_pattern_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pattern_logits: torch.Tensor,
        seq_len: int,
        device: torch.device,
        normalized_mask: Optional[torch.Tensor] = None,
        return_pattern_masks: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        outputs = []
        pattern_masks = []
        batch = q.size(0)
        for batch_index in range(batch):
            pattern_mask = self._build_task_pattern_mask(
                pattern_logits[batch_index],
                seq_len,
                device,
            )
            if return_pattern_masks:
                pattern_masks.append(pattern_mask)

            positions, valid_mask = pattern_mask_to_indices(pattern_mask)
            sample_mask = (
                None if normalized_mask is None else normalized_mask[batch_index : batch_index + 1]
            )
            sample_out = self._compute_sparse_attention_from_indices(
                q[batch_index : batch_index + 1],
                k[batch_index : batch_index + 1],
                v[batch_index : batch_index + 1],
                positions,
                valid_mask,
                mask=sample_mask,
            )
            outputs.append(sample_out)

        stacked_masks = torch.stack(pattern_masks, dim=0) if return_pattern_masks else None
        return torch.cat(outputs, dim=0), stacked_masks

    def _compute_grouped_pattern_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pattern_logits: torch.Tensor,
        seq_len: int,
        device: torch.device,
        normalized_mask: Optional[torch.Tensor] = None,
        return_pattern_masks: bool = False,
        group_hint_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not bool(self.continual_config.grouped_indexed_attention):
            return self._compute_legacy_pattern_attention(
                q,
                k,
                v,
                pattern_logits,
                seq_len,
                device,
                normalized_mask=normalized_mask,
                return_pattern_masks=return_pattern_masks,
            )
        if group_hint_ids is not None and group_hint_ids.numel() == q.size(0):
            if group_hint_ids.unique().numel() == q.size(0):
                return self._compute_legacy_pattern_attention(
                    q,
                    k,
                    v,
                    pattern_logits,
                    seq_len,
                    device,
                    normalized_mask=normalized_mask,
                    return_pattern_masks=return_pattern_masks,
                )

        top_k = min(self.continual_config.top_k_patterns, len(self.pattern_bank))
        selected_patterns = pattern_logits.topk(k=top_k, dim=-1).indices
        canonical_patterns = torch.sort(selected_patterns, dim=-1).values.reshape(q.size(0), -1)
        _, group_ids = torch.unique(canonical_patterns, dim=0, return_inverse=True)
        num_groups = int(group_ids.max().item()) + 1
        if num_groups == q.size(0):
            return self._compute_legacy_pattern_attention(
                q,
                k,
                v,
                pattern_logits,
                seq_len,
                device,
                normalized_mask=normalized_mask,
                return_pattern_masks=return_pattern_masks,
            )

        outputs = [None] * q.size(0)
        pattern_masks = [None] * q.size(0) if return_pattern_masks else None
        for group_id in range(num_groups):
            group_index_tensor = torch.nonzero(group_ids == group_id, as_tuple=False).squeeze(-1)
            group_indices = group_index_tensor.tolist()
            first_index = group_indices[0]
            top_indices = selected_patterns[first_index]
            pattern_mask = self._build_pattern_mask_from_indices(top_indices, seq_len, device)
            positions, valid_mask = pattern_mask_to_indices(pattern_mask)
            max_group_size = int(self.continual_config.grouped_indexed_attention_max_group_size)
            chunk_size = len(group_indices) if max_group_size <= 0 else max(1, max_group_size)
            for chunk_start in range(0, len(group_indices), chunk_size):
                chunk_indices = group_indices[chunk_start : chunk_start + chunk_size]
                chunk_index_tensor = torch.tensor(chunk_indices, device=q.device, dtype=torch.long)
                group_mask = None
                if normalized_mask is not None:
                    group_mask = normalized_mask.index_select(0, chunk_index_tensor)
                group_out = self._compute_sparse_attention_from_indices(
                    q.index_select(0, chunk_index_tensor),
                    k.index_select(0, chunk_index_tensor),
                    v.index_select(0, chunk_index_tensor),
                    positions,
                    valid_mask,
                    mask=group_mask,
                )
                for offset, batch_index in enumerate(chunk_indices):
                    outputs[batch_index] = group_out[offset : offset + 1]
                    if pattern_masks is not None:
                        pattern_masks[batch_index] = pattern_mask

        attn_out = torch.cat(outputs, dim=0)
        stacked_masks = torch.stack(pattern_masks, dim=0) if pattern_masks is not None else None
        return attn_out, stacked_masks

    def _effective_support_scores(
        self,
        pattern_weights: torch.Tensor,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        base_masks = self._get_base_pattern_masks(seq_len, device).to(dtype=pattern_weights.dtype)
        return torch.einsum("bhp,phij->bhij", pattern_weights, base_masks)

    def compute_continual_regularization(
        self,
        task_ids: torch.Tensor,
        pattern_weights: torch.Tensor,
        head_importance: torch.Tensor,
        seq_len: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        effective_support = self._effective_support_scores(pattern_weights, seq_len, device)
        base_masks = self._get_base_pattern_masks(seq_len, device).to(dtype=pattern_weights.dtype)
        overlap_terms = []

        batch = task_ids.size(0)
        for left in range(batch):
            for right in range(left + 1, batch):
                if task_ids[left] != task_ids[right]:
                    overlap_terms.append(
                        (effective_support[left] * effective_support[right]).mean()
                    )

        remembered_task_ids = torch.nonzero(self.task_memory_seen, as_tuple=False).squeeze(-1)
        for batch_index in range(batch):
            other_task_ids = remembered_task_ids[remembered_task_ids != task_ids[batch_index]]
            if other_task_ids.numel() == 0:
                continue

            remembered_pattern_weights = self.task_pattern_memory[other_task_ids]
            remembered_support = torch.einsum(
                "thp,phij->thij", remembered_pattern_weights, base_masks
            )
            current_support = effective_support[batch_index].unsqueeze(0)
            overlap_terms.append((current_support * remembered_support).mean())

        if overlap_terms:
            overlap_loss = torch.stack(overlap_terms).mean()
        else:
            overlap_loss = head_importance.new_zeros(())

        seen_mask = self.task_memory_seen[task_ids]
        if seen_mask.any():
            remembered = self.task_head_memory[task_ids[seen_mask]]
            stability_loss = F.mse_loss(head_importance[seen_mask], remembered)
        else:
            stability_loss = head_importance.new_zeros(())

        return {
            "overlap_loss": overlap_loss,
            "stability_loss": stability_loss,
        }

    @torch.no_grad()
    def update_task_memory(
        self,
        task_ids: torch.Tensor,
        head_importance: torch.Tensor,
        pattern_weights: Optional[torch.Tensor] = None,
    ):
        unique_task_ids = task_ids.unique(sorted=True)
        for task_id in unique_task_ids.tolist():
            task_mask = task_ids == task_id
            task_importance = head_importance[task_mask].mean(dim=0)
            task_pattern_weights = None
            if pattern_weights is not None:
                task_pattern_weights = pattern_weights[task_mask].mean(dim=0)

            if self.task_memory_seen[task_id]:
                self.task_head_memory[task_id].mul_(self.continual_config.memory_momentum)
                self.task_head_memory[task_id].add_(
                    task_importance * (1.0 - self.continual_config.memory_momentum)
                )
                if task_pattern_weights is not None:
                    self.task_pattern_memory[task_id].mul_(self.continual_config.memory_momentum)
                    self.task_pattern_memory[task_id].add_(
                        task_pattern_weights * (1.0 - self.continual_config.memory_momentum)
                    )
            else:
                self.task_head_memory[task_id].copy_(task_importance)
                if task_pattern_weights is not None:
                    self.task_pattern_memory[task_id].copy_(task_pattern_weights)
                self.task_memory_seen[task_id] = True

    def forward(
        self,
        x: torch.Tensor,
        task_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_info: bool = False,
    ):
        batch, seq_len, dim = x.shape
        if task_ids.dim() == 0:
            task_ids = task_ids.view(1).expand(batch)
        if task_ids.dim() != 1 or task_ids.size(0) != batch:
            raise ValueError("task_ids must be a tensor of shape [batch]")

        residual = x
        x = self.norm(x)

        gate_info = self.task_gate(x, task_ids)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda tensor: tensor.reshape(batch, seq_len, self.num_heads, self.dim_head).transpose(
                1, 2
            ),
            qkv,
        )
        q = q * self.scale

        normalized_mask = None
        if mask is not None:
            normalized_mask = normalize_attention_mask(mask, batch, self.num_heads, seq_len)

        attn_out, task_pattern_masks = self._compute_grouped_pattern_attention(
            q,
            k,
            v,
            gate_info["pattern_logits"],
            seq_len,
            x.device,
            normalized_mask=normalized_mask,
            return_pattern_masks=return_info,
            group_hint_ids=task_ids,
        )
        attn_out = attn_out.transpose(1, 2).reshape(batch, seq_len, dim)
        attn_out = self.to_out(attn_out)

        x = residual + attn_out

        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = residual + x

        if not return_info:
            return x, None

        regularization = self.compute_continual_regularization(
            task_ids=task_ids,
            pattern_weights=gate_info["pattern_weights"],
            head_importance=gate_info["head_importance"],
            seq_len=seq_len,
            device=x.device,
        )
        selected_patterns = (
            gate_info["pattern_logits"]
            .topk(
                k=min(self.continual_config.top_k_patterns, len(self.pattern_bank)),
                dim=-1,
            )
            .indices
        )

        info = {
            "pattern_logits": gate_info["pattern_logits"],
            "pattern_weights": gate_info["pattern_weights"],
            "head_importance": gate_info["head_importance"],
            "selected_patterns": selected_patterns,
            "task_pattern_masks": task_pattern_masks,
            **regularization,
        }
        return x, info


class PrototypeContinualASAMLayer(ContinualASAMLayer):
    """Task-agnostic continual ASAM using learned prototype routing."""

    def __init__(self, config: ContinualASAMConfig):
        super().__init__(config)
        self.prototype_gate = PrototypeSparseGate(
            dim=config.dim,
            num_heads=config.num_heads,
            num_patterns=len(self.pattern_bank),
            num_prototypes=config.num_prototypes,
            prototype_embed_dim=config.prototype_embed_dim,
            hidden_dim=config.task_hidden_dim,
            routing_temperature=config.routing_temperature,
            top_k=config.prototype_top_k,
            prior_strength=config.prototype_prior_strength,
            prior_floor=config.prototype_prior_floor,
            routing_strategy=config.prototype_routing_strategy,
            sinkhorn_epsilon=config.prototype_sinkhorn_epsilon,
            sinkhorn_iters=config.prototype_sinkhorn_iters,
            capacity_blend=config.prototype_capacity_blend,
            masked_sinkhorn_candidate_k=config.prototype_masked_sinkhorn_candidate_k,
            masked_sinkhorn_capacity_bias=config.prototype_masked_sinkhorn_capacity_bias,
        )
        self.register_buffer(
            "prototype_head_memory", torch.zeros(config.num_prototypes, config.num_heads)
        )
        self.register_buffer(
            "prototype_pattern_memory",
            torch.zeros(config.num_prototypes, config.num_heads, len(self.pattern_bank)),
        )
        self.register_buffer(
            "prototype_memory_seen", torch.zeros(config.num_prototypes, dtype=torch.bool)
        )
        self.register_buffer("prototype_usage_ema", torch.zeros(config.num_prototypes))
        self.register_buffer("prototype_capacity_ema", torch.zeros(config.num_prototypes))
        self.register_buffer("prototype_support_ema", torch.zeros(config.num_prototypes))
        self.register_buffer("prototype_excess_ema", torch.zeros(config.num_prototypes))
        self.register_buffer(
            "prototype_latent_ema",
            torch.zeros(config.num_prototypes, config.prototype_embed_dim),
        )
        self.register_buffer(
            "task_prototype_memory",
            torch.zeros(config.num_tasks, config.num_prototypes),
        )
        self.register_buffer("task_prototype_seen", torch.zeros(config.num_tasks, dtype=torch.bool))
        self.register_buffer("task_transport_weights", torch.zeros(config.num_tasks))
        self.register_buffer("task_transport_base_weight", torch.zeros(1))

    def _build_prototype_prior(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        task_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.prototype_memory_seen.any():
            global_prior = torch.full(
                (batch_size, self.continual_config.num_prototypes),
                1.0 / self.continual_config.num_prototypes,
                device=device,
                dtype=dtype,
            )
        else:
            seen_mask = self.prototype_memory_seen.to(device=device)
            prior = self.prototype_usage_ema.to(device=device, dtype=dtype).clamp_min(
                self.continual_config.prototype_prior_floor
            )
            prior = (
                prior * seen_mask.to(dtype)
                + (~seen_mask).to(dtype) * self.continual_config.prototype_prior_floor
            )
            prior = prior / prior.sum().clamp_min(1e-6)
            global_prior = prior.unsqueeze(0).expand(batch_size, -1)

        if (
            task_ids is None
            or not self.task_prototype_seen.any()
            or float(self.task_transport_base_weight.item()) <= 0.0
        ):
            return global_prior

        if task_ids.dim() == 0:
            task_ids = task_ids.view(1).expand(batch_size)
        if task_ids.dim() != 1 or task_ids.size(0) != batch_size:
            raise ValueError("task_ids must be a tensor of shape [batch] for prototype routing")

        clamped_task_ids = (
            task_ids.to(device=device).long().clamp(min=0, max=self.continual_config.num_tasks - 1)
        )
        task_seen = self.task_prototype_seen[clamped_task_ids].to(device=device)
        if not task_seen.any():
            return global_prior

        task_prior = self.task_prototype_memory[clamped_task_ids].to(device=device, dtype=dtype)
        task_prior = task_prior.clamp_min(self.continual_config.prototype_prior_floor)
        task_prior = task_prior / task_prior.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        task_prior = torch.where(task_seen.unsqueeze(-1), task_prior, global_prior)

        task_weights = (
            self.task_transport_weights[clamped_task_ids]
            .to(device=device, dtype=dtype)
            .clamp_min(0.0)
        )
        base_weight = max(float(self.task_transport_base_weight.item()), 1e-6)
        positive_slack = (task_weights - base_weight).clamp_min(0.0)
        mix = positive_slack / (positive_slack + base_weight)
        mix = (self.prototype_gate.capacity_blend * mix).clamp(0.0, 1.0)
        mix = mix * task_seen.to(dtype)

        task_bias = (task_prior - global_prior).clamp_min(0.0)
        bias_mass = task_bias.sum(dim=-1, keepdim=True)
        normalized_bias = torch.where(
            bias_mass > 0,
            task_bias / bias_mass.clamp_min(1e-6),
            torch.zeros_like(task_bias),
        )
        conditioned = global_prior + mix.unsqueeze(-1) * normalized_bias
        conditioned = conditioned.clamp_min(self.continual_config.prototype_prior_floor)
        return conditioned / conditioned.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def _normalize_prototype_vector(
        self,
        vector: torch.Tensor,
        fallback: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if vector.norm().item() > 1e-6:
            return F.normalize(vector, dim=-1)
        if fallback is not None and fallback.norm().item() > 1e-6:
            return F.normalize(fallback, dim=-1)
        return F.normalize(torch.randn_like(vector), dim=-1)

    def _prototype_anchor(self, prototype_index: int) -> torch.Tensor:
        latent = self.prototype_latent_ema[prototype_index]
        embedding = self.prototype_gate.prototype_embeddings[prototype_index]
        return self._normalize_prototype_vector(latent, fallback=embedding)

    def _transport_barycenter(self, weights: torch.Tensor) -> Optional[torch.Tensor]:
        positive_indices = torch.nonzero(weights > 0, as_tuple=False).squeeze(-1)
        if positive_indices.numel() == 0:
            return None

        anchor_weights = []
        anchors = []
        for index in positive_indices.tolist():
            latent = self.prototype_latent_ema[index]
            embedding = self.prototype_gate.prototype_embeddings[index]
            if latent.norm().item() > 1e-6:
                anchors.append(F.normalize(latent, dim=-1))
                anchor_weights.append(weights[index])
            elif embedding.norm().item() > 1e-6:
                anchors.append(F.normalize(embedding, dim=-1))
                anchor_weights.append(weights[index])

        if not anchors:
            return None

        normalized = torch.stack(anchor_weights)
        normalized = normalized / normalized.sum().clamp_min(1e-6)
        barycenter = torch.einsum("p,pd->d", normalized, torch.stack(anchors, dim=0))
        return self._normalize_prototype_vector(barycenter)

    def _birkhoff_lifecycle_transition(
        self,
        usage: torch.Tensor,
        capacity: torch.Tensor,
    ) -> torch.Tensor:
        num_prototypes = self.continual_config.num_prototypes
        device = usage.device
        dtype = usage.dtype
        work_dtype = torch.float64
        anchors = torch.stack(
            [
                self._prototype_anchor(index).detach().to(device=device, dtype=work_dtype)
                for index in range(num_prototypes)
            ],
            dim=0,
        )
        similarity = torch.matmul(anchors, anchors.transpose(0, 1)).clamp(-1.0, 1.0)
        epsilon = max(float(self.continual_config.prototype_birkhoff_epsilon), 1e-6)
        diagonal_bias = float(self.continual_config.prototype_birkhoff_diag_bias)
        gap_weight = float(self.continual_config.prototype_birkhoff_gap_weight)
        usage = usage.detach().to(device=device, dtype=work_dtype)
        capacity = capacity.detach().to(device=device, dtype=work_dtype)
        excess = (usage - capacity).clamp_min(0.0)
        deficit = (capacity - usage).clamp_min(0.0)
        if excess.sum().item() > 1e-6 and deficit.sum().item() > 1e-6:
            source_pressure = excess / excess.sum().clamp_min(1e-6)
            target_pressure = deficit / deficit.sum().clamp_min(1e-6)
            pressure = torch.outer(source_pressure, target_pressure) * float(num_prototypes)
        else:
            pressure = torch.zeros_like(similarity)

        logits = similarity / epsilon
        logits = logits + diagonal_bias * torch.eye(num_prototypes, device=device, dtype=work_dtype)
        logits = logits + gap_weight * pressure
        logits = logits - logits.max()
        transition = torch.exp(logits).clamp_min(1e-12)
        sinkhorn_iters = max(256, int(self.continual_config.prototype_birkhoff_sinkhorn_iters))
        for _ in range(sinkhorn_iters):
            transition = transition / transition.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            transition = transition / transition.sum(dim=0, keepdim=True).clamp_min(1e-9)
        for _ in range(8):
            transition = transition / transition.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            transition = transition / transition.sum(dim=0, keepdim=True).clamp_min(1e-9)
        return transition.to(dtype=dtype)

    def _birkhoff_lifecycle_stats(
        self,
        base_strength: float,
        effective_strength: float = 0.0,
        offdiag_mass: float = 0.0,
        row_error: float = 0.0,
        col_error: float = 0.0,
        pre_gap: float = 0.0,
        post_gap: float = 0.0,
    ) -> Dict[str, float]:
        gate_factor = effective_strength / base_strength if base_strength > 0.0 else 0.0
        return {
            "birkhoff_base_strength": float(base_strength),
            "birkhoff_strength": float(effective_strength),
            "birkhoff_gate_factor": float(gate_factor),
            "birkhoff_offdiag_mass": float(offdiag_mass),
            "birkhoff_applied_offdiag_mass": float(effective_strength * offdiag_mass),
            "birkhoff_row_error": float(row_error),
            "birkhoff_col_error": float(col_error),
            "birkhoff_pre_gap": float(pre_gap),
            "birkhoff_post_gap": float(post_gap),
            "birkhoff_gap_delta": float(post_gap - pre_gap),
        }

    @torch.no_grad()
    def _apply_birkhoff_lifecycle_transport(self) -> Dict[str, float]:
        base_strength = float(
            min(max(self.continual_config.prototype_birkhoff_transport_strength, 0.0), 1.0)
        )
        num_prototypes = self.continual_config.num_prototypes
        if base_strength <= 0.0 or num_prototypes <= 1:
            return self._birkhoff_lifecycle_stats(base_strength=base_strength)

        usage = self.prototype_usage_ema.clone().clamp_min(0.0)
        capacity = self.prototype_capacity_ema.clone().clamp_min(0.0)
        if capacity.sum().item() <= 0:
            capacity.fill_(1.0 / max(1, capacity.numel()))
        pre_gap = float((usage - capacity).abs().mean().item())
        transition = self._birkhoff_lifecycle_transition(usage, capacity)
        transport = transition.transpose(0, 1)
        identity = torch.eye(num_prototypes, device=transition.device, dtype=transition.dtype)
        offdiag_mass = float(
            ((transition * (1.0 - identity)).sum() / max(1, num_prototypes)).item()
        )
        row_error = float((transition.sum(dim=-1) - 1.0).abs().max().item())
        col_error = float((transition.sum(dim=0) - 1.0).abs().max().item())

        def transported_vector_with_strength(
            vector: torch.Tensor, candidate_strength: float
        ) -> torch.Tensor:
            return (1.0 - candidate_strength) * vector + candidate_strength * torch.matmul(
                transport, vector
            )

        def projected_gap(candidate_strength: float) -> float:
            if candidate_strength <= 0.0:
                return pre_gap
            next_usage = transported_vector_with_strength(usage, candidate_strength).clamp_min(0.0)
            next_capacity = transported_vector_with_strength(
                capacity, candidate_strength
            ).clamp_min(0.0)
            return float((next_usage - next_capacity).abs().mean().item())

        effective_strength = base_strength
        if bool(self.continual_config.prototype_birkhoff_adaptive_gate):
            gap_target = float(self.continual_config.prototype_birkhoff_gap_target)
            if gap_target > 0.0:
                effective_strength *= min(1.0, pre_gap / max(gap_target, 1e-12))

            max_applied_offdiag_mass = float(
                self.continual_config.prototype_birkhoff_max_applied_offdiag_mass
            )
            if max_applied_offdiag_mass > 0.0 and offdiag_mass > 0.0:
                effective_strength = min(
                    effective_strength, max_applied_offdiag_mass / offdiag_mass
                )

            gap_tolerance = max(float(self.continual_config.prototype_birkhoff_gap_tolerance), 0.0)
            post_gap = projected_gap(effective_strength)
            for _ in range(16):
                if post_gap <= pre_gap + gap_tolerance:
                    break
                effective_strength *= 0.5
                post_gap = projected_gap(effective_strength)

            if effective_strength < max(
                float(self.continual_config.prototype_birkhoff_min_effective_strength), 0.0
            ):
                effective_strength = 0.0
                post_gap = pre_gap
        else:
            post_gap = projected_gap(effective_strength)

        if effective_strength <= 0.0:
            return self._birkhoff_lifecycle_stats(
                base_strength=base_strength,
                effective_strength=0.0,
                offdiag_mass=offdiag_mass,
                row_error=row_error,
                col_error=col_error,
                pre_gap=pre_gap,
                post_gap=post_gap,
            )

        def transported_vector(vector: torch.Tensor) -> torch.Tensor:
            return transported_vector_with_strength(vector, effective_strength)

        def transported_matrix(matrix: torch.Tensor) -> torch.Tensor:
            return (1.0 - effective_strength) * matrix + effective_strength * torch.matmul(
                transport, matrix
            )

        embeddings = self.prototype_gate.prototype_embeddings.detach()
        mixed_embeddings = transported_matrix(embeddings)
        self.prototype_gate.prototype_embeddings.copy_(F.normalize(mixed_embeddings, dim=-1))
        self.prototype_head_memory.copy_(transported_matrix(self.prototype_head_memory))
        flat_patterns = self.prototype_pattern_memory.view(num_prototypes, -1)
        self.prototype_pattern_memory.copy_(
            transported_matrix(flat_patterns).view_as(self.prototype_pattern_memory)
        )
        mixed_latents = transported_matrix(self.prototype_latent_ema)
        latent_norms = mixed_latents.norm(dim=-1, keepdim=True)
        self.prototype_latent_ema.copy_(
            torch.where(
                latent_norms > 1e-6, mixed_latents / latent_norms.clamp_min(1e-6), mixed_latents
            )
        )
        self.prototype_usage_ema.copy_(transported_vector(self.prototype_usage_ema).clamp_min(0.0))
        self.prototype_capacity_ema.copy_(
            transported_vector(self.prototype_capacity_ema).clamp_min(0.0)
        )
        self.prototype_support_ema.copy_(
            transported_vector(self.prototype_support_ema).clamp(0.0, 1.0)
        )
        self.prototype_excess_ema.copy_(self.prototype_usage_ema - self.prototype_capacity_ema)
        post_gap = float(
            (self.prototype_usage_ema - self.prototype_capacity_ema).abs().mean().item()
        )

        return self._birkhoff_lifecycle_stats(
            base_strength=base_strength,
            effective_strength=effective_strength,
            offdiag_mass=offdiag_mass,
            row_error=row_error,
            col_error=col_error,
            pre_gap=pre_gap,
            post_gap=post_gap,
        )

    def compute_prototype_regularization(
        self,
        prototype_weights: torch.Tensor,
        routing_prior: torch.Tensor,
        pattern_weights: torch.Tensor,
        head_importance: torch.Tensor,
        prototype_capacity: torch.Tensor,
        prototype_latents: torch.Tensor,
        seq_len: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        effective_support = self._effective_support_scores(pattern_weights, seq_len, device)
        overlap_terms = []

        average_usage = prototype_weights.mean(dim=0)
        balance_loss = F.kl_div(
            average_usage.clamp_min(1e-6).log(),
            prototype_capacity.clamp_min(1e-6),
            reduction="sum",
        )

        normalized_prototypes = F.normalize(self.prototype_gate.prototype_embeddings, dim=-1)
        prototype_similarity = torch.matmul(
            normalized_prototypes, normalized_prototypes.transpose(0, 1)
        )
        off_diagonal = prototype_similarity - torch.eye(
            prototype_similarity.size(0),
            device=prototype_similarity.device,
            dtype=prototype_similarity.dtype,
        )
        diversity_loss = off_diagonal.pow(2).mean()

        normalized_latents = F.normalize(prototype_latents, dim=-1)
        anchor_bank = torch.stack(
            [
                self._prototype_anchor(index).to(
                    device=normalized_latents.device,
                    dtype=normalized_latents.dtype,
                )
                for index in range(self.continual_config.num_prototypes)
            ],
            dim=0,
        )
        transport_cost = 1.0 - torch.matmul(normalized_latents, anchor_bank.transpose(0, 1))
        transport_loss_per_sample = (prototype_weights * transport_cost.clamp_min(0.0)).sum(dim=-1)
        transport_loss = transport_loss_per_sample.mean()

        batch = prototype_weights.size(0)
        for left in range(batch):
            for right in range(left + 1, batch):
                overlap_terms.append((effective_support[left] * effective_support[right]).mean())

        seen_mask = self.prototype_memory_seen.to(dtype=prototype_weights.dtype)
        seen_sum = seen_mask.sum()
        if seen_sum.item() > 0:
            normalized_seen_weights = prototype_weights * seen_mask.unsqueeze(0)
            normalized_seen_weights = normalized_seen_weights / normalized_seen_weights.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-6)

            remembered_head = torch.einsum(
                "bp,ph->bh", normalized_seen_weights, self.prototype_head_memory
            )
            routing_stability_loss = torch.sum(
                prototype_weights.clamp_min(1e-6)
                * (prototype_weights.clamp_min(1e-6).log() - routing_prior.clamp_min(1e-6).log()),
                dim=-1,
            ).mean()
            stability_loss = F.mse_loss(head_importance, remembered_head) + routing_stability_loss

            base_masks = self._get_base_pattern_masks(seq_len, device).to(
                dtype=pattern_weights.dtype
            )
            remembered_support = torch.einsum(
                "bp,phm,mhij->bhij",
                normalized_seen_weights,
                self.prototype_pattern_memory,
                base_masks,
            )
            overlap_terms.append((effective_support * remembered_support).mean())
        else:
            stability_loss = head_importance.new_zeros(())
            routing_stability_loss = head_importance.new_zeros(())

        if overlap_terms:
            overlap_loss = torch.stack(overlap_terms).mean()
        else:
            overlap_loss = head_importance.new_zeros(())

        return {
            "overlap_loss": overlap_loss,
            "stability_loss": stability_loss,
            "routing_stability_loss": routing_stability_loss,
            "balance_loss": balance_loss,
            "diversity_loss": diversity_loss,
            "transport_loss": transport_loss,
            "transport_loss_per_sample": transport_loss_per_sample,
        }

    @torch.no_grad()
    def set_task_transport_weights(
        self,
        task_transport_weights: torch.Tensor,
        base_weight: float,
    ):
        if task_transport_weights.dim() != 1:
            raise ValueError("task_transport_weights must have shape [num_tasks]")
        if task_transport_weights.numel() != self.continual_config.num_tasks:
            raise ValueError("task_transport_weights must match the configured num_tasks")
        self.task_transport_weights.copy_(task_transport_weights.to(self.task_transport_weights))
        self.task_transport_base_weight.fill_(max(float(base_weight), 0.0))

    @torch.no_grad()
    def update_prototype_memory(
        self,
        head_importance: torch.Tensor,
        pattern_weights: torch.Tensor,
        prototype_weights: torch.Tensor,
        prototype_capacity: Optional[torch.Tensor] = None,
        prototype_support: Optional[torch.Tensor] = None,
        prototype_latents: Optional[torch.Tensor] = None,
        task_ids: Optional[torch.Tensor] = None,
    ):
        average_usage = prototype_weights.mean(dim=0)
        if prototype_capacity is None:
            prototype_capacity = torch.full_like(average_usage, 1.0 / average_usage.numel())
        else:
            prototype_capacity = prototype_capacity.to(
                device=average_usage.device, dtype=average_usage.dtype
            )
            prototype_capacity = prototype_capacity / prototype_capacity.sum().clamp_min(1e-6)

        if prototype_support is None:
            average_support = (prototype_weights > 0).to(dtype=average_usage.dtype).mean(dim=0)
        else:
            average_support = prototype_support.to(
                device=average_usage.device, dtype=average_usage.dtype
            ).mean(dim=0)

        excess = average_usage - prototype_capacity
        momentum = self.continual_config.prototype_usage_momentum
        self.prototype_usage_ema.mul_(momentum)
        self.prototype_usage_ema.add_(average_usage * (1.0 - momentum))
        self.prototype_capacity_ema.mul_(momentum)
        self.prototype_capacity_ema.add_(prototype_capacity * (1.0 - momentum))
        self.prototype_support_ema.mul_(momentum)
        self.prototype_support_ema.add_(average_support * (1.0 - momentum))
        self.prototype_excess_ema.mul_(momentum)
        self.prototype_excess_ema.add_(excess * (1.0 - momentum))

        if prototype_latents is not None:
            prototype_latents = prototype_latents.to(
                device=prototype_weights.device, dtype=prototype_weights.dtype
            )

        if task_ids is not None:
            task_ids = (
                task_ids.to(device=prototype_weights.device)
                .long()
                .clamp(
                    min=0,
                    max=self.continual_config.num_tasks - 1,
                )
            )
            task_momentum = self.continual_config.prototype_usage_momentum
            for task_id in task_ids.unique(sorted=True).tolist():
                task_mask = task_ids == task_id
                if not task_mask.any():
                    continue
                task_usage = prototype_weights[task_mask].mean(dim=0)
                task_usage = task_usage / task_usage.sum().clamp_min(1e-6)
                if self.task_prototype_seen[task_id]:
                    self.task_prototype_memory[task_id].mul_(task_momentum)
                    self.task_prototype_memory[task_id].add_(task_usage * (1.0 - task_momentum))
                else:
                    self.task_prototype_memory[task_id].copy_(task_usage)
                    self.task_prototype_seen[task_id] = True

        responsibilities = prototype_weights.sum(dim=0)
        for prototype_id in range(prototype_weights.size(-1)):
            responsibility = responsibilities[prototype_id]
            if responsibility.item() <= 0:
                continue

            sample_weights = prototype_weights[:, prototype_id] / responsibility.clamp_min(1e-6)
            prototype_head = torch.einsum("b,bh->h", sample_weights, head_importance)
            prototype_pattern = torch.einsum("b,bhm->hm", sample_weights, pattern_weights)
            prototype_latent = None
            if prototype_latents is not None:
                prototype_latent = torch.einsum("b,bd->d", sample_weights, prototype_latents)

            if self.prototype_memory_seen[prototype_id]:
                self.prototype_head_memory[prototype_id].mul_(self.continual_config.memory_momentum)
                self.prototype_head_memory[prototype_id].add_(
                    prototype_head * (1.0 - self.continual_config.memory_momentum)
                )
                self.prototype_pattern_memory[prototype_id].mul_(
                    self.continual_config.memory_momentum
                )
                self.prototype_pattern_memory[prototype_id].add_(
                    prototype_pattern * (1.0 - self.continual_config.memory_momentum)
                )
                if prototype_latent is not None:
                    self.prototype_latent_ema[prototype_id].mul_(
                        self.continual_config.memory_momentum
                    )
                    self.prototype_latent_ema[prototype_id].add_(
                        prototype_latent * (1.0 - self.continual_config.memory_momentum)
                    )
            else:
                self.prototype_head_memory[prototype_id].copy_(prototype_head)
                self.prototype_pattern_memory[prototype_id].copy_(prototype_pattern)
                if prototype_latent is not None:
                    self.prototype_latent_ema[prototype_id].copy_(prototype_latent)
                self.prototype_memory_seen[prototype_id] = True

    @torch.no_grad()
    def refresh_prototypes(
        self,
        reset_threshold: Optional[float] = None,
        split_threshold: Optional[float] = None,
        noise_scale: Optional[float] = None,
    ) -> Dict[str, float]:
        reset_threshold = (
            self.continual_config.prototype_reset_threshold
            if reset_threshold is None
            else reset_threshold
        )
        split_threshold = (
            self.continual_config.prototype_split_threshold
            if split_threshold is None
            else split_threshold
        )
        noise_scale = (
            self.continual_config.prototype_noise_scale if noise_scale is None else noise_scale
        )

        usage = self.prototype_usage_ema.clone()
        capacity = self.prototype_capacity_ema.clone()
        support = self.prototype_support_ema.clone()
        excess = self.prototype_excess_ema.clone()
        relocation_strength = min(
            max(self.continual_config.prototype_relocation_strength, 0.0), 1.0
        )
        merge_threshold = min(max(self.continual_config.prototype_merge_threshold, -1.0), 1.0)
        merge_usage_threshold = max(0.0, self.continual_config.prototype_merge_usage_threshold)
        if capacity.sum().item() <= 0:
            capacity.fill_(1.0 / max(1, capacity.numel()))
            excess = usage - capacity

        deficit = capacity - usage
        gap = (usage - capacity).abs()
        deficit_barycenter = self._transport_barycenter(deficit.clamp_min(0.0))

        split_candidates = (
            torch.nonzero(
                (excess > split_threshold)
                | ((usage > split_threshold) & (support > reset_threshold)),
                as_tuple=False,
            )
            .squeeze(-1)
            .tolist()
        )
        split_candidates = sorted(
            split_candidates,
            key=lambda idx: (float(excess[idx]), float(usage[idx])),
            reverse=True,
        )

        target_candidates = (
            torch.nonzero(
                (deficit > reset_threshold)
                | ((usage < reset_threshold) & (support < split_threshold)),
                as_tuple=False,
            )
            .squeeze(-1)
            .tolist()
        )
        target_candidates = sorted(
            target_candidates,
            key=lambda idx: (float(deficit[idx]), float(-usage[idx])),
            reverse=True,
        )

        reset_count = 0
        split_count = 0
        merge_count = 0
        merge_similarity_total = 0.0
        used_targets = set()
        reset_skip = set()

        for source_index in split_candidates:
            target_index = None
            for candidate in target_candidates:
                if candidate != source_index and candidate not in used_targets:
                    target_index = candidate
                    break
            if target_index is None:
                break

            source_embedding = self.prototype_gate.prototype_embeddings[source_index]
            target_embedding = self.prototype_gate.prototype_embeddings[target_index]
            source_anchor = self._prototype_anchor(source_index)
            if deficit_barycenter is None:
                deficit_barycenter = source_anchor
            transfer_share = deficit[target_index] / (
                deficit[target_index] + excess[source_index].clamp_min(1e-6)
            )
            transfer_share = transfer_share.clamp(0.25, 0.75)
            transport_anchor = self._normalize_prototype_vector(
                (1.0 - transfer_share) * source_anchor + transfer_share * deficit_barycenter,
                fallback=source_anchor,
            )
            source_relocated = self._normalize_prototype_vector(
                (1.0 - 0.5 * relocation_strength) * source_embedding
                + (0.5 * relocation_strength) * source_anchor,
                fallback=source_anchor,
            )
            target_relocated = self._normalize_prototype_vector(
                (1.0 - relocation_strength) * target_embedding
                + relocation_strength * transport_anchor,
                fallback=transport_anchor,
            )
            noise = torch.randn_like(source_embedding) * noise_scale
            self.prototype_gate.prototype_embeddings[source_index].copy_(source_relocated)
            self.prototype_gate.prototype_embeddings[target_index].copy_(
                self._normalize_prototype_vector(
                    target_relocated + noise, fallback=transport_anchor
                )
            )
            self.prototype_head_memory[target_index].copy_(self.prototype_head_memory[source_index])
            self.prototype_pattern_memory[target_index].copy_(
                self.prototype_pattern_memory[source_index]
            )
            self.prototype_memory_seen[target_index] = self.prototype_memory_seen[source_index]
            self.prototype_latent_ema[target_index].copy_(transport_anchor)
            self.prototype_latent_ema[source_index].copy_(source_anchor)

            source_usage = self.prototype_usage_ema[source_index].clone()
            source_capacity = self.prototype_capacity_ema[source_index].clone()
            source_support = self.prototype_support_ema[source_index].clone()

            self.prototype_usage_ema[source_index].copy_(source_usage * (1.0 - transfer_share))
            self.prototype_usage_ema[target_index].copy_(source_usage * transfer_share)
            self.prototype_capacity_ema[source_index].copy_(
                source_capacity * (1.0 - transfer_share)
            )
            self.prototype_capacity_ema[target_index].copy_(source_capacity * transfer_share)
            self.prototype_support_ema[source_index].copy_(source_support * (1.0 - transfer_share))
            self.prototype_support_ema[target_index].copy_(source_support * transfer_share)
            self.prototype_excess_ema[source_index].copy_(
                self.prototype_usage_ema[source_index] - self.prototype_capacity_ema[source_index]
            )
            self.prototype_excess_ema[target_index].copy_(
                self.prototype_usage_ema[target_index] - self.prototype_capacity_ema[target_index]
            )

            used_targets.add(target_index)
            reset_skip.add(target_index)
            split_count += 1

        usage = self.prototype_usage_ema.clone()
        capacity = self.prototype_capacity_ema.clone()
        support = self.prototype_support_ema.clone()
        if capacity.sum().item() <= 0:
            capacity.fill_(1.0 / max(1, capacity.numel()))
        excess = usage - capacity
        deficit = capacity - usage
        gap = excess.abs()
        deficit_barycenter = self._transport_barycenter(deficit.clamp_min(0.0))

        seen_indices = (
            torch.nonzero(self.prototype_memory_seen, as_tuple=False).squeeze(-1).tolist()
        )
        if len(seen_indices) >= 2:
            anchors = torch.stack(
                [
                    self._prototype_anchor(index)
                    for index in range(self.continual_config.num_prototypes)
                ],
                dim=0,
            )
            similarity = torch.matmul(anchors, anchors.transpose(0, 1))
            merge_candidates = []
            for left_pos, left_index in enumerate(seen_indices):
                for right_index in seen_indices[left_pos + 1 :]:
                    pair_similarity = float(similarity[left_index, right_index].item())
                    lower_usage = float(min(usage[left_index].item(), usage[right_index].item()))
                    pair_excess = float(max(excess[left_index].item(), excess[right_index].item()))
                    if pair_similarity < merge_threshold:
                        continue
                    if lower_usage >= merge_usage_threshold:
                        continue
                    if pair_excess > split_threshold:
                        continue
                    merge_candidates.append((pair_similarity, left_index, right_index))

            consumed = set()
            for pair_similarity, left_index, right_index in sorted(merge_candidates, reverse=True):
                if left_index in consumed or right_index in consumed:
                    continue

                keep_index, merge_index = (
                    (left_index, right_index)
                    if usage[left_index].item() >= usage[right_index].item()
                    else (right_index, left_index)
                )
                keep_anchor = self._prototype_anchor(keep_index)
                merge_anchor = self._prototype_anchor(merge_index)
                keep_weight = float((usage[keep_index] + capacity[keep_index]).item())
                merge_weight = float((usage[merge_index] + capacity[merge_index]).item())
                total_weight = max(1e-6, keep_weight + merge_weight)
                merged_anchor = self._normalize_prototype_vector(
                    (keep_weight / total_weight) * keep_anchor
                    + (merge_weight / total_weight) * merge_anchor,
                    fallback=keep_anchor,
                )
                current_keep = self.prototype_gate.prototype_embeddings[keep_index]
                self.prototype_gate.prototype_embeddings[keep_index].copy_(
                    self._normalize_prototype_vector(
                        (1.0 - relocation_strength) * current_keep
                        + relocation_strength * merged_anchor,
                        fallback=merged_anchor,
                    )
                )
                self.prototype_head_memory[keep_index].copy_(
                    (
                        keep_weight * self.prototype_head_memory[keep_index]
                        + merge_weight * self.prototype_head_memory[merge_index]
                    )
                    / total_weight
                )
                self.prototype_pattern_memory[keep_index].copy_(
                    (
                        keep_weight * self.prototype_pattern_memory[keep_index]
                        + merge_weight * self.prototype_pattern_memory[merge_index]
                    )
                    / total_weight
                )
                self.prototype_memory_seen[keep_index] = (
                    self.prototype_memory_seen[keep_index] | self.prototype_memory_seen[merge_index]
                )
                self.prototype_latent_ema[keep_index].copy_(merged_anchor)

                merged_usage = (
                    self.prototype_usage_ema[keep_index] + self.prototype_usage_ema[merge_index]
                ).clamp_max(1.0)
                merged_capacity = (
                    self.prototype_capacity_ema[keep_index]
                    + self.prototype_capacity_ema[merge_index]
                ).clamp_max(1.0)
                merged_support = 1.0 - (
                    (1.0 - self.prototype_support_ema[keep_index].clamp(0.0, 1.0))
                    * (1.0 - self.prototype_support_ema[merge_index].clamp(0.0, 1.0))
                )
                self.prototype_usage_ema[keep_index].copy_(merged_usage)
                self.prototype_capacity_ema[keep_index].copy_(merged_capacity)
                self.prototype_support_ema[keep_index].copy_(merged_support)
                self.prototype_excess_ema[keep_index].copy_(merged_usage - merged_capacity)

                current_merge = self.prototype_gate.prototype_embeddings[merge_index]
                if deficit_barycenter is not None:
                    relocated = self._normalize_prototype_vector(
                        (1.0 - relocation_strength) * current_merge
                        + relocation_strength * deficit_barycenter,
                        fallback=deficit_barycenter,
                    )
                    new_embedding = self._normalize_prototype_vector(
                        relocated + torch.randn_like(current_merge) * noise_scale,
                        fallback=deficit_barycenter,
                    )
                    self.prototype_latent_ema[merge_index].copy_(deficit_barycenter)
                else:
                    new_embedding = F.normalize(torch.randn_like(current_merge), dim=-1)
                    self.prototype_latent_ema[merge_index].zero_()
                self.prototype_gate.prototype_embeddings[merge_index].copy_(new_embedding)
                self.prototype_head_memory[merge_index].zero_()
                self.prototype_pattern_memory[merge_index].zero_()
                self.prototype_memory_seen[merge_index] = False
                self.prototype_usage_ema[merge_index].zero_()
                self.prototype_capacity_ema[merge_index].zero_()
                self.prototype_support_ema[merge_index].zero_()
                self.prototype_excess_ema[merge_index].zero_()

                consumed.update({keep_index, merge_index})
                reset_skip.add(merge_index)
                merge_count += 1
                merge_similarity_total += pair_similarity

        usage = self.prototype_usage_ema.clone()
        capacity = self.prototype_capacity_ema.clone()
        support = self.prototype_support_ema.clone()
        if capacity.sum().item() <= 0:
            capacity.fill_(1.0 / max(1, capacity.numel()))
        excess = usage - capacity
        deficit = capacity - usage
        gap = excess.abs()
        deficit_barycenter = self._transport_barycenter(deficit.clamp_min(0.0))

        reset_candidates = [
            idx
            for idx in range(self.continual_config.num_prototypes)
            if idx not in reset_skip
            and usage[idx].item() < reset_threshold
            and capacity[idx].item() < reset_threshold
            and support[idx].item() < reset_threshold
        ]
        for target_index in reset_candidates:
            current_embedding = self.prototype_gate.prototype_embeddings[target_index]
            if deficit_barycenter is not None:
                relocated = self._normalize_prototype_vector(
                    (1.0 - relocation_strength) * current_embedding
                    + relocation_strength * deficit_barycenter,
                    fallback=deficit_barycenter,
                )
                new_embedding = self._normalize_prototype_vector(
                    relocated + torch.randn_like(current_embedding) * noise_scale,
                    fallback=deficit_barycenter,
                )
                self.prototype_latent_ema[target_index].copy_(deficit_barycenter)
            else:
                new_embedding = F.normalize(torch.randn_like(current_embedding), dim=-1)
                self.prototype_latent_ema[target_index].zero_()
            self.prototype_gate.prototype_embeddings[target_index].copy_(new_embedding)
            self.prototype_head_memory[target_index].zero_()
            self.prototype_pattern_memory[target_index].zero_()
            self.prototype_memory_seen[target_index] = False
            self.prototype_usage_ema[target_index].zero_()
            self.prototype_capacity_ema[target_index].zero_()
            self.prototype_support_ema[target_index].zero_()
            self.prototype_excess_ema[target_index].zero_()
            reset_count += 1

        birkhoff_stats = self._apply_birkhoff_lifecycle_transport()
        usage = self.prototype_usage_ema.clone()
        capacity = self.prototype_capacity_ema.clone()
        if capacity.sum().item() <= 0:
            capacity.fill_(1.0 / max(1, capacity.numel()))
        excess = usage - capacity
        gap = excess.abs()

        return {
            "reset_count": reset_count,
            "split_count": split_count,
            "merge_count": merge_count,
            "mean_transport_gap": float(gap.mean().item()),
            "max_transport_gap": float(gap.max().item()),
            "mean_excess": float(excess.mean().item()),
            "mean_merge_similarity": float(merge_similarity_total / max(1, merge_count)),
            **birkhoff_stats,
        }

    def forward(
        self,
        x: torch.Tensor,
        task_ids: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_info: bool = False,
    ):
        batch, seq_len, dim = x.shape
        residual = x
        x = self.norm(x)

        routing_prior = self._build_prototype_prior(batch, x.device, x.dtype, task_ids=task_ids)
        gate_info = self.prototype_gate(x, routing_prior=routing_prior)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda tensor: tensor.reshape(batch, seq_len, self.num_heads, self.dim_head).transpose(
                1, 2
            ),
            qkv,
        )
        q = q * self.scale

        normalized_mask = None
        if mask is not None:
            normalized_mask = normalize_attention_mask(mask, batch, self.num_heads, seq_len)

        attn_out, prototype_pattern_masks = self._compute_grouped_pattern_attention(
            q,
            k,
            v,
            gate_info["pattern_logits"],
            seq_len,
            x.device,
            normalized_mask=normalized_mask,
            return_pattern_masks=return_info,
            group_hint_ids=task_ids,
        )
        attn_out = attn_out.transpose(1, 2).reshape(batch, seq_len, dim)
        attn_out = self.to_out(attn_out)

        x = residual + attn_out

        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = residual + x

        if not return_info:
            return x, None

        regularization = self.compute_prototype_regularization(
            prototype_weights=gate_info["prototype_weights"],
            routing_prior=gate_info["prototype_prior"],
            pattern_weights=gate_info["pattern_weights"],
            head_importance=gate_info["head_importance"],
            prototype_capacity=gate_info["prototype_capacity"],
            prototype_latents=gate_info["prototype_latents"],
            seq_len=seq_len,
            device=x.device,
        )
        selected_patterns = (
            gate_info["pattern_logits"]
            .topk(
                k=min(self.continual_config.top_k_patterns, len(self.pattern_bank)),
                dim=-1,
            )
            .indices
        )

        info = {
            "prototype_logits": gate_info["prototype_logits"],
            "proximal_logits": gate_info["proximal_logits"],
            "prototype_prior": gate_info["prototype_prior"],
            "prototype_capacity": gate_info["prototype_capacity"],
            "prototype_target_capacity": gate_info["prototype_target_capacity"],
            "prototype_support": gate_info["prototype_support"],
            "prototype_weights": gate_info["prototype_weights"],
            "prototype_usage": self.prototype_usage_ema.detach().clone(),
            "prototype_latents": gate_info["prototype_latents"],
            "prototype_capacity_ema": self.prototype_capacity_ema.detach().clone(),
            "prototype_support_ema": self.prototype_support_ema.detach().clone(),
            "prototype_excess_ema": self.prototype_excess_ema.detach().clone(),
            "pattern_logits": gate_info["pattern_logits"],
            "pattern_weights": gate_info["pattern_weights"],
            "head_importance": gate_info["head_importance"],
            "selected_patterns": selected_patterns,
            "prototype_pattern_masks": prototype_pattern_masks,
            **regularization,
        }
        for key in [
            "candidate_support_residual",
            "support_projection_residual",
            "support_residual_delta",
            "target_capacity_residual",
            "effective_capacity_residual",
            "support_density",
            "support_size",
            "support_active_prototypes",
            "support_weight_leakage",
            "capacity_bias_selection_rate",
        ]:
            info[key] = gate_info[key]
        return x, info
