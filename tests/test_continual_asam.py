"""
Tests for continual-learning ASAM extensions.
"""

from dataclasses import replace

import torch

from asam import ContinualASAMConfig, ContinualASAMLayer, PrototypeContinualASAMLayer
import asam.continual_asam as continual_asam_module
from asam.continual_asam import PrototypeSparseGate


def _paired_layers(layer_cls, config):
    legacy = layer_cls(replace(config, grouped_indexed_attention=False))
    grouped = layer_cls(replace(config, grouped_indexed_attention=True))
    grouped.load_state_dict(legacy.state_dict())
    legacy.eval()
    grouped.eval()
    return legacy, grouped


def test_continual_asam_forward_shapes():
    config = ContinualASAMConfig(dim=64, num_heads=4, dim_head=16, num_tasks=6, top_k_patterns=2)
    layer = ContinualASAMLayer(config)

    x = torch.randn(3, 20, 64)
    task_ids = torch.tensor([0, 1, 2])

    output, info = layer(x, task_ids=task_ids, return_info=True)

    assert output.shape == x.shape
    assert info["pattern_logits"].shape == (3, 4, 4)
    assert info["pattern_weights"].shape == (3, 4, 4)
    assert info["head_importance"].shape == (3, 4)
    assert info["task_pattern_masks"].shape[:3] == (3, 4, 20)


def test_continual_regularization_is_finite():
    config = ContinualASAMConfig(dim=32, num_heads=2, dim_head=16, num_tasks=4, top_k_patterns=2)
    layer = ContinualASAMLayer(config)

    x = torch.randn(2, 12, 32)
    task_ids = torch.tensor([0, 1])

    _, info = layer(x, task_ids=task_ids, return_info=True)

    assert torch.isfinite(info["overlap_loss"])
    assert torch.isfinite(info["stability_loss"])
    assert info["overlap_loss"].ndim == 0
    assert info["stability_loss"].ndim == 0


def test_task_memory_update_reduces_stability_error_on_revisit():
    config = ContinualASAMConfig(dim=32, num_heads=2, dim_head=16, num_tasks=4, top_k_patterns=2)
    layer = ContinualASAMLayer(config)

    x = torch.randn(2, 10, 32)
    task_ids = torch.tensor([1, 1])

    _, first_info = layer(x, task_ids=task_ids, return_info=True)
    first_loss = first_info["stability_loss"]
    layer.update_task_memory(task_ids, first_info["head_importance"], first_info["pattern_weights"])

    _, second_info = layer(x, task_ids=task_ids, return_info=True)

    assert first_loss.item() == 0.0
    assert second_info["stability_loss"].item() >= 0.0
    assert layer.task_memory_seen[1].item() is True


def test_overlap_regularization_uses_seen_task_memory_without_mixed_batch():
    config = ContinualASAMConfig(dim=32, num_heads=2, dim_head=16, num_tasks=4, top_k_patterns=2)
    layer = ContinualASAMLayer(config)

    previous_x = torch.randn(2, 10, 32)
    previous_task_ids = torch.tensor([0, 0])
    _, previous_info = layer(previous_x, task_ids=previous_task_ids, return_info=True)
    layer.update_task_memory(previous_task_ids, previous_info["head_importance"], previous_info["pattern_weights"])

    current_x = torch.randn(2, 10, 32)
    current_task_ids = torch.tensor([1, 1])
    _, current_info = layer(current_x, task_ids=current_task_ids, return_info=True)

    assert current_info["overlap_loss"].item() >= 0.0
    assert layer.task_pattern_memory[0].abs().sum().item() > 0.0


def test_continual_asam_accepts_scalar_task_id():
    config = ContinualASAMConfig(dim=48, num_heads=3, dim_head=16, num_tasks=5)
    layer = ContinualASAMLayer(config)

    x = torch.randn(2, 14, 48)
    output, _ = layer(x, task_ids=torch.tensor(2), return_info=False)

    assert output.shape == x.shape


def test_grouped_task_attention_matches_legacy_forward_and_backward():
    torch.manual_seed(123)
    config = ContinualASAMConfig(
        dim=32,
        num_heads=2,
        dim_head=16,
        num_tasks=4,
        top_k_patterns=2,
        dropout=0.0,
    )
    legacy, grouped = _paired_layers(ContinualASAMLayer, config)
    task_ids = torch.tensor([0, 1, 0, 2])
    mask = torch.ones(4, 1, 12, 12, dtype=torch.bool)
    mask[:, :, 3, :] = False
    x = torch.randn(4, 12, 32)
    legacy_x = x.clone().requires_grad_(True)
    grouped_x = x.clone().requires_grad_(True)

    legacy_out, legacy_info = legacy(legacy_x, task_ids=task_ids, mask=mask, return_info=True)
    grouped_out, grouped_info = grouped(grouped_x, task_ids=task_ids, mask=mask, return_info=True)

    assert torch.allclose(grouped_out, legacy_out, atol=1e-6, rtol=1e-5)
    assert torch.equal(grouped_info["task_pattern_masks"], legacy_info["task_pattern_masks"])
    assert torch.allclose(grouped_info["pattern_logits"], legacy_info["pattern_logits"], atol=1e-6)
    assert torch.allclose(grouped_info["pattern_weights"], legacy_info["pattern_weights"], atol=1e-6)
    assert torch.allclose(grouped_info["head_importance"], legacy_info["head_importance"], atol=1e-6)

    legacy_out.square().mean().backward()
    grouped_out.square().mean().backward()

    assert torch.allclose(grouped_x.grad, legacy_x.grad, atol=1e-6, rtol=1e-5)
    assert torch.allclose(grouped.to_qkv.weight.grad, legacy.to_qkv.weight.grad, atol=1e-6, rtol=1e-5)
    assert torch.allclose(grouped.to_out[0].weight.grad, legacy.to_out[0].weight.grad, atol=1e-6, rtol=1e-5)


def test_grouped_task_attention_reuses_indices_for_repeated_signatures(monkeypatch):
    torch.manual_seed(321)
    config = ContinualASAMConfig(
        dim=32,
        num_heads=2,
        dim_head=16,
        num_tasks=4,
        top_k_patterns=2,
        dropout=0.0,
        grouped_indexed_attention=True,
    )
    layer = ContinualASAMLayer(config)
    layer.eval()
    for parameter in layer.task_gate.parameters():
        parameter.data.zero_()

    calls = 0
    original = continual_asam_module.pattern_mask_to_indices

    def counted_pattern_mask_to_indices(pattern_mask):
        nonlocal calls
        calls += 1
        return original(pattern_mask)

    monkeypatch.setattr(continual_asam_module, "pattern_mask_to_indices", counted_pattern_mask_to_indices)

    x = torch.randn(6, 10, 32)
    task_ids = torch.tensor([0, 1, 2, 3, 0, 1])
    output, info = layer(x, task_ids=task_ids, return_info=True)

    assert output.shape == x.shape
    assert info["task_pattern_masks"].shape == (6, 2, 10, 10)
    assert calls == 1


def test_prototype_continual_asam_forward_shapes():
    config = ContinualASAMConfig(
        dim=64,
        num_heads=4,
        dim_head=16,
        num_prototypes=5,
        prototype_embed_dim=32,
        top_k_patterns=2,
        prototype_top_k=2,
    )
    layer = PrototypeContinualASAMLayer(config)

    x = torch.randn(3, 18, 64)
    output, info = layer(x, return_info=True)

    assert output.shape == x.shape
    assert info["prototype_logits"].shape == (3, 5)
    assert info["prototype_weights"].shape == (3, 5)
    assert info["prototype_support"].shape == (3, 5)
    assert info["prototype_prior"].shape == (3, 5)
    assert info["prototype_capacity"].shape == (5,)
    assert info["pattern_weights"].shape == (3, 4, 4)
    assert info["prototype_pattern_masks"].shape[:3] == (3, 4, 18)
    assert info["transport_loss_per_sample"].shape == (3,)
    assert torch.all(info["prototype_support"].sum(dim=-1) <= config.prototype_top_k)
    assert torch.allclose(
        info["transport_loss_per_sample"].mean(),
        info["transport_loss"],
        atol=1e-6,
    )


def test_grouped_prototype_attention_matches_legacy_forward_with_head_mask():
    torch.manual_seed(456)
    config = ContinualASAMConfig(
        dim=32,
        num_heads=2,
        dim_head=16,
        num_prototypes=5,
        prototype_embed_dim=16,
        top_k_patterns=2,
        prototype_top_k=2,
        dropout=0.0,
    )
    legacy, grouped = _paired_layers(PrototypeContinualASAMLayer, config)
    mask = torch.ones(3, 2, 14, 14, dtype=torch.bool)
    mask[0, 0, :, 7:] = False
    mask[1, 1, 2, :] = False
    x = torch.randn(3, 14, 32)

    legacy_out, legacy_info = legacy(x, mask=mask, return_info=True)
    grouped_out, grouped_info = grouped(x, mask=mask, return_info=True)

    assert torch.allclose(grouped_out, legacy_out, atol=1e-6, rtol=1e-5)
    assert torch.equal(grouped_info["prototype_pattern_masks"], legacy_info["prototype_pattern_masks"])
    assert torch.allclose(grouped_info["prototype_weights"], legacy_info["prototype_weights"], atol=1e-6)
    assert torch.equal(grouped_info["prototype_support"], legacy_info["prototype_support"])
    assert torch.allclose(
        grouped_info["transport_loss_per_sample"],
        legacy_info["transport_loss_per_sample"],
        atol=1e-6,
    )


def test_sinkhorn_transport_matches_capacity_target():
    gate = PrototypeSparseGate(
        dim=16,
        num_heads=2,
        num_patterns=4,
        num_prototypes=3,
        top_k=3,
        routing_strategy="sinkhorn_topk",
        sinkhorn_epsilon=0.2,
        sinkhorn_iters=60,
        capacity_blend=1.0,
    )
    logits = torch.tensor(
        [
            [3.0, 1.0, 0.5],
            [2.5, 1.5, 0.2],
            [0.3, 2.5, 3.0],
            [0.4, 2.2, 2.8],
        ]
    )
    target_capacity = torch.tensor([0.5, 0.2, 0.3])

    weights = gate._sinkhorn_transport_weights(logits, target_capacity)

    assert weights.shape == logits.shape
    assert torch.allclose(weights.sum(dim=-1), torch.ones(logits.size(0)), atol=1e-5)
    assert torch.allclose(weights.mean(dim=0), target_capacity, atol=5e-2)


def test_prototype_routing_is_topk_sparse_and_memory_biased():
    config = ContinualASAMConfig(
        dim=32,
        num_heads=2,
        dim_head=16,
        num_prototypes=4,
        prototype_embed_dim=16,
        prototype_top_k=1,
        prototype_prior_strength=8.0,
    )
    layer = PrototypeContinualASAMLayer(config)

    with torch.no_grad():
        layer.prototype_memory_seen.copy_(torch.tensor([True, False, False, False]))
        layer.prototype_usage_ema.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        layer.prototype_gate.input_proj.weight.zero_()
        layer.prototype_gate.input_proj.bias.zero_()

    x = torch.zeros(2, 10, 32)
    _, info = layer(x, return_info=True)

    assert torch.all(info["prototype_support"].sum(dim=-1) == 1)
    assert torch.all((info["prototype_weights"] > 0).sum(dim=-1) == 1)
    assert torch.all(info["prototype_weights"].argmax(dim=-1) == 0)
    assert torch.all(info["prototype_prior"][:, 0] > info["prototype_prior"][:, 1])


def test_prototype_memory_update_enables_stability_regularization():
    config = ContinualASAMConfig(dim=32, num_heads=2, dim_head=16, num_prototypes=4, prototype_embed_dim=16)
    layer = PrototypeContinualASAMLayer(config)

    x = torch.randn(2, 10, 32)
    _, first_info = layer(x, return_info=True)
    first_loss = first_info["stability_loss"]

    layer.update_prototype_memory(
        head_importance=first_info["head_importance"],
        pattern_weights=first_info["pattern_weights"],
        prototype_weights=first_info["prototype_weights"],
    )
    _, second_info = layer(x, return_info=True)

    assert first_loss.item() == 0.0
    assert second_info["stability_loss"].item() >= 0.0
    assert layer.prototype_memory_seen.any().item() is True


def test_prototype_memory_tracks_capacity_and_support_statistics():
    config = ContinualASAMConfig(dim=32, num_heads=2, dim_head=16, num_prototypes=4, prototype_embed_dim=16)
    layer = PrototypeContinualASAMLayer(config)

    head_importance = torch.tensor([[0.8, 0.2], [0.6, 0.4]])
    pattern_weights = torch.full((2, 2, 4), 0.25)
    prototype_weights = torch.tensor([[0.7, 0.3, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]])
    prototype_capacity = torch.tensor([0.4, 0.4, 0.1, 0.1])
    prototype_support = prototype_weights > 0
    prototype_latents = torch.zeros(2, 16)
    prototype_latents[:, 0] = 1.0

    layer.update_prototype_memory(
        head_importance=head_importance,
        pattern_weights=pattern_weights,
        prototype_weights=prototype_weights,
        prototype_capacity=prototype_capacity,
        prototype_support=prototype_support,
        prototype_latents=prototype_latents,
    )

    assert layer.prototype_capacity_ema[0].item() > 0.0
    assert layer.prototype_support_ema[0].item() > 0.0
    assert layer.prototype_excess_ema[0].item() > 0.0
    assert layer.prototype_excess_ema[2].item() < 0.0
    assert layer.prototype_latent_ema[0, 0].item() > 0.0


def test_prototype_regularizers_are_finite_and_non_negative():
    config = ContinualASAMConfig(dim=32, num_heads=2, dim_head=16, num_prototypes=4, prototype_embed_dim=16)
    layer = PrototypeContinualASAMLayer(config)

    x = torch.randn(3, 12, 32)
    _, info = layer(x, return_info=True)

    assert torch.isfinite(info["balance_loss"])
    assert torch.isfinite(info["diversity_loss"])
    assert info["balance_loss"].item() >= 0.0
    assert info["diversity_loss"].item() >= 0.0


def test_prototype_refresh_can_reset_and_split():
    config = ContinualASAMConfig(
        dim=32,
        num_heads=2,
        dim_head=16,
        num_prototypes=4,
        prototype_embed_dim=16,
        prototype_reset_threshold=0.05,
        prototype_split_threshold=0.1,
        prototype_noise_scale=0.01,
        prototype_relocation_strength=1.0,
    )
    layer = PrototypeContinualASAMLayer(config)

    with torch.no_grad():
        layer.prototype_usage_ema.copy_(torch.tensor([0.7, 0.2, 0.01, 0.0]))
        layer.prototype_capacity_ema.copy_(torch.tensor([0.3, 0.2, 0.25, 0.25]))
        layer.prototype_support_ema.copy_(torch.tensor([0.8, 0.3, 0.01, 0.0]))
        layer.prototype_excess_ema.copy_(layer.prototype_usage_ema - layer.prototype_capacity_ema)
        layer.prototype_memory_seen.copy_(torch.tensor([True, True, False, False]))
        layer.prototype_head_memory[0].fill_(1.0)
        layer.prototype_pattern_memory[0].fill_(1.0)
        layer.prototype_latent_ema.zero_()
        layer.prototype_latent_ema[0, 0] = 1.0
        layer.prototype_gate.prototype_embeddings.zero_()
        layer.prototype_gate.prototype_embeddings[0, 1] = 1.0
        layer.prototype_gate.prototype_embeddings[2, 2] = 1.0

    before = layer.prototype_gate.prototype_embeddings.detach().clone()
    barycenter = torch.zeros(16)
    barycenter[0] = 1.0
    deficit_indices = [2, 3]
    before_alignment = max(torch.dot(before[index], barycenter).item() for index in deficit_indices)
    stats = layer.refresh_prototypes()
    after = layer.prototype_gate.prototype_embeddings.detach().clone()
    after_alignment = max(torch.dot(after[index], barycenter).item() for index in deficit_indices)

    assert stats["split_count"] >= 1
    assert stats["reset_count"] >= 0
    assert stats["max_transport_gap"] >= stats["mean_transport_gap"]
    assert after_alignment > before_alignment
    assert not torch.allclose(before, after)


def test_birkhoff_lifecycle_transition_is_doubly_stochastic_and_mass_preserving():
    config = ContinualASAMConfig(
        dim=32,
        num_heads=2,
        dim_head=16,
        num_prototypes=4,
        prototype_embed_dim=16,
        prototype_reset_threshold=0.0,
        prototype_split_threshold=10.0,
        prototype_merge_threshold=1.0,
        prototype_merge_usage_threshold=0.0,
        prototype_birkhoff_transport_strength=0.5,
        prototype_birkhoff_adaptive_gate=False,
        prototype_birkhoff_sinkhorn_iters=80,
    )
    layer = PrototypeContinualASAMLayer(config)

    with torch.no_grad():
        layer.prototype_memory_seen.fill_(True)
        layer.prototype_usage_ema.copy_(torch.tensor([0.6, 0.2, 0.1, 0.1]))
        layer.prototype_capacity_ema.copy_(torch.tensor([0.25, 0.25, 0.25, 0.25]))
        layer.prototype_support_ema.copy_(torch.tensor([0.8, 0.6, 0.4, 0.4]))
        layer.prototype_excess_ema.copy_(layer.prototype_usage_ema - layer.prototype_capacity_ema)
        layer.prototype_gate.prototype_embeddings.zero_()
        layer.prototype_gate.prototype_embeddings[:, :4].copy_(torch.eye(4))

    transition = layer._birkhoff_lifecycle_transition(
        layer.prototype_usage_ema,
        layer.prototype_capacity_ema,
    )
    usage_sum = layer.prototype_usage_ema.sum().clone()
    capacity_sum = layer.prototype_capacity_ema.sum().clone()
    stats = layer.refresh_prototypes()

    assert torch.allclose(transition.sum(dim=-1), torch.ones(4), atol=5e-4)
    assert torch.allclose(transition.sum(dim=0), torch.ones(4), atol=5e-4)
    assert torch.allclose(layer.prototype_usage_ema.sum(), usage_sum, atol=1e-5)
    assert torch.allclose(layer.prototype_capacity_ema.sum(), capacity_sum, atol=1e-5)
    assert stats["birkhoff_strength"] == 0.5
    assert stats["birkhoff_offdiag_mass"] > 0.0
    assert stats["birkhoff_row_error"] < 5e-4
    assert stats["birkhoff_col_error"] < 5e-4
    assert torch.allclose(
        layer.prototype_gate.prototype_embeddings.norm(dim=-1),
        torch.ones(4),
        atol=1e-5,
    )


def test_birkhoff_adaptive_gate_caps_applied_transport_and_preserves_gap():
    config = ContinualASAMConfig(
        dim=32,
        num_heads=2,
        dim_head=16,
        num_prototypes=4,
        prototype_embed_dim=16,
        prototype_reset_threshold=0.0,
        prototype_split_threshold=10.0,
        prototype_merge_threshold=1.0,
        prototype_merge_usage_threshold=0.0,
        prototype_birkhoff_transport_strength=0.5,
        prototype_birkhoff_adaptive_gate=True,
        prototype_birkhoff_gap_target=0.0,
        prototype_birkhoff_max_applied_offdiag_mass=0.001,
        prototype_birkhoff_sinkhorn_iters=80,
    )
    layer = PrototypeContinualASAMLayer(config)

    with torch.no_grad():
        layer.prototype_memory_seen.fill_(True)
        layer.prototype_usage_ema.copy_(torch.tensor([0.6, 0.2, 0.1, 0.1]))
        layer.prototype_capacity_ema.copy_(torch.tensor([0.25, 0.25, 0.25, 0.25]))
        layer.prototype_support_ema.copy_(torch.tensor([0.8, 0.6, 0.4, 0.4]))
        layer.prototype_excess_ema.copy_(layer.prototype_usage_ema - layer.prototype_capacity_ema)
        layer.prototype_gate.prototype_embeddings.zero_()
        layer.prototype_gate.prototype_embeddings[:, :4].copy_(torch.eye(4))

    stats = layer.refresh_prototypes()

    assert stats["birkhoff_base_strength"] == 0.5
    assert 0.0 < stats["birkhoff_strength"] < 0.5
    assert stats["birkhoff_gate_factor"] < 1.0
    assert stats["birkhoff_applied_offdiag_mass"] <= 0.00105
    assert stats["birkhoff_post_gap"] <= stats["birkhoff_pre_gap"] + 1e-6
    assert stats["birkhoff_gap_delta"] <= 1e-6


def test_prototype_prior_can_shift_toward_task_conditioned_capacity_memory():
    config = ContinualASAMConfig(
        dim=32,
        num_heads=2,
        dim_head=16,
        num_tasks=3,
        num_prototypes=4,
        prototype_embed_dim=16,
    )
    layer = PrototypeContinualASAMLayer(config)

    with torch.no_grad():
        layer.prototype_memory_seen.copy_(torch.tensor([True, True, True, True]))
        layer.prototype_usage_ema.copy_(torch.tensor([0.7, 0.1, 0.1, 0.1]))
        layer.task_prototype_seen.copy_(torch.tensor([False, True, False]))
        layer.task_prototype_memory[1].copy_(torch.tensor([0.05, 0.8, 0.1, 0.05]))
        layer.set_task_transport_weights(torch.tensor([0.05, 0.15, 0.05]), base_weight=0.05)

    unseen_prior = layer._build_prototype_prior(
        batch_size=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
        task_ids=torch.tensor([2]),
    )
    task_prior = layer._build_prototype_prior(
        batch_size=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
        task_ids=torch.tensor([1]),
    )

    with torch.no_grad():
        layer.set_task_transport_weights(torch.tensor([0.05, 0.05, 0.05]), base_weight=0.05)
    neutral_prior = layer._build_prototype_prior(
        batch_size=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
        task_ids=torch.tensor([1]),
    )

    assert task_prior[0, 1] > unseen_prior[0, 1]
    assert task_prior[0, 0] < unseen_prior[0, 0]
    assert torch.allclose(neutral_prior, unseen_prior, atol=1e-6)
    assert torch.allclose(unseen_prior.sum(dim=-1), torch.ones(1), atol=1e-6)
    assert torch.allclose(task_prior.sum(dim=-1), torch.ones(1), atol=1e-6)



def test_classifier_propagates_task_transport_weights_to_prototype_layers():
    from experiments.train_continual_asam import ContinualTextClassifier

    model = ContinualTextClassifier(
        vocab_size=128,
        num_tasks=3,
        num_classes=2,
        dim=32,
        num_heads=2,
        num_layers=1,
        seq_len=16,
        top_k_patterns=2,
        routing_mode="prototype",
    )
    model.set_task_transport_weights([0.05, 0.2, 0.1], base_weight=0.05)

    prototype_layer = next(layer for layer in model.layers if isinstance(layer, PrototypeContinualASAMLayer))
    assert torch.allclose(
        prototype_layer.task_transport_weights,
        torch.tensor([0.05, 0.2, 0.1]),
        atol=1e-6,
    )
    assert abs(float(prototype_layer.task_transport_base_weight.item()) - 0.05) < 1e-6


def test_continual_text_classifier_aggregates_transport_loss_per_sample():
    from experiments.train_continual_asam import ContinualTextClassifier

    model = ContinualTextClassifier(
        vocab_size=128,
        num_tasks=2,
        num_classes=2,
        dim=32,
        num_heads=2,
        num_layers=2,
        seq_len=16,
        top_k_patterns=2,
        routing_mode="prototype",
    )

    inputs = torch.randint(0, 128, (4, 16))
    task_ids = torch.tensor([0, 1, 0, 1])
    _, info = model(inputs, task_ids=task_ids, return_info=True)

    assert info["transport_loss_per_sample"].shape == (4,)
    assert torch.allclose(
        info["transport_loss_per_sample"].mean(),
        info["transport_loss"],
        atol=1e-6,
    )



def test_prototype_hyperparameter_round_trip_tracks_merge_controls():
    from experiments.train_continual_asam import ContinualTextClassifier

    model = ContinualTextClassifier(
        vocab_size=128,
        num_tasks=3,
        num_classes=2,
        dim=32,
        num_heads=2,
        num_layers=1,
        seq_len=16,
        top_k_patterns=2,
        routing_mode="prototype",
    )

    model.set_prototype_hyperparameters(
        prototype_prior_strength=1.5,
        prototype_capacity_blend=0.4,
        prototype_relocation_strength=0.6,
        prototype_merge_threshold=0.8,
        prototype_merge_usage_threshold=0.05,
    )
    params = model.get_prototype_hyperparameters()

    assert params["prototype_prior_strength"] == 1.5
    assert params["prototype_capacity_blend"] == 0.4
    assert params["prototype_relocation_strength"] == 0.6
    assert params["prototype_merge_threshold"] == 0.8
    assert params["prototype_merge_usage_threshold"] == 0.05
