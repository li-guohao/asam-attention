"""HuggingFace Transformers-compatible model classes for ASAM."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, SequenceClassifierOutput

from .asam_layer import ASAMConfig, ASAMEncoder, ASAMLayer


class ASAMHFConfig(PretrainedConfig):
    """HuggingFace-compatible configuration for ASAM models.

    Mirrors ASAMConfig fields while following HF conventions.
    """

    model_type = "asam"

    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.1,
        pattern_type: str = "hierarchical",
        window_size: int = 128,
        stride: int = 32,
        num_clusters: int = 32,
        use_adaptive_gate: bool = True,
        gate_hidden_dim: int = 128,
        use_gradient_checkpointing: bool = False,
        num_layers: int = 6,
        num_labels: int = 2,
        vocab_size: int = 30000,
        max_position_embeddings: int = 8192,
        pad_token_id: int = 0,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            **kwargs,
        )
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.pattern_type = pattern_type
        self.window_size = window_size
        self.stride = stride
        self.num_clusters = num_clusters
        self.use_adaptive_gate = use_adaptive_gate
        self.gate_hidden_dim = gate_hidden_dim
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.num_layers = num_layers
        self.num_labels = num_labels
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings

    def to_asam_config(self) -> ASAMConfig:
        """Convert to internal ASAMConfig."""
        return ASAMConfig(
            dim=self.dim,
            num_heads=self.num_heads,
            dim_head=self.dim_head,
            dropout=self.dropout,
            pattern_type=self.pattern_type,
            window_size=self.window_size,
            stride=self.stride,
            num_clusters=self.num_clusters,
            use_adaptive_gate=self.use_adaptive_gate,
            gate_hidden_dim=self.gate_hidden_dim,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
        )


class ASAMHFModel(PreTrainedModel):
    """HuggingFace-compatible ASAM base model.

    Wraps ASAMEncoder with token embeddings for text processing.
    Outputs last_hidden_state via BaseModelOutputWithPast.
    """

    config_class = ASAMHFConfig
    base_model_prefix = "asam"
    _no_split_modules = ["ASAMLayer"]

    def __init__(self, config: ASAMHFConfig):
        super().__init__(config)
        asam_config = config.to_asam_config()

        self.token_embedding = nn.Embedding(
            config.vocab_size, config.dim, padding_idx=config.pad_token_id
        )
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.dim)
        self.embed_dropout = nn.Dropout(config.dropout)

        self.encoder = ASAMEncoder(asam_config, num_layers=config.num_layers)

        self.post_init()

    def _init_weights(self, module: nn.Module):
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> BaseModelOutputWithPast:
        batch, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.token_embedding(input_ids)
        x = x + self.position_embedding(positions)
        x = self.embed_dropout(x)

        # Build attention mask if provided
        mask_4d = None
        if attention_mask is not None:
            mask_4d = attention_mask[:, None, None, :].to(dtype=torch.bool)
            mask_4d = mask_4d.expand(-1, self.config.num_heads, seq_len, -1)

        hidden_states = self.encoder(x, mask=mask_4d)

        if not return_dict:
            return (hidden_states,)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


class ASAMHFForSequenceClassification(PreTrainedModel):
    """ASAM model with a sequence classification head.

    Built on ASAMHFModel with a mean-pooling + linear classifier.
    """

    config_class = ASAMHFConfig
    base_model_prefix = "asam"

    def __init__(self, config: ASAMHFConfig):
        super().__init__(config)
        self.asam = ASAMHFModel(config)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> SequenceClassifierOutput:
        outputs = self.asam(input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)

        if not return_dict:
            output = (logits,) + (outputs,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )


# Register with HuggingFace auto classes
ASAMHFConfig.register_for_auto_class()
ASAMHFModel.register_for_auto_class("AutoModel")
ASAMHFForSequenceClassification.register_for_auto_class("AutoModelForSequenceClassification")
