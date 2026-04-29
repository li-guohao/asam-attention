"""
Optimized ASAM Layer
====================

This is an optimized version of ASAM that achieves true sparse computation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math

from .adaptive_gate import AdaptiveGate
from .sparse_patterns import SparsePattern, LocalSparsePattern, HierarchicalSparsePattern
from ._common import (
    normalize_attention_mask,
    gather_values_by_positions,
)


class OptimizedASAMLayer(nn.Module):
    """
    Optimized ASAM layer with true sparse attention.
    
    Key optimizations:
    1. True O(n*window) local attention (not O(n^2) + mask)
    2. Gradient checkpointing for memory efficiency
    3. Mixed precision support
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 128,
        stride: int = 32,
        dropout: float = 0.1,
        use_adaptive_gate: bool = True,
        pattern_type: str = 'local',
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.stride = stride
        self.head_dim = dim // num_heads
        self.pattern_type = pattern_type
        self._local_window_mask_cache = {}
        self._local_window_index_cache = {}
        self._local_attention_mask_cache = {}
        self._strided_attention_index_cache = {}

        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        # Q, K, V projections
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Adaptive gate
        self.use_adaptive_gate = use_adaptive_gate
        if use_adaptive_gate:
            self.adaptive_gate = AdaptiveGate(dim, num_heads=num_heads)
        else:
            self.register_parameter('adaptive_gate', None)
        
        # Layer norm
        self.norm = nn.LayerNorm(dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.qkv.bias)
        nn.init.zeros_(self.out_proj.bias)
    
    def _compute_local_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        window_size: int,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        True O(n*window) local attention.
        
        Args:
            q, k, v: [batch, heads, seq, dim]
        Returns:
            output: [batch, heads, seq, dim]
        """
        _, _, seq_len, dim_head = q.shape

        if mask is not None:
            window_positions, valid_window_mask = self._get_local_window_indices(seq_len, window_size, q.device)
            return self._compute_indexed_sparse_attention(
                q,
                k,
                v,
                window_positions,
                valid_window_mask,
                mask=mask,
            )

        w = window_size // 2
        
        # Use unfold for efficient window extraction
        # Pad for boundary handling
        k_padded = F.pad(k, (0, 0, w, w), mode='constant', value=0)
        v_padded = F.pad(v, (0, 0, w, w), mode='constant', value=0)
        
        # Extract windows: [batch, heads, seq, window, dim]
        k_windows = k_padded.unfold(2, 2*w + 1, 1).permute(0, 1, 2, 4, 3)
        v_windows = v_padded.unfold(2, 2*w + 1, 1).permute(0, 1, 2, 4, 3)
        
        # Compute attention: q [batch, heads, seq, 1, dim] @ k [batch, heads, seq, dim, window]
        q_expanded = q.unsqueeze(-2)  # [batch, heads, seq, 1, dim]
        scores = torch.matmul(q_expanded, k_windows.transpose(-2, -1)) / math.sqrt(dim_head)
        scores = scores.squeeze(-2)  # [batch, heads, seq, window]

        valid_window_mask = self._get_local_window_mask(seq_len, window_size, q.device)
        scores = scores.masked_fill(~valid_window_mask, float('-inf'))
        
        # Softmax and apply
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)
        
        # [batch, heads, seq, 1, window] @ [batch, heads, seq, window, dim]
        out = torch.matmul(attn.unsqueeze(-2), v_windows).squeeze(-2)
        
        return out
    
    def _compute_strided_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        stride: int = 32,
        local_window: int = 16,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Exact strided attention over the union of local and global strided tokens.
        """
        seq_len = q.size(-2)
        combined_positions, combined_valid_mask = self._get_strided_attention_indices(
            seq_len,
            stride,
            local_window,
            q.device,
        )
        return self._compute_indexed_sparse_attention(
            q,
            k,
            v,
            combined_positions,
            combined_valid_mask,
            mask=mask,
        )

    def _compute_indexed_sparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        positions: torch.Tensor,
        valid_mask: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, heads, seq_len, dim_head = q.shape
        context_size = positions.size(-1)

        gathered_k = gather_values_by_positions(k, positions)
        gathered_v = gather_values_by_positions(v, positions)

        scores = torch.matmul(q.unsqueeze(-2), gathered_k.transpose(-2, -1)).squeeze(-2)
        scores = scores / math.sqrt(dim_head)

        combined_mask = valid_mask.view(1, 1, seq_len, context_size)
        if mask is not None:
            normalized_mask = normalize_attention_mask(mask, batch, heads, seq_len)
            gather_index = positions.view(1, 1, seq_len, context_size).expand(batch, heads, -1, -1)
            gathered_mask = normalized_mask.gather(-1, gather_index)
            combined_mask = combined_mask & gathered_mask

        scores = scores.masked_fill(~combined_mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        return torch.matmul(attn.unsqueeze(-2), gathered_v).squeeze(-2)


    def _get_strided_local_window(self) -> int:
        return max(1, min(self.window_size // 2, self.stride // 2))

    def _get_strided_attention_indices(
        self,
        seq_len: int,
        stride: int,
        local_window: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_key = (device.type, device.index, seq_len, stride, local_window)
        cached_value = self._strided_attention_index_cache.get(cache_key)
        if cached_value is not None:
            return cached_value

        local_positions, local_valid_mask = self._get_local_window_indices(seq_len, local_window * 2, device)
        strided_positions = torch.arange(0, seq_len, stride, device=device)
        query_positions = torch.arange(seq_len, device=device)
        strided_duplicates = (query_positions.unsqueeze(-1) - strided_positions.view(1, -1)).abs() <= local_window

        combined_positions = torch.cat(
            [local_positions, strided_positions.view(1, -1).expand(seq_len, -1)],
            dim=-1,
        )
        combined_valid_mask = torch.cat(
            [local_valid_mask, ~strided_duplicates],
            dim=-1,
        )

        cached_value = (combined_positions, combined_valid_mask)
        self._strided_attention_index_cache[cache_key] = cached_value
        return cached_value

    def _get_local_window_indices(
        self,
        seq_len: int,
        window_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_key = (device.type, device.index, seq_len, window_size)
        cached_value = self._local_window_index_cache.get(cache_key)
        if cached_value is not None:
            return cached_value

        w = window_size // 2
        positions = torch.arange(seq_len, device=device)
        offsets = torch.arange(-w, w + 1, device=device)
        window_positions = positions.unsqueeze(-1) + offsets.unsqueeze(0)
        valid_mask = (window_positions >= 0) & (window_positions < seq_len)
        clamped_positions = window_positions.clamp(0, seq_len - 1)

        cached_value = (clamped_positions, valid_mask)
        self._local_window_index_cache[cache_key] = cached_value
        return cached_value

    def _get_local_window_mask(
        self,
        seq_len: int,
        window_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        cache_key = (device.type, device.index, seq_len, window_size)
        cached_mask = self._local_window_mask_cache.get(cache_key)
        if cached_mask is not None:
            return cached_mask

        _, valid_mask = self._get_local_window_indices(seq_len, window_size, device)
        mask = valid_mask.unsqueeze(0).unsqueeze(0)

        self._local_window_mask_cache[cache_key] = mask
        return mask

    def _estimate_sparse_ratio(self, seq_len: int, device: torch.device) -> float:
        if self.pattern_type == 'strided':
            _, valid_mask = self._get_strided_attention_indices(
                seq_len,
                self.stride,
                self._get_strided_local_window(),
                device,
            )
        else:
            _, valid_mask = self._get_local_window_indices(seq_len, self.window_size, device)

        average_connections = valid_mask.sum(dim=-1).float().mean().item()
        return max(0.0, 1.0 - (average_connections / max(1, seq_len)))

    def _get_dense_local_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        cache_key = (device.type, device.index, seq_len, self.window_size)
        cached_mask = self._local_attention_mask_cache.get(cache_key)
        if cached_mask is not None:
            return cached_mask

        w = self.window_size // 2
        positions = torch.arange(seq_len, device=device)
        local_mask = (positions.view(-1, 1) - positions.view(1, -1)).abs() <= w

        self._local_attention_mask_cache[cache_key] = local_mask
        return local_mask

    def _fallback_attention_with_mask(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        if hasattr(F, 'scaled_dot_product_attention'):
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False,
            )

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        scores = scores.masked_fill(~attn_mask, torch.finfo(scores.dtype).min)
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)
        return torch.matmul(attn, v)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_info: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        """
        Forward pass with optimized sparse attention.
        
        Args:
            x: [batch, seq_len, dim]
            mask: optional mask
            return_info: return debug info
            
        Returns:
            output: [batch, seq_len, dim]
            info: dict with gate values, etc.
        """
        batch, seq_len, dim = x.shape
        residual = x
        
        # Pre-norm
        x = self.norm(x)
        
        # Project to Q, K, V
        qkv = self.qkv(x).reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        gate_value = None
        confidence = None
        pattern_logits = None
        if return_info:
            if self.use_adaptive_gate:
                gate_value, confidence, pattern_logits = self.adaptive_gate(x)
            else:
                gate_value = x.new_full((batch, self.num_heads, seq_len), 0.5)
                confidence = x.new_full((batch, self.num_heads), 0.5)
                pattern_logits = x.new_zeros((batch, 4))
        
        # Select attention type based on pattern_type and gate
        normalized_mask = None
        if mask is not None:
            normalized_mask = normalize_attention_mask(mask, batch, heads, seq_len)

        if self.pattern_type == 'local':
            attn_out = self._compute_local_attention(q, k, v, self.window_size, mask=normalized_mask)
        elif self.pattern_type == 'strided':
            attn_out = self._compute_strided_attention(
                q,
                k,
                v,
                stride=self.stride,
                local_window=self._get_strided_local_window(),
                mask=normalized_mask,
            )
        elif normalized_mask is not None:
            attn_out = self._fallback_attention_with_mask(q, k, v, normalized_mask)
        else:
            # Fallback to local for hierarchical
            attn_out = self._compute_local_attention(q, k, v, self.window_size)
        
        # Merge heads
        attn_out = attn_out.transpose(1, 2).reshape(batch, seq_len, dim)
        attn_out = self.out_proj(attn_out)
        attn_out = self.dropout(attn_out)
        
        # Residual connection
        out = residual + attn_out
        
        # FFN
        residual = out
        out = residual + self.ffn(out)
        
        if return_info:
            info = {
                'gate_values': gate_value,
                'confidence': confidence,
                'pattern_logits': pattern_logits,
                'sparse_ratio': self._estimate_sparse_ratio(seq_len, x.device),
            }
            return out, info
        
        return out, None
