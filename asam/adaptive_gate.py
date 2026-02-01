"""
Adaptive Gating Mechanism
=========================

An original contribution: dynamic selection between sparse and dense attention
based on input characteristics and learned confidence estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class AdaptiveGate(nn.Module):
    """
    Adaptive Gating Module that dynamically controls the balance between
    sparse and full attention based on input complexity.
    
    Key Innovation:
    - Uses input-dependent gating to determine when sparse attention is sufficient
    - Incorporates a confidence estimation mechanism
    - Allows differentiable soft switching between sparse and dense modes
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        hidden_dim: int = 128,
        num_pools: int = 4,
        temperature: float = 1.0
    ):
        """
        Args:
            dim: Model dimension
            num_heads: Number of attention heads
            hidden_dim: Hidden dimension for gating network
            num_pools: Number of pooling scales for multi-scale features
            temperature: Temperature for gating softmax
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.temperature = temperature
        
        # Multi-scale pooling for capturing different granularities
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool1d(max(1, dim // (2 ** i)))
            for i in range(num_pools)
        ])
        
        # Feature extraction network
        pool_dim = sum(max(1, dim // (2 ** i)) for i in range(num_pools))
        self.feature_proj = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        # Complexity estimator: estimates how "complex" the input is
        self.complexity_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_heads),
        )
        
        # Confidence predictor: predicts confidence in sparse attention
        self.confidence_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_heads),
            nn.Sigmoid(),
        )
        
        # Pattern selector: selects which sparse pattern to use
        self.pattern_selector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4),  # 4 pattern types
        )
        
        # Learnable threshold for sparse/dense switching
        self.sparse_threshold = nn.Parameter(torch.ones(num_heads) * 0.5)
        
        # Gating temperature (learnable)
        self.gate_temp = nn.Parameter(torch.ones(1) * 0.1)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract multi-scale features from input.
        
        Args:
            x: [batch, seq_len, dim]
            
        Returns:
            features: [batch, seq_len, hidden_dim]
        """
        batch, seq_len, dim = x.shape
        
        # Transpose for pooling: [batch, dim, seq_len]
        x_t = x.transpose(1, 2)
        
        # Multi-scale pooling
        pooled_features = []
        for pool in self.pools:
            # Pool and flatten
            pooled = pool(x_t)  # [batch, dim, pooled_dim]
            pooled_features.append(pooled)
        
        # Concatenate and project
        multi_scale = torch.cat(pooled_features, dim=-1)  # [batch, dim, pool_dim]
        multi_scale = multi_scale.transpose(1, 2)  # [batch, pool_dim, dim]
        
        # Average over sequence dimension with attention weights
        attn_weights = F.softmax(
            torch.bmm(multi_scale, x_t).mean(dim=-1, keepdim=True), 
            dim=1
        )  # [batch, pool_dim, 1]
        
        pooled_seq = (multi_scale * attn_weights).sum(dim=2)  # [batch, pool_dim]
        
        # Project to hidden dim
        features = self.feature_proj(pooled_seq)  # [batch, hidden_dim]
        
        # Expand to sequence length
        features = features.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, hidden_dim]
        
        return features
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute adaptive gating outputs.
        
        Args:
            x: Input tensor [batch, seq_len, dim]
            
        Returns:
            gate_values: Sparse attention gate [batch, num_heads, seq_len]
            confidence: Confidence in sparse attention [batch, num_heads]
            pattern_logits: Pattern selection logits [batch, 4]
        """
        batch, seq_len, dim = x.shape
        
        # Extract features
        features = self.extract_features(x)  # [batch, seq_len, hidden_dim]
        
        # Compute complexity scores
        complexity = self.complexity_estimator(features)  # [batch, seq_len, num_heads]
        complexity = complexity.mean(dim=1)  # [batch, num_heads]
        
        # Compute confidence
        confidence = self.confidence_predictor(features.mean(dim=1))  # [batch, num_heads]
        
        # Compute gate values (soft switching)
        # Lower complexity and higher confidence -> more sparse attention
        effective_threshold = self.sparse_threshold.view(1, -1).expand(batch, -1)
        gate_logits = (effective_threshold - complexity) * confidence / self.gate_temp.abs()
        gate_values = torch.sigmoid(gate_logits)  # [batch, num_heads]
        
        # Pattern selection
        pattern_logits = self.pattern_selector(features.mean(dim=1))  # [batch, 4]
        
        # Expand gate to sequence dimension
        gate_values = gate_values.unsqueeze(-1).expand(-1, -1, seq_len)  # [batch, num_heads, seq_len]
        
        return gate_values, confidence, pattern_logits
    
    def get_attention_mode(self, gate_values: torch.Tensor) -> torch.Tensor:
        """
        Determine attention mode based on gate values.
        
        Returns:
            Boolean tensor indicating sparse (True) or dense (False) mode
        """
        return gate_values.mean(dim=-1) > 0.5


class DynamicSparseDenseAttention(nn.Module):
    """
    Combined sparse-dense attention that adaptively switches between modes.
    Original contribution: differentiable sparse-dense hybrid attention.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        
        inner_dim = dim_head * num_heads
        
        # Projections
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )
        
        # Adaptive gate
        self.gate = AdaptiveGate(dim, num_heads)
        
    def forward(
        self, 
        x: torch.Tensor, 
        sparse_attn_fn=None,
        mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass with adaptive sparse/dense attention.
        
        Args:
            x: [batch, seq_len, dim]
            sparse_attn_fn: Function for sparse attention computation
            mask: Optional attention mask
            
        Returns:
            output: [batch, seq_len, dim]
            info: Dictionary with gating information
        """
        batch, seq_len, dim = x.shape
        
        # Compute gating
        gate_values, confidence, pattern_logits = self.gate(x)
        # gate_values: [batch, num_heads, seq_len]
        
        # Project to Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(batch, seq_len, self.num_heads, self.dim_head).transpose(1, 2), qkv)
        # [batch, heads, seq_len, dim_head]
        
        # Compute attention scores
        q = q * self.scale
        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # [batch, heads, seq_len, seq_len]
        
        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        
        # Sparse attention (if provided)
        if sparse_attn_fn is not None:
            sparse_attn = sparse_attn_fn(q, k, v)  # [batch, heads, seq_len, dim_head]
        else:
            sparse_attn = None
        
        # Dense attention
        dense_attn_weights = F.softmax(attn_scores, dim=-1)
        dense_attn = torch.matmul(dense_attn_weights, v)  # [batch, heads, seq_len, dim_head]
        
        # Adaptive combination
        if sparse_attn is not None:
            # Reshape gate for broadcasting
            g = gate_values.unsqueeze(-1)  # [batch, heads, seq_len, 1]
            out = g * sparse_attn + (1 - g) * dense_attn
        else:
            out = dense_attn
        
        # Reshape and project
        out = out.transpose(1, 2).reshape(batch, seq_len, -1)
        out = self.to_out(out)
        
        info = {
            'gate_values': gate_values,
            'confidence': confidence,
            'pattern_logits': pattern_logits,
            'sparse_ratio': (gate_values > 0.5).float().mean().item(),
        }
        
        return out, info
