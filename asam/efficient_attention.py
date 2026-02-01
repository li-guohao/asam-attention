"""
Efficient Attention Implementations
====================================

Uses PyTorch 2.0+ scaled_dot_product_attention with memory-efficient attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class EfficientASAMLayer(nn.Module):
    """
    ASAM Layer using PyTorch 2.0 efficient attention kernels.
    
    Features:
    - Uses torch.nn.functional.scaled_dot_product_attention
    - Supports local attention via block-sparse mask
    - Memory efficient (flash attention on RTX 3060)
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 128,
        dropout: float = 0.1,
        use_local_attention: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.use_local_attention = use_local_attention
        self.scale = self.head_dim ** -0.5
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        # Q, K, V projections
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
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
    
    def _create_local_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create local attention mask for efficient attention."""
        w = self.window_size // 2
        
        # Create causal + local mask
        i = torch.arange(seq_len, device=device).view(-1, 1)
        j = torch.arange(seq_len, device=device).view(1, -1)
        
        # Local window mask
        local_mask = (i - j).abs() <= w
        
        return local_mask
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_info: bool = False,
    ):
        """
        Forward using PyTorch efficient attention.
        
        Args:
            x: [batch, seq_len, dim]
            mask: optional mask
            
        Returns:
            output: [batch, seq_len, dim]
            info: dict with debug info
        """
        batch, seq_len, dim = x.shape
        residual = x
        
        # Pre-norm
        x = self.norm(x)
        
        # Project to Q, K, V
        qkv = self.qkv(x).reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Use PyTorch 2.0 efficient attention
        # This automatically selects flash attention on RTX 3060
        if hasattr(F, 'scaled_dot_product_attention'):
            # PyTorch 2.0+ 
            if self.use_local_attention and seq_len > self.window_size:
                # Create local attention mask
                local_mask = self._create_local_mask(seq_len, x.device)
                if mask is not None:
                    attn_mask = mask & local_mask
                else:
                    attn_mask = local_mask
                
                # Efficient attention with mask
                attn_out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    is_causal=False,
                )
            else:
                # Standard efficient attention (may use flash attention)
                attn_out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=mask,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    is_causal=False,
                )
        else:
            # Fallback for older PyTorch
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if mask is not None:
                scores = scores.masked_fill(~mask, float('-inf'))
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            attn_out = torch.matmul(attn, v)
        
        # Merge heads
        attn_out = attn_out.transpose(1, 2).reshape(batch, seq_len, dim)
        attn_out = self.out_proj(attn_out)
        attn_out = self.dropout(attn_out)
        
        # Residual + FFN
        out = residual + attn_out
        out = out + self.ffn(out)
        
        if return_info:
            info = {
                'sparse_ratio': 1.0 - (self.window_size / seq_len) if self.use_local_attention else 1.0,
            }
            return out, info
        
        return out, None


class FlashASAMLayer(EfficientASAMLayer):
    """
    ASAM Layer that always uses Flash Attention (when available).
    Best for RTX 3060.
    """
    
    def __init__(self, *args, **kwargs):
        # Force use_local_attention=False to allow flash attention
        kwargs['use_local_attention'] = False
        super().__init__(*args, **kwargs)
    
    def forward(self, x, mask=None, return_info=False):
        # Always use efficient attention (flash when available)
        return super().forward(x, mask, return_info)
