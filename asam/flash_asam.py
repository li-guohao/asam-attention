"""
Flash Attention Integration for ASAM
=====================================

Combines ASAM's adaptive sparsity with Flash Attention's memory efficiency.
This provides the best of both worlds:
- ASAM's adaptive pattern selection
- Flash Attention's IO-aware exact attention computation

Requirements: pip install flash-attn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from .asam_layer import ASAMConfig


class FlashASAMLayer(nn.Module):
    """
    ASAM layer with Flash Attention backend for the dense path.
    
    The sparse path uses custom efficient kernels, while the dense path
    uses Flash Attention for maximum memory efficiency.
    """
    
    def __init__(self, config: ASAMConfig):
        super().__init__()
        self.config = config
        
        # Try to import flash attention
        try:
            from flash_attn import flash_attn_func
            self.flash_attn_func = flash_attn_func
            self.use_flash = True
        except ImportError:
            print("Warning: flash-attn not installed. Falling back to standard attention.")
            print("Install with: pip install flash-attn")
            self.use_flash = False
        
        # Rest of initialization same as ASAMLayer
        # ... (simplified for brevity)
    
    def flash_attention_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = False
    ) -> torch.Tensor:
        """
        Flash Attention forward pass.
        
        Args:
            q, k, v: [batch, seq_len, num_heads, head_dim]
            causal: Whether to use causal masking
            
        Returns:
            output: [batch, seq_len, num_heads, head_dim]
        """
        if self.use_flash:
            # Flash attention expects [batch, seq_len, num_heads, head_dim]
            return self.flash_attn_func(
                q, k, v,
                causal=causal,
                softmax_scale=None  # Use default 1/sqrt(d)
            )
        else:
            # Fallback to standard attention
            scores = torch.einsum('bshd,bthd->bsht', q, k) / math.sqrt(q.size(-1))
            if causal:
                mask = torch.triu(torch.ones(q.size(1), k.size(1)), diagonal=1).bool()
                scores = scores.masked_fill(mask.to(scores.device), float('-inf'))
            attn = F.softmax(scores, dim=-1)
            return torch.einsum('bsht,bthd->bshd', attn, v)


class SparseFlashAttention(nn.Module):
    """
    Sparse attention pattern computed with memory-efficient kernels.
    """
    
    def __init__(self, config: ASAMConfig):
        super().__init__()
        self.config = config
        
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pattern_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute sparse attention with pattern mask.
        
        Uses scatter-gather for memory efficiency instead of full O(n²) storage.
        """
        batch, seq_len, num_heads, head_dim = q.shape
        
        # Get non-zero indices from pattern
        # pattern_mask: [seq_len, seq_len] boolean
        row_indices, col_indices = pattern_mask.nonzero(as_tuple=True)
        
        # Gather relevant Q, K pairs
        q_sparse = q[:, row_indices, :, :]  # [batch, num_sparse, heads, dim]
        k_sparse = k[:, col_indices, :, :]  # [batch, num_sparse, heads, dim]
        
        # Compute attention scores for sparse pairs only
        scores = torch.einsum('bshd,bshd->bsh', q_sparse, k_sparse) / math.sqrt(head_dim)
        
        # Softmax per query
        # Need to group by query position
        output = torch.zeros_like(q)
        for i in range(seq_len):
            mask_i = (row_indices == i)
            if mask_i.any():
                scores_i = scores[:, mask_i, :]
                v_i = v[:, col_indices[mask_i], :, :]
                
                attn_i = F.softmax(scores_i, dim=1)
                output[:, i, :, :] = torch.einsum('bsh,bshd->bhd', attn_i, v_i)
        
        return output


class HybridASAM(nn.Module):
    """
    Hybrid attention that combines Flash Attention for local windows
    with ASAM's adaptive global attention.
    """
    
    def __init__(
        self,
        config: ASAMConfig,
        local_window_size: int = 512,
        use_flash_local: bool = True
    ):
        super().__init__()
        self.config = config
        self.local_window_size = local_window_size
        self.use_flash_local = use_flash_local
        
        inner_dim = config.dim_head * config.num_heads
        self.to_qkv = nn.Linear(config.dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, config.dim)
        
        # Try import flash attention
        try:
            from flash_attn import flash_attn_func
            self.flash_attn_func = flash_attn_func
            self.has_flash = True
        except ImportError:
            self.has_flash = False
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Forward pass with local Flash Attention + sparse global attention.
        """
        batch, seq_len, dim = x.shape
        
        # Project to Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: t.reshape(batch, seq_len, self.config.num_heads, self.config.dim_head),
            qkv
        )
        
        # 1. Local attention with Flash
        local_out = self._local_attention(q, k, v)
        
        # 2. Sparse global attention (ASAM style)
        global_out = self._sparse_global_attention(q, k, v)
        
        # 3. Combine
        # Learnable combination weights
        combine = torch.sigmoid(self.combine_gate(x))  # [batch, seq_len, 1]
        out = combine * local_out + (1 - combine) * global_out
        
        # Reshape and project
        out = out.reshape(batch, seq_len, -1)
        return self.to_out(out)
    
    def _local_attention(self, q, k, v):
        """Local window attention using Flash Attention."""
        batch, seq_len, heads, dim = q.shape
        window = self.local_window_size
        
        outputs = []
        for i in range(0, seq_len, window):
            end_i = min(i + window, seq_len)
            
            # Get local window
            q_win = q[:, i:end_i, :, :]
            k_win = k[:, max(0, i-window//2):min(seq_len, i+window+window//2), :, :]
            v_win = v[:, max(0, i-window//2):min(seq_len, i+window+window//2), :, :]
            
            if self.has_flash and self.use_flash_local:
                # Use Flash Attention
                out_win = self.flash_attn_func(q_win, k_win, v_win, causal=False)
            else:
                # Standard attention
                scores = torch.einsum('bshd,bthd->bsht', q_win, k_win) / math.sqrt(dim)
                attn = F.softmax(scores, dim=-1)
                out_win = torch.einsum('bsht,bthd->bshd', attn, v_win)
            
            outputs.append(out_win)
        
        return torch.cat(outputs, dim=1)
    
    def _sparse_global_attention(self, q, k, v):
        """Sparse global attention (strided pattern)."""
        # Simplified strided pattern
        batch, seq_len, heads, dim = q.shape
        stride = 32
        
        # Only attend to strided positions
        k_strided = k[:, ::stride, :, :]
        v_strided = v[:, ::stride, :, :]
        
        scores = torch.einsum('bshd,bthd->bsht', q, k_strided) / math.sqrt(dim)
        attn = F.softmax(scores, dim=-1)
        return torch.einsum('bsht,bthd->bshd', attn, v_strided)


def benchmark_flash_vs_standard(seq_lengths=[512, 1024, 2048, 4096], device='cuda'):
    """
    Benchmark Flash Attention vs standard attention.
    """
    import time
    
    results = []
    
    for seq_len in seq_lengths:
        batch, heads, dim = 2, 8, 64
        
        q = torch.randn(batch, seq_len, heads, dim, device=device)
        k = torch.randn(batch, seq_len, heads, dim, device=device)
        v = torch.randn(batch, seq_len, heads, dim, device=device)
        
        # Standard attention
        torch.cuda.synchronize()
        start = time.time()
        scores = torch.einsum('bshd,bthd->bsht', q, k) / math.sqrt(dim)
        attn = F.softmax(scores, dim=-1)
        out_std = torch.einsum('bsht,bthd->bshd', attn, v)
        torch.cuda.synchronize()
        time_std = (time.time() - start) * 1000
        
        # Flash attention (if available)
        try:
            from flash_attn import flash_attn_func
            torch.cuda.synchronize()
            start = time.time()
            out_flash = flash_attn_func(q, k, v)
            torch.cuda.synchronize()
            time_flash = (time.time() - start) * 1000
            
            speedup = time_std / time_flash
            print(f"Seq Len {seq_len}: Standard={time_std:.2f}ms, Flash={time_flash:.2f}ms, Speedup={speedup:.2f}x")
            
            results.append({
                'seq_len': seq_len,
                'standard': time_std,
                'flash': time_flash,
                'speedup': speedup
            })
        except ImportError:
            print(f"Seq Len {seq_len}: Standard={time_std:.2f}ms (Flash not available)")
    
    return results


if __name__ == "__main__":
    # Test if flash attention is available
    try:
        import flash_attn
        print("✓ Flash Attention is available")
        print(f"  Version: {flash_attn.__version__}")
        
        # Run benchmark
        if torch.cuda.is_available():
            print("\nRunning benchmark...")
            results = benchmark_flash_vs_standard()
    except ImportError:
        print("✗ Flash Attention not installed")
        print("  Install with: pip install flash-attn")
