"""
ASAM Attention Visualization and Analysis
==========================================

Visualize attention patterns, gating behavior, and sparsity.
"""

import os
import sys
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from asam import ASAMConfig, ASAMLayer
from asam.sparse_patterns import ClusteredSparsePattern, HierarchicalSparsePattern


class AttentionVisualizer:
    """Visualize ASAM attention patterns and behavior."""

    def __init__(self, model: ASAMLayer):
        self.model = model
        self.model.eval()

    def _probe_input(self, seq_len: int, device: torch.device) -> torch.Tensor:
        values = torch.linspace(
            -1.0, 1.0, steps=seq_len * self.model.config.dim, device=device
        )
        return values.view(1, seq_len, self.model.config.dim)

    def _reshape_qkv(
        self,
        qkv: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch: int,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return tuple(
            tensor.reshape(
                batch, seq_len, self.model.num_heads, self.model.dim_head
            ).transpose(1, 2)
            for tensor in qkv
        )

    def _expand_mask(
        self,
        mask: Optional[torch.Tensor],
        batch: int,
        heads: int,
        seq_len: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if mask is None:
            return None

        mask = mask.to(device=device, dtype=torch.bool)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            if mask.shape[0] != batch:
                raise ValueError("3D masks must have shape [batch, seq_len, seq_len].")
            mask = mask.unsqueeze(1)
        elif mask.dim() != 4:
            raise ValueError("Attention mask must be 2D, 3D, or 4D.")

        if mask.shape[-2:] != (seq_len, seq_len):
            raise ValueError("Attention mask sequence dimensions must match the input.")

        return mask.expand(batch, heads, seq_len, seq_len)

    def _masked_softmax(
        self, scores: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)
        return torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

    def _get_pattern_mask(
        self,
        pattern_module,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> torch.Tensor:
        batch, heads, seq_len, _ = q.shape
        device = q.device

        if isinstance(pattern_module, ClusteredSparsePattern):
            q_assign, k_assign = pattern_module.compute_cluster_assignment(q, k)
            cluster_affinity = torch.einsum(
                "b h q c, b h k c -> b h q k", q_assign, k_assign
            )
            return cluster_affinity > 0.1

        if isinstance(pattern_module, HierarchicalSparsePattern):
            pattern_mask = pattern_module.combine_patterns(device)
            if pattern_mask.dim() == 3:
                return pattern_mask.unsqueeze(0).expand(batch, -1, -1, -1)
            return pattern_mask.unsqueeze(0).unsqueeze(0).expand(batch, heads, -1, -1)

        pattern_mask = pattern_module.get_pattern(device)
        if pattern_mask.dim() == 2:
            return pattern_mask.unsqueeze(0).unsqueeze(0).expand(batch, heads, -1, -1)
        if pattern_mask.dim() == 3:
            return pattern_mask.unsqueeze(0).expand(batch, -1, -1, -1)
        raise ValueError("Unsupported sparse pattern dimensions.")

    def _get_pattern_matrix(self, seq_len: int) -> np.ndarray:
        device = next(self.model.parameters()).device
        pattern_module = self.model._get_pattern(seq_len, device)

        if isinstance(pattern_module, ClusteredSparsePattern):
            with torch.no_grad():
                probe = self._probe_input(seq_len, device)
                x_norm = self.model.norm(probe)
                qkv_source = (
                    self.model.adaptive_attn.to_qkv
                    if self.model.adaptive_attn is not None
                    else self.model.to_qkv
                )
                qkv = qkv_source(x_norm).chunk(3, dim=-1)
                q, k, _ = self._reshape_qkv(qkv, batch=1, seq_len=seq_len)
                scale = (
                    self.model.adaptive_attn.scale
                    if self.model.adaptive_attn is not None
                    else self.model.scale
                )
                q = q * scale
                pattern = (
                    self._get_pattern_mask(pattern_module, q, k)[0].float().mean(dim=0)
                )
                return pattern.cpu().numpy()

        if isinstance(pattern_module, HierarchicalSparsePattern):
            pattern = pattern_module.combine_patterns(device).float()
        else:
            pattern = pattern_module.get_pattern(device).float()

        if pattern.dim() == 3:
            pattern = pattern.mean(dim=0)
        return pattern.cpu().numpy()

    def _extract_attention_map(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[np.ndarray, Dict[str, torch.Tensor]]:
        device = next(self.model.parameters()).device
        x = x.to(device)

        with torch.no_grad():
            batch, seq_len, _ = x.shape
            x_norm = self.model.norm(x)
            attn_mask = self._expand_mask(
                mask, batch, self.model.num_heads, seq_len, device
            )

            if self.model.adaptive_attn is not None:
                gate_values, confidence, pattern_logits = self.model.adaptive_attn.gate(
                    x_norm
                )
                qkv = self.model.adaptive_attn.to_qkv(x_norm).chunk(3, dim=-1)
                q, k, _ = self._reshape_qkv(qkv, batch=batch, seq_len=seq_len)
                q = q * self.model.adaptive_attn.scale

                scores = torch.matmul(q, k.transpose(-2, -1))
                dense_weights = self._masked_softmax(scores, attn_mask)

                pattern_module = self.model._get_pattern(seq_len, device)
                sparse_mask = self._get_pattern_mask(pattern_module, q, k)
                if attn_mask is not None:
                    sparse_mask = sparse_mask & attn_mask
                sparse_weights = self._masked_softmax(scores, sparse_mask)

                gate = gate_values.unsqueeze(-1)
                blended_weights = gate * sparse_weights + (1.0 - gate) * dense_weights
                attention_map = blended_weights[0].mean(dim=0).cpu().numpy()
                info = {
                    "gate_values": gate_values.cpu(),
                    "confidence": confidence.cpu(),
                    "pattern_logits": pattern_logits.cpu(),
                    "sparse_ratio": (gate_values > 0.5).float().mean().item(),
                }
                return attention_map, info

            qkv = self.model.to_qkv(x_norm).chunk(3, dim=-1)
            q, k, _ = self._reshape_qkv(qkv, batch=batch, seq_len=seq_len)
            q = q * self.model.scale
            scores = torch.matmul(q, k.transpose(-2, -1))

            pattern_module = self.model._get_pattern(seq_len, device)
            sparse_mask = self._get_pattern_mask(pattern_module, q, k)
            if attn_mask is not None:
                sparse_mask = sparse_mask & attn_mask
            sparse_weights = self._masked_softmax(scores, sparse_mask)
            attention_map = sparse_weights[0].mean(dim=0).cpu().numpy()
            return attention_map, {}

    def visualize_sparse_pattern(self, seq_len: int = 128, save_path: str = None):
        """Visualize the sparse attention pattern."""
        pattern = self._get_pattern_matrix(seq_len)

        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(
            pattern[:seq_len, :seq_len], cmap="Blues", cbar=True, ax=ax, square=True
        )
        ax.set_title(f"Sparse Attention Pattern ({self.model.config.pattern_type})")
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")

        sparsity = 1.0 - pattern[:seq_len, :seq_len].mean()
        ax.text(
            0.02,
            0.98,
            f"Sparsity: {sparsity:.2%}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat"),
        )

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved to {save_path}")
        else:
            plt.show()

        plt.close()
        return fig

    def visualize_gate_behavior(self, x: torch.Tensor, save_path: str = None):
        """Visualize adaptive gate behavior."""
        if not self.model.config.use_adaptive_gate:
            print("Adaptive gate not enabled")
            return None

        with torch.no_grad():
            _, info = self.model(
                x.to(next(self.model.parameters()).device), return_info=True
            )

        if not info:
            print("No gating information available")
            return None

        gate_values = info["gate_values"].cpu().numpy()
        confidence = info["confidence"].cpu().numpy()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax = axes[0, 0]
        im = ax.imshow(gate_values[0], aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1)
        ax.set_title("Gate Values (Sparse Attention Weight)")
        ax.set_xlabel("Sequence Position")
        ax.set_ylabel("Attention Head")
        plt.colorbar(im, ax=ax)

        ax = axes[0, 1]
        ax.hist(
            gate_values.flatten(),
            bins=50,
            alpha=0.7,
            color="steelblue",
            edgecolor="black",
        )
        ax.axvline(
            gate_values.mean(),
            color="red",
            linestyle="--",
            label=f"Mean: {gate_values.mean():.3f}",
        )
        ax.set_title("Gate Value Distribution")
        ax.set_xlabel("Gate Value")
        ax.set_ylabel("Count")
        ax.legend()

        ax = axes[1, 0]
        heads = range(len(confidence[0]))
        ax.bar(heads, confidence[0], color="coral", edgecolor="black")
        ax.set_title("Confidence per Head")
        ax.set_xlabel("Attention Head")
        ax.set_ylabel("Confidence")
        ax.set_ylim(0, 1)

        ax = axes[1, 1]
        mean_gates = gate_values[0].mean(axis=0)
        std_gates = gate_values[0].std(axis=0)
        positions = range(len(mean_gates))
        ax.plot(positions, mean_gates, label="Mean", color="blue")
        ax.fill_between(
            positions,
            mean_gates - std_gates,
            mean_gates + std_gates,
            alpha=0.3,
            label="+/- 1 std",
        )
        ax.set_title("Gate Statistics over Sequence")
        ax.set_xlabel("Position")
        ax.set_ylabel("Gate Value")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved to {save_path}")
        else:
            plt.show()

        plt.close()
        return fig

    def visualize_attention_rollout(
        self,
        x: torch.Tensor,
        save_path: str = None,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Visualize the real single-layer attention map.

        The public method name is kept for backward compatibility, but the plot is
        a head-averaged attention map for the current ASAM layer rather than a
        simulated rollout.
        """
        attention_map, info = self._extract_attention_map(x, mask=mask)

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(attention_map, cmap="viridis", ax=ax, square=True)
        title = (
            "Adaptive Attention Map (Single Layer)"
            if self.model.adaptive_attn is not None
            else "Sparse Attention Map (Single Layer)"
        )
        ax.set_title(title)
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")

        if "sparse_ratio" in info:
            ax.text(
                0.02,
                0.98,
                f"Sparse routing ratio: {info['sparse_ratio']:.2%}",
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
            )

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved to {save_path}")
        else:
            plt.show()

        plt.close()
        return fig

    def compare_patterns(self, seq_len: int = 128, save_path: str = None):
        """Compare different sparse patterns."""
        patterns = ["local", "strided", "random", "clustered", "hierarchical"]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, pattern_type in enumerate(patterns):
            config = ASAMConfig(dim=256, num_heads=4, pattern_type=pattern_type)
            model = ASAMLayer(config)
            model.eval()
            pattern_visualizer = AttentionVisualizer(model)
            pattern = pattern_visualizer._get_pattern_matrix(seq_len)

            ax = axes[idx]
            sns.heatmap(
                pattern[:seq_len, :seq_len],
                cmap="Blues",
                ax=ax,
                cbar=True,
                square=True,
                cbar_kws={"shrink": 0.6},
            )
            sparsity = 1.0 - pattern[:seq_len, :seq_len].mean()
            ax.set_title(f"{pattern_type.capitalize()}\nSparsity: {sparsity:.1%}")
            ax.set_xlabel("")
            ax.set_ylabel("")

        fig.delaxes(axes[-1])
        plt.suptitle("Sparse Pattern Comparison", fontsize=16, y=0.98)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved to {save_path}")
        else:
            plt.show()

        plt.close()
        return fig

    def plot_complexity_scaling(self, max_seq_len: int = 8192, save_path: str = None):
        """Plot computational complexity scaling."""
        seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192]
        seq_lengths = [seq_len for seq_len in seq_lengths if seq_len <= max_seq_len]

        standard = [seq_len**2 for seq_len in seq_lengths]
        local = [seq_len * 256 for seq_len in seq_lengths]
        sparse = [seq_len * (seq_len // 32 + 128) for seq_len in seq_lengths]
        linformer = [seq_len * 256 for seq_len in seq_lengths]
        performer = [seq_len * 256 for seq_len in seq_lengths]
        asam = [seq_len * np.sqrt(seq_len) * 50 for seq_len in seq_lengths]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(seq_lengths, standard, "o-", label="Standard (O(n^2))", linewidth=2)
        ax.plot(seq_lengths, local, "s-", label="Local (O(n*w))", linewidth=2)
        ax.plot(seq_lengths, sparse, "^-", label="Sparse (O(n*s))", linewidth=2)
        ax.plot(seq_lengths, linformer, "v-", label="Linformer (O(n*k))", linewidth=2)
        ax.plot(seq_lengths, performer, "d-", label="Performer (O(n*m))", linewidth=2)
        ax.plot(
            seq_lengths,
            asam,
            "h-",
            label="ASAM (approx. O(n*sqrt(n)))",
            linewidth=2,
            color="red",
        )

        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Relative Computation")
        ax.set_title("Computational Complexity Comparison")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved to {save_path}")
        else:
            plt.show()

        plt.close()
        return fig


def create_visualization_report(output_dir: str = "visualizations"):
    """Create a comprehensive visualization report."""
    os.makedirs(output_dir, exist_ok=True)

    print("Creating ASAM Visualization Report...")
    print("=" * 60)

    config = ASAMConfig(
        dim=256,
        num_heads=4,
        pattern_type="hierarchical",
        use_adaptive_gate=True,
    )
    model = ASAMLayer(config)
    visualizer = AttentionVisualizer(model)
    x = torch.randn(1, 256, 256)

    print("\n1. Visualizing sparse pattern...")
    visualizer.visualize_sparse_pattern(
        seq_len=128,
        save_path=os.path.join(output_dir, "sparse_pattern.png"),
    )

    print("\n2. Visualizing gate behavior...")
    visualizer.visualize_gate_behavior(
        x,
        save_path=os.path.join(output_dir, "gate_behavior.png"),
    )

    print("\n3. Visualizing attention map...")
    visualizer.visualize_attention_rollout(
        x,
        save_path=os.path.join(output_dir, "attention_rollout.png"),
    )

    print("\n4. Comparing patterns...")
    visualizer.compare_patterns(
        seq_len=128,
        save_path=os.path.join(output_dir, "pattern_comparison.png"),
    )

    print("\n5. Plotting complexity scaling...")
    visualizer.plot_complexity_scaling(
        max_seq_len=8192,
        save_path=os.path.join(output_dir, "complexity_scaling.png"),
    )

    print(f"\nDone. All visualizations saved to {output_dir}/")


if __name__ == "__main__":
    create_visualization_report()
