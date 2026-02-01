"""
ASAM Attention Visualization and Analysis
==========================================

Visualize attention patterns, gating behavior, and sparsity.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from asam import ASAMLayer, ASAMConfig


class AttentionVisualizer:
    """Visualize ASAM attention patterns and behavior."""
    
    def __init__(self, model: ASAMLayer):
        self.model = model
        self.model.eval()
    
    def visualize_sparse_pattern(self, seq_len: int = 128, save_path: str = None):
        """
        Visualize the sparse attention pattern.
        """
        # Get pattern from model
        pattern_module = list(self.model._pattern_modules.values())[0]
        device = next(self.model.parameters()).device
        
        if hasattr(pattern_module, 'combine_patterns'):
            pattern = pattern_module.combine_patterns(device)
        else:
            pattern = pattern_module.get_pattern(device)
        
        pattern = pattern.cpu().numpy()
        
        # If pattern has head dimension, average over heads
        if pattern.ndim == 3:
            pattern = pattern.mean(axis=0)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(pattern[:seq_len, :seq_len], cmap='Blues', cbar=True, ax=ax, square=True)
        ax.set_title(f'Sparse Attention Pattern ({self.model.config.pattern_type})')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        
        # Calculate sparsity
        sparsity = 1.0 - pattern.mean()
        ax.text(0.02, 0.98, f'Sparsity: {sparsity:.2%}', 
                transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        return fig
    
    def visualize_gate_behavior(self, x: torch.Tensor, save_path: str = None):
        """
        Visualize adaptive gate behavior.
        """
        if not self.model.config.use_adaptive_gate:
            print("Adaptive gate not enabled")
            return None
        
        with torch.no_grad():
            _, info = self.model(x, return_info=True)
        
        if not info:
            print("No gating information available")
            return None
        
        gate_values = info['gate_values'].cpu().numpy()  # [batch, heads, seq_len]
        confidence = info['confidence'].cpu().numpy()  # [batch, heads]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Gate values heatmap
        ax = axes[0, 0]
        im = ax.imshow(gate_values[0], aspect='auto', cmap='RdYlBu_r', vmin=0, vmax=1)
        ax.set_title('Gate Values (Sparse Attention Weight)')
        ax.set_xlabel('Sequence Position')
        ax.set_ylabel('Attention Head')
        plt.colorbar(im, ax=ax)
        
        # Gate distribution
        ax = axes[0, 1]
        ax.hist(gate_values.flatten(), bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(gate_values.mean(), color='red', linestyle='--', label=f'Mean: {gate_values.mean():.3f}')
        ax.set_title('Gate Value Distribution')
        ax.set_xlabel('Gate Value')
        ax.set_ylabel('Count')
        ax.legend()
        
        # Confidence per head
        ax = axes[1, 0]
        heads = range(len(confidence[0]))
        ax.bar(heads, confidence[0], color='coral', edgecolor='black')
        ax.set_title('Confidence per Head')
        ax.set_xlabel('Attention Head')
        ax.set_ylabel('Confidence')
        ax.set_ylim(0, 1)
        
        # Gate statistics over sequence
        ax = axes[1, 1]
        mean_gates = gate_values[0].mean(axis=0)
        std_gates = gate_values[0].std(axis=0)
        positions = range(len(mean_gates))
        
        ax.plot(positions, mean_gates, label='Mean', color='blue')
        ax.fill_between(positions, mean_gates - std_gates, mean_gates + std_gates, 
                        alpha=0.3, label='±1 std')
        ax.set_title('Gate Statistics over Sequence')
        ax.set_xlabel('Position')
        ax.set_ylabel('Gate Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        return fig
    
    def visualize_attention_rollout(self, x: torch.Tensor, save_path: str = None):
        """
        Visualize attention rollout across layers.
        """
        # This would require modifying the model to return attention weights
        # For now, show a placeholder visualization
        
        seq_len = x.size(1)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create simulated attention weights for visualization
        attention = torch.softmax(torch.randn(seq_len, seq_len), dim=-1)
        attention = attention.numpy()
        
        sns.heatmap(attention, cmap='viridis', ax=ax, square=True)
        ax.set_title('Attention Rollout (Simulated)')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
        return fig
    
    def compare_patterns(self, seq_len: int = 128, save_path: str = None):
        """
        Compare different sparse patterns.
        """
        patterns = ['local', 'strided', 'random', 'clustered', 'hierarchical']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, pattern_type in enumerate(patterns):
            config = ASAMConfig(
                dim=256,
                num_heads=4,
                pattern_type=pattern_type
            )
            model = ASAMLayer(config)
            model.eval()
            
            # Get pattern
            if hasattr(model, '_pattern_modules') and model._pattern_modules:
                pattern_module = list(model._pattern_modules.values())[0]
                pattern = pattern_module.get_pattern(torch.device('cpu'))
                pattern = pattern.numpy()
                
                if pattern.ndim == 3:
                    pattern = pattern[0]  # Take first head
            else:
                pattern = np.ones((seq_len, seq_len))
            
            ax = axes[idx]
            sns.heatmap(pattern[:seq_len, :seq_len], cmap='Blues', ax=ax, 
                       cbar=True, square=True, cbar_kws={'shrink': 0.6})
            sparsity = 1.0 - pattern[:seq_len, :seq_len].mean()
            ax.set_title(f'{pattern_type.capitalize()}\nSparsity: {sparsity:.1%}')
            ax.set_xlabel('')
            ax.set_ylabel('')
        
        # Remove extra subplot
        fig.delaxes(axes[-1])
        
        plt.suptitle('Sparse Pattern Comparison', fontsize=16, y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        return fig
    
    def plot_complexity_scaling(self, max_seq_len: int = 8192, save_path: str = None):
        """
        Plot computational complexity scaling.
        """
        seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192]
        seq_lengths = [s for s in seq_lengths if s <= max_seq_len]
        
        # Theoretical complexity
        standard = [s**2 for s in seq_lengths]
        local = [s * 256 for s in seq_lengths]  # window=256
        sparse = [s * (s // 32 + 128) for s in seq_lengths]  # stride=32 + local
        linformer = [s * 256 for s in seq_lengths]  # k=256
        performer = [s * 256 for s in seq_lengths]  # m=256
        asam = [s * np.sqrt(s) * 50 for s in seq_lengths]  # estimated
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(seq_lengths, standard, 'o-', label='Standard (O(n²))', linewidth=2)
        ax.plot(seq_lengths, local, 's-', label='Local (O(n×w))', linewidth=2)
        ax.plot(seq_lengths, sparse, '^-', label='Sparse (O(n×s))', linewidth=2)
        ax.plot(seq_lengths, linformer, 'v-', label='Linformer (O(n×k))', linewidth=2)
        ax.plot(seq_lengths, performer, 'd-', label='Performer (O(n×m))', linewidth=2)
        ax.plot(seq_lengths, asam, 'h-', label='ASAM (O(n×√n))', linewidth=2, color='red')
        
        ax.set_xlabel('Sequence Length')
        ax.set_ylabel('Relative Computation')
        ax.set_title('Computational Complexity Comparison')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        return fig


def create_visualization_report(output_dir: str = 'visualizations'):
    """
    Create a comprehensive visualization report.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating ASAM Visualization Report...")
    print("="*60)
    
    # Create model
    config = ASAMConfig(
        dim=256,
        num_heads=4,
        pattern_type="hierarchical",
        use_adaptive_gate=True
    )
    model = ASAMLayer(config)
    visualizer = AttentionVisualizer(model)
    
    # Generate visualizations
    print("\n1. Visualizing sparse pattern...")
    visualizer.visualize_sparse_pattern(
        seq_len=128,
        save_path=os.path.join(output_dir, 'sparse_pattern.png')
    )
    
    print("\n2. Visualizing gate behavior...")
    x = torch.randn(1, 256, 256)
    visualizer.visualize_gate_behavior(
        x,
        save_path=os.path.join(output_dir, 'gate_behavior.png')
    )
    
    print("\n3. Comparing patterns...")
    visualizer.compare_patterns(
        seq_len=128,
        save_path=os.path.join(output_dir, 'pattern_comparison.png')
    )
    
    print("\n4. Plotting complexity scaling...")
    visualizer.plot_complexity_scaling(
        max_seq_len=8192,
        save_path=os.path.join(output_dir, 'complexity_scaling.png')
    )
    
    print(f"\n✓ All visualizations saved to {output_dir}/")


if __name__ == "__main__":
    create_visualization_report()
