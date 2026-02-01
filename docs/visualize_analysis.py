#!/usr/bin/env python3
"""
ASAM Performance Analysis Visualization
=======================================
"""

import matplotlib.pyplot as plt
import numpy as np
import json

# Load data
data = {
    "forward": {
        "128": {"original": 7.73, "flash": 2.60, "speedup": 2.97},
        "256": {"original": 11.63, "flash": 2.62, "speedup": 4.44},
        "512": {"original": 11.60, "flash": 2.59, "speedup": 4.47},
        "1024": {"original": 16.23, "flash": 6.01, "speedup": 2.70},
        "2048": {"original": 43.73, "flash": 15.04, "speedup": 2.91},
    },
    "training": {
        "128": {"fp32": 11.35, "fp16": 11.18, "speedup": 1.02},
        "256": {"fp32": 9.88, "fp16": 11.54, "speedup": 0.86},
        "512": {"fp32": 16.45, "fp16": 14.86, "speedup": 1.11},
        "1024": {"fp32": 41.66, "fp16": 20.66, "speedup": 2.02},
    }
}

# Extract data
seq_lens = [128, 256, 512, 1024, 2048]
train_lens = [128, 256, 512, 1024]

orig_times = [data["forward"][str(s)]["original"] for s in seq_lens]
flash_times = [data["forward"][str(s)]["flash"] for s in seq_lens]
speedups = [data["forward"][str(s)]["speedup"] for s in seq_lens]

fp32_times = [data["training"][str(s)]["fp32"] for s in train_lens]
fp16_times = [data["training"][str(s)]["fp16"] for s in train_lens]
train_speedups = [data["training"][str(s)]["speedup"] for s in train_lens]

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('ASAM Performance Analysis on RTX 3060', fontsize=16, fontweight='bold')

# Plot 1: Forward pass time comparison
ax = axes[0, 0]
x = np.arange(len(seq_lens))
width = 0.35
bars1 = ax.bar(x - width/2, orig_times, width, label='Original', color='#e74c3c', alpha=0.8)
bars2 = ax.bar(x + width/2, flash_times, width, label='Flash Attention', color='#2ecc71', alpha=0.8)
ax.set_xlabel('Sequence Length', fontsize=12)
ax.set_ylabel('Time (ms)', fontsize=12)
ax.set_title('Forward Pass: Original vs Flash Attention', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(seq_lens)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

# Plot 2: Speedup ratio
ax = axes[0, 1]
colors = ['#3498db' if s < 4.0 else '#e74c3c' for s in speedups]
bars = ax.bar(seq_lens, speedups, color=colors, alpha=0.8)
ax.axhline(y=4.0, color='red', linestyle='--', linewidth=2, label='4x Target')
ax.axhline(y=np.mean(speedups), color='orange', linestyle='--', linewidth=2, label=f'Mean: {np.mean(speedups):.2f}x')
ax.set_xlabel('Sequence Length', fontsize=12)
ax.set_ylabel('Speedup Ratio', fontsize=12)
ax.set_title('Flash Attention Speedup vs Sequence Length', fontsize=13)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar, speedup in zip(bars, speedups):
    height = bar.get_height()
    ax.annotate(f'{speedup:.2f}x',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 3: Training time comparison
ax = axes[1, 0]
x = np.arange(len(train_lens))
bars1 = ax.bar(x - width/2, fp32_times, width, label='FP32', color='#9b59b6', alpha=0.8)
bars2 = ax.bar(x + width/2, fp16_times, width, label='FP16 (Mixed)', color='#f39c12', alpha=0.8)
ax.set_xlabel('Sequence Length', fontsize=12)
ax.set_ylabel('Time (ms)', fontsize=12)
ax.set_title('Training: FP32 vs Mixed Precision (FP16)', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(train_lens)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 4: Combined optimization potential
ax = axes[1, 1]
# Calculate combined speedup: Flash × FP16
combined_speedups = []
for s in train_lens:
    flash_spd = data["forward"][str(s)]["speedup"]
    fp16_spd = data["training"][str(s)]["speedup"]
    combined_speedups.append(flash_spd * fp16_spd)

bars = ax.bar(train_lens, combined_speedups, color='#1abc9c', alpha=0.8)
ax.axhline(y=4.0, color='red', linestyle='--', linewidth=2, label='4x Target')
ax.set_xlabel('Sequence Length', fontsize=12)
ax.set_ylabel('Combined Speedup', fontsize=12)
ax.set_title('Combined Optimization:\nFlash Attention × Mixed Precision', fontsize=13)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar, spd in zip(bars, combined_speedups):
    height = bar.get_height()
    ax.annotate(f'{spd:.2f}x',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('experiments/performance_analysis.png', dpi=150, bbox_inches='tight')
print("Chart saved to: experiments/performance_analysis.png")

# Print summary
print("\n" + "="*70)
print("ANALYSIS SUMMARY")
print("="*70)
print("\n[Forward Pass - Flash Attention]")
print(f"  Best speedup: {max(speedups):.2f}x at {seq_lens[np.argmax(speedups)]} tokens")
print(f"  Worst speedup: {min(speedups):.2f}x at {seq_lens[np.argmin(speedups)]} tokens")
print(f"  Average speedup: {np.mean(speedups):.2f}x")

print("\n[Training - Mixed Precision]")
print(f"  Best speedup: {max(train_speedups):.2f}x at {train_lens[np.argmax(train_speedups)]} tokens")
print(f"  Average speedup: {np.mean(train_speedups):.2f}x")

print("\n[Combined Optimization]")
print(f"  1024 tokens: {combined_speedups[-1]:.2f}x speedup potential")
print(f"  Expected throughput increase: {(combined_speedups[-1]-1)*100:.0f}%")

print("\n" + "="*70)
