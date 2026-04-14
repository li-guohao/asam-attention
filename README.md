# ASAM: Adaptive Sparse Attention Module

[中文说明](README.zh-CN.md)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/li-guohao/asam-attention)](https://github.com/li-guohao/asam-attention/releases)

ASAM is a research-oriented attention module that combines **adaptive sparse patterns** with **hardware-aware optimized implementations** for long-sequence modeling.

This repository includes:

- the original `ASAMLayer` implementation,
- efficient PyTorch 2.x attention variants built on `scaled_dot_product_attention`,
- a true sparse optimized path for local and strided attention,
- profiling and benchmarking scripts for sparse pattern construction and runtime behavior.

## Highlights

- Adaptive sparse attention with local, strided, random, clustered, and hierarchical patterns
- Optimized inference paths for consumer GPUs
- Flash / SDPA-based efficient attention variants for PyTorch 2.x
- Mixed precision training examples
- Sparse pattern profiling and optimization reports
- End-to-end tests covering core layers and sparse pattern behavior

## Latest Update

The latest release, [`v1.1.1`](https://github.com/li-guohao/asam-attention/releases/tag/v1.1.1), focuses on:

- sparse path optimizations,
- pattern construction acceleration,
- hierarchical pattern caching,
- clustered assignment optimization,
- profiling support via `benchmarks/profile_patterns.py`.

See:

- [Changelog](CHANGELOG.md)
- [Performance Optimization Report](docs/performance_optimization_report.md)

## Implementations

### 1. `ASAMLayer`

The main adaptive sparse attention layer with gating and pattern selection.

```python
from asam import ASAMLayer, ASAMConfig

config = ASAMConfig(
    dim=256,
    num_heads=4,
    pattern_type="local",
    use_adaptive_gate=True,
)

layer = ASAMLayer(config)
```

### 2. `EfficientASAMLayer` / `FlashASAMLayer`

PyTorch 2.x efficient attention variants built on `scaled_dot_product_attention`.

```python
from asam.efficient_attention import FlashASAMLayer

layer = FlashASAMLayer(dim=256, num_heads=4, window_size=128)
```

### 3. `OptimizedASAMLayer`

An optimized sparse implementation for local and strided attention.

```python
from asam.asam_layer_optimized import OptimizedASAMLayer

layer = OptimizedASAMLayer(dim=256, num_heads=4, window_size=128)
```

## Performance Snapshot

Representative results included in this repository:

| Component | Before | After | Gain |
|---|---:|---:|---:|
| `OptimizedASAMLayer` | 32.84 ms | 24.58 ms | `1.34x` |
| `EfficientASAMLayer` | 12.19 ms | 11.30 ms | `1.08x` |
| `LocalSparsePattern.build_pattern()` | 28.55 ms | 17.81 ms | `1.60x` |
| `StridedSparsePattern.build_pattern()` | 82.09 ms | 18.10 ms | `4.54x` |
| `RandomSparsePattern.build_pattern()` | 540.76 ms | 239.46 ms | `2.26x` |
| `HierarchicalSparsePattern.combine_patterns()` on CUDA | 43.06 ms | 2.94 ms | `14.65x` |

These numbers come from local measurements in the current development environment and should be treated as reference results, not universal benchmarks.

## Installation

### Clone the repository

```bash
git clone https://github.com/li-guohao/asam-attention.git
cd asam-attention
```

### Create an environment

```bash
python -m venv .venv
```

- Windows: `.venv\Scripts\activate`
- macOS / Linux: `source .venv/bin/activate`

### Install dependencies

Install PyTorch first, then install ASAM from source:

```bash
pip install torch torchvision
pip install -e .
```

For development:

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic usage

```python
import torch
from asam import ASAMLayer, ASAMConfig

config = ASAMConfig(
    dim=256,
    num_heads=4,
    pattern_type="local",
    use_adaptive_gate=True,
)

layer = ASAMLayer(config)
x = torch.randn(2, 512, 256)

output, info = layer(x, return_info=True)
print(output.shape)
print(info["sparse_ratio"])
```

### Efficient attention usage

```python
import torch
from asam.efficient_attention import FlashASAMLayer

layer = FlashASAMLayer(dim=256, num_heads=4, window_size=128)
x = torch.randn(2, 512, 256)

output, info = layer(x, return_info=True)
print(output.shape)
print(info["sparse_ratio"])
```

### Example scripts

```bash
python examples/basic_usage.py
python examples/optimized_usage.py
python examples/benchmark.py
```

## Benchmarks and Profiling

### Run benchmark scripts

```bash
python experiments/run_final_benchmark.py
python experiments/benchmark_optimized.py
```

### Profile sparse patterns

```bash
python benchmarks/profile_patterns.py --seq-len 2048 --devices auto
```

### Export profiling results

```bash
python benchmarks/profile_patterns.py --seq-len 2048 --devices auto --json-out benchmarks/pattern_profile.json
```

## Project Structure

```text
asam-attention/
├── asam/                         # Core library
│   ├── asam_layer.py             # Main ASAM implementation
│   ├── asam_layer_optimized.py   # Optimized sparse attention path
│   ├── efficient_attention.py    # SDPA / Flash-style attention layers
│   ├── adaptive_gate.py          # Adaptive gating module
│   ├── sparse_patterns.py        # Sparse pattern implementations
│   └── __init__.py
├── benchmarks/                   # Benchmark and profiling tools
├── docs/                         # Documentation
├── examples/                     # Usage examples
├── experiments/                  # Experiment scripts
├── tests/                        # Unit tests
├── CHANGELOG.md
└── README.md
```

## Documentation

- [Chinese README](README.zh-CN.md)
- [Changelog](CHANGELOG.md)
- [Performance Analysis Report](docs/analysis_report.md)
- [Performance Optimization Report](docs/performance_optimization_report.md)
- [API Documentation](docs/API.md)
- [Technical Deep Dive](docs/TECHNICAL.md)
- [Experiments Guide](docs/EXPERIMENTS_GUIDE.md)

## Testing

Run the full test suite:

```bash
python -m pytest tests -q
```

Run selected tests:

```bash
python tests/test_basic.py
python tests/test_efficient.py
python tests/test_asam.py
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Optional CUDA GPU for optimized and benchmark paths

## Notes

- The repository includes both baseline and optimized implementations.
- Some optimization claims depend on GPU architecture and sequence length.
- The profiling scripts are intended to help you evaluate the trade-offs on your own hardware.

## License

This project is released under the [MIT License](LICENSE).
