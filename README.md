# ASAM: Adaptive Sparse Attention Module ⚡

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**ASAM** is an efficient attention mechanism that combines adaptive sparsity patterns with hardware-optimized implementations, achieving **5.45x speedup** on consumer GPUs.

## 🚀 Key Features

- **⚡ Flash Attention Integration**: Up to 4.5x faster forward pass
- **🔧 Mixed Precision Training**: Additional 2x speedup with FP16
- **💾 Memory Efficient**: Process 2-4x longer sequences
- **🎯 Adaptive Sparsity**: Dynamic pattern selection based on input
- **💻 Consumer GPU Optimized**: Tested on RTX 3060 12GB

## 📊 Performance Highlights

### Speedup on RTX 3060

| Sequence Length | Forward Speedup | Training Speedup | Memory Savings |
|----------------|----------------|-----------------|----------------|
| 256 | **4.44x** | - | **2x** |
| 512 | **4.47x** | 1.1x | **2x** |
| 1024 | **2.70x** | **2.02x** | **4x** |
| **Combined** | - | **5.45x** | **~75%** |

*Full analysis: [docs/analysis_report.md](docs/analysis_report.md)*

## 🛠️ Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/asam-attention.git
cd asam-attention

# Create virtual environment (Python 3.8+)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install ASAM
pip install -e .
```

### Basic Usage

```python
import torch
from asam.efficient_attention import FlashASAMLayer

# Create optimized ASAM layer
layer = FlashASAMLayer(
    dim=256,           # Model dimension
    num_heads=4,       # Number of attention heads
    window_size=128,   # Local attention window
)

# Forward pass
x = torch.randn(2, 512, 256)  # [batch, seq_len, dim]
output, info = layer(x, return_info=True)
print(f"Output shape: {output.shape}")
print(f"Sparse ratio: {info['sparse_ratio']:.1%}")
```

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

model = YourModel().cuda()
scaler = GradScaler()

for x, y in dataloader:
    optimizer.zero_grad()
    
    # Automatic mixed precision
    with autocast():
        output = model(x)
        loss = criterion(output, y)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## 📈 Benchmarks

Run the benchmark suite:

```bash
# Full benchmark (requires GPU)
python experiments/run_final_benchmark.py

# Visualize results
python docs/visualize_analysis.py
```

## 🏗️ Project Structure

```
asam-attention/
├── asam/                      # Core library
│   ├── __init__.py
│   ├── asam_layer.py          # Original ASAM implementation
│   ├── efficient_attention.py # Flash Attention optimized
│   ├── asam_layer_optimized.py # True sparse attention
│   ├── adaptive_gate.py       # Adaptive gating mechanism
│   └── sparse_patterns.py     # Sparse attention patterns
├── experiments/               # Benchmark scripts
│   ├── run_final_benchmark.py
│   ├── benchmark_optimized.py
│   └── train_mixed_precision.py
├── tests/                     # Unit tests
├── examples/                  # Usage examples
└── docs/                      # Documentation
```

## 🔬 Optimization Details

### 1. Flash Attention
- Uses PyTorch 2.0+ `scaled_dot_product_attention`
- Automatically selects fastest kernel (Flash/Memory-Efficient/Math)
- Reduces HBM reads from O(N²) to O(N)

### 2. Mixed Precision
- FP16 forward/backward passes
- Tensor Cores acceleration on RTX GPUs
- 50% memory reduction for large sequences

### 3. Adaptive Sparsity
- Local window attention: O(n×window) complexity
- Strided global attention for long-range dependencies
- Dynamic pattern selection based on input characteristics

## 📚 Documentation

- [Performance Analysis Report](docs/analysis_report.md) - Detailed benchmark analysis
- [API Documentation](docs/API.md) - Full API reference
- [Technical Deep Dive](docs/TECHNICAL.md) - Architecture details

## 🧪 Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python tests/test_efficient.py
```

## 🖥️ Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support (tested on RTX 3060)
- **VRAM**: 4GB+ for inference, 8GB+ for training
- **Python**: 3.8 or higher
- **PyTorch**: 2.0 or higher (for Flash Attention)

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📝 Citation

If you use ASAM in your research, please cite:

```bibtex
@software{asam_attention,
  title={ASAM: Adaptive Sparse Attention Module},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/asam-attention}
}
```

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- Flash Attention implementation based on [Dao et al.](https://arxiv.org/abs/2205.14135)
- PyTorch team for `scaled_dot_product_attention`
- Inspired by BigBird, Longformer, and other sparse attention works

---

**Made with ❤️ for efficient deep learning**
