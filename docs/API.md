# ASAM API Reference

Complete API documentation for the ASAM library.

## Table of Contents
- [Core Components](#core-components)
- [Configuration](#configuration)
- [Sparse Patterns](#sparse-patterns)
- [Adaptive Gating](#adaptive-gating)
- [Utilities](#utilities)
- [Advanced Features](#advanced-features)

---

## Core Components

### ASAMLayer

Main ASAM attention layer.

```python
class asam.ASAMLayer(config: ASAMConfig)
```

**Parameters:**
- `config` (ASAMConfig): Configuration object

**Methods:**

#### forward
```python
forward(x: torch.Tensor, mask: Optional[torch.Tensor] = None, return_info: bool = False) -> Tuple[torch.Tensor, Optional[Dict]]
```

Forward pass through ASAM layer.

**Args:**
- `x` (Tensor): Input tensor of shape `[batch, seq_len, dim]`
- `mask` (Optional[Tensor]): Attention mask
- `return_info` (bool): Whether to return gating information

**Returns:**
- `output` (Tensor): Output tensor of shape `[batch, seq_len, dim]`
- `info` (Optional[Dict]): Dictionary containing:
  - `gate_values`: Sparse attention gate values
  - `confidence`: Confidence in sparse attention
  - `pattern_logits`: Pattern selection logits
  - `sparse_ratio`: Ratio of sparse attention used

**Example:**
```python
config = ASAMConfig(dim=512, num_heads=8)
layer = ASAMLayer(config)

x = torch.randn(2, 1024, 512)
output, info = layer(x, return_info=True)

print(f"Sparse ratio: {info['sparse_ratio']:.2%}")
```

---

### ASAMEncoder

Multi-layer ASAM encoder.

```python
class asam.ASAMEncoder(config: ASAMConfig, num_layers: int = 6)
```

**Parameters:**
- `config` (ASAMConfig): Configuration shared across all layers
- `num_layers` (int): Number of ASAM layers

**Example:**
```python
config = ASAMConfig(dim=512, num_heads=8)
encoder = ASAMEncoder(config, num_layers=6)

x = torch.randn(2, 1024, 512)
output = encoder(x)
```

---

## Configuration

### ASAMConfig

Configuration class for ASAM layers.

```python
@dataclass
class ASAMConfig:
    dim: int = 512
    num_heads: int = 8
    dim_head: int = 64
    dropout: float = 0.1
    pattern_type: str = "hierarchical"
    window_size: int = 128
    stride: int = 32
    num_clusters: int = 32
    use_adaptive_gate: bool = True
    gate_hidden_dim: int = 128
    use_gradient_checkpointing: bool = False
```

**Attributes:**

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `dim` | int | 512 | Model dimension |
| `num_heads` | int | 8 | Number of attention heads |
| `dim_head` | int | 64 | Dimension per head (auto-computed if None) |
| `dropout` | float | 0.1 | Dropout rate |
| `pattern_type` | str | "hierarchical" | Sparse pattern type |
| `window_size` | int | 128 | Window size for local pattern |
| `stride` | int | 32 | Stride for strided pattern |
| `num_clusters` | int | 32 | Number of clusters for clustered pattern |
| `use_adaptive_gate` | bool | True | Enable adaptive gating |
| `gate_hidden_dim` | int | 128 | Hidden dimension for gating network |
| `use_gradient_checkpointing` | bool | False | Enable gradient checkpointing |

**Pattern Types:**
- `"local"`: Sliding window attention
- `"strided"`: Fixed stride intervals
- `"random"`: Random sampling per head
- `"clustered"`: Learnable cluster centroids
- `"hierarchical"`: Multi-scale combination (recommended)

---

## Sparse Patterns

### LocalSparsePattern

Sliding window sparse pattern.

```python
class asam.sparse_patterns.LocalSparsePattern(seq_len: int, window_size: int = 128)
```

**Complexity:** O(n × window_size)

**Example:**
```python
pattern = LocalSparsePattern(seq_len=512, window_size=128)
mask = pattern.get_pattern(device)  # [seq_len, seq_len] boolean
```

---

### StridedSparsePattern

Strided sparse pattern.

```python
class asam.sparse_patterns.StridedSparsePattern(seq_len: int, stride: int = 32, local_window: int = 16)
```

**Complexity:** O(n × n/stride)

---

### ClusteredSparsePattern

Cluster-based sparse pattern with learnable centroids.

```python
class asam.sparse_patterns.ClusteredSparsePattern(
    seq_len: int,
    num_clusters: int = 32,
    num_heads: int = 8,
    dim_head: int = 64
)
```

**Methods:**

#### compute_cluster_assignment
```python
compute_cluster_assignment(queries: torch.Tensor, keys: torch.Tensor) -> Tuple[Tensor, Tensor]
```

Compute soft cluster assignments.

**Returns:**
- `q_assign`: [batch, heads, seq_len, num_clusters]
- `k_assign`: [batch, heads, seq_len, num_clusters]

---

### HierarchicalSparsePattern

Multi-scale hierarchical pattern.

```python
class asam.sparse_patterns.HierarchicalSparsePattern(
    seq_len: int,
    scales: List[int] = [4, 16, 64],
    num_heads: int = 8
)
```

**Complexity:** O(n × √n)

---

## Adaptive Gating

### AdaptiveGate

Adaptive gating module for sparse/dense attention switching.

```python
class asam.adaptive_gate.AdaptiveGate(
    dim: int,
    num_heads: int = 8,
    hidden_dim: int = 128,
    num_pools: int = 4
)
```

**Methods:**

#### forward
```python
forward(x: torch.Tensor) -> Tuple[Tensor, Tensor, Tensor]
```

**Returns:**
- `gate_values`: [batch, num_heads, seq_len] - Sparse attention weight
- `confidence`: [batch, num_heads] - Confidence in sparse attention
- `pattern_logits`: [batch, 4] - Pattern selection logits

---

## Utilities

### compute_attention_sparsity

Calculate sparsity ratio of attention mask.

```python
asam.utils.compute_attention_sparsity(attention_mask: torch.Tensor) -> float
```

**Returns:** Sparsity ratio (0.0 = dense, 1.0 = completely sparse)

---

### measure_attention_efficiency

Benchmark attention computation.

```python
asam.utils.measure_attention_efficiency(
    attn_fn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_warmup: int = 3,
    num_runs: int = 10
) -> Tuple[float, float, float]
```

**Returns:** (avg_time_ms, peak_memory_mb, flops)

---

### memory_efficient_attention

Chunked attention for very long sequences.

```python
asam.utils.memory_efficient_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int = 1024,
    mask: Optional[Tensor] = None
) -> torch.Tensor
```

---

## Advanced Features

### Flash Attention Integration

```python
from asam.flash_asam import FlashASAMLayer, HybridASAM

# Flash-enabled ASAM
config = ASAMConfig(dim=512, num_heads=8)
layer = FlashASAMLayer(config)

# Hybrid local (Flash) + global (ASAM)
hybrid = HybridASAM(config, local_window_size=512)
```

---

### Quantization

```python
from asam.quantization import quantize_asam_model

# INT8 quantization
quantized_model = quantize_asam_model(model, dtype=torch.qint8)

# FP16 quantization
fp16_model = quantize_asam_model(model, dtype=torch.float16)
```

---

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for x, y in dataloader:
    with autocast():
        output = model(x)
        loss = criterion(output, y)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## Type Aliases

```python
from typing import Tuple, Optional, Dict

# Common return types
ASAMOutput = Tuple[torch.Tensor, Optional[Dict]]
GateOutput = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
PatternMask = torch.Tensor  # Boolean tensor
```

---

## Constants

```python
# Default configuration values
DEFAULT_DIM = 512
DEFAULT_NUM_HEADS = 8
DEFAULT_WINDOW_SIZE = 128
DEFAULT_STRIDE = 32
DEFAULT_NUM_CLUSTERS = 32

# Pattern type strings
PATTERN_LOCAL = "local"
PATTERN_STRIDED = "strided"
PATTERN_RANDOM = "random"
PATTERN_CLUSTERED = "clustered"
PATTERN_HIERARCHICAL = "hierarchical"
```

---

## Error Handling

Common exceptions:

```python
try:
    layer = ASAMLayer(config)
except ValueError as e:
    # Invalid configuration
    print(f"Config error: {e}")

try:
    output = layer(x)
except RuntimeError as e:
    # CUDA OOM or shape mismatch
    print(f"Runtime error: {e}")
```

---

## Version Information

```python
import asam

print(asam.__version__)  # "0.1.0"
```

---

*For more examples, see the [examples/](examples/) directory.*
