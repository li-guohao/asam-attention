# Plan B: 代码整理与重构

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 解决 FlashASAMLayer 命名冲突，抽取公共模块消除代码重复，补全 `__init__.py` 导出，添加类型标注，统一版本号

**Architecture:** 新建 `asam/_common.py` 存放共享函数；`flash_asam.py::FlashASAMLayer` 重命名为 `FlashAttnASAMLayer`；`__init__.py` 从 10 个导出扩充到 18 个；公开 API 类添加类型标注和 docstring

**Tech Stack:** Python, PyTorch, typing

**Dependency:** 应在 Plan A 完成后执行（Plan A 创建了 pyproject.toml 和 setup.py）

---

### Task B.1: 创建公共模块 asam/_common.py

**Files:**
- Create: `asam/_common.py`

- [ ] **Step 1: 写入 asam/_common.py**

```python
"""
Shared utility functions used across ASAM attention layer implementations.

These are extracted from asam_layer.py and asam_layer_optimized.py to
eliminate code duplication.
"""

from __future__ import annotations

import torch
from typing import Optional, Tuple


def normalize_attention_mask(
    mask: torch.Tensor,
    batch: int,
    heads: int,
    seq_len: int,
) -> torch.Tensor:
    """Normalize attention mask to 4D [batch, heads, seq_len, seq_len].

    Accepts masks of shape [seq_len, seq_len], [batch, seq_len, seq_len],
    or [batch, heads, seq_len, seq_len] and expands them to a consistent
    4D boolean tensor. This is used by both the original ASAMLayer and
    OptimizedASAMLayer to handle user-provided masks.

    Args:
        mask: Input attention mask.
        batch: Target batch size.
        heads: Target number of attention heads.
        seq_len: Target sequence length.

    Returns:
        Boolean mask of shape [batch, heads, seq_len, seq_len].

    Raises:
        ValueError: If mask dimensions or sizes are incompatible.
    """
    if mask.dtype != torch.bool:
        mask = mask.to(torch.bool)

    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)
    elif mask.dim() != 4:
        raise ValueError("attention mask must have 2, 3, or 4 dimensions")

    if mask.size(-2) != seq_len or mask.size(-1) != seq_len:
        raise ValueError("attention mask must match sequence length")

    if mask.size(0) == 1 and batch != 1:
        mask = mask.expand(batch, -1, -1, -1)
    elif mask.size(0) != batch:
        raise ValueError("attention mask batch dimension is not broadcastable")

    if mask.size(1) == 1 and heads != 1:
        mask = mask.expand(-1, heads, -1, -1)
    elif mask.size(1) != heads:
        raise ValueError("attention mask head dimension is not broadcastable")

    return mask


def gather_values_by_positions(
    tensor: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Gather values from a tensor at specified positions along dim=2.

    Given a tensor of shape [batch, heads, seq_len, dim_head] and positions
    of shape [seq_len, context_size], returns a tensor of shape
    [batch, heads, seq_len, context_size, dim_head] where position (i, j)
    selects tensor[:, :, positions[i, j], :].

    Used in sparse attention to gather only the keys/values that are
    included in the sparse pattern, avoiding O(n^2) memory.

    Args:
        tensor: Source tensor [batch, heads, seq_len, dim_head].
        positions: Index tensor [seq_len, context_size] with values in [0, seq_len).

    Returns:
        Gathered tensor [batch, heads, seq_len, context_size, dim_head].
    """
    batch, heads, seq_len, dim_head = tensor.shape
    context_size = positions.size(-1)

    expanded_tensor = tensor.unsqueeze(3).expand(-1, -1, -1, context_size, -1)
    gather_index = positions.unsqueeze(0).unsqueeze(-1).expand(
        batch, -1, -1, -1, dim_head
    )
    return torch.gather(expanded_tensor, 2, gather_index)


def expand_pattern_mask(
    pattern_mask: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    """Expand 2D pattern mask to 3D with per-head dimension.

    Args:
        pattern_mask: Mask of shape [seq_len, seq_len] or [heads, seq_len, seq_len].
        num_heads: Target number of attention heads.

    Returns:
        Expanded mask [num_heads, seq_len, seq_len].

    Raises:
        ValueError: If pattern_mask has unexpected dimensions.
    """
    if pattern_mask.dim() == 2:
        return pattern_mask.unsqueeze(0).expand(num_heads, -1, -1)
    if pattern_mask.dim() == 3:
        return pattern_mask
    raise ValueError("pattern mask must have 2 or 3 dimensions")


def pattern_mask_to_indices(
    pattern_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a boolean pattern mask to index tensors for gather-based attention.

    For each query position, sorts attendable key positions by presence
    (True before False) and returns the top-k indices along with a validity mask.
    This enables efficient sparse attention via torch.gather without building
    full O(n^2) intermediate tensors.

    Args:
        pattern_mask: Boolean mask [num_heads, seq_len, seq_len] where True
            indicates that query i may attend to key j.

    Returns:
        positions: LongTensor [num_heads, seq_len, max_connections] with
            key indices for each query position.
        valid_mask: BoolTensor [num_heads, seq_len, max_connections] where
            True indicates a valid (not padding) connection.
    """
    num_connections = pattern_mask.sum(dim=-1)
    max_connections = max(1, int(num_connections.max().item()))

    sorted_indices = torch.argsort(
        pattern_mask.to(torch.int64), dim=-1, descending=True
    )
    positions = sorted_indices[..., :max_connections]
    valid_mask = (
        torch.arange(max_connections, device=pattern_mask.device).view(1, 1, -1)
        < num_connections.unsqueeze(-1)
    )

    return positions, valid_mask
```

- [ ] **Step 2: Commit**

```bash
git add asam/_common.py
git commit -m "refactor: extract shared attention utilities to asam/_common.py"
```


### Task B.2: 修改 asam_layer.py — 删除重复函数，改为 import

**Files:**
- Modify: `asam/asam_layer.py`

- [ ] **Step 1: 添加导入**

在 `asam/asam_layer.py` 第 14-21 行（`from .sparse_patterns import` 和 `from .adaptive_gate import` 之后），添加：

```python
from ._common import (
    normalize_attention_mask,
    gather_values_by_positions,
    expand_pattern_mask,
    pattern_mask_to_indices,
)
```

- [ ] **Step 2: 删除本地的 `_normalize_attention_mask` 方法**

删除 `ASAMLayer` 类中的 `_normalize_attention_mask` 方法（原第 202-232 行），并将调用处从 `self._normalize_attention_mask(...)` 改为 `normalize_attention_mask(...)`。

具体修改：在 `_compute_sparse_attention` 方法中（约第 168 行），将：
```python
normalized_mask = self._normalize_attention_mask(mask, batch, heads, seq_len)
```
改为：
```python
normalized_mask = normalize_attention_mask(mask, batch, heads, seq_len)
```

- [ ] **Step 3: 删除本地的 `_gather_values_by_positions` 方法**

删除 `ASAMLayer` 类中的 `_gather_values_by_positions` 方法（原第 263-269 行），并将调用处从 `self._gather_values_by_positions(...)` 改为 `gather_values_by_positions(...)`。

在 `_compute_sparse_attention_from_indices` 方法中（约第 246-247 行），将：
```python
gathered_k = self._gather_values_by_positions(k, positions)
gathered_v = self._gather_values_by_positions(v, positions)
```
改为：
```python
gathered_k = gather_values_by_positions(k, positions)
gathered_v = gather_values_by_positions(v, positions)
```

- [ ] **Step 4: 删除本地的 `_expand_pattern_mask` 方法**

删除 `ASAMLayer` 类中的 `_expand_pattern_mask` 方法（原第 292-297 行），并将调用处从 `self._expand_pattern_mask(...)` 改为 `expand_pattern_mask(...)`。

在 `_get_pattern_indices` 方法中（约第 278-287 行），将：
```python
pattern_mask = self._expand_pattern_mask(pattern_mask)
```
和
```python
pattern_mask = self._expand_pattern_mask(pattern.get_pattern(device))
```
改为直接调用函数：
```python
pattern_mask = expand_pattern_mask(pattern_mask, self.num_heads)
```
和
```python
pattern_mask = expand_pattern_mask(pattern.get_pattern(device), self.num_heads)
```

- [ ] **Step 5: 删除本地的 `_pattern_mask_to_indices` 方法**

删除 `ASAMLayer` 类中的 `_pattern_mask_to_indices` 方法（原第 299-307 行），并将调用处从 `self._pattern_mask_to_indices(...)` 改为 `pattern_mask_to_indices(...)`。

在 `_get_pattern_indices` 方法中（约第 281-288 行），将：
```python
return self._pattern_mask_to_indices(pattern_mask)
```
改为：
```python
return pattern_mask_to_indices(pattern_mask)
```

- [ ] **Step 6: 验证删除干净**

```bash
grep -n "_normalize_attention_mask\|_gather_values_by_positions\|_expand_pattern_mask\|_pattern_mask_to_indices" "E:\ASAM Adaptive Sparse Attention Module\repo_tmp\asam\asam_layer.py"
```

Expected: 仅有 import 语句和注释引用，无方法定义

- [ ] **Step 7: 运行测试**

```bash
python -m pytest tests/test_basic.py tests/test_asam.py -v
```

Expected: 全部通过

- [ ] **Step 8: Commit**

```bash
git add asam/asam_layer.py
git commit -m "refactor: replace local utility methods with _common.py imports in asam_layer.py"
```


### Task B.3: 修改 asam_layer_optimized.py — 删除重复函数

**Files:**
- Modify: `asam/asam_layer_optimized.py`

- [ ] **Step 1: 添加导入**

在 `asam/asam_layer_optimized.py` 的 import 区域添加：

```python
from ._common import (
    normalize_attention_mask,
    gather_values_by_positions,
)
```

- [ ] **Step 2: 删除本地的 `_normalize_attention_mask` 方法**

删除 `OptimizedASAMLayer` 类中的 `_normalize_attention_mask` 方法（原约第 211-241 行）。

将所有 `self._normalize_attention_mask(...)` 调用改为 `normalize_attention_mask(...)`。

- [ ] **Step 3: 删除本地的 `_gather_values_by_positions` 方法**

删除 `OptimizedASAMLayer` 类中的 `_gather_values_by_positions` 方法（原约第 203-209 行）。

将所有 `self._gather_values_by_positions(...)` 调用改为 `gather_values_by_positions(...)`。

- [ ] **Step 4: 验证删除干净**

```bash
grep -n "_normalize_attention_mask\|_gather_values_by_positions" "E:\ASAM Adaptive Sparse Attention Module\repo_tmp\asam\asam_layer_optimized.py"
```

Expected: 仅有 import 语句，无方法定义

- [ ] **Step 5: 运行测试**

```bash
python -m pytest tests/test_optimized_layer.py tests/test_efficient.py -v
```

Expected: 全部通过

- [ ] **Step 6: Commit**

```bash
git add asam/asam_layer_optimized.py
git commit -m "refactor: replace local utility methods with _common.py imports in asam_layer_optimized.py"
```


### Task B.4: 更新 continual_asam.py 的导入

**Files:**
- Modify: `asam/continual_asam.py`

- [ ] **Step 1: 添加导入**

在 `asam/continual_asam.py` 的 import 区域添加：

```python
from ._common import (
    expand_pattern_mask,
    pattern_mask_to_indices,
)
```

- [ ] **Step 2: 替换调用**

将 `ContinualASAMLayer` 中所有 `self._expand_pattern_mask(...)` 改为 `expand_pattern_mask(...)`，注意传入 `self.num_heads` 参数。

将 `ContinualASAMLayer` 中所有 `self._pattern_mask_to_indices(...)` 改为 `pattern_mask_to_indices(...)`。

（注意：`ContinualASAMLayer` 继承自 `ASAMLayer`，`_normalize_attention_mask` 和 `_gather_values_by_positions` 等已通过 `ASAMLayer` 的方法调用，不需要在此文件显式修改调用，但显式 import 可避免继承链断裂。）

- [ ] **Step 3: 运行测试**

```bash
python -m pytest tests/test_continual_asam.py -v
```

Expected: 全部通过

- [ ] **Step 4: Commit**

```bash
git add asam/continual_asam.py
git commit -m "refactor: use _common.py imports in continual_asam.py"
```


### Task B.5: 解决 FlashASAMLayer 命名冲突

**Files:**
- Modify: `asam/flash_asam.py`

- [ ] **Step 1: 重命名 flash_asam.py 中的类**

将 `asam/flash_asam.py` 第 22 行：
```python
class FlashASAMLayer(nn.Module):
```
改为：
```python
class FlashAttnASAMLayer(nn.Module):
```

同时将文件中所有 `FlashASAMLayer` 引用改为 `FlashAttnASAMLayer`（包括类型注解、docstring 等）。

- [ ] **Step 2: 验证命名不冲突**

```bash
grep -rn "class FlashASAMLayer" "E:\ASAM Adaptive Sparse Attention Module\repo_tmp\asam/"
```

Expected: 仅在 `efficient_attention.py` 中出现一次，`flash_asam.py` 中已改为 `FlashAttnASAMLayer`

- [ ] **Step 3: Commit**

```bash
git add asam/flash_asam.py
git commit -m "refactor: rename FlashASAMLayer→FlashAttnASAMLayer in flash_asam.py to resolve naming conflict"
```


### Task B.6: 补全 flash_asam.py 中 FlashAttnASAMLayer 的实现

**Files:**
- Modify: `asam/flash_asam.py`

- [ ] **Step 1: 补全 __init__ 方法**

将 `FlashAttnASAMLayer.__init__` 中第 44-45 行的占位注释：
```python
        # Rest of initialization same as ASAMLayer
        # ... (simplified for brevity)
```

替换为完整的初始化代码：

```python
        inner_dim = config.dim_head * config.num_heads

        # Q, K, V projections
        self.to_qkv = nn.Linear(config.dim, inner_dim * 3, bias=False)

        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, config.dim),
            nn.Dropout(config.dropout),
        )

        # Layer normalization (pre-norm)
        self.norm = nn.LayerNorm(config.dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.dim, config.dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim * 4, config.dim),
            nn.Dropout(config.dropout),
        )
        self.ffn_norm = nn.LayerNorm(config.dim)

        # Adaptive attention mechanism
        if config.use_adaptive_gate:
            from .adaptive_gate import DynamicSparseDenseAttention
            self.adaptive_attn = DynamicSparseDenseAttention(
                dim=config.dim,
                num_heads=config.num_heads,
                dim_head=config.dim_head,
                dropout=config.dropout,
            )
        else:
            self.adaptive_attn = None
```

- [ ] **Step 2: 运行测试**

```bash
python -c "
from asam.asam_layer import ASAMConfig
from asam.flash_asam import FlashAttnASAMLayer
import torch
config = ASAMConfig(dim=256, num_heads=4)
layer = FlashAttnASAMLayer(config)
x = torch.randn(2, 128, 256)
out, info = layer(x, return_info=True)
print(f'output shape: {out.shape}')
"
```

Expected: 输出 `output shape: torch.Size([2, 128, 256])`

- [ ] **Step 3: Commit**

```bash
git add asam/flash_asam.py
git commit -m "fix: complete FlashAttnASAMLayer.__init__ implementation"
```


### Task B.7: 补全 __init__.py 导出 + 版本号更新

**Files:**
- Modify: `asam/__init__.py`

- [ ] **Step 1: 更新版本号**

将 `asam/__init__.py` 第 22 行：
```python
__version__ = "1.1.1"
```
改为：
```python
__version__ = "1.2.0"
```

- [ ] **Step 2: 补全导出**

将 `asam/__init__.py` 第 23-34 行的 `__all__` 替换为：

```python
__all__ = [
    # Core layer
    "ASAMLayer",
    "ASAMConfig",
    "ASAMEncoder",
    # Efficient variants
    "FlashASAMLayer",       # from efficient_attention (SDPA-based)
    "FlashAttnASAMLayer",   # from flash_asam (flash-attn library)
    "EfficientASAMLayer",   # from efficient_attention
    "OptimizedASAMLayer",   # from asam_layer_optimized
    "HybridASAM",           # from flash_asam
    # Continual learning
    "ContinualASAMLayer",
    "ContinualASAMConfig",
    "PrototypeContinualASAMLayer",
    # Sparse patterns
    "LocalSparsePattern",
    "StridedSparsePattern",
    "RandomSparsePattern",
    "ClusteredSparsePattern",
    "HierarchicalSparsePattern",
    # Gating
    "AdaptiveGate",
    "DynamicSparseDenseAttention",
]
```

- [ ] **Step 3: 更新导入语句**

在文件顶部 import 区域添加缺失的导入：

```python
from .efficient_attention import FlashASAMLayer, EfficientASAMLayer
from .asam_layer_optimized import OptimizedASAMLayer
from .flash_asam import FlashAttnASAMLayer, HybridASAM
from .sparse_patterns import HierarchicalSparsePattern
from .adaptive_gate import DynamicSparseDenseAttention
```

- [ ] **Step 4: 验证所有导出可导入**

```bash
python -c "
from asam import (
    ASAMLayer, ASAMConfig, ASAMEncoder,
    FlashASAMLayer, FlashAttnASAMLayer, EfficientASAMLayer,
    OptimizedASAMLayer, HybridASAM,
    ContinualASAMLayer, ContinualASAMConfig, PrototypeContinualASAMLayer,
    LocalSparsePattern, StridedSparsePattern, RandomSparsePattern,
    ClusteredSparsePattern, HierarchicalSparsePattern,
    AdaptiveGate, DynamicSparseDenseAttention,
)
print('All imports OK')
print(f'Version: {__import__(\"asam\").__version__}')
assert FlashASAMLayer is not FlashAttnASAMLayer, 'Naming conflict not resolved'
print('Naming check OK')
"
```

Expected: 输出 `All imports OK` 和 `Version: 1.2.0` 和 `Naming check OK`

- [ ] **Step 5: 运行全部测试**

```bash
python -m pytest tests/ -q
```

Expected: 全部通过，不少于 32 个测试

- [ ] **Step 6: Commit**

```bash
git add asam/__init__.py
git commit -m "feat: bump version to 1.2.0 and expand public API exports from 10 to 18 symbols"
```


### Task B.8: 添加类型标注（公开 API）

**Files:**
- Modify: `asam/efficient_attention.py`, `asam/adaptive_gate.py`, `asam/sparse_patterns.py`, `asam/flash_asam.py`, `asam/asam_layer_optimized.py`

- [ ] **Step 1: 为 EfficientASAMLayer 和 FlashASAMLayer 添加类型标注**

在 `asam/efficient_attention.py` 中，为 `EfficientASAMLayer.__init__`、`FlashASAMLayer.__init__` 和 `forward` 方法添加类型标注：

```python
from __future__ import annotations
from typing import Optional, Dict, Tuple

class EfficientASAMLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 128,
        dropout: float = 0.1,
        use_local_attention: bool = True,
    ) -> None: ...

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_info: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]: ...
```

- [ ] **Step 2: 为 DynamicSparseDenseAttention 和 AdaptiveGate 添加类型标注**

在 `asam/adaptive_gate.py` 中：

```python
class AdaptiveGate(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        hidden_dim: int = 128,
        num_pools: int = 4,
        temperature: float = 1.0,
    ) -> None: ...

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

class DynamicSparseDenseAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ) -> None: ...

    def forward(
        self,
        x: torch.Tensor,
        sparse_attn_fn: Optional[callable] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]: ...
```

- [ ] **Step 3: 为 OptimizedASAMLayer 添加类型标注**

在 `asam/asam_layer_optimized.py` 中：

```python
class OptimizedASAMLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 128,
        stride: int = 32,
        dropout: float = 0.1,
        use_adaptive_gate: bool = True,
        pattern_type: str = "local",
    ) -> None: ...

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_info: bool = False,
    ) -> Tuple[torch.Tensor, Optional[dict]]: ...
```

- [ ] **Step 4: 为 HybridASAM 添加类型标注**

在 `asam/flash_asam.py` 中：

```python
class HybridASAM(nn.Module):
    def __init__(
        self,
        config: ASAMConfig,
        local_window_size: int = 512,
        use_flash_local: bool = True,
    ) -> None: ...

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor: ...
```

- [ ] **Step 5: 为 HierarchicalSparsePattern 添加类型标注**

在 `asam/sparse_patterns.py` 中：

```python
class HierarchicalSparsePattern(SparsePattern):
    def __init__(
        self,
        seq_len: int,
        scales: Optional[List[int]] = None,
        num_heads: int = 8,
    ) -> None: ...

    def combine_patterns(self, device: torch.device) -> torch.Tensor: ...
```

- [ ] **Step 6: 运行 mypy 检查**

```bash
python -m mypy asam/efficient_attention.py asam/adaptive_gate.py asam/sparse_patterns.py asam/flash_asam.py asam/asam_layer_optimized.py --ignore-missing-imports
```

Expected: 无新增类型错误（忽略 torch 等第三方库的 missing import）

- [ ] **Step 7: 运行全部测试**

```bash
python -m pytest tests/ -q
```

Expected: 全部通过

- [ ] **Step 8: Commit**

```bash
git add asam/efficient_attention.py asam/adaptive_gate.py asam/sparse_patterns.py asam/flash_asam.py asam/asam_layer_optimized.py
git commit -m "docs: add type annotations to public API classes"
```


### Task B.9: Plan B 最终验证

- [ ] **Step 1: 运行全部测试**

```bash
python -m pytest tests/ -q -v
```

Expected: 所有测试通过，测试数 >= 32

- [ ] **Step 2: 验证唯一函数定义**

```bash
grep -rn "def _normalize_attention_mask\|def _gather_values_by_positions\|def _pattern_mask_to_indices\|def _expand_pattern_mask" "E:\ASAM Adaptive Sparse Attention Module\repo_tmp\asam/"
```

Expected: 仅在 `_common.py` 中出现（`normalize_attention_mask` 等不带下划线前缀的函数名）

- [ ] **Step 3: 验证导入完整性**

```bash
python -c "import asam; print(len(asam.__all__)); [print(f'  {x}') for x in sorted(asam.__all__)]"
```

Expected: 输出 18 个符号，且 `FlashASAMLayer` 和 `FlashAttnASAMLayer` 为不同对象
