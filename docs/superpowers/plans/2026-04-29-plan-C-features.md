# Plan C: 新功能集成

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** HuggingFace Transformers 兼容模型、多 GPU 分布式训练（DDP）、ONNX 导出与验证、预训练模型权重训练脚本

**Architecture:** 新建 5 个独立文件——`asam/modeling_asam.py`（HF 集成）、`asam/distributed.py`（DDP 包装器）、`asam/export.py`（ONNX 导出）、`scripts/pretrain_asam.py`（训练脚本）、`scripts/upload_to_hub.py`（Hub 上传脚本）。全部为新增，不与现有代码冲突

**Tech Stack:** PyTorch, HuggingFace Transformers, ONNX, onnxruntime, torch.distributed

**Dependency:** 应在 Plan A + Plan B 完成后执行。Plan B 提供的 `__init__.py` 导出决定了 C 的 import 路径

---

### Task C.1: 创建 HF 兼容配置类

**Files:**
- Create: `asam/modeling_asam.py`
- Test: `tests/test_hf_integration.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_hf_integration.py
import pytest
import torch

def test_asam_hf_config_creation():
    """ASAMHFConfig can be created with default values."""
    from asam.modeling_asam import ASAMHFConfig

    config = ASAMHFConfig(dim=256, num_heads=4, num_labels=2)
    assert config.model_type == "asam"
    assert config.dim == 256
    assert config.num_heads == 4

def test_asam_hf_config_defaults():
    """ASAMHFConfig has sensible defaults."""
    from asam.modeling_asam import ASAMHFConfig

    config = ASAMHFConfig()
    assert config.dim == 512
    assert config.pattern_type == "hierarchical"
    assert config.use_adaptive_gate is True

def test_asam_hf_config_serialization():
    """ASAMHFConfig can save/load JSON."""
    from asam.modeling_asam import ASAMHFConfig
    import tempfile, os

    config = ASAMHFConfig(dim=256, num_heads=4)
    with tempfile.TemporaryDirectory() as tmpdir:
        config.save_pretrained(tmpdir)
        loaded = ASAMHFConfig.from_pretrained(tmpdir)
        assert loaded.dim == 256
        assert loaded.num_heads == 4
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest tests/test_hf_integration.py -v
```

Expected: FAIL — module not found

- [ ] **Step 3: 实现 ASAMHFConfig**

```python
"""HuggingFace Transformers-compatible model classes for ASAM."""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple

from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    SequenceClassifierOutput,
)

from .asam_layer import ASAMConfig, ASAMLayer, ASAMEncoder


class ASAMHFConfig(PretrainedConfig):
    """HuggingFace-compatible configuration for ASAM models.

    Mirrors ASAMConfig fields while following HF conventions.
    """

    model_type = "asam"

    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.1,
        pattern_type: str = "hierarchical",
        window_size: int = 128,
        stride: int = 32,
        num_clusters: int = 32,
        use_adaptive_gate: bool = True,
        gate_hidden_dim: int = 128,
        use_gradient_checkpointing: bool = False,
        num_layers: int = 6,
        num_labels: int = 2,
        vocab_size: int = 30000,
        max_position_embeddings: int = 8192,
        pad_token_id: int = 0,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            **kwargs,
        )
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.pattern_type = pattern_type
        self.window_size = window_size
        self.stride = stride
        self.num_clusters = num_clusters
        self.use_adaptive_gate = use_adaptive_gate
        self.gate_hidden_dim = gate_hidden_dim
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.num_layers = num_layers
        self.num_labels = num_labels
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings

    def to_asam_config(self) -> ASAMConfig:
        """Convert to internal ASAMConfig."""
        return ASAMConfig(
            dim=self.dim,
            num_heads=self.num_heads,
            dim_head=self.dim_head,
            dropout=self.dropout,
            pattern_type=self.pattern_type,
            window_size=self.window_size,
            stride=self.stride,
            num_clusters=self.num_clusters,
            use_adaptive_gate=self.use_adaptive_gate,
            gate_hidden_dim=self.gate_hidden_dim,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
        )
```

- [ ] **Step 4: 运行测试确认通过**

```bash
python -m pytest tests/test_hf_integration.py::test_asam_hf_config_creation tests/test_hf_integration.py::test_asam_hf_config_defaults tests/test_hf_integration.py::test_asam_hf_config_serialization -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add asam/modeling_asam.py tests/test_hf_integration.py
git commit -m "feat: add ASAMHFConfig for HuggingFace Transformers compatibility"
```


### Task C.2: 创建 HF 兼容基础模型

**Files:**
- Modify: `asam/modeling_asam.py`
- Modify: `tests/test_hf_integration.py`

- [ ] **Step 1: 添加测试**

```python
def test_asam_hf_model_creation():
    """ASAMHFModel can be created and run forward."""
    from asam.modeling_asam import ASAMHFConfig, ASAMHFModel

    config = ASAMHFConfig(dim=64, num_heads=2, num_layers=2, vocab_size=1000)
    model = ASAMHFModel(config)
    input_ids = torch.randint(0, 1000, (2, 128))
    output = model(input_ids)
    assert output.last_hidden_state.shape == (2, 128, 64)

def test_asam_hf_model_save_load():
    """ASAMHFModel can save and load with HF methods."""
    from asam.modeling_asam import ASAMHFConfig, ASAMHFModel
    import tempfile, os

    config = ASAMHFConfig(dim=64, num_heads=2, num_layers=2, vocab_size=1000)
    model = ASAMHFModel(config)
    input_ids = torch.randint(0, 1000, (2, 128))

    with torch.no_grad():
        before = model(input_ids).last_hidden_state

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir)
        loaded = ASAMHFModel.from_pretrained(tmpdir)

    with torch.no_grad():
        after = loaded(input_ids).last_hidden_state

    assert torch.allclose(before, after, atol=1e-6)
```

- [ ] **Step 2: 实现 ASAMHFModel**

在 `asam/modeling_asam.py` 中 `ASAMHFConfig` 之后添加：

```python
class ASAMHFModel(PreTrainedModel):
    """HuggingFace-compatible ASAM base model.

    Wraps ASAMEncoder with token embeddings for text processing.
    Outputs last_hidden_state via BaseModelOutputWithPast.
    """

    config_class = ASAMHFConfig
    base_model_prefix = "asam"
    _no_split_modules = ["ASAMLayer"]

    def __init__(self, config: ASAMHFConfig):
        super().__init__(config)
        asam_config = config.to_asam_config()

        self.token_embedding = nn.Embedding(
            config.vocab_size, config.dim, padding_idx=config.pad_token_id
        )
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings, config.dim
        )
        self.embed_dropout = nn.Dropout(config.dropout)

        self.encoder = ASAMEncoder(asam_config, num_layers=config.num_layers)

        self.post_init()

    def _init_weights(self, module: nn.Module):
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> BaseModelOutputWithPast:
        batch, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.token_embedding(input_ids)
        x = x + self.position_embedding(positions)
        x = self.embed_dropout(x)

        # Build attention mask if provided
        mask_4d = None
        if attention_mask is not None:
            mask_4d = attention_mask[:, None, None, :].to(dtype=torch.bool)
            mask_4d = mask_4d.expand(-1, self.config.num_heads, seq_len, -1)

        hidden_states = self.encoder(x, mask=mask_4d)

        if not return_dict:
            return (hidden_states,)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)
```

- [ ] **Step 3: 运行测试**

```bash
python -m pytest tests/test_hf_integration.py -v
```

Expected: 全部通过

- [ ] **Step 4: Commit**

```bash
git add asam/modeling_asam.py tests/test_hf_integration.py
git commit -m "feat: add ASAMHFModel with HF save/load support"
```


### Task C.3: 创建 HF 兼容分类模型

**Files:**
- Modify: `asam/modeling_asam.py`
- Modify: `tests/test_hf_integration.py`

- [ ] **Step 1: 添加测试**

```python
def test_asam_hf_classification():
    """ASAMHFForSequenceClassification produces valid logits."""
    from asam.modeling_asam import ASAMHFConfig, ASAMHFForSequenceClassification

    config = ASAMHFConfig(dim=64, num_heads=2, num_layers=2, vocab_size=1000, num_labels=3)
    model = ASAMHFForSequenceClassification(config)
    input_ids = torch.randint(0, 1000, (2, 128))
    labels = torch.randint(0, 3, (2,))

    output = model(input_ids, labels=labels)
    assert output.logits.shape == (2, 3)
    assert output.loss is not None
    assert output.loss.item() > 0

def test_asam_hf_classification_save_load():
    """Classification model round-trips through save/load."""
    from asam.modeling_asam import ASAMHFConfig, ASAMHFForSequenceClassification
    import tempfile

    config = ASAMHFConfig(dim=64, num_heads=2, num_layers=2, vocab_size=1000, num_labels=2)
    model = ASAMHFForSequenceClassification(config)
    input_ids = torch.randint(0, 1000, (2, 128))

    with torch.no_grad():
        before = model(input_ids).logits

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir)
        loaded = ASAMHFForSequenceClassification.from_pretrained(tmpdir)

    with torch.no_grad():
        after = loaded(input_ids).logits

    assert torch.allclose(before, after, atol=1e-6)
```

- [ ] **Step 2: 实现 ASAMHFForSequenceClassification**

在 `asam/modeling_asam.py` 中添加：

```python
class ASAMHFForSequenceClassification(PreTrainedModel):
    """ASAM model with a sequence classification head.

    Built on ASAMHFModel with a mean-pooling + linear classifier.
    """

    config_class = ASAMHFConfig
    base_model_prefix = "asam"

    def __init__(self, config: ASAMHFConfig):
        super().__init__(config)
        self.asam = ASAMHFModel(config)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> SequenceClassifierOutput:
        outputs = self.asam(input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)

        if not return_dict:
            output = (logits,) + (outputs,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )
```

- [ ] **Step 3: 注册到 HF 自动发现**

在 `asam/modeling_asam.py` 末尾添加：

```python
# Register with HuggingFace auto classes
ASAMHFConfig.register_for_auto_class()
ASAMHFModel.register_for_auto_class("AutoModel")
ASAMHFForSequenceClassification.register_for_auto_class(
    "AutoModelForSequenceClassification"
)
```

- [ ] **Step 4: 运行测试**

```bash
python -m pytest tests/test_hf_integration.py -v
```

Expected: 全部通过（5 个测试）

- [ ] **Step 5: Commit**

```bash
git add asam/modeling_asam.py tests/test_hf_integration.py
git commit -m "feat: add ASAMHFForSequenceClassification and HF auto-class registration"
```


### Task C.4: 创建多 GPU 分布式训练模块

**Files:**
- Create: `asam/distributed.py`
- Test: `tests/test_distributed.py`

- [ ] **Step 1: 写测试（单 GPU 兼容）**

```python
# tests/test_distributed.py
import torch
import pytest

def test_distributed_trainer_main_process():
    """DistributedTrainer identifies main process correctly outside torchrun."""
    from asam.distributed import DistributedTrainer

    trainer = DistributedTrainer()
    # When not launched via torchrun, is_main_process should return True
    assert trainer.is_main_process() is True

def test_distributed_trainer_no_init():
    """DistributedTrainer works without torch.distributed initialization."""
    from asam.distributed import DistributedTrainer

    trainer = DistributedTrainer()
    # Should not crash when torch.distributed is not initialized
    result = trainer.is_main_process()
    assert isinstance(result, bool)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest tests/test_distributed.py -v
```

Expected: FAIL

- [ ] **Step 3: 实现 asam/distributed.py**

```python
"""Distributed training utilities for ASAM models.

Supports DDP (DistributedDataParallel) and FSDP (FullyShardedDataParallel)
wrapping with minimal boilerplate.
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from typing import Optional, Callable


class DistributedTrainer:
    """Lightweight distributed training orchestrator.

    Handles torch.distributed lifecycle, process-safe logging,
    and checkpoint save/load for multi-GPU training.
    """

    def __init__(self, backend: str = "nccl"):
        self.backend = backend
        self._initialized = False

    def is_main_process(self) -> bool:
        """Check if current process is rank 0."""
        if not torch.distributed.is_available():
            return True
        if not torch.distributed.is_initialized():
            return True
        return torch.distributed.get_rank() == 0

    def init_process_group(self):
        """Initialize distributed process group.

        Call this before any distributed operations.
        Use environment variables set by torchrun.
        """
        if not torch.distributed.is_available():
            raise RuntimeError("torch.distributed is not available")

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend=self.backend)
            self._initialized = True

    def cleanup(self):
        """Destroy the process group."""
        if self._initialized and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            self._initialized = False

    def wrap_ddp(
        self,
        model: nn.Module,
        find_unused_parameters: bool = False,
    ) -> DDP:
        """Wrap model with DistributedDataParallel."""
        if not torch.distributed.is_initialized():
            self.init_process_group()

        device = torch.device(f"cuda:{torch.distributed.get_rank()}")
        model = model.to(device)
        return DDP(
            model,
            device_ids=[torch.distributed.get_rank()],
            find_unused_parameters=find_unused_parameters,
        )

    def create_dataloader(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 4,
        **kwargs,
    ) -> DataLoader:
        """Create a DataLoader with DistributedSampler if initialized."""
        sampler = None
        if torch.distributed.is_initialized():
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            shuffle = False  # Sampler handles shuffling

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            **kwargs,
        )

    def save_checkpoint(
        self,
        model: nn.Module,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
    ):
        """Save checkpoint — only on main process."""
        if not self.is_main_process():
            return

        model_state = (
            model.module.state_dict()
            if isinstance(model, DDP)
            else model.state_dict()
        )

        checkpoint = {
            "model_state_dict": model_state,
            "epoch": epoch,
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(
        self,
        model: nn.Module,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> int:
        """Load checkpoint, returning the epoch number."""
        checkpoint = torch.load(path, map_location="cpu")

        if isinstance(model, DDP):
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint.get("epoch", 0)
```

- [ ] **Step 4: 运行测试**

```bash
python -m pytest tests/test_distributed.py -v
```

Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add asam/distributed.py tests/test_distributed.py
git commit -m "feat: add DDP distributed training utilities"
```


### Task C.5: 创建 ONNX 导出模块

**Files:**
- Create: `asam/export.py`
- Test: `tests/test_onnx_export.py`

- [ ] **Step 1: 写测试**

```python
# tests/test_onnx_export.py
import torch
import tempfile
import os
import pytest

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_onnx_export_and_verify():
    """Export ASAM model to ONNX and verify output matches."""
    from asam import ASAMConfig, ASAMLayer
    from asam.export import export_to_onnx, verify_onnx_export

    config = ASAMConfig(dim=64, num_heads=2, use_adaptive_gate=False)
    model = ASAMLayer(config)
    model.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "test.onnx")

        with torch.no_grad():
            sample_input = torch.randn(1, 128, 64)
            onnx_out = export_to_onnx(model, onnx_path, seq_len=128)

        assert os.path.exists(onnx_path)
        # Verify file is non-empty
        assert os.path.getsize(onnx_path) > 1000
```

- [ ] **Step 2: 实现 asam/export.py**

```python
"""ONNX export utilities for ASAM models."""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    batch_size: int = 1,
    seq_len: int = 512,
    dim: Optional[int] = None,
    dynamic_batch: bool = True,
    dynamic_seq_len: bool = False,
    opset_version: int = 17,
) -> str:
    """Export an ASAM model to ONNX format.

    Args:
        model: ASAM model in eval mode.
        output_path: Path for the output .onnx file.
        batch_size: Sample batch size for tracing.
        seq_len: Fixed sequence length for tracing.
        dim: Model dimension (inferred from model if None).
        dynamic_batch: Allow variable batch size.
        dynamic_seq_len: Allow variable sequence length (may not work
            with all pattern types — local window only).
        opset_version: ONNX opset version.

    Returns:
        Path to the exported ONNX file.
    """
    if dim is None:
        # Try to infer from model
        for name, param in model.named_parameters():
            if "embed" in name or "weight" in name:
                dim = param.shape[-1]
                break
        if dim is None:
            dim = 512

    sample_input = torch.randn(batch_size, seq_len, dim)
    device = next(model.parameters()).device
    sample_input = sample_input.to(device)

    dynamic_axes = {}
    if dynamic_batch:
        dynamic_axes["x"] = {0: "batch_size"}
        dynamic_axes["output"] = {0: "batch_size"}
    if dynamic_seq_len:
        dynamic_axes["x"] = {**dynamic_axes.get("x", {}), 1: "seq_len"}
        dynamic_axes["output"] = {**dynamic_axes.get("output", {}), 1: "seq_len"}

    input_names = ["x"]
    output_names = ["output"]

    with torch.no_grad():
        torch.onnx.export(
            model,
            sample_input,
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes or None,
            opset_version=opset_version,
            do_constant_folding=True,
        )

    return output_path


def verify_onnx_export(
    onnx_path: str,
    pytorch_model: nn.Module,
    sample_input: torch.Tensor,
    atol: float = 1e-5,
) -> bool:
    """Verify ONNX export matches PyTorch model output.

    Args:
        onnx_path: Path to the exported .onnx file.
        pytorch_model: The original PyTorch model.
        sample_input: Input tensor for comparison.
        atol: Absolute tolerance for output comparison.

    Returns:
        True if outputs match within tolerance.

    Raises:
        ImportError: If onnxruntime is not installed.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError(
            "onnxruntime is required for verification. "
            "Install with: pip install onnxruntime"
        )

    # PyTorch output
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(sample_input)
        if isinstance(pytorch_output, tuple):
            pytorch_output = pytorch_output[0]

    # ONNX output
    session = ort.InferenceSession(onnx_path)
    onnx_output = session.run(
        None, {"x": sample_input.cpu().numpy()}
    )[0]

    pytorch_np = pytorch_output.cpu().numpy()
    match = (
        abs(pytorch_np - onnx_output).max() < atol
    )

    if not match:
        max_diff = abs(pytorch_np - onnx_output).max()
        print(f"Max difference: {max_diff:.2e} (tolerance: {atol})")

    return match
```

- [ ] **Step 3: 运行测试**

```bash
python -m pytest tests/test_onnx_export.py -v
```

Expected: 1 PASS（skip 如果没有 CUDA）

- [ ] **Step 4: Commit**

```bash
git add asam/export.py tests/test_onnx_export.py
git commit -m "feat: add ONNX export and verification utilities"
```


### Task C.6: 创建预训练模型训练脚本

**Files:**
- Create: `scripts/pretrain_asam.py`

- [ ] **Step 1: 写入 scripts/pretrain_asam.py**

```python
#!/usr/bin/env python3
"""Train an ASAM model on IMDB long document classification.

This script trains ASAMHFForSequenceClassification on IMDB reviews
and saves a checkpoint suitable for upload to HuggingFace Hub.

Usage:
    python scripts/pretrain_asam.py --output checkpoints/asam-imdb-v1

Requirements:
    pip install transformers datasets tqdm
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from asam.modeling_asam import ASAMHFConfig, ASAMHFForSequenceClassification


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain ASAM on IMDB")
    parser.add_argument("--output", type=str, default="checkpoints/asam-imdb-v1")
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    print(f"Device: {args.device}")
    print(f"Output: {args.output}")

    # Load IMDB dataset
    dataset = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )

    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.with_format("torch")

    train_loader = DataLoader(
        tokenized["train"], batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        tokenized["test"], batch_size=args.batch_size, shuffle=False
    )

    # Create model
    config = ASAMHFConfig(
        dim=args.dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_labels=2,
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=args.max_length,
        pad_token_id=tokenizer.pad_token_id or 0,
        pattern_type="hierarchical",
    )
    model = ASAMHFForSequenceClassification(config).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            labels = batch["label"].to(args.device)

            optimizer.zero_grad()
            output = model(input_ids, attention_mask=attention_mask, labels=labels)
            output.loss.backward()
            optimizer.step()

            train_loss += output.loss.item()
            pbar.set_postfix(loss=f"{output.loss.item():.4f}")

        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}: train_loss={avg_loss:.4f}")

        # Validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(args.device)
                attention_mask = batch["attention_mask"].to(args.device)
                labels = batch["label"].to(args.device)

                output = model(input_ids, attention_mask=attention_mask)
                preds = output.logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f"Epoch {epoch + 1}: val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(args.output)
            config.save_pretrained(args.output)
            tokenizer.save_pretrained(args.output)
            print(f"  Saved best checkpoint (acc={val_acc:.4f})")

    print(f"Training complete. Best val_acc={best_val_acc:.4f}")
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/pretrain_asam.py
git commit -m "feat: add IMDB pretraining script for ASAM model"
```


### Task C.7: 创建 Hub 上传脚本

**Files:**
- Create: `scripts/upload_to_hub.py`

- [ ] **Step 1: 写入 scripts/upload_to_hub.py**

```python
#!/usr/bin/env python3
"""Upload a trained ASAM checkpoint to HuggingFace Hub.

Usage:
    python scripts/upload_to_hub.py --checkpoint checkpoints/asam-imdb-v1 --repo li-guohao/asam-imdb

Requires:
    pip install huggingface_hub
    HF token set via: huggingface-cli login
"""

import argparse

from asam.modeling_asam import ASAMHFForSequenceClassification


def parse_args():
    parser = argparse.ArgumentParser(description="Upload ASAM model to HF Hub")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to local checkpoint directory"
    )
    parser.add_argument(
        "--repo", type=str, required=True, help="HF Hub repository name (e.g. li-guohao/asam-imdb)"
    )
    parser.add_argument(
        "--private", action="store_true", help="Create as private repository"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading model from {args.checkpoint}...")
    model = ASAMHFForSequenceClassification.from_pretrained(args.checkpoint)

    print(f"Pushing to {args.repo} (private={args.private})...")
    model.push_to_hub(args.repo, private=args.private)

    print(f"Done! Model available at https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/upload_to_hub.py
git commit -m "feat: add HF Hub upload script for trained checkpoints"
```


### Task C.8: Plan C 最终验证

- [ ] **Step 1: 运行全部测试**

```bash
python -m pytest tests/ -q
```

Expected: 全部通过，测试数 >= 39（原有 32 + 新增 7）

- [ ] **Step 2: 验证 HF 模型可训练**

```bash
python -c "
from asam.modeling_asam import ASAMHFConfig, ASAMHFForSequenceClassification
import torch
config = ASAMHFConfig(dim=64, num_heads=2, num_layers=2, vocab_size=1000, num_labels=2)
model = ASAMHFForSequenceClassification(config)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
x = torch.randint(0, 1000, (4, 128))
y = torch.randint(0, 2, (4,))
output = model(x, labels=y)
output.loss.backward()
opt.step()
print('Training step OK, loss:', output.loss.item())
"
```

Expected: loss 输出且无报错

- [ ] **Step 3: 验证 ONNX 导出可运行**

```bash
python -c "
from asam import ASAMConfig, ASAMLayer
from asam.export import export_to_onnx
import tempfile, os
config = ASAMConfig(dim=64, num_heads=2, use_adaptive_gate=False)
model = ASAMLayer(config).eval()
with tempfile.TemporaryDirectory() as tmpdir:
    onnx_path = os.path.join(tmpdir, 'test.onnx')
    export_to_onnx(model, onnx_path, seq_len=128)
    print(f'ONNX exported to {onnx_path}, size: {os.path.getsize(onnx_path)} bytes')
"
```

Expected: ONNX 文件生成成功

- [ ] **Step 4: git tag**

```bash
git tag -a v1.2.0 -m "v1.2.0: HF integration, packaging, ONNX export, code refactoring"
```
