# Plan A: 打包与基础设施

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 项目可 `pip install -e .` 安装，GitHub Actions CI 正确触发，版本号统一 v1.2.0

**Architecture:** 创建 pyproject.toml（现代打包标准）+ setup.py（兼容），补全 requirements.txt，创建 .gitignore 和 MANIFEST.in，写 CHANGELOG.md，修复 CI 配置

**Tech Stack:** setuptools, wheel, pip, GitHub Actions

**Note:** `asam/__init__.py` 的版本号修改在 Plan B 中处理（B 也修改此文件，避免冲突）。

---

### Task A.1: 目录结构检查

**Files:**
- 已存在，当前任务无创建或修改

- [ ] **Step 1: 确认项目根目录**

```bash
ls E:\ASAM\ Adaptive\ Sparse\ Attention\ Module\repo_tmp/
```

Expected: 看到 `asam/`, `tests/`, `docs/`, `README.md`, `requirements.txt` 等

- [ ] **Step 2: 确认当前版本号**

```bash
grep __version__ "E:\ASAM Adaptive Sparse Attention Module\repo_tmp\asam\__init__.py"
```

Expected: `__version__ = "1.1.1"`


### Task A.2: 创建 pyproject.toml

**Files:**
- Create: `pyproject.toml`
- Modify: 无

- [ ] **Step 1: 写入 pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "asam-attention"
version = "1.2.0"
description = "Adaptive Sparse Attention Mechanism for long-sequence modeling"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.8"
authors = [{name = "Guohao Li", email = "liguohao@gmail.com"}]
keywords = ["attention", "sparse-attention", "transformer", "deep-learning", "pytorch"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
    "torch>=2.0.0",
    "numpy>=1.20.0",
]

[project.optional-dependencies]
viz = ["matplotlib>=3.5.0", "seaborn>=0.12.0"]
dev = ["pytest>=7.0", "black>=22.0", "flake8>=4.0", "mypy>=1.0", "isort>=5.0"]
hf = ["transformers>=4.30", "datasets>=2.14", "huggingface_hub>=0.16"]
export = ["onnx>=1.14", "onnxruntime>=1.15"]
all = ["asam-attention[viz,dev,hf,export]"]

[project.urls]
Homepage = "https://github.com/li-guohao/asam-attention"
Documentation = "https://github.com/li-guohao/asam-attention#readme"
Repository = "https://github.com/li-guohao/asam-attention"
Issues = "https://github.com/li-guohao/asam-attention/issues"

[tool.setuptools.packages.find]
include = ["asam*"]

[tool.black]
line-length = 100
target-version = ["py38", "py39", "py310", "py311", "py312"]

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

- [ ] **Step 2: 验证 pyproject.toml 语法**

```bash
python -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))" 2>&1 || python -c "import toml; toml.load('pyproject.toml')" 2>&1 || echo "Need to validate manually"
```

Expected: 无报错（tomllib/toml 可能存在可能不存在，如都不存在则手动检查无语法问题）

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add pyproject.toml for v1.2.0 packaging"
```


### Task A.3: 创建 setup.py（兼容旧 pip）

**Files:**
- Create: `setup.py`

- [ ] **Step 1: 写入 setup.py**

```python
"""Compatibility shim for older pip versions.
All configuration lives in pyproject.toml."""
from setuptools import setup

setup()
```

- [ ] **Step 2: 测试 pip install -e**

```bash
pip install -e . 2>&1
```

Expected: 成功安装，输出含 `Successfully installed asam-attention-1.2.0`

- [ ] **Step 3: 验证导入**

```bash
python -c "import asam; print(asam.__version__)"
```

Expected: 输出 `1.1.1`（版本号在 Plan B 中更新为 `1.2.0`）

- [ ] **Step 4: Commit**

```bash
git add setup.py
git commit -m "chore: add setup.py compatibility shim"
```


### Task A.4: 补全 requirements.txt

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: 将 requirements.txt 替换为完整版**

```text
# Core dependencies
torch>=2.0.0
numpy>=1.20.0
torchvision>=0.15.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.12.0

# Development
pytest>=7.0.0
black>=22.0.0
flake8>=4.0.0
mypy>=1.0.0
isort>=5.0.0

# HuggingFace integration (optional but recommended)
transformers>=4.30.0
datasets>=2.14.0
huggingface_hub>=0.16.0

# ONNX export (optional)
onnx>=1.14.0
onnxruntime>=1.15.0

# Training utilities
tqdm>=4.64.0
```

- [ ] **Step 2: Commit**

```bash
git add requirements.txt
git commit -m "chore: complete requirements.txt with optional dependencies"
```


### Task A.5: 创建 .gitignore

**Files:**
- Create: `.gitignore`

- [ ] **Step 1: 写入 .gitignore**

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
*.egg-info/
dist/
build/
eggs/
*.egg
.eggs/

# Virtual environments
.venv/
venv/
.env
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Testing
.pytest_cache/
.coverage
htmlcov/
coverage.xml
.tox/

# Type checking
.mypy_cache/
.dmypy.json

# OS
.DS_Store
Thumbs.db

# Project specific
*.pt
*.pth
checkpoints/
experiments/*_results/
*.log
wandb/

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Misc
*.bak
*.orig
```

- [ ] **Step 2: Commit**

```bash
git add .gitignore
git commit -m "chore: add .gitignore with Python standard patterns"
```


### Task A.6: 创建 MANIFEST.in

**Files:**
- Create: `MANIFEST.in`

- [ ] **Step 1: 写入 MANIFEST.in**

```
include README.md
include README.zh-CN.md
include LICENSE
include CHANGELOG.md
include requirements.txt
include pyproject.toml
include setup.py

recursive-include docs *.md *.svg
recursive-include examples *.py
recursive-include tutorials *.py
recursive-include paper/figures *.pdf *.png
recursive-include scripts *.py
recursive-exclude * __pycache__
recursive-exclude * *.pyc

global-exclude .gitignore
global-exclude .DS_Store
```

- [ ] **Step 2: Commit**

```bash
git add MANIFEST.in
git commit -m "chore: add MANIFEST.in for sdist inclusion"
```


### Task A.7: 创建 CHANGELOG.md

**Files:**
- Create: `CHANGELOG.md`

- [ ] **Step 1: 写入 CHANGELOG.md**

```markdown
# Changelog

All notable changes to ASAM (Adaptive Sparse Attention Mechanism).

## [1.2.0] - 2026-04-29

### Added
- `pyproject.toml` and `setup.py` for standard Python packaging
- HuggingFace Transformers integration (`ASAMHFModel`, `ASAMHFForSequenceClassification`)
- Multi-GPU distributed training support (DDP, FSDP)
- ONNX export with accuracy verification
- Real LRA benchmark pipeline with measured (not simulated) results
- Pretrained model weights training script
- `.gitignore`, `MANIFEST.in`, `CHANGELOG.md`

### Changed
- Unified version to `1.2.0` across all files
- Resolved `FlashASAMLayer` naming conflict: `flash_asam.py` class renamed to `FlashAttnASAMLayer`
- Extracted shared utility functions to `asam/_common.py` to eliminate code duplication
- Expanded `__init__.py` exports from 10 to 18 public symbols
- Added type annotations to all public API classes
- Fixed GitHub Actions CI branch trigger from `main` to `master`

### Fixed
- Completed incomplete `FlashAttnASAMLayer.__init__` implementation in `flash_asam.py`
- Fixed CI benchmark step that used non-existent `--quick` flag

## [1.1.1] - 2026-02

### Changed
- Sparse pattern construction performance optimization (1.6-14.6x speedup)
- Hierarchical pattern GPU caching
- Clustered assignment computation via batched matmul
- OptimizedASAMLayer gate lazy computation
- EfficientASAMLayer local mask cache reuse

## [1.1.0] - 2026-01

### Added
- Flash Attention integration (`FlashASAMLayer` with 3-4.5x forward speedup)
- Mixed precision training support (additional 2x training speedup)
- `EfficientASAMLayer` and `OptimizedASAMLayer` variants
- Comprehensive performance analysis report (RTX 3060)

## [1.0.0] - 2025-12

### Added
- Initial release
- `ASAMLayer` with adaptive gating and sparse pattern selection
- Five sparse patterns: local, strided, random, clustered, hierarchical
- `AdaptiveGate` with complexity estimation and confidence prediction
- `ClusteredSparsePattern` with learnable centroids
- `HierarchicalSparsePattern` with multi-scale combination
- Long Range Arena benchmark suite
- SOTA comparison vs Transformer, Longformer, Linformer, Performer
- Comprehensive documentation (TECHNICAL.md, SURVEY.md, API.md)
```

- [ ] **Step 2: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs: add CHANGELOG.md covering v1.0.0 through v1.2.0"
```


### Task A.8: 修复 GitHub Actions CI

**Files:**
- Modify: `.github/workflows/tests.yml`

- [ ] **Step 1: 修改分支名**

将第 3 行：
```yaml
    branches: [ main, develop ]
```
改为：
```yaml
    branches: [ master ]
```

- [ ] **Step 2: 修改 benchmark job 分支条件**

将第 80 行：
```yaml
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
```
改为：
```yaml
    if: github.event_name == 'push' && github.ref == 'refs/heads/master'
```

- [ ] **Step 3: 删除不存在的 --quick 参数**

将第 97 行：
```yaml
        python benchmarks/sota_comparison.py --quick
```
改为：
```yaml
        python benchmarks/sota_comparison.py
```

- [ ] **Step 4: 增加 Python 3.12 到 matrix**

在第 9 行 `python-version` 列表末尾添加 `'3.12'`：
```yaml
        python-version: ['3.8', '3.9', '3.10', '3.11', '3.12']
```

- [ ] **Step 5: lint job 中移除 || true**

将第 71 行：
```yaml
        black --check asam/ || true
```
改为：
```yaml
        black --check asam/
```

将第 75 行：
```yaml
        isort --check-only asam/ || true
```
改为：
```yaml
        isort --check-only asam/
```

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/tests.yml
git commit -m "ci: fix branch trigger (main→master), add Python 3.12, remove --quick flag, harden lint"
```


### Task A.9: 最终验证

- [ ] **Step 1: 完整安装测试**

```bash
pip install -e .
python -c "import asam; print(asam.__version__); from asam import ASAMLayer, ASAMConfig; print('OK')"
```

Expected: `1.1.1` — 版本号会在 Plan B 执行后变为 `1.2.0`；导入成功

- [ ] **Step 2: 测试依赖安装**

```bash
pip install -e ".[dev]"
pytest tests/ -q
```

Expected: 32 passed（Plan B 后会增加到更多）

- [ ] **Step 3: 检查 git 状态**

```bash
git status
git log --oneline -10
```

Expected: 干净的工作目录，10 个以内的新 commit
