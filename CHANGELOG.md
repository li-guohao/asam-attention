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
