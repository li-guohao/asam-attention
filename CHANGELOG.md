# Changelog

All notable changes to ASAM (Adaptive Sparse Attention Mechanism).

## [1.2.1] - 2026-08-02

### Fixed
- Paper reproducibility: reran the BPE strategy table (`r2_agnews_bpe_3ep.json`,
  3 epochs, 3 seeds) and the EWC/SI/MAS baseline comparison
  (`r2_baseline_comparison.json`, unified task-incremental multi-head protocol);
  the paper tables and abstract now match the stored artifacts.
- Theory section: Theorem 1 is now a valid $\epsilon$-consistency statement;
  the unproven forgetting corollary was replaced with an explicitly informal
  surrogate-control interpretation.
- `datasets/text_dataset.py`: synthetic dataset fallback is disabled by default;
  set `ASAM_ALLOW_DATASET_FALLBACK=1` to allow it. Datasets now expose a
  `data_source` attribute (`huggingface` / `synthetic_fallback`).
- `datasets/text_dataset.py`: `_fetch_raw_texts` now uses the `text` field when
  `title`/`description` are absent, so BPE tokenizer training uses real
  headlines with current AG News cache formats.
- Added `paper/references.bib` and a table-vs-artifact consistency test.
- README release link updated to `v1.2.0`.

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
