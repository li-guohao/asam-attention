# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows a tag-based release workflow.

## [v1.1.1] - 2026-04-14

Release: [ASAM v1.1.1 - Sparse Path Optimizations and Profiling](https://github.com/li-guohao/asam-attention/releases/tag/v1.1.1)

### Added

- Added `benchmarks/profile_patterns.py` for profiling sparse pattern build time, cache reuse, effective combine cost, memory footprint, and sparsity.
- Added `docs/performance_optimization_report.md` summarizing the implemented optimizations and measured gains.
- Added regression coverage for lazy gate execution, local mask caching, random pattern determinism, hierarchical cache reuse, and clustered assignment equivalence.

### Changed

- Optimized `OptimizedASAMLayer` by fixing `AdaptiveGate` head configuration, skipping unnecessary gate work unless debug info is requested, and improving boundary handling for local attention.
- Improved `EfficientASAMLayer` with reusable local attention mask caching and corrected sparse ratio estimation.
- Vectorized `LocalSparsePattern`, `StridedSparsePattern`, and `RandomSparsePattern` construction.
- Added per-device sparse pattern caching to reduce repeated device transfer overhead.
- Accelerated `HierarchicalSparsePattern.combine_patterns()` using cached pattern stacks.
- Optimized `ClusteredSparsePattern.compute_cluster_assignment()` with batched matrix multiplication.

### Performance

- `OptimizedASAMLayer`: `32.84 ms -> 24.58 ms` (`1.34x`)
- `EfficientASAMLayer`: `12.19 ms -> 11.30 ms` (`1.08x`)
- `LocalSparsePattern.build_pattern()`: `28.55 ms -> 17.81 ms` (`1.60x`)
- `StridedSparsePattern.build_pattern()`: `82.09 ms -> 18.10 ms` (`4.54x`)
- `RandomSparsePattern.build_pattern()`: `540.76 ms -> 239.46 ms` (`2.26x`)
- `HierarchicalSparsePattern.combine_patterns()` on CPU: `774.75 ms -> 158.47 ms` (`4.89x`)
- `HierarchicalSparsePattern.combine_patterns()` on CUDA: `43.06 ms -> 2.94 ms` (`14.65x`)
- Clustered full path: `2.540 ms -> 2.252 ms` (`1.13x`)

### Validation

- Test suite result: `32 passed`

## [v1.1.0] - 2026-02-01

Release: [ASAM v1.1.0 - Flash Attention Optimization](https://github.com/li-guohao/asam-attention/releases/tag/v1.1.0)

### Added

- Added `FlashASAMLayer` for hardware-optimized attention execution.
- Added `EfficientASAMLayer` for memory-efficient attention computation.
- Added mixed precision training support.
- Added benchmark and analysis tooling for RTX 3060 evaluation.

### Performance

- Up to `4.5x` faster forward pass.
- Up to `2.0x` training speedup with mixed precision.
- Up to `5.45x` combined speedup at `1024` tokens.
- Up to `50-75%` memory savings on RTX 3060.

[v1.1.1]: https://github.com/li-guohao/asam-attention/releases/tag/v1.1.1
[v1.1.0]: https://github.com/li-guohao/asam-attention/releases/tag/v1.1.0
