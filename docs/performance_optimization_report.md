# ASAM Performance Optimization Report

## Overview

This report summarizes the targeted performance work completed in the local workspace for ASAM.
The goal was to improve the highest-value hot paths without changing the public API or destabilizing existing tests.

All results below were measured in the current environment during this session and should be treated as machine-specific reference numbers.

## Scope

The following modules were optimized:

- `asam/asam_layer_optimized.py`
- `asam/efficient_attention.py`
- `asam/sparse_patterns.py`
- `benchmarks/profile_patterns.py`

The following attempted optimization was intentionally **not** kept:

- Hard-routing changes for the main `ASAMLayer` path, because they produced regressions on the current GPU during validation.

## Implemented Optimizations

### 1. `OptimizedASAMLayer`

- Fixed `AdaptiveGate` initialization to use the real `num_heads` instead of the default head count.
- Avoided computing gate outputs unless `return_info=True`.
- Added a valid local-window mask so boundary tokens no longer attend to padded zero values.
- Added a fallback path for external attention masks.

### 2. `EfficientASAMLayer`

- Added a reusable local attention mask cache keyed by device and sequence length.
- Corrected `sparse_ratio` estimation to reflect the actual effective local window.

### 3. Sparse pattern construction

- Vectorized `LocalSparsePattern.build_pattern()`.
- Vectorized `StridedSparsePattern.build_pattern()`.
- Reworked `RandomSparsePattern.build_pattern()` to:
  - avoid mutating global RNG state,
  - support explicit seeding,
  - generate row-wise random connections in a vectorized way.
- Added per-device pattern caching to avoid repeated `.to(device)` costs.

### 4. `HierarchicalSparsePattern`

- Added cached expanded pattern stacks per device.
- Replaced repeated `expand + stack + float()` work inside `combine_patterns()` with a cached stack plus `einsum`.

### 5. `ClusteredSparsePattern`

- Reworked `compute_cluster_assignment()` to use batched matrix multiplication instead of `einsum`.
- Kept `apply_cluster_mask()` on `einsum`, because that remained faster on the current machine.

## Validation

### Test status

- Final test result: `32 passed`
- Command used:

```bash
python -m pytest tests -q
```

### Added regression coverage

New or expanded tests cover:

- gate shape correctness,
- lazy gate computation,
- local attention boundary stability,
- efficient local mask cache reuse,
- external attention mask support,
- strided pattern semantics,
- random pattern determinism,
- random pattern global RNG isolation,
- hierarchical cache population,
- hierarchical weight update correctness,
- clustered assignment equivalence to the reference formula.

## Benchmark Summary

### Layer-level improvements

| Component | Before | After | Speedup |
|---|---:|---:|---:|
| `OptimizedASAMLayer` | 32.844 ms | 24.585 ms | **1.34x** |
| `EfficientASAMLayer` | 12.194 ms | 11.295 ms | **1.08x** |

### Pattern construction improvements

| Pattern | Before | After | Speedup |
|---|---:|---:|---:|
| `LocalSparsePattern.build_pattern()` | 28.552 ms | 17.809 ms | **1.60x** |
| `StridedSparsePattern.build_pattern()` | 82.093 ms | 18.100 ms | **4.54x** |
| `RandomSparsePattern.build_pattern()` | 540.760 ms | 239.462 ms | **2.26x** |

### Hierarchical pattern improvements

| Operation | Before | After | Speedup |
|---|---:|---:|---:|
| `HierarchicalSparsePattern.combine_patterns()` on CPU | 774.748 ms | 158.467 ms | **4.89x** |
| `HierarchicalSparsePattern.combine_patterns()` on CUDA | 43.055 ms | 2.938 ms | **14.65x** |

### Clustered pattern improvements

| Operation | Before | After | Speedup |
|---|---:|---:|---:|
| `compute_cluster_assignment()` | 2.132 ms | 1.349 ms | **1.58x** |
| Full clustered path (`assignment + mask`) | 2.540 ms | 2.252 ms | **1.13x** |

## Pattern Profiling Snapshot (`seq_len=2048`)

The new profiling script was run with:

```bash
python benchmarks/profile_patterns.py --seq-len 2048 --devices auto --json-out benchmarks/pattern_profile.json
```

### CPU

| Pattern | Build | First get | Cached get | Effective | Memory | Sparsity |
|---|---:|---:|---:|---:|---:|---:|
| `local` | 8.962 ms | 8.566 ms | 0.002 ms | - | 4.000 MB | 93.80% |
| `strided` | 9.237 ms | 9.502 ms | 0.001 ms | - | 4.000 MB | 95.32% |
| `random` | 291.851 ms | 292.074 ms | 0.002 ms | - | 32.000 MB | 96.88% |
| `hierarchical` | 9.572 ms | 9.468 ms | 0.002 ms | 309.958 ms | 32.000 MB | 93.12% |

### CUDA

| Pattern | Build | First get | Cached get | Effective | Memory | Sparsity |
|---|---:|---:|---:|---:|---:|---:|
| `local` | 9.074 ms | 40.234 ms | 0.002 ms | - | 4.000 MB | 93.80% |
| `strided` | 8.778 ms | 8.930 ms | 0.004 ms | - | 4.000 MB | 95.32% |
| `random` | 295.228 ms | 298.333 ms | 0.004 ms | - | 32.000 MB | 96.88% |
| `hierarchical` | 10.193 ms | 10.721 ms | 0.004 ms | 11.107 ms | 32.000 MB | 93.12% |

### Notes

- Device caching is working as intended: cached gets drop to near-zero cost.
- `RandomSparsePattern` remains the heaviest static pattern builder.
- `HierarchicalSparsePattern` benefits strongly from caching and GPU-side reuse.
- The first CUDA access for some patterns is dominated by the initial host-to-device transfer.

## Output Artifacts

- Profiling script: `benchmarks/profile_patterns.py`
- Profiling JSON: `benchmarks/pattern_profile.json`

## Recommended Next Steps

1. Add an optional benchmark mode for pattern memory scaling across multiple sequence lengths.
2. Consider precomputing or persisting random patterns for fixed experiment settings.
3. Investigate whether hierarchical pattern stacks should be invalidated automatically if internal pattern modules change.
4. Benchmark mixed precision behavior for clustered and hierarchical paths separately.

## Conclusion

The largest wins in this round came from:

- removing unnecessary work on the optimized attention path,
- vectorizing sparse pattern construction,
- caching per-device sparse masks and hierarchical stacks,
- tightening clustered assignment computation.

These changes improve both one-time setup costs and repeated runtime access, while keeping the repository fully green under the current test suite.
