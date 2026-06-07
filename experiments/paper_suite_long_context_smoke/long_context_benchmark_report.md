# Long-Context ASAM Paper Suite

Diagnostic only: this is an LRA-style synthetic operator/runtime sweep, not an official LRA result.

## Config

- Sequence lengths: `[32, 64, 128]`
- Models: `['asam', 'transformer', 'local', 'longformer_style']`
- Device: `cpu`
- Batch size: `1`
- Width / heads: `32` / `2`

## Results

| Model | Seq Len | Success | Latency ms | Memory MB | Finite Rate |
| --- | ---: | --- | ---: | ---: | ---: |
| `asam` | 32 | True | 0.4533 | 0.0000 | 1.0000 |
| `transformer` | 32 | True | 0.1383 | 0.0000 | 1.0000 |
| `local` | 32 | True | 0.1382 | 0.0000 | 1.0000 |
| `longformer_style` | 32 | True | 0.1277 | 0.0000 | 1.0000 |
| `asam` | 64 | True | 0.5491 | 0.0000 | 1.0000 |
| `transformer` | 64 | True | 0.2695 | 0.0000 | 1.0000 |
| `local` | 64 | True | 0.1467 | 0.0000 | 1.0000 |
| `longformer_style` | 64 | True | 0.1459 | 0.0000 | 1.0000 |
| `asam` | 128 | True | 0.7178 | 0.0000 | 1.0000 |
| `transformer` | 128 | True | 0.1537 | 0.0000 | 1.0000 |
| `local` | 128 | True | 0.2024 | 0.0000 | 1.0000 |
| `longformer_style` | 128 | True | 0.1888 | 0.0000 | 1.0000 |

## Claim Boundary

- These numbers validate the benchmark harness, artifact provenance, and operator behavior.
- They must not be reported as official Long Range Arena results or hardware speedup claims.
- Successful rows: `12/12`.