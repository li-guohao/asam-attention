# Continual Operator Ablation Summary

- Dataset: `split_ag_news`
- Number of operator settings: `7`
- Seeds per setting: `1`
- Best average accuracy: `sinkhorn_topk (0.5000)`
- Lowest average forgetting: `sinkhorn_topk (0.2500)`

## Aggregated Table

| Strategy | Routing | Transport W | Merge Usage | Relocation | Accuracy (mean±std) | Forgetting (mean±std) | BWT (mean±std) | Final Gap (mean±std) | Final Transport (mean±std) | Final Candidate Residual (mean±std) | Final Support Residual (mean±std) | Final Delta (mean±std) | Final Density (mean±std) | Runs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sinkhorn_topk | sinkhorn_topk | 0.05 | 0.10 | 0.75 | 0.5000±0.0000 | 0.2500±0.0000 | -0.2500±0.0000 | 0.0236±0.0000 | 0.0740±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 1 |
| kl_topk | kl_topk | 0.05 | 0.10 | 0.75 | 0.5000±0.0000 | 0.2500±0.0000 | -0.2500±0.0000 | 0.0149±0.0000 | 0.0503±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 1 |
| masked_sinkhorn_topk | masked_sinkhorn_topk | 0.05 | 0.10 | 0.75 | 0.5000±0.0000 | 0.2500±0.0000 | -0.2500±0.0000 | 0.0000±0.0000 | 0.0606±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 1 |
| sinkhorn_support_masked | sinkhorn_support_masked | 0.05 | 0.10 | 0.75 | 0.5000±0.0000 | 0.2500±0.0000 | -0.2500±0.0000 | 0.0000±0.0000 | 0.0740±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 1 |
| no_transport | sinkhorn_topk | 0.00 | 0.10 | 0.75 | 0.5000±0.0000 | 0.2500±0.0000 | -0.2500±0.0000 | 0.0236±0.0000 | 0.0704±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 1 |
| no_merge | sinkhorn_topk | 0.05 | 0.00 | 0.75 | 0.5000±0.0000 | 0.2500±0.0000 | -0.2500±0.0000 | 0.0174±0.0000 | 0.0912±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 1 |
| no_relocation | sinkhorn_topk | 0.05 | 0.10 | 0.00 | 0.5000±0.0000 | 0.2500±0.0000 | -0.2500±0.0000 | 0.0236±0.0000 | 0.0674±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 1 |

## Notes

- `sinkhorn_topk` is the full capacity-aware routing baseline.
- `kl_topk` removes Sinkhorn balancing while keeping sparse prototype routing.
- `masked_sinkhorn_topk` runs Sinkhorn directly on the sparse top-k support.
- `sinkhorn_support_masked` uses dense Sinkhorn to select top-k support, then masked Sinkhorn reroutes on that support.
- `no_transport` disables the transport-loss training term.
- `no_merge` disables merge events through `prototype_merge_usage_threshold=0`.
- `no_relocation` disables relocation updates through `prototype_relocation_strength=0`.