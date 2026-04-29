# Continual Operator Ablation Summary

- Dataset: `split_ag_news`
- Number of operator settings: `5`
- Seeds per setting: `2`
- Best average accuracy: `sinkhorn_topk (0.6250)`
- Lowest average forgetting: `kl_topk (-0.0312)`

## Aggregated Table

| Strategy | Routing | Transport W | Merge Usage | Relocation | Accuracy (mean±std) | Forgetting (mean±std) | BWT (mean±std) | Final Gap (mean±std) | Final Transport (mean±std) | Runs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sinkhorn_topk | sinkhorn_topk | 0.05 | 0.10 | 0.75 | 0.6250±0.1250 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.1099±0.0164 | 2 |
| kl_topk | kl_topk | 0.05 | 0.10 | 0.75 | 0.5156±0.0156 | -0.0312±0.0312 | 0.0312±0.0312 | 0.0052±0.0009 | 0.1070±0.0181 | 2 |
| no_transport | sinkhorn_topk | 0.00 | 0.10 | 0.75 | 0.6250±0.1250 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.1140±0.0176 | 2 |
| no_merge | sinkhorn_topk | 0.05 | 0.00 | 0.75 | 0.6250±0.1250 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.1099±0.0164 | 2 |
| no_relocation | sinkhorn_topk | 0.05 | 0.10 | 0.00 | 0.6250±0.1250 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.1099±0.0164 | 2 |

## Notes

- `sinkhorn_topk` is the full capacity-aware routing baseline.
- `kl_topk` removes Sinkhorn balancing while keeping sparse prototype routing.
- `no_transport` disables the transport-loss training term.
- `no_merge` disables merge events through `prototype_merge_usage_threshold=0`.
- `no_relocation` disables relocation updates through `prototype_relocation_strength=0`.