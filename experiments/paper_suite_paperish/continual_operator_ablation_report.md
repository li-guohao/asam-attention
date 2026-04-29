# Continual Operator Ablation Summary

- Dataset: `split_ag_news`
- Number of operator settings: `5`
- Seeds per setting: `3`
- Best average accuracy: `sinkhorn_topk (0.4974)`
- Lowest average forgetting: `sinkhorn_topk (0.0469)`

## Aggregated Table

| Strategy | Routing | Transport W | Merge Usage | Relocation | Accuracy (mean±std) | Forgetting (mean±std) | BWT (mean±std) | Final Gap (mean±std) | Final Transport (mean±std) | Runs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sinkhorn_topk | sinkhorn_topk | 0.05 | 0.10 | 0.75 | 0.4974±0.0295 | 0.0469±0.0837 | -0.0469±0.0837 | 0.0000±0.0000 | 0.0381±0.0076 | 3 |
| kl_topk | kl_topk | 0.05 | 0.10 | 0.75 | 0.4974±0.0295 | 0.0469±0.0837 | -0.0469±0.0837 | 0.0008±0.0003 | 0.0370±0.0071 | 3 |
| no_transport | sinkhorn_topk | 0.00 | 0.10 | 0.75 | 0.4948±0.0258 | 0.0469±0.0837 | -0.0469±0.0837 | 0.0000±0.0000 | 0.0418±0.0098 | 3 |
| no_merge | sinkhorn_topk | 0.05 | 0.00 | 0.75 | 0.4974±0.0295 | 0.0469±0.0837 | -0.0469±0.0837 | 0.0000±0.0000 | 0.0381±0.0076 | 3 |
| no_relocation | sinkhorn_topk | 0.05 | 0.10 | 0.00 | 0.4974±0.0295 | 0.0469±0.0837 | -0.0469±0.0837 | 0.0000±0.0000 | 0.0381±0.0076 | 3 |

## Notes

- `sinkhorn_topk` is the full capacity-aware routing baseline.
- `kl_topk` removes Sinkhorn balancing while keeping sparse prototype routing.
- `no_transport` disables the transport-loss training term.
- `no_merge` disables merge events through `prototype_merge_usage_threshold=0`.
- `no_relocation` disables relocation updates through `prototype_relocation_strength=0`.