# Continual Operator Ablation Summary

- Dataset: `split_ag_news`
- Number of operator settings: `5`
- Seeds per setting: `3`
- Best average accuracy: `sinkhorn_topk (0.5286)`
- Lowest average forgetting: `kl_topk (-0.0573)`

## Aggregated Table

| Strategy | Routing | Transport W | Merge Usage | Relocation | Accuracy (mean±std) | Forgetting (mean±std) | BWT (mean±std) | Final Gap (mean±std) | Final Transport (mean±std) | Runs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sinkhorn_topk | sinkhorn_topk | 0.05 | 0.10 | 0.75 | 0.5286±0.0415 | -0.0417±0.0820 | 0.0417±0.0820 | 0.0570±0.0123 | 0.0392±0.0071 | 3 |
| kl_topk | kl_topk | 0.05 | 0.10 | 0.75 | 0.5130±0.0461 | -0.0573±0.0940 | 0.0573±0.0940 | 0.0293±0.0075 | 0.0421±0.0027 | 3 |
| no_transport | sinkhorn_topk | 0.00 | 0.10 | 0.75 | 0.5104±0.0425 | -0.0469±0.0797 | 0.0469±0.0797 | 0.0651±0.0031 | 0.0474±0.0096 | 3 |
| no_merge | sinkhorn_topk | 0.05 | 0.00 | 0.75 | 0.5286±0.0415 | -0.0521±0.0966 | 0.0521±0.0966 | 0.0580±0.0131 | 0.0383±0.0071 | 3 |
| no_relocation | sinkhorn_topk | 0.05 | 0.10 | 0.00 | 0.5286±0.0415 | -0.0417±0.0820 | 0.0417±0.0820 | 0.0366±0.0121 | 0.0377±0.0071 | 3 |

## Notes

- `sinkhorn_topk` is the full capacity-aware routing baseline.
- `kl_topk` removes Sinkhorn balancing while keeping sparse prototype routing.
- `no_transport` disables the transport-loss training term.
- `no_merge` disables merge events through `prototype_merge_usage_threshold=0`.
- `no_relocation` disables relocation updates through `prototype_relocation_strength=0`.