# Continual Ablation Summary

- Dataset: `split_ag_news`
- Number of strategies: `4`
- Seeds per strategy: `3`
- Best average accuracy: `no_adaptation (0.5312)`
- Lowest average forgetting: `no_adaptation (-0.0625)`

## Aggregated Table

| Strategy | Routing | Accuracy (mean?std) | Forgetting (mean?std) | BWT (mean?std) | Final Gap (mean?std) | Runs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| task_routing | task | 0.5234?0.0447 | -0.0469?0.0776 | 0.0469?0.0776 | 0.0000?0.0000 | 3 |
| no_adaptation | prototype | 0.5312?0.0418 | -0.0625?0.0920 | 0.0625?0.0920 | 0.1034?0.0040 | 3 |
| correlation | prototype | 0.5312?0.0418 | -0.0625?0.0920 | 0.0625?0.0920 | 0.1035?0.0028 | 3 |
| meta_secant | prototype | 0.5312?0.0418 | -0.0625?0.0920 | 0.0625?0.0920 | 0.1035?0.0028 | 3 |

## Notes

- `task_routing` is the explicit task-ID continual baseline.
- `no_adaptation` uses prototype routing without online hyperparameter updates.
- `correlation` uses the diagnostic controller.
- `meta_secant` uses the secant-style meta update.