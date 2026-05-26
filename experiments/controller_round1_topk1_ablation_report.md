# Continual Ablation Summary

- Dataset: `split_ag_news`
- Number of strategies: `4`
- Seeds per strategy: `3`
- Best average accuracy: `task_routing (0.5234)`
- Lowest average forgetting: `no_adaptation (-0.0573)`

## Aggregated Table

| Strategy | Routing | Accuracy (mean?std) | Forgetting (mean?std) | BWT (mean?std) | Final Gap (mean?std) | Runs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| task_routing | task | 0.5234?0.0447 | -0.0469?0.0776 | 0.0469?0.0776 | 0.0000?0.0000 | 3 |
| no_adaptation | prototype | 0.5130?0.0461 | -0.0573?0.0940 | 0.0573?0.0940 | 0.1051?0.0063 | 3 |
| correlation | prototype | 0.5130?0.0461 | -0.0573?0.0940 | 0.0573?0.0940 | 0.1058?0.0061 | 3 |
| meta_secant | prototype | 0.5130?0.0461 | -0.0573?0.0940 | 0.0573?0.0940 | 0.1072?0.0056 | 3 |

## Notes

- `task_routing` is the explicit task-ID continual baseline.
- `no_adaptation` uses prototype routing without online hyperparameter updates.
- `correlation` uses the diagnostic controller.
- `meta_secant` uses the secant-style meta update.