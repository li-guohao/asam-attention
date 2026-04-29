# Continual Ablation Summary

- Dataset: `split_ag_news`
- Number of strategies: `4`
- Seeds per strategy: `2`
- Best average accuracy: `task_routing (0.6250)`
- Lowest average forgetting: `task_routing (-0.2188)`

## Aggregated Table

| Strategy | Routing | Accuracy (mean?std) | Forgetting (mean?std) | BWT (mean?std) | Final Gap (mean?std) | Runs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| task_routing | task | 0.6250?0.1250 | -0.2188?0.2188 | 0.2188?0.2188 | 0.0000?0.0000 | 2 |
| no_adaptation | prototype | 0.6250?0.1250 | 0.0000?0.0000 | 0.0000?0.0000 | 0.0000?0.0000 | 2 |
| correlation | prototype | 0.6250?0.1250 | 0.0000?0.0000 | 0.0000?0.0000 | 0.0000?0.0000 | 2 |
| meta_secant | prototype | 0.6250?0.1250 | 0.0000?0.0000 | 0.0000?0.0000 | 0.0000?0.0000 | 2 |

## Notes

- `task_routing` is the explicit task-ID continual baseline.
- `no_adaptation` uses prototype routing without online hyperparameter updates.
- `correlation` uses the diagnostic controller.
- `meta_secant` uses the secant-style meta update.