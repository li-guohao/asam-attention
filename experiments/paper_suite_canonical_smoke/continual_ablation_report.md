# Continual Ablation Summary

- Dataset: `split_ag_news`
- Number of strategies: `5`
- Seeds per strategy: `1`
- Best average accuracy: `task_routing (0.6250)`
- Lowest average forgetting: `task_routing (-0.2500)`

## Aggregated Table

| Strategy | Routing | Accuracy (mean?std) | Forgetting (mean?std) | BWT (mean?std) | Final Gap (mean?std) | Runs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| task_routing | task | 0.6250?0.0000 | -0.2500?0.0000 | 0.2500?0.0000 | 0.0000?0.0000 | 1 |
| no_adaptation | prototype | 0.5000?0.0000 | 0.2500?0.0000 | -0.2500?0.0000 | 0.0236?0.0000 | 1 |
| correlation | prototype | 0.5000?0.0000 | 0.2500?0.0000 | -0.2500?0.0000 | 0.0240?0.0000 | 1 |
| dual_transport | prototype | 0.5000?0.0000 | 0.2500?0.0000 | -0.2500?0.0000 | 0.0236?0.0000 | 1 |
| meta_secant | prototype | 0.5000?0.0000 | 0.2500?0.0000 | -0.2500?0.0000 | 0.0240?0.0000 | 1 |

## Notes

- `task_routing` is the explicit task-ID continual baseline.
- `no_adaptation` uses prototype routing without online hyperparameter updates.
- `correlation` uses the diagnostic controller.
- `dual_transport` uses a forgetting-constrained dual update on transport regularization.
- `meta_secant` uses the secant-style meta update.