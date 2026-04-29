# Continual Ablation Summary

- Dataset: `split_ag_news`
- Number of strategies: `4`
- Seeds per strategy: `5`
- Best average accuracy: `no_adaptation (0.5016)`
- Lowest average forgetting: `no_adaptation (-0.0078)`

## Aggregated Table

| Strategy | Routing | Accuracy (mean?std) | Forgetting (mean?std) | BWT (mean?std) | Final Gap (mean?std) | Runs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| task_routing | task | 0.4961?0.0215 | 0.0109?0.0348 | -0.0109?0.0348 | 0.0000?0.0000 | 5 |
| no_adaptation | prototype | 0.5016?0.0149 | -0.0078?0.0324 | 0.0078?0.0324 | 0.1089?0.0025 | 5 |
| correlation | prototype | 0.5016?0.0149 | -0.0078?0.0324 | 0.0078?0.0324 | 0.1097?0.0024 | 5 |
| meta_secant | prototype | 0.5016?0.0149 | -0.0078?0.0324 | 0.0078?0.0324 | 0.1097?0.0024 | 5 |

## Notes

- `task_routing` is the explicit task-ID continual baseline.
- `no_adaptation` uses prototype routing without online hyperparameter updates.
- `correlation` uses the diagnostic controller.
- `meta_secant` uses the secant-style meta update.