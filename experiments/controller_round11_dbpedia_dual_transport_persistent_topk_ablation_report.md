# Continual Ablation Summary

- Dataset: `split_dbpedia`
- Number of strategies: `5`
- Seeds per strategy: `3`
- Best average accuracy: `task_routing (0.5238)`
- Lowest average forgetting: `correlation (0.0500)`

## Aggregated Table

| Strategy | Routing | Accuracy (mean?std) | Forgetting (mean?std) | BWT (mean?std) | Final Gap (mean?std) | Runs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| task_routing | task | 0.5238?0.0178 | 0.0500?0.0272 | 0.0111?0.0208 | 0.0000?0.0000 | 3 |
| no_adaptation | prototype | 0.5190?0.0294 | 0.0556?0.0342 | -0.0000?0.0136 | 0.0285?0.0003 | 3 |
| correlation | prototype | 0.5190?0.0294 | 0.0500?0.0360 | 0.0000?0.0136 | 0.0463?0.0005 | 3 |
| dual_transport | prototype | 0.5190?0.0294 | 0.0556?0.0342 | -0.0000?0.0136 | 0.0285?0.0002 | 3 |
| meta_secant | prototype | 0.5000?0.0117 | 0.0833?0.0136 | -0.0222?0.0342 | 0.0417?0.0034 | 3 |

## Notes

- `task_routing` is the explicit task-ID continual baseline.
- `no_adaptation` uses prototype routing without online hyperparameter updates.
- `correlation` uses the diagnostic controller.
- `dual_transport` uses a forgetting-constrained dual update on transport regularization.
- `meta_secant` uses the secant-style meta update.