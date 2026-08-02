# Continual Ablation Summary

- Dataset: `split_ag_news`
- Protocol: `task_incremental_multihead`
- Label mode: `local`
- Head mode: `multi`
- Train task-id mode: `oracle`
- Eval task-id mode: `oracle`
- Number of strategies: `5`
- Seeds per strategy: `3`
- Best average accuracy: `task_routing (0.5312)`
- Lowest average forgetting: `no_adaptation (0.0000)`

## Aggregated Table

| Strategy | Routing | Accuracy (mean?std) | Forgetting (mean?std) | BWT (mean?std) | Final Gap (mean?std) | Runs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| task_routing | task | 0.5312?0.0255 | 0.0417?0.0295 | -0.0417?0.0295 | 0.0000?0.0000 | 3 |
| no_adaptation | prototype | 0.5104?0.0147 | 0.0000?0.0510 | 0.0000?0.0510 | 0.0477?0.0047 | 3 |
| correlation | prototype | 0.5104?0.0147 | 0.0000?0.0510 | 0.0000?0.0510 | 0.0477?0.0045 | 3 |
| dual_transport | prototype | 0.5104?0.0147 | 0.0000?0.0510 | 0.0000?0.0510 | 0.0477?0.0047 | 3 |
| meta_secant | prototype | 0.5104?0.0147 | 0.0000?0.0510 | 0.0000?0.0510 | 0.0470?0.0037 | 3 |

## Notes

- `task_routing` is the explicit task-ID continual baseline.
- `no_adaptation` uses prototype routing without online hyperparameter updates.
- `correlation` uses the diagnostic controller.
- `dual_transport` uses a forgetting-constrained dual update on transport regularization.
- `meta_secant` uses the secant-style meta update.