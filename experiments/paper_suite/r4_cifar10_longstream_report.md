# Continual Ablation Summary

- Dataset: `split_cifar10`
- Protocol: `class_incremental_singlehead`
- Label mode: `global`
- Head mode: `single`
- Train task-id mode: `oracle`
- Eval task-id mode: `none`
- Number of strategies: `4`
- Seeds per strategy: `3`
- Best average accuracy: `no_adaptation (0.1111)`
- Lowest average forgetting: `no_adaptation (0.5192)`

## Aggregated Table

| Strategy | Routing | Accuracy (mean?std) | Forgetting (mean?std) | BWT (mean?std) | Final Gap (mean?std) | Runs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| no_adaptation | prototype | 0.1111?0.0157 | 0.5192?0.0157 | -0.5192?0.0157 | 0.0672?0.0020 | 3 |
| correlation | prototype | 0.1056?0.0039 | 0.5224?0.0198 | -0.5224?0.0198 | 0.0754?0.0066 | 3 |
| dual_transport | prototype | 0.1111?0.0039 | 0.5224?0.0198 | -0.5224?0.0198 | 0.0596?0.0015 | 3 |
| meta_secant | prototype | 0.1056?0.0039 | 0.5192?0.0157 | -0.5192?0.0157 | 0.0964?0.0060 | 3 |

## Notes

- `no_adaptation` uses prototype routing without online hyperparameter updates.
- `correlation` uses the diagnostic controller.
- `dual_transport` uses a forgetting-constrained dual update on transport regularization.
- `meta_secant` uses the secant-style meta update.

## Skipped Strategies

- `task_routing`: eval_task_id_mode='none' requires prototype routing