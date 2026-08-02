# Continual Ablation Summary

- Dataset: `split_dbpedia`
- Protocol: `class_incremental_singlehead`
- Label mode: `global`
- Head mode: `single`
- Train task-id mode: `oracle`
- Eval task-id mode: `none`
- Number of strategies: `4`
- Seeds per strategy: `3`
- Best average accuracy: `correlation (0.0726)`
- Lowest average forgetting: `correlation (0.3361)`

## Aggregated Table

| Strategy | Routing | Accuracy (mean?std) | Forgetting (mean?std) | BWT (mean?std) | Final Gap (mean?std) | Runs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| no_adaptation | prototype | 0.0655?0.0269 | 0.3639?0.0104 | -0.1944?0.0349 | 0.0473?0.0027 | 3 |
| correlation | prototype | 0.0726?0.0255 | 0.3361?0.0104 | -0.1917?0.0180 | 0.0567?0.0015 | 3 |
| dual_transport | prototype | 0.0595?0.0237 | 0.3611?0.0142 | -0.2056?0.0410 | 0.0450?0.0011 | 3 |
| meta_secant | prototype | 0.0631?0.0211 | 0.3556?0.0239 | -0.1944?0.0342 | 0.0722?0.0027 | 3 |

## Notes

- `no_adaptation` uses prototype routing without online hyperparameter updates.
- `correlation` uses the diagnostic controller.
- `dual_transport` uses a forgetting-constrained dual update on transport regularization.
- `meta_secant` uses the secant-style meta update.

## Skipped Strategies

- `task_routing`: eval_task_id_mode='none' requires prototype routing