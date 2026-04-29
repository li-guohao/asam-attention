# Continual Ablation Summary

- Dataset: `split_arxiv`
- Number of strategies: `4`
- Seeds per strategy: `2`
- Best average accuracy: `no_adaptation (0.5938)`
- Lowest average forgetting: `task_routing (0.0000)`

## Aggregated Table

| Strategy | Routing | Accuracy (mean?std) | Forgetting (mean?std) | BWT (mean?std) | Final Gap (mean?std) | Runs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| task_routing | task | 0.5000?0.0000 | 0.0000?0.0000 | 0.0000?0.0000 | 0.0000?0.0000 | 2 |
| no_adaptation | prototype | 0.5938?0.0312 | 0.0833?0.0833 | 0.0000?0.1667 | 0.0368?0.0028 | 2 |
| correlation | prototype | 0.5938?0.0312 | 0.0833?0.0833 | 0.0000?0.1667 | 0.0354?0.0023 | 2 |
| meta_secant | prototype | 0.5938?0.0312 | 0.0833?0.0833 | 0.0000?0.1667 | 0.0426?0.0071 | 2 |

## Notes

- `task_routing` is the explicit task-ID continual baseline.
- `no_adaptation` uses prototype routing without online hyperparameter updates.
- `correlation` uses the diagnostic controller.
- `meta_secant` uses the secant-style meta update.