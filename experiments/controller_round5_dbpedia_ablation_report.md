# Continual Ablation Summary

- Dataset: `split_dbpedia`
- Number of strategies: `4`
- Seeds per strategy: `2`
- Best average accuracy: `no_adaptation (0.5357)`
- Lowest average forgetting: `task_routing (0.0333)`

## Aggregated Table

| Strategy | Routing | Accuracy (mean?std) | Forgetting (mean?std) | BWT (mean?std) | Final Gap (mean?std) | Runs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| task_routing | task | 0.5357?0.0071 | 0.0333?0.0167 | 0.0250?0.0083 | 0.0000?0.0000 | 2 |
| no_adaptation | prototype | 0.5357?0.0214 | 0.0583?0.0417 | 0.0000?0.0167 | 0.0283?0.0002 | 2 |
| correlation | prototype | 0.5357?0.0214 | 0.0417?0.0417 | 0.0000?0.0167 | 0.0466?0.0002 | 2 |
| meta_secant | prototype | 0.5071?0.0071 | 0.0833?0.0167 | -0.0250?0.0083 | 0.0480?0.0065 | 2 |

## Notes

- `task_routing` is the explicit task-ID continual baseline.
- `no_adaptation` uses prototype routing without online hyperparameter updates.
- `correlation` uses the diagnostic controller.
- `meta_secant` uses the secant-style meta update.