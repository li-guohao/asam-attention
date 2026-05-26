# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.4766`
- Avg forgetting: `0.0312`
- Backward transfer: `-0.0312`

## Artifacts

- Plot image: `capacity_round1_ablation_correlation_seed42_plots.png`
- Raw JSON: `capacity_round1_ablation_correlation_seed42.json`
- Resolved prototypes: `4`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6880`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.03125]`
- Stage transport gap trace: `[0.05441046133637428, 0.0655251294374466]`
- Stage transport loss trace: `[0.11797323008067906, 0.04767694161273539]`
- Stage merge-count trace: `[0.0, 0.0]`
- Stage routing stability trace: `[0.26707378309220076, 0.4907774683088064]`
- Forgetting vs routing stability correlation: `1.0`
- Forgetting vs transport gap correlation: `1.0`
- Forgetting vs transport loss correlation: `-0.9999999999999999`
- Forgetting vs mean abs excess correlation: `1.0`
- Forgetting vs merge-count correlation: `None`

## Hyperparameter Adaptation

- Adaptation steps: `1`
- Adaptation strategy: `correlation`
- Final hyperparameters: `{'prototype_prior_strength': 1.0168322031253405, 'prototype_capacity_blend': 0.4956904077145737, 'prototype_relocation_strength': 0.7556698538188357, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1}`