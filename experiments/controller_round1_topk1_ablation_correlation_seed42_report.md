# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.4766`
- Avg forgetting: `0.0312`
- Backward transfer: `-0.0312`

## Artifacts

- Plot image: `controller_round1_topk1_ablation_correlation_seed42_plots.png`
- Raw JSON: `controller_round1_topk1_ablation_correlation_seed42.json`
- Resolved prototypes: `4`
- Prototype top-k: `1`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.0000`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.03125]`
- Stage transport gap trace: `[0.09602127224206924, 0.10159338265657425]`
- Stage transport loss trace: `[0.1180221107788384, 0.046828396152704954]`
- Stage merge-count trace: `[1.0, 1.0]`
- Stage routing stability trace: `[0.3362783221527934, 0.1806830505374819]`
- Forgetting vs routing stability correlation: `-0.9999999999999999`
- Forgetting vs transport gap correlation: `1.0`
- Forgetting vs transport loss correlation: `-1.0`
- Forgetting vs mean abs excess correlation: `1.0`
- Forgetting vs merge-count correlation: `None`

## Hyperparameter Adaptation

- Adaptation steps: `1`
- Adaptation strategy: `correlation`
- Final hyperparameters: `{'prototype_prior_strength': 1.0212398160870289, 'prototype_capacity_blend': 0.49464891547104345, 'prototype_relocation_strength': 0.7577516163815744, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'transport_weight': 0.05}`