# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.4766`
- Avg forgetting: `0.1406`
- Backward transfer: `-0.1406`

## Artifacts

- Plot image: `continual_ablation_correlation_seed44_plots.png`
- Raw JSON: `continual_ablation_correlation_seed44.json`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6900`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.140625]`
- Stage transport gap trace: `[0.0, 0.0]`
- Stage transport loss trace: `[0.09771448688115925, 0.030071596731431782]`
- Stage merge-count trace: `[0.0, 0.0]`
- Stage routing stability trace: `[0.015999344846932217, 0.0044593132406589575]`
- Forgetting vs routing stability correlation: `-1.0`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `-1.0`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`

## Hyperparameter Adaptation

- Adaptation steps: `1`
- Adaptation strategy: `correlation`
- Final hyperparameters: `{'prototype_prior_strength': 1.001000459178675, 'prototype_capacity_blend': 0.4975571376062582, 'prototype_relocation_strength': 0.7524428621720289, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1}`