# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.5391`
- Avg forgetting: `-0.0625`
- Backward transfer: `0.0625`

## Artifacts

- Plot image: `continual_ablation_correlation_seed42_plots.png`
- Raw JSON: `continual_ablation_correlation_seed42.json`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6910`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.0, 4.470348358154297e-08]`
- Stage transport loss trace: `[0.11891891364939511, 0.04826872143894434]`
- Stage merge-count trace: `[0.0, 0.0]`
- Stage routing stability trace: `[0.008211900116293691, 0.0032312217299477197]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`

## Hyperparameter Adaptation

- Adaptation steps: `1`
- Adaptation strategy: `correlation`
- Final hyperparameters: `{'prototype_prior_strength': 1.0005133754893816, 'prototype_capacity_blend': 0.4970270270635245, 'prototype_relocation_strength': 0.7529729728412349, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1}`