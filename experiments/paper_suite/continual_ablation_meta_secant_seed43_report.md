# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.4688`
- Avg forgetting: `0.0000`
- Backward transfer: `0.0000`

## Artifacts

- Plot image: `continual_ablation_meta_secant_seed43_plots.png`
- Raw JSON: `continual_ablation_meta_secant_seed43.json`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6670`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.0, 1.4901161193847656e-08]`
- Stage transport loss trace: `[0.2937461622059345, 0.054877555929124355]`
- Stage merge-count trace: `[0.0, 0.0]`
- Stage routing stability trace: `[0.03158281370997429, 0.05145367980003357]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`

## Hyperparameter Adaptation

- Adaptation steps: `1`
- Adaptation strategy: `meta_secant`
- Final hyperparameters: `{'prototype_prior_strength': 1.0019758753310124, 'prototype_capacity_blend': 0.49265634594485164, 'prototype_relocation_strength': 0.7573436540551484, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1}`