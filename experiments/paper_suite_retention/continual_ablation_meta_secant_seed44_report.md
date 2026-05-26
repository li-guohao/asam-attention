# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.4844`
- Avg forgetting: `-0.0156`
- Backward transfer: `0.0156`

## Artifacts

- Plot image: `continual_ablation_meta_secant_seed44_plots.png`
- Raw JSON: `continual_ablation_meta_secant_seed44.json`
- Resolved prototypes: `4`
- Prototype top-k: `1`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.0000`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.05461576581001282, 0.11445216834545135]`
- Stage transport loss trace: `[0.09822813048958778, 0.031205360079184175]`
- Stage merge-count trace: `[0.0, 1.0]`
- Stage routing stability trace: `[0.7416652627289295, 0.4681613575667143]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`

## Hyperparameter Adaptation

- Adaptation steps: `1`
- Adaptation strategy: `meta_secant`
- Final hyperparameters: `{'prototype_prior_strength': 1.047445223578705, 'prototype_capacity_blend': 0.4961789026390761, 'prototype_relocation_strength': 0.7551864915527403, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1}`