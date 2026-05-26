# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.5000`
- Avg forgetting: `0.0156`
- Backward transfer: `-0.0156`

## Artifacts

- Plot image: `frontier_round1_ablation_correlation_seed44_plots.png`
- Raw JSON: `frontier_round1_ablation_correlation_seed44.json`
- Resolved prototypes: `4`
- Prototype top-k: `1`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.0000`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.015625]`
- Stage transport gap trace: `[0.10723854601383209, 0.1090092658996582]`
- Stage transport loss trace: `[0.07364481489639729, 0.046175718249287456]`
- Stage merge-count trace: `[1.0, 1.0]`
- Stage routing stability trace: `[0.19165249972138554, 0.08434180560288951]`
- Forgetting vs routing stability correlation: `-0.9999999999999998`
- Forgetting vs transport gap correlation: `1.0`
- Forgetting vs transport loss correlation: `-0.9999999999999998`
- Forgetting vs mean abs excess correlation: `1.0`
- Forgetting vs merge-count correlation: `None`

## Hyperparameter Adaptation

- Adaptation steps: `1`
- Adaptation strategy: `correlation`
- Final hyperparameters: `{'prototype_prior_strength': 1.0120503081421284, 'prototype_capacity_blend': 0.49547791611694264, 'prototype_relocation_strength': 0.7572030476731015, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1}`