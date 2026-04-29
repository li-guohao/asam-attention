# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.5312`
- Avg forgetting: `0.0000`
- Backward transfer: `0.0000`

## Artifacts

- Plot image: `continual_ablation_correlation_seed42_plots.png`
- Raw JSON: `continual_ablation_correlation_seed42.json`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6876`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[7.450580596923828e-09, 1.4901161193847656e-08]`
- Stage transport loss trace: `[0.3015690827742219, 0.07350195664912462]`
- Stage merge-count trace: `[0.0, 0.0]`
- Stage routing stability trace: `[0.009641786338761449, 0.013019013917073607]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`

## Hyperparameter Adaptation

- Adaptation steps: `1`
- Adaptation strategy: `correlation`
- Final hyperparameters: `{'prototype_prior_strength': 1.0006027932530484, 'prototype_capacity_blend': 0.4924607726952061, 'prototype_relocation_strength': 0.7575392274418846, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1}`