# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.5391`
- Avg forgetting: `-0.0312`
- Backward transfer: `0.0312`

## Artifacts

- Plot image: `continual_ablation_correlation_seed44_plots.png`
- Raw JSON: `continual_ablation_correlation_seed44.json`
- Resolved prototypes: `4`
- Prototype top-k: `1`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.0000`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.054615747183561325, 0.10751543194055557]`
- Stage transport loss trace: `[0.10144962242338806, 0.041393549530766904]`
- Stage merge-count trace: `[1.0, 1.0]`
- Stage routing stability trace: `[0.7236646860837936, 0.389402287080884]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`

## Hyperparameter Adaptation

- Adaptation steps: `1`
- Adaptation strategy: `correlation`
- Final hyperparameters: `{'prototype_prior_strength': 1.0462674725822996, 'prototype_capacity_blend': 0.49609836575982624, 'prototype_relocation_strength': 0.7552670279197627, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1}`