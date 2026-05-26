# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.5781`
- Avg forgetting: `-0.1562`
- Backward transfer: `0.1562`

## Artifacts

- Plot image: `continual_ablation_correlation_seed43_plots.png`
- Raw JSON: `continual_ablation_correlation_seed43.json`
- Resolved prototypes: `4`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6909`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.04514103755354881, 0.06572993099689484]`
- Stage transport loss trace: `[0.10023545741569251, 0.03974902513436973]`
- Stage merge-count trace: `[1.0, 0.0]`
- Stage routing stability trace: `[0.38712432980537415, 0.18231327272951603]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`

## Hyperparameter Adaptation

- Adaptation steps: `1`
- Adaptation strategy: `correlation`
- Final hyperparameters: `{'prototype_prior_strength': 1.0244903512183976, 'prototype_capacity_blend': 0.496365587625769, 'prototype_relocation_strength': 0.7547629383130697, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1}`