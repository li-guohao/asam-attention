# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.5312`
- Avg forgetting: `0.0000`
- Backward transfer: `0.0000`

## Artifacts

- Plot image: `continual_ablation_correlation_seed44_plots.png`
- Raw JSON: `continual_ablation_correlation_seed44.json`
- Resolved prototypes: `4`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6758`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.031249813735485077, 0.039608001708984375]`
- Stage transport loss trace: `[0.0976022103568539, 0.03038001188542694]`
- Stage merge-count trace: `[0.0, 1.0]`
- Stage routing stability trace: `[0.43515569902956486, 0.6216953285038471]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`

## Hyperparameter Adaptation

- Adaptation steps: `1`
- Adaptation strategy: `correlation`
- Final hyperparameters: `{'prototype_prior_strength': 1.0275704517208166, 'prototype_capacity_blend': 0.4967786992638139, 'prototype_relocation_strength': 0.7540025459456956, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1}`