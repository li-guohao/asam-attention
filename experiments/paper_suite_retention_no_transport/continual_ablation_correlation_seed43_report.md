# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.5781`
- Avg forgetting: `-0.1875`
- Backward transfer: `0.1875`

## Artifacts

- Plot image: `continual_ablation_correlation_seed43_plots.png`
- Raw JSON: `continual_ablation_correlation_seed43.json`
- Resolved prototypes: `4`
- Prototype top-k: `1`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.0000`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.08895877003669739, 0.10140223056077957]`
- Stage transport loss trace: `[0.10672796866856515, 0.05442850850522518]`
- Stage merge-count trace: `[1.0, 1.0]`
- Stage routing stability trace: `[0.5107057197019458, 0.18046582327224314]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`

## Hyperparameter Adaptation

- Adaptation steps: `1`
- Adaptation strategy: `correlation`
- Final hyperparameters: `{'prototype_prior_strength': 1.0324339857416858, 'prototype_capacity_blend': 0.49510783157893457, 'prototype_relocation_strength': 0.757116137718549, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1}`