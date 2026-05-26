# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.5000`
- Avg forgetting: `0.2500`
- Backward transfer: `-0.2500`

## Artifacts

- Plot image: `task_level_dual_transport_smoke_plots.png`
- Raw JSON: `task_level_dual_transport_smoke.json`
- Resolved prototypes: `4`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6786`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.25]`
- Stage transport gap trace: `[0.009920045733451843, 0.018125634640455246]`
- Stage transport loss trace: `[0.5380902718752623, 0.06131333112716675]`
- Stage merge-count trace: `[2.0, 1.0]`
- Stage routing stability trace: `[0.3059500455856323, 0.0385311059653759]`
- Forgetting vs routing stability correlation: `-1.0`
- Forgetting vs transport gap correlation: `1.0`
- Forgetting vs transport loss correlation: `-0.9999999999999999`
- Forgetting vs mean abs excess correlation: `1.0`
- Forgetting vs merge-count correlation: `-0.9999999999999999`

## Hyperparameter Adaptation

- Adaptation steps: `1`
- Adaptation strategy: `dual_transport`
- Final hyperparameters: `{'prototype_prior_strength': 1.0, 'prototype_capacity_blend': 0.5, 'prototype_relocation_strength': 0.75, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 2, 'transport_weight': 0.05, 'task_transport_weights': [0.05, 0.05]}`