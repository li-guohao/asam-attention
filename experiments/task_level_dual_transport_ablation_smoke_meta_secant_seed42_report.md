# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.5000`
- Avg forgetting: `-0.3333`
- Backward transfer: `0.3333`

## Artifacts

- Plot image: `task_level_dual_transport_ablation_smoke_meta_secant_seed42_plots.png`
- Raw JSON: `task_level_dual_transport_ablation_smoke_meta_secant_seed42.json`
- Resolved prototypes: `4`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6605`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.02656858041882515, 0.023301538079977036]`
- Stage transport loss trace: `[0.36989953617254895, 0.07177984217802684]`
- Stage merge-count trace: `[1.0, 1.0]`
- Stage routing stability trace: `[0.3352906306584676, 0.5085202952226003]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`

## Hyperparameter Adaptation

- Adaptation steps: `1`
- Adaptation strategy: `meta_secant`
- Final hyperparameters: `{'prototype_prior_strength': 1.043843940428263, 'prototype_capacity_blend': 0.48743143922959764, 'prototype_relocation_strength': 0.7617033381325503, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 2, 'transport_weight': 0.05, 'task_transport_weights': [0.05, 0.05]}`