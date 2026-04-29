# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.5781`
- Avg forgetting: `-0.1875`
- Backward transfer: `0.1875`

## Artifacts

- Plot image: `controller_round1_topk1_ablation_meta_secant_seed43_plots.png`
- Raw JSON: `controller_round1_topk1_ablation_meta_secant_seed43.json`
- Resolved prototypes: `4`
- Prototype top-k: `1`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.0000`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.09039053320884705, 0.10304281115531921]`
- Stage transport loss trace: `[0.10049432562664151, 0.03854600142221898]`
- Stage merge-count trace: `[1.0, 1.0]`
- Stage routing stability trace: `[0.49980129674077034, 0.17679002531804144]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`

## Hyperparameter Adaptation

- Adaptation steps: `1`
- Adaptation strategy: `meta_secant`
- Final hyperparameters: `{'prototype_prior_strength': 1.070498401992909, 'prototype_capacity_blend': 0.4861888253013603, 'prototype_relocation_strength': 0.7737938489764928, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'transport_weight': 0.0}`