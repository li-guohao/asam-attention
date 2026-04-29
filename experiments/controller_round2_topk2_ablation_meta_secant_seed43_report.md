# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.5781`
- Avg forgetting: `-0.1562`
- Backward transfer: `0.1562`

## Artifacts

- Plot image: `controller_round2_topk2_ablation_meta_secant_seed43_plots.png`
- Raw JSON: `controller_round2_topk2_ablation_meta_secant_seed43.json`
- Resolved prototypes: `4`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.0000`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.04514103755354881, 0.10779495537281036]`
- Stage transport loss trace: `[0.10023545741569251, 0.04321022843942046]`
- Stage merge-count trace: `[1.0, 1.0]`
- Stage routing stability trace: `[0.38712432980537415, 0.3100649146363139]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`

## Hyperparameter Adaptation

- Adaptation steps: `1`
- Adaptation strategy: `meta_secant`
- Final hyperparameters: `{'prototype_prior_strength': 1.051344247281853, 'prototype_capacity_blend': 0.4918514838704141, 'prototype_relocation_strength': 0.765320737939328, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 1, 'transport_weight': 0.006741286933954621}`