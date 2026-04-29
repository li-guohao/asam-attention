# Continual Text Benchmark Report

- Dataset: `split_dbpedia`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `7`
- Avg accuracy: `0.4857`
- Avg forgetting: `0.0667`
- Backward transfer: `0.0000`

## Artifacts

- Plot image: `controller_round11_dbpedia_dual_transport_persistent_topk_ablation_correlation_seed44_plots.png`
- Raw JSON: `controller_round11_dbpedia_dual_transport_persistent_topk_ablation_correlation_seed44.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6850`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.0, 0.033333333333333326, 0.09999999999999999, 0.06, 0.06666666666666665]`
- Stage transport gap trace: `[0.02266412042081356, 0.031225210055708885, 0.035184867680072784, 0.03706130012869835, 0.03920667991042137, 0.04053189605474472, 0.04571561887860298]`
- Stage transport loss trace: `[0.33453848709662753, 0.05751737952232361, 0.046069297939538956, 0.07008909309903781, 0.06995415190855662, 0.052373060335715614, 0.0562176468471686]`
- Stage merge-count trace: `[0.0, 1.0, 2.0, 2.0, 2.0, 3.0, 1.0]`
- Stage routing stability trace: `[0.07595475266377132, 0.05862844487031301, 0.2556687891483307, 0.2979591290156047, 0.3927622189124425, 0.8188056945800781, 0.32869406541188556]`
- Forgetting vs routing stability correlation: `0.6184211019072765`
- Forgetting vs transport gap correlation: `0.7317800631631645`
- Forgetting vs transport loss correlation: `-0.3717928302273884`
- Forgetting vs mean abs excess correlation: `0.7317801276699302`
- Forgetting vs merge-count correlation: `0.44967308386531346`

## Hyperparameter Adaptation

- Adaptation steps: `6`
- Adaptation strategy: `correlation`
- Final hyperparameters: `{'prototype_prior_strength': 1.4943379206423564, 'prototype_capacity_blend': 0.2790310289707938, 'prototype_relocation_strength': 0.8790865326188727, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 2, 'transport_weight': 0.05}`