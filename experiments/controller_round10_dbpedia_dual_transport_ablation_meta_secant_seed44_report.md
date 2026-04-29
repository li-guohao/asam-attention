# Continual Text Benchmark Report

- Dataset: `split_dbpedia`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `7`
- Avg accuracy: `0.5000`
- Avg forgetting: `0.0667`
- Backward transfer: `0.0167`

## Artifacts

- Plot image: `controller_round10_dbpedia_dual_transport_ablation_meta_secant_seed44_plots.png`
- Raw JSON: `controller_round10_dbpedia_dual_transport_ablation_meta_secant_seed44.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6750`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.0, 0.033333333333333326, 0.04999999999999999, 0.07999999999999999, 0.08333333333333333]`
- Stage transport gap trace: `[0.02266412042081356, 0.03129138424992561, 0.036697305738925934, 0.03847505524754524, 0.04047393426299095, 0.04028450325131416, 0.04280809685587883]`
- Stage transport loss trace: `[0.33453848709662753, 0.057517419258753456, 0.046068926652272545, 0.07971868912378947, 0.0745216856400172, 0.06515810390313466, 0.04822681720058123]`
- Stage merge-count trace: `[0.0, 1.0, 2.0, 3.0, 2.0, 3.0, 1.0]`
- Stage routing stability trace: `[0.07595475266377132, 0.05862857773900032, 0.2557205061117808, 0.38071055213610333, 0.46010248859723407, 0.8419157862663269, 0.31708986560503644]`
- Forgetting vs routing stability correlation: `0.7639960849108673`
- Forgetting vs transport gap correlation: `0.7682013821779395`
- Forgetting vs transport loss correlation: `-0.40032010362147286`
- Forgetting vs mean abs excess correlation: `0.7682013428805484`
- Forgetting vs merge-count correlation: `0.4055353355259886`

## Hyperparameter Adaptation

- Adaptation steps: `6`
- Adaptation strategy: `meta_secant`
- Final hyperparameters: `{'prototype_prior_strength': 1.0918336208371924, 'prototype_capacity_blend': 0.407630181555876, 'prototype_relocation_strength': 0.868135058598292, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 2, 'transport_weight': 0.4153904479821585}`