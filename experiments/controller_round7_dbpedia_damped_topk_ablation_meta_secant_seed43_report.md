# Continual Text Benchmark Report

- Dataset: `split_dbpedia`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `7`
- Avg accuracy: `0.5429`
- Avg forgetting: `0.0500`
- Backward transfer: `0.0167`

## Artifacts

- Plot image: `controller_round7_dbpedia_damped_topk_ablation_meta_secant_seed43_plots.png`
- Raw JSON: `controller_round7_dbpedia_damped_topk_ablation_meta_secant_seed43.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.0002`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.0, 0.033333333333333326, 0.125, 0.1, 0.05000000000000001]`
- Stage transport gap trace: `[0.02266412042081356, 0.03129138797521591, 0.03904566541314125, 0.04315286874771118, 0.048003219068050385, 0.050413601100444794, 0.05136226490139961]`
- Stage transport loss trace: `[0.3345447393755118, 0.0466744564473629, 0.05236586804191271, 0.05329296117027601, 0.08467226972182591, 0.06121501699090004, 0.0426896425584952]`
- Stage merge-count trace: `[0.0, 1.0, 2.0, 2.0, 1.0, 0.0, 2.0]`
- Stage routing stability trace: `[0.07099447896083196, 0.07201472048958142, 0.26607981820901233, 0.23003400365511575, 0.43955238660176593, 0.9087343017260233, 0.7870805263519287]`
- Forgetting vs routing stability correlation: `0.688232252481794`
- Forgetting vs transport gap correlation: `0.749936397050088`
- Forgetting vs transport loss correlation: `-0.28157657223162813`
- Forgetting vs mean abs excess correlation: `0.7499363184777218`
- Forgetting vs merge-count correlation: `-0.22017967084286305`

## Hyperparameter Adaptation

- Adaptation steps: `6`
- Adaptation strategy: `meta_secant`
- Final hyperparameters: `{'prototype_prior_strength': 2.365987322311939, 'prototype_capacity_blend': 0.29874912345136684, 'prototype_relocation_strength': 0.9662148475990228, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 1, 'transport_weight': 0.4960202622713968}`