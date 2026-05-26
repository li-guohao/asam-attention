# Continual Text Benchmark Report

- Dataset: `split_arxiv`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `4`
- Avg accuracy: `0.6250`
- Avg forgetting: `0.0000`
- Backward transfer: `0.1667`

## Artifacts

- Plot image: `controller_round4_arxiv_fallback_ablation_meta_secant_seed43_plots.png`
- Raw JSON: `controller_round4_arxiv_fallback_ablation_meta_secant_seed43.json`
- Resolved prototypes: `8`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.0001`
- Prototype heatmap rows: `4`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.0, 0.0]`
- Stage transport gap trace: `[0.01845387928187847, 0.03712955862283707, 0.042164575308561325, 0.04969204217195511]`
- Stage transport loss trace: `[0.4728832356631756, 0.058851320296525955, 0.11716978251934052, 0.1628035381436348]`
- Stage merge-count trace: `[3.0, 1.0, 1.0, 1.0]`
- Stage routing stability trace: `[0.399270236492157, 0.3034415617585182, 0.6818397492170334, 0.16500435769557953]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`

## Hyperparameter Adaptation

- Adaptation steps: `3`
- Adaptation strategy: `meta_secant`
- Final hyperparameters: `{'prototype_prior_strength': 1.4792557939747475, 'prototype_capacity_blend': 0.4030514168767369, 'prototype_relocation_strength': 0.8519671423023748, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 1, 'transport_weight': 0.0}`