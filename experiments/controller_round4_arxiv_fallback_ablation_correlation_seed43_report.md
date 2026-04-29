# Continual Text Benchmark Report

- Dataset: `split_arxiv`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `4`
- Avg accuracy: `0.6250`
- Avg forgetting: `0.0000`
- Backward transfer: `0.1667`

## Artifacts

- Plot image: `controller_round4_arxiv_fallback_ablation_correlation_seed43_plots.png`
- Raw JSON: `controller_round4_arxiv_fallback_ablation_correlation_seed43.json`
- Resolved prototypes: `8`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.5962`
- Prototype heatmap rows: `4`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.0, 0.0]`
- Stage transport gap trace: `[0.01845387928187847, 0.02747834473848343, 0.03269197419285774, 0.037713877856731415]`
- Stage transport loss trace: `[0.4728832356631756, 0.07469772547483444, 0.10636233165860176, 0.1222214512526989]`
- Stage merge-count trace: `[3.0, 1.0, 1.0, 1.0]`
- Stage routing stability trace: `[0.399270236492157, 0.21669021993875504, 0.3399289697408676, 0.24443769454956055]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`

## Hyperparameter Adaptation

- Adaptation steps: `3`
- Adaptation strategy: `correlation`
- Final hyperparameters: `{'prototype_prior_strength': 1.0615637842397798, 'prototype_capacity_blend': 0.4816858126461739, 'prototype_relocation_strength': 0.7702797922305763, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 2, 'transport_weight': 0.05}`