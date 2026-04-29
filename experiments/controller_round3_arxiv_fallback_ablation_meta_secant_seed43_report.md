# Continual Text Benchmark Report

- Dataset: `split_arxiv`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `4`
- Avg accuracy: `0.6250`
- Avg forgetting: `0.0000`
- Backward transfer: `0.1667`

## Artifacts

- Plot image: `controller_round3_arxiv_fallback_ablation_meta_secant_seed43_plots.png`
- Raw JSON: `controller_round3_arxiv_fallback_ablation_meta_secant_seed43.json`
- Resolved prototypes: `8`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.0001`
- Prototype heatmap rows: `4`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.0, 0.0]`
- Stage transport gap trace: `[0.01845387928187847, 0.03712955862283707, 0.053593430668115616, 0.06893645226955414]`
- Stage transport loss trace: `[0.4728832356631756, 0.058851320296525955, 0.2916127145290375, 0.11306625232100487]`
- Stage merge-count trace: `[3.0, 1.0, 1.0, 1.0]`
- Stage routing stability trace: `[0.399270236492157, 0.3034415617585182, 2.052436947822571, 0.9211260676383972]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`

## Hyperparameter Adaptation

- Adaptation steps: `3`
- Adaptation strategy: `meta_secant`
- Final hyperparameters: `{'prototype_prior_strength': 1.7400291849868395, 'prototype_capacity_blend': 0.0777826391801047, 'prototype_relocation_strength': 0.8494540799154824, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 1, 'transport_weight': 0.0}`