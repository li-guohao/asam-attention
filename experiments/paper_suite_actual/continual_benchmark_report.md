# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.5000`
- Avg forgetting: `0.0000`
- Backward transfer: `0.0000`

## Artifacts

- Plot image: `continual_benchmark_plots.png`
- Raw JSON: `continual_benchmark.json`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6925`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[7.450580596923828e-09, 1.4901161193847656e-08]`
- Stage transport loss trace: `[0.2931173974648118, 0.12635421566665173]`
- Stage merge-count trace: `[0.0, 0.0]`
- Stage routing stability trace: `[0.001191278930491535, 0.0016423719062004238]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`

## Hyperparameter Adaptation

- Adaptation steps: `1`
- Adaptation strategy: `meta_secant`
- Final hyperparameters: `{'prototype_prior_strength': 1.0000744577049931, 'prototype_capacity_blend': 0.49267206496279686, 'prototype_relocation_strength': 0.7573279353091493, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1}`