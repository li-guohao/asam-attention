# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.4766`
- Avg forgetting: `0.0625`
- Backward transfer: `-0.0625`

## Artifacts

- Plot image: `continual_ablation_correlation_seed43_plots.png`
- Raw JSON: `continual_ablation_correlation_seed43.json`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6901`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0625]`
- Stage transport gap trace: `[1.4901161193847656e-08, 2.9802322387695312e-08]`
- Stage transport loss trace: `[0.10476192517671734, 0.03584296815097332]`
- Stage merge-count trace: `[0.0, 0.0]`
- Stage routing stability trace: `[0.03255378088215366, 0.00795010250294581]`
- Forgetting vs routing stability correlation: `-1.0`
- Forgetting vs transport gap correlation: `0.9999999999999999`
- Forgetting vs transport loss correlation: `-0.9999999999999999`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`

## Hyperparameter Adaptation

- Adaptation steps: `1`
- Adaptation strategy: `correlation`
- Final hyperparameters: `{'prototype_prior_strength': 1.0020366825311913, 'prototype_capacity_blend': 0.49738095164042245, 'prototype_relocation_strength': 0.752619048874476, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1}`