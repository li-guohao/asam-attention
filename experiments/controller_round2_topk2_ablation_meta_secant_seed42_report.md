# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.4766`
- Avg forgetting: `0.0312`
- Backward transfer: `-0.0312`

## Artifacts

- Plot image: `controller_round2_topk2_ablation_meta_secant_seed42_plots.png`
- Raw JSON: `controller_round2_topk2_ablation_meta_secant_seed42.json`
- Resolved prototypes: `4`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.0000`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.03125]`
- Stage transport gap trace: `[0.05441046133637428, 0.1630621701478958]`
- Stage transport loss trace: `[0.11797323008067906, 0.04935666988603771]`
- Stage merge-count trace: `[0.0, 1.0]`
- Stage routing stability trace: `[0.26707378309220076, 0.7117655668407679]`
- Forgetting vs routing stability correlation: `1.0`
- Forgetting vs transport gap correlation: `1.0`
- Forgetting vs transport loss correlation: `-1.0`
- Forgetting vs mean abs excess correlation: `0.9999999999999998`
- Forgetting vs merge-count correlation: `0.9999999999999999`

## Hyperparameter Adaptation

- Adaptation steps: `1`
- Adaptation strategy: `meta_secant`
- Final hyperparameters: `{'prototype_prior_strength': 1.036027879697728, 'prototype_capacity_blend': 0.49024936158093624, 'prototype_relocation_strength': 0.7634781522443518, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 1, 'transport_weight': 0.0}`