# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.5117`
- Avg forgetting: `-0.0391`
- Backward transfer: `0.0391`

## Artifacts

- Plot image: `frontier_round1_ablation_correlation_seed46_plots.png`
- Raw JSON: `frontier_round1_ablation_correlation_seed46.json`
- Resolved prototypes: `4`
- Prototype top-k: `1`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.0000`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.10825492441654205, 0.10944297164678574]`
- Stage transport loss trace: `[0.06327887164661661, 0.03394194337306544]`
- Stage merge-count trace: `[1.0, 1.0]`
- Stage routing stability trace: `[0.1613588455657009, 0.10607563110534102]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`

## Hyperparameter Adaptation

- Adaptation steps: `1`
- Adaptation strategy: `correlation`
- Final hyperparameters: `{'prototype_prior_strength': 1.010135952113803, 'prototype_capacity_blend': 0.49571165514498716, 'prototype_relocation_strength': 0.7569947180119925, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1}`