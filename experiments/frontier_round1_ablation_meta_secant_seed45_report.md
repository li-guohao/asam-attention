# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.5234`
- Avg forgetting: `-0.0469`
- Backward transfer: `0.0469`

## Artifacts

- Plot image: `frontier_round1_ablation_meta_secant_seed45_plots.png`
- Raw JSON: `frontier_round1_ablation_meta_secant_seed45.json`
- Resolved prototypes: `4`
- Prototype top-k: `1`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.0000`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.10764577984809875, 0.1090434193611145]`
- Stage transport loss trace: `[0.07708439778070897, 0.05007734661921859]`
- Stage merge-count trace: `[1.0, 1.0]`
- Stage routing stability trace: `[0.29959317814791575, 0.12944444548338652]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`

## Hyperparameter Adaptation

- Adaptation steps: `1`
- Adaptation strategy: `meta_secant`
- Final hyperparameters: `{'prototype_prior_strength': 1.0189009777737343, 'prototype_capacity_blend': 0.4953817455127137, 'prototype_relocation_strength': 0.7573093989369226, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1}`