# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.4766`
- Avg forgetting: `0.0312`
- Backward transfer: `-0.0312`

## Artifacts

- Plot image: `continual_benchmark_plots.png`
- Raw JSON: `continual_benchmark.json`
- Resolved prototypes: `4`
- Prototype top-k: `1`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.0000`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.03125]`
- Stage transport gap trace: `[0.09737126529216766, 0.10172488540410995]`
- Stage transport loss trace: `[0.12562380614690483, 0.07188825635239482]`
- Stage merge-count trace: `[1.0, 1.0]`
- Stage routing stability trace: `[0.33304902561940253, 0.1742657849099487]`
- Forgetting vs routing stability correlation: `-0.9999999999999999`
- Forgetting vs transport gap correlation: `1.0`
- Forgetting vs transport loss correlation: `-1.0`
- Forgetting vs mean abs excess correlation: `1.0`
- Forgetting vs merge-count correlation: `None`

## Hyperparameter Adaptation

- Adaptation steps: `1`
- Adaptation strategy: `meta_secant`
- Final hyperparameters: `{'prototype_prior_strength': 1.021033718998757, 'prototype_capacity_blend': 0.4944251232140232, 'prototype_relocation_strength': 0.758009158418281, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1}`