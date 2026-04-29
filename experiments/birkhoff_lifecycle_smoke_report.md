# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.5000`
- Avg forgetting: `0.2500`
- Backward transfer: `-0.2500`

## Artifacts

- Plot image: `birkhoff_lifecycle_smoke_plots.png`
- Raw JSON: `birkhoff_lifecycle_smoke.json`
- Resolved prototypes: `4`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6836`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.25]`
- Stage transport gap trace: `[0.009054004214704037, 0.017156224697828293]`
- Stage transport loss trace: `[0.5380902718752623, 0.059923989698290825]`
- Stage merge-count trace: `[2.0, 1.0]`
- Stage routing stability trace: `[0.3059500455856323, 0.03372975531965494]`
- Forgetting vs routing stability correlation: `-1.0`
- Forgetting vs transport gap correlation: `0.9999999999999999`
- Forgetting vs transport loss correlation: `-0.9999999999999999`
- Forgetting vs mean abs excess correlation: `1.0`
- Forgetting vs merge-count correlation: `-0.9999999999999999`

## Hyperparameter Adaptation

- Adaptation steps: `1`
- Adaptation strategy: `meta_secant`
- Final hyperparameters: `{'prototype_prior_strength': 1.0393356877302276, 'prototype_capacity_blend': 0.4854159926871944, 'prototype_relocation_strength': 0.7587805016555649, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 2, 'transport_weight': 0.05, 'task_transport_weights': [0.05, 0.05]}`