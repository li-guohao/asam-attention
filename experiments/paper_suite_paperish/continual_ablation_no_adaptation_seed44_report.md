# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.4766`
- Avg forgetting: `0.1406`
- Backward transfer: `-0.1406`

## Artifacts

- Plot image: `continual_ablation_no_adaptation_seed44_plots.png`
- Raw JSON: `continual_ablation_no_adaptation_seed44.json`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6900`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.140625]`
- Stage transport gap trace: `[0.0, 1.4901161193847656e-08]`
- Stage transport loss trace: `[0.09771448688115925, 0.030071593588218093]`
- Stage merge-count trace: `[0.0, 0.0]`
- Stage routing stability trace: `[0.015999344846932217, 0.00445931628200924]`
- Forgetting vs routing stability correlation: `-1.0`
- Forgetting vs transport gap correlation: `1.0`
- Forgetting vs transport loss correlation: `-1.0`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`