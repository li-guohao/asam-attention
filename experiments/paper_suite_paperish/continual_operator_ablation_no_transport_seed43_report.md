# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.4766`
- Avg forgetting: `0.0625`
- Backward transfer: `-0.0625`

## Artifacts

- Plot image: `continual_operator_ablation_no_transport_seed43_plots.png`
- Raw JSON: `continual_operator_ablation_no_transport_seed43.json`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6904`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0625]`
- Stage transport gap trace: `[0.0, 1.4901161193847656e-08]`
- Stage transport loss trace: `[0.10558493598364294, 0.03877452632877976]`
- Stage merge-count trace: `[0.0, 0.0]`
- Stage routing stability trace: `[0.030668933992274106, 0.007030405409750529]`
- Forgetting vs routing stability correlation: `-1.0`
- Forgetting vs transport gap correlation: `0.9999999999999999`
- Forgetting vs transport loss correlation: `-0.9999999999999999`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`