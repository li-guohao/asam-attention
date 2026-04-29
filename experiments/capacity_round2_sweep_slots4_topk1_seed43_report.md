# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.5000`
- Avg forgetting: `-0.0781`
- Backward transfer: `0.0781`

## Artifacts

- Plot image: `capacity_round2_sweep_slots4_topk1_seed43_plots.png`
- Raw JSON: `capacity_round2_sweep_slots4_topk1_seed43.json`
- Resolved prototypes: `8`
- Prototype top-k: `1`
- Prototype slots/task: `4`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.0001`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.07539541274309158, 0.0829571783542633]`
- Stage transport loss trace: `[0.09781857766211033, 0.03385610750410706]`
- Stage merge-count trace: `[2.0, 1.0]`
- Stage routing stability trace: `[0.3478967337869108, 0.13460925710387528]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`