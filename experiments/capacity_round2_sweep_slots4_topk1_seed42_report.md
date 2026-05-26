# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.4922`
- Avg forgetting: `0.1094`
- Backward transfer: `-0.1094`

## Artifacts

- Plot image: `capacity_round2_sweep_slots4_topk1_seed42_plots.png`
- Raw JSON: `capacity_round2_sweep_slots4_topk1_seed42.json`
- Resolved prototypes: `8`
- Prototype top-k: `1`
- Prototype slots/task: `4`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.0001`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.109375]`
- Stage transport gap trace: `[0.07441522181034088, 0.08547233045101166]`
- Stage transport loss trace: `[0.11826934549026191, 0.04417153960093856]`
- Stage merge-count trace: `[1.0, 1.0]`
- Stage routing stability trace: `[0.14431429840624332, 0.13534526946023107]`
- Forgetting vs routing stability correlation: `-1.0`
- Forgetting vs transport gap correlation: `1.0`
- Forgetting vs transport loss correlation: `-1.0`
- Forgetting vs mean abs excess correlation: `1.0`
- Forgetting vs merge-count correlation: `None`