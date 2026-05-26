# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.5078`
- Avg forgetting: `-0.0781`
- Backward transfer: `0.0781`

## Artifacts

- Plot image: `capacity_round2_sweep_slots4_topk2_seed43_plots.png`
- Raw JSON: `capacity_round2_sweep_slots4_topk2_seed43.json`
- Resolved prototypes: `8`
- Prototype top-k: `2`
- Prototype slots/task: `4`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6904`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.04673845320940018, 0.0759158805012703]`
- Stage transport loss trace: `[0.09781662747263908, 0.035695700091309845]`
- Stage merge-count trace: `[3.0, 0.0]`
- Stage routing stability trace: `[0.6155849024653435, 0.1508574930485338]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`