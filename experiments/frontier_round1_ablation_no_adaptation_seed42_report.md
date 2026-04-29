# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.4922`
- Avg forgetting: `-0.0078`
- Backward transfer: `0.0078`

## Artifacts

- Plot image: `frontier_round1_ablation_no_adaptation_seed42_plots.png`
- Raw JSON: `frontier_round1_ablation_no_adaptation_seed42.json`
- Resolved prototypes: `4`
- Prototype top-k: `1`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.0000`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.1068597063422203, 0.10580839216709137]`
- Stage transport loss trace: `[0.09344689117278904, 0.06849211337976158]`
- Stage merge-count trace: `[1.0, 1.0]`
- Stage routing stability trace: `[0.21236920155934058, 0.15255845326464623]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`