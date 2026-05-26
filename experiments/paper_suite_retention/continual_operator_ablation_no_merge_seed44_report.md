# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.4844`
- Avg forgetting: `-0.0156`
- Backward transfer: `0.0156`

## Artifacts

- Plot image: `continual_operator_ablation_no_merge_seed44_plots.png`
- Raw JSON: `continual_operator_ablation_no_merge_seed44.json`
- Resolved prototypes: `4`
- Prototype top-k: `1`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.0000`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.05461576581001282, 0.16648060083389282]`
- Stage transport loss trace: `[0.09822813048958778, 0.03120536368805915]`
- Stage merge-count trace: `[0.0, 0.0]`
- Stage routing stability trace: `[0.7416652627289295, 0.4681613575667143]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`