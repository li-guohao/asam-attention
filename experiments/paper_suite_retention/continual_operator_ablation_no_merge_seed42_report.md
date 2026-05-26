# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.5000`
- Avg forgetting: `-0.0312`
- Backward transfer: `0.0312`

## Artifacts

- Plot image: `continual_operator_ablation_no_merge_seed42_plots.png`
- Raw JSON: `continual_operator_ablation_no_merge_seed42.json`
- Resolved prototypes: `4`
- Prototype top-k: `1`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.0000`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.1390341818332672, 0.15411251783370972]`
- Stage transport loss trace: `[0.1180221107788384, 0.04516256367787719]`
- Stage merge-count trace: `[0.0, 0.0]`
- Stage routing stability trace: `[0.3362783221527934, 0.20824363827705383]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`