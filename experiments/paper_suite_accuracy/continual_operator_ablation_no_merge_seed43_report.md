# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.5781`
- Avg forgetting: `-0.1875`
- Backward transfer: `0.1875`

## Artifacts

- Plot image: `continual_operator_ablation_no_merge_seed43_plots.png`
- Raw JSON: `continual_operator_ablation_no_merge_seed43.json`
- Resolved prototypes: `4`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6918`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.04514104127883911, 0.0686282217502594]`
- Stage transport loss trace: `[0.10023545741569251, 0.03686828108038753]`
- Stage merge-count trace: `[0.0, 0.0]`
- Stage routing stability trace: `[0.38712432980537415, 0.22626062482595444]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`