# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.5781`
- Avg forgetting: `-0.1875`
- Backward transfer: `0.1875`

## Artifacts

- Plot image: `continual_operator_ablation_no_relocation_seed43_plots.png`
- Raw JSON: `continual_operator_ablation_no_relocation_seed43.json`
- Resolved prototypes: `4`
- Prototype top-k: `1`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.0000`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.09039053320884705, 0.10098409652709961]`
- Stage transport loss trace: `[0.10049432562664151, 0.03508030949160457]`
- Stage merge-count trace: `[1.0, 1.0]`
- Stage routing stability trace: `[0.49980129674077034, 0.26290373457595706]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`