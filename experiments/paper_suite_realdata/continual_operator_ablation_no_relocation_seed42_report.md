# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.5312`
- Avg forgetting: `0.0000`
- Backward transfer: `0.0000`

## Artifacts

- Plot image: `continual_operator_ablation_no_relocation_seed42_plots.png`
- Raw JSON: `continual_operator_ablation_no_relocation_seed42.json`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6876`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[7.450580596923828e-09, 1.4901161193847656e-08]`
- Stage transport loss trace: `[0.3015690827742219, 0.07350194733589888]`
- Stage merge-count trace: `[0.0, 0.0]`
- Stage routing stability trace: `[0.009641786338761449, 0.013019025092944503]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`