# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.7500`
- Avg forgetting: `0.0000`
- Backward transfer: `0.0000`

## Artifacts

- Plot image: `continual_operator_ablation_no_transport_seed43_plots.png`
- Raw JSON: `continual_operator_ablation_no_transport_seed43.json`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6867`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[1.4901161193847656e-08, 1.4901161193847656e-08]`
- Stage transport loss trace: `[0.2932009007781744, 0.09638916701078415]`
- Stage merge-count trace: `[0.0, 0.0]`
- Stage routing stability trace: `[0.013604371226392686, 0.021308912429958582]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`