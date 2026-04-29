# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.4688`
- Avg forgetting: `0.0000`
- Backward transfer: `0.0000`

## Artifacts

- Plot image: `continual_operator_ablation_no_merge_seed43_plots.png`
- Raw JSON: `continual_operator_ablation_no_merge_seed43.json`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6670`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.0, 0.0]`
- Stage transport loss trace: `[0.2937461622059345, 0.05487756337970495]`
- Stage merge-count trace: `[0.0, 0.0]`
- Stage routing stability trace: `[0.03158281370997429, 0.05145370867103338]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`