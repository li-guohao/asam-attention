# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `kl_topk`
- Tasks: `2`
- Avg accuracy: `0.5391`
- Avg forgetting: `-0.0625`
- Backward transfer: `0.0625`

## Artifacts

- Plot image: `continual_operator_ablation_kl_topk_seed42_plots.png`
- Raw JSON: `continual_operator_ablation_kl_topk_seed42.json`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6921`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.0006502866744995117, 0.0011526644229888916]`
- Stage transport loss trace: `[0.11854650173336267, 0.04650372313335538]`
- Stage merge-count trace: `[0.0, 0.0]`
- Stage routing stability trace: `[0.0001412662440998247, 0.00010353482798564073]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`