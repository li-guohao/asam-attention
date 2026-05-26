# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.7500`
- Avg forgetting: `0.0000`
- Backward transfer: `0.0000`

## Artifacts

- Plot image: `continual_operator_ablation_no_relocation_seed43_plots.png`
- Raw JSON: `continual_operator_ablation_no_relocation_seed43.json`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6856`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.0, 0.0]`
- Stage transport loss trace: `[0.29275134298950434, 0.09350526332855225]`
- Stage merge-count trace: `[0.0, 0.0]`
- Stage routing stability trace: `[0.02389890095219016, 0.019580159801989794]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`