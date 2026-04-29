# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.5000`
- Avg forgetting: `0.0000`
- Backward transfer: `0.0000`

## Artifacts

- Plot image: `continual_ablation_no_adaptation_seed42_plots.png`
- Raw JSON: `continual_ablation_no_adaptation_seed42.json`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6925`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[7.450580596923828e-09, 0.0]`
- Stage transport loss trace: `[0.2931173974648118, 0.12635422870516777]`
- Stage merge-count trace: `[0.0, 0.0]`
- Stage routing stability trace: `[0.001191278930491535, 0.0016423663473688066]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`