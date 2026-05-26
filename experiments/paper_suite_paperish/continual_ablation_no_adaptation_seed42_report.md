# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.5391`
- Avg forgetting: `-0.0625`
- Backward transfer: `0.0625`

## Artifacts

- Plot image: `continual_ablation_no_adaptation_seed42_plots.png`
- Raw JSON: `continual_ablation_no_adaptation_seed42.json`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6910`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.0, 1.4901161193847656e-08]`
- Stage transport loss trace: `[0.11891891364939511, 0.04826871817931533]`
- Stage merge-count trace: `[0.0, 0.0]`
- Stage routing stability trace: `[0.008211900116293691, 0.003231208582292311]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`