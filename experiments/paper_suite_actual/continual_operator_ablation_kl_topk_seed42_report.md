# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `kl_topk`
- Tasks: `2`
- Avg accuracy: `0.5000`
- Avg forgetting: `0.0000`
- Backward transfer: `0.0000`

## Artifacts

- Plot image: `continual_operator_ablation_kl_topk_seed42_plots.png`
- Raw JSON: `continual_operator_ablation_kl_topk_seed42.json`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6881`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.005568049848079681, 0.006063491106033325]`
- Stage transport loss trace: `[0.29276447370648384, 0.12507020495831966]`
- Stage merge-count trace: `[0.0, 0.0]`
- Stage routing stability trace: `[0.00038185347511898726, 0.0001189151143989875]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`