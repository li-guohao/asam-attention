# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `kl_topk`
- Tasks: `2`
- Avg accuracy: `0.5312`
- Avg forgetting: `0.0000`
- Backward transfer: `0.0000`

## Artifacts

- Plot image: `continual_operator_ablation_kl_topk_seed42_plots.png`
- Raw JSON: `continual_operator_ablation_kl_topk_seed42.json`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6922`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.002614237368106842, 0.0031774938106536865]`
- Stage transport loss trace: `[0.30179584864526987, 0.07327897846698761]`
- Stage merge-count trace: `[0.0, 0.0]`
- Stage routing stability trace: `[0.00020857555864495225, 0.00019954250092268921]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`