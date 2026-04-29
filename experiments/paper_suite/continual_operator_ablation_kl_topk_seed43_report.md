# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `kl_topk`
- Tasks: `2`
- Avg accuracy: `0.4688`
- Avg forgetting: `0.0000`
- Backward transfer: `0.0000`

## Artifacts

- Plot image: `continual_operator_ablation_kl_topk_seed43_plots.png`
- Raw JSON: `continual_operator_ablation_kl_topk_seed43.json`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6915`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.0023951977491378784, 0.0036837011575698853]`
- Stage transport loss trace: `[0.29609982296824455, 0.055391646921634674]`
- Stage merge-count trace: `[0.0, 0.0]`
- Stage routing stability trace: `[0.00046111608389765024, 0.0008524315344402567]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`