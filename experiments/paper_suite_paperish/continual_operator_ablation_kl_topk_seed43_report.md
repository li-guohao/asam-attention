# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `kl_topk`
- Tasks: `2`
- Avg accuracy: `0.4766`
- Avg forgetting: `0.0625`
- Backward transfer: `-0.0625`

## Artifacts

- Plot image: `continual_operator_ablation_kl_topk_seed43_plots.png`
- Raw JSON: `continual_operator_ablation_kl_topk_seed43.json`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6918`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0625]`
- Stage transport gap trace: `[0.0016274452209472656, 0.0004073232412338257]`
- Stage transport loss trace: `[0.10543091560248286, 0.03483954758848995]`
- Stage merge-count trace: `[0.0, 0.0]`
- Stage routing stability trace: `[0.000584742669161642, 0.0003850889834211557]`
- Forgetting vs routing stability correlation: `-1.0`
- Forgetting vs transport gap correlation: `-0.9999999999999999`
- Forgetting vs transport loss correlation: `-1.0`
- Forgetting vs mean abs excess correlation: `-1.0`
- Forgetting vs merge-count correlation: `None`