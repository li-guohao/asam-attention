# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `kl_topk`
- Tasks: `2`
- Avg accuracy: `0.5312`
- Avg forgetting: `-0.0625`
- Backward transfer: `0.0625`

## Artifacts

- Plot image: `continual_operator_ablation_kl_topk_seed43_plots.png`
- Raw JSON: `continual_operator_ablation_kl_topk_seed43.json`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6917`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.0065518394112586975, 0.004298657178878784]`
- Stage transport loss trace: `[0.29427594132721424, 0.0889586117118597]`
- Stage merge-count trace: `[0.0, 0.0]`
- Stage routing stability trace: `[0.0025007938384078443, 0.0003237017663195729]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`