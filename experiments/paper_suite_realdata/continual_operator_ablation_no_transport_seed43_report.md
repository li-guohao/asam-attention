# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.4688`
- Avg forgetting: `0.0000`
- Backward transfer: `0.0000`

## Artifacts

- Plot image: `continual_operator_ablation_no_transport_seed43_plots.png`
- Raw JSON: `continual_operator_ablation_no_transport_seed43.json`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6684`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.0, 0.0]`
- Stage transport loss trace: `[0.2937467051669955, 0.05581843666732311]`
- Stage merge-count trace: `[0.0, 0.0]`
- Stage routing stability trace: `[0.030443690717220306, 0.04813469294458628]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`