# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.5000`
- Avg forgetting: `-0.3333`
- Backward transfer: `0.3333`

## Artifacts

- Plot image: `task_level_dual_transport_ablation_smoke_no_adaptation_seed42_plots.png`
- Raw JSON: `task_level_dual_transport_ablation_smoke_no_adaptation_seed42.json`
- Resolved prototypes: `4`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6580`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.02656858041882515, 0.022973142564296722]`
- Stage transport loss trace: `[0.36989953617254895, 0.07062020152807236]`
- Stage merge-count trace: `[1.0, 1.0]`
- Stage routing stability trace: `[0.3352906306584676, 0.455181360244751]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`