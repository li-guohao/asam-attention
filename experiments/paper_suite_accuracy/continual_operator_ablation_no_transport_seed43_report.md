# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.5703`
- Avg forgetting: `-0.1562`
- Backward transfer: `0.1562`

## Artifacts

- Plot image: `continual_operator_ablation_no_transport_seed43_plots.png`
- Raw JSON: `continual_operator_ablation_no_transport_seed43.json`
- Resolved prototypes: `4`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6910`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.044611770659685135, 0.06550092995166779]`
- Stage transport loss trace: `[0.10202434309758246, 0.04487247671931982]`
- Stage merge-count trace: `[1.0, 0.0]`
- Stage routing stability trace: `[0.3881137054413557, 0.1809664994943887]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`