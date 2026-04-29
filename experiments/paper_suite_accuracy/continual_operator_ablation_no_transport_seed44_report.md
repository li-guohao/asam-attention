# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.4844`
- Avg forgetting: `-0.0156`
- Backward transfer: `0.0156`

## Artifacts

- Plot image: `continual_operator_ablation_no_transport_seed44_plots.png`
- Raw JSON: `continual_operator_ablation_no_transport_seed44.json`
- Resolved prototypes: `4`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6922`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.03964436426758766, 0.06876486539840698]`
- Stage transport loss trace: `[0.09883308666758239, 0.037197008612565696]`
- Stage merge-count trace: `[0.0, 0.0]`
- Stage routing stability trace: `[0.4365296419709921, 0.23819544212892652]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`