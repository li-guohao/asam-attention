# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.4766`
- Avg forgetting: `0.0312`
- Backward transfer: `-0.0312`

## Artifacts

- Plot image: `continual_operator_ablation_no_transport_seed42_plots.png`
- Raw JSON: `continual_operator_ablation_no_transport_seed42.json`
- Resolved prototypes: `4`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6861`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.03125]`
- Stage transport gap trace: `[0.05420253798365593, 0.06110057979822159]`
- Stage transport loss trace: `[0.12187546980567276, 0.06024879473261535]`
- Stage merge-count trace: `[0.0, 0.0]`
- Stage routing stability trace: `[0.27402779227122664, 0.5227441936731339]`
- Forgetting vs routing stability correlation: `1.0`
- Forgetting vs transport gap correlation: `0.9999999999999998`
- Forgetting vs transport loss correlation: `-1.0`
- Forgetting vs mean abs excess correlation: `0.9999999999999998`
- Forgetting vs merge-count correlation: `None`