# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.4766`
- Avg forgetting: `0.0312`
- Backward transfer: `-0.0312`

## Artifacts

- Plot image: `continual_ablation_no_adaptation_seed42_plots.png`
- Raw JSON: `continual_ablation_no_adaptation_seed42.json`
- Resolved prototypes: `4`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6881`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.03125]`
- Stage transport gap trace: `[0.05441046133637428, 0.06589968502521515]`
- Stage transport loss trace: `[0.11797323008067906, 0.04764856304973364]`
- Stage merge-count trace: `[0.0, 0.0]`
- Stage routing stability trace: `[0.26707378309220076, 0.484918424859643]`
- Forgetting vs routing stability correlation: `0.9999999999999998`
- Forgetting vs transport gap correlation: `1.0`
- Forgetting vs transport loss correlation: `-0.9999999999999999`
- Forgetting vs mean abs excess correlation: `1.0`
- Forgetting vs merge-count correlation: `None`