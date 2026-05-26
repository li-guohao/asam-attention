# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.4766`
- Avg forgetting: `0.0312`
- Backward transfer: `-0.0312`

## Artifacts

- Plot image: `controller_round1_topk1_ablation_no_adaptation_seed42_plots.png`
- Raw JSON: `controller_round1_topk1_ablation_no_adaptation_seed42.json`
- Resolved prototypes: `4`
- Prototype top-k: `1`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.0000`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.03125]`
- Stage transport gap trace: `[0.09602127224206924, 0.10063104331493378]`
- Stage transport loss trace: `[0.1180221107788384, 0.04682839824818075]`
- Stage merge-count trace: `[1.0, 1.0]`
- Stage routing stability trace: `[0.3362783221527934, 0.1806830505374819]`
- Forgetting vs routing stability correlation: `-0.9999999999999999`
- Forgetting vs transport gap correlation: `0.9999999999999999`
- Forgetting vs transport loss correlation: `-0.9999999999999999`
- Forgetting vs mean abs excess correlation: `1.0`
- Forgetting vs merge-count correlation: `None`