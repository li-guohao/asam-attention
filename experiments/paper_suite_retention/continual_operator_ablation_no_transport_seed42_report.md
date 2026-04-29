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
- Prototype top-k: `1`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.0000`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.03125]`
- Stage transport gap trace: `[0.09737126529216766, 0.100723035633564]`
- Stage transport loss trace: `[0.12562380614690483, 0.0719130972865969]`
- Stage merge-count trace: `[1.0, 1.0]`
- Stage routing stability trace: `[0.33304902561940253, 0.1742657849099487]`
- Forgetting vs routing stability correlation: `-0.9999999999999999`
- Forgetting vs transport gap correlation: `1.0`
- Forgetting vs transport loss correlation: `-1.0`
- Forgetting vs mean abs excess correlation: `1.0`
- Forgetting vs merge-count correlation: `None`