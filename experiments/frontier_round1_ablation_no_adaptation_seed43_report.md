# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.4805`
- Avg forgetting: `0.0391`
- Backward transfer: `-0.0391`

## Artifacts

- Plot image: `frontier_round1_ablation_no_adaptation_seed43_plots.png`
- Raw JSON: `frontier_round1_ablation_no_adaptation_seed43.json`
- Resolved prototypes: `4`
- Prototype top-k: `1`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.0000`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0390625]`
- Stage transport gap trace: `[0.06653288006782532, 0.11349552869796753]`
- Stage transport loss trace: `[0.07437814015429467, 0.05063143849838525]`
- Stage merge-count trace: `[1.0, 1.0]`
- Stage routing stability trace: `[0.676790508441627, 0.2120400252752006]`
- Forgetting vs routing stability correlation: `-1.0`
- Forgetting vs transport gap correlation: `1.0`
- Forgetting vs transport loss correlation: `-1.0`
- Forgetting vs mean abs excess correlation: `1.0`
- Forgetting vs merge-count correlation: `None`