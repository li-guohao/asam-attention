# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.5156`
- Avg forgetting: `0.1094`
- Backward transfer: `-0.1094`

## Artifacts

- Plot image: `capacity_round2_sweep_slots4_topk2_seed42_plots.png`
- Raw JSON: `capacity_round2_sweep_slots4_topk2_seed42.json`
- Resolved prototypes: `8`
- Prototype top-k: `2`
- Prototype slots/task: `4`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6917`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.109375]`
- Stage transport gap trace: `[0.040986210107803345, 0.07191076874732971]`
- Stage transport loss trace: `[0.12018012325279415, 0.04984753066673875]`
- Stage merge-count trace: `[3.0, 1.0]`
- Stage routing stability trace: `[0.268473441246897, 0.20469480915926397]`
- Forgetting vs routing stability correlation: `-0.9999999999999999`
- Forgetting vs transport gap correlation: `1.0`
- Forgetting vs transport loss correlation: `-1.0`
- Forgetting vs mean abs excess correlation: `1.0`
- Forgetting vs merge-count correlation: `-1.0`