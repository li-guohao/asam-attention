# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.5000`
- Avg forgetting: `0.0000`
- Backward transfer: `0.0000`

## Artifacts

- Plot image: `continual_operator_ablation_no_transport_seed42_plots.png`
- Raw JSON: `continual_operator_ablation_no_transport_seed42.json`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6914`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.0, 0.0]`
- Stage transport loss trace: `[0.2931748563423753, 0.13163909129798412]`
- Stage merge-count trace: `[0.0, 0.0]`
- Stage routing stability trace: `[0.0005601258017122746, 0.001100107874663081]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`