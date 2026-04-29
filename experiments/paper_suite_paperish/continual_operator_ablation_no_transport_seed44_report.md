# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.4766`
- Avg forgetting: `0.1406`
- Backward transfer: `-0.1406`

## Artifacts

- Plot image: `continual_operator_ablation_no_transport_seed44_plots.png`
- Raw JSON: `continual_operator_ablation_no_transport_seed44.json`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6901`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.140625]`
- Stage transport gap trace: `[0.0, 4.470348358154297e-08]`
- Stage transport loss trace: `[0.09717287379316986, 0.03155418857932091]`
- Stage merge-count trace: `[0.0, 0.0]`
- Stage routing stability trace: `[0.012929731019539759, 0.004378715129860211]`
- Forgetting vs routing stability correlation: `-1.0`
- Forgetting vs transport gap correlation: `1.0`
- Forgetting vs transport loss correlation: `-0.9999999999999999`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`