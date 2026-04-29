# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `kl_topk`
- Tasks: `2`
- Avg accuracy: `0.4766`
- Avg forgetting: `0.1406`
- Backward transfer: `-0.1406`

## Artifacts

- Plot image: `continual_operator_ablation_kl_topk_seed44_plots.png`
- Raw JSON: `continual_operator_ablation_kl_topk_seed44.json`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6915`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.140625]`
- Stage transport gap trace: `[0.00020709633827209473, 0.0007631629705429077]`
- Stage transport loss trace: `[0.09850228426512331, 0.029595918022096157]`
- Stage merge-count trace: `[0.0, 0.0]`
- Stage routing stability trace: `[0.0003581768960430054, 0.00019257458279753337]`
- Forgetting vs routing stability correlation: `-1.0`
- Forgetting vs transport gap correlation: `1.0`
- Forgetting vs transport loss correlation: `-1.0`
- Forgetting vs mean abs excess correlation: `1.0`
- Forgetting vs merge-count correlation: `None`