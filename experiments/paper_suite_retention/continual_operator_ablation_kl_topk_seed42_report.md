# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `kl_topk`
- Tasks: `2`
- Avg accuracy: `0.4531`
- Avg forgetting: `0.0000`
- Backward transfer: `0.0000`

## Artifacts

- Plot image: `continual_operator_ablation_kl_topk_seed42_plots.png`
- Raw JSON: `continual_operator_ablation_kl_topk_seed42.json`
- Resolved prototypes: `4`
- Prototype top-k: `1`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.0000`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.009592272341251373, 0.0298672616481781]`
- Stage transport loss trace: `[0.11593220406211913, 0.04424446739722043]`
- Stage merge-count trace: `[1.0, 0.0]`
- Stage routing stability trace: `[0.10529157659038901, 0.20442561013624072]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`