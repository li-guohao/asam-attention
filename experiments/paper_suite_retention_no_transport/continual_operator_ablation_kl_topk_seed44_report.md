# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `kl_topk`
- Tasks: `2`
- Avg accuracy: `0.5391`
- Avg forgetting: `-0.0312`
- Backward transfer: `0.0312`

## Artifacts

- Plot image: `continual_operator_ablation_kl_topk_seed44_plots.png`
- Raw JSON: `continual_operator_ablation_kl_topk_seed44.json`
- Resolved prototypes: `4`
- Prototype top-k: `1`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.0000`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.0178535059094429, 0.035072848200798035]`
- Stage transport loss trace: `[0.09540411352645606, 0.029802850214764476]`
- Stage merge-count trace: `[0.0, 0.0]`
- Stage routing stability trace: `[0.15828983497340232, 0.24717685673385859]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`