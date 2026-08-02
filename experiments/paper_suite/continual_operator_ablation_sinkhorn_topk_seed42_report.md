# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.4688`
- Avg forgetting: `-0.1875`
- Backward transfer: `0.1875`

## Artifacts

- Plot image: `continual_operator_ablation_sinkhorn_topk_seed42_plots.png`
- Raw JSON: `continual_operator_ablation_sinkhorn_topk_seed42.json`
- Resolved prototypes: `4`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6737`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.0161803737282753, 0.0203110184520483]`
- Stage transport loss trace: `[0.30774285923689604, 0.07579363323748112]`
- Stage merge-count trace: `[2.0, 1.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.010856033286143955, 0.013540680209795635]`
- Stage Birkhoff gate-factor trace: `[0.5428016643071978, 0.6770340104897817]`
- Stage Birkhoff offdiag-mass trace: `[0.5526880621910095, 0.017509061843156815]`
- Stage Birkhoff applied-offdiag trace: `[0.006, 0.00023708460719172136]`
- Stage Birkhoff gap-delta trace: `[-0.000314345583319664, -1.862645149230957e-09]`
- Stage Birkhoff row-error trace: `[8.940696716308594e-07, 1.1086463928222656e-05]`
- Stage Birkhoff col-error trace: `[0.0, 5.960464477539063e-08]`
- Stage routing stability trace: `[0.3603001981973648, 0.12868799222633243]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`