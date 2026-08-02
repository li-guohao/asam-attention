# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.4375`
- Avg forgetting: `-0.1250`
- Backward transfer: `0.1250`

## Artifacts

- Plot image: `continual_operator_ablation_no_relocation_seed42_plots.png`
- Raw JSON: `continual_operator_ablation_no_relocation_seed42.json`
- Resolved prototypes: `4`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6683`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.0161803737282753, 0.021318916231393814]`
- Stage transport loss trace: `[0.30774285923689604, 0.07688441686332226]`
- Stage merge-count trace: `[2.0, 1.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.010856033286143955, 0.0071063072731097545]`
- Stage Birkhoff gate-factor trace: `[0.5428016643071978, 0.3553153636554877]`
- Stage Birkhoff offdiag-mass trace: `[0.5526880621910095, 0.024437088519334793]`
- Stage Birkhoff applied-offdiag trace: `[0.006, 0.00017365745987857573]`
- Stage Birkhoff gap-delta trace: `[-0.000314345583319664, -5.587935447692871e-09]`
- Stage Birkhoff row-error trace: `[8.940696716308594e-07, 1.5616416931152344e-05]`
- Stage Birkhoff col-error trace: `[0.0, 0.0]`
- Stage routing stability trace: `[0.3603001981973648, 0.07132363878190517]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`