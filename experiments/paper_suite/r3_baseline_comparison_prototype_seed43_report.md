# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Protocol: `task_incremental_multihead`
- Label mode: `local`
- Head mode: `multi`
- Train task-id mode: `oracle`
- Eval task-id mode: `oracle`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Output classes: `2`
- Avg accuracy: `0.5000`
- Avg forgetting: `0.0000`
- Backward transfer: `0.0000`

## Artifacts

- Plot image: `r3_baseline_comparison_prototype_seed43_plots.png`
- Raw JSON: `r3_baseline_comparison_prototype_seed43.json`
- Resolved prototypes: `4`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6915`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.003566295839846134, 0.019132597371935844]`
- Stage transport loss trace: `[0.28526044078171253, 0.05955701041966677]`
- Stage merge-count trace: `[2.0, 0.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.002387503782908122, 0.003188766849537691]`
- Stage Birkhoff gate-factor trace: `[0.1193751891454061, 0.15943834247688454]`
- Stage Birkhoff offdiag-mass trace: `[0.5543926954269409, 0.043859973549842834]`
- Stage Birkhoff applied-offdiag trace: `[0.0013236146575484518, 0.0001398592296773388]`
- Stage Birkhoff gap-delta trace: `[-1.4959834516048431e-05, -3.725290298461914e-09]`
- Stage Birkhoff row-error trace: `[9.5367431640625e-07, 0.0]`
- Stage Birkhoff col-error trace: `[0.0, 0.0]`
- Stage routing stability trace: `[0.5194668620824814, 0.18236878386233002]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`