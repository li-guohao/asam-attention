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

- Plot image: `r2_baseline_comparison_prototype_seed43_plots.png`
- Raw JSON: `r2_baseline_comparison_prototype_seed43.json`
- Resolved prototypes: `4`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6922`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.003984242677688599, 0.017887266352772713]`
- Stage transport loss trace: `[0.28494357503950596, 0.05226545315235853]`
- Stage merge-count trace: `[2.0, 0.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.002668616672356924, 0.005962423359354338]`
- Stage Birkhoff gate-factor trace: `[0.1334308336178462, 0.29812116796771687]`
- Stage Birkhoff offdiag-mass trace: `[0.5544494986534119, 0.043782010674476624]`
- Stage Birkhoff applied-offdiag trace: `[0.0014796131760864328, 0.00026104688316500035]`
- Stage Birkhoff gap-delta trace: `[-1.86823308467865e-05, -3.725290298461914e-09]`
- Stage Birkhoff row-error trace: `[9.5367431640625e-07, 0.0]`
- Stage Birkhoff col-error trace: `[5.960464477539063e-08, 0.0]`
- Stage routing stability trace: `[0.5217461138963699, 0.20921031071338803]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`