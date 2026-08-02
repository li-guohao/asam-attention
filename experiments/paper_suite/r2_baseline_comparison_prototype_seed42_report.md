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

- Plot image: `r2_baseline_comparison_prototype_seed42_plots.png`
- Raw JSON: `r2_baseline_comparison_prototype_seed42.json`
- Resolved prototypes: `4`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6681`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.008833654224872589, 0.019975164905190468]`
- Stage transport loss trace: `[0.29260125290602446, 0.06502240523695946]`
- Stage merge-count trace: `[2.0, 1.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.0059511611859003714, 0.006658390164375305]`
- Stage Birkhoff gate-factor trace: `[0.29755805929501855, 0.33291950821876526]`
- Stage Birkhoff offdiag-mass trace: `[0.5533660054206848, 0.019285520538687706]`
- Stage Birkhoff applied-offdiag trace: `[0.003293170293056314, 0.00012841052026965616]`
- Stage Birkhoff gap-delta trace: `[-9.308755397796631e-05, -5.587935447692871e-09]`
- Stage Birkhoff row-error trace: `[8.940696716308594e-07, 1.2159347534179688e-05]`
- Stage Birkhoff col-error trace: `[5.960464477539063e-08, 5.960464477539063e-08]`
- Stage routing stability trace: `[0.3912890776991844, 0.12520722672343254]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`