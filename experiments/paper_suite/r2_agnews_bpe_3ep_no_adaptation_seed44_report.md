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

- Plot image: `r2_agnews_bpe_3ep_no_adaptation_seed44_plots.png`
- Raw JSON: `r2_agnews_bpe_3ep_no_adaptation_seed44.json`
- Resolved prototypes: `4`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.9573`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.03245999664068222, 0.05361491069197655]`
- Stage transport loss trace: `[0.17949930392205715, 0.05209843209013343]`
- Stage merge-count trace: `[1.0, 0.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.02, 0.0075]`
- Stage Birkhoff gate-factor trace: `[1.0, 0.375]`
- Stage Birkhoff offdiag-mass trace: `[0.03685238026082516, 0.05028531514108181]`
- Stage Birkhoff applied-offdiag trace: `[0.0007370476052165032, 0.00037801727652549746]`
- Stage Birkhoff gap-delta trace: `[0.0, -1.862645149230957e-09]`
- Stage Birkhoff row-error trace: `[9.894371032714844e-06, 5.960464477539063e-08]`
- Stage Birkhoff col-error trace: `[0.0, 5.960464477539063e-08]`
- Stage routing stability trace: `[0.4675624618927638, 0.4055791099866231]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`