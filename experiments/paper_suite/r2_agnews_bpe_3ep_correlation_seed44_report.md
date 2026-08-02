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

- Plot image: `r2_agnews_bpe_3ep_correlation_seed44_plots.png`
- Raw JSON: `r2_agnews_bpe_3ep_correlation_seed44.json`
- Resolved prototypes: `4`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.9746`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.03245999664068222, 0.05336855351924896]`
- Stage transport loss trace: `[0.17949930392205715, 0.052425970789045095]`
- Stage merge-count trace: `[1.0, 0.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.02, 0.01125]`
- Stage Birkhoff gate-factor trace: `[1.0, 0.5625]`
- Stage Birkhoff offdiag-mass trace: `[0.03685238026082516, 0.050189223140478134]`
- Stage Birkhoff applied-offdiag trace: `[0.0007370476052165032, 0.0005624421685934067]`
- Stage Birkhoff gap-delta trace: `[0.0, 0.0]`
- Stage Birkhoff row-error trace: `[9.894371032714844e-06, 5.960464477539063e-08]`
- Stage Birkhoff col-error trace: `[0.0, 5.960464477539063e-08]`
- Stage routing stability trace: `[0.4675624618927638, 0.4129274810353915]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`

## Hyperparameter Adaptation

- Adaptation steps: `1`
- Adaptation strategy: `correlation`
- Final hyperparameters: `{'prototype_prior_strength': 1.0296538253618364, 'prototype_capacity_blend': 0.49470101748425177, 'prototype_relocation_strength': 0.7561104824300855, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 2, 'transport_weight': 0.05}`