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

- Plot image: `r2_agnews_bpe_3ep_dual_transport_seed44_plots.png`
- Raw JSON: `r2_agnews_bpe_3ep_dual_transport_seed44.json`
- Resolved prototypes: `4`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.9574`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.03245999477803707, 0.05361490696668625]`
- Stage transport loss trace: `[0.1794993088891109, 0.0520984378332893]`
- Stage merge-count trace: `[1.0, 0.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.02, 0.0125]`
- Stage Birkhoff gate-factor trace: `[1.0, 0.625]`
- Stage Birkhoff offdiag-mass trace: `[0.03685237839818001, 0.050285305827856064]`
- Stage Birkhoff applied-offdiag trace: `[0.0007370475679636002, 0.0006311986036598682]`
- Stage Birkhoff gap-delta trace: `[0.0, -1.862645149230957e-09]`
- Stage Birkhoff row-error trace: `[9.894371032714844e-06, 5.960464477539063e-08]`
- Stage Birkhoff col-error trace: `[0.0, 5.960464477539063e-08]`
- Stage routing stability trace: `[0.4675624792774518, 0.4055791087448597]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`

## Hyperparameter Adaptation

- Adaptation steps: `1`
- Adaptation strategy: `dual_transport`
- Final hyperparameters: `{'prototype_prior_strength': 1.0, 'prototype_capacity_blend': 0.5, 'prototype_relocation_strength': 0.75, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 2, 'transport_weight': 0.05, 'task_transport_weights': [0.05, 0.05]}`