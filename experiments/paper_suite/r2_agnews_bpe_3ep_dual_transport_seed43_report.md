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
- Avg accuracy: `0.5312`
- Avg forgetting: `-0.0625`
- Backward transfer: `0.0625`

## Artifacts

- Plot image: `r2_agnews_bpe_3ep_dual_transport_seed43_plots.png`
- Raw JSON: `r2_agnews_bpe_3ep_dual_transport_seed43.json`
- Resolved prototypes: `4`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `1.1724`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.025294032879173756, 0.04718690365552902]`
- Stage transport loss trace: `[0.17818852389852205, 0.052102589048445225]`
- Stage merge-count trace: `[2.0, 0.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.009672796974579494, 0.015]`
- Stage Birkhoff gate-factor trace: `[0.4836398487289747, 0.75]`
- Stage Birkhoff offdiag-mass trace: `[0.024960827082395554, 0.04671580530703068]`
- Stage Birkhoff applied-offdiag trace: `[0.00024162211692720095, 0.0007024972140789032]`
- Stage Birkhoff gap-delta trace: `[-3.725290298461914e-09, -7.450580596923828e-09]`
- Stage Birkhoff row-error trace: `[1.5616416931152344e-05, 0.0]`
- Stage Birkhoff col-error trace: `[5.960464477539063e-08, 0.0]`
- Stage routing stability trace: `[0.5717415312925974, 0.39092957476774853]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`

## Hyperparameter Adaptation

- Adaptation steps: `1`
- Adaptation strategy: `dual_transport`
- Final hyperparameters: `{'prototype_prior_strength': 1.0, 'prototype_capacity_blend': 0.5, 'prototype_relocation_strength': 0.75, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 2, 'transport_weight': 0.05, 'task_transport_weights': [0.05, 0.05]}`