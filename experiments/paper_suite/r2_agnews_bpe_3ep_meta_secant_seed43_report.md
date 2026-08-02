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

- Plot image: `r2_agnews_bpe_3ep_meta_secant_seed43_plots.png`
- Raw JSON: `r2_agnews_bpe_3ep_meta_secant_seed43.json`
- Resolved prototypes: `4`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `1.1726`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.025294032879173756, 0.047317443415522575]`
- Stage transport loss trace: `[0.17818851893146834, 0.05205608687053124]`
- Stage merge-count trace: `[2.0, 0.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.009672795111934345, 0.02]`
- Stage Birkhoff gate-factor trace: `[0.48363975559671724, 1.0]`
- Stage Birkhoff offdiag-mass trace: `[0.024960828013718128, 0.04666745103895664]`
- Stage Birkhoff applied-offdiag trace: `[0.00024162207478289208, 0.0009333490207791328]`
- Stage Birkhoff gap-delta trace: `[-3.725290298461914e-09, -3.725290298461914e-09]`
- Stage Birkhoff row-error trace: `[1.5616416931152344e-05, 5.960464477539063e-08]`
- Stage Birkhoff col-error trace: `[5.960464477539063e-08, 5.960464477539063e-08]`
- Stage routing stability trace: `[0.5717415461937586, 0.40143586571017903]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`

## Hyperparameter Adaptation

- Adaptation steps: `1`
- Adaptation strategy: `meta_secant`
- Final hyperparameters: `{'prototype_prior_strength': 1.074898742384429, 'prototype_capacity_blend': 0.492383532923485, 'prototype_relocation_strength': 0.7674552927580722, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 2, 'transport_weight': 0.05}`