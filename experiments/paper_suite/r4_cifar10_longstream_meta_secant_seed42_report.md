# Continual Text Benchmark Report

- Dataset: `split_cifar10`
- Protocol: `class_incremental_singlehead`
- Label mode: `global`
- Head mode: `single`
- Train task-id mode: `oracle`
- Eval task-id mode: `none`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `5`
- Output classes: `10`
- Avg accuracy: `0.1000`
- Avg forgetting: `0.5192`
- Backward transfer: `-0.5192`

## Artifacts

- Plot image: `r4_cifar10_longstream_meta_secant_seed42_plots.png`
- Raw JSON: `r4_cifar10_longstream_meta_secant_seed42.json`
- Resolved prototypes: `10`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6879`
- Prototype heatmap rows: `5`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.5, 0.4807692307692308, 0.5256410256410257, 0.5192307692307693]`
- Stage transport gap trace: `[0.06356785073876381, 0.0627520028501749, 0.06932481378316879, 0.08425955846905708, 0.08795949444174767]`
- Stage transport loss trace: `[0.2280122126851763, 0.0943276672845795, 0.061530267198880516, 0.0434761643409729, 0.03648989616582791]`
- Stage merge-count trace: `[0.0, 2.0, 1.0, 3.0, 2.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff gate-factor trace: `[1.0, 1.0, 1.0, 1.0, 1.0]`
- Stage Birkhoff offdiag-mass trace: `[0.03330528549849987, 0.2189386934041977, 0.19097867608070374, 0.27214328944683075, 0.22227367013692856]`
- Stage Birkhoff applied-offdiag trace: `[0.0006661057099699975, 0.0043787738680839535, 0.003819573521614075, 0.0054428657889366155, 0.004445473402738571]`
- Stage Birkhoff gap-delta trace: `[-4.4792890548706055e-05, -0.0005508307367563248, -0.0004947707056999207, -0.0014798715710639954, -0.001241687685251236]`
- Stage Birkhoff row-error trace: `[0.00023889541625976562, 1.1920928955078125e-07, 1.1920928955078125e-07, 5.960464477539063e-08, 1.1920928955078125e-07]`
- Stage Birkhoff col-error trace: `[5.960464477539063e-08, 5.960464477539063e-08, 1.1920928955078125e-07, 1.1920928955078125e-07, 5.960464477539063e-08]`
- Stage routing stability trace: `[0.09788058724786554, 0.29257586491959436, 0.25097271019504186, 0.41057832006897244, 0.17922890103525585]`
- Forgetting vs routing stability correlation: `0.718455608633935`
- Forgetting vs transport gap correlation: `0.5258134687066467`
- Forgetting vs transport loss correlation: `-0.9682914226561166`
- Forgetting vs mean abs excess correlation: `0.5258136318234301`
- Forgetting vs merge-count correlation: `0.8254318357092731`

## Hyperparameter Adaptation

- Adaptation steps: `4`
- Adaptation strategy: `meta_secant`
- Final hyperparameters: `{'prototype_prior_strength': 1.2708633797002142, 'prototype_capacity_blend': 0.39358743448462835, 'prototype_masked_sinkhorn_capacity_bias': 0.0, 'prototype_relocation_strength': 0.8376584826254796, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 1, 'transport_weight': 0.9699194173564838}`