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
- Avg accuracy: `0.1083`
- Avg forgetting: `0.5000`
- Backward transfer: `-0.5000`

## Artifacts

- Plot image: `r4_cifar10_longstream_meta_secant_seed44_plots.png`
- Raw JSON: `r4_cifar10_longstream_meta_secant_seed44.json`
- Resolved prototypes: `10`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.0268`
- Prototype heatmap rows: `5`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.5, 0.5, 0.5, 0.5]`
- Stage transport gap trace: `[0.05522916465997696, 0.06924431398510933, 0.07423948496580124, 0.10001851245760918, 0.09982109069824219]`
- Stage transport loss trace: `[0.2506973317691258, 0.0939095417658488, 0.06472027124393553, 0.04041228931219805, 0.035738557370172605]`
- Stage merge-count trace: `[3.0, 4.0, 0.0, 2.0, 2.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff gate-factor trace: `[1.0, 1.0, 1.0, 1.0, 1.0]`
- Stage Birkhoff offdiag-mass trace: `[0.024379843845963478, 0.17036202549934387, 0.13269587233662605, 0.1925446167588234, 0.16718114167451859]`
- Stage Birkhoff applied-offdiag trace: `[0.0004875968769192696, 0.003407240509986878, 0.002653917446732521, 0.003850892335176468, 0.003343622833490372]`
- Stage Birkhoff gap-delta trace: `[-1.486949622631073e-05, -0.00042478740215301514, -0.0003172047436237335, -0.0011923201382160187, -0.000927019864320755]`
- Stage Birkhoff row-error trace: `[7.915496826171875e-05, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage Birkhoff col-error trace: `[1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 5.960464477539063e-08, 5.960464477539063e-08]`
- Stage routing stability trace: `[0.37118517855803174, 0.3050602624813716, 0.21189028982605254, 0.3447926136709395, 0.1869214963581827]`
- Forgetting vs routing stability correlation: `-0.6005329810912519`
- Forgetting vs transport gap correlation: `0.6939576933972773`
- Forgetting vs transport loss correlation: `-0.9656172907154301`
- Forgetting vs mean abs excess correlation: `0.6939576924707439`
- Forgetting vs merge-count correlation: `-0.3015113445777635`

## Hyperparameter Adaptation

- Adaptation steps: `4`
- Adaptation strategy: `meta_secant`
- Final hyperparameters: `{'prototype_prior_strength': 1.3064235651382754, 'prototype_capacity_blend': 0.32315526781896725, 'prototype_masked_sinkhorn_capacity_bias': 0.0, 'prototype_relocation_strength': 0.9426861464054543, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 1, 'transport_weight': 1.0}`