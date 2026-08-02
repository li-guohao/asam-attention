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
- Avg forgetting: `0.5385`
- Backward transfer: `-0.5385`

## Artifacts

- Plot image: `r4_cifar10_longstream_meta_secant_seed43_plots.png`
- Raw JSON: `r4_cifar10_longstream_meta_secant_seed43.json`
- Resolved prototypes: `10`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6719`
- Prototype heatmap rows: `5`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.5769230769230769, 0.5769230769230769, 0.5512820512820512, 0.5384615384615384]`
- Stage transport gap trace: `[0.05978173576295376, 0.06798956170678139, 0.07281399145722389, 0.09547598287463188, 0.10143954679369926]`
- Stage transport loss trace: `[0.2395775555854752, 0.09773430441107069, 0.06621593006310009, 0.04534154997340271, 0.031083640228543017]`
- Stage merge-count trace: `[2.0, 2.0, 1.0, 3.0, 2.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff gate-factor trace: `[1.0, 1.0, 1.0, 1.0, 1.0]`
- Stage Birkhoff offdiag-mass trace: `[0.028503685258328915, 0.16672810167074203, 0.1582716703414917, 0.22290769964456558, 0.18888800591230392]`
- Stage Birkhoff applied-offdiag trace: `[0.0005700737051665783, 0.0033345620334148405, 0.003165433406829834, 0.004458153992891312, 0.0037777601182460785]`
- Stage Birkhoff gap-delta trace: `[-3.108754754066467e-05, -0.00041750073432922363, -0.00040650367736816406, -0.0014127790927886963, -0.0011645928025245667]`
- Stage Birkhoff row-error trace: `[0.00022834539413452148, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage Birkhoff col-error trace: `[5.960464477539063e-08, 1.1920928955078125e-07, 1.1920928955078125e-07, 5.960464477539063e-08, 5.960464477539063e-08]`
- Stage routing stability trace: `[0.21364867900099074, 0.24505570459933507, 0.24128184006327674, 0.38467123359441757, 0.15932398206657833]`
- Forgetting vs routing stability correlation: `0.24149144110660836`
- Forgetting vs transport gap correlation: `0.5578336546889952`
- Forgetting vs transport loss correlation: `-0.9347692119704611`
- Forgetting vs mean abs excess correlation: `0.5578336458352308`
- Forgetting vs merge-count correlation: `-0.03606092229873097`

## Hyperparameter Adaptation

- Adaptation steps: `4`
- Adaptation strategy: `meta_secant`
- Final hyperparameters: `{'prototype_prior_strength': 1.327042259763241, 'prototype_capacity_blend': 0.30244568303193675, 'prototype_masked_sinkhorn_capacity_bias': 0.0, 'prototype_relocation_strength': 0.9595302204876167, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 1, 'transport_weight': 1.0}`