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
- Avg accuracy: `0.1167`
- Avg forgetting: `0.5000`
- Backward transfer: `-0.5000`

## Artifacts

- Plot image: `r4_cifar10_longstream_dual_transport_seed44_plots.png`
- Raw JSON: `r4_cifar10_longstream_dual_transport_seed44.json`
- Resolved prototypes: `10`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.9917`
- Prototype heatmap rows: `5`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.5, 0.5, 0.5, 0.5]`
- Stage transport gap trace: `[0.055229175835847855, 0.06456406973302364, 0.06184859201312065, 0.061089809983968735, 0.06036814860999584]`
- Stage transport loss trace: `[0.2506973321239154, 0.093568069594247, 0.07580822564306713, 0.07319480500050954, 0.07597167293230693]`
- Stage merge-count trace: `[3.0, 3.0, 2.0, 2.0, 3.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff gate-factor trace: `[1.0, 1.0, 1.0, 1.0, 1.0]`
- Stage Birkhoff offdiag-mass trace: `[0.024379846639931202, 0.17150818556547165, 0.15147478133440018, 0.1423921249806881, 0.14561781659722328]`
- Stage Birkhoff applied-offdiag trace: `[0.000487596932798624, 0.003430163711309433, 0.0030294956266880037, 0.002847842499613762, 0.0029123563319444655]`
- Stage Birkhoff gap-delta trace: `[-1.4865770936012268e-05, -0.00041631050407886505, -0.00036239251494407654, -0.00033580511808395386, -0.0003175418823957443]`
- Stage Birkhoff row-error trace: `[7.915496826171875e-05, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage Birkhoff col-error trace: `[1.1920928955078125e-07, 5.960464477539063e-08, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage routing stability trace: `[0.3711851750101362, 0.2991612766470228, 0.2829915831486384, 0.32413344156174434, 0.3113998356792662]`
- Forgetting vs routing stability correlation: `-0.891165069530856`
- Forgetting vs transport gap correlation: `0.8847078639764058`
- Forgetting vs transport loss correlation: `-0.9944154684907229`
- Forgetting vs mean abs excess correlation: `0.8847078837027114`
- Forgetting vs merge-count correlation: `-0.40824829046386296`

## Hyperparameter Adaptation

- Adaptation steps: `4`
- Adaptation strategy: `dual_transport`
- Final hyperparameters: `{'prototype_prior_strength': 1.0, 'prototype_capacity_blend': 0.5, 'prototype_masked_sinkhorn_capacity_bias': 0.0, 'prototype_relocation_strength': 0.75, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 2, 'transport_weight': 0.09452697869997735, 'task_transport_weights': [0.12071394435449255, 0.09253934254376828, 0.08894095001504441, 0.07591367788660414, 0.09452697869997735]}`