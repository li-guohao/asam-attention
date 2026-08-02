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
- Avg forgetting: `0.5481`
- Backward transfer: `-0.5481`

## Artifacts

- Plot image: `r4_cifar10_longstream_dual_transport_seed43_plots.png`
- Raw JSON: `r4_cifar10_longstream_dual_transport_seed43.json`
- Resolved prototypes: `10`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `1.3658`
- Prototype heatmap rows: `5`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.5769230769230769, 0.5769230769230769, 0.5512820512820512, 0.548076923076923]`
- Stage transport gap trace: `[0.05978173203766346, 0.06710800901055336, 0.06077507324516773, 0.06292331963777542, 0.06080259196460247]`
- Stage transport loss trace: `[0.23957755594026475, 0.09820926260380518, 0.08239391836382094, 0.09087115739073072, 0.06666361437075669]`
- Stage merge-count trace: `[2.0, 2.0, 3.0, 2.0, 3.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff gate-factor trace: `[1.0, 1.0, 1.0, 1.0, 1.0]`
- Stage Birkhoff offdiag-mass trace: `[0.02850368618965149, 0.1686246246099472, 0.18970081955194473, 0.15270011872053146, 0.19647053629159927]`
- Stage Birkhoff applied-offdiag trace: `[0.0005700737237930298, 0.0033724924921989443, 0.0037940163910388947, 0.0030540023744106293, 0.003929410725831986]`
- Stage Birkhoff gap-delta trace: `[-3.108568489551544e-05, -0.00041897594928741455, -0.0004638936370611191, -0.0003281589597463608, -0.0005280133336782455]`
- Stage Birkhoff row-error trace: `[0.00022834539413452148, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage Birkhoff col-error trace: `[5.960464477539063e-08, 5.960464477539063e-08, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage routing stability trace: `[0.21364867935578027, 0.2359848189212027, 0.26634112781002406, 0.37390059658459257, 0.2607533687518703]`
- Forgetting vs routing stability correlation: `0.48492096759746867`
- Forgetting vs transport gap correlation: `0.49536951074337526`
- Forgetting vs transport loss correlation: `-0.9795493262894951`
- Forgetting vs mean abs excess correlation: `0.4953695914457209`
- Forgetting vs merge-count correlation: `0.40474930228150474`

## Hyperparameter Adaptation

- Adaptation steps: `4`
- Adaptation strategy: `dual_transport`
- Final hyperparameters: `{'prototype_prior_strength': 1.0, 'prototype_capacity_blend': 0.5, 'prototype_masked_sinkhorn_capacity_bias': 0.0, 'prototype_relocation_strength': 0.75, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 2, 'transport_weight': 0.1032853173180015, 'task_transport_weights': [0.13779694076165785, 0.102164107204686, 0.09190051539098561, 0.08127970591467654, 0.1032853173180015]}`