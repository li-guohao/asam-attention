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

- Plot image: `r4_cifar10_longstream_correlation_seed42_plots.png`
- Raw JSON: `r4_cifar10_longstream_correlation_seed42.json`
- Resolved prototypes: `10`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.8385`
- Prototype heatmap rows: `5`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.5, 0.4807692307692308, 0.5256410256410257, 0.5192307692307693]`
- Stage transport gap trace: `[0.06356785073876381, 0.06236104667186737, 0.07456053420901299, 0.08480177074670792, 0.08471161872148514]`
- Stage transport loss trace: `[0.2280122126851763, 0.09468372698341097, 0.07202516620357831, 0.07461617496751603, 0.07944387632111709]`
- Stage merge-count trace: `[0.0, 2.0, 1.0, 2.0, 2.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff gate-factor trace: `[1.0, 1.0, 1.0, 1.0, 1.0]`
- Stage Birkhoff offdiag-mass trace: `[0.03330528549849987, 0.22082343697547913, 0.19919367879629135, 0.21750714629888535, 0.20383217930793762]`
- Stage Birkhoff applied-offdiag trace: `[0.0006661057099699975, 0.004416468739509583, 0.003983873575925827, 0.004350142925977707, 0.004076643586158753]`
- Stage Birkhoff gap-delta trace: `[-4.4792890548706055e-05, -0.0005533769726753235, -0.0005385540425777435, -0.0006867200136184692, -0.0006267614662647247]`
- Stage Birkhoff row-error trace: `[0.00023889541625976562, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 5.960464477539063e-08]`
- Stage Birkhoff col-error trace: `[5.960464477539063e-08, 1.1920928955078125e-07, 1.1920928955078125e-07, 5.960464477539063e-08, 1.1920928955078125e-07]`
- Stage routing stability trace: `[0.09788058724786554, 0.28496571878592175, 0.24658803712754024, 0.24304694469485963, 0.3717224796613057]`
- Forgetting vs routing stability correlation: `0.8638961154437447`
- Forgetting vs transport gap correlation: `0.5738994614229979`
- Forgetting vs transport loss correlation: `-0.9883352946014168`
- Forgetting vs mean abs excess correlation: `0.5738995175868962`
- Forgetting vs merge-count correlation: `0.9039339244538892`

## Hyperparameter Adaptation

- Adaptation steps: `4`
- Adaptation strategy: `correlation`
- Final hyperparameters: `{'prototype_prior_strength': 2.1446710280714187, 'prototype_capacity_blend': 0.2542499233696523, 'prototype_masked_sinkhorn_capacity_bias': 0.0, 'prototype_relocation_strength': 0.9572606912219012, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 2, 'transport_weight': 0.05}`