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
- Avg forgetting: `0.5192`
- Backward transfer: `-0.5192`

## Artifacts

- Plot image: `r4_cifar10_longstream_dual_transport_seed42_plots.png`
- Raw JSON: `r4_cifar10_longstream_dual_transport_seed42.json`
- Resolved prototypes: `10`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.7255`
- Prototype heatmap rows: `5`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.5, 0.4807692307692308, 0.5256410256410257, 0.5192307692307693]`
- Stage transport gap trace: `[0.06356785073876381, 0.061638034880161285, 0.0639013722538948, 0.05974568612873554, 0.05748951435089111]`
- Stage transport loss trace: `[0.22801220949207032, 0.09471040112631661, 0.07359922606320608, 0.07738564926243964, 0.07653986993763182]`
- Stage merge-count trace: `[0.0, 2.0, 2.0, 2.0, 3.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff gate-factor trace: `[1.0, 1.0, 1.0, 1.0, 1.0]`
- Stage Birkhoff offdiag-mass trace: `[0.03330529108643532, 0.22083543241024017, 0.18151260912418365, 0.1940402239561081, 0.19318142533302307]`
- Stage Birkhoff applied-offdiag trace: `[0.0006661058217287064, 0.004416708648204804, 0.003630252182483673, 0.003880804479122162, 0.0038636285066604614]`
- Stage Birkhoff gap-delta trace: `[-4.4792890548706055e-05, -0.0005471110343933105, -0.00042498111724853516, -0.0004491303116083145, -0.0004473552107810974]`
- Stage Birkhoff row-error trace: `[0.00023889541625976562, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage Birkhoff col-error trace: `[5.960464477539063e-08, 5.960464477539063e-08, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage routing stability trace: `[0.0978805837886674, 0.28496114448422477, 0.3320945075580052, 0.32631153435934157, 0.29345705691311097]`
- Forgetting vs routing stability correlation: `0.9712579983057636`
- Forgetting vs transport gap correlation: `-0.5367002432578275`
- Forgetting vs transport loss correlation: `-0.9896329975198858`
- Forgetting vs mean abs excess correlation: `-0.5366996295358617`
- Forgetting vs merge-count correlation: `0.9286919881737385`

## Hyperparameter Adaptation

- Adaptation steps: `4`
- Adaptation strategy: `dual_transport`
- Final hyperparameters: `{'prototype_prior_strength': 1.0, 'prototype_capacity_blend': 0.5, 'prototype_masked_sinkhorn_capacity_bias': 0.0, 'prototype_relocation_strength': 0.75, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 2, 'transport_weight': 0.09583317920498366, 'task_transport_weights': [0.12185724520925434, 0.08866104463773164, 0.0973707393623494, 0.07544368761059927, 0.09583317920498366]}`