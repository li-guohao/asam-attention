# Continual Text Benchmark Report

- Dataset: `split_dbpedia`
- Protocol: `class_incremental_singlehead`
- Label mode: `global`
- Head mode: `single`
- Train task-id mode: `oracle`
- Eval task-id mode: `none`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `7`
- Output classes: `14`
- Avg accuracy: `0.1071`
- Avg forgetting: `0.3333`
- Backward transfer: `-0.2000`

## Artifacts

- Plot image: `r4_dbpedia_longstream_correlation_seed44_plots.png`
- Raw JSON: `r4_dbpedia_longstream_correlation_seed44.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `1.1684`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.375, 0.4166666666666667, 0.4, 0.37, 0.3333333333333333]`
- Stage transport gap trace: `[0.04704268276691437, 0.046629998832941055, 0.046438200399279594, 0.04908857308328152, 0.05253606103360653, 0.055824391543865204, 0.057478148490190506]`
- Stage transport loss trace: `[0.17218391249577206, 0.09330257227023443, 0.07613083496689796, 0.07386061822374662, 0.05799135069052378, 0.08583059708277384, 0.0872015447045366]`
- Stage merge-count trace: `[2.0, 3.0, 3.0, 4.0, 6.0, 4.0, 7.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff gate-factor trace: `[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]`
- Stage Birkhoff offdiag-mass trace: `[0.01647813990712166, 0.21585917472839355, 0.20407848805189133, 0.20724251866340637, 0.20268288999795914, 0.1900477558374405, 0.20448633283376694]`
- Stage Birkhoff applied-offdiag trace: `[0.00032956279814243315, 0.0043171834945678715, 0.004081569761037827, 0.004144850373268127, 0.004053657799959183, 0.0038009551167488094, 0.004089726656675339]`
- Stage Birkhoff gap-delta trace: `[-1.7309561371803284e-05, -0.00033801235258579254, -0.00030638836324214935, -0.0003221798688173294, -0.0003350861370563507, -0.00028328411281108856, -0.0003671012818813324]`
- Stage Birkhoff row-error trace: `[6.985664367675781e-05, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage Birkhoff col-error trace: `[5.960464477539063e-08, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage routing stability trace: `[0.053996566062172256, 0.7744457642237346, 0.6329203963279724, 0.5751946846644084, 0.522293770313263, 0.761953063805898, 0.7088533192873001]`
- Forgetting vs routing stability correlation: `0.4068881730498643`
- Forgetting vs transport gap correlation: `0.5024827439472141`
- Forgetting vs transport loss correlation: `-0.7632702632235671`
- Forgetting vs mean abs excess correlation: `0.5024828436862266`
- Forgetting vs merge-count correlation: `0.5810977993732518`

## Hyperparameter Adaptation

- Adaptation steps: `6`
- Adaptation strategy: `correlation`
- Final hyperparameters: `{'prototype_prior_strength': 2.0552241327983185, 'prototype_capacity_blend': 0.1813535652624576, 'prototype_masked_sinkhorn_capacity_bias': 0.0, 'prototype_relocation_strength': 0.99555631719906, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 2, 'transport_weight': 0.05}`