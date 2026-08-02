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

- Plot image: `r4_cifar10_longstream_correlation_seed44_plots.png`
- Raw JSON: `r4_cifar10_longstream_correlation_seed44.json`
- Resolved prototypes: `10`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `1.0542`
- Prototype heatmap rows: `5`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.5, 0.5, 0.5, 0.5]`
- Stage transport gap trace: `[0.05522916465997696, 0.06530085951089859, 0.08011942356824875, 0.09254428371787071, 0.07059169933199883]`
- Stage transport loss trace: `[0.2506973317691258, 0.09346394631124678, 0.07799357104869116, 0.0745936930179596, 0.08542945856849353]`
- Stage merge-count trace: `[3.0, 4.0, 2.0, 2.0, 4.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff gate-factor trace: `[1.0, 1.0, 1.0, 1.0, 1.0]`
- Stage Birkhoff offdiag-mass trace: `[0.024379843845963478, 0.16566616669297218, 0.16992053389549255, 0.1651463769376278, 0.1975395604968071]`
- Stage Birkhoff applied-offdiag trace: `[0.0004875968769192696, 0.0033133233338594435, 0.003398410677909851, 0.0033029275387525562, 0.003950791209936142]`
- Stage Birkhoff gap-delta trace: `[-1.486949622631073e-05, -0.0003698877990245819, -0.00047529861330986023, -0.0005361177027225494, -0.0005168206989765167]`
- Stage Birkhoff row-error trace: `[7.915496826171875e-05, 5.960464477539063e-08, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage Birkhoff col-error trace: `[1.1920928955078125e-07, 1.1920928955078125e-07, 5.960464477539063e-08, 5.960464477539063e-08, 1.1920928955078125e-07]`
- Stage routing stability trace: `[0.37118517855803174, 0.3024882851612, 0.26018676197245005, 0.34924279082389104, 0.6397770080301497]`
- Forgetting vs routing stability correlation: `0.05025023889929393`
- Forgetting vs transport gap correlation: `0.6871847887666155`
- Forgetting vs transport loss correlation: `-0.9953492590285482`
- Forgetting vs mean abs excess correlation: `0.6871847207356939`
- Forgetting vs merge-count correlation: `0.0`

## Hyperparameter Adaptation

- Adaptation steps: `4`
- Adaptation strategy: `correlation`
- Final hyperparameters: `{'prototype_prior_strength': 1.5184558624453395, 'prototype_capacity_blend': 0.09121228165340023, 'prototype_masked_sinkhorn_capacity_bias': 0.0, 'prototype_relocation_strength': 1.0, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 2, 'transport_weight': 0.05}`