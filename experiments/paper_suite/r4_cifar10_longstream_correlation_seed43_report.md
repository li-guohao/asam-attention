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

- Plot image: `r4_cifar10_longstream_correlation_seed43_plots.png`
- Raw JSON: `r4_cifar10_longstream_correlation_seed43.json`
- Resolved prototypes: `10`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `1.3847`
- Prototype heatmap rows: `5`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.5769230769230769, 0.5769230769230769, 0.5512820512820512, 0.548076923076923]`
- Stage transport gap trace: `[0.05978173576295376, 0.06733259558677673, 0.07713067159056664, 0.08458461612462997, 0.07100491598248482]`
- Stage transport loss trace: `[0.2395775555854752, 0.0979763890306155, 0.08336841350510008, 0.08219845983244124, 0.0718798635320531]`
- Stage merge-count trace: `[2.0, 2.0, 2.0, 3.0, 4.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff gate-factor trace: `[1.0, 1.0, 1.0, 1.0, 1.0]`
- Stage Birkhoff offdiag-mass trace: `[0.028503685258328915, 0.167759507894516, 0.16738475859165192, 0.20207522064447403, 0.22247715294361115]`
- Stage Birkhoff applied-offdiag trace: `[0.0005700737051665783, 0.00335519015789032, 0.003347695171833038, 0.004041504412889481, 0.004449543058872223]`
- Stage Birkhoff gap-delta trace: `[-3.108754754066467e-05, -0.0004159994423389435, -0.0004900433123111725, -0.0006802305579185486, -0.0006308555603027344]`
- Stage Birkhoff row-error trace: `[0.00022834539413452148, 5.960464477539063e-08, 5.960464477539063e-08, 1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage Birkhoff col-error trace: `[5.960464477539063e-08, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage routing stability trace: `[0.21364867900099074, 0.2470500891407331, 0.30802770313762484, 0.3983700679881232, 0.6528928445445167]`
- Forgetting vs routing stability correlation: `0.43622351864338127`
- Forgetting vs transport gap correlation: `0.7066006886830623`
- Forgetting vs transport loss correlation: `-0.984277177052437`
- Forgetting vs mean abs excess correlation: `0.7066006081733063`
- Forgetting vs merge-count correlation: `0.32739891895375806`

## Hyperparameter Adaptation

- Adaptation steps: `4`
- Adaptation strategy: `correlation`
- Final hyperparameters: `{'prototype_prior_strength': 2.1373132737385285, 'prototype_capacity_blend': 0.0683103269950685, 'prototype_masked_sinkhorn_capacity_bias': 0.0, 'prototype_relocation_strength': 1.0, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 2, 'transport_weight': 0.05}`