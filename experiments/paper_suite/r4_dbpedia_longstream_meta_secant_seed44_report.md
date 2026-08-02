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
- Avg accuracy: `0.0929`
- Avg forgetting: `0.3833`
- Backward transfer: `-0.2333`

## Artifacts

- Plot image: `r4_dbpedia_longstream_meta_secant_seed44_plots.png`
- Raw JSON: `r4_dbpedia_longstream_meta_secant_seed44.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.2626`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.375, 0.4166666666666667, 0.35, 0.39, 0.3833333333333333]`
- Stage transport gap trace: `[0.04704268276691437, 0.04693981818854809, 0.04589774273335934, 0.049261294305324554, 0.05962604470551014, 0.07077346742153168, 0.07338766753673553]`
- Stage transport loss trace: `[0.17218391249577206, 0.09329236249128978, 0.07715320338805516, 0.067706049233675, 0.040723887955149016, 0.04511537825067838, 0.03323446105544766]`
- Stage merge-count trace: `[2.0, 3.0, 2.0, 4.0, 4.0, 3.0, 3.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff gate-factor trace: `[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]`
- Stage Birkhoff offdiag-mass trace: `[0.01647813990712166, 0.21590367704629898, 0.2028093934059143, 0.20452600717544556, 0.23429936170578003, 0.22003602981567383, 0.2331203892827034]`
- Stage Birkhoff applied-offdiag trace: `[0.00032956279814243315, 0.00431807354092598, 0.004056187868118287, 0.004090520143508911, 0.004685987234115601, 0.004400720596313476, 0.004662407785654068]`
- Stage Birkhoff gap-delta trace: `[-1.7309561371803284e-05, -0.0003403685986995697, -0.0003047138452529907, -0.00030937977135181427, -0.0008153486996889114, -0.0006910599768161774, -0.0007522478699684143]`
- Stage Birkhoff row-error trace: `[6.985664367675781e-05, 1.7881393432617188e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage Birkhoff col-error trace: `[5.960464477539063e-08, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 5.960464477539063e-08, 1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage routing stability trace: `[0.053996566062172256, 0.7745834251244863, 0.6340525825818379, 0.5801954984664917, 0.7896427035331726, 0.5935989081859588, 0.5122502793868383]`
- Forgetting vs routing stability correlation: `0.3828904645117211`
- Forgetting vs transport gap correlation: `0.5120983618145337`
- Forgetting vs transport loss correlation: `-0.8015011774960701`
- Forgetting vs mean abs excess correlation: `0.5120983899253022`
- Forgetting vs merge-count correlation: `0.42543193239295707`

## Hyperparameter Adaptation

- Adaptation steps: `6`
- Adaptation strategy: `meta_secant`
- Final hyperparameters: `{'prototype_prior_strength': 1.8518620874935148, 'prototype_capacity_blend': 0.2982527511084612, 'prototype_masked_sinkhorn_capacity_bias': 0.0, 'prototype_relocation_strength': 1.0, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 1, 'transport_weight': 1.0}`