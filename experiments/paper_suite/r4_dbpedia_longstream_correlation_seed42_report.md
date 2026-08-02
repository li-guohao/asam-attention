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
- Avg accuracy: `0.0464`
- Avg forgetting: `0.3500`
- Backward transfer: `-0.1667`

## Artifacts

- Plot image: `r4_dbpedia_longstream_correlation_seed42_plots.png`
- Raw JSON: `r4_dbpedia_longstream_correlation_seed42.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `1.1066`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.35, 0.39999999999999997, 0.4125, 0.38, 0.35833333333333334]`
- Stage transport gap trace: `[0.047090066596865654, 0.04686155542731285, 0.04548872821033001, 0.0461414884775877, 0.04984455928206444, 0.054293831810355186, 0.05802421644330025]`
- Stage transport loss trace: `[0.18818894972403843, 0.10635616034269332, 0.08035205701986949, 0.07152785807847976, 0.05932388380169869, 0.09055943315227827, 0.064429411975046]`
- Stage merge-count trace: `[2.0, 3.0, 4.0, 3.0, 5.0, 3.0, 3.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff gate-factor trace: `[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]`
- Stage Birkhoff offdiag-mass trace: `[0.014787494204938412, 0.19625288993120193, 0.18438253551721573, 0.18259555101394653, 0.1888839304447174, 0.1668795719742775, 0.16577432304620743]`
- Stage Birkhoff applied-offdiag trace: `[0.00029574988409876825, 0.003925057798624039, 0.0036876507103443144, 0.0036519110202789308, 0.003777678608894348, 0.00333759143948555, 0.0033154864609241485]`
- Stage Birkhoff gap-delta trace: `[-1.3152137398719788e-05, -0.00027065351605415344, -0.0002263784408569336, -0.00024928152561187744, -0.00027763843536376953, -0.00020866096019744873, -0.00020421668887138367]`
- Stage Birkhoff row-error trace: `[0.00011098384857177734, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 2.384185791015625e-07, 1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage Birkhoff col-error trace: `[1.1920928955078125e-07, 1.1920928955078125e-07, 5.960464477539063e-08, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage routing stability trace: `[0.054933303222060206, 0.7388898332913717, 0.6792050043741862, 0.6165623545646668, 0.5744392911593119, 0.6896598060925802, 0.7148763686418533]`
- Forgetting vs routing stability correlation: `0.5033181992610027`
- Forgetting vs transport gap correlation: `0.36274610399397583`
- Forgetting vs transport loss correlation: `-0.8192106765016575`
- Forgetting vs mean abs excess correlation: `0.3627463720753966`
- Forgetting vs merge-count correlation: `0.5928328545072867`

## Hyperparameter Adaptation

- Adaptation steps: `6`
- Adaptation strategy: `correlation`
- Final hyperparameters: `{'prototype_prior_strength': 2.1581045618736034, 'prototype_capacity_blend': 0.2964718090710528, 'prototype_masked_sinkhorn_capacity_bias': 0.0, 'prototype_relocation_strength': 0.9372074903694602, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 2, 'transport_weight': 0.05}`