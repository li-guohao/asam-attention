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
- Avg accuracy: `0.0643`
- Avg forgetting: `0.3250`
- Backward transfer: `-0.2083`

## Artifacts

- Plot image: `r4_dbpedia_longstream_correlation_seed43_plots.png`
- Raw JSON: `r4_dbpedia_longstream_correlation_seed43.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `1.3179`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.35, 0.36666666666666664, 0.3125, 0.35, 0.325]`
- Stage transport gap trace: `[0.04676903411746025, 0.05003691464662552, 0.04489201679825783, 0.04989962838590145, 0.050843892619013786, 0.05099308490753174, 0.05469451658427715]`
- Stage transport loss trace: `[0.20554628918568293, 0.12813340425491332, 0.09337303514281908, 0.08384022116661072, 0.07146042883396149, 0.11514092485109965, 0.08738750188301007]`
- Stage merge-count trace: `[2.0, 1.0, 4.0, 4.0, 4.0, 4.0, 5.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff gate-factor trace: `[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]`
- Stage Birkhoff offdiag-mass trace: `[0.015963700134307146, 0.16937556117773056, 0.1956036016345024, 0.19299815595149994, 0.19267088174819946, 0.20044878125190735, 0.20112042874097824]`
- Stage Birkhoff applied-offdiag trace: `[0.0003192740026861429, 0.003387511223554611, 0.0039120720326900485, 0.0038599631190299986, 0.003853417634963989, 0.004008975625038147, 0.004022408574819565]`
- Stage Birkhoff gap-delta trace: `[-1.52587890625e-05, -0.0002673175185918808, -0.00028117187321186066, -0.00030251964926719666, -0.0003161691129207611, -0.00036245398223400116, -0.0004031136631965637]`
- Stage Birkhoff row-error trace: `[0.00010842084884643555, 1.1920928955078125e-07, 1.7881393432617188e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage Birkhoff col-error trace: `[1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 5.960464477539063e-08, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage routing stability trace: `[0.08344089984893799, 0.6261384844779968, 0.5076792975266774, 0.5481885929902395, 0.6039423684279124, 0.6665323177973429, 0.5559572453300158]`
- Forgetting vs routing stability correlation: `0.5416267884404726`
- Forgetting vs transport gap correlation: `0.2424249198910026`
- Forgetting vs transport loss correlation: `-0.8025484980776805`
- Forgetting vs mean abs excess correlation: `0.24242554905615546`
- Forgetting vs merge-count correlation: `0.9262685031388972`

## Hyperparameter Adaptation

- Adaptation steps: `6`
- Adaptation strategy: `correlation`
- Final hyperparameters: `{'prototype_prior_strength': 2.015428144182196, 'prototype_capacity_blend': 0.34404217096911355, 'prototype_masked_sinkhorn_capacity_bias': 0.0, 'prototype_relocation_strength': 0.90665974570866, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 2, 'transport_weight': 0.05}`