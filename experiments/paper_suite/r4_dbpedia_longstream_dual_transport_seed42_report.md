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
- Avg forgetting: `0.3667`
- Backward transfer: `-0.1583`

## Artifacts

- Plot image: `r4_dbpedia_longstream_dual_transport_seed42_plots.png`
- Raw JSON: `r4_dbpedia_longstream_dual_transport_seed42.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.8289`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.35, 0.39999999999999997, 0.425, 0.4, 0.375]`
- Stage transport gap trace: `[0.0470900684595108, 0.046474022790789604, 0.04742603190243244, 0.04489630460739136, 0.04524426907300949, 0.04552173614501953, 0.04652257822453976]`
- Stage transport loss trace: `[0.18818894823392232, 0.10657281130552292, 0.08040863871574402, 0.07615855286518733, 0.0674797902504603, 0.0839742325246334, 0.06237373128533363]`
- Stage merge-count trace: `[2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 3.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff gate-factor trace: `[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]`
- Stage Birkhoff offdiag-mass trace: `[0.014787493739277124, 0.196124829351902, 0.17217306792736053, 0.17543622851371765, 0.16760101914405823, 0.15502840280532837, 0.15712029859423637]`
- Stage Birkhoff applied-offdiag trace: `[0.0002957498747855425, 0.003922496587038041, 0.003443461358547211, 0.0035087245702743533, 0.003352020382881165, 0.0031005680561065675, 0.0031424059718847272]`
- Stage Birkhoff gap-delta trace: `[-1.3154000043869019e-05, -0.00026798248291015625, -0.0002201087772846222, -0.0002490803599357605, -0.00023299269378185272, -0.00020780600607395172, -0.00025318749248981476]`
- Stage Birkhoff row-error trace: `[0.00011098384857177734, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.7881393432617188e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.7881393432617188e-07]`
- Stage Birkhoff col-error trace: `[1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage routing stability trace: `[0.054933299372593565, 0.7354664484659831, 0.6926225423812866, 0.6450496594111125, 0.5892274220784505, 0.5168147047360738, 0.46502084533373517]`
- Forgetting vs routing stability correlation: `0.38478861932917796`
- Forgetting vs transport gap correlation: `-0.5303328657326148`
- Forgetting vs transport loss correlation: `-0.820808637251608`
- Forgetting vs mean abs excess correlation: `-0.5303319197604468`
- Forgetting vs merge-count correlation: `0.7030946630882342`

## Hyperparameter Adaptation

- Adaptation steps: `6`
- Adaptation strategy: `dual_transport`
- Final hyperparameters: `{'prototype_prior_strength': 1.0, 'prototype_capacity_blend': 0.5, 'prototype_masked_sinkhorn_capacity_bias': 0.0, 'prototype_relocation_strength': 0.75, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 2, 'transport_weight': 0.11674294892259733, 'task_transport_weights': [0.23133850539500245, 0.1160478219922157, 0.09715996718014684, 0.08060859569423186, 0.08059731511992757, 0.09470548815405942, 0.11674294892259733]}`