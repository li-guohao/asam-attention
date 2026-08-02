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
- Avg forgetting: `0.3750`
- Backward transfer: `-0.2583`

## Artifacts

- Plot image: `r4_dbpedia_longstream_dual_transport_seed44_plots.png`
- Raw JSON: `r4_dbpedia_longstream_dual_transport_seed44.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `1.0611`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.375, 0.4166666666666667, 0.4, 0.39, 0.375]`
- Stage transport gap trace: `[0.04704267717897892, 0.04628353752195835, 0.04566430114209652, 0.04278123378753662, 0.04271011799573898, 0.040451766923069954, 0.04438534565269947]`
- Stage transport loss trace: `[0.17218390901883443, 0.09351337427894274, 0.07651349132259687, 0.07781004657347997, 0.06361986001332601, 0.0830038254459699, 0.06436944007873535]`
- Stage merge-count trace: `[2.0, 3.0, 3.0, 3.0, 4.0, 7.0, 5.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff gate-factor trace: `[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]`
- Stage Birkhoff offdiag-mass trace: `[0.016478140838444233, 0.21394655108451843, 0.2040623500943184, 0.20892807096242905, 0.2081911414861679, 0.22424203157424927, 0.2153634950518608]`
- Stage Birkhoff applied-offdiag trace: `[0.00032956281676888466, 0.004278931021690368, 0.0040812470018863675, 0.004178561419248581, 0.004163822829723358, 0.004484840631484985, 0.004307269901037216]`
- Stage Birkhoff gap-delta trace: `[-1.7307698726654053e-05, -0.000321732833981514, -0.00029286742210388184, -0.00032106228172779083, -0.0003826506435871124, -0.00033867545425891876, -0.0003474578261375427]`
- Stage Birkhoff row-error trace: `[6.985664367675781e-05, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage Birkhoff col-error trace: `[1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage routing stability trace: `[0.05399654358625412, 0.7851994375387827, 0.6566090782483419, 0.6935556610425313, 0.6958750049273174, 0.6547916889190674, 0.5815025692184767]`
- Forgetting vs routing stability correlation: `0.48207531168403506`
- Forgetting vs transport gap correlation: `-0.7464072243457265`
- Forgetting vs transport loss correlation: `-0.7703513106818667`
- Forgetting vs mean abs excess correlation: `-0.7464077748846536`
- Forgetting vs merge-count correlation: `0.5363040589759578`

## Hyperparameter Adaptation

- Adaptation steps: `6`
- Adaptation strategy: `dual_transport`
- Final hyperparameters: `{'prototype_prior_strength': 1.0, 'prototype_capacity_blend': 0.5, 'prototype_masked_sinkhorn_capacity_bias': 0.0, 'prototype_relocation_strength': 0.75, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 2, 'transport_weight': 0.11886622957867639, 'task_transport_weights': [0.25928877439330894, 0.11616395519349633, 0.06307643987432378, 0.09524477277997839, 0.08324697696899859, 0.09617645826195233, 0.11886622957867639]}`