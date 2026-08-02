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
- Avg accuracy: `0.0393`
- Avg forgetting: `0.3417`
- Backward transfer: `-0.2000`

## Artifacts

- Plot image: `r4_dbpedia_longstream_dual_transport_seed43_plots.png`
- Raw JSON: `r4_dbpedia_longstream_dual_transport_seed43.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `1.3095`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.35, 0.3499999999999999, 0.33749999999999997, 0.36, 0.3416666666666666]`
- Stage transport gap trace: `[0.04676903411746025, 0.04959182254970074, 0.04688389599323273, 0.04679172858595848, 0.0444782730191946, 0.04594971239566803, 0.0441288985311985]`
- Stage transport loss trace: `[0.2055462881922722, 0.12813258270422617, 0.09268421481053034, 0.08396069606145223, 0.07400554244716963, 0.08650594602028529, 0.06970142511030038]`
- Stage merge-count trace: `[2.0, 1.0, 3.0, 4.0, 4.0, 5.0, 5.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff gate-factor trace: `[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]`
- Stage Birkhoff offdiag-mass trace: `[0.01596370106562972, 0.16933489590883255, 0.18882235139608383, 0.19002872705459595, 0.18698328733444214, 0.1874898299574852, 0.19260365515947342]`
- Stage Birkhoff applied-offdiag trace: `[0.0003192740213125944, 0.003386697918176651, 0.0037764470279216767, 0.003800574541091919, 0.0037396657466888432, 0.003749796599149704, 0.0038520731031894684]`
- Stage Birkhoff gap-delta trace: `[-1.526065170764923e-05, -0.0002648700028657913, -0.00028074532747268677, -0.0003111511468887329, -0.0002832990139722824, -0.00028850138187408447, -0.0002504810690879822]`
- Stage Birkhoff row-error trace: `[0.00010842084884643555, 1.1920928955078125e-07, 1.1920928955078125e-07, 5.960464477539063e-08, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage Birkhoff col-error trace: `[1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage routing stability trace: `[0.0834408663213253, 0.6256438453992208, 0.5039200584093729, 0.642973800500234, 0.6358356416225434, 0.574905784924825, 0.4861713573336601]`
- Forgetting vs routing stability correlation: `0.5288271011816168`
- Forgetting vs transport gap correlation: `-0.6643691958939423`
- Forgetting vs transport loss correlation: `-0.8644299163227408`
- Forgetting vs mean abs excess correlation: `-0.6643682000557398`
- Forgetting vs merge-count correlation: `0.87314802701271`

## Hyperparameter Adaptation

- Adaptation steps: `6`
- Adaptation strategy: `dual_transport`
- Final hyperparameters: `{'prototype_prior_strength': 1.0, 'prototype_capacity_blend': 0.5, 'prototype_masked_sinkhorn_capacity_bias': 0.0, 'prototype_relocation_strength': 0.75, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 2, 'transport_weight': 0.10876997345253901, 'task_transport_weights': [0.22954167345528953, 0.08942516009144058, 0.08053404650328327, 0.08644346980552119, 0.07780371240806341, 0.0888717784516361, 0.10876997345253901]}`