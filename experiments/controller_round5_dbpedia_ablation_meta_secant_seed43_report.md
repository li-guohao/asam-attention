# Continual Text Benchmark Report

- Dataset: `split_dbpedia`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `7`
- Avg accuracy: `0.5143`
- Avg forgetting: `0.1000`
- Backward transfer: `-0.0167`

## Artifacts

- Plot image: `controller_round5_dbpedia_ablation_meta_secant_seed43_plots.png`
- Raw JSON: `controller_round5_dbpedia_ablation_meta_secant_seed43.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.0002`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.0, 0.0, 0.2, 0.06000000000000001, 0.09999999999999999]`
- Stage transport gap trace: `[0.02266412042081356, 0.03129138797521591, 0.032543689012527466, 0.03712480515241623, 0.04393627494573593, 0.04303189739584923, 0.04150151461362839]`
- Stage transport loss trace: `[0.3345447393755118, 0.04644830400745074, 0.05460722868641218, 0.05085285007953644, 0.09562158832947414, 0.06972537438074748, 0.045637513200441994]`
- Stage merge-count trace: `[0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]`
- Stage routing stability trace: `[0.07099447896083196, 0.727764626344045, 0.7376309434572855, 0.597135086854299, 0.7137591342131296, 0.40271474917729694, 0.4802017907301585]`
- Forgetting vs routing stability correlation: `0.20847709762413924`
- Forgetting vs transport gap correlation: `0.7228722786023032`
- Forgetting vs transport loss correlation: `-0.16586894171100966`
- Forgetting vs mean abs excess correlation: `0.7228724267045841`
- Forgetting vs merge-count correlation: `0.18639968498479859`

## Hyperparameter Adaptation

- Adaptation steps: `6`
- Adaptation strategy: `meta_secant`
- Final hyperparameters: `{'prototype_prior_strength': 0.585545558778796, 'prototype_capacity_blend': 0.7401987266964531, 'prototype_relocation_strength': 0.5247943261082019, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 1, 'transport_weight': 0.0}`