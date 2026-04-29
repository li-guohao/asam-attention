# Continual Text Benchmark Report

- Dataset: `split_dbpedia`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `7`
- Avg accuracy: `0.5286`
- Avg forgetting: `0.0500`
- Backward transfer: `-0.0167`

## Artifacts

- Plot image: `dbpedia_conservative_dual_transport_seed42_plots.png`
- Raw JSON: `dbpedia_conservative_dual_transport_seed42.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6899`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.09999999999999998, 0.0, 0.07499999999999998, 0.019999999999999997, 0.04999999999999999]`
- Stage transport gap trace: `[0.02266411855816841, 0.03096446953713894, 0.03400234505534172, 0.034033093601465225, 0.03232492506504059, 0.03022763505578041, 0.027479728683829308]`
- Stage transport loss trace: `[0.3713108276327451, 0.08413244908054669, 0.08234322567780812, 0.11132263392210007, 0.0954369530081749, 0.09099990377823512, 0.08160431186358134]`
- Stage merge-count trace: `[0.0, 1.0, 2.0, 3.0, 2.0, 2.0, 2.0]`
- Stage routing stability trace: `[0.08705365657806396, 0.06886688868204753, 0.2775874783595403, 0.45619525015354156, 0.5222697059313456, 0.7590165436267853, 0.4554381271203359]`
- Forgetting vs routing stability correlation: `0.21401462912185382`
- Forgetting vs transport gap correlation: `0.3979004804316408`
- Forgetting vs transport loss correlation: `-0.41549022528865687`
- Forgetting vs mean abs excess correlation: `0.39790069759386204`
- Forgetting vs merge-count correlation: `0.3011922004347898`

## Hyperparameter Adaptation

- Adaptation steps: `6`
- Adaptation strategy: `dual_transport`
- Final hyperparameters: `{'prototype_prior_strength': 1.0, 'prototype_capacity_blend': 0.5, 'prototype_relocation_strength': 0.75, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 2, 'transport_weight': 0.05508068835850813, 'task_transport_weights': [0.053373233849416876, 0.0593687853564627, 0.057374485360979104, 0.05289643731808602, 0.05289643731808602, 0.054574750948018044, 0.05508068835850813]}`