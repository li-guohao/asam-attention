# Continual Text Benchmark Report

- Dataset: `split_dbpedia`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `7`
- Avg accuracy: `0.5571`
- Avg forgetting: `0.0167`
- Backward transfer: `0.0167`

## Artifacts

- Plot image: `controller_round11_dbpedia_dual_transport_persistent_topk_ablation_dual_transport_seed42_plots.png`
- Raw JSON: `controller_round11_dbpedia_dual_transport_persistent_topk_ablation_dual_transport_seed42.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6783`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.09999999999999998, 0.0, 0.07499999999999998, 0.019999999999999997, 0.033333333333333326]`
- Stage transport gap trace: `[0.02266411855816841, 0.03096446767449379, 0.03400234505534172, 0.0340232290327549, 0.03232082352042198, 0.030234893783926964, 0.02817694842815399]`
- Stage transport loss trace: `[0.3713108276327451, 0.0841324453552564, 0.08234323312838872, 0.1110679879784584, 0.09353916843732198, 0.08536079525947571, 0.07905204469958942]`
- Stage merge-count trace: `[0.0, 1.0, 2.0, 3.0, 2.0, 3.0, 1.0]`
- Stage routing stability trace: `[0.08705365657806396, 0.06886690482497215, 0.2775875876347224, 0.4568408677975337, 0.5066307584444681, 0.7429041564464569, 0.29028333723545074]`
- Forgetting vs routing stability correlation: `0.1900030220715752`
- Forgetting vs transport gap correlation: `0.45926317605181566`
- Forgetting vs transport loss correlation: `-0.3866094872249685`
- Forgetting vs mean abs excess correlation: `0.4592634148241562`
- Forgetting vs merge-count correlation: `0.19347967609750383`

## Hyperparameter Adaptation

- Adaptation steps: `6`
- Adaptation strategy: `dual_transport`
- Final hyperparameters: `{'prototype_prior_strength': 1.0, 'prototype_capacity_blend': 0.5, 'prototype_relocation_strength': 0.75, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 2, 'transport_weight': 0.1673054807198544}`