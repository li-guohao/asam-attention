# Continual Text Benchmark Report

- Dataset: `split_dbpedia`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `7`
- Avg accuracy: `0.5571`
- Avg forgetting: `0.0167`
- Backward transfer: `0.0167`

## Artifacts

- Plot image: `dbpedia_tasklevel_dual_transport_seed42_plots.png`
- Raw JSON: `dbpedia_tasklevel_dual_transport_seed42.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6783`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.09999999999999998, 0.0, 0.07499999999999998, 0.019999999999999997, 0.033333333333333326]`
- Stage transport gap trace: `[0.02266411855816841, 0.03096446767449379, 0.03400234505534172, 0.034020520746707916, 0.03231928497552872, 0.030221736058592796, 0.028158698230981827]`
- Stage transport loss trace: `[0.3713108276327451, 0.0841324453552564, 0.08234323312838872, 0.11124328523874283, 0.09457497298717499, 0.08736761907736461, 0.08277257283528645]`
- Stage merge-count trace: `[0.0, 1.0, 2.0, 3.0, 2.0, 3.0, 1.0]`
- Stage routing stability trace: `[0.08705365657806396, 0.06886690482497215, 0.2775875876347224, 0.45711231231689453, 0.506648580233256, 0.7439905305703481, 0.2915022373199463]`
- Forgetting vs routing stability correlation: `0.1894271919244764`
- Forgetting vs transport gap correlation: `0.4593111908132008`
- Forgetting vs transport loss correlation: `-0.3877637318444226`
- Forgetting vs mean abs excess correlation: `0.45931132145532383`
- Forgetting vs merge-count correlation: `0.19347967609750383`

## Hyperparameter Adaptation

- Adaptation steps: `6`
- Adaptation strategy: `dual_transport`
- Final hyperparameters: `{'prototype_prior_strength': 1.0, 'prototype_capacity_blend': 0.5, 'prototype_relocation_strength': 0.75, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 2, 'transport_weight': 0.0549857310116833, 'task_transport_weights': [0.053467183871758285, 0.059533597074396226, 0.05763581393172702, 0.05233699941149352, 0.05233699941149352, 0.05460379236923122, 0.0549857310116833]}`