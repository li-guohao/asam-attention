# Continual Text Benchmark Report

- Dataset: `split_dbpedia`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `7`
- Avg accuracy: `0.4857`
- Avg forgetting: `0.0500`
- Backward transfer: `-0.0000`

## Artifacts

- Plot image: `controller_round12_dbpedia_taskwise_dual_transport_ablation_dual_transport_seed44_plots.png`
- Raw JSON: `controller_round12_dbpedia_taskwise_dual_transport_ablation_dual_transport_seed44.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6728`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.0, 0.033333333333333326, 0.0, 0.039999999999999994, 0.049999999999999996]`
- Stage transport gap trace: `[0.02266411855816841, 0.03096446767449379, 0.03400234133005142, 0.034824538975954056, 0.032827228307724, 0.030600642785429955, 0.02877122163772583]`
- Stage transport loss trace: `[0.3345384808878104, 0.05751736586292585, 0.046068708101908364, 0.0752869447072347, 0.0724903258184592, 0.06125239903728167, 0.05919246996442477]`
- Stage merge-count trace: `[0.0, 1.0, 2.0, 3.0, 2.0, 3.0, 1.0]`
- Stage routing stability trace: `[0.07595474769671758, 0.05862787986795107, 0.25576816002527875, 0.3816535572210948, 0.46839672327041626, 0.7482935984929403, 0.27195194860299426]`
- Forgetting vs routing stability correlation: `0.5083045927889877`
- Forgetting vs transport gap correlation: `0.07480078974262307`
- Forgetting vs transport loss correlation: `-0.32394017481540616`
- Forgetting vs mean abs excess correlation: `0.07480049319580719`
- Forgetting vs merge-count correlation: `0.3898813605230921`

## Hyperparameter Adaptation

- Adaptation steps: `6`
- Adaptation strategy: `dual_transport`
- Final hyperparameters: `{'prototype_prior_strength': 1.0, 'prototype_capacity_blend': 0.5, 'prototype_relocation_strength': 0.75, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 2, 'transport_weight': 0.08868792223640613, 'task_transport_weights': [0.18648163022783892, 0.11376393546039858, 0.05, 0.05000000000000001, 0.06594098386509964, 0.06594098386509964, 0.08868792223640613]}`