# Continual Text Benchmark Report

- Dataset: `split_dbpedia`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `7`
- Avg accuracy: `0.5571`
- Avg forgetting: `0.0167`
- Backward transfer: `0.0167`

## Artifacts

- Plot image: `controller_round12_dbpedia_taskwise_dual_transport_ablation_dual_transport_seed42_plots.png`
- Raw JSON: `controller_round12_dbpedia_taskwise_dual_transport_ablation_dual_transport_seed42.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6783`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.09999999999999998, 0.0, 0.07499999999999998, 0.019999999999999997, 0.033333333333333326]`
- Stage transport gap trace: `[0.02266411855816841, 0.03096446767449379, 0.03400234133005142, 0.03402147814631462, 0.03231983259320259, 0.030228544026613235, 0.02816861681640148]`
- Stage transport loss trace: `[0.3713108201821645, 0.08413244287172954, 0.08234321574370067, 0.11119864135980606, 0.09418392926454544, 0.08631879339615504, 0.08044323821862538]`
- Stage merge-count trace: `[0.0, 1.0, 2.0, 3.0, 2.0, 3.0, 1.0]`
- Stage routing stability trace: `[0.08705366154511769, 0.06886693835258484, 0.27758756776650745, 0.45701650778452557, 0.5066401362419128, 0.7434226969877878, 0.29081903398036957]`
- Forgetting vs routing stability correlation: `0.18969797557573365`
- Forgetting vs transport gap correlation: `0.4593061134974201`
- Forgetting vs transport loss correlation: `-0.3869179700046962`
- Forgetting vs mean abs excess correlation: `0.4593063499640211`
- Forgetting vs merge-count correlation: `0.19347967609750383`

## Hyperparameter Adaptation

- Adaptation steps: `6`
- Adaptation strategy: `dual_transport`
- Final hyperparameters: `{'prototype_prior_strength': 1.0, 'prototype_capacity_blend': 0.5, 'prototype_relocation_strength': 0.75, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 2, 'transport_weight': 0.14094872071200773, 'task_transport_weights': [0.114543194634219, 0.2199245820442835, 0.1908129702322185, 0.09302879642281266, 0.09302879642281266, 0.13435398451570008, 0.14094872071200773]}`