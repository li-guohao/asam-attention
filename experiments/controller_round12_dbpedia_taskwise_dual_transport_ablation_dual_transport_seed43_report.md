# Continual Text Benchmark Report

- Dataset: `split_dbpedia`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `7`
- Avg accuracy: `0.5143`
- Avg forgetting: `0.1000`
- Backward transfer: `-0.0167`

## Artifacts

- Plot image: `controller_round12_dbpedia_taskwise_dual_transport_ablation_dual_transport_seed43_plots.png`
- Raw JSON: `controller_round12_dbpedia_taskwise_dual_transport_ablation_dual_transport_seed43.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6920`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.0, 0.0, 0.15, 0.1, 0.09999999999999999]`
- Stage transport gap trace: `[0.02266412042081356, 0.03096446767449379, 0.034002337604761124, 0.03425821289420128, 0.03305686265230179, 0.03176787495613098, 0.02847723476588726]`
- Stage transport loss trace: `[0.33454474434256554, 0.04667459800839424, 0.052381012588739395, 0.05250952889521917, 0.0956251894434293, 0.0668511800467968, 0.052006080746650696]`
- Stage merge-count trace: `[0.0, 1.0, 2.0, 2.0, 2.0, 1.0, 2.0]`
- Stage routing stability trace: `[0.07099447151025136, 0.07201261942585309, 0.26692145069440204, 0.2205283691485723, 0.399707759420077, 0.5084399183591207, 0.3801455001036326]`
- Forgetting vs routing stability correlation: `0.8184727835665165`
- Forgetting vs transport gap correlation: `0.14146071466629268`
- Forgetting vs transport loss correlation: `-0.2169742700575058`
- Forgetting vs mean abs excess correlation: `0.14146021661142377`
- Forgetting vs merge-count correlation: `0.3281650616569468`

## Hyperparameter Adaptation

- Adaptation steps: `6`
- Adaptation strategy: `dual_transport`
- Final hyperparameters: `{'prototype_prior_strength': 1.0, 'prototype_capacity_blend': 0.5, 'prototype_relocation_strength': 0.75, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 2, 'transport_weight': 0.17414817817819617, 'task_transport_weights': [0.2284126383873323, 0.22841263838733233, 0.05, 0.3534126383873324, 0.05000000000000001, 0.13465115390717983, 0.17414817817819617]}`