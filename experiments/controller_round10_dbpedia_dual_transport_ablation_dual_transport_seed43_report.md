# Continual Text Benchmark Report

- Dataset: `split_dbpedia`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `7`
- Avg accuracy: `0.5143`
- Avg forgetting: `0.1000`
- Backward transfer: `-0.0167`

## Artifacts

- Plot image: `controller_round10_dbpedia_dual_transport_ablation_dual_transport_seed43_plots.png`
- Raw JSON: `controller_round10_dbpedia_dual_transport_ablation_dual_transport_seed43.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.0002`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.0, 0.0, 0.15, 0.1, 0.09999999999999999]`
- Stage transport gap trace: `[0.02266412042081356, 0.03096446767449379, 0.03400234133005142, 0.03425821289420128, 0.03305686265230179, 0.03176995739340782, 0.031881965696811676]`
- Stage transport loss trace: `[0.3345447393755118, 0.04667459552486738, 0.052381022522846855, 0.05250950405995051, 0.09562518199284871, 0.0666705680390199, 0.03932172432541847]`
- Stage merge-count trace: `[0.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0]`
- Stage routing stability trace: `[0.07099447896083196, 0.0720126082499822, 0.2669214556614558, 0.22052839895089468, 0.3997077097495397, 0.5082377394040426, 0.8612810174624125]`
- Forgetting vs routing stability correlation: `0.7116613278314808`
- Forgetting vs transport gap correlation: `0.2564198741659265`
- Forgetting vs transport loss correlation: `-0.23044914744436995`
- Forgetting vs mean abs excess correlation: `0.2564194807237544`
- Forgetting vs merge-count correlation: `0.17078251276599335`

## Hyperparameter Adaptation

- Adaptation steps: `6`
- Adaptation strategy: `dual_transport`
- Final hyperparameters: `{'prototype_prior_strength': 1.0, 'prototype_capacity_blend': 0.5, 'prototype_relocation_strength': 0.75, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 1, 'transport_weight': 0.1908903212596973}`