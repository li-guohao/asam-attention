# Continual Text Benchmark Report

- Dataset: `split_dbpedia`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `7`
- Avg accuracy: `0.5143`
- Avg forgetting: `0.0833`
- Backward transfer: `-0.0167`

## Artifacts

- Plot image: `controller_round8_dbpedia_sustained_topk_ablation_meta_secant_seed43_plots.png`
- Raw JSON: `controller_round8_dbpedia_sustained_topk_ablation_meta_secant_seed43.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.0002`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.0, 0.033333333333333326, 0.125, 0.1, 0.08333333333333333]`
- Stage transport gap trace: `[0.02266412042081356, 0.03129138797521591, 0.03904566541314125, 0.04315286874771118, 0.048003219068050385, 0.04751884937286377, 0.04727837070822716]`
- Stage transport loss trace: `[0.3345447393755118, 0.0466744564473629, 0.05236586804191271, 0.05329296117027601, 0.08467226972182591, 0.06346750631928444, 0.04323116938273112]`
- Stage merge-count trace: `[0.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0]`
- Stage routing stability trace: `[0.07099447896083196, 0.07201472048958142, 0.26607981820901233, 0.23003400365511575, 0.43955238660176593, 0.5179835657278696, 0.9605444471041361]`
- Forgetting vs routing stability correlation: `0.6988867959126988`
- Forgetting vs transport gap correlation: `0.8145473133771359`
- Forgetting vs transport loss correlation: `-0.31975731661697326`
- Forgetting vs mean abs excess correlation: `0.8145472354915403`
- Forgetting vs merge-count correlation: `-0.07021387563287308`

## Hyperparameter Adaptation

- Adaptation steps: `6`
- Adaptation strategy: `meta_secant`
- Final hyperparameters: `{'prototype_prior_strength': 1.7009625508451278, 'prototype_capacity_blend': 0.38123487536693357, 'prototype_relocation_strength': 0.8750447283090211, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 1, 'transport_weight': 0.4932906055119831}`