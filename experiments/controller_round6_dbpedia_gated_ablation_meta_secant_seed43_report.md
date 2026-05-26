# Continual Text Benchmark Report

- Dataset: `split_dbpedia`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `7`
- Avg accuracy: `0.5286`
- Avg forgetting: `0.0667`
- Backward transfer: `-0.0000`

## Artifacts

- Plot image: `controller_round6_dbpedia_gated_ablation_meta_secant_seed43_plots.png`
- Raw JSON: `controller_round6_dbpedia_gated_ablation_meta_secant_seed43.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.0002`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.0, 0.033333333333333326, 0.125, 0.08, 0.06666666666666667]`
- Stage transport gap trace: `[0.02266412042081356, 0.03129138797521591, 0.03904566541314125, 0.04315286874771118, 0.05170328542590141, 0.0522015206515789, 0.05222263187170029]`
- Stage transport loss trace: `[0.3345447393755118, 0.0466744564473629, 0.05236586804191271, 0.05329296117027601, 0.07541229327519734, 0.05113885551691055, 0.04117498298486074]`
- Stage merge-count trace: `[0.0, 1.0, 2.0, 2.0, 0.0, 1.0, 1.0]`
- Stage routing stability trace: `[0.07099447896083196, 0.07201472048958142, 0.26607981820901233, 0.23003400365511575, 0.7486537098884583, 0.7658271193504333, 0.6751874486605326]`
- Forgetting vs routing stability correlation: `0.91494471310548`
- Forgetting vs transport gap correlation: `0.8346671721231754`
- Forgetting vs transport loss correlation: `-0.33513478955968967`
- Forgetting vs mean abs excess correlation: `0.8346671490563896`
- Forgetting vs merge-count correlation: `-0.38339463711730704`

## Hyperparameter Adaptation

- Adaptation steps: `6`
- Adaptation strategy: `meta_secant`
- Final hyperparameters: `{'prototype_prior_strength': 1.6547102178666497, 'prototype_capacity_blend': 0.42341095048523797, 'prototype_relocation_strength': 0.8344154871484003, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 1, 'transport_weight': 0.49603322624217605}`