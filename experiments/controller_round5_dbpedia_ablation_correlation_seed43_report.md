# Continual Text Benchmark Report

- Dataset: `split_dbpedia`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `7`
- Avg accuracy: `0.5143`
- Avg forgetting: `0.0833`
- Backward transfer: `-0.0167`

## Artifacts

- Plot image: `controller_round5_dbpedia_ablation_correlation_seed43_plots.png`
- Raw JSON: `controller_round5_dbpedia_ablation_correlation_seed43.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6879`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.0, 0.033333333333333326, 0.125, 0.08, 0.08333333333333333]`
- Stage transport gap trace: `[0.02266412042081356, 0.031225210055708885, 0.0351775698363781, 0.03755251690745354, 0.03889232128858566, 0.04137597233057022, 0.04681863263249397]`
- Stage transport loss trace: `[0.3345447393755118, 0.04667448252439499, 0.052370380610227585, 0.05337297543883324, 0.0968744233250618, 0.05567557364702225, 0.05791558946172396]`
- Stage merge-count trace: `[0.0, 1.0, 2.0, 2.0, 2.0, 3.0, 1.0]`
- Stage routing stability trace: `[0.07099447896083196, 0.07201427842179935, 0.26578818758328754, 0.2293295611937841, 0.4706011513868968, 0.7379644910494486, 0.2510276287794113]`
- Forgetting vs routing stability correlation: `0.6999877543768117`
- Forgetting vs transport gap correlation: `0.7124592616882681`
- Forgetting vs transport loss correlation: `-0.27935426185098716`
- Forgetting vs mean abs excess correlation: `0.7124590632809603`
- Forgetting vs merge-count correlation: `0.4549710158952626`

## Hyperparameter Adaptation

- Adaptation steps: `6`
- Adaptation strategy: `correlation`
- Final hyperparameters: `{'prototype_prior_strength': 1.4990201333157183, 'prototype_capacity_blend': 0.2780372942663165, 'prototype_relocation_strength': 0.8819835321589954, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 2, 'transport_weight': 0.05}`