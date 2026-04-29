# Continual Text Benchmark Report

- Dataset: `split_dbpedia`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `7`
- Avg accuracy: `0.5143`
- Avg forgetting: `0.0833`
- Backward transfer: `-0.0167`

## Artifacts

- Plot image: `controller_round9_dbpedia_damped_secant_ablation_meta_secant_seed43_plots.png`
- Raw JSON: `controller_round9_dbpedia_damped_secant_ablation_meta_secant_seed43.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.0002`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.0, 0.033333333333333326, 0.125, 0.08, 0.08333333333333333]`
- Stage transport gap trace: `[0.02266412042081356, 0.03129138797521591, 0.03669184818863869, 0.039403971284627914, 0.04101996496319771, 0.04124782234430313, 0.0452253557741642]`
- Stage transport loss trace: `[0.3345447393755118, 0.0466744564473629, 0.05236814171075821, 0.053348896404107414, 0.09691074738899867, 0.05312361692388853, 0.043608419597148895]`
- Stage merge-count trace: `[0.0, 1.0, 2.0, 2.0, 2.0, 3.0, 0.0]`
- Stage routing stability trace: `[0.07099447896083196, 0.07201472048958142, 0.26589590807755786, 0.22946385542551676, 0.47609185179074603, 0.732269416252772, 0.6801419258117676]`
- Forgetting vs routing stability correlation: `0.801868470423256`
- Forgetting vs transport gap correlation: `0.728411752519912`
- Forgetting vs transport loss correlation: `-0.29527432413191895`
- Forgetting vs mean abs excess correlation: `0.7284116861279251`
- Forgetting vs merge-count correlation: `0.28276689414683354`

## Hyperparameter Adaptation

- Adaptation steps: `6`
- Adaptation strategy: `meta_secant`
- Final hyperparameters: `{'prototype_prior_strength': 1.3548936292650438, 'prototype_capacity_blend': 0.4370467077428593, 'prototype_relocation_strength': 0.8258921462910191, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 1, 'transport_weight': 0.4435940245262238}`