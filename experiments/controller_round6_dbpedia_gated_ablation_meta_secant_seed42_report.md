# Continual Text Benchmark Report

- Dataset: `split_dbpedia`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `7`
- Avg accuracy: `0.5000`
- Avg forgetting: `0.0833`
- Backward transfer: `-0.0500`

## Artifacts

- Plot image: `controller_round6_dbpedia_gated_ablation_meta_secant_seed42_plots.png`
- Raw JSON: `controller_round6_dbpedia_gated_ablation_meta_secant_seed42.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.0002`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.09999999999999998, 0.0, 0.024999999999999994, 0.039999999999999994, 0.08333333333333331]`
- Stage transport gap trace: `[0.02266411855816841, 0.03131822124123573, 0.03909580782055855, 0.04279690980911255, 0.04511449113488197, 0.048370976001024246, 0.05098041519522667]`
- Stage transport loss trace: `[0.3713108276327451, 0.0841325173775355, 0.0823353777329127, 0.11366267502307892, 0.0704650307695071, 0.06896875922878583, 0.059782529870669045]`
- Stage merge-count trace: `[0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 1.0]`
- Stage routing stability trace: `[0.08705365657806396, 0.06887710839509964, 0.2783011496067047, 0.8165884017944336, 0.8182356158892313, 0.8886727293332418, 0.7270312507947286]`
- Forgetting vs routing stability correlation: `0.15001579546991328`
- Forgetting vs transport gap correlation: `0.5109272855553927`
- Forgetting vs transport loss correlation: `-0.4476599251906543`
- Forgetting vs mean abs excess correlation: `0.510927186789349`
- Forgetting vs merge-count correlation: `0.6421445720561135`

## Hyperparameter Adaptation

- Adaptation steps: `6`
- Adaptation strategy: `meta_secant`
- Final hyperparameters: `{'prototype_prior_strength': 2.086904009868352, 'prototype_capacity_blend': 0.4463720760211424, 'prototype_relocation_strength': 0.8470269178762895, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 1, 'transport_weight': 0.3512563921813465}`