# Continual Text Benchmark Report

- Dataset: `split_dbpedia`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `7`
- Avg accuracy: `0.4857`
- Avg forgetting: `0.1000`
- Backward transfer: `-0.0667`

## Artifacts

- Plot image: `controller_round10_dbpedia_dual_transport_ablation_meta_secant_seed42_plots.png`
- Raw JSON: `controller_round10_dbpedia_dual_transport_ablation_meta_secant_seed42.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6781`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.09999999999999998, 0.0, 0.04999999999999999, 0.039999999999999994, 0.09999999999999998]`
- Stage transport gap trace: `[0.02266411855816841, 0.03131822124123573, 0.036753393709659576, 0.03764078766107559, 0.03659600391983986, 0.03546641021966934, 0.03719903156161308]`
- Stage transport loss trace: `[0.3713108276327451, 0.0841325173775355, 0.08233966181675594, 0.10812631497780482, 0.08223559459050496, 0.07894002149502437, 0.06796231865882874]`
- Stage merge-count trace: `[0.0, 1.0, 2.0, 3.0, 2.0, 3.0, 1.0]`
- Stage routing stability trace: `[0.08705365657806396, 0.06887710839509964, 0.27784063418706256, 0.46142179270585376, 0.3832521637280782, 0.8939800063769022, 0.3750911056995392]`
- Forgetting vs routing stability correlation: `0.19637313136907406`
- Forgetting vs transport gap correlation: `0.5482016034192582`
- Forgetting vs transport loss correlation: `-0.4741457361033767`
- Forgetting vs mean abs excess correlation: `0.5482015049772583`
- Forgetting vs merge-count correlation: `0.07628394235874981`

## Hyperparameter Adaptation

- Adaptation steps: `6`
- Adaptation strategy: `meta_secant`
- Final hyperparameters: `{'prototype_prior_strength': 1.1568737254508579, 'prototype_capacity_blend': 0.48597775798541026, 'prototype_relocation_strength': 0.8053259054368186, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 2, 'transport_weight': 0.382540471461791}`