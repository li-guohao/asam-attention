# Continual Text Benchmark Report

- Dataset: `split_dbpedia`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `7`
- Avg accuracy: `0.4857`
- Avg forgetting: `0.1000`
- Backward transfer: `-0.0667`

## Artifacts

- Plot image: `controller_round7_dbpedia_damped_topk_ablation_meta_secant_seed42_plots.png`
- Raw JSON: `controller_round7_dbpedia_damped_topk_ablation_meta_secant_seed42.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.0002`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.09999999999999998, 0.0, 0.04999999999999999, 0.039999999999999994, 0.09999999999999998]`
- Stage transport gap trace: `[0.02266411855816841, 0.03131822124123573, 0.03909580782055855, 0.039680931717157364, 0.03771885856986046, 0.03618471696972847, 0.03748925402760506]`
- Stage transport loss trace: `[0.3713108276327451, 0.0841325173775355, 0.0823353777329127, 0.10787977029879887, 0.08199641729394595, 0.0786765751739343, 0.06276903549830119]`
- Stage merge-count trace: `[0.0, 1.0, 2.0, 3.0, 2.0, 3.0, 0.0]`
- Stage routing stability trace: `[0.08705365657806396, 0.06887710839509964, 0.2783011496067047, 0.4653339833021164, 0.3829480856657028, 0.8932474255561829, 0.7651679118474325]`
- Forgetting vs routing stability correlation: `0.4387551001729731`
- Forgetting vs transport gap correlation: `0.5386729752590478`
- Forgetting vs transport loss correlation: `-0.4819593431689542`
- Forgetting vs mean abs excess correlation: `0.5386729064059506`
- Forgetting vs merge-count correlation: `-0.10423183193213746`

## Hyperparameter Adaptation

- Adaptation steps: `6`
- Adaptation strategy: `meta_secant`
- Final hyperparameters: `{'prototype_prior_strength': 0.9887293457743151, 'prototype_capacity_blend': 0.5688242305505649, 'prototype_relocation_strength': 0.6941015781484976, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 1, 'transport_weight': 0.39920153477197595}`