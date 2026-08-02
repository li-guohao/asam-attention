# Continual Text Benchmark Report

- Dataset: `split_dbpedia`
- Protocol: `class_incremental_singlehead`
- Label mode: `global`
- Head mode: `single`
- Train task-id mode: `oracle`
- Eval task-id mode: `none`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `7`
- Output classes: `14`
- Avg accuracy: `0.0500`
- Avg forgetting: `0.3250`
- Backward transfer: `-0.2000`

## Artifacts

- Plot image: `r4_dbpedia_longstream_meta_secant_seed43_plots.png`
- Raw JSON: `r4_dbpedia_longstream_meta_secant_seed43.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6933`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.35, 0.36666666666666664, 0.2875, 0.36, 0.325]`
- Stage transport gap trace: `[0.04676903411746025, 0.05052451230585575, 0.047469235956668854, 0.04813351668417454, 0.06536851078271866, 0.0736638717353344, 0.06839605048298836]`
- Stage transport loss trace: `[0.20554628918568293, 0.1272031421462695, 0.09216864009698232, 0.07572786460320155, 0.049622234205404916, 0.05368704224626223, 0.030354768969118595]`
- Stage merge-count trace: `[2.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff gate-factor trace: `[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]`
- Stage Birkhoff offdiag-mass trace: `[0.015963700134307146, 0.16920564323663712, 0.17846955358982086, 0.1957334503531456, 0.2144082933664322, 0.2007376328110695, 0.2208562195301056]`
- Stage Birkhoff applied-offdiag trace: `[0.0003192740026861429, 0.003384112864732742, 0.0035693910717964177, 0.003914669007062912, 0.004288165867328644, 0.00401475265622139, 0.004417124390602111]`
- Stage Birkhoff gap-delta trace: `[-1.52587890625e-05, -0.000268617644906044, -0.0002531353384256363, -0.0002828184515237808, -0.0006617680191993713, -0.00055704265832901, -0.0006321761757135391]`
- Stage Birkhoff row-error trace: `[0.00010842084884643555, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 2.384185791015625e-07, 1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage Birkhoff col-error trace: `[1.1920928955078125e-07, 1.1920928955078125e-07, 5.960464477539063e-08, 5.960464477539063e-08, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage routing stability trace: `[0.08344089984893799, 0.6314701755841573, 0.4992563784122467, 0.5227755784988404, 0.7109742561976115, 0.5259996632734935, 0.5308608785271645]`
- Forgetting vs routing stability correlation: `0.4365167116355134`
- Forgetting vs transport gap correlation: `0.4516537116762871`
- Forgetting vs transport loss correlation: `-0.8279137643207617`
- Forgetting vs mean abs excess correlation: `0.45165382281071215`
- Forgetting vs merge-count correlation: `0.4491183650065698`

## Hyperparameter Adaptation

- Adaptation steps: `6`
- Adaptation strategy: `meta_secant`
- Final hyperparameters: `{'prototype_prior_strength': 1.7029518886627721, 'prototype_capacity_blend': 0.34365363888190004, 'prototype_masked_sinkhorn_capacity_bias': 0.0, 'prototype_relocation_strength': 0.9562077091312492, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 1, 'transport_weight': 0.8707005904548033}`