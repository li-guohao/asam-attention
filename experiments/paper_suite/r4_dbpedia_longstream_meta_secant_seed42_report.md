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
- Avg accuracy: `0.0464`
- Avg forgetting: `0.3583`
- Backward transfer: `-0.1500`

## Artifacts

- Plot image: `r4_dbpedia_longstream_meta_secant_seed42_plots.png`
- Raw JSON: `r4_dbpedia_longstream_meta_secant_seed42.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6933`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.35, 0.39999999999999997, 0.425, 0.4, 0.375]`
- Stage transport gap trace: `[0.047090066596865654, 0.04717163369059563, 0.04535434953868389, 0.04757899418473244, 0.0641109012067318, 0.06799613684415817, 0.0747663639485836]`
- Stage transport loss trace: `[0.18818894972403843, 0.10635527471701305, 0.0804415004948775, 0.06570731153090795, 0.043148579200108846, 0.044726012150446574, 0.02534673665650189]`
- Stage merge-count trace: `[2.0, 3.0, 3.0, 4.0, 3.0, 3.0, 3.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff gate-factor trace: `[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]`
- Stage Birkhoff offdiag-mass trace: `[0.014787494204938412, 0.19634681195020676, 0.17315462976694107, 0.1784345954656601, 0.19324874877929688, 0.2004801332950592, 0.21276380121707916]`
- Stage Birkhoff applied-offdiag trace: `[0.00029574988409876825, 0.003926936239004135, 0.0034630925953388217, 0.0035686919093132017, 0.0038649749755859373, 0.004009602665901184, 0.004255276024341583]`
- Stage Birkhoff gap-delta trace: `[-1.3152137398719788e-05, -0.0002729613333940506, -0.0002193395048379898, -0.00027189403772354126, -0.0005505681037902832, -0.0005304962396621704, -0.000598657876253128]`
- Stage Birkhoff row-error trace: `[0.00011098384857177734, 5.960464477539063e-08, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.7881393432617188e-07, 1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage Birkhoff col-error trace: `[1.1920928955078125e-07, 5.960464477539063e-08, 5.960464477539063e-08, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 5.960464477539063e-08]`
- Stage routing stability trace: `[0.054933303222060206, 0.7389023184776307, 0.6743643323580424, 0.6115217208862305, 0.814191210269928, 0.5348283966382344, 0.49931369721889496]`
- Forgetting vs routing stability correlation: `0.4665865393465218`
- Forgetting vs transport gap correlation: `0.5394837799882112`
- Forgetting vs transport loss correlation: `-0.8547457216325256`
- Forgetting vs mean abs excess correlation: `0.5394838230237088`
- Forgetting vs merge-count correlation: `0.6022898276056528`

## Hyperparameter Adaptation

- Adaptation steps: `6`
- Adaptation strategy: `meta_secant`
- Final hyperparameters: `{'prototype_prior_strength': 1.8772927421525825, 'prototype_capacity_blend': 0.27320054456026316, 'prototype_masked_sinkhorn_capacity_bias': 0.0, 'prototype_relocation_strength': 1.0, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 1, 'transport_weight': 1.0}`