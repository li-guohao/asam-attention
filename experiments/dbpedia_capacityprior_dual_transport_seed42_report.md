# Continual Text Benchmark Report

- Dataset: `split_dbpedia`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `7`
- Avg accuracy: `0.5429`
- Avg forgetting: `0.0333`
- Backward transfer: `0.0000`

## Artifacts

- Plot image: `dbpedia_capacityprior_dual_transport_seed42_plots.png`
- Raw JSON: `dbpedia_capacityprior_dual_transport_seed42.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6839`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.09999999999999998, 0.0, 0.04999999999999999, 0.0, 0.033333333333333326]`
- Stage transport gap trace: `[0.022245284169912338, 0.030583636835217476, 0.03546183556318283, 0.03464173525571823, 0.03628203645348549, 0.03170120343565941, 0.027226444333791733]`
- Stage transport loss trace: `[0.3713109915455182, 0.08412895848353703, 0.0964970091978709, 0.11168038348356883, 0.11004930237929027, 0.10329084595044453, 0.11992691208918889]`
- Stage merge-count trace: `[0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]`
- Stage routing stability trace: `[0.06434488296508789, 0.06364478170871735, 0.2890919049580892, 0.6057633757591248, 0.46730151772499084, 0.6533596714337667, 0.8436099489529928]`
- Forgetting vs routing stability correlation: `0.03129838118488615`
- Forgetting vs transport gap correlation: `0.4785776544760598`
- Forgetting vs transport loss correlation: `-0.29794593092253224`
- Forgetting vs mean abs excess correlation: `0.4785773941960022`
- Forgetting vs merge-count correlation: `0.026324906324632795`

## Hyperparameter Adaptation

- Adaptation steps: `6`
- Adaptation strategy: `dual_transport`
- Final hyperparameters: `{'prototype_prior_strength': 1.0, 'prototype_capacity_blend': 0.5, 'prototype_relocation_strength': 0.75, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 2, 'transport_weight': 0.05392184005149323, 'task_transport_weights': [0.05354382349827741, 0.057606009239960156, 0.05365478035447105, 0.05240229358237878, 0.05240229358237878, 0.05392184005149323, 0.05392184005149323]}`