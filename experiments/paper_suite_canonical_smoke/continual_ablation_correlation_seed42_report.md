# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.5000`
- Avg forgetting: `0.2500`
- Backward transfer: `-0.2500`

## Artifacts

- Plot image: `continual_ablation_correlation_seed42_plots.png`
- Raw JSON: `continual_ablation_correlation_seed42.json`
- Resolved prototypes: `4`
- Prototype top-k: `1`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.0000`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.25]`
- Stage transport gap trace: `[0.006203977856785059, 0.024006683379411697]`
- Stage transport loss trace: `[0.5289209447801113, 0.07373064756393433]`
- Stage merge-count trace: `[1.0, 1.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.004166667039195697, 0.011608082632635242]`
- Stage Birkhoff gate-factor trace: `[0.20833335195978483, 0.5804041316317621]`
- Stage Birkhoff offdiag-mass trace: `[0.5412599444389343, 0.5168812274932861]`
- Stage Birkhoff applied-offdiag trace: `[0.0022552499701306017, 0.006]`
- Stage Birkhoff gap-delta trace: `[-4.6022702008485794e-05, -0.0004439260810613632]`
- Stage Birkhoff row-error trace: `[5.960464477539063e-08, 5.960464477539063e-08]`
- Stage Birkhoff col-error trace: `[0.0, 0.0]`
- Stage routing stability trace: `[0.6931285858154297, 0.7952829003334045]`
- Forgetting vs routing stability correlation: `1.0`
- Forgetting vs transport gap correlation: `0.9999999999999999`
- Forgetting vs transport loss correlation: `-1.0`
- Forgetting vs mean abs excess correlation: `1.0`
- Forgetting vs merge-count correlation: `None`

## Hyperparameter Adaptation

- Adaptation steps: `1`
- Adaptation strategy: `correlation`
- Final hyperparameters: `{'prototype_prior_strength': 1.0442725688028573, 'prototype_capacity_blend': 0.48662187693602166, 'prototype_masked_sinkhorn_capacity_bias': 0.0, 'prototype_relocation_strength': 0.763533222512342, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 1, 'transport_weight': 0.0, 'task_transport_weights': [0.0, 0.0]}`