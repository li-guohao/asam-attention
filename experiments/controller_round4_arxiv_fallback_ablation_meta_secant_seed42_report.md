# Continual Text Benchmark Report

- Dataset: `split_arxiv`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `4`
- Avg accuracy: `0.5625`
- Avg forgetting: `0.1667`
- Backward transfer: `-0.1667`

## Artifacts

- Plot image: `controller_round4_arxiv_fallback_ablation_meta_secant_seed42_plots.png`
- Raw JSON: `controller_round4_arxiv_fallback_ablation_meta_secant_seed42.json`
- Resolved prototypes: `8`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.0001`
- Prototype heatmap rows: `4`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.25, 0.25, 0.16666666666666666]`
- Stage transport gap trace: `[0.017980769276618958, 0.03007485345005989, 0.033549964427948, 0.035531334578990936]`
- Stage transport loss trace: `[0.46701505221426487, 0.07625407725572586, 0.13242561370134354, 0.1427086815237999]`
- Stage merge-count trace: `[2.0, 1.0, 1.0, 0.0]`
- Stage routing stability trace: `[0.3672025203704834, 0.49168506264686584, 0.2090659886598587, 0.3791990429162979]`
- Forgetting vs routing stability correlation: `-0.06824551593360102`
- Forgetting vs transport gap correlation: `0.8289344648941828`
- Forgetting vs transport loss correlation: `-0.9639311761850053`
- Forgetting vs mean abs excess correlation: `0.8289344815548221`
- Forgetting vs merge-count correlation: `-0.5773502691896257`

## Hyperparameter Adaptation

- Adaptation steps: `3`
- Adaptation strategy: `meta_secant`
- Final hyperparameters: `{'prototype_prior_strength': 0.7061257767598098, 'prototype_capacity_blend': 0.6881557975481951, 'prototype_relocation_strength': 0.5606474685025635, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 1, 'transport_weight': 0.0}`