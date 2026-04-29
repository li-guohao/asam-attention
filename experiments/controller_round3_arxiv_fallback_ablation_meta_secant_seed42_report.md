# Continual Text Benchmark Report

- Dataset: `split_arxiv`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `4`
- Avg accuracy: `0.5625`
- Avg forgetting: `0.1667`
- Backward transfer: `-0.1667`

## Artifacts

- Plot image: `controller_round3_arxiv_fallback_ablation_meta_secant_seed42_plots.png`
- Raw JSON: `controller_round3_arxiv_fallback_ablation_meta_secant_seed42.json`
- Resolved prototypes: `8`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.9015`
- Prototype heatmap rows: `4`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.25, 0.25, 0.16666666666666666]`
- Stage transport gap trace: `[0.017980769276618958, 0.03007485345005989, 0.022232143208384514, 0.011519579216837883]`
- Stage transport loss trace: `[0.46701505221426487, 0.07625407725572586, 0.13722586631774902, 0.1454247608780861]`
- Stage merge-count trace: `[2.0, 1.0, 1.0, 2.0]`
- Stage routing stability trace: `[0.3672025203704834, 0.49168506264686584, 0.016498271375894547, 0.16557611525058746]`
- Forgetting vs routing stability correlation: `-0.25288018004947876`
- Forgetting vs transport gap correlation: `0.4950941828280382`
- Forgetting vs transport loss correlation: `-0.9627320374989579`
- Forgetting vs mean abs excess correlation: `0.4950940960092375`
- Forgetting vs merge-count correlation: `-0.8164965809277261`

## Hyperparameter Adaptation

- Adaptation steps: `3`
- Adaptation strategy: `meta_secant`
- Final hyperparameters: `{'prototype_prior_strength': 0.37250242692295077, 'prototype_capacity_blend': 1.0, 'prototype_relocation_strength': 0.0, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 3, 'transport_weight': 1.0}`