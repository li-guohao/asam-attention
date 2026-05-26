# Continual Text Benchmark Report

- Dataset: `split_arxiv`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `4`
- Avg accuracy: `0.5625`
- Avg forgetting: `0.1667`
- Backward transfer: `-0.1667`

## Artifacts

- Plot image: `controller_round4_arxiv_fallback_ablation_correlation_seed42_plots.png`
- Raw JSON: `controller_round4_arxiv_fallback_ablation_correlation_seed42.json`
- Resolved prototypes: `8`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6920`
- Prototype heatmap rows: `4`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.25, 0.25, 0.16666666666666666]`
- Stage transport gap trace: `[0.017980769276618958, 0.03340910002589226, 0.040366653352975845, 0.033108167350292206]`
- Stage transport loss trace: `[0.46701505221426487, 0.07488108426332474, 0.21753890812397003, 0.22863269597291946]`
- Stage merge-count trace: `[2.0, 0.0, 1.0, 1.0]`
- Stage routing stability trace: `[0.3672025203704834, 0.02678514551371336, 0.5361483618617058, 1.2160936444997787]`
- Forgetting vs routing stability correlation: `-0.08081529796914395`
- Forgetting vs transport gap correlation: `0.9442422211249514`
- Forgetting vs transport loss correlation: `-0.9305333753265517`
- Forgetting vs mean abs excess correlation: `0.9442422436068431`
- Forgetting vs merge-count correlation: `-0.8660254037844386`

## Hyperparameter Adaptation

- Adaptation steps: `3`
- Adaptation strategy: `correlation`
- Final hyperparameters: `{'prototype_prior_strength': 1.1713225578627486, 'prototype_capacity_blend': 0.24344150296339265, 'prototype_relocation_strength': 0.9112130567897101, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 2, 'transport_weight': 0.05}`