# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Protocol: `task_incremental_multihead`
- Label mode: `local`
- Head mode: `multi`
- Train task-id mode: `oracle`
- Eval task-id mode: `oracle`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Output classes: `2`
- Avg accuracy: `0.5000`
- Avg forgetting: `0.0625`
- Backward transfer: `-0.0625`

## Artifacts

- Plot image: `r2_agnews_bpe_3ep_meta_secant_seed42_plots.png`
- Raw JSON: `r2_agnews_bpe_3ep_meta_secant_seed42.json`
- Resolved prototypes: `4`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.7779`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0625]`
- Stage transport gap trace: `[0.03247445076704025, 0.042360806837677956]`
- Stage transport loss trace: `[0.16937775909900665, 0.05270648514851928]`
- Stage merge-count trace: `[0.0, 0.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.014675804016490778, 0.0125]`
- Stage Birkhoff gate-factor trace: `[0.7337902008245389, 0.625]`
- Stage Birkhoff offdiag-mass trace: `[0.047854166477918625, 0.04923243634402752]`
- Stage Birkhoff applied-offdiag trace: `[0.0007036116488744043, 0.0006139578018337487]`
- Stage Birkhoff gap-delta trace: `[-9.313225746154785e-10, 0.0]`
- Stage Birkhoff row-error trace: `[1.1920928955078125e-07, 0.0]`
- Stage Birkhoff col-error trace: `[1.1920928955078125e-07, 0.0]`
- Stage routing stability trace: `[0.5098772123456001, 0.5902616828680038]`
- Forgetting vs routing stability correlation: `0.9999999999999999`
- Forgetting vs transport gap correlation: `0.9999999999999999`
- Forgetting vs transport loss correlation: `-0.9999999999999998`
- Forgetting vs mean abs excess correlation: `0.9999999999999999`
- Forgetting vs merge-count correlation: `None`

## Hyperparameter Adaptation

- Adaptation steps: `1`
- Adaptation strategy: `meta_secant`
- Final hyperparameters: `{'prototype_prior_strength': 1.0669889341756973, 'prototype_capacity_blend': 0.4917062496766448, 'prototype_relocation_strength': 0.7668062366545201, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 2, 'transport_weight': 0.05}`