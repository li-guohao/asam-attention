# Continual Text Benchmark Report

- Dataset: `split_dbpedia`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `7`
- Avg accuracy: `0.5571`
- Avg forgetting: `-0.0000`
- Backward transfer: `0.0167`

## Artifacts

- Plot image: `controller_round8_dbpedia_sustained_topk_ablation_correlation_seed42_plots.png`
- Raw JSON: `controller_round8_dbpedia_sustained_topk_ablation_correlation_seed42.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6863`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.09999999999999998, 0.0, 0.04999999999999999, 0.039999999999999994, 0.033333333333333326]`
- Stage transport gap trace: `[0.02266411855816841, 0.031252045184373856, 0.03524533286690712, 0.03716359660029411, 0.04121944680809975, 0.04177752882242203, 0.04643313214182854]`
- Stage transport loss trace: `[0.3713108276327451, 0.08413246770699818, 0.08234247068564098, 0.10301996022462845, 0.08478686213493347, 0.09346873313188553, 0.0739149699608485]`
- Stage merge-count trace: `[0.0, 1.0, 2.0, 3.0, 2.0, 3.0, 2.0]`
- Stage routing stability trace: `[0.08705365657806396, 0.0688751923541228, 0.2775864948829015, 0.5595849901437759, 0.3818548421065013, 0.9336614807446798, 0.46655137340227765]`
- Forgetting vs routing stability correlation: `0.17002003235139576`
- Forgetting vs transport gap correlation: `0.372772727670589`
- Forgetting vs transport loss correlation: `-0.4130550702262265`
- Forgetting vs mean abs excess correlation: `0.3727725311418576`
- Forgetting vs merge-count correlation: `0.30537034398149926`

## Hyperparameter Adaptation

- Adaptation steps: `6`
- Adaptation strategy: `correlation`
- Final hyperparameters: `{'prototype_prior_strength': 1.394087742703059, 'prototype_capacity_blend': 0.26883084579080185, 'prototype_relocation_strength': 0.8861723994309266, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 2, 'transport_weight': 0.05}`