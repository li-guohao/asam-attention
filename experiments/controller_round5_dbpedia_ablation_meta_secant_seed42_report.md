# Continual Text Benchmark Report

- Dataset: `split_dbpedia`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `7`
- Avg accuracy: `0.5000`
- Avg forgetting: `0.0667`
- Backward transfer: `-0.0333`

## Artifacts

- Plot image: `controller_round5_dbpedia_ablation_meta_secant_seed42_plots.png`
- Raw JSON: `controller_round5_dbpedia_ablation_meta_secant_seed42.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.0002`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.09999999999999998, 0.033333333333333326, 0.024999999999999994, 0.039999999999999994, 0.06666666666666665]`
- Stage transport gap trace: `[0.02266411855816841, 0.03274178132414818, 0.03471428155899048, 0.039999302476644516, 0.04499145597219467, 0.048542167991399765, 0.05448724702000618]`
- Stage transport loss trace: `[0.3713108276327451, 0.08415660013755162, 0.08134105304876964, 0.10307032366593678, 0.11140148838361104, 0.07687028994162877, 0.07751649618148804]`
- Stage merge-count trace: `[0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]`
- Stage routing stability trace: `[0.08705365657806396, 0.7130622267723083, 0.4692271848519643, 0.4806317190329234, 0.5561873118082682, 0.8430306216080984, 0.6510584851106008]`
- Forgetting vs routing stability correlation: `0.19463343591454335`
- Forgetting vs transport gap correlation: `0.42176199364409966`
- Forgetting vs transport loss correlation: `-0.5042483545223606`
- Forgetting vs mean abs excess correlation: `0.4217618486620493`
- Forgetting vs merge-count correlation: `-0.11775452099599147`

## Hyperparameter Adaptation

- Adaptation steps: `6`
- Adaptation strategy: `meta_secant`
- Final hyperparameters: `{'prototype_prior_strength': 0.7893630248855602, 'prototype_capacity_blend': 0.3009141928449359, 'prototype_relocation_strength': 0.9524795471692826, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 1, 'transport_weight': 0.0}`