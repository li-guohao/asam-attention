# Continual Text Benchmark Report

- Dataset: `split_dbpedia`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `7`
- Avg accuracy: `0.4857`
- Avg forgetting: `0.0500`
- Backward transfer: `-0.0000`

## Artifacts

- Plot image: `dbpedia_tasklevel_dual_transport_seed44_plots.png`
- Raw JSON: `dbpedia_tasklevel_dual_transport_seed44.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6728`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.0, 0.033333333333333326, 0.0, 0.039999999999999994, 0.049999999999999996]`
- Stage transport gap trace: `[0.02266412042081356, 0.03096446953713894, 0.03400234505534172, 0.034824542701244354, 0.0328272320330143, 0.030598964542150497, 0.028770221397280693]`
- Stage transport loss trace: `[0.33453848709662753, 0.057517352203528084, 0.0460686981678009, 0.0752869260807832, 0.07249033078551292, 0.061314452439546585, 0.05960619325439135]`
- Stage merge-count trace: `[0.0, 1.0, 2.0, 3.0, 2.0, 3.0, 1.0]`
- Stage routing stability trace: `[0.07595475266377132, 0.05862789476911227, 0.25576816002527875, 0.3816535572210948, 0.46839673320452374, 0.7484124004840851, 0.2719711661338806]`
- Forgetting vs routing stability correlation: `0.5083328195247729`
- Forgetting vs transport gap correlation: `0.07467264363796516`
- Forgetting vs transport loss correlation: `-0.3229806670363814`
- Forgetting vs mean abs excess correlation: `0.07467250646973247`
- Forgetting vs merge-count correlation: `0.3898813605230921`

## Hyperparameter Adaptation

- Adaptation steps: `6`
- Adaptation strategy: `dual_transport`
- Final hyperparameters: `{'prototype_prior_strength': 1.0, 'prototype_capacity_blend': 0.5, 'prototype_relocation_strength': 0.75, 'prototype_merge_threshold': 0.9, 'prototype_merge_usage_threshold': 0.1, 'prototype_top_k': 2, 'transport_weight': 0.05204982523140648, 'task_transport_weights': [0.057391000650849094, 0.053271967158393156, 0.05, 0.05000000000000001, 0.05081799178959829, 0.050817991789598296, 0.05204982523140648]}`