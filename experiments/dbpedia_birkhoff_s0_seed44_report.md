# Continual Text Benchmark Report

- Dataset: `split_dbpedia`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `7`
- Avg accuracy: `0.4857`
- Avg forgetting: `0.0500`
- Backward transfer: `-0.0000`

## Artifacts

- Plot image: `dbpedia_birkhoff_s0_seed44_plots.png`
- Raw JSON: `dbpedia_birkhoff_s0_seed44.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6728`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.0, 0.033333333333333326, 0.0, 0.039999999999999994, 0.049999999999999996]`
- Stage transport gap trace: `[0.02266412042081356, 0.03096446953713894, 0.03400234505534172, 0.034824542701244354, 0.0328272320330143, 0.030598944053053856, 0.028770221397280693]`
- Stage transport loss trace: `[0.33453848709662753, 0.057517352203528084, 0.0460686981678009, 0.0752869260807832, 0.07249033078551292, 0.06131768847505251, 0.05962743734320005]`
- Stage merge-count trace: `[0.0, 1.0, 2.0, 3.0, 2.0, 3.0, 1.0]`
- Stage Birkhoff offdiag-mass trace: `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`
- Stage Birkhoff row-error trace: `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`
- Stage Birkhoff col-error trace: `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`
- Stage routing stability trace: `[0.07595475266377132, 0.05862789476911227, 0.25576816002527875, 0.3816535572210948, 0.46839673320452374, 0.7484135429064432, 0.2719714840253194]`
- Forgetting vs routing stability correlation: `0.5083332344167597`
- Forgetting vs transport gap correlation: `0.07467181404808618`
- Forgetting vs transport loss correlation: `-0.32293129408797505`
- Forgetting vs mean abs excess correlation: `0.07467177235251123`
- Forgetting vs merge-count correlation: `0.3898813605230921`