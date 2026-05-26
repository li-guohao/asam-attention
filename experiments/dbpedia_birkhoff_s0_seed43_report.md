# Continual Text Benchmark Report

- Dataset: `split_dbpedia`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `7`
- Avg accuracy: `0.5143`
- Avg forgetting: `0.1000`
- Backward transfer: `-0.0167`

## Artifacts

- Plot image: `dbpedia_birkhoff_s0_seed43_plots.png`
- Raw JSON: `dbpedia_birkhoff_s0_seed43.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6920`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.0, 0.0, 0.15, 0.1, 0.09999999999999999]`
- Stage transport gap trace: `[0.02266412042081356, 0.03096446767449379, 0.03400234133005142, 0.03425821289420128, 0.03305686265230179, 0.03176749497652054, 0.028477013111114502]`
- Stage transport loss trace: `[0.3345447393755118, 0.04667459552486738, 0.052381022522846855, 0.05250950405995051, 0.09562518199284871, 0.06687570984164874, 0.0524793341755867]`
- Stage merge-count trace: `[0.0, 1.0, 2.0, 2.0, 2.0, 1.0, 2.0]`
- Stage Birkhoff offdiag-mass trace: `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`
- Stage Birkhoff row-error trace: `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`
- Stage Birkhoff col-error trace: `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`
- Stage routing stability trace: `[0.07099447896083196, 0.0720126082499822, 0.2669214556614558, 0.22052839895089468, 0.3997077097495397, 0.5084754625956217, 0.3801509936650594]`
- Forgetting vs routing stability correlation: `0.8184607880311893`
- Forgetting vs transport gap correlation: `0.14144137983833036`
- Forgetting vs transport loss correlation: `-0.21643782672697487`
- Forgetting vs mean abs excess correlation: `0.14144106466900858`
- Forgetting vs merge-count correlation: `0.3281650616569468`