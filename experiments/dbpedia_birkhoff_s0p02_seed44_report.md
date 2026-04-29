# Continual Text Benchmark Report

- Dataset: `split_dbpedia`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `7`
- Avg accuracy: `0.4714`
- Avg forgetting: `0.0833`
- Backward transfer: `-0.0500`

## Artifacts

- Plot image: `dbpedia_birkhoff_s0p02_seed44_plots.png`
- Raw JSON: `dbpedia_birkhoff_s0p02_seed44.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6933`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.0, 0.16666666666666666, 0.09999999999999999, 0.07999999999999999, 0.08333333333333333]`
- Stage transport gap trace: `[0.02265702374279499, 0.03057817928493023, 0.03408271074295044, 0.03425062820315361, 0.03051978535950184, 0.03006276674568653, 0.026022527366876602]`
- Stage transport loss trace: `[0.33453848709662753, 0.057583704590797424, 0.05690311764677366, 0.056428153067827225, 0.05885439986983935, 0.03915772711237272, 0.04666056980689367]`
- Stage merge-count trace: `[0.0, 2.0, 1.0, 1.0, 3.0, 2.0, 2.0]`
- Stage Birkhoff offdiag-mass trace: `[0.009832995012402534, 0.27508118748664856, 0.2744068205356598, 0.2704664170742035, 0.310243159532547, 0.34331974387168884, 0.31348085403442383]`
- Stage Birkhoff row-error trace: `[9.334087371826172e-05, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 5.960464477539063e-08, 5.960464477539063e-08]`
- Stage Birkhoff col-error trace: `[1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 5.960464477539063e-08, 1.7881393432617188e-07]`
- Stage routing stability trace: `[0.07595475266377132, 0.059106639275948204, 0.02448515345652898, 0.33826204140981037, 0.6164098083972931, 0.8204657435417175, 0.5110138853391012]`
- Forgetting vs routing stability correlation: `0.6326919221641653`
- Forgetting vs transport gap correlation: `0.3397272718207074`
- Forgetting vs transport loss correlation: `-0.4285839161661533`
- Forgetting vs mean abs excess correlation: `0.3397272197631495`
- Forgetting vs merge-count correlation: `0.3130245821865503`