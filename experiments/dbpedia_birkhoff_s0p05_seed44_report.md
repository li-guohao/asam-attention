# Continual Text Benchmark Report

- Dataset: `split_dbpedia`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `7`
- Avg accuracy: `0.4714`
- Avg forgetting: `0.1000`
- Backward transfer: `-0.0333`

## Artifacts

- Plot image: `dbpedia_birkhoff_s0p05_seed44_plots.png`
- Raw JSON: `dbpedia_birkhoff_s0p05_seed44.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6933`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.0, 0.06666666666666667, 0.12499999999999999, 0.07999999999999999, 0.10000000000000002]`
- Stage transport gap trace: `[0.02264636941254139, 0.029998958110809326, 0.034624453634023666, 0.03556745499372482, 0.03145349770784378, 0.03362112119793892, 0.03364742919802666]`
- Stage transport loss trace: `[0.33453848709662753, 0.057583779096603394, 0.056939929723739624, 0.06086370597283045, 0.05069756259520849, 0.0403339775900046, 0.03582401697834333]`
- Stage merge-count trace: `[0.0, 2.0, 1.0, 1.0, 3.0, 2.0, 2.0]`
- Stage Birkhoff offdiag-mass trace: `[0.009832995012402534, 0.27508121728897095, 0.265956312417984, 0.2529540956020355, 0.2708924114704132, 0.2602445185184479, 0.2573549747467041]`
- Stage Birkhoff row-error trace: `[9.334087371826172e-05, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage Birkhoff col-error trace: `[1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 5.960464477539063e-08]`
- Stage routing stability trace: `[0.07595475266377132, 0.059091909478108086, 0.02476753108203411, 0.33657770852247876, 0.7802802721659342, 0.8309703270594279, 0.5173498193422953]`
- Forgetting vs routing stability correlation: `0.9091761517450174`
- Forgetting vs transport gap correlation: `0.4261758577545923`
- Forgetting vs transport loss correlation: `-0.48610753911670807`
- Forgetting vs mean abs excess correlation: `0.42617583109728574`
- Forgetting vs merge-count correlation: `0.7037101306822643`