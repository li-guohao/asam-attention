# Continual Text Benchmark Report

- Dataset: `split_dbpedia`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `7`
- Avg accuracy: `0.5143`
- Avg forgetting: `0.0500`
- Backward transfer: `0.0167`

## Artifacts

- Plot image: `dbpedia_birkhoff_s0p1_seed44_plots.png`
- Raw JSON: `dbpedia_birkhoff_s0p1_seed44.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6613`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.0, 0.06666666666666667, 0.12499999999999999, 0.07999999999999999, 0.049999999999999996]`
- Stage transport gap trace: `[0.02262861095368862, 0.029034176841378212, 0.033152125775814056, 0.03376097232103348, 0.029773922637104988, 0.031476687639951706, 0.03164710849523544]`
- Stage transport loss trace: `[0.33453848709662753, 0.05758384863535563, 0.057001845290263496, 0.060922433932622276, 0.05072082703312238, 0.040320431192715965, 0.03844778363903364]`
- Stage merge-count trace: `[0.0, 2.0, 1.0, 1.0, 3.0, 3.0, 2.0]`
- Stage Birkhoff offdiag-mass trace: `[0.009832995012402534, 0.27508118748664856, 0.26578304171562195, 0.2528771460056305, 0.2690508961677551, 0.267153799533844, 0.26693734526634216]`
- Stage Birkhoff row-error trace: `[9.334087371826172e-05, 1.1920928955078125e-07, 1.1920928955078125e-07, 5.960464477539063e-08, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage Birkhoff col-error trace: `[1.1920928955078125e-07, 5.960464477539063e-08, 1.1920928955078125e-07, 5.960464477539063e-08, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage routing stability trace: `[0.07595475266377132, 0.05906728282570839, 0.025254936267932255, 0.33701736728350323, 0.7678269743919373, 0.8246466716130575, 0.3857647776603699]`
- Forgetting vs routing stability correlation: `0.925041370433926`
- Forgetting vs transport gap correlation: `0.3254361707428093`
- Forgetting vs transport loss correlation: `-0.440884349821859`
- Forgetting vs mean abs excess correlation: `0.32543622721044824`
- Forgetting vs merge-count correlation: `0.7091898060020739`