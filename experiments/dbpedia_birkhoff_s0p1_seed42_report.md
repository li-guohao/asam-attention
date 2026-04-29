# Continual Text Benchmark Report

- Dataset: `split_dbpedia`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `7`
- Avg accuracy: `0.4714`
- Avg forgetting: `0.0833`
- Backward transfer: `-0.0500`

## Artifacts

- Plot image: `dbpedia_birkhoff_s0p1_seed42_plots.png`
- Raw JSON: `dbpedia_birkhoff_s0p1_seed42.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6933`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.09999999999999998, 0.033333333333333326, 0.024999999999999994, 0.039999999999999994, 0.08333333333333333]`
- Stage transport gap trace: `[0.022629519924521446, 0.029031461104750633, 0.033151187002658844, 0.03278094902634621, 0.031249897554516792, 0.033198051154613495, 0.03220105543732643]`
- Stage transport loss trace: `[0.3713108276327451, 0.08427676310141881, 0.0815198024113973, 0.08490723371505737, 0.08073335140943527, 0.06889443472027779, 0.06136614580949148]`
- Stage merge-count trace: `[0.0, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0]`
- Stage Birkhoff offdiag-mass trace: `[0.010014851577579975, 0.2750193774700165, 0.26557567715644836, 0.2476506233215332, 0.2600488066673279, 0.24506154656410217, 0.2772359848022461]`
- Stage Birkhoff row-error trace: `[8.96453857421875e-05, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage Birkhoff col-error trace: `[5.960464477539063e-08, 1.1920928955078125e-07, 5.960464477539063e-08, 5.960464477539063e-08, 5.960464477539063e-08, 1.1920928955078125e-07, 5.960464477539063e-08]`
- Stage routing stability trace: `[0.08705365657806396, 0.0698522577683131, 0.026188271741072338, 0.4997028708457947, 0.5379119912783304, 0.863446036974589, 0.48962854345639545]`
- Forgetting vs routing stability correlation: `0.06187041111592321`
- Forgetting vs transport gap correlation: `0.657121242122991`
- Forgetting vs transport loss correlation: `-0.4891665373313796`
- Forgetting vs mean abs excess correlation: `0.6571212096100549`
- Forgetting vs merge-count correlation: `0.11363126128793528`