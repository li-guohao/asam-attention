# Continual Text Benchmark Report

- Dataset: `split_dbpedia`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `7`
- Avg accuracy: `0.5000`
- Avg forgetting: `0.0833`
- Backward transfer: `-0.0167`

## Artifacts

- Plot image: `dbpedia_birkhoff_s0p02_seed42_plots.png`
- Raw JSON: `dbpedia_birkhoff_s0p02_seed42.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6932`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.09999999999999998, 0.033333333333333326, 0.04999999999999999, 0.019999999999999997, 0.08333333333333333]`
- Stage transport gap trace: `[0.022657202556729317, 0.030577633529901505, 0.034082405269145966, 0.03056979551911354, 0.028172696009278297, 0.02738732099533081, 0.024154961109161377]`
- Stage transport loss trace: `[0.3713108276327451, 0.08427647749582927, 0.08137617508570354, 0.07828308021028836, 0.07468150059382121, 0.06375292936960857, 0.05702219406763712]`
- Stage merge-count trace: `[0.0, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0]`
- Stage Birkhoff offdiag-mass trace: `[0.010014851577579975, 0.2750193774700165, 0.27419301867485046, 0.2813500463962555, 0.35844144225120544, 0.3712708055973053, 0.18811611831188202]`
- Stage Birkhoff row-error trace: `[8.96453857421875e-05, 0.0, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage Birkhoff col-error trace: `[5.960464477539063e-08, 5.960464477539063e-08, 1.1920928955078125e-07, 5.960464477539063e-08, 5.960464477539063e-08, 1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage routing stability trace: `[0.08705365657806396, 0.06991225481033325, 0.02544269027809302, 0.7259848117828369, 0.5432146886984507, 0.8535179793834686, 0.4932941297690074]`
- Forgetting vs routing stability correlation: `-0.03163030017732095`
- Forgetting vs transport gap correlation: `0.32756996682622136`
- Forgetting vs transport loss correlation: `-0.47730459874978903`
- Forgetting vs mean abs excess correlation: `0.3275696764609944`
- Forgetting vs merge-count correlation: `0.12354240390414761`