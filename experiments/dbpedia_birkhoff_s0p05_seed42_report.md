# Continual Text Benchmark Report

- Dataset: `split_dbpedia`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `7`
- Avg accuracy: `0.4714`
- Avg forgetting: `0.0833`
- Backward transfer: `-0.0500`

## Artifacts

- Plot image: `dbpedia_birkhoff_s0p05_seed42_plots.png`
- Raw JSON: `dbpedia_birkhoff_s0p05_seed42.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6933`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.09999999999999998, 0.033333333333333326, 0.024999999999999994, 0.039999999999999994, 0.08333333333333333]`
- Stage transport gap trace: `[0.022646820172667503, 0.029997605830430984, 0.03462393209338188, 0.03448040038347244, 0.03278375789523125, 0.03508633002638817, 0.0343426875770092]`
- Stage transport loss trace: `[0.3713108276327451, 0.08427660788098972, 0.08142975469430287, 0.08486562718947728, 0.08077021688222885, 0.06894843777020772, 0.06137432033816973]`
- Stage merge-count trace: `[0.0, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0]`
- Stage Birkhoff offdiag-mass trace: `[0.010014851577579975, 0.2750193774700165, 0.26574426889419556, 0.24760831892490387, 0.26294153928756714, 0.24514524638652802, 0.24661259353160858]`
- Stage Birkhoff row-error trace: `[8.96453857421875e-05, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage Birkhoff col-error trace: `[5.960464477539063e-08, 5.960464477539063e-08, 5.960464477539063e-08, 1.1920928955078125e-07, 5.960464477539063e-08, 1.1920928955078125e-07, 5.960464477539063e-08]`
- Stage routing stability trace: `[0.08705365657806396, 0.06988975033164024, 0.025716178740064304, 0.5006698270638784, 0.537688821554184, 0.8691337704658508, 0.4872369170188904]`
- Forgetting vs routing stability correlation: `0.05970004336345336`
- Forgetting vs transport gap correlation: `0.6605623287334738`
- Forgetting vs transport loss correlation: `-0.48937047537266526`
- Forgetting vs mean abs excess correlation: `0.6605625471104076`
- Forgetting vs merge-count correlation: `0.11363126128793528`