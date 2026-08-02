# Continual Text Benchmark Report

- Dataset: `split_dbpedia`
- Protocol: `class_incremental_singlehead`
- Label mode: `global`
- Head mode: `single`
- Train task-id mode: `oracle`
- Eval task-id mode: `none`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `7`
- Output classes: `14`
- Avg accuracy: `0.0464`
- Avg forgetting: `0.3500`
- Backward transfer: `-0.1833`

## Artifacts

- Plot image: `r4_dbpedia_longstream_no_adaptation_seed43_plots.png`
- Raw JSON: `r4_dbpedia_longstream_no_adaptation_seed43.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.9670`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.35, 0.3499999999999999, 0.32499999999999996, 0.37, 0.35000000000000003]`
- Stage transport gap trace: `[0.04676903411746025, 0.049591824412345886, 0.046883903443813324, 0.04736063629388809, 0.048798562958836555, 0.047105977311730385, 0.04763219691812992]`
- Stage transport loss trace: `[0.20554628918568293, 0.12813258667786917, 0.09268421456217765, 0.08545900781949362, 0.06785076757272085, 0.08845563183228175, 0.07524897903203964]`
- Stage merge-count trace: `[2.0, 1.0, 3.0, 4.0, 3.0, 4.0, 3.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff gate-factor trace: `[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]`
- Stage Birkhoff offdiag-mass trace: `[0.015963700134307146, 0.16933492571115494, 0.18882235884666443, 0.18324336409568787, 0.1907409429550171, 0.18383827805519104, 0.1834041029214859]`
- Stage Birkhoff applied-offdiag trace: `[0.0003192740026861429, 0.003386698514223099, 0.0037764471769332884, 0.0036648672819137576, 0.0038148188591003423, 0.0036767655611038208, 0.003668082058429718]`
- Stage Birkhoff gap-delta trace: `[-1.52587890625e-05, -0.00026487186551094055, -0.0002807416021823883, -0.0002602227032184601, -0.00028946809470653534, -0.00023707933723926544, -0.00024322979152202606]`
- Stage Birkhoff row-error trace: `[0.00010842084884643555, 5.960464477539063e-08, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 2.384185791015625e-07]`
- Stage Birkhoff col-error trace: `[1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 5.960464477539063e-08, 1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage routing stability trace: `[0.08344089984893799, 0.6256438553333282, 0.5039200822512309, 0.5424715558687846, 0.5099225342273712, 0.5102116982142131, 0.4976188391447067]`
- Forgetting vs routing stability correlation: `0.4398874938228206`
- Forgetting vs transport gap correlation: `-0.32238633747999484`
- Forgetting vs transport loss correlation: `-0.8543507055775459`
- Forgetting vs mean abs excess correlation: `-0.3223848851116266`
- Forgetting vs merge-count correlation: `0.8847785821772134`