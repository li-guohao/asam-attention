# Continual Text Benchmark Report

- Dataset: `split_dbpedia`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `7`
- Avg accuracy: `0.5286`
- Avg forgetting: `0.0833`
- Backward transfer: `-0.0000`

## Artifacts

- Plot image: `dbpedia_birkhoff_s0p05_seed43_plots.png`
- Raw JSON: `dbpedia_birkhoff_s0p05_seed43.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6933`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.0, 0.13333333333333333, 0.12500000000000003, 0.10000000000000002, 0.08333333333333336]`
- Stage transport gap trace: `[0.022647133097052574, 0.029999330639839172, 0.034317389130592346, 0.03720206394791603, 0.036680661141872406, 0.032643742859363556, 0.03733951598405838]`
- Stage transport loss trace: `[0.3345447393755118, 0.046694912016391754, 0.054137200117111206, 0.048055982838074364, 0.06886669869224231, 0.047505141546328865, 0.03788479541738828]`
- Stage merge-count trace: `[0.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0]`
- Stage Birkhoff offdiag-mass trace: `[0.009588897228240967, 0.27515754103660583, 0.28147992491722107, 0.26802247762680054, 0.250546932220459, 0.2723469138145447, 0.2538699805736542]`
- Stage Birkhoff row-error trace: `[0.00010913610458374023, 1.1920928955078125e-07, 5.960464477539063e-08, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage Birkhoff col-error trace: `[5.960464477539063e-08, 5.960464477539063e-08, 5.960464477539063e-08, 5.960464477539063e-08, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage routing stability trace: `[0.07099447896083196, 0.07358943670988083, 0.025784183914462726, 0.018987292423844337, 0.4246586511532466, 0.6967035531997681, 0.5221777856349945]`
- Forgetting vs routing stability correlation: `0.5171238165118605`
- Forgetting vs transport gap correlation: `0.6981557211886922`
- Forgetting vs transport loss correlation: `-0.4374003913770258`
- Forgetting vs mean abs excess correlation: `0.6981558149601708`
- Forgetting vs merge-count correlation: `0.12359606041364168`