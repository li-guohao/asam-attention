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
- Avg accuracy: `0.1429`
- Avg forgetting: `0.3833`
- Backward transfer: `-0.3750`

## Artifacts

- Plot image: `r4_dbpedia_notransport_seed42_plots.png`
- Raw JSON: `r4_dbpedia_notransport_seed42.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.7006`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.5, 0.25, 0.3333333333333333, 0.375, 0.4, 0.3833333333333333]`
- Stage transport gap trace: `[0.04708949476480484, 0.05396921746432781, 0.051689401268959045, 0.05446546524763107, 0.054979028180241585, 0.05227872356772423, 0.05483604036271572]`
- Stage transport loss trace: `[0.09368630306174358, 0.027228332683444025, 0.032191862786809605, 0.04490481441219648, 0.03426420105000337, 0.0514253335694472, 0.07756534467140834]`
- Stage merge-count trace: `[2.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff gate-factor trace: `[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]`
- Stage Birkhoff offdiag-mass trace: `[0.015068068634718657, 0.19360288232564926, 0.18368970602750778, 0.1614215485751629, 0.15363110229372978, 0.14710545539855957, 0.16716713830828667]`
- Stage Birkhoff applied-offdiag trace: `[0.00030136137269437313, 0.0038720576465129854, 0.003673794120550156, 0.0032284309715032576, 0.0030726220458745955, 0.002942109107971191, 0.0033433427661657337]`
- Stage Birkhoff gap-delta trace: `[-1.372210681438446e-05, -0.0003441907465457916, -0.00032593682408332825, -0.0002717766910791397, -0.0002443641424179077, -0.00022412464022636414, -0.00031940266489982605]`
- Stage Birkhoff row-error trace: `[0.0001291036605834961, 1.7881393432617188e-07, 1.1920928955078125e-07, 2.384185791015625e-07, 2.384185791015625e-07, 2.980232238769531e-07, 4.291534423828125e-06]`
- Stage Birkhoff col-error trace: `[1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 5.960464477539063e-08, 1.1920928955078125e-07, 1.1920928955078125e-07, 5.960464477539063e-08]`
- Stage routing stability trace: `[0.03098141650358836, 0.24219002972046536, 0.19267112215360005, 0.22709105213483174, 0.19458525329828263, 0.19445317089557648, 0.24931268145640692]`
- Forgetting vs routing stability correlation: `0.9138234473202483`
- Forgetting vs transport gap correlation: `0.8722840477681695`
- Forgetting vs transport loss correlation: `-0.6624462857767098`
- Forgetting vs mean abs excess correlation: `0.8722842185976354`
- Forgetting vs merge-count correlation: `-0.14499545920068813`