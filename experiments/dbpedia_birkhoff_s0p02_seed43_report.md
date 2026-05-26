# Continual Text Benchmark Report

- Dataset: `split_dbpedia`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `7`
- Avg accuracy: `0.6143`
- Avg forgetting: `-0.0167`
- Backward transfer: `0.1167`

## Artifacts

- Plot image: `dbpedia_birkhoff_s0p02_seed43_plots.png`
- Raw JSON: `dbpedia_birkhoff_s0p02_seed43.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6856`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.0, 0.10000000000000002, 0.125, 0.06000000000000001, 0.0]`
- Stage transport gap trace: `[0.022657332941889763, 0.030578330159187317, 0.033937957137823105, 0.03450978919863701, 0.03156920149922371, 0.02639160118997097, 0.024504708126187325]`
- Stage transport loss trace: `[0.3345447393755118, 0.0466947170595328, 0.05410561834772428, 0.05162115270892779, 0.05917501077055931, 0.0598423977692922, 0.04213282962640127]`
- Stage merge-count trace: `[0.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0]`
- Stage Birkhoff offdiag-mass trace: `[0.009588897228240967, 0.27515748143196106, 0.293800413608551, 0.29816851019859314, 0.34905174374580383, 0.31410637497901917, 0.1878613382577896]`
- Stage Birkhoff row-error trace: `[0.00010913610458374023, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 2.384185791015625e-07, 1.1920928955078125e-07, 5.960464477539063e-08]`
- Stage Birkhoff col-error trace: `[5.960464477539063e-08, 1.1920928955078125e-07, 5.960464477539063e-08, 1.1920928955078125e-07, 5.960464477539063e-08, 5.960464477539063e-08, 5.960464477539063e-08]`
- Stage routing stability trace: `[0.07099447896083196, 0.07361268997192383, 0.025508354728420574, 0.018658637379606564, 0.4265528917312622, 0.5510193606217703, 0.36136989792188007]`
- Forgetting vs routing stability correlation: `0.3617463539051282`
- Forgetting vs transport gap correlation: `0.4404140285868315`
- Forgetting vs transport loss correlation: `-0.2945887449681326`
- Forgetting vs mean abs excess correlation: `0.4404137592459271`
- Forgetting vs merge-count correlation: `0.08655166907136455`