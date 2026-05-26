# Continual Text Benchmark Report

- Dataset: `split_dbpedia`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `7`
- Avg accuracy: `0.5571`
- Avg forgetting: `0.0167`
- Backward transfer: `0.0167`

## Artifacts

- Plot image: `controller_round9_dbpedia_damped_secant_ablation_no_adaptation_seed42_plots.png`
- Raw JSON: `controller_round9_dbpedia_damped_secant_ablation_no_adaptation_seed42.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6783`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.09999999999999998, 0.0, 0.07499999999999998, 0.019999999999999997, 0.033333333333333326]`
- Stage transport gap trace: `[0.02266411855816841, 0.03096446767449379, 0.03400234505534172, 0.03402048721909523, 0.03231927007436752, 0.030221164226531982, 0.02815782092511654]`
- Stage transport loss trace: `[0.3713108276327451, 0.0841324453552564, 0.08234323312838872, 0.1112477034330368, 0.09459826350212097, 0.08744602153698604, 0.08294343948364258]`
- Stage merge-count trace: `[0.0, 1.0, 2.0, 3.0, 2.0, 3.0, 1.0]`
- Stage routing stability trace: `[0.08705365657806396, 0.06886690482497215, 0.2775875876347224, 0.4571160425742467, 0.5066493252913157, 0.7440377573172251, 0.2915623386700948]`
- Forgetting vs routing stability correlation: `0.189407471956671`
- Forgetting vs transport gap correlation: `0.459309726672153`
- Forgetting vs transport loss correlation: `-0.387835457178221`
- Forgetting vs mean abs excess correlation: `0.45930985254892387`
- Forgetting vs merge-count correlation: `0.19347967609750383`