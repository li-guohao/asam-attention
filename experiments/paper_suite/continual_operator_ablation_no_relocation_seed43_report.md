# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.5000`
- Avg forgetting: `0.0625`
- Backward transfer: `-0.0625`

## Artifacts

- Plot image: `continual_operator_ablation_no_relocation_seed43_plots.png`
- Raw JSON: `continual_operator_ablation_no_relocation_seed43.json`
- Resolved prototypes: `4`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6790`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0625]`
- Stage transport gap trace: `[0.000533288752194494, 0.020739439874887466]`
- Stage transport loss trace: `[0.27926598861813545, 0.05473058018833399]`
- Stage merge-count trace: `[2.0, 0.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.00035574535528818774, 0.01382629821697871]`
- Stage Birkhoff gate-factor trace: `[0.017787267764409385, 0.6913149108489355]`
- Stage Birkhoff offdiag-mass trace: `[0.5550507307052612, 0.051614195108413696]`
- Stage Birkhoff applied-offdiag trace: `[0.00019745671939771137, 0.0007136332537982516]`
- Stage Birkhoff gap-delta trace: `[-3.292807377874851e-07, -7.450580596923828e-09]`
- Stage Birkhoff row-error trace: `[1.1324882507324219e-06, 0.0]`
- Stage Birkhoff col-error trace: `[0.0, 0.0]`
- Stage routing stability trace: `[0.514634370803833, 0.12533452548086643]`
- Forgetting vs routing stability correlation: `-0.9999999999999998`
- Forgetting vs transport gap correlation: `1.0`
- Forgetting vs transport loss correlation: `-0.9999999999999998`
- Forgetting vs mean abs excess correlation: `1.0`
- Forgetting vs merge-count correlation: `-0.9999999999999999`