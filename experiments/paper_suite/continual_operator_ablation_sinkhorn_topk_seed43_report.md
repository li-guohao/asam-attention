# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.5000`
- Avg forgetting: `0.0000`
- Backward transfer: `0.0000`

## Artifacts

- Plot image: `continual_operator_ablation_sinkhorn_topk_seed43_plots.png`
- Raw JSON: `continual_operator_ablation_sinkhorn_topk_seed43.json`
- Resolved prototypes: `4`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6909`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.000533288752194494, 0.018613101914525032]`
- Stage transport loss trace: `[0.27926598861813545, 0.05591362901031971]`
- Stage merge-count trace: `[2.0, 0.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.00035574535528818774, 0.012408735851446789]`
- Stage Birkhoff gate-factor trace: `[0.017787267764409385, 0.6204367925723394]`
- Stage Birkhoff offdiag-mass trace: `[0.5550507307052612, 0.04500620812177658]`
- Stage Birkhoff applied-offdiag trace: `[0.00019745671939771137, 0.0005584701482583647]`
- Stage Birkhoff gap-delta trace: `[-3.292807377874851e-07, -1.862645149230957e-09]`
- Stage Birkhoff row-error trace: `[1.1324882507324219e-06, 0.0]`
- Stage Birkhoff col-error trace: `[0.0, 0.0]`
- Stage routing stability trace: `[0.514634370803833, 0.21021737752016634]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`