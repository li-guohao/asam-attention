# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.5312`
- Avg forgetting: `0.0625`
- Backward transfer: `-0.0625`

## Artifacts

- Plot image: `continual_operator_ablation_no_merge_seed43_plots.png`
- Raw JSON: `continual_operator_ablation_no_merge_seed43.json`
- Resolved prototypes: `4`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6780`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0625]`
- Stage transport gap trace: `[0.0116901695728302, 0.02288799174129963]`
- Stage transport loss trace: `[0.27926598861813545, 0.05621141940355301]`
- Stage merge-count trace: `[0.0, 0.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.007818436870972317, 0.015367568780978523]`
- Stage Birkhoff gate-factor trace: `[0.3909218435486158, 0.7683784390489261]`
- Stage Birkhoff offdiag-mass trace: `[0.2171376347541809, 0.23397493362426758]`
- Stage Birkhoff applied-offdiag trace: `[0.0016976768896378078, 0.0035956258854958167]`
- Stage Birkhoff gap-delta trace: `[-3.748573362827301e-05, -0.00016336143016815186]`
- Stage Birkhoff row-error trace: `[0.0, 0.0]`
- Stage Birkhoff col-error trace: `[0.0, 0.0]`
- Stage routing stability trace: `[0.514634370803833, 0.686998724937439]`
- Forgetting vs routing stability correlation: `0.9999999999999999`
- Forgetting vs transport gap correlation: `1.0`
- Forgetting vs transport loss correlation: `-1.0`
- Forgetting vs mean abs excess correlation: `1.0`
- Forgetting vs merge-count correlation: `None`