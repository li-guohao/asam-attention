# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `masked_sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.5000`
- Avg forgetting: `0.2500`
- Backward transfer: `-0.2500`

## Artifacts

- Plot image: `continual_operator_ablation_masked_sinkhorn_topk_seed42_plots.png`
- Raw JSON: `continual_operator_ablation_masked_sinkhorn_topk_seed42.json`
- Resolved prototypes: `4`
- Prototype top-k: `1`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.0000`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.25]`
- Stage transport gap trace: `[0.0, 0.0]`
- Stage transport loss trace: `[0.5176981277763844, 0.06059965491294861]`
- Stage merge-count trace: `[1.0, 1.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.0, 0.0]`
- Stage Birkhoff gate-factor trace: `[0.0, 0.0]`
- Stage Birkhoff offdiag-mass trace: `[0.000985347549431026, 0.0007662258576601744]`
- Stage Birkhoff applied-offdiag trace: `[0.0, 0.0]`
- Stage Birkhoff gap-delta trace: `[0.0, 0.0]`
- Stage Birkhoff row-error trace: `[2.980232238769531e-07, 1.1920928955078125e-07]`
- Stage Birkhoff col-error trace: `[0.0, 0.0]`
- Stage routing stability trace: `[0.3564591705799103, 0.013312571682035923]`
- Forgetting vs routing stability correlation: `-1.0`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `-0.9999999999999998`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`