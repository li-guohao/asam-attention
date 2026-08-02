# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.4688`
- Avg forgetting: `-0.1875`
- Backward transfer: `0.1875`

## Artifacts

- Plot image: `continual_operator_ablation_no_transport_seed42_plots.png`
- Raw JSON: `continual_operator_ablation_no_transport_seed42.json`
- Resolved prototypes: `4`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6719`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.018252983689308167, 0.020312227308750153]`
- Stage transport loss trace: `[0.3077769074589014, 0.07837479189038277]`
- Stage merge-count trace: `[2.0, 1.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.01085639506579735, 0.013541491081317268]`
- Stage Birkhoff gate-factor trace: `[0.5428197532898674, 0.6770745540658634]`
- Stage Birkhoff offdiag-mass trace: `[0.5526696443557739, 0.017470350489020348]`
- Stage Birkhoff applied-offdiag trace: `[0.006, 0.0002365745953345558]`
- Stage Birkhoff gap-delta trace: `[-0.000354645773768425, -9.313225746154785e-09]`
- Stage Birkhoff row-error trace: `[8.344650268554688e-07, 1.1086463928222656e-05]`
- Stage Birkhoff col-error trace: `[0.0, 5.960464477539063e-08]`
- Stage routing stability trace: `[0.3627556413412094, 0.1293521337211132]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`