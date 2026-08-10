# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.5000`
- Avg forgetting: `0.2500`
- Backward transfer: `-0.2500`

## Artifacts

- Plot image: `continual_operator_ablation_no_relocation_seed42_plots.png`
- Raw JSON: `continual_operator_ablation_no_relocation_seed42.json`
- Resolved prototypes: `4`
- Prototype top-k: `1`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.0000`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.25]`
- Stage transport gap trace: `[0.006203977856785059, 0.02363797463476658]`
- Stage transport loss trace: `[0.5290975086390972, 0.07212680578231812]`
- Stage merge-count trace: `[1.0, 1.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.004166667039195697, 0.011624978866466556]`
- Stage Birkhoff gate-factor trace: `[0.20833335195978483, 0.5812489433233278]`
- Stage Birkhoff offdiag-mass trace: `[0.5412585139274597, 0.5161299705505371]`
- Stage Birkhoff applied-offdiag trace: `[0.0022552440096655912, 0.006]`
- Stage Birkhoff gap-delta trace: `[-4.6022702008485794e-05, -0.0004363376647233963]`
- Stage Birkhoff row-error trace: `[5.960464477539063e-08, 5.960464477539063e-08]`
- Stage Birkhoff col-error trace: `[0.0, 0.0]`
- Stage routing stability trace: `[0.6931285858154297, 0.7952829003334045]`
- Forgetting vs routing stability correlation: `1.0`
- Forgetting vs transport gap correlation: `1.0`
- Forgetting vs transport loss correlation: `-1.0`
- Forgetting vs mean abs excess correlation: `0.9999999999999999`
- Forgetting vs merge-count correlation: `None`