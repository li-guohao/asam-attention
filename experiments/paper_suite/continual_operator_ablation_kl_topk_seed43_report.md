# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `kl_topk`
- Tasks: `2`
- Avg accuracy: `0.5000`
- Avg forgetting: `0.0000`
- Backward transfer: `0.0000`

## Artifacts

- Plot image: `continual_operator_ablation_kl_topk_seed43_plots.png`
- Raw JSON: `continual_operator_ablation_kl_topk_seed43.json`
- Resolved prototypes: `4`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.0162`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.01641698181629181, 0.008369918912649155]`
- Stage transport loss trace: `[0.27185256872326136, 0.05722772981971502]`
- Stage merge-count trace: `[2.0, 1.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.010948559010811702, 0.0027899766961733503]`
- Stage Birkhoff gate-factor trace: `[0.5474279505405851, 0.1394988348086675]`
- Stage Birkhoff offdiag-mass trace: `[0.5480173230171204, 0.04727112874388695]`
- Stage Birkhoff applied-offdiag trace: `[0.006, 0.00013188534759725481]`
- Stage Birkhoff gap-delta trace: `[-0.0003428887575864792, -1.1175870895385742e-08]`
- Stage Birkhoff row-error trace: `[1.0132789611816406e-06, 0.0]`
- Stage Birkhoff col-error trace: `[5.960464477539063e-08, 0.0]`
- Stage routing stability trace: `[0.1165397446602583, 0.0048174309195019305]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`