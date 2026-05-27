# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Dataset source (train): `huggingface`
- Dataset source (val): `huggingface`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.5000`
- Avg forgetting: `0.2500`
- Backward transfer: `-0.2500`

## Artifacts

- Plot image: `continual_operator_ablation_no_merge_seed42_plots.png`
- Raw JSON: `continual_operator_ablation_no_merge_seed42.json`
- Resolved prototypes: `4`
- Prototype top-k: `1`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.0000`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.25]`
- Stage transport gap trace: `[0.006203612312674522, 0.01737036369740963]`
- Stage transport loss trace: `[0.5290975086390972, 0.09124793484807014]`
- Stage merge-count trace: `[0.0, 0.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.004166667039195697, 0.011064872367919473]`
- Stage Birkhoff gate-factor trace: `[0.20833335195978483, 0.5532436183959736]`
- Stage Birkhoff offdiag-mass trace: `[0.5381021499633789, 0.5422565937042236]`
- Stage Birkhoff applied-offdiag trace: `[0.002242092491972751, 0.006]`
- Stage Birkhoff gap-delta trace: `[-4.638824611902237e-05, -0.000348089262843132]`
- Stage Birkhoff row-error trace: `[0.0, 0.0]`
- Stage Birkhoff col-error trace: `[0.0, 5.960464477539063e-08]`
- Stage routing stability trace: `[0.6931285858154297, 1.3294206261634827]`
- Forgetting vs routing stability correlation: `1.0`
- Forgetting vs transport gap correlation: `1.0`
- Forgetting vs transport loss correlation: `-1.0`
- Forgetting vs mean abs excess correlation: `1.0`
- Forgetting vs merge-count correlation: `None`