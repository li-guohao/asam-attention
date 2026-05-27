# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Dataset source (train): `huggingface`
- Dataset source (val): `huggingface`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_support_masked`
- Tasks: `2`
- Avg accuracy: `0.5000`
- Avg forgetting: `0.2500`
- Backward transfer: `-0.2500`

## Artifacts

- Plot image: `continual_operator_ablation_sinkhorn_support_masked_seed42_plots.png`
- Raw JSON: `continual_operator_ablation_sinkhorn_support_masked_seed42.json`
- Resolved prototypes: `4`
- Prototype top-k: `1`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.0000`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.25]`
- Stage transport gap trace: `[0.0, 0.0]`
- Stage transport loss trace: `[0.5290975086390972, 0.07397337630391121]`
- Stage merge-count trace: `[1.0, 1.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.0, 0.0]`
- Stage Birkhoff gate-factor trace: `[0.0, 0.0]`
- Stage Birkhoff offdiag-mass trace: `[0.0180203877389431, 0.019804546609520912]`
- Stage Birkhoff applied-offdiag trace: `[0.0, 0.0]`
- Stage Birkhoff gap-delta trace: `[0.0, 0.0]`
- Stage Birkhoff row-error trace: `[9.894371032714844e-06, 1.2874603271484375e-05]`
- Stage Birkhoff col-error trace: `[5.960464477539063e-08, 0.0]`
- Stage routing stability trace: `[0.6931285858154297, 0.7937499284744263]`
- Forgetting vs routing stability correlation: `1.0`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `-1.0`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`