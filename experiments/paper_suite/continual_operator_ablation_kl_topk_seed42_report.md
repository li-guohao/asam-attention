# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `kl_topk`
- Tasks: `2`
- Avg accuracy: `0.5312`
- Avg forgetting: `-0.3125`
- Backward transfer: `0.3125`

## Artifacts

- Plot image: `continual_operator_ablation_kl_topk_seed42_plots.png`
- Raw JSON: `continual_operator_ablation_kl_topk_seed42.json`
- Resolved prototypes: `4`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6760`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.010916772298514843, 0.01763218641281128]`
- Stage transport loss trace: `[0.29722958616912365, 0.07698699831962585]`
- Stage merge-count trace: `[1.0, 0.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.007298055415352186, 0.011889491230249405]`
- Stage Birkhoff gate-factor trace: `[0.3649027707676093, 0.5944745615124702]`
- Stage Birkhoff offdiag-mass trace: `[0.17452137172222137, 0.4340418875217438]`
- Stage Birkhoff applied-offdiag trace: `[0.0012736666419920495, 0.005160537215250671]`
- Stage Birkhoff gap-delta trace: `[-3.0310824513435364e-05, -0.00020205043256282806]`
- Stage Birkhoff row-error trace: `[5.960464477539063e-08, 0.0]`
- Stage Birkhoff col-error trace: `[0.0, 0.0]`
- Stage routing stability trace: `[0.09322948195040226, 0.07350532431155443]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`