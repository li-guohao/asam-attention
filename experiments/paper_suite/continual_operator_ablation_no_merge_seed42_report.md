# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.4688`
- Avg forgetting: `-0.1875`
- Backward transfer: `0.1875`

## Artifacts

- Plot image: `continual_operator_ablation_no_merge_seed42_plots.png`
- Raw JSON: `continual_operator_ablation_no_merge_seed42.json`
- Resolved prototypes: `4`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6828`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.025844166055321693, 0.020117713138461113]`
- Stage transport loss trace: `[0.30774285923689604, 0.07161451317369938]`
- Stage merge-count trace: `[0.0, 0.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.013829359995400439, 0.013411812484264374]`
- Stage Birkhoff gate-factor trace: `[0.6914679997700219, 0.6705906242132187]`
- Stage Birkhoff offdiag-mass trace: `[0.43385955691337585, 0.050391387194395065]`
- Stage Birkhoff applied-offdiag trace: `[0.006, 0.0006758398358731876]`
- Stage Birkhoff gap-delta trace: `[-0.00044736452400684357, -5.587935447692871e-09]`
- Stage Birkhoff row-error trace: `[5.960464477539063e-08, 0.0]`
- Stage Birkhoff col-error trace: `[0.0, 0.0]`
- Stage routing stability trace: `[0.3603001981973648, 0.4460565894842148]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`