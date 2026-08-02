# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.5000`
- Avg forgetting: `0.0000`
- Backward transfer: `0.0000`

## Artifacts

- Plot image: `continual_operator_ablation_no_transport_seed43_plots.png`
- Raw JSON: `continual_operator_ablation_no_transport_seed43.json`
- Resolved prototypes: `4`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6909`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.0004317226994317025, 0.018612993881106377]`
- Stage transport loss trace: `[0.2786545529961586, 0.056157857179641724]`
- Stage merge-count trace: `[2.0, 0.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.00028796245654424035, 0.012408662587404251]`
- Stage Birkhoff gate-factor trace: `[0.014398122827212017, 0.6204331293702126]`
- Stage Birkhoff offdiag-mass trace: `[0.5550552606582642, 0.045014359056949615]`
- Stage Birkhoff applied-offdiag trace: `[0.0001598350763769574, 0.0005585679931259524]`
- Stage Birkhoff gap-delta trace: `[-2.20985384657979e-07, 0.0]`
- Stage Birkhoff row-error trace: `[1.1324882507324219e-06, 0.0]`
- Stage Birkhoff col-error trace: `[0.0, 0.0]`
- Stage routing stability trace: `[0.5147922337055206, 0.21020364481955767]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`