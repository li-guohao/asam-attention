# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Protocol: `task_incremental_multihead`
- Label mode: `local`
- Head mode: `multi`
- Train task-id mode: `oracle`
- Eval task-id mode: `oracle`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Output classes: `2`
- Avg accuracy: `0.5312`
- Avg forgetting: `0.0625`
- Backward transfer: `-0.0625`

## Artifacts

- Plot image: `r3_baseline_comparison_prototype_seed42_plots.png`
- Raw JSON: `r3_baseline_comparison_prototype_seed42.json`
- Resolved prototypes: `4`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6698`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0625]`
- Stage transport gap trace: `[0.008570894598960876, 0.019137196242809296]`
- Stage transport loss trace: `[0.29425351694226265, 0.07179965265095234]`
- Stage merge-count trace: `[2.0, 1.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.005772305031617483, 0.012758130828539532]`
- Stage Birkhoff gate-factor trace: `[0.28861525158087414, 0.6379065414269766]`
- Stage Birkhoff offdiag-mass trace: `[0.5535090565681458, 0.01937105879187584]`
- Stage Birkhoff applied-offdiag trace: `[0.003195023112274154, 0.00024713850235408286]`
- Stage Birkhoff gap-delta trace: `[-8.756294846534729e-05, 0.0]`
- Stage Birkhoff row-error trace: `[8.344650268554688e-07, 1.2755393981933594e-05]`
- Stage Birkhoff col-error trace: `[0.0, 0.0]`
- Stage routing stability trace: `[0.39108747243881226, 0.18304728472139686]`
- Forgetting vs routing stability correlation: `-0.9999999999999998`
- Forgetting vs transport gap correlation: `0.9999999999999999`
- Forgetting vs transport loss correlation: `-1.0`
- Forgetting vs mean abs excess correlation: `1.0`
- Forgetting vs merge-count correlation: `-0.9999999999999999`