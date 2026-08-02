# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Protocol: `task_incremental_multihead`
- Label mode: `local`
- Head mode: `multi`
- Train task-id mode: `oracle`
- Eval task-id mode: `oracle`
- Routing mode: `task`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Output classes: `2`
- Avg accuracy: `0.4375`
- Avg forgetting: `0.1875`
- Backward transfer: `-0.1875`

## Artifacts

- Plot image: `r2_baseline_comparison_task_routing_seed43_plots.png`
- Raw JSON: `r2_baseline_comparison_task_routing_seed43.json`