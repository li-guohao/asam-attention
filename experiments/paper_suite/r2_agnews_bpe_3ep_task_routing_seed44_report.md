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
- Avg accuracy: `0.5000`
- Avg forgetting: `0.0625`
- Backward transfer: `-0.0625`

## Artifacts

- Plot image: `r2_agnews_bpe_3ep_task_routing_seed44_plots.png`
- Raw JSON: `r2_agnews_bpe_3ep_task_routing_seed44.json`