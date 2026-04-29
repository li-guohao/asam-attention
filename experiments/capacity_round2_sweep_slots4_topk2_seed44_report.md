# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.5000`
- Avg forgetting: `-0.1250`
- Backward transfer: `0.1250`

## Artifacts

- Plot image: `capacity_round2_sweep_slots4_topk2_seed44_plots.png`
- Raw JSON: `capacity_round2_sweep_slots4_topk2_seed44.json`
- Resolved prototypes: `8`
- Prototype top-k: `2`
- Prototype slots/task: `4`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6931`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.05422726646065712, 0.07617558538913727]`
- Stage transport loss trace: `[0.09610406402498484, 0.03355858183931559]`
- Stage merge-count trace: `[1.0, 1.0]`
- Stage routing stability trace: `[0.14584550389554352, 0.19837269047275186]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`