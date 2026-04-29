# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.4219`
- Avg forgetting: `-0.0469`
- Backward transfer: `0.0469`

## Artifacts

- Plot image: `capacity_round2_sweep_slots4_topk1_seed44_plots.png`
- Raw JSON: `capacity_round2_sweep_slots4_topk1_seed44.json`
- Resolved prototypes: `8`
- Prototype top-k: `1`
- Prototype slots/task: `4`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.0001`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.07583489269018173, 0.08542240411043167]`
- Stage transport loss trace: `[0.09556552290450782, 0.0320742396870628]`
- Stage merge-count trace: `[1.0, 1.0]`
- Stage routing stability trace: `[0.22479090024717152, 0.13854070543311536]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`