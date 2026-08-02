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
- Avg forgetting: `-0.0625`
- Backward transfer: `0.0625`

## Artifacts

- Plot image: `r2_agnews_bpe_3ep_no_adaptation_seed43_plots.png`
- Raw JSON: `r2_agnews_bpe_3ep_no_adaptation_seed43.json`
- Resolved prototypes: `4`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `1.1724`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0]`
- Stage transport gap trace: `[0.025294032879173756, 0.047186894342303276]`
- Stage transport loss trace: `[0.17818851893146834, 0.052102580200880766]`
- Stage merge-count trace: `[2.0, 0.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.009672795111934345, 0.015]`
- Stage Birkhoff gate-factor trace: `[0.48363975559671724, 0.75]`
- Stage Birkhoff offdiag-mass trace: `[0.024960828013718128, 0.04671579971909523]`
- Stage Birkhoff applied-offdiag trace: `[0.00024162207478289208, 0.0007024971395730973]`
- Stage Birkhoff gap-delta trace: `[-3.725290298461914e-09, -3.725290298461914e-09]`
- Stage Birkhoff row-error trace: `[1.5616416931152344e-05, 5.960464477539063e-08]`
- Stage Birkhoff col-error trace: `[5.960464477539063e-08, 5.960464477539063e-08]`
- Stage routing stability trace: `[0.5717415461937586, 0.39092956855893135]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`