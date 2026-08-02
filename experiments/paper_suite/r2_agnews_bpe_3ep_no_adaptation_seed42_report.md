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
- Avg accuracy: `0.5000`
- Avg forgetting: `0.0625`
- Backward transfer: `-0.0625`

## Artifacts

- Plot image: `r2_agnews_bpe_3ep_no_adaptation_seed42_plots.png`
- Raw JSON: `r2_agnews_bpe_3ep_no_adaptation_seed42.json`
- Resolved prototypes: `4`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.7876`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0625]`
- Stage transport gap trace: `[0.03247445076704025, 0.04224349930882454]`
- Stage transport loss trace: `[0.16937775909900665, 0.052672410383820534]`
- Stage merge-count trace: `[0.0, 0.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.014675804016490778, 0.02]`
- Stage Birkhoff gate-factor trace: `[0.7337902008245389, 1.0]`
- Stage Birkhoff offdiag-mass trace: `[0.047854166477918625, 0.04920538514852524]`
- Stage Birkhoff applied-offdiag trace: `[0.0007036116488744043, 0.0009841077029705048]`
- Stage Birkhoff gap-delta trace: `[-9.313225746154785e-10, 0.0]`
- Stage Birkhoff row-error trace: `[1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage Birkhoff col-error trace: `[1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage routing stability trace: `[0.5098772123456001, 0.587575264275074]`
- Forgetting vs routing stability correlation: `0.9999999999999999`
- Forgetting vs transport gap correlation: `0.9999999999999999`
- Forgetting vs transport loss correlation: `-1.0`
- Forgetting vs mean abs excess correlation: `0.9999999999999999`
- Forgetting vs merge-count correlation: `None`