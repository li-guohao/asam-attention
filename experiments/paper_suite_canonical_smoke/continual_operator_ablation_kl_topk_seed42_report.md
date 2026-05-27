# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Dataset source (train): `huggingface`
- Dataset source (val): `huggingface`
- Routing mode: `prototype`
- Routing strategy: `kl_topk`
- Tasks: `2`
- Avg accuracy: `0.5000`
- Avg forgetting: `0.2500`
- Backward transfer: `-0.2500`

## Artifacts

- Plot image: `continual_operator_ablation_kl_topk_seed42_plots.png`
- Raw JSON: `continual_operator_ablation_kl_topk_seed42.json`
- Resolved prototypes: `4`
- Prototype top-k: `1`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.0000`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.25]`
- Stage transport gap trace: `[0.023465121164917946, 0.014903162606060505]`
- Stage transport loss trace: `[0.5176981277763844, 0.050325531512498856]`
- Stage merge-count trace: `[1.0, 0.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.015653595328330994, 0.010057363659143448]`
- Stage Birkhoff gate-factor trace: `[0.7826797664165497, 0.5028681829571724]`
- Stage Birkhoff offdiag-mass trace: `[0.021166551858186722, 0.4660147726535797]`
- Stage Birkhoff applied-offdiag trace: `[0.0003313326372841874, 0.004686880039110108]`
- Stage Birkhoff gap-delta trace: `[-1.5271827578544617e-05, -0.0001828828826546669]`
- Stage Birkhoff row-error trace: `[6.556510925292969e-07, 0.0]`
- Stage Birkhoff col-error trace: `[0.0, 0.0]`
- Stage routing stability trace: `[0.3564591705799103, 0.013317860197275877]`
- Forgetting vs routing stability correlation: `-0.9999999999999999`
- Forgetting vs transport gap correlation: `-1.0`
- Forgetting vs transport loss correlation: `-1.0`
- Forgetting vs mean abs excess correlation: `-1.0`
- Forgetting vs merge-count correlation: `-0.9999999999999999`