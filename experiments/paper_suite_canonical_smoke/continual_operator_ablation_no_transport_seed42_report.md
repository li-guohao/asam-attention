# Continual Text Benchmark Report

- Dataset: `split_ag_news`
- Dataset source (train): `huggingface`
- Dataset source (val): `huggingface`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `2`
- Avg accuracy: `0.5000`
- Avg forgetting: `0.2500`
- Backward transfer: `-0.2500`

## Artifacts

- Plot image: `continual_operator_ablation_no_transport_seed42_plots.png`
- Raw JSON: `continual_operator_ablation_no_transport_seed42.json`
- Resolved prototypes: `4`
- Prototype top-k: `1`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.0000`
- Prototype heatmap rows: `2`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.25]`
- Stage transport gap trace: `[0.006203977856785059, 0.023569395765662193]`
- Stage transport loss trace: `[0.5289209447801113, 0.07035461068153381]`
- Stage merge-count trace: `[1.0, 1.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.004166667039195697, 0.011028717550929183]`
- Stage Birkhoff gate-factor trace: `[0.20833335195978483, 0.5514358775464592]`
- Stage Birkhoff offdiag-mass trace: `[0.5412599444389343, 0.5440342426300049]`
- Stage Birkhoff applied-offdiag trace: `[0.0022552499701306017, 0.006]`
- Stage Birkhoff gap-delta trace: `[-4.6022702008485794e-05, -0.0005049221217632294]`
- Stage Birkhoff row-error trace: `[5.960464477539063e-08, 1.7881393432617188e-07]`
- Stage Birkhoff col-error trace: `[0.0, 0.0]`
- Stage routing stability trace: `[0.6931285858154297, 0.7372403144836426]`
- Forgetting vs routing stability correlation: `1.0`
- Forgetting vs transport gap correlation: `0.9999999999999999`
- Forgetting vs transport loss correlation: `-1.0`
- Forgetting vs mean abs excess correlation: `1.0`
- Forgetting vs merge-count correlation: `None`