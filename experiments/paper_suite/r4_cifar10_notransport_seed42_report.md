# Continual Text Benchmark Report

- Dataset: `split_cifar10`
- Protocol: `class_incremental_singlehead`
- Label mode: `global`
- Head mode: `single`
- Train task-id mode: `oracle`
- Eval task-id mode: `none`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `5`
- Output classes: `10`
- Avg accuracy: `0.1000`
- Avg forgetting: `0.5192`
- Backward transfer: `-0.5192`

## Artifacts

- Plot image: `r4_cifar10_notransport_seed42_plots.png`
- Raw JSON: `r4_cifar10_notransport_seed42.json`
- Resolved prototypes: `10`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.7602`
- Prototype heatmap rows: `5`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.5, 0.4807692307692308, 0.5256410256410257, 0.5192307692307693]`
- Stage transport gap trace: `[0.06369251757860184, 0.0599390659481287, 0.06860150769352913, 0.07030735164880753, 0.06812762469053268]`
- Stage transport loss trace: `[0.2467681680406843, 0.12651232984803973, 0.12211574507611138, 0.15011211271796907, 0.1723833754658699]`
- Stage merge-count trace: `[0.0, 2.0, 1.0, 1.0, 2.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff gate-factor trace: `[1.0, 1.0, 1.0, 1.0, 1.0]`
- Stage Birkhoff offdiag-mass trace: `[0.03153661545366049, 0.22867292165756226, 0.15843822807073593, 0.14182376861572266, 0.14739441126585007]`
- Stage Birkhoff applied-offdiag trace: `[0.0006307323090732098, 0.004573458433151245, 0.0031687645614147187, 0.0028364753723144533, 0.0029478882253170012]`
- Stage Birkhoff gap-delta trace: `[-3.9208680391311646e-05, -0.0005416199564933777, -0.0003775171935558319, -0.0003117397427558899, -0.0003678537905216217]`
- Stage Birkhoff row-error trace: `[0.00026494264602661133, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage Birkhoff col-error trace: `[1.1920928955078125e-07, 1.1920928955078125e-07, 5.960464477539063e-08, 1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage routing stability trace: `[0.08997870059240431, 0.31616882234811783, 0.2854818052479199, 0.22664032273349308, 0.2832799930539396]`
- Forgetting vs routing stability correlation: `0.9139788696901174`
- Forgetting vs transport gap correlation: `0.34245205787841737`
- Forgetting vs transport loss correlation: `-0.8898520783391528`
- Forgetting vs mean abs excess correlation: `0.3424521841909842`
- Forgetting vs merge-count correlation: `0.8078180457925052`