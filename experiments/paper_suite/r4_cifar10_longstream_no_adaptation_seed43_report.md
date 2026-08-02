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
- Avg forgetting: `0.5385`
- Backward transfer: `-0.5385`

## Artifacts

- Plot image: `r4_cifar10_longstream_no_adaptation_seed43_plots.png`
- Raw JSON: `r4_cifar10_longstream_no_adaptation_seed43.json`
- Resolved prototypes: `10`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `1.3615`
- Prototype heatmap rows: `5`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.5769230769230769, 0.5769230769230769, 0.5512820512820512, 0.5384615384615384]`
- Stage transport gap trace: `[0.05978173576295376, 0.06710801273584366, 0.06952160969376564, 0.06498561426997185, 0.06512707471847534]`
- Stage transport loss trace: `[0.2395775555854752, 0.09820925905590966, 0.0831451252812431, 0.08432086965157873, 0.07491853750414318]`
- Stage merge-count trace: `[2.0, 2.0, 1.0, 3.0, 2.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff gate-factor trace: `[1.0, 1.0, 1.0, 1.0, 1.0]`
- Stage Birkhoff offdiag-mass trace: `[0.028503685258328915, 0.1686246171593666, 0.15858659148216248, 0.17523950338363647, 0.15978176891803741]`
- Stage Birkhoff applied-offdiag trace: `[0.0005700737051665783, 0.0033724923431873322, 0.0031717318296432496, 0.003504790067672729, 0.003195635378360748]`
- Stage Birkhoff gap-delta trace: `[-3.108754754066467e-05, -0.000418979674577713, -0.0003827400505542755, -0.000469183549284935, -0.00039265304803848267]`
- Stage Birkhoff row-error trace: `[0.00022834539413452148, 1.1920928955078125e-07, 1.1920928955078125e-07, 5.960464477539063e-08, 1.1920928955078125e-07]`
- Stage Birkhoff col-error trace: `[5.960464477539063e-08, 1.1920928955078125e-07, 5.960464477539063e-08, 1.1920928955078125e-07, 5.960464477539063e-08]`
- Stage routing stability trace: `[0.21364867900099074, 0.23598484517562957, 0.23775251635483333, 0.21662627515338717, 0.18831021711230278]`
- Forgetting vs routing stability correlation: `0.19724675833031644`
- Forgetting vs transport gap correlation: `0.8857989603859112`
- Forgetting vs transport loss correlation: `-0.9846348154795256`
- Forgetting vs mean abs excess correlation: `0.8857991694861643`
- Forgetting vs merge-count correlation: `-0.03606092229873097`