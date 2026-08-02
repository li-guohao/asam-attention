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

- Plot image: `r4_cifar10_longstream_no_adaptation_seed42_plots.png`
- Raw JSON: `r4_cifar10_longstream_no_adaptation_seed42.json`
- Resolved prototypes: `10`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.7388`
- Prototype heatmap rows: `5`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.5, 0.4807692307692308, 0.5256410256410257, 0.5192307692307693]`
- Stage transport gap trace: `[0.06356785073876381, 0.061638038605451584, 0.0686543695628643, 0.07028363645076752, 0.06989701092243195]`
- Stage transport loss trace: `[0.2280122126851763, 0.09471039722363155, 0.07196536660194397, 0.07460358135756992, 0.08445258314410846]`
- Stage merge-count trace: `[0.0, 2.0, 1.0, 1.0, 3.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff gate-factor trace: `[1.0, 1.0, 1.0, 1.0, 1.0]`
- Stage Birkhoff offdiag-mass trace: `[0.03330528549849987, 0.22083543241024017, 0.1858956143260002, 0.16704003512859344, 0.19012270867824554]`
- Stage Birkhoff applied-offdiag trace: `[0.0006661057099699975, 0.004416708648204804, 0.003717912286520004, 0.003340800702571869, 0.003802454173564911]`
- Stage Birkhoff gap-delta trace: `[-4.4792890548706055e-05, -0.0005471110343933105, -0.0004703439772129059, -0.00042882561683654785, -0.0005339831113815308]`
- Stage Birkhoff row-error trace: `[0.00023889541625976562, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage Birkhoff col-error trace: `[5.960464477539063e-08, 5.960464477539063e-08, 1.1920928955078125e-07, 5.960464477539063e-08, 5.960464477539063e-08]`
- Stage routing stability trace: `[0.09788058724786554, 0.2849611579662278, 0.25102415397053673, 0.20719172663631893, 0.2535470293627845]`
- Forgetting vs routing stability correlation: `0.9072812363405597`
- Forgetting vs transport gap correlation: `0.48349969350879`
- Forgetting vs transport loss correlation: `-0.986726799732571`
- Forgetting vs mean abs excess correlation: `0.483500165674388`
- Forgetting vs merge-count correlation: `0.7029164657914051`