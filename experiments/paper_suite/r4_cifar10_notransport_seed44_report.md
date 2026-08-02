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
- Avg accuracy: `0.1167`
- Avg forgetting: `0.5000`
- Backward transfer: `-0.5000`

## Artifacts

- Plot image: `r4_cifar10_notransport_seed44_plots.png`
- Raw JSON: `r4_cifar10_notransport_seed44.json`
- Resolved prototypes: `10`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `1.0312`
- Prototype heatmap rows: `5`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.5, 0.5, 0.5, 0.5]`
- Stage transport gap trace: `[0.05524932220578194, 0.05753368139266968, 0.06354628875851631, 0.07058630511164665, 0.07116769999265671]`
- Stage transport loss trace: `[0.2758052015588397, 0.13014205864497594, 0.12376349703187034, 0.11579885085423787, 0.12599811454614004]`
- Stage merge-count trace: `[3.0, 3.0, 3.0, 0.0, 1.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff gate-factor trace: `[1.0, 1.0, 1.0, 1.0, 1.0]`
- Stage Birkhoff offdiag-mass trace: `[0.024303555488586426, 0.18207571282982826, 0.17021531611680984, 0.11376134119927883, 0.11700748652219772]`
- Stage Birkhoff applied-offdiag trace: `[0.0004860711097717285, 0.0036415142565965654, 0.003404306322336197, 0.002275226823985577, 0.0023401497304439546]`
- Stage Birkhoff gap-delta trace: `[-1.640617847442627e-05, -0.0005080066621303558, -0.0004440862685441971, -0.00023984909057617188, -0.0002801753580570221]`
- Stage Birkhoff row-error trace: `[9.02414321899414e-05, 5.960464477539063e-08, 1.3709068298339844e-06, 1.3947486877441406e-05, 4.506111145019531e-05]`
- Stage Birkhoff col-error trace: `[1.1920928955078125e-07, 1.1920928955078125e-07, 5.960464477539063e-08, 1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage routing stability trace: `[0.40283688051359995, 0.37084229929106577, 0.3368412068202382, 0.18127755749793278, 0.18890984935892952]`
- Forgetting vs routing stability correlation: `-0.5731807449798518`
- Forgetting vs transport gap correlation: `0.641573283749995`
- Forgetting vs transport loss correlation: `-0.997059448730403`
- Forgetting vs mean abs excess correlation: `0.6415732896617211`
- Forgetting vs merge-count correlation: `-0.39528470752104733`