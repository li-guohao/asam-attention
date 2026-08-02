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
- Avg accuracy: `0.1083`
- Avg forgetting: `0.5385`
- Backward transfer: `-0.5385`

## Artifacts

- Plot image: `r4_cifar10_notransport_seed43_plots.png`
- Raw JSON: `r4_cifar10_notransport_seed43.json`
- Resolved prototypes: `10`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `1.3762`
- Prototype heatmap rows: `5`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.5769230769230769, 0.5769230769230769, 0.5512820512820512, 0.5384615384615384]`
- Stage transport gap trace: `[0.05979343317449093, 0.06619343906641006, 0.06766632199287415, 0.06738617643713951, 0.06047219969332218]`
- Stage transport loss trace: `[0.2633837894314811, 0.13349683511824834, 0.14206359393539883, 0.14314205376874833, 0.1218030382361677]`
- Stage merge-count trace: `[2.0, 2.0, 1.0, 1.0, 3.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff gate-factor trace: `[1.0, 1.0, 1.0, 1.0, 1.0]`
- Stage Birkhoff offdiag-mass trace: `[0.028357908129692078, 0.15614379942417145, 0.13841607421636581, 0.1255120299756527, 0.16565868631005287]`
- Stage Birkhoff applied-offdiag trace: `[0.0005671581625938415, 0.003122875988483429, 0.0027683214843273163, 0.0025102405995130537, 0.0033131737262010576]`
- Stage Birkhoff gap-delta trace: `[-3.099068999290466e-05, -0.00036611035466194153, -0.0002864561975002289, -0.0002756156027317047, -0.0004567001014947891]`
- Stage Birkhoff row-error trace: `[0.0002200007438659668, 1.1920928955078125e-07, 1.7881393432617188e-07, 4.976987838745117e-05, 2.1755695343017578e-05]`
- Stage Birkhoff col-error trace: `[1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage routing stability trace: `[0.21112617140724546, 0.2519207132004556, 0.2879607854854493, 0.27487065075408845, 0.2698960014515453]`
- Forgetting vs routing stability correlation: `0.8975584996745951`
- Forgetting vs transport gap correlation: `0.6888825547346322`
- Forgetting vs transport loss correlation: `-0.9816314243726855`
- Forgetting vs mean abs excess correlation: `0.6888823066311262`
- Forgetting vs merge-count correlation: `-0.16762373050613785`