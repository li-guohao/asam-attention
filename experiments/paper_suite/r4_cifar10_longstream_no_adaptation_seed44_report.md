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
- Avg accuracy: `0.1333`
- Avg forgetting: `0.5000`
- Backward transfer: `-0.5000`

## Artifacts

- Plot image: `r4_cifar10_longstream_no_adaptation_seed44_plots.png`
- Raw JSON: `r4_cifar10_longstream_no_adaptation_seed44.json`
- Resolved prototypes: `10`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.7188`
- Prototype heatmap rows: `5`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.5, 0.5, 0.5, 0.5]`
- Stage transport gap trace: `[0.05522916465997696, 0.06456406973302364, 0.06615982204675674, 0.06644904986023903, 0.06660225242376328]`
- Stage transport loss trace: `[0.2506973317691258, 0.09356806604635148, 0.07472992209451539, 0.07216695394544374, 0.08039183252387577]`
- Stage merge-count trace: `[3.0, 3.0, 1.0, 1.0, 1.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff gate-factor trace: `[1.0, 1.0, 1.0, 1.0, 1.0]`
- Stage Birkhoff offdiag-mass trace: `[0.024379843845963478, 0.17150817811489105, 0.14873094484210014, 0.1334967017173767, 0.13580160215497017]`
- Stage Birkhoff applied-offdiag trace: `[0.0004875968769192696, 0.003430163562297821, 0.002974618896842003, 0.002669934034347534, 0.0027160320430994036]`
- Stage Birkhoff gap-delta trace: `[-1.486949622631073e-05, -0.0004163142293691635, -0.00034877657890319824, -0.0003020484000444412, -0.00029770471155643463]`
- Stage Birkhoff row-error trace: `[7.915496826171875e-05, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage Birkhoff col-error trace: `[1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 5.960464477539063e-08]`
- Stage routing stability trace: `[0.37118517855803174, 0.29916126493896755, 0.1878007480076381, 0.1755508810636543, 0.19352995169659457]`
- Forgetting vs routing stability correlation: `-0.8171215047967993`
- Forgetting vs transport gap correlation: `0.9859346017379492`
- Forgetting vs transport loss correlation: `-0.9941765577245342`
- Forgetting vs mean abs excess correlation: `0.9859346502306526`
- Forgetting vs merge-count correlation: `-0.6123724356957942`