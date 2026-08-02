# Continual Text Benchmark Report

- Dataset: `split_dbpedia`
- Protocol: `class_incremental_singlehead`
- Label mode: `global`
- Head mode: `single`
- Train task-id mode: `oracle`
- Eval task-id mode: `none`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `7`
- Output classes: `14`
- Avg accuracy: `0.1429`
- Avg forgetting: `0.4833`
- Backward transfer: `-0.4833`

## Artifacts

- Plot image: `r4_dbpedia_notransport_seed43_plots.png`
- Raw JSON: `r4_dbpedia_notransport_seed43.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.7478`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.7, 0.475, 0.5166666666666666, 0.475, 0.48, 0.48333333333333334]`
- Stage transport gap trace: `[0.04708939231932163, 0.05426529236137867, 0.05190047807991505, 0.05436665564775467, 0.05492791347205639, 0.05237889662384987, 0.04957975819706917]`
- Stage transport loss trace: `[0.11134085108836492, 0.031739044065276785, 0.03604142653445403, 0.03865854988495509, 0.02473573163151741, 0.03953263635436694, 0.03924007946625352]`
- Stage merge-count trace: `[2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 5.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff gate-factor trace: `[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]`
- Stage Birkhoff offdiag-mass trace: `[0.015056807547807693, 0.19208793342113495, 0.1936689391732216, 0.1770695000886917, 0.16892007738351822, 0.17469556629657745, 0.1938479319214821]`
- Stage Birkhoff applied-offdiag trace: `[0.00030113615095615385, 0.003841758668422699, 0.0038733787834644316, 0.003541390001773834, 0.0033784015476703644, 0.0034939113259315493, 0.0038769586384296423]`
- Stage Birkhoff gap-delta trace: `[-1.3828277587890625e-05, -0.00033255666494369507, -0.00029299966990947723, -0.0002812407910823822, -0.00024879537522792816, -0.00027419812977313995, -0.0003066975623369217]`
- Stage Birkhoff row-error trace: `[0.00014138221740722656, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.7881393432617188e-07]`
- Stage Birkhoff col-error trace: `[1.1920928955078125e-07, 5.960464477539063e-08, 1.1920928955078125e-07, 0.0, 1.1920928955078125e-07, 5.960464477539063e-08, 5.960464477539063e-08]`
- Stage routing stability trace: `[0.0319468192756176, 0.23340989003578821, 0.15461123983065286, 0.23827014565467836, 0.1843477765719096, 0.3417801102002462, 0.39365630969405174]`
- Forgetting vs routing stability correlation: `0.6276153565631843`
- Forgetting vs transport gap correlation: `0.8002533233950637`
- Forgetting vs transport loss correlation: `-0.9235732211854065`
- Forgetting vs mean abs excess correlation: `0.8002534148272555`
- Forgetting vs merge-count correlation: `0.050341479292313696`