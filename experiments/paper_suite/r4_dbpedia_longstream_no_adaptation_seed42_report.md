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
- Avg accuracy: `0.0464`
- Avg forgetting: `0.3667`
- Backward transfer: `-0.1583`

## Artifacts

- Plot image: `r4_dbpedia_longstream_no_adaptation_seed42_plots.png`
- Raw JSON: `r4_dbpedia_longstream_no_adaptation_seed42.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `1.1257`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.35, 0.39999999999999997, 0.425, 0.4, 0.375]`
- Stage transport gap trace: `[0.047090066596865654, 0.04647402465343475, 0.047426024451851845, 0.04511002264916897, 0.04986052215099335, 0.04984778352081776, 0.05034688487648964]`
- Stage transport loss trace: `[0.18818894972403843, 0.10657281602422396, 0.08040863002339999, 0.07345650171240171, 0.06713395764430365, 0.08578511103987693, 0.06504653735707204]`
- Stage merge-count trace: `[2.0, 3.0, 3.0, 4.0, 3.0, 4.0, 3.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff gate-factor trace: `[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]`
- Stage Birkhoff offdiag-mass trace: `[0.014787494204938412, 0.1961248368024826, 0.17217306792736053, 0.17393051832914352, 0.17206571996212006, 0.1614348515868187, 0.1641179770231247]`
- Stage Birkhoff applied-offdiag trace: `[0.00029574988409876825, 0.003922496736049652, 0.003443461358547211, 0.003478610366582871, 0.0034413143992424013, 0.003228697031736374, 0.003282359540462494]`
- Stage Birkhoff gap-delta trace: `[-1.3152137398719788e-05, -0.000267980620265007, -0.00022011250257492065, -0.0002121664583683014, -0.00019205361604690552, -0.00018769875168800354, -0.0002201758325099945]`
- Stage Birkhoff row-error trace: `[0.00011098384857177734, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.7881393432617188e-07]`
- Stage Birkhoff col-error trace: `[1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 5.960464477539063e-08, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage routing stability trace: `[0.054933303222060206, 0.7354664285977681, 0.6926225463549296, 0.6247898578643799, 0.5148247400919597, 0.5019413709640503, 0.4844826286037763]`
- Forgetting vs routing stability correlation: `0.34013632116044396`
- Forgetting vs transport gap correlation: `0.43692726992382996`
- Forgetting vs transport loss correlation: `-0.8215907594688213`
- Forgetting vs mean abs excess correlation: `0.4369279467793651`
- Forgetting vs merge-count correlation: `0.6568850551721374`