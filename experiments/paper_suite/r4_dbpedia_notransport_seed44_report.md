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
- Avg accuracy: `0.1179`
- Avg forgetting: `0.3667`
- Backward transfer: `-0.3250`

## Artifacts

- Plot image: `r4_dbpedia_notransport_seed44_plots.png`
- Raw JSON: `r4_dbpedia_notransport_seed44.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.8065`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.7, 0.44999999999999996, 0.39999999999999997, 0.4125, 0.43, 0.3666666666666667]`
- Stage transport gap trace: `[0.04709227941930294, 0.05345291830599308, 0.05590115301311016, 0.053380049765110016, 0.05356512404978275, 0.049047039821743965, 0.05161813460290432]`
- Stage transport loss trace: `[0.08144362922757864, 0.02892765241364638, 0.034059131021300953, 0.03116980878015359, 0.03755900611480077, 0.06324928601582845, 0.05232667208959659]`
- Stage merge-count trace: `[2.0, 2.0, 2.0, 2.0, 1.0, 3.0, 3.0]`
- Stage Birkhoff base-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff effective-strength trace: `[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]`
- Stage Birkhoff gate-factor trace: `[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]`
- Stage Birkhoff offdiag-mass trace: `[0.014143133535981178, 0.1734890341758728, 0.19769325107336044, 0.17719584703445435, 0.16418495029211044, 0.16310817003250122, 0.177288219332695]`
- Stage Birkhoff applied-offdiag trace: `[0.00028286267071962355, 0.0034697806835174557, 0.003953865021467209, 0.0035439169406890868, 0.003283699005842209, 0.0032621634006500247, 0.0035457643866538997]`
- Stage Birkhoff gap-delta trace: `[-1.093931496143341e-05, -0.00026769377291202545, -0.0003287326544523239, -0.000280601903796196, -0.00023350678384304047, -0.0002283044159412384, -0.0002650003880262375]`
- Stage Birkhoff row-error trace: `[0.00016361474990844727, 0.0003077387809753418, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage Birkhoff col-error trace: `[5.960464477539063e-08, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage routing stability trace: `[0.03059800627330939, 0.25870844423770906, 0.17706000606218975, 0.22456873108943304, 0.29147119919459025, 0.42789511879285175, 0.40870506688952446]`
- Forgetting vs routing stability correlation: `0.5193656262607587`
- Forgetting vs transport gap correlation: `0.6807405524063924`
- Forgetting vs transport loss correlation: `-0.8103275954508817`
- Forgetting vs mean abs excess correlation: `0.680740612015178`
- Forgetting vs merge-count correlation: `-0.011728718810281192`