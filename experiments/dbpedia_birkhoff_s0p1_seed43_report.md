# Continual Text Benchmark Report

- Dataset: `split_dbpedia`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `7`
- Avg accuracy: `0.5286`
- Avg forgetting: `0.0833`
- Backward transfer: `-0.0000`

## Artifacts

- Plot image: `dbpedia_birkhoff_s0p1_seed43_plots.png`
- Raw JSON: `dbpedia_birkhoff_s0p1_seed43.json`
- Resolved prototypes: `14`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6933`
- Prototype heatmap rows: `7`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.0, 0.13333333333333333, 0.12500000000000003, 0.10000000000000002, 0.08333333333333336]`
- Stage transport gap trace: `[0.022630149498581886, 0.02903488464653492, 0.03255090489983559, 0.03516003116965294, 0.03459738940000534, 0.030578557401895523, 0.03527117520570755]`
- Stage transport loss trace: `[0.3345447393755118, 0.04669520755608877, 0.05419024949272474, 0.048091297348340355, 0.0689390202363332, 0.04752168928583463, 0.037968602031469345]`
- Stage merge-count trace: `[0.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0]`
- Stage Birkhoff offdiag-mass trace: `[0.009588897228240967, 0.27515754103660583, 0.28118985891342163, 0.2674476206302643, 0.2503243684768677, 0.2724947929382324, 0.2534043490886688]`
- Stage Birkhoff row-error trace: `[0.00010913610458374023, 1.1920928955078125e-07, 1.1920928955078125e-07, 2.384185791015625e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07]`
- Stage Birkhoff col-error trace: `[5.960464477539063e-08, 1.1920928955078125e-07, 1.1920928955078125e-07, 1.1920928955078125e-07, 5.960464477539063e-08, 5.960464477539063e-08, 5.960464477539063e-08]`
- Stage routing stability trace: `[0.07099447896083196, 0.07355048755804698, 0.026260067398349445, 0.019648009911179543, 0.42654890318711597, 0.6860180298487345, 0.5198915402094523]`
- Forgetting vs routing stability correlation: `0.5198817828804345`
- Forgetting vs transport gap correlation: `0.6832180746921809`
- Forgetting vs transport loss correlation: `-0.4373158772356011`
- Forgetting vs mean abs excess correlation: `0.6832181489896969`
- Forgetting vs merge-count correlation: `0.12359606041364168`