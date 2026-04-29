# Continual Text Benchmark Report

- Dataset: `split_arxiv`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `4`
- Avg accuracy: `0.6250`
- Avg forgetting: `0.0000`
- Backward transfer: `0.1667`

## Artifacts

- Plot image: `controller_round3_arxiv_fallback_ablation_no_adaptation_seed43_plots.png`
- Raw JSON: `controller_round3_arxiv_fallback_ablation_no_adaptation_seed43.json`
- Resolved prototypes: `8`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6282`
- Prototype heatmap rows: `4`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.0, 0.0, 0.0]`
- Stage transport gap trace: `[0.01845387928187847, 0.02715342864394188, 0.03327466547489166, 0.03399764746427536]`
- Stage transport loss trace: `[0.4728832356631756, 0.07379623502492905, 0.10961311310529709, 0.126817025244236]`
- Stage merge-count trace: `[3.0, 1.0, 1.0, 1.0]`
- Stage routing stability trace: `[0.399270236492157, 0.2119368016719818, 0.24925664067268372, 0.17834408581256866]`
- Forgetting vs routing stability correlation: `None`
- Forgetting vs transport gap correlation: `None`
- Forgetting vs transport loss correlation: `None`
- Forgetting vs mean abs excess correlation: `None`
- Forgetting vs merge-count correlation: `None`