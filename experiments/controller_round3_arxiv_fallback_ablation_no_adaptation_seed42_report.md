# Continual Text Benchmark Report

- Dataset: `split_arxiv`
- Routing mode: `prototype`
- Routing strategy: `sinkhorn_topk`
- Tasks: `4`
- Avg accuracy: `0.5625`
- Avg forgetting: `0.1667`
- Backward transfer: `-0.1667`

## Artifacts

- Plot image: `controller_round3_arxiv_fallback_ablation_no_adaptation_seed42_plots.png`
- Raw JSON: `controller_round3_arxiv_fallback_ablation_no_adaptation_seed42.json`
- Resolved prototypes: `8`
- Prototype top-k: `2`
- Prototype slots/task: `2`

## Prototype Diagnostics

- Final-stage mean routing entropy: `0.6921`
- Prototype heatmap rows: `4`

## Theory Diagnostics

- Stage forgetting trace: `[0.0, 0.25, 0.25, 0.16666666666666666]`
- Stage transport gap trace: `[0.017980769276618958, 0.03299206495285034, 0.03644147515296936, 0.03966112434864044]`
- Stage transport loss trace: `[0.46701505221426487, 0.07488161697983742, 0.14462322741746902, 0.14025647938251495]`
- Stage merge-count trace: `[2.0, 0.0, 1.0, 1.0]`
- Stage routing stability trace: `[0.3672025203704834, 0.026772175915539265, 0.016370405908674, 0.3308316767215729]`
- Forgetting vs routing stability correlation: `-0.8589811523565457`
- Forgetting vs transport gap correlation: `0.8229339073738893`
- Forgetting vs transport loss correlation: `-0.9544414131044924`
- Forgetting vs mean abs excess correlation: `0.8229339522217807`
- Forgetting vs merge-count correlation: `-0.8660254037844386`