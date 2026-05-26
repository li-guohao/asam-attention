# Continual ASAM Paper Suite

## Run Config

- Dataset: `split_ag_news`
- Output directory: `experiments\paper_suite_realdata`
- Device: `cpu`
- Seeds for ablation: `2`

## Benchmark

- Meta-secant avg accuracy: `0.5312`
- Meta-secant avg forgetting: `0.0000`
- Meta-secant backward transfer: `0.0000`
- Benchmark JSON: `continual_benchmark.json`
- Benchmark report: `continual_benchmark_report.md`

## Ablation

- Best avg accuracy: `no_adaptation (0.5000)`
- Lowest avg forgetting: `no_adaptation (0.0000)`
- Ablation JSON: `continual_ablation.json`
- Ablation report: `continual_ablation_report.md`
- Ablation table: `continual_ablation_table.md`
- Ablation CSV: `continual_ablation.csv`
- Ablation plot: `continual_ablation.png`

## Operator Ablation

- Best operator avg accuracy: `sinkhorn_topk (0.5000)`
- Lowest operator avg forgetting: `sinkhorn_topk (0.0000)`
- Operator Ablation JSON: `continual_operator_ablation.json`
- Operator Ablation report: `continual_operator_ablation_report.md`
- Operator Ablation table: `continual_operator_ablation_table.md`
- Operator Ablation CSV: `continual_operator_ablation.csv`
- Operator Ablation plot: `continual_operator_ablation.png`

## Recommendation

- Use the ablation report and CSV as the main strategy-level paper tables.
- Use the operator ablation report and CSV for the mechanistic appendix and operator study.
- Use the benchmark report as the detailed continual-learning diagnostics appendix.
- Use the plot artifacts directly in slides and drafts.