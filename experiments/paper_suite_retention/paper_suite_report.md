# Continual ASAM Paper Suite

## Run Config

- Dataset: `split_ag_news`
- Output directory: `experiments\paper_suite_retention`
- Device: `cpu`
- Seeds for ablation: `3`
- Candidate profile: `retention`
- Prototype layout: `num_prototypes=0, slots_per_task=2, top_k=1`

## Benchmark

- Meta-secant avg accuracy: `0.4766`
- Meta-secant avg forgetting: `0.0312`
- Meta-secant backward transfer: `-0.0312`
- Benchmark JSON: `continual_benchmark.json`
- Benchmark report: `continual_benchmark_report.md`

## Ablation

- Best avg accuracy: `task_routing (0.5234)`
- Lowest avg forgetting: `no_adaptation (-0.0573)`
- Ablation JSON: `continual_ablation.json`
- Ablation report: `continual_ablation_report.md`
- Ablation table: `continual_ablation_table.md`
- Ablation CSV: `continual_ablation.csv`
- Ablation plot: `continual_ablation.png`

## Operator Ablation

- Best operator avg accuracy: `no_transport (0.5312)`
- Lowest operator avg forgetting: `kl_topk (-0.0625)`
- Operator Ablation JSON: `continual_operator_ablation.json`
- Operator Ablation report: `continual_operator_ablation_report.md`
- Operator Ablation table: `continual_operator_ablation_table.md`
- Operator Ablation CSV: `continual_operator_ablation.csv`
- Operator Ablation plot: `continual_operator_ablation.png`
- Profile note: Retention-oriented prototype routing preset from the capacity sweep.

## Paper Sync

- Source paper TeX: `asam_paper.tex`
- Synced paper TeX: `asam_paper_retention.tex`
- Standalone appendix TeX: `continual_appendix_retention.tex`

## Recommendation

- Use the ablation report and CSV as the main strategy-level paper tables.
- Use the operator ablation report and CSV for the mechanistic appendix and operator study.
- Use the benchmark report as the detailed continual-learning diagnostics appendix.
- Use the plot artifacts directly in slides and drafts.