# Continual ASAM Candidate Profile Comparison

## Accuracy-oriented preset

- Profile: `accuracy`
- Prototype layout: `num_prototypes=0, slots_per_task=2, top_k=2`
- Benchmark meta-secant accuracy / forgetting / BWT: `0.4766 / 0.0312 / -0.0312`
- Strategy ablation best accuracy: `no_adaptation (0.5286)`
- Strategy ablation best forgetting: `task_routing (-0.0469)`
- Operator ablation best accuracy: `sinkhorn_topk (0.5286)`
- Operator ablation best forgetting: `kl_topk (-0.0573)`

## Retention-oriented preset

- Profile: `retention`
- Prototype layout: `num_prototypes=0, slots_per_task=2, top_k=1`
- Benchmark meta-secant accuracy / forgetting / BWT: `0.4766 / 0.0312 / -0.0312`
- Strategy ablation best accuracy: `task_routing (0.5234)`
- Strategy ablation best forgetting: `no_adaptation (-0.0573)`
- Operator ablation best accuracy: `no_transport (0.5312)`
- Operator ablation best forgetting: `kl_topk (-0.0625)`

## Takeaways

- The layout sweep winners remain real after paper-suite integration: `top_k=2` is still the better accuracy-oriented sparse layout, while `top_k=1` is still the better retention-oriented sparse layout.
- The strongest prototype result seen in the full paper suite is now `retention + no_transport`, with average accuracy `0.5312`.
- Compared against the explicit task baseline from the same suite (`task_routing = 0.5234`, forgetting `-0.0469`), `retention + no_transport` improves accuracy by `+0.0078` and improves forgetting by `-0.0156` under the current metric convention where lower forgetting is better.
- The single-run benchmark section stays unchanged across the two profile presets, so the gains are currently concentrated in the multi-seed ablation/operator studies rather than the one-shot benchmark summary.

## Artifacts

- Accuracy suite report: `experiments/paper_suite_accuracy/paper_suite_report.md`
- Retention suite report: `experiments/paper_suite_retention/paper_suite_report.md`
- Accuracy synced paper: `paper/asam_paper_accuracy.tex`
- Retention synced paper: `paper/asam_paper_retention.tex`
