# Continual ASAM Frontier Update

## Code changes

- Added a new full-pipeline preset `retention_no_transport` to the paper suite.
- Fixed a real plumbing gap: `transport_weight` now propagates through `experiments/run_continual_text_ablation.py`, so operator-level findings can be promoted into the main strategy ablation.
- The paper-suite report now records both sparse layout and transport weight.

## Main paper-suite result (256/128, 3 seeds)

Source: `experiments/paper_suite_retention_no_transport/continual_ablation_report.md`

- `task_routing`: accuracy `0.5234`, forgetting `-0.0469`
- `no_adaptation`: accuracy `0.5312`, forgetting `-0.0625`
- `correlation`: accuracy `0.5312`, forgetting `-0.0625`
- `meta_secant`: accuracy `0.5312`, forgetting `-0.0625`

### Interpretation

- The promoted preset does not only win inside operator ablation anymore; it now wins in the main strategy ablation too.
- Relative to `task_routing`, the prototype path gains `+0.0078` accuracy and `-0.0156` forgetting under the current metric sign convention.
- This is the clearest end-to-end positive signal obtained so far.

## Stronger-budget confirmation (512/256, 5 seeds)

Source: `experiments/frontier_round1_ablation_report.md`

- `task_routing`: accuracy `0.4961`, forgetting `0.0109`
- `no_adaptation`: accuracy `0.5016`, forgetting `-0.0078`
- `correlation`: accuracy `0.5016`, forgetting `-0.0078`
- `meta_secant`: accuracy `0.5016`, forgetting `-0.0078`

### Interpretation

- The gain survives at a larger budget, but it becomes smaller.
- The prototype route still beats `task_routing` on both metrics, so the direction remains promising.
- The main remaining bottleneck is no longer sparse layout selection; it is that `no_adaptation`, `correlation`, and `meta_secant` are effectively tied.

## Practical takeaway

- The best current continual preset is `prototype_slots_per_task=2`, `prototype_top_k=1`, `transport_weight=0.0`.
- The next high-value improvement target is the adaptation controller, not the sparse router itself.
