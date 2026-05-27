# Canonical Paper Suite Smoke Run

This directory contains a small, CPU-only canonical smoke run for the continual
ASAM paper pipeline. It is intended as provenance and pipeline-authenticity
evidence, not as a paper-scale result.

## Command

```powershell
$env:PYTHONPATH=(Get-Location).Path
python scripts/run_continual_paper_suite.py `
  --output-dir experiments/paper_suite_canonical_smoke `
  --candidate-profile retention_no_transport `
  --num-seeds 1 `
  --max-train-samples 16 `
  --max-val-samples 8 `
  --max-length 64 `
  --batch-size 4 `
  --dim 32 `
  --num-heads 2 `
  --device cpu
```

## Provenance Checks

- `paper_suite_manifest.json` records the pre-run git commit, dirty state,
  platform, Python/PyTorch versions, dataset bounds, redacted argv, timestamps,
  and SHA-256 hashes for generated outputs.
- `continual_ablation.json` includes the `dual_transport` strategy.
- `continual_operator_ablation.json` includes `sinkhorn_topk`, `kl_topk`, and
  `masked_sinkhorn_topk` operator variants.
- Local audit command:

```powershell
python scripts/audit_experiment_artifacts.py experiments/paper_suite_canonical_smoke
```

The suite audited as `CURRENT` with zero blocking issues. The audit reports
non-blocking semantic-duplicate warnings because some smoke variants
intentionally share equivalent tiny-run outputs under the constrained one-seed
setting.
