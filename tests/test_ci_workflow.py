from pathlib import Path

WORKFLOW_DIR = Path(".github/workflows")
CI_WORKFLOW = WORKFLOW_DIR / "tests.yml"
NIGHTLY_WORKFLOW = WORKFLOW_DIR / "nightly.yml"
RELEASE_WORKFLOW = WORKFLOW_DIR / "release.yml"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_ci_workflow_runs_on_all_pushes_and_pull_requests():
    workflow = _read(CI_WORKFLOW)

    assert "push:" in workflow
    assert "pull_request:" in workflow
    assert "workflow_dispatch:" in workflow
    assert "branches: [ master ]" not in workflow
    assert "branches: [master]" not in workflow


def test_ci_workflow_uses_current_github_action_versions():
    workflow = _read(CI_WORKFLOW)

    assert "actions/checkout@v4" in workflow
    assert "actions/setup-python@v5" in workflow
    assert "actions/upload-artifact@v4" in workflow
    assert "actions/checkout@v3" not in workflow
    assert "actions/setup-python@v4" not in workflow
    assert "actions/cache@v3" not in workflow
    assert "actions/upload-artifact@v3" not in workflow


def test_ci_workflow_runs_fast_gating_tests_not_full_suite():
    workflow = _read(CI_WORKFLOW)

    assert "pytest -q \\" in workflow
    assert "tests/test_basic.py" in workflow
    assert (
        "tests/test_continual_asam.py::test_sinkhorn_transport_matches_capacity_target" in workflow
    )
    assert (
        "tests/test_continual_ablation.py::test_build_benchmark_args_forwards_transport_weight_override"
        in workflow
    )
    assert "tests/test_continual_ablation.py \\" not in workflow
    assert "tests/test_continual_asam.py \\" not in workflow
    assert "tests/test_experiment_artifact_audit.py" in workflow
    assert "tests/test_continual_real_benchmark.py" not in workflow
    assert "tests/test_continual_training.py" not in workflow
    assert "pytest tests/" not in workflow
    assert "benchmarks/sota_comparison.py" not in workflow


def test_ci_workflow_installs_plotting_dependency_and_audits_artifacts():
    workflow = _read(CI_WORKFLOW)

    assert '".[dev,viz]"' in workflow
    assert "scripts/audit_experiment_artifacts.py" in workflow
    assert "experiments/paper_suite_canonical_smoke" in workflow
    assert "experiment_artifact_audit.json" in workflow
    assert (
        "scripts/audit_experiment_artifacts.py > experiment_artifact_audit.json || true"
        not in workflow
    )
    assert "|| true" not in workflow


def test_nightly_workflow_runs_full_matrix_and_full_suite():
    workflow = _read(NIGHTLY_WORKFLOW)

    assert "schedule:" in workflow
    assert "workflow_dispatch:" in workflow
    assert "pull_request:" not in workflow
    assert 'python-version: ["3.8", "3.9", "3.10", "3.11", "3.12"]' in workflow
    assert "pytest tests/ -v" in workflow
    assert "benchmarks/sota_comparison.py" in workflow


def test_release_workflow_builds_package_artifacts_without_publishing():
    workflow = _read(RELEASE_WORKFLOW)

    assert "tags:" in workflow
    assert '"v*"' in workflow
    assert "python -m build" in workflow
    assert "twine check dist/*" in workflow
    assert "actions/upload-artifact@v4" in workflow
    assert "pypi" not in workflow.lower()
