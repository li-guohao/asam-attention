from pathlib import Path


def test_tests_workflow_runs_on_all_pushes_and_pull_requests():
    workflow = Path(".github/workflows/tests.yml").read_text(encoding="utf-8")

    assert "push:" in workflow
    assert "pull_request:" in workflow
    assert "branches: [ master ]" not in workflow
