"""Static guards for Release Full token permissions."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "build.yml"
RELEASE_SMOKE = REPO_ROOT / "scripts" / "release_smoke.py"


def _workflow_text() -> str:
    return RELEASE_WORKFLOW.read_text(encoding="utf-8").replace("\r\n", "\n")


def _job_block(workflow: str, job_name: str) -> str:
    match = re.search(
        rf"^  {re.escape(job_name)}:\n(?P<body>.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)",
        workflow,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match, f"missing job block {job_name!r}"
    return match.group("body")


def test_release_full_workflow_defaults_to_read_only_token():
    workflow = _workflow_text()

    assert "\npermissions:\n  contents: read\n\njobs:\n" in workflow
    assert "\npermissions:\n  contents: write\n\njobs:\n" not in workflow


def test_release_full_build_job_has_read_only_token():
    build_job = _job_block(_workflow_text(), "build")

    assert "    permissions:\n      contents: read\n\n    strategy:\n" in build_job
    assert "contents: write" not in build_job
    assert "GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}" not in build_job
    assert "gh release upload" not in build_job


def test_release_upload_job_is_the_only_write_token_boundary():
    workflow = _workflow_text()
    release_job = _job_block(workflow, "release-upload")
    workflow_without_release_job = workflow.replace(release_job, "")

    assert "if: startsWith(github.ref, 'refs/tags/')" in release_job
    assert "needs: build" in release_job
    assert "\n    permissions:\n      contents: write\n\n    steps:\n" in release_job
    assert "actions/download-artifact@v4" in release_job
    assert "gh release upload" in release_job
    assert "GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}" in release_job
    assert "contents: write" not in workflow_without_release_job
    assert "GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}" not in workflow_without_release_job
    assert "gh release upload" not in workflow_without_release_job


def test_release_smoke_runs_workflow_permission_guard():
    smoke = RELEASE_SMOKE.read_text(encoding="utf-8")

    assert "tests/test_workflow_permissions.py" in smoke
