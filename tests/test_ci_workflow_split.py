"""F206 guardrails for the PR-fast / release-full workflow split."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "build.yml"
PR_FAST_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "pr-fast.yml"
RELEASE_SMOKE = REPO_ROOT / "scripts" / "release_smoke.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def test_release_full_workflow_no_longer_runs_on_pull_requests():
    text = _read(RELEASE_WORKFLOW)

    assert "name: Release Full" in text
    assert "pull_request:" not in text
    assert "workflow_dispatch:" in text
    assert "push:" in text
    assert "branches: [main]" in text
    assert "tags:" in text
    assert "windows-latest" in text
    assert "ubuntu-latest" in text
    assert "macos-latest" in text


def test_pr_fast_workflow_is_linux_only_and_pull_request_scoped():
    text = _read(PR_FAST_WORKFLOW)

    assert "name: PR Fast" in text
    assert "pull_request:" in text
    assert "runs-on: ubuntu-latest" in text
    assert "windows-latest" not in text
    assert "macos-latest" not in text
    assert "pyinstaller" not in text.lower()
    assert "Build Windows installer" not in text
    assert "Notarize macOS bundle" not in text


def test_pr_fast_workflow_runs_fast_release_smoke_subset():
    """RESEARCH_FEATURE_PLAN_2026-05-25.md E3: supply-chain triad must run on PR.

    pip-audit, npm-advisory, and esbuild-pin MUST run on every PR so a
    vulnerable dep fails fast. Only cross-OS / heavyweight gates remain
    skipped (panel-source, adobe-premierepro-versions).
    """
    text = _read(PR_FAST_WORKFLOW)

    assert "python scripts/release_smoke.py --json" in text
    # Gates that MUST be skipped on PR-fast (cross-OS-required).
    for step in ("panel-source", "adobe-premierepro-versions"):
        assert f"--skip {step}" in text, (
            f"PR-fast must skip the cross-OS gate {step!r}"
        )
    # Gates that MUST NOT be skipped on PR-fast (cheap supply-chain checks).
    for step in ("pip-audit", "npm-advisory", "esbuild-pin"):
        assert f"--skip {step}" not in text, (
            f"PR-fast must run the supply-chain gate {step!r} "
            "(RESEARCH_FEATURE_PLAN_2026-05-25.md E3)"
        )


def test_f206_guardrail_is_in_release_smoke_pytest_fast():
    smoke = _read(RELEASE_SMOKE)

    assert "tests/test_ci_workflow_split.py" in smoke
