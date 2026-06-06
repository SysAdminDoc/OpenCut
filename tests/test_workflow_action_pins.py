"""Static guards for immutable GitHub Actions references."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
RELEASE_SMOKE = REPO_ROOT / "scripts" / "release_smoke.py"

USE_RE = re.compile(r"^\s*-?\s*uses:\s*(?P<target>\S+)(?:\s+#\s*(?P<comment>\S+))?\s*$")
FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

EXPECTED_ACTION_PINS = {
    "actions/attest": ("281a49d4cbb0a72c9575a50d18f6deb515a11deb", "v4"),
    "actions/checkout": ("34e114876b0b11c390a56381ad16ebd13914f8d5", "v4"),
    "actions/download-artifact": ("d3f86a106a0bac45b974a628896c90dbdf5c8093", "v4"),
    "actions/github-script": ("f28e40c7f34bde8b3046d885e986cb6290c5673b", "v7"),
    "actions/setup-node": ("49933ea5288caeca8642d1e84afbd3f7d6820020", "v4"),
    "actions/setup-python": ("a26af69be951a213d495a4c3e4e4022e16d87065", "v5"),
    "actions/upload-artifact": ("ea165f8d65b6e75b540449e92b4886f43607fa02", "v4"),
}


def _workflow_uses_refs() -> list[tuple[Path, int, str, str | None]]:
    refs: list[tuple[Path, int, str, str | None]] = []
    for workflow in sorted(WORKFLOW_DIR.glob("*.yml")):
        for line_number, line in enumerate(workflow.read_text(encoding="utf-8").splitlines(), start=1):
            match = USE_RE.match(line)
            if match:
                refs.append((workflow, line_number, match.group("target"), match.group("comment")))
    return refs


def test_workflow_actions_are_full_sha_pinned_with_version_comments():
    refs = _workflow_uses_refs()

    assert refs
    for workflow, line_number, target, comment in refs:
        if target.startswith("./"):
            continue

        assert "@" in target, f"{workflow}:{line_number} missing @ref"
        action, ref = target.split("@", 1)
        expected = EXPECTED_ACTION_PINS.get(action)
        assert expected is not None, f"{workflow}:{line_number} has unreviewed action {action!r}"
        expected_sha, expected_comment = expected
        assert FULL_SHA_RE.fullmatch(ref), f"{workflow}:{line_number} uses mutable action ref {target!r}"
        assert ref == expected_sha, f"{workflow}:{line_number} pin drift for {action!r}"
        assert comment == expected_comment, f"{workflow}:{line_number} missing adjacent {expected_comment} comment"


def test_release_smoke_runs_workflow_action_pin_guard():
    smoke = RELEASE_SMOKE.read_text(encoding="utf-8")

    assert "tests/test_workflow_action_pins.py" in smoke
