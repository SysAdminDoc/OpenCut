"""Static tests for the Adobe Premiere Pro typings tracker workflow."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "adobe-premierepro-versions.yml"
LABELS = REPO_ROOT / ".github" / "labels.yml"


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_probe_step_captures_exit_code_before_success_exit():
    text = _workflow_text()
    probe_block = re.search(
        r"- name: Probe registry \(drift check\)(?P<body>.*?)- name: Print drift summary",
        text,
        re.S,
    )
    assert probe_block, "probe step must exist before the summary step"
    body = probe_block.group("body")
    assert "continue-on-error: true" not in body
    assert "set +e" in body
    assert "probe_rc=$?" in body
    assert "set -e" in body
    assert 'echo "exit_code=${probe_rc}" >> "$GITHUB_OUTPUT"' in body
    assert "exit 0" in body


def test_tracker_issue_labels_are_shared_between_search_and_create():
    text = _workflow_text()
    assert "const trackerLabels = ['f251', 'uxp', 'tracking'];" in text
    assert "labels: trackerLabels.join(',')" in text
    assert "labels: trackerLabels," in text


def test_tracker_labels_are_declared_in_label_manifest():
    labels_text = LABELS.read_text(encoding="utf-8")
    names = set(re.findall(r'- name: "([^"]+)"', labels_text))
    assert {"f251", "uxp", "tracking"}.issubset(names)
