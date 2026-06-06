"""F251 — @adobe/premierepro npm version tracker tests.

The tool fetches Adobe's published `latest`, `beta`, and `release-*`
dist-tags and diffs them against a committed snapshot. CI surfaces drift as a
notification rather than a hard failure, so the contract these tests
pin is:

1. The committed snapshot file exists and validates.
2. The diff function correctly classifies (no-drift / new-beta /
   first-time / network-error) cases.
3. ``--check`` exits 0 on no-drift and 2 on drift, with stable JSON
   output suitable for the wrapping CI step.
4. Offline mode does not crash and produces a recognisable placeholder
   snapshot.
5. Adobe stable release-channel tags are tracked alongside latest/beta.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from opencut.tools import adobe_premierepro_versions as tool

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = (
    REPO_ROOT / "opencut" / "_generated" / "adobe_premierepro_versions.json"
)


# ---------------------------------------------------------------------------
# Committed snapshot shape
# ---------------------------------------------------------------------------


def test_committed_snapshot_exists():
    assert SNAPSHOT.is_file(), (
        "F251 snapshot must be committed at opencut/_generated/"
        "adobe_premierepro_versions.json"
    )


def test_committed_snapshot_is_valid_json():
    data = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert data["package"] == "@adobe/premierepro"
    assert data["snapshot_version"] == tool.SNAPSHOT_VERSION
    assert data["status"] in {"ok", "network_error", "parse_error"}


def test_committed_snapshot_carries_dist_tags_when_ok():
    data = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    if data["status"] != "ok":
        pytest.skip("snapshot was captured offline; dist-tag contract not asserted")
    tags = data.get("dist_tags") or {}
    # At least one tracked tag must be present in a healthy snapshot.
    assert any(tag in tags for tag in tool.TRACKED_TAGS), (
        f"Committed snapshot must carry at least one tracked tag "
        f"(expected any of {tool.TRACKED_TAGS}); got {sorted(tags)}"
    )


def test_committed_snapshot_tracks_release_channel_tags_when_present():
    data = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    if data["status"] != "ok":
        pytest.skip("snapshot was captured offline; dist-tag contract not asserted")
    release_tags = {
        tag: version
        for tag, version in (data.get("dist_tags") or {}).items()
        if tag.startswith("release-")
    }
    if not release_tags:
        pytest.skip("Adobe has not published release-* tags in this snapshot")
    tracked = data.get("tracked_dist_tags") or {}
    for tag, version in release_tags.items():
        assert tracked.get(tag) == version
        assert version in (data.get("tracked_versions") or [])


# ---------------------------------------------------------------------------
# Diff function contract
# ---------------------------------------------------------------------------


def _ok_snapshot(**overrides) -> dict:
    base = {
        "package": tool.PACKAGE,
        "status": "ok",
        "error": None,
        "snapshot_version": tool.SNAPSHOT_VERSION,
        "recorded_at": "2026-05-17T00:00:00Z",
        "dist_tags": {
            "beta": "26.3.0-beta.67",
            "latest": "26.2.0",
            "release-26.2": "26.2.1",
        },
        "tracked_dist_tags": {
            "latest": "26.2.0",
            "beta": "26.3.0-beta.67",
            "release-26.2": "26.2.1",
        },
        "tracked_versions": ["26.2.0", "26.3.0-beta.67", "26.2.1"],
        "latest_release_versions": ["26.3.0-beta.67", "26.2.0"],
        "release_count": 2,
        "notes": [],
    }
    base.update(overrides)
    return base


def test_diff_no_drift_when_identical():
    snap = _ok_snapshot()
    diff = tool.diff_snapshots(snap, snap)
    assert diff == {"changed": False, "fields": {}}


def test_diff_ignores_timestamp_changes():
    """Recorded-at timestamp drifts every minute — it must not flag."""
    committed = _ok_snapshot(recorded_at="2026-05-17T00:00:00Z")
    live = _ok_snapshot(recorded_at="2026-05-17T01:23:45Z")
    diff = tool.diff_snapshots(committed, live)
    assert not diff["changed"]


def test_diff_flags_new_beta_release():
    committed = _ok_snapshot()
    live = _ok_snapshot(
        dist_tags={"beta": "26.3.0-beta.99", "latest": "26.2.0"},
        latest_release_versions=["26.3.0-beta.99", "26.3.0-beta.67", "26.2.0"],
        release_count=3,
    )
    diff = tool.diff_snapshots(committed, live)
    assert diff["changed"] is True
    assert "dist_tags" in diff["fields"]
    assert diff["fields"]["dist_tags"]["beta"] == {
        "from": "26.3.0-beta.67",
        "to": "26.3.0-beta.99",
    }
    added = diff["fields"]["latest_release_versions"]["added"]
    assert "26.3.0-beta.99" in added


def test_diff_flags_new_latest_release():
    committed = _ok_snapshot()
    live = _ok_snapshot(
        dist_tags={"beta": "26.3.0-beta.67", "latest": "26.3.0"},
        release_count=3,
    )
    diff = tool.diff_snapshots(committed, live)
    assert diff["changed"] is True
    assert diff["fields"]["dist_tags"]["latest"] == {
        "from": "26.2.0",
        "to": "26.3.0",
    }


def test_select_tracked_dist_tags_includes_release_channels():
    dist_tags = {
        "latest": "26.2.0",
        "beta": "26.3.0-beta.85",
        "release-26.1": "26.1.4",
        "release-26.2": "26.2.1",
        "next": "27.0.0-alpha.1",
    }
    tracked = tool._select_tracked_dist_tags(dist_tags)
    assert list(tracked) == ["latest", "beta", "release-26.1", "release-26.2"]
    assert "next" not in tracked


def test_diff_first_time_when_committed_is_none():
    live = _ok_snapshot()
    diff = tool.diff_snapshots(None, live)
    assert diff["changed"] is True
    assert "status" in diff["fields"]


def test_diff_flags_network_error_status():
    committed = _ok_snapshot()
    live = _ok_snapshot(status="network_error")
    diff = tool.diff_snapshots(committed, live)
    assert diff["changed"]
    assert diff["fields"]["status"] == {"from": "ok", "to": "network_error"}


# ---------------------------------------------------------------------------
# Offline / fallback behaviour
# ---------------------------------------------------------------------------


def test_offline_mode_produces_placeholder():
    result = tool.build_versions(offline=True)
    assert result["status"] == "network_error"
    assert "Offline placeholder" in (result.get("notes") or [""])[0]


def test_semver_key_handles_prerelease_and_weird_suffixes():
    # Stable sort: 26.3.0 > 26.3.0-beta.67 > 26.2.0 > 26.2.0-rc.1
    versions = ["26.2.0", "26.3.0-beta.67", "26.3.0", "26.2.0-rc.1"]
    versions.sort(key=tool._semver_key, reverse=True)
    assert versions[0] == "26.3.0"
    assert versions[-1] == "26.2.0-rc.1"


# ---------------------------------------------------------------------------
# CLI contract
# ---------------------------------------------------------------------------


def _run_cli(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "opencut.tools.adobe_premierepro_versions", *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=60,
    )


def test_cli_offline_check_exit_code_is_drift():
    """Offline `--check` will compare network_error against committed snapshot;
    a healthy committed snapshot means drift is detected → exit 2."""
    result = _run_cli("--check", "--offline", "--json")
    if result.returncode == 0:
        # Committed snapshot was already captured offline; tolerate it but
        # at least assert the JSON shape is intact.
        payload = json.loads(result.stdout)
        assert payload["package"] == "@adobe/premierepro"
        return
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["package"] == "@adobe/premierepro"
    assert "drift" in payload
    assert payload["drift"]["changed"] is True


def test_cli_offline_check_text_mode_does_not_crash():
    result = _run_cli("--check", "--offline")
    assert result.returncode in (0, 2)
    assert "@adobe/premierepro" in result.stdout


def test_cli_offline_write_emits_placeholder(tmp_path):
    target = tmp_path / "snapshot.json"
    result = _run_cli("--offline", "--output", str(target))
    assert result.returncode == 0
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["status"] == "network_error"
    assert data["package"] == "@adobe/premierepro"
