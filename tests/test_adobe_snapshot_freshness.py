"""A tracked platform snapshot must not age out silently.

``opencut/_generated/adobe_premierepro_versions.json`` carries a ``recorded_at``
that every refresh wrote and nothing ever read. The snapshot sat at 2026-06-25
while OpenCut planned its UXP migration against it, through exactly the period
Adobe was retiring ExtendScript support in Premiere Pro.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from opencut.tools.adobe_premierepro_versions import (
    MAX_SNAPSHOT_AGE_DAYS,
    check_snapshot_freshness,
    main,
    snapshot_age_days,
)

NOW = datetime(2026, 9, 5, tzinfo=timezone.utc)


def _snapshot(days_old: float, **overrides) -> dict:
    recorded = NOW - timedelta(days=days_old)
    payload = {
        "status": "ok",
        "error": None,
        "recorded_at": recorded.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "snapshot_version": 2,
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# Age
# ---------------------------------------------------------------------------

def test_a_recent_snapshot_passes():
    verdict = check_snapshot_freshness(_snapshot(10), now=NOW)
    assert verdict["state"] == "fresh"
    assert verdict["ok"] is True


def test_a_snapshot_past_the_limit_fails_and_names_the_file_and_age():
    verdict = check_snapshot_freshness(_snapshot(MAX_SNAPSHOT_AGE_DAYS + 5), now=NOW)
    assert verdict["state"] == "stale"
    assert verdict["ok"] is False
    assert "adobe_premierepro_versions.json" in verdict["detail"]
    assert str(MAX_SNAPSHOT_AGE_DAYS) in verdict["detail"]
    assert "adobe_premierepro_versions" in verdict["detail"], "the fix command must be named"


def test_the_boundary_is_not_stale():
    assert check_snapshot_freshness(_snapshot(MAX_SNAPSHOT_AGE_DAYS - 0.5), now=NOW)["ok"] is True


def test_refreshing_the_snapshot_clears_the_failure():
    """The acceptance: a refresh has to be what fixes it."""
    stale = _snapshot(MAX_SNAPSHOT_AGE_DAYS + 30)
    assert check_snapshot_freshness(stale, now=NOW)["ok"] is False

    refreshed = _snapshot(0)
    assert check_snapshot_freshness(refreshed, now=NOW)["ok"] is True


# ---------------------------------------------------------------------------
# Old is not the same as broken
# ---------------------------------------------------------------------------

def test_an_upstream_failure_is_reported_separately_from_staleness():
    """A placeholder written when the registry was unreachable is not stale data."""
    broken = _snapshot(2, status="error", error="registry unreachable")
    verdict = check_snapshot_freshness(broken, now=NOW)
    assert verdict["state"] == "upstream_error"
    assert verdict["ok"] is False
    assert "registry unreachable" in verdict["detail"]
    assert "stale" not in verdict["state"]


def test_an_upstream_failure_is_flagged_even_when_recent():
    broken = _snapshot(0, status="error", error="502")
    assert check_snapshot_freshness(broken, now=NOW)["state"] == "upstream_error"


def test_a_snapshot_without_a_timestamp_is_not_silently_accepted():
    undated = _snapshot(1)
    undated.pop("recorded_at")
    assert check_snapshot_freshness(undated, now=NOW)["state"] == "undated"


def test_an_unparseable_timestamp_is_not_silently_accepted():
    assert check_snapshot_freshness(_snapshot(1, recorded_at="whenever"), now=NOW)["state"] == "undated"


def test_a_missing_snapshot_is_reported(tmp_path, monkeypatch):
    """Point at a path that really is absent.

    This originally called the checker without controlling SNAPSHOT_PATH, so it
    only passed while "missing" was the single verdict for a None snapshot. It
    is not: a file that exists but will not parse also yields None, and calling
    that "missing" sends the reader looking for a file that is right there.
    """
    from opencut.tools import adobe_premierepro_versions as module

    monkeypatch.setattr(module, "SNAPSHOT_PATH", tmp_path / "absent.json")
    verdict = check_snapshot_freshness(None, now=NOW)
    assert verdict["state"] == "missing"
    assert verdict["ok"] is False


def test_snapshot_age_handles_a_naive_timestamp():
    assert snapshot_age_days({"recorded_at": "2026-09-01T00:00:00"}, now=NOW) == pytest.approx(4.0, abs=0.1)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

def test_cli_exits_nonzero_on_a_stale_snapshot(tmp_path, capsys):
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(_snapshot(MAX_SNAPSHOT_AGE_DAYS + 100)), encoding="utf-8")
    assert main(["--check-freshness", "--output", str(path)]) == 1
    assert "stale" in capsys.readouterr().out


def test_cli_exits_zero_on_a_fresh_snapshot(tmp_path, capsys):
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(_snapshot(1)), encoding="utf-8")
    assert main(["--check-freshness", "--output", str(path)]) == 0
    assert "fresh" in capsys.readouterr().out


def test_cli_makes_no_network_call(tmp_path, monkeypatch):
    """The gate has to run in an offline release build."""
    from opencut.tools import adobe_premierepro_versions as module

    def _boom(*args, **kwargs):
        raise AssertionError("the freshness check reached the network")

    monkeypatch.setattr(module, "_http_get", _boom)
    monkeypatch.setattr(module, "fetch_registry", _boom)

    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(_snapshot(1)), encoding="utf-8")
    assert main(["--check-freshness", "--output", str(path)]) == 0


def test_the_release_gate_runs_the_freshness_check():
    import importlib.util
    import sys
    from pathlib import Path

    spec_path = Path(__file__).resolve().parents[1] / "scripts" / "release_smoke.py"
    spec = importlib.util.spec_from_file_location("release_smoke_for_freshness_test", spec_path)
    module = importlib.util.module_from_spec(spec)
    # Register before exec: the module defines dataclasses, and dataclasses
    # resolves annotations through sys.modules[cls.__module__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    commands = [" ".join(str(part) for part in cmd) for _label, cmd in module.GENERATED_DOC_CHECKS]
    assert any("--check-freshness" in cmd for cmd in commands), (
        "the release gate does not run the snapshot freshness check"
    )


def test_the_committed_snapshot_is_currently_fresh():
    """If this fails, refresh the snapshot -- that is the point of the gate."""
    from opencut.tools.adobe_premierepro_versions import load_committed_snapshot

    verdict = check_snapshot_freshness(load_committed_snapshot())
    assert verdict["ok"], verdict["detail"]


# ---------------------------------------------------------------------------
# Regressions found by adversarial review
# ---------------------------------------------------------------------------

def test_a_future_dated_snapshot_is_refused():
    """A stamp in the future makes the age test unsatisfiable forever."""
    verdict = check_snapshot_freshness(_snapshot(-3000), now=NOW)
    assert verdict["ok"] is False
    assert verdict["state"] == "future_dated"
    assert "future" in verdict["detail"]


def test_a_snapshot_written_moments_ago_is_not_treated_as_future_dated():
    """Second-level clock jitter must not trip the guard."""
    assert check_snapshot_freshness(_snapshot(-0.0001), now=NOW)["ok"] is True


def test_a_corrupt_snapshot_is_not_reported_as_missing(tmp_path, monkeypatch):
    """The file is there; saying it is absent sends you looking in the wrong place."""
    from opencut.tools import adobe_premierepro_versions as module

    corrupt = tmp_path / "snapshot.json"
    corrupt.write_text("{ not json", encoding="utf-8")
    monkeypatch.setattr(module, "SNAPSHOT_PATH", corrupt)

    assert module.load_committed_snapshot(corrupt) is None
    verdict = module.check_snapshot_freshness(None, now=NOW)
    assert verdict["state"] == "unreadable"
    assert "could not be parsed" in verdict["detail"]


def test_a_genuinely_absent_snapshot_still_reports_missing(tmp_path, monkeypatch):
    from opencut.tools import adobe_premierepro_versions as module

    monkeypatch.setattr(module, "SNAPSHOT_PATH", tmp_path / "nope.json")
    assert module.check_snapshot_freshness(None, now=NOW)["state"] == "missing"
