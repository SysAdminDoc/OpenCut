"""F176 follow-up — tests for the opt-in eval-dataset download runner.

The runner is a pure planner by default — every test here exercises
the planning path without touching the network. One execution test
uses a local file:// URL so we exercise the download path end-to-end
without a real HTTP request.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from opencut.tools import download_eval_dataset as runner

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Planning — pure functions
# ---------------------------------------------------------------------------


def test_plan_for_unknown_dataset_returns_unknown_status(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENCUT_DOWNLOAD_EVAL", raising=False)
    plan = runner.build_plan("not-a-real-dataset", target_dir=tmp_path)
    assert plan.status == "unknown"
    assert "F176 registry" in plan.reason


def test_plan_blocks_without_opt_in_env(tmp_path, monkeypatch):
    """Default state: OPENCUT_DOWNLOAD_EVAL unset → blocked."""
    monkeypatch.delenv("OPENCUT_DOWNLOAD_EVAL", raising=False)
    plan = runner.build_plan("davis_2017", target_dir=tmp_path)
    assert plan.status == "blocked"
    assert "OPENCUT_DOWNLOAD_EVAL" in plan.reason


def test_plan_force_overrides_env_check(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENCUT_DOWNLOAD_EVAL", raising=False)
    plan = runner.build_plan("davis_2017", target_dir=tmp_path, force=True)
    # davis_2017 is acquisition="auto" + commercial_use_ok=True, so force
    # alone is enough to clear all gates.
    assert plan.status == "ok"
    assert plan.dataset_id == "davis_2017"
    assert plan.download_url


def test_plan_env_var_opt_in_unblocks(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENCUT_DOWNLOAD_EVAL", "1")
    plan = runner.build_plan("davis_2017", target_dir=tmp_path)
    assert plan.status == "ok"


def test_plan_blocks_manual_acquisition_dataset(tmp_path, monkeypatch):
    """REDS is acquisition='manual' — must require the explicit licence flag."""
    monkeypatch.setenv("OPENCUT_DOWNLOAD_EVAL", "1")
    plan = runner.build_plan("reds_120", target_dir=tmp_path)
    assert plan.status == "blocked"
    assert "acquisition='manual'" in plan.reason
    assert "--accept-noncommercial-license" in plan.reason


def test_plan_accept_noncommercial_unblocks_manual_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENCUT_DOWNLOAD_EVAL", "1")
    plan = runner.build_plan(
        "reds_120",
        target_dir=tmp_path,
        accept_noncommercial_license=True,
    )
    # REDS has no download_url in our registry, so it should land on
    # the third gate (missing URL) rather than the licence gate.
    assert plan.status == "blocked"
    assert "no canonical download_url" in plan.reason


def test_plan_target_dir_resolution(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENCUT_DOWNLOAD_EVAL", "1")
    plan = runner.build_plan("davis_2017", target_dir=tmp_path)
    assert plan.target_dir.endswith("davis_2017")
    # Must live inside the supplied target_dir.
    assert plan.target_dir.startswith(str(tmp_path))


def test_plan_default_target_dir_under_dot_opencut(monkeypatch):
    monkeypatch.setenv("OPENCUT_DOWNLOAD_EVAL", "1")
    plan = runner.build_plan("davis_2017")
    assert ".opencut" in plan.target_dir
    assert plan.target_dir.endswith("davis_2017")


def test_plan_dry_run_flag_defaults_true(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENCUT_DOWNLOAD_EVAL", "1")
    plan = runner.build_plan("davis_2017", target_dir=tmp_path)
    # The planner never touches the network; dry_run stays True until
    # execute_plan runs.
    assert plan.dry_run is True


# ---------------------------------------------------------------------------
# Execution — uses a file:// URL to avoid touching the real internet
# ---------------------------------------------------------------------------


def test_execute_plan_downloads_via_file_url(tmp_path, monkeypatch):
    """Build a plan, point download_url at a local file:// URL, execute."""
    monkeypatch.setenv("OPENCUT_DOWNLOAD_EVAL", "1")

    # Stage a fake "dataset asset" on disk.
    asset = tmp_path / "fake_asset.bin"
    asset.write_bytes(b"hello, world -- opencut eval payload")
    asset_url = asset.as_uri()  # file:///...

    plan = runner.build_plan("davis_2017", target_dir=tmp_path)
    assert plan.status == "ok"
    # Swap the upstream download URL for the local one.
    plan.download_url = asset_url

    finished = runner.execute_plan(plan)
    assert finished.status == "ok"
    assert finished.dry_run is False
    assert "downloaded to" in finished.reason

    downloaded_file = Path(plan.target_dir) / asset.name
    assert downloaded_file.is_file()
    assert downloaded_file.read_bytes() == asset.read_bytes()


def test_execute_plan_skips_when_status_not_ok(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENCUT_DOWNLOAD_EVAL", raising=False)
    plan = runner.build_plan("davis_2017", target_dir=tmp_path)
    assert plan.status == "blocked"
    finished = runner.execute_plan(plan)
    # execute_plan must not flip a blocked plan to ok.
    assert finished.status == "blocked"
    assert finished.dry_run is False


def test_execute_plan_records_failure_reason(tmp_path, monkeypatch):
    """A bad URL must surface as status='blocked' with the IO error."""
    monkeypatch.setenv("OPENCUT_DOWNLOAD_EVAL", "1")
    plan = runner.build_plan("davis_2017", target_dir=tmp_path)
    assert plan.status == "ok"
    plan.download_url = "file:///definitely/not/a/real/path/xyz.bin"

    finished = runner.execute_plan(plan)
    assert finished.status == "blocked"
    assert "download failed" in finished.reason


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _run_cli(*args, env=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "opencut.tools.download_eval_dataset", *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=60,
        env=env,
    )


def test_cli_list_prints_all_datasets():
    result = _run_cli("--list")
    assert result.returncode == 0
    assert "davis_2017" in result.stdout
    assert "vbench" in result.stdout
    # Modality flag should appear for at least one row.
    assert "[video" in result.stdout or "[ video" in result.stdout


def test_cli_list_json_emits_manifest():
    result = _run_cli("--list", "--json")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["count"] >= 10
    assert payload["modalities"]


def test_cli_list_filtered_by_modality():
    result = _run_cli("--list", "--modality", "speech", "--json")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert all(d["modality"] == "speech" for d in payload["datasets"])


def test_cli_blocked_when_opt_in_not_set(monkeypatch):
    import os as _os

    env = _os.environ.copy()
    env.pop("OPENCUT_DOWNLOAD_EVAL", None)
    result = _run_cli("davis_2017", env=env)
    # Exit code 2 == blocked
    assert result.returncode == 2
    assert "BLOCKED" in result.stdout
    assert "OPENCUT_DOWNLOAD_EVAL" in result.stdout


def test_cli_force_unblocks(tmp_path, monkeypatch):
    import os as _os

    env = _os.environ.copy()
    env.pop("OPENCUT_DOWNLOAD_EVAL", None)
    result = _run_cli(
        "davis_2017",
        "--force",
        "--target-dir", str(tmp_path),
        "--json",
        env=env,
    )
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["dry_run"] is True  # without --execute


def test_cli_unknown_dataset_exits_3():
    result = _run_cli("not-a-real-dataset", "--force")
    assert result.returncode == 3
    assert "UNKNOWN" in result.stdout


def test_cli_required_dataset_id_when_not_listing():
    result = _run_cli()
    assert result.returncode != 0
    assert "dataset_id is required" in (result.stderr + result.stdout)
