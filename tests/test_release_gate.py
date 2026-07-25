"""Fail-closed local release receipt tests."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "release_gate.py"


def _module():
    spec = importlib.util.spec_from_file_location("release_gate_under_test", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _receipt(module, now: datetime, source: dict | None = None) -> dict:
    return {
        "schema_version": module.RECEIPT_SCHEMA_VERSION,
        "status": "ok",
        "strict": True,
        "generated_at": now.isoformat().replace("+00:00", "Z"),
        "source": source or {"commit": "abc123", "branch": "main", "dirty_paths": []},
        "steps": [{"name": name, "status": "ok"} for name in sorted(module.REQUIRED_STEPS)],
    }


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_porcelain_parser_preserves_status_columns_and_ignores_local_docs():
    module = _module()

    assert module._parse_status_paths(" M RESEARCH.md\n M ROADMAP.md\n M opencut/server.py\n?? new file.txt\n") == [
        "new file.txt",
        "opencut/server.py",
    ]


def test_valid_receipt_is_bound_to_current_clean_commit(tmp_path):
    module = _module()
    now = datetime(2026, 7, 25, 12, 0, tzinfo=timezone.utc)
    path = tmp_path / "receipt.json"
    payload = _receipt(module, now)
    _write(path, payload)

    assert (
        module.validate_receipt(
            path,
            now=now + timedelta(minutes=10),
            source_state={"commit": "abc123", "branch": "main", "dirty_paths": []},
        )
        == payload
    )


@pytest.mark.parametrize("failure", ["stale", "missing-step", "skipped-step", "source-drift", "dirty"])
def test_receipt_fails_closed_for_incomplete_or_stale_evidence(tmp_path, failure):
    module = _module()
    now = datetime(2026, 7, 25, 12, 0, tzinfo=timezone.utc)
    path = tmp_path / "receipt.json"
    payload = _receipt(module, now)
    current = {"commit": "abc123", "branch": "main", "dirty_paths": []}

    if failure == "stale":
        clock = now + timedelta(hours=3)
    else:
        clock = now + timedelta(minutes=5)
    if failure == "missing-step":
        payload["steps"].pop()
    if failure == "skipped-step":
        payload["steps"][0]["status"] = "skipped"
    if failure == "source-drift":
        current["commit"] = "different"
    if failure == "dirty":
        current["dirty_paths"] = ["opencut/server.py"]
    _write(path, payload)

    with pytest.raises(module.ReleaseGateError):
        module.validate_receipt(path, now=clock, source_state=current)


def test_failed_smoke_never_writes_receipt(monkeypatch, tmp_path):
    module = _module()
    target = tmp_path / "receipt.json"
    monkeypatch.setattr(
        module,
        "current_source_state",
        lambda: {"commit": "abc123", "branch": "main", "dirty_paths": []},
    )
    monkeypatch.setattr(
        module,
        "_run",
        lambda *args, **kwargs: module.subprocess.CompletedProcess(
            args[0], 1, json.dumps({"status": "fail", "steps": []}), ""
        ),
    )

    with pytest.raises(module.ReleaseGateError):
        module.run_verification(target)
    assert not target.exists()


def test_source_drift_during_smoke_never_writes_receipt(monkeypatch, tmp_path):
    module = _module()
    target = tmp_path / "receipt.json"
    source_states = iter(
        [
            {"commit": "abc123", "branch": "main", "dirty_paths": []},
            {"commit": "different", "branch": "main", "dirty_paths": []},
        ]
    )
    monkeypatch.setattr(module, "current_source_state", lambda: next(source_states))
    monkeypatch.setattr(
        module,
        "_run",
        lambda *args, **kwargs: module.subprocess.CompletedProcess(
            args[0],
            0,
            json.dumps(
                {
                    "status": "ok",
                    "steps": [{"name": name, "status": "ok"} for name in sorted(module.REQUIRED_STEPS)],
                }
            ),
            "",
        ),
    )

    with pytest.raises(module.ReleaseGateError, match="source changed"):
        module.run_verification(target)
    assert not target.exists()


def test_artifact_promotion_requires_smoke_before_tag(monkeypatch, tmp_path):
    module = _module()
    receipt = tmp_path / "receipt.json"
    artifact = tmp_path / "OpenCut-WPF-Setup-1.44.0.exe"
    promotion = tmp_path / "promotion.json"
    receipt.write_text("{}\n", encoding="utf-8")
    artifact.write_bytes(b"installer")
    calls: list[tuple] = []

    monkeypatch.setattr(
        module,
        "validate_receipt",
        lambda path: {"source": {"commit": "abc123", "branch": "main"}},
    )
    monkeypatch.setattr(
        module,
        "_smoke_artifact",
        lambda path, kind: {
            "name": path.name,
            "sha256": module._sha256(path),
            "smoke": kind,
            "status": "ok",
        },
    )
    monkeypatch.setattr(module, "_git", lambda *args: "")
    monkeypatch.setattr(
        module,
        "_run",
        lambda command, **kwargs: (
            calls.append(tuple(command)) or module.subprocess.CompletedProcess(command, 0, "", "")
        ),
    )
    fake_root = tmp_path / "repo"
    (fake_root / "opencut").mkdir(parents=True)
    (fake_root / "opencut" / "__init__.py").write_text(
        '__version__ = "1.44.0"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "REPO_ROOT", fake_root)

    payload = module.promote_artifact(
        receipt,
        artifact,
        promotion,
        artifact_kind="wpf",
        tag="v1.44.0",
    )

    assert payload["artifact"]["status"] == "ok"
    assert payload["artifact"]["smoke"] == "wpf"
    assert json.loads(promotion.read_text(encoding="utf-8"))["tag"] == "v1.44.0"
    assert calls == [("git", "tag", "-a", "v1.44.0", "-m", "OpenCut v1.44.0")]
