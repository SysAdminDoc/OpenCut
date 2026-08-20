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
            args[0],
            1,
            json.dumps(
                {
                    "status": "fail",
                    "steps": [
                        {"name": "panel-rendered", "status": "fail"},
                        {"name": "adobe-premierepro-versions", "status": "warn"},
                    ],
                }
            ),
            "",
        ),
    )

    with pytest.raises(module.ReleaseGateError, match=r"panel-rendered(?!.*adobe)"):
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


# ---------------------------------------------------------------------------
# F318 — published digests for unsigned artifacts
# ---------------------------------------------------------------------------


def test_digests_cover_every_downloadable_artifact(tmp_path):
    module = _module()
    artifacts = tmp_path / "dist"
    (artifacts / "nested").mkdir(parents=True)
    (artifacts / "OpenCut-Setup-9.9.9.exe").write_bytes(b"installer")
    (artifacts / "nested" / "opencut-9.9.9.whl").write_bytes(b"wheel")

    payload = module.digest_artifacts(artifacts, tmp_path / "release-digests.json")

    names = {entry["name"] for entry in payload["artifacts"]}
    assert names == {"OpenCut-Setup-9.9.9.exe", "nested/opencut-9.9.9.whl"}
    assert payload["algorithm"] == "sha256"
    assert payload["unsigned"] is True
    assert all(len(entry["sha256"]) == 64 for entry in payload["artifacts"])


def test_digests_match_the_bytes_on_disk(tmp_path):
    import hashlib

    module = _module()
    artifacts = tmp_path / "dist"
    artifacts.mkdir()
    body = b"exactly these bytes"
    (artifacts / "OpenCut-Setup-9.9.9.exe").write_bytes(body)

    payload = module.digest_artifacts(artifacts, tmp_path / "digests.json")

    assert payload["artifacts"][0]["sha256"] == hashlib.sha256(body).hexdigest()
    assert payload["artifacts"][0]["size_bytes"] == len(body)


def test_build_logs_are_not_published_as_artifacts(tmp_path):
    module = _module()
    artifacts = tmp_path / "dist"
    artifacts.mkdir()
    (artifacts / "OpenCut-Setup-9.9.9.exe").write_bytes(b"installer")
    (artifacts / "build.log").write_text("noise", encoding="utf-8")
    (artifacts / "receipt.json").write_text("{}", encoding="utf-8")

    payload = module.digest_artifacts(artifacts, tmp_path / "digests.json")

    assert [entry["name"] for entry in payload["artifacts"]] == ["OpenCut-Setup-9.9.9.exe"]


def test_an_empty_artifact_directory_fails_rather_than_publishing_nothing(tmp_path):
    """A digest file with no entries would read as 'verified' to a user."""
    module = _module()
    artifacts = tmp_path / "dist"
    artifacts.mkdir()
    (artifacts / "build.log").write_text("noise", encoding="utf-8")

    with pytest.raises(module.ReleaseGateError):
        module.digest_artifacts(artifacts, tmp_path / "digests.json")


def test_missing_artifact_directory_is_refused(tmp_path):
    module = _module()

    with pytest.raises(module.ReleaseGateError):
        module.digest_artifacts(tmp_path / "absent", tmp_path / "digests.json")


def test_digest_file_tells_the_user_how_to_check_it(tmp_path):
    module = _module()
    artifacts = tmp_path / "dist"
    artifacts.mkdir()
    (artifacts / "OpenCut-Setup-9.9.9.exe").write_bytes(b"installer")

    output = tmp_path / "digests.json"
    module.digest_artifacts(artifacts, output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert "Get-FileHash" in payload["verify_hint"]
    assert "sha256sum" in payload["verify_hint"]


def test_readme_states_artifacts_are_unsigned_and_how_to_verify():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")

    assert "OpenCut ships unsigned" in readme
    assert "More info" in readme and "Run anyway" in readme
    assert "Get-FileHash" in readme
    assert "sha256sum" in readme
    assert "release-digests.json" in readme
