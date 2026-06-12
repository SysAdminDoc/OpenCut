"""Tests for release-facing version sync surfaces."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SYNC_VERSION_PATH = REPO_ROOT / "scripts" / "sync_version.py"


def _sync_version_module():
    spec = importlib.util.spec_from_file_location("sync_version_under_test", SYNC_VERSION_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _targets(module, rel_path: str):
    return [target for target in module.TARGETS if target[0] == rel_path]


def test_version_tokens_include_security_minor_series():
    module = _sync_version_module()

    tokens = module.version_tokens("1.33.0")

    assert tokens["series"] == "1.33.x"
    assert tokens["previous_series"] == "1.32.x"
    assert tokens["critical_series"] == "1.31.x"
    assert tokens["latest_minor"] == "1.33"
    assert tokens["eol_minor"] == "1.30"


def test_security_policy_targets_sync_minor_series(monkeypatch, tmp_path):
    module = _sync_version_module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    security = tmp_path / "SECURITY.md"
    security.write_text(
        "\n".join(
            [
                "OpenCut ships rapidly. We actively support the **latest minor** (`1.32.x`) "
                "and the one immediately preceding it (`1.31.x`).",
                "",
                "| Version | Supported         | Security fixes until |",
                "|---------|-------------------|----------------------|",
                "| 1.32.x  | ✅ Active         | —                    |",
                "| 1.31.x  | ✅ Previous       | +90 days after 1.32  |",
                "| 1.30.x  | ⚠️ Critical only  | +30 days after 1.32  |",
                "| ≤ 1.29  | ❌ End of life    | n/a                  |",
                "",
            ]
        ),
        encoding="utf-8",
    )

    security_targets = _targets(module, "SECURITY.md")
    assert security_targets
    assert not all(
        module.check_file(path, pattern, replacement, "1.33.0")
        for path, pattern, replacement in security_targets
    )

    for path, pattern, replacement in security_targets:
        module.sync_file(path, pattern, replacement, "1.33.0")

    text = security.read_text(encoding="utf-8")
    assert "`1.33.x`" in text
    assert "`1.32.x`" in text
    assert "| 1.33.x" in text
    assert "| 1.32.x" in text
    assert "| 1.31.x" in text
    assert "| ≤ 1.30" in text
    assert "+90 days after 1.33" in text
    assert "+30 days after 1.33" in text
    assert all(
        module.check_file(path, pattern, replacement, "1.33.0")
        for path, pattern, replacement in security_targets
    )


def test_panel_lock_and_c2pa_targets_sync_release_version(monkeypatch, tmp_path):
    module = _sync_version_module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    lock = tmp_path / "extension" / "com.opencut.panel" / "package-lock.json"
    c2pa = tmp_path / "opencut" / "core" / "c2pa_sidecar.py"
    lock.parent.mkdir(parents=True)
    c2pa.parent.mkdir(parents=True)
    lock.write_text(
        """{
  "name": "opencut-panel",
  "version": "1.32.0",
  "lockfileVersion": 3,
  "packages": {
    "": {
      "name": "opencut-panel",
      "version": "1.32.0"
    },
    "node_modules/lightningcss": {
      "version": "1.32.0"
    }
  }
}
""",
        encoding="utf-8",
    )
    c2pa.write_text(
        'CLAIM_GENERATOR_DEFAULT = "OpenCut/1.32.0 (sidecar; c2pa-spec 2.3)"\n',
        encoding="utf-8",
    )

    targets = [
        *_targets(module, "extension/com.opencut.panel/package-lock.json"),
        *_targets(module, "opencut/core/c2pa_sidecar.py"),
    ]
    assert targets
    assert not all(
        module.check_file(path, pattern, replacement, "1.33.0")
        for path, pattern, replacement in targets
    )

    for path, pattern, replacement in targets:
        module.sync_file(path, pattern, replacement, "1.33.0")

    lock_text = lock.read_text(encoding="utf-8")
    c2pa_text = c2pa.read_text(encoding="utf-8")
    assert '"version": "1.33.0"' in lock_text
    assert 'OpenCut/1.33.0 (sidecar; c2pa-spec 2.3)' in c2pa_text
    assert '"node_modules/lightningcss": {\n      "version": "1.32.0"' in lock_text
    assert all(
        module.check_file(path, pattern, replacement, "1.33.0")
        for path, pattern, replacement in targets
    )
