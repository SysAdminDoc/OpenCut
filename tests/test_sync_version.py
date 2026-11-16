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
                "| 1.32.x  | ✅ Active         | n/a                  |",
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
        module.check_file(path, pattern, replacement, "1.33.0") for path, pattern, replacement in security_targets
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
    assert "| 1.33.x  | ✅ Active         | n/a" in text
    assert "+90 days after 1.33" in text
    assert "+30 days after 1.33" in text
    assert all(
        module.check_file(path, pattern, replacement, "1.33.0") for path, pattern, replacement in security_targets
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
        'CLAIM_GENERATOR_DEFAULT = "OpenCut/1.32.0 (sidecar; c2pa-spec 2.4)"\n',
        encoding="utf-8",
    )

    targets = [
        *_targets(module, "extension/com.opencut.panel/package-lock.json"),
        *_targets(module, "opencut/core/c2pa_sidecar.py"),
    ]
    assert targets
    assert not all(module.check_file(path, pattern, replacement, "1.33.0") for path, pattern, replacement in targets)

    for path, pattern, replacement in targets:
        module.sync_file(path, pattern, replacement, "1.33.0")

    lock_text = lock.read_text(encoding="utf-8")
    c2pa_text = c2pa.read_text(encoding="utf-8")
    assert '"version": "1.33.0"' in lock_text
    assert "OpenCut/1.33.0 (sidecar; c2pa-spec 2.4)" in c2pa_text
    assert '"node_modules/lightningcss": {\n      "version": "1.32.0"' in lock_text
    assert all(module.check_file(path, pattern, replacement, "1.33.0") for path, pattern, replacement in targets)


def test_panel_update_and_about_versions_sync_together(monkeypatch, tmp_path):
    module = _sync_version_module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    panel = tmp_path / "extension" / "com.opencut.panel" / "client" / "index.html"
    uxp = tmp_path / "extension" / "com.opencut.uxp" / "index.html"
    panel.parent.mkdir(parents=True)
    uxp.parent.mkdir(parents=True)
    panel.write_text(
        '<span class="settings-value" id="updateCurrentVersion">1.32.0</span>\n'
        '<span class="settings-label" data-i18n="settings.version">Version</span>\n'
        '<span class="settings-value">1.32.0</span>\n',
        encoding="utf-8",
    )
    uxp.write_text(
        '<strong class="oc-inline-value" id="uxpUpdateCurrentVersion">1.32.0</strong>\n',
        encoding="utf-8",
    )

    targets = [
        *_targets(module, "extension/com.opencut.panel/client/index.html"),
        *_targets(module, "extension/com.opencut.uxp/index.html"),
    ]
    assert len(targets) == 5
    for path, pattern, replacement in targets:
        module.sync_file(path, pattern, replacement, "1.33.0")

    assert panel.read_text(encoding="utf-8").count("1.33.0") == 2
    assert uxp.read_text(encoding="utf-8").count("1.33.0") == 1
    assert all(
        module.check_file(path, pattern, replacement, "1.33.0")
        for path, pattern, replacement in targets
    )


def test_sync_file_preserves_crlf_bytes(monkeypatch, tmp_path):
    module = _sync_version_module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    target = tmp_path / "version.txt"
    target.write_bytes(b'version = "1.41.0"\r\nnext = true\r\n')

    changed = module.sync_file(
        "version.txt",
        r'^(version\s*=\s*")[^"]+(")',
        r"\g<1>{v}\g<2>",
        "1.42.0",
    )

    assert changed is True
    assert target.read_bytes() == b'version = "1.42.0"\r\nnext = true\r\n'


def test_security_policy_check_passes_on_a_crlf_checkout(monkeypatch, tmp_path):
    """``--check`` must not report a mismatch a ``--set`` can never clear.

    ``$`` under ``re.MULTILINE`` matches before the ``\\n``, so a pattern ending
    in ``[^\\n]*$`` pulls the ``\\r`` of a CRLF row into the match while the
    rebuilt replacement carries none. The comparison then comes out unequal on
    an already-correct file, on every run.
    """
    module = _sync_version_module()
    monkeypatch.setattr(module, "ROOT", tmp_path)
    security = tmp_path / "SECURITY.md"
    security.write_bytes(
        "\r\n".join(
            [
                "OpenCut ships rapidly. We actively support the **latest minor** (`1.43.x`) "
                "and the one immediately preceding it (`1.42.x`).",
                "",
                "| Version | Supported         | Security fixes until |",
                "|---------|-------------------|----------------------|",
                "| 1.43.x  | ✅ Active         | n/a                  |",
                "| 1.42.x  | ✅ Previous       | +90 days after 1.43  |",
                "| 1.41.x  | ⚠️ Critical only  | +30 days after 1.43  |",
                "| ≤ 1.40  | ❌ End of life    | n/a                  |",
                "",
            ]
        ).encode("utf-8")
    )

    security_targets = _targets(module, "SECURITY.md")
    assert security_targets
    assert all(
        module.check_file(path, pattern, replacement, "1.43.0")
        for path, pattern, replacement in security_targets
    ), "an in-sync CRLF SECURITY.md must not report a mismatch"

    # And a real bump still rewrites the row without stranding the CR.
    for path, pattern, replacement in security_targets:
        module.sync_file(path, pattern, replacement, "1.44.0")

    raw = security.read_bytes()
    assert b"| 1.44.x  | \xe2\x9c\x85 Active" in raw
    assert b"\r\r" not in raw
    assert raw.count(b"\n") == raw.count(b"\r\n"), "CRLF endings must survive the rewrite"
    assert all(
        module.check_file(path, pattern, replacement, "1.44.0")
        for path, pattern, replacement in security_targets
    )


def test_smoke_manifest_version_is_tracked():
    module = _sync_version_module()

    assert _targets(module, "installer/src/OpenCut.Installer/Properties/app.smoke.manifest")


def test_version_bump_runs_before_the_gate_without_a_receipt(monkeypatch, tmp_path):
    # The gate runs once, on the bump commit, so the bump itself can't wait
    # for a receipt. A checkout with no build/ folder at all must still bump.
    module = _sync_version_module()
    init_py = tmp_path / "opencut" / "__init__.py"
    init_py.parent.mkdir()
    init_py.write_bytes(b'__version__ = "1.43.0"\n')
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "INIT_PY", init_py)
    monkeypatch.setattr(module, "TARGETS", [])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module.sys, "argv", ["sync_version.py", "--set", "1.44.0"])

    module.main()

    assert not (tmp_path / "build").exists()
    assert init_py.read_bytes() == b'__version__ = "1.44.0"\n'
