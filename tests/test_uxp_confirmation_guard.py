"""RA-20 - UXP confirmation guardrails."""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UXP_ROOT = REPO_ROOT / "extension" / "com.opencut.uxp"
LIVE_MANIFEST = UXP_ROOT / "manifest.json"
WEBVIEW_CONFIG = UXP_ROOT / "bolt-webview" / "uxp.config.ts"
UXP_MAIN = UXP_ROOT / "main.js"
SOURCE_SUFFIXES = {".html", ".js", ".mjs", ".ts", ".tsx", ".jsx"}
FORBIDDEN_DIALOG_PATTERNS = {
    "window.alert": re.compile(r"\bwindow\.alert\b"),
    "window.confirm": re.compile(r"\bwindow\.confirm\b"),
    "window.prompt": re.compile(r"\bwindow\.prompt\b"),
    "alert()": re.compile(r"(?<![\w.])alert\s*\("),
    "confirm()": re.compile(r"(?<![\w.])confirm\s*\("),
    "prompt()": re.compile(r"(?<![\w.])prompt\s*\("),
}


def _uxp_source_files() -> list[Path]:
    return sorted(
        path
        for path in UXP_ROOT.rglob("*")
        if path.is_file()
        and path.suffix in SOURCE_SUFFIXES
        and "node_modules" not in path.parts
        and "dist" not in path.parts
    )


def test_uxp_sources_do_not_use_beta_alert_confirm_prompt_globals():
    violations: list[str] = []

    for path in _uxp_source_files():
        source = path.read_text(encoding="utf-8")
        for label, pattern in FORBIDDEN_DIALOG_PATTERNS.items():
            if pattern.search(source):
                rel = path.relative_to(REPO_ROOT).as_posix()
                violations.append(f"{rel}: {label}")

    assert not violations, "Beta UXP dialog API usage found:\n" + "\n".join(violations)


def test_manifest_does_not_enable_beta_alert_feature_flag():
    manifest = json.loads(LIVE_MANIFEST.read_text(encoding="utf-8"))
    config = WEBVIEW_CONFIG.read_text(encoding="utf-8")

    assert manifest.get("featureFlags", {}).get("enableAlerts") is not True
    assert "enableAlerts" not in config


def test_clear_index_uses_panel_local_second_click_confirmation():
    source = UXP_MAIN.read_text(encoding="utf-8")

    assert "const INLINE_CONFIRM_MS = 8000" in source
    assert "function requireClearIndexConfirmation" in source
    assert "Confirm Clear" in source
    assert "setTimeout(() => resetClearIndexConfirmation(), INLINE_CONFIRM_MS)" in source
    assert "if (!requireClearIndexConfirmation())" in source
    assert "window.confirm" not in source
