"""CEP confirmation guardrails for panel-local dialogs."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CEP_CLIENT = REPO_ROOT / "extension" / "com.opencut.panel" / "client"
MAIN_JS = CEP_CLIENT / "main.js"
SOURCE_SUFFIXES = {".html", ".js", ".mjs"}
FORBIDDEN_DIALOG_PATTERNS = {
    "window.alert": re.compile(r"\bwindow\.alert\b"),
    "window.confirm": re.compile(r"\bwindow\.confirm\b"),
    "window.prompt": re.compile(r"\bwindow\.prompt\b"),
    "alert()": re.compile(r"(?<![\w.])alert\s*\("),
    "confirm()": re.compile(r"(?<![\w.])confirm\s*\("),
    "prompt()": re.compile(r"(?<![\w.])prompt\s*\("),
}


def _cep_source_files() -> list[Path]:
    return sorted(
        path
        for path in CEP_CLIENT.rglob("*")
        if path.is_file()
        and path.suffix in SOURCE_SUFFIXES
        and "node_modules" not in path.parts
        and "dist" not in path.parts
    )


def test_cep_sources_do_not_use_native_alert_confirm_prompt_globals():
    violations: list[str] = []

    for path in _cep_source_files():
        source = path.read_text(encoding="utf-8")
        for label, pattern in FORBIDDEN_DIALOG_PATTERNS.items():
            if pattern.search(source):
                rel = path.relative_to(REPO_ROOT).as_posix()
                violations.append(f"{rel}: {label}")

    assert not violations, "CEP native dialog usage found:\n" + "\n".join(violations)


def test_cep_panel_local_dialogs_are_focus_managed_and_action_specific():
    source = MAIN_JS.read_text(encoding="utf-8")

    assert "function showPanelDialog" in source
    assert "function showPanelConfirm" in source
    assert "function showPanelPrompt" in source
    assert "function showPanelChoice" in source
    assert "activateOverlay(overlay" in source
    assert "overlay._ocCloseOverlay" in source
    assert "journal.clear_confirm_button" in source
    assert "search.clear_confirm_button" in source
    assert "gist.visibility_title" in source
