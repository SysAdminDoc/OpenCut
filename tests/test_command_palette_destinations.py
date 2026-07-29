"""Every command-palette entry must resolve to a destination that exists.

Commands used to name sub-tab IDs that were never in the markup — the
Settings panel has no sub-tabs at all — so Workflow Presets, Project
Templates, Keyboard Shortcuts, Job History, and dependency recovery silently
landed on a broad default page instead of the control they advertised.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PANEL = REPO_ROOT / "extension" / "com.opencut.panel" / "client"
MAIN_JS = PANEL / "main.js"
INDEX_HTML = PANEL / "index.html"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _markup_ids() -> tuple[set[str], set[str], set[str]]:
    html = _read(INDEX_HTML)
    nav_ids = set(re.findall(r'data-nav="([^"]+)"', html))
    sub_ids = set(re.findall(r'data-sub="([^"]+)"', html))
    element_ids = set(re.findall(r'id="([^"]+)"', html))
    return nav_ids, sub_ids, element_ids


def _command_entries() -> list[tuple[str, str, str, str]]:
    main = _read(MAIN_JS)
    start = main.index("var _commandIndex = [")
    end = main.index("\n    ];", start)
    block = main[start:end]
    entries = re.findall(
        r'\{\s*name:\s*"([^"]+)".*?tab:\s*"([^"]+)"'
        r'(?:,\s*sub:\s*"([^"]*)")?(?:,\s*focusId:\s*"([^"]*)")?',
        block,
    )
    assert entries, "command index could not be parsed"
    return entries


def test_command_index_is_parsed():
    # Guard the regex itself: a silent parse failure would make every
    # destination assertion below vacuously pass.
    assert len(_command_entries()) >= 40


@pytest.mark.parametrize("entry", _command_entries(), ids=lambda e: e[0])
def test_every_command_resolves_to_a_real_destination(entry):
    name, tab, sub, focus_id = entry
    nav_ids, sub_ids, element_ids = _markup_ids()

    assert tab in nav_ids, f"{name}: no nav tab '{tab}'"
    if focus_id:
        assert focus_id in element_ids, f"{name}: no element '#{focus_id}'"
    elif sub:
        assert sub in sub_ids, f"{name}: no sub-tab '{sub}'"


def test_named_regressions_open_their_promised_control():
    """The five destinations the audit found landing on a default page."""
    entries = {name: (tab, sub, focus) for name, tab, sub, focus in _command_entries()}

    for name, expected_tab, expected_focus in (
        ("Workflow Presets", "export", "workflowPreset"),
        ("Project Templates", "settings", "templateSelect"),
        ("Keyboard Shortcuts", "settings", "shortcutReference"),
        ("Job History", "settings", "jobHistory"),
    ):
        tab, _sub, focus = entries[name]
        assert tab == expected_tab, name
        assert focus == expected_focus, name

    # Dependency recovery is routed from the error-code table, not the palette.
    main = _read(MAIN_JS)
    assert '"MISSING_DEPENDENCY": { tab: "settings", focusId: "depsStatusLine"' in main
    assert "depsStatusLine" in _markup_ids()[2]


def test_settings_panel_still_has_no_sub_tabs():
    """Pin the fact that made the old destinations unreachable.

    If Settings ever gains sub-tabs, the commands above can move back to a
    sub-tab — but until then a `sub:` there is silently ignored.
    """
    _nav_ids, sub_ids, _element_ids = _markup_ids()
    assert not any(sub.startswith("set-") for sub in sub_ids)


def test_navigation_honours_the_focus_target():
    main = _read(MAIN_JS)
    assert "function navigateToTab(tab, sub, focusId)" in main
    assert "if (focusId && focusPanelTarget(focusId)) return;" in main
    # Palette activation and error-code recovery must both pass it through.
    assert "navigateToTab(item.tab, item.sub, item.focusId);" in main
    assert "navigateToTab(action.tab, action.sub || null, action.focusId || null);" in main
    # A static card must not become a permanent tab stop.
    assert 'target.setAttribute("tabindex", "-1")' in main
