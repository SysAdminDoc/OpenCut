"""Semantic project-facts documentation gate.

Generated-doc checks validate counts but miss user-facing facts that silently
contradict source. This gate derives the fact from the authoritative source and
fails when a committed doc/comment disagrees, so a seeded contradiction (wrong
UXP tab count, a stale CSRF endpoint, a wrong Vite major) breaks the build.

Only committed files are checked (Roadmap_Blocked.md / CLAUDE.md are gitignored
local working notes and are intentionally out of scope).
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UXP_INDEX = REPO_ROOT / "extension" / "com.opencut.uxp" / "index.html"
UXP_MAIN = REPO_ROOT / "extension" / "com.opencut.uxp" / "main.js"
UXP_BACKEND_CLIENT = REPO_ROOT / "extension" / "com.opencut.uxp" / "backend-client.js"
PANEL_PKG = REPO_ROOT / "extension" / "com.opencut.panel" / "package.json"
README = REPO_ROOT / "README.md"
NODE_ADVISORIES = REPO_ROOT / "docs" / "NODE_ADVISORIES.md"


def _uxp_tabs() -> list[str]:
    """Authoritative UXP tab list: the role=tab buttons in index.html."""
    html = UXP_INDEX.read_text(encoding="utf-8")
    tabs = []
    for match in re.finditer(r'class="oc-tab(?:\s+active)?"[^>]*data-tab="([^"]+)"', html):
        tabs.append(match.group(1))
    # Also catch the first (active) tab whose class ordering differs.
    for match in re.finditer(r'data-tab="([^"]+)"[^>]*role="tab"', html):
        if match.group(1) not in tabs:
            tabs.append(match.group(1))
    return tabs


def test_readme_uxp_tab_count_matches_source():
    tabs = _uxp_tabs()
    assert len(tabs) >= 8, f"unexpectedly few UXP tabs parsed: {tabs}"
    assert "agent" in tabs, "UXP index.html should expose the Agent tab"

    readme = README.read_text(encoding="utf-8")
    # The UXP feature bullet: "**N tabs** -- Cut & Clean, Captions, ...".
    match = re.search(r"\*\*(\d+) tabs\*\* -- (.+)", readme)
    assert match, "README UXP tab bullet not found"
    stated = int(match.group(1))
    tab_list = match.group(2)
    assert stated == len(tabs), (
        f"README claims {stated} UXP tabs but index.html has {len(tabs)}: {tabs}"
    )
    # The Agent tab was the previously-omitted one; guard it explicitly.
    assert "Agent" in tab_list, "README UXP tab list omits the Agent tab"


def test_uxp_architecture_box_tab_count_matches_source():
    tabs = _uxp_tabs()
    readme = README.read_text(encoding="utf-8")
    assert f"{len(tabs)} tabs, modern JS" in readme, (
        f"README UXP architecture box should read '{len(tabs)} tabs, modern JS'"
    )


def test_uxp_csrf_comment_matches_implementation():
    main = UXP_MAIN.read_text(encoding="utf-8")
    client = UXP_BACKEND_CLIENT.read_text(encoding="utf-8")
    # Implementation reads the token from /health and sends X-OpenCut-Token.
    assert 'get("/health")' in client
    assert "X-OpenCut-Token" in client
    # The stale "/csrf or /api/csrf" comment must not reappear.
    assert "/csrf" not in main + client, "stale /csrf CSRF reference re-introduced in UXP runtime"


def test_node_advisories_vite_major_matches_package():
    pkg = PANEL_PKG.read_text(encoding="utf-8")
    match = re.search(r'"vite":\s*"[^0-9]*(\d+)\.', pkg)
    assert match, "vite version not found in panel package.json"
    major = match.group(1)
    advisories = NODE_ADVISORIES.read_text(encoding="utf-8")
    assert f"Vite {major}" in advisories, (
        f"docs/NODE_ADVISORIES.md should reference Vite {major} to match package.json"
    )
    # A superseded major must not be described as the current supported line.
    assert f"Vite {int(major) - 3}" not in advisories or "waiver" in advisories.lower()
