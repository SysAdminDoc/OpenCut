"""Semantic project-facts documentation gate.

Generated-doc checks validate counts but miss user-facing facts that silently
contradict source. This gate derives the fact from the authoritative source and
fails when a committed doc/comment disagrees, so a seeded contradiction (wrong
UXP tab count, a stale CSRF endpoint, a wrong Vite major) breaks the build.

Only committed files are checked (Roadmap_Blocked.md / CLAUDE.md are gitignored
local working notes and are intentionally out of scope).
"""

import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlparse

from opencut.project_facts import build_project_facts, validate_project_facts

REPO_ROOT = Path(__file__).resolve().parents[1]
UXP_INDEX = REPO_ROOT / "extension" / "com.opencut.uxp" / "index.html"
UXP_MAIN = REPO_ROOT / "extension" / "com.opencut.uxp" / "main.js"
UXP_BACKEND_CLIENT = REPO_ROOT / "extension" / "com.opencut.uxp" / "backend-client.js"
PANEL_PKG = REPO_ROOT / "extension" / "com.opencut.panel" / "package.json"
README = REPO_ROOT / "README.md"
NODE_ADVISORIES = REPO_ROOT / "docs" / "NODE_ADVISORIES.md"
PROJECT_FACTS = REPO_ROOT / "opencut" / "_generated" / "project_facts.json"

# CHANGELOG records old user-facing commands and claims verbatim. It is the
# one intentional exemption from current semantic-claim checks; link integrity
# still applies because a historical link should remain navigable.
SEMANTIC_FACT_EXEMPTIONS = {
    "CHANGELOG.md": "immutable historical release notes",
}
CURRENT_SEMANTIC_DOCS = (
    "README.md",
    "CONTRIBUTING.md",
    "DEVELOPMENT.md",
    "docs/INSTALLER_POLICY.md",
    "docs/LINUX_DISTRIBUTION.md",
    "docs/RELEASE_PROVENANCE.md",
)
LINK_INTEGRITY_EXEMPTIONS = {
    "ROADMAP.md": "links to the permitted gitignored Roadmap_Blocked.md tracker",
}

INLINE_LINK_RE = re.compile(
    r"!?\[[^\]]*\]\((?P<target><[^>]+>|[^)\s]+)(?:\s+['\"][^)]*['\"])?\)"
)
REFERENCE_LINK_RE = re.compile(r"^\s*\[[^\]]+\]:\s*(?P<target><[^>]+>|\S+)", re.MULTILINE)


def _committed_markdown() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "*.md"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return [REPO_ROOT / line for line in result.stdout.splitlines() if line]


def _local_link_target(markdown: Path, raw_target: str) -> Path | None:
    target = raw_target.strip().strip("<>")
    parsed = urlparse(target)
    if not target or target.startswith("#") or parsed.scheme or target.startswith("//"):
        return None
    path_text = unquote(target.split("#", 1)[0].split("?", 1)[0])
    if not path_text:
        return None
    if path_text.startswith("/"):
        return (REPO_ROOT / path_text.lstrip("/")).resolve()
    return (markdown.parent / path_text).resolve()


def _manifest() -> dict:
    return json.loads(PROJECT_FACTS.read_text(encoding="utf-8"))


def test_generated_project_facts_match_authoritative_sources():
    committed = _manifest()
    generated = build_project_facts()
    validate_project_facts(generated)

    assert committed == generated, (
        "project_facts.json is stale; run "
        "`python -m opencut.tools.dump_project_facts`"
    )


def test_readme_runtime_route_ffmpeg_and_distribution_claims_match_manifest():
    facts = _manifest()
    readme = README.read_text(encoding="utf-8")
    version = facts["project"]["version"]
    python_versions = facts["runtime"]["python_versions"]
    routes = facts["routes"]
    ffmpeg = facts["ffmpeg"]
    channels = facts["distribution"]["channels"]

    assert f"version-{version}-blue" in readme
    assert f"Python-{python_versions[0]}--{python_versions[-1]}" in readme
    platform_badge = "%20%7C%20".join(facts["runtime"]["platform_display"])
    assert platform_badge in readme
    assert f"**{routes['shipped']:,} shipped API routes**" in readme
    assert f"{routes['stubs']} strategic 501 stubs" in readme
    assert f"FFmpeg {ffmpeg['release_floor']} or newer" in readme

    source_command = channels["source_checkout"]["install_command"]
    assert channels["source_checkout"]["available"] is True
    assert source_command in readme
    assert channels["pypi"]["available"] is False
    assert channels["homebrew"]["available"] is False
    assert channels["winget"]["available"] is False
    assert "pip install opencut-ppro" not in readme
    assert "pip install \"opencut-ppro" not in readme
    assert "pip install 'opencut-ppro" not in readme

    installer = channels["github_windows_installer"]
    assert installer["available"] is True
    assert f"OpenCut v{installer['published_version']}" in readme
    assert installer["url"] in readme
    if not installer["current"]:
        assert "It predates the current source tree" in readme


def test_current_docs_distinguish_optional_egress_and_guarded_execution():
    facts = _manifest()
    assert "CHANGELOG.md" in SEMANTIC_FACT_EXEMPTIONS
    current_docs = [REPO_ROOT / relative for relative in CURRENT_SEMANTIC_DOCS]
    current_text = "\n".join(path.read_text(encoding="utf-8") for path in current_docs)

    assert "Everything runs locally. No data leaves your machine." not in current_text
    assert "No `eval`/`exec`/`pickle`" not in current_text
    assert "pip install opencut-ppro" not in current_text
    assert facts["trust"]["local_only_environment_variable"] in README.read_text(
        encoding="utf-8"
    )
    for relative_path in facts["trust"]["guarded_dynamic_execution"]:
        source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert "eval(" in source or "exec(" in source
    assert (REPO_ROOT / facts["trust"]["restricted_checkpoint_loading"]).is_file()


def test_every_local_markdown_link_resolves_to_a_committed_file():
    markdown_files = [
        path for path in _committed_markdown()
        if path.relative_to(REPO_ROOT).as_posix() not in LINK_INTEGRITY_EXEMPTIONS
    ]
    tracked = {path.resolve() for path in markdown_files}
    tracked.update(
        (REPO_ROOT / line).resolve()
        for line in subprocess.run(
            ["git", "ls-files"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.splitlines()
        if line
    )
    broken: list[str] = []
    for markdown in markdown_files:
        text = markdown.read_text(encoding="utf-8")
        matches = [*INLINE_LINK_RE.finditer(text), *REFERENCE_LINK_RE.finditer(text)]
        for match in matches:
            target = _local_link_target(markdown, match.group("target"))
            if target is not None and target not in tracked:
                line = text.count("\n", 0, match.start()) + 1
                try:
                    display = target.relative_to(REPO_ROOT).as_posix()
                except ValueError:
                    display = str(target)
                broken.append(
                    f"{markdown.relative_to(REPO_ROOT).as_posix()}:{line} -> {display}"
                )

    assert not broken, "local Markdown links must target Git-tracked files:\n" + "\n".join(broken)


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
