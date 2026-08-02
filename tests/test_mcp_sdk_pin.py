"""The `mcp` extra is a convenience, not a runtime dependency.

F137 originally capped the extra at ``<2`` on the grounds that the 2.x
FastMCP -> McpServer rewrite "breaks our JSON-RPC server". It cannot:
``opencut/mcp_server.py`` implements JSON-RPC directly and never imports the
SDK, so the cap excluded SDK 2 users for no benefit.

The cap is gone; these tests enforce the invariant that made it unnecessary
in the first place. If OpenCut ever does start importing `mcp`, the
no-import test fails and whoever adds the import has to make a deliberate
decision about pinning.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
PACKAGE_ROOT = REPO_ROOT / "opencut"


def _mcp_extra_block() -> str:
    """Return the text block containing the `mcp` extra dependency."""
    text = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r"^mcp\s*=\s*\[(?P<body>.*?)^\]", text, re.M | re.S)
    assert match, "`mcp` extra not found in pyproject.toml"
    return match.group("body")


def test_mcp_extra_pins_minimum_to_1_26_or_higher():
    body = _mcp_extra_block()
    m = re.search(r'"mcp\s*>=\s*(\d+)\.(\d+)', body)
    assert m, f"`mcp` extra must declare a minimum version. Found: {body!r}"
    major, minor = int(m.group(1)), int(m.group(2))
    assert (major, minor) >= (1, 26), (
        f"`mcp` minimum must stay at or above the 1.26 baseline; got {major}.{minor}"
    )


def test_mcp_extra_does_not_exclude_sdk_2():
    body = _mcp_extra_block()
    assert "<2" not in body, (
        "The `<2` cap excluded SDK 2 users while OpenCut does not import the "
        "SDK at all. Re-add it only alongside a real import that breaks."
    )


def test_mcp_extra_is_single_entry():
    """Defence against an inadvertent extra constraint that loosens the pin."""
    body = _mcp_extra_block()
    deps = [line.strip() for line in body.splitlines() if line.strip().startswith('"')]
    assert len(deps) == 1, (
        f"`mcp` extra must contain exactly one constraint; got {deps}"
    )
    assert deps[0].startswith('"mcp>='), (
        f"`mcp` extra must start with the canonical >= pin; got {deps[0]!r}"
    )


def _imports_mcp_sdk(path: Path) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:  # pragma: no cover - a broken file is another test's job
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == "mcp" or alias.name.startswith("mcp.")
                   for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level == 0 and (module == "mcp" or module.startswith("mcp.")):
                return True
    return False


def test_opencut_never_imports_the_mcp_sdk():
    """This is what makes the unbounded version range safe."""
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in sorted(PACKAGE_ROOT.rglob("*.py"))
        if _imports_mcp_sdk(path)
    ]
    assert not offenders, (
        "opencut/ imports the `mcp` SDK: " + ", ".join(offenders) + ". "
        "Either drop the import or restore an explicit version bound in "
        "pyproject.toml, because the SDK's public API is no longer irrelevant."
    )
