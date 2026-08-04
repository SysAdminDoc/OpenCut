"""The `mcp` extra stays on the tested 1.x client-tooling line.

OpenCut's server implementation speaks JSON-RPC directly and does not import
the SDK, but the extra is still consumed by users as reference client
tooling. Keep the dependency bounded to one major API line until the 1.x to
2.x protocol/client break has a dedicated compatibility pass.
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


def test_mcp_extra_is_bounded_to_the_tested_major_line():
    body = _mcp_extra_block()
    assert "<2" in body, (
        "The mcp extra must stay on the tested 1.x SDK line until the 2.x "
        "protocol/client compatibility pass is complete."
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
