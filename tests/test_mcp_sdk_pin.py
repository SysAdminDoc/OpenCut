"""F137 — pin the `mcp` SDK to the stable 1.x line.

The MCP 2.x SDK is a pre-alpha rewrite (FastMCP -> McpServer) that
breaks our `opencut.mcp_server.MCP_TOOLS` JSON-RPC server. F137 pins
the `mcp` extra to `>=1.26,<2` so a transitive resolver upgrade
cannot pull in the breaking rewrite by accident.

These tests live next to the pin so any future automated dependency
upgrade lands with a failing test.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _mcp_extra_block() -> str:
    """Return the text block containing the `mcp` extra dependency."""
    text = PYPROJECT.read_text(encoding="utf-8")
    # Find the [project.optional-dependencies] table entry for `mcp`.
    match = re.search(r"^mcp\s*=\s*\[(?P<body>.*?)^\]", text, re.M | re.S)
    assert match, "`mcp` extra not found in pyproject.toml"
    return match.group("body")


def test_mcp_extra_pins_minimum_to_1_26_or_higher():
    body = _mcp_extra_block()
    m = re.search(r'"mcp\s*>=\s*(\d+)\.(\d+)', body)
    assert m, f"`mcp` extra must declare a minimum version. Found: {body!r}"
    major, minor = int(m.group(1)), int(m.group(2))
    assert major == 1, f"`mcp` must stay on the 1.x line; got {major}.x"
    assert minor >= 26, (
        f"`mcp` minimum must be >=1.26 (F137 baseline); got 1.{minor}"
    )


def test_mcp_extra_caps_below_2():
    body = _mcp_extra_block()
    assert "<2" in body, (
        "F137 requires an upper bound of <2 on the `mcp` extra so the "
        "pre-alpha 2.x rewrite cannot be pulled in by transitive resolution. "
        f"Found: {body!r}"
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
