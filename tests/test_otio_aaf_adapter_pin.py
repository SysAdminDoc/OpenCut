"""F126 — pin `otio-aaf-adapter` so the AAF export path stays working.

OpenTimelineIO split adapters out of core into separate packages
starting at 0.15. Without an explicit `otio-aaf-adapter` pin in the
`opencut[otio]` extra, the existing `opencut.export.otio_export.
export_aaf()` raises `ImportError("OTIO AAF adapter not installed")`
the moment a user runs the canonical `pip install "opencut[otio]"`.

These tests pin three invariants without needing the adapter actually
installed on this VM:

1. Both the `otio` extra and the `all` extra declare a pinned
   `otio-aaf-adapter>=0.6,<1` constraint.
2. The version range is sane (matches the F126 baseline).
3. The shipped `check_aaf_available()` honour the package name when
   the adapter is present, and degrades gracefully when it isn't.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
OTIO_EXPORT = REPO_ROOT / "opencut" / "export" / "otio_export.py"


def _extras_block(name: str) -> str:
    text = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(rf"^{re.escape(name)}\s*=\s*\[(?P<body>.*?)^\]", text, re.M | re.S)
    assert match, f"`{name}` extra not found in pyproject.toml"
    return match.group("body")


def _has_aaf_adapter(body: str) -> bool:
    return any(
        line.strip().startswith('"otio-aaf-adapter')
        for line in body.splitlines()
    )


def _adapter_spec(body: str) -> str | None:
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith('"otio-aaf-adapter'):
            return stripped
    return None


def test_otio_extra_pins_aaf_adapter():
    body = _extras_block("otio")
    assert _has_aaf_adapter(body), (
        "`opencut[otio]` must declare `otio-aaf-adapter` (F126); the AAF "
        "export path needs it after the OTIO 0.15+ adapter split."
    )


def test_all_extra_pins_aaf_adapter():
    body = _extras_block("all")
    assert _has_aaf_adapter(body), (
        "`opencut[all]` must declare `otio-aaf-adapter` (F126) so installs "
        "via the kitchen-sink extra get AAF support out of the box."
    )


def test_aaf_adapter_spec_caps_below_1_and_floors_above_0_6():
    body = _extras_block("otio")
    spec = _adapter_spec(body) or ""
    # Spec includes commas, quotes, etc. Strip to the version range.
    m = re.search(r'>=\s*(\d+)\.(\d+).*<\s*(\d+)', spec)
    assert m, f"could not parse the otio-aaf-adapter spec: {spec!r}"
    major_min, minor_min, major_cap = int(m.group(1)), int(m.group(2)), int(m.group(3))
    assert (major_min, minor_min) >= (0, 6), (
        f"F126 baseline minimum is otio-aaf-adapter>=0.6; got {major_min}.{minor_min}"
    )
    assert major_cap <= 1, (
        f"otio-aaf-adapter is pinned <1 to keep within the 0.x line; got <{major_cap}"
    )


def test_otio_export_handles_missing_adapter():
    """`check_aaf_available` must fall through cleanly when the adapter is absent.

    Without this guard, a user who installs `opencut` (no extras) would
    still get a ModuleNotFoundError on import, defeating the F126 intent
    of pinning the adapter into an opt-in extra.
    """
    text = OTIO_EXPORT.read_text(encoding="utf-8")
    assert "def check_aaf_available" in text
    # Must check otio_aaf_adapter first, then fall back to OTIO's adapter
    # registry. The fallback is important because some installs register
    # the AAF adapter via OTIO's plugin metadata without exposing the
    # `otio_aaf_adapter` shim module.
    aaf_block = text.split("def check_aaf_available")[1].split("def ")[0]
    assert "import otio_aaf_adapter" in aaf_block, (
        "check_aaf_available must probe `import otio_aaf_adapter` directly"
    )
    assert "available_adapter_names" in aaf_block, (
        "check_aaf_available must also probe OTIO's adapter registry as fallback"
    )


def test_otio_export_documents_install_hint():
    text = OTIO_EXPORT.read_text(encoding="utf-8")
    assert "pip install otio-aaf-adapter" in text, (
        "otio_export.export_aaf must surface the F126 install hint in its "
        "ImportError message so users know which package to grab"
    )
