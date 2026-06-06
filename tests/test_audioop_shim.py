"""F123 — Python 3.13 audioop shim + pydub-pin retirement.

CPython 3.13 removed ``audioop`` from the standard library. OpenCut
itself does not import audioop or pydub, but downstream plugins might
install pydub alongside OpenCut and crash on the missing stdlib
module. `opencut.core.audioop_shim` provides a one-call bridge that
aliases the ``audioop_lts`` PyPI backport into ``sys.modules['audioop']``
when needed.

These tests pin:

1. The shim contract: ``not_needed`` on <3.13, ``already_present`` on
   3.13 with audioop available, ``installed`` after a successful
   ``audioop_lts`` shim, ``needs_install`` when neither is reachable.
2. The pyproject.toml `[standard]`, `[audio]`, and `[all]` extras
   no longer pin pydub. The OpenCut tree has zero `import pydub` calls
   and the dependency was vestigial.
3. The dependency-check entry in `routes/system.py` is intentionally
   kept (informational only).
"""

from __future__ import annotations

import importlib
import re
import sys
import types
from pathlib import Path

import pytest

from opencut.core import audioop_shim

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"


# ---------------------------------------------------------------------------
# Shim contract
# ---------------------------------------------------------------------------


def test_needs_shim_matches_runtime_version():
    runtime_needs = sys.version_info >= (3, 13)
    assert audioop_shim.needs_shim() is runtime_needs


def test_install_returns_not_needed_on_pre_3_13(monkeypatch):
    monkeypatch.setattr(audioop_shim, "needs_shim", lambda: False)
    info = audioop_shim.install_audioop_shim()
    assert info["status"] == "not_needed"
    assert info["backend"] == "audioop"


def test_install_detects_already_present_audioop(monkeypatch):
    monkeypatch.setattr(audioop_shim, "needs_shim", lambda: True)
    monkeypatch.setattr(audioop_shim, "audioop_importable", lambda: True)
    info = audioop_shim.install_audioop_shim()
    assert info["status"] == "already_present"
    assert info["backend"] == "audioop"


def test_install_aliases_audioop_lts_when_available(monkeypatch):
    monkeypatch.setattr(audioop_shim, "needs_shim", lambda: True)
    monkeypatch.setattr(audioop_shim, "audioop_importable", lambda: False)

    fake_module = types.ModuleType("audioop_lts")
    fake_module.MARKER = "test-marker"

    real_import = importlib.import_module

    def fake_import(name):
        if name == "audioop_lts":
            return fake_module
        return real_import(name)

    monkeypatch.setattr(audioop_shim.importlib, "import_module", fake_import)
    monkeypatch.delitem(sys.modules, "audioop", raising=False)
    try:
        info = audioop_shim.install_audioop_shim()
        assert info["status"] == "installed"
        assert info["backend"] == "audioop_lts"
        # The shim must register the backport under the legacy name so
        # downstream `import audioop` resolves to it.
        assert sys.modules["audioop"] is fake_module
    finally:
        sys.modules.pop("audioop", None)


def test_install_surfaces_actionable_hint_when_backport_missing(monkeypatch):
    monkeypatch.setattr(audioop_shim, "needs_shim", lambda: True)
    monkeypatch.setattr(audioop_shim, "audioop_importable", lambda: False)

    def fake_import(name):
        if name == "audioop_lts":
            raise ImportError("No module named 'audioop_lts'")
        return importlib.import_module(name)

    monkeypatch.setattr(audioop_shim.importlib, "import_module", fake_import)
    info = audioop_shim.install_audioop_shim()
    assert info["status"] == "needs_install"
    assert "audioop-lts" in info["hint"]
    assert "pip install" in info["hint"]


# ---------------------------------------------------------------------------
# Pyproject.toml: no pydub pin
# ---------------------------------------------------------------------------


def _extra_body(extra_name: str) -> str:
    text = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(rf"^{re.escape(extra_name)}\s*=\s*\[(?P<body>.*?)^\]", text, re.M | re.S)
    assert match, f"`{extra_name}` extra not found in pyproject.toml"
    return match.group("body")


@pytest.mark.parametrize("extra", ["standard", "audio", "all"])
def test_extra_does_not_pin_pydub(extra):
    body = _extra_body(extra)
    deps = [
        line.strip().strip('"').split(">")[0].split("=")[0]
        for line in body.splitlines()
        if line.strip().startswith('"')
    ]
    assert "pydub" not in deps, (
        f"`opencut[{extra}]` must not pin pydub (F123 — pydub is unused "
        f"and complicates Python 3.13 because of the stdlib audioop removal). "
        f"Got deps: {deps}"
    )


# ---------------------------------------------------------------------------
# Sanity: zero pydub imports in the live tree
# ---------------------------------------------------------------------------


def test_no_pydub_imports_in_opencut_tree():
    """OpenCut must not import pydub anywhere — F123 retired the dependency."""
    opencut_dir = REPO_ROOT / "opencut"
    offenders = []
    for py in opencut_dir.rglob("*.py"):
        try:
            text = py.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if re.search(r"^\s*(?:from|import)\s+pydub\b", text, re.M):
            offenders.append(str(py.relative_to(REPO_ROOT)))
    assert not offenders, "F123 forbids new pydub imports. Found in:\n  - " + "\n  - ".join(offenders)
