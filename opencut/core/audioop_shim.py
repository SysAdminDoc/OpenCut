"""F123 — Python 3.13 compatibility shim for the stdlib ``audioop`` module.

CPython 3.13 removed ``audioop`` from the standard library. Several
audio packages we *might* coexist with (``pydub`` in particular) still
do ``import audioop`` unconditionally. Without a shim, OpenCut would
crash at *their* import time on Python 3.13 even if we don't use
them ourselves.

This module provides one helper, ``install_audioop_shim()``, that:

1. No-ops on Python < 3.13 (stdlib ``audioop`` is still present).
2. On Python 3.13+, checks whether ``audioop`` is already importable
   (e.g. because the operator installed a binary patch). If yes, no
   action.
3. Otherwise, tries to import ``audioop_lts`` (the PSF-blessed backport
   on PyPI) and registers it under ``sys.modules['audioop']`` so any
   later ``import audioop`` resolves to the backport.
4. Returns a small dict describing what happened so callers can log it
   or render it in the dependency-status panel.

Usage::

    from opencut.core.audioop_shim import install_audioop_shim
    info = install_audioop_shim()
    if info["status"] == "needs_install":
        # Operator could pip install audioop-lts to enable pydub.
        ...

This is a *non-mandatory* shim — OpenCut itself does not import
audioop or pydub. The shim exists so users who pip install pydub on
top of opencut don't get a confusing ``ModuleNotFoundError: audioop``
on Python 3.13.

References:
- PEP 594 (stdlib removal of audioop in 3.13)
- audioop-lts on PyPI: https://pypi.org/project/audioop-lts/
"""

from __future__ import annotations

import importlib
import sys
from typing import Dict


def needs_shim() -> bool:
    """Return True when the running Python no longer ships ``audioop``."""
    return sys.version_info >= (3, 13)


def audioop_importable() -> bool:
    """Return True when ``import audioop`` works in the current interpreter."""
    try:
        importlib.import_module("audioop")
        return True
    except ImportError:
        return False


def install_audioop_shim() -> Dict[str, str]:
    """Install the ``audioop`` shim if needed.

    Returns a dict with keys:

    * ``status``:
        - ``"not_needed"`` on Python < 3.13.
        - ``"already_present"`` when ``audioop`` is already importable.
        - ``"installed"`` when we successfully aliased ``audioop_lts``
          into ``sys.modules['audioop']``.
        - ``"needs_install"`` when neither stdlib audioop nor the
          backport is available; the operator should
          ``pip install audioop-lts``.
        - ``"error"`` on unexpected failure.
    * ``backend``: the importable module name we resolved to (``""``
      when no shim is in place).
    * ``hint``: an actionable remediation string for the panel.
    """
    if not needs_shim():
        return {
            "status": "not_needed",
            "backend": "audioop",
            "hint": "",
        }

    if audioop_importable():
        return {
            "status": "already_present",
            "backend": "audioop",
            "hint": "",
        }

    # Python 3.13+ and audioop missing — try the backport.
    try:
        backport = importlib.import_module("audioop_lts")
    except ImportError:
        return {
            "status": "needs_install",
            "backend": "",
            "hint": (
                "Python 3.13 removed `audioop` from the stdlib. Install "
                "the PyPI backport with `pip install audioop-lts` if any "
                "OpenCut-adjacent package (e.g. pydub) still imports it."
            ),
        }
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "status": "error",
            "backend": "",
            "hint": f"audioop-lts import failed: {exc}",
        }

    # Register the backport under the legacy name so `import audioop`
    # transparently resolves to it for the rest of the process.
    sys.modules["audioop"] = backport
    return {
        "status": "installed",
        "backend": "audioop_lts",
        "hint": "",
    }
