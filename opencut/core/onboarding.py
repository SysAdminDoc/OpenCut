"""
OpenCut First-Run Onboarding State (Wave H1.8, v1.25.0)

Persists whether the user has seen the onboarding wizard, plus their
current step.  Stored in ``~/.opencut/onboarding.json`` via the
user_data wrappers so parallel Flask threads can't race on the write.

The wizard UI lives in ``client/onboarding.js``; this module only
tracks state. Delete the JSON file to re-trigger the tour.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from opencut.user_data import read_user_file, write_user_file

logger = logging.getLogger("opencut")

STATE_FILE = "onboarding.json"
MAX_STEP = 20


def check_onboarding_available() -> bool:
    """Always True — stdlib + existing user_data wrappers."""
    return True


def get_state() -> Dict[str, Any]:
    """Return the persisted onboarding state."""
    state = read_user_file(STATE_FILE, default={}) or {}
    try:
        step = int(state.get("step") or 0)
    except (TypeError, ValueError):
        step = 0
    try:
        updated = float(state.get("updated_at") or 0.0)
    except (TypeError, ValueError):
        updated = 0.0
    return {
        "seen": bool(state.get("seen")),
        "step": max(0, min(step, MAX_STEP)),
        "updated_at": updated,
    }


def set_state(
    seen: Optional[bool] = None,
    step: Optional[int] = None,
) -> Dict[str, Any]:
    """Patch onboarding state. Only the fields passed are updated."""
    state = read_user_file(STATE_FILE, default={}) or {}
    if seen is not None:
        state["seen"] = bool(seen)
    if step is not None:
        try:
            state["step"] = max(0, min(MAX_STEP, int(step)))
        except (TypeError, ValueError):
            pass
    state["updated_at"] = time.time()
    write_user_file(STATE_FILE, state)
    return get_state()


def reset() -> Dict[str, Any]:
    """Wipe onboarding state so the tour runs again on next panel open."""
    write_user_file(STATE_FILE, {})
    return get_state()


__all__ = [
    "check_onboarding_available",
    "get_state",
    "set_state",
    "reset",
]
