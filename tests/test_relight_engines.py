"""Registry coverage for the IC-Light v1 relight engine.

The stub was re-aimed from IC-Light v2 (non-commercial, weights never released)
to IC-Light **v1** (Apache-2.0, real public weights). These tests pin the
registry wiring, the availability gating, the model-download entry, and the
removal of the v2 framing.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_iclight_registered_in_relight_domain():
    from opencut.core.engine_registry import get_registry

    reg = get_registry()
    names = {e.name for e in reg.get_engines("relight")}
    assert "iclight" in names


def test_iclight_gated_by_availability_check():
    from opencut.checks import check_iclight_available
    from opencut.core.engine_registry import get_registry

    reg = get_registry()
    engine = reg.get_engine("relight", "iclight")
    assert engine is not None
    assert engine.is_available == check_iclight_available()


def test_iclight_check_returns_bool():
    from opencut.checks import check_iclight_available

    assert isinstance(check_iclight_available(), bool)


def test_iclight_targets_v1_apache_license():
    from opencut.core import relight_iclight

    assert relight_iclight.MODEL_LICENSE == "Apache-2.0"
    assert relight_iclight.MODEL_NAME == "ic-light-v1"
    assert "IC-Light" in relight_iclight.MODEL_ID


def test_iclight_v2_framing_removed_from_module():
    """The module docstring/hints must no longer present this as IC-Light v2."""
    src = (REPO_ROOT / "opencut" / "core" / "relight_iclight.py").read_text(encoding="utf-8")
    # No "v2"/"V2" as the *targeted* model. The only allowed mention is the
    # explicit note that v2 is intentionally NOT targeted.
    assert "IC-Light v1" in src
    assert "V2 Per-Frame Relight" not in src
    assert "IC-Light V2 LoRA" not in src


def test_iclight_has_model_catalog_entry():
    from opencut.core.model_manager import KNOWN_MODELS

    assert "ic-light-v1" in KNOWN_MODELS
    entry = KNOWN_MODELS["ic-light-v1"]
    assert entry["category"] == "relight"
    assert entry["url"].startswith("https://")


def test_iclight_relight_raises_without_dependency():
    import pytest

    from opencut.core import relight_iclight

    if relight_iclight.check_iclight_available():
        with pytest.raises(NotImplementedError):
            relight_iclight.relight(video_path="x.mp4")
    else:
        with pytest.raises(RuntimeError):
            relight_iclight.relight(video_path="x.mp4")
