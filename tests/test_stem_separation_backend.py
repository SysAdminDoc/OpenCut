"""Stem separation must default to a backend that is still maintained.

`facebookresearch/demucs` was archived on 2024-04-24, yet it was the declared
dependency, the route default, and the only option the panel offered.
`python-audio-separator` was already wired into the route and probed by the
engine registry, but appeared in no extra and in no requirements file - its
only install guidance was a runtime error string.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.requirements import Requirement

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
PANEL_HTML = REPO_ROOT / "extension" / "com.opencut.panel" / "client" / "index.html"

MAINTAINED = "audio-separator"
ARCHIVED = "demucs"


def _extras() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ]


def _names(entries) -> dict:
    result = {}
    for entry in entries:
        try:
            requirement = Requirement(entry)
        except Exception:
            continue
        result[requirement.name.lower()] = entry
    return result


def test_the_audio_extra_installs_a_maintained_separation_backend():
    extras = _extras()
    for extra in ("audio", "torch-stack"):
        names = _names(extras[extra])
        assert MAINTAINED in names, (
            f"the '{extra}' extra declares no maintained separation backend; "
            f"it has only {sorted(names)}"
        )


def test_archived_demucs_is_not_the_recommended_install_hint():
    from opencut import model_cards

    cards = {card.check_name: card for card in model_cards.CARDS}

    maintained = cards.get("check_audio_separator_available")
    assert maintained is not None, "no card for the maintained separation backend"
    assert "audio]" in maintained.install_hint or "audio-separator" in maintained.install_hint

    archived = cards.get("check_demucs_available")
    assert archived is not None
    lowered = (archived.label + archived.quality_notes + archived.install_hint).lower()
    assert "archived" in lowered, "the Demucs card must state that it is archived upstream"
    assert "2024-04-24" in archived.quality_notes + archived.install_hint


def test_dependency_dashboard_lists_the_maintained_backend():
    source = (
        REPO_ROOT / "opencut" / "routes" / "system_runtime_routes.py"
    ).read_text(encoding="utf-8")
    assert '"audio-separator": "audio_separator"' in source
    assert '"audio-separator": \'pip install "opencut-ppro[audio]"\'' in source
    # The archived backend must not point at the recommended extra.
    assert '"demucs": \'pip install "opencut-ppro[audio]"\'' not in source


def test_checks_expose_both_backends_and_a_rollup():
    from opencut import checks

    assert callable(checks.check_audio_separator_available)
    assert callable(checks.check_demucs_available)
    assert isinstance(checks.check_stem_separation_available(), bool)


class TestBackendSelection:
    """The route picks the backend from the model when none is given."""

    @staticmethod
    def _resolve(data: dict) -> str:
        """Mirror of the route's selection, driven by the route's own source."""
        allowed_demucs = {
            "htdemucs", "htdemucs_ft", "htdemucs_6s",
            "mdx", "mdx_extra", "mdx_q", "mdx_extra_q",
        }
        allowed_separator = {"mel_band_roformer", "bs_roformer", "scnet", "mdx23c", "htdemucs"}
        requested_model = str(data.get("model", "") or "").strip()
        backend = data.get("backend")
        if backend not in ("demucs", "audio-separator"):
            if requested_model and requested_model in allowed_demucs - allowed_separator:
                backend = "demucs"
            else:
                backend = "audio-separator"
        return backend

    def test_no_model_and_no_backend_selects_the_maintained_one(self):
        assert self._resolve({}) == "audio-separator"

    def test_a_demucs_only_model_still_reaches_demucs(self):
        assert self._resolve({"model": "htdemucs_ft"}) == "demucs"
        assert self._resolve({"model": "mdx_extra"}) == "demucs"

    def test_a_shared_model_name_goes_to_the_maintained_backend(self):
        assert self._resolve({"model": "htdemucs"}) == "audio-separator"

    def test_an_explicit_backend_always_wins(self):
        assert self._resolve({"backend": "demucs", "model": "htdemucs"}) == "demucs"
        assert (
            self._resolve({"backend": "audio-separator", "model": "bs_roformer"})
            == "audio-separator"
        )

    def test_the_route_implements_this_selection(self):
        source = (REPO_ROOT / "opencut" / "routes" / "audio.py").read_text(encoding="utf-8")
        assert "allowed_demucs - allowed_separator" in source
        assert 'backend = "audio-separator"' in source
        assert 'data.get("backend", "demucs")' not in source


def test_panel_offers_the_maintained_models_first():
    html = PANEL_HTML.read_text(encoding="utf-8")
    start = html.index('<select id="separateModel">')
    block = html[start : html.index("</select>", start)]

    assert 'value="mel_band_roformer" selected' in block, (
        "the panel still defaults to an archived-backend model"
    )
    for model in ("bs_roformer", "scnet"):
        assert f'value="{model}"' in block
    # The legacy options remain reachable, and are labelled as such.
    assert 'value="htdemucs_ft"' in block
    assert block.count("legacy backend") >= 3
