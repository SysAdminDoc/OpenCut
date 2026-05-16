"""Tests for the OCIO validator (F109)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from opencut.core import ocio_validate as ov


def test_validate_returns_unavailable_when_pyopencolorio_missing(monkeypatch):
    """When PyOpenColorIO isn't installed, the validator must still return cleanly."""

    # Force the import to fail by patching sys.modules.
    import builtins

    real_import = builtins.__import__

    def _hijack(name, *args, **kwargs):
        if name == "PyOpenColorIO":
            raise ImportError("simulated missing dep")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _hijack)

    result = ov.validate_ocio()

    assert result.available is False
    assert any(f.rule == "ocio_not_installed" for f in result.findings)
    assert any("PyOpenColorIO" in f.message for f in result.findings)


class _StubColorSpace:
    def __init__(self, name: str):
        self._name = name

    def getName(self) -> str:
        return self._name


class _StubConfig:
    def __init__(self):
        self._roles = {"scene_linear": "ACEScg", "default": "ACEScg"}
        self._color_spaces = ["ACES2065-1", "ACEScg", "sRGB"]
        self._displays = ["sRGB", "Rec709"]
        self._views = {"sRGB": ["sRGB", "Raw"], "Rec709": ["Rec.709"]}
        self._view_transforms = ["sRGB", "Rec.709"]
        self._search_path = "luts"

    def getWorkingDir(self) -> str:
        return "/tmp/aces"

    def getColorSpaces(self):
        return [_StubColorSpace(name) for name in self._color_spaces]

    def getColorSpace(self, name):
        if name in self._roles:
            return _StubColorSpace(self._roles[name])
        return _StubColorSpace(name)

    def getRoleNames(self):
        return list(self._roles)

    def getLooks(self):
        return []

    def getDefaultDisplay(self):
        return self._displays[0]

    def getDefaultView(self, display):
        return self._views[display][0]

    def getViewTransformNames(self):
        return list(self._view_transforms)

    def getViewTransform(self, name):
        return name if name in self._view_transforms else None

    def getSearchPath(self):
        return self._search_path


class _StubOcioModule:
    __version__ = "2.4.0"
    _config = _StubConfig()

    def GetCurrentConfig(self):
        return self._config


@pytest.fixture()
def stub_ocio(monkeypatch):
    """Inject a stub PyOpenColorIO module so we can exercise the validator."""
    import sys

    stub = _StubOcioModule()
    monkeypatch.setitem(sys.modules, "PyOpenColorIO", stub)
    yield stub
    sys.modules.pop("PyOpenColorIO", None)


def test_validate_reports_config_metadata(stub_ocio):
    result = ov.validate_ocio()

    assert result.available is True
    assert result.version == "2.4.0"
    assert "ACES2065-1" in result.color_spaces
    assert result.default_display == "sRGB"
    assert result.default_view == "sRGB"
    assert "scene_linear" in result.roles


def test_validate_flags_missing_aces_space(stub_ocio, monkeypatch):
    """Drop ACES spaces from the stub config and assert the warning fires."""
    stub_ocio._config._color_spaces = ["sRGB", "Linear"]
    stub_ocio._config._roles = {"scene_linear": "Linear", "default": "Linear"}

    result = ov.validate_ocio()

    rules = {f.rule for f in result.findings}
    assert "missing_aces_space" in rules


def test_validate_flags_missing_default_role(stub_ocio):
    stub_ocio._config._roles = {"scene_linear": "ACEScg"}  # no "default"

    result = ov.validate_ocio()

    findings = [f for f in result.findings if f.rule == "missing_role"]
    assert any("default" in f.message for f in findings)


def test_validate_flags_no_default_display(stub_ocio):
    stub_ocio._config._displays = []
    # Override the relevant methods to return blanks.
    stub_ocio._config.getDefaultDisplay = lambda: ""  # type: ignore[assignment]
    stub_ocio._config.getDefaultView = lambda display: ""  # type: ignore[assignment]

    result = ov.validate_ocio()

    rules = {f.rule for f in result.findings}
    assert "no_default_display" in rules


def test_validate_flags_broken_view_transform(stub_ocio):
    def _broken(name):
        raise RuntimeError(f"cannot resolve {name}")

    stub_ocio._config.getViewTransform = _broken  # type: ignore[assignment]

    result = ov.validate_ocio()

    findings = [f for f in result.findings if f.rule == "bad_view_transform"]
    assert findings and "cannot resolve" in findings[0].message


def test_route_returns_payload_even_without_ocio(client):
    resp = client.get("/system/ocio")
    assert resp.status_code == 200
    payload = resp.get_json()
    # Without OCIO installed the validator surfaces the install hint.
    if not payload["available"]:
        assert any(f["rule"] == "ocio_not_installed" for f in payload["findings"])
    assert "findings" in payload
