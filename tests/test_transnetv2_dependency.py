"""TransNetV2 PyPI dependency compatibility."""

from __future__ import annotations

from types import SimpleNamespace


def test_transnetv2_check_accepts_pytorch_package(monkeypatch):
    from opencut import checks

    seen = []

    def _fake_try_import(name):
        seen.append(name)
        return object() if name == "transnetv2_pytorch" else None

    monkeypatch.setattr(checks, "_try_import", _fake_try_import)

    assert checks.check_transnetv2_available() is True
    assert seen == ["transnetv2", "transnetv2_pytorch"]


def test_scene_detection_loader_falls_back_to_pytorch_package(monkeypatch):
    from opencut.core import scene_detect

    sentinel = object()

    def _fake_import_module(name):
        if name == "transnetv2":
            raise ImportError("legacy package absent")
        if name == "transnetv2_pytorch":
            return SimpleNamespace(TransNetV2=sentinel)
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(scene_detect.importlib, "import_module", _fake_import_module)

    assert scene_detect._load_transnetv2_class() is sentinel


def test_scene_detection_loader_error_names_installable_package(monkeypatch):
    from opencut.core import scene_detect

    monkeypatch.setattr(
        scene_detect.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError(name)),
    )

    try:
        scene_detect._load_transnetv2_class()
    except ImportError as exc:
        assert "transnetv2-pytorch" in str(exc)
    else:
        raise AssertionError("expected ImportError")
