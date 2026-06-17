"""TransNetV2 / AutoShot scene detection engine compatibility."""

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


# --- AutoShot ---


def test_autoshot_check_detects_module(monkeypatch):
    from opencut import checks

    monkeypatch.setattr(checks, "_try_import", lambda name: object() if name == "autoshot" else None)
    assert checks.check_autoshot_available() is True


def test_autoshot_check_returns_false_when_missing(monkeypatch):
    from opencut import checks

    monkeypatch.setattr(checks, "_try_import", lambda name: None)
    assert checks.check_autoshot_available() is False


def test_autoshot_loader_raises_when_missing(monkeypatch):
    from opencut.core import scene_detect

    monkeypatch.setattr(
        scene_detect.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError(name)),
    )

    try:
        scene_detect._load_autoshot_model()
    except ImportError as exc:
        assert "AutoShot" in str(exc)
    else:
        raise AssertionError("expected ImportError")


def test_autoshot_engine_registered():
    from opencut.core.engine_registry import get_registry

    reg = get_registry()
    engines = reg.get_engines("scene_detection")
    names = [e.name for e in engines]
    assert "autoshot" in names
    autoshot = next(e for e in engines if e.name == "autoshot")
    assert autoshot.priority > 70  # higher than TransNetV2


def test_detect_scenes_auto_prefers_autoshot(monkeypatch):
    from opencut.core import scene_detect

    class FakeModel:
        def predict(self, path):
            return [1.0, 3.0, 5.0]

    monkeypatch.setattr(
        scene_detect,
        "_load_autoshot_model",
        lambda: FakeModel(),
    )
    monkeypatch.setattr(
        scene_detect.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout='{"format":{"duration":"10.0"}}', stderr=""),
    )

    result = scene_detect.detect_scenes_auto("/fake/video.mp4")
    assert result.total_scenes >= 1
