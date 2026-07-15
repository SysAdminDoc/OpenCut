"""Registry + resolver coverage for the Depth Anything 3 backend.

DA3-Small (Apache-2.0) is the preferred depth backend
for CineFocus bokeh/parallax, with Depth-Anything-V2-Small retained as the
automatic fallback. These tests pin the registry priority, resolver,
availability gate, model card, and CineFocus adapter without downloading
model weights.
"""

from __future__ import annotations

import sys
import types


def test_da3_registered_and_preferred_over_v2():
    from opencut.core.engine_registry import get_registry

    reg = get_registry()
    names = {e.name for e in reg.get_engines("depth_estimation")}
    assert "depth_anything_3" in names
    assert "depth_anything_v2" in names  # fallback retained
    da3 = reg.get_engine("depth_estimation", "depth_anything_3")
    da2 = reg.get_engine("depth_estimation", "depth_anything_v2")
    assert da3.priority > da2.priority


def test_da3_gated_by_availability_check():
    from opencut.checks import check_depth_anything_3_available
    from opencut.core.engine_registry import get_registry

    reg = get_registry()
    engine = reg.get_engine("depth_estimation", "depth_anything_3")
    assert engine is not None
    assert engine.is_available == check_depth_anything_3_available()


def test_da3_check_returns_bool():
    from opencut.checks import check_depth_anything_3_available

    assert isinstance(check_depth_anything_3_available(), bool)


def test_da3_model_metadata():
    from opencut.core import depth_anything_3 as da3

    assert da3.MODEL_LICENSE == "Apache-2.0"
    assert da3.model_id("small") == "depth-anything/DA3-SMALL"
    # Unknown sizes fall back to small.
    assert da3.model_id("bogus") == "depth-anything/DA3-SMALL"


def test_resolve_depth_model_falls_back_to_da2():
    from opencut.core import depth_anything_3 as da3

    backend, model = da3.resolve_depth_model("small")
    if da3.check_depth_anything_3_available():
        assert backend == "da3"
        assert model == "depth-anything/DA3-SMALL"
    else:
        assert backend == "da2"
        assert model == "depth-anything/Depth-Anything-V2-Small-hf"


def test_da3_has_model_card():
    from opencut.model_cards import cards_by_check_name

    card = cards_by_check_name()["check_depth_anything_3_available"]
    assert card.license == "Apache-2.0"
    assert "depth-anything-3==0.1.1" in card.install_hint


def test_cinefocus_uses_central_depth_loader():
    """CineFocus must route through the DA3-preferred loader, not hardcode DA2."""
    import inspect

    from opencut.core import cinefocus

    src = inspect.getsource(cinefocus)
    assert "_load_depth_backend" in src
    # The bokeh/parallax paths must not still hardcode the DA2 load directly.
    assert src.count('AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")') == 0


def test_cinefocus_accepts_da3_prediction_shape():
    import numpy as np

    from opencut.core.cinefocus import _estimate_depth_frame

    class Prediction:
        depth = np.array([[[2.0, 4.0], [6.0, 10.0]]], dtype=np.float32)

    class Model:
        def inference(self, images):
            assert len(images) == 1
            return Prediction()

    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    depth = _estimate_depth_frame(frame, Model(), None, "cpu")
    assert depth.shape == (2, 2)
    assert float(depth.min()) == 0.0
    assert float(depth.max()) == 1.0


def test_cinefocus_loads_official_da3_api(monkeypatch):
    import torch

    from opencut.core import cinefocus
    from opencut.core import depth_anything_3 as da3

    calls = {}

    class Model:
        def to(self, *, device):
            calls["device"] = device
            return self

        def eval(self):
            calls["evaluated"] = True

    class DepthAnything3:
        @classmethod
        def from_pretrained(cls, model_id):
            calls["model_id"] = model_id
            return Model()

    package = types.ModuleType("depth_anything_3")
    package.__path__ = []
    api = types.ModuleType("depth_anything_3.api")
    api.DepthAnything3 = DepthAnything3
    monkeypatch.setitem(sys.modules, "depth_anything_3", package)
    monkeypatch.setitem(sys.modules, "depth_anything_3.api", api)
    monkeypatch.setattr(da3, "check_depth_anything_3_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    model, processor, device, backend = cinefocus._load_depth_backend()

    assert isinstance(model, Model)
    assert processor is None
    assert device == "cpu"
    assert backend == "depth-anything-3"
    assert calls == {
        "device": "cpu",
        "evaluated": True,
        "model_id": "depth-anything/DA3-SMALL",
    }
