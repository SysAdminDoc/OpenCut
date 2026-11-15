"""F463: the mask, matte and RGBA writers keep their pixels under Pillow 13.

Pillow 13 removed ``Image.fromarray(..., mode=...)`` where the mode changes
the array's data type (it raises now) and refuses to save I-mode images as
PNG. Each writer that passes ``mode`` is driven through its real code path
with a fixture, and the PNG it writes is compared with pixels worked out by
hand from that writer's formula. Warnings are errors here, so a deprecated
``mode`` use fails on Pillow 12 as well.
"""

from __future__ import annotations

import contextlib
import os
import sys
import types

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.filterwarnings("error")


def _png(path):
    with Image.open(path) as img:
        img.load()
        return img.mode, img.size, img.tobytes()


class _FakeTensor:
    """Just enough of a torch tensor for the writers' ``.cpu().numpy()`` calls."""

    def __init__(self, arr):
        self.arr = arr

    def cpu(self):
        return self

    def numpy(self):
        return self.arr

    def permute(self, *_dims):
        return self

    def unsqueeze(self, _dim):
        return self

    def float(self):
        return self

    def to(self, _device):
        return self

    def __truediv__(self, _other):
        return self

    def __getitem__(self, _index):
        return self


# ---------------------------------------------------------------------------
# matte_birefnet: alpha post-processing (mode="L") and cutout (mode "RGBA")
# ---------------------------------------------------------------------------
def test_birefnet_alpha_matte_keeps_its_pixels():
    from opencut.core.matte_birefnet import _postprocess_alpha

    mask = np.array([[[[0.0, 0.5, 1.0], [0.25, 0.75, 0.1]]]], dtype=np.float32)
    img = _postprocess_alpha(mask, (3, 2))

    assert img.mode == "L"
    assert img.size == (3, 2)
    # value * 255, truncated: 0, 127.5, 255, 63.75, 191.25, 25.5
    assert img.tobytes() == bytes([0, 127, 255, 63, 191, 25])


def test_birefnet_cutout_premultiplies_rgb(tmp_path):
    from opencut.core.matte_birefnet import _compose_output

    src = Image.new("RGB", (3, 1))
    src.putdata([(200, 100, 50), (10, 20, 30), (90, 90, 90)])
    alpha = Image.new("L", (3, 1))
    alpha.putdata([128, 255, 0])
    out = tmp_path / "cutout.png"

    _compose_output(src, alpha, str(out), "cutout")

    # RGB * alpha / 255, truncated: 200 * 128 / 255 = 100.39 and so on.
    assert _png(out) == (
        "RGBA",
        (3, 1),
        bytes([100, 50, 25, 128, 10, 20, 30, 255, 0, 0, 0, 0]),
    )


# ---------------------------------------------------------------------------
# physics_remove: combined removal mask (mode="L")
# ---------------------------------------------------------------------------
class _StopAfterMasks(Exception):
    pass


def test_physics_removal_mask_is_the_dilated_box(tmp_path, monkeypatch):
    from opencut.core import physics_remove

    video = tmp_path / "in.mp4"
    video.write_bytes(b"not decoded")
    work = tmp_path / "work"
    work.mkdir()

    def fake_ffmpeg(cmd, timeout=None):
        pattern = cmd[-1]
        for n in (1, 2):
            Image.new("RGB", (40, 30), (n, n, n)).save(pattern % n)

    def stop_at_inpaint(pct, _msg=""):
        if pct >= 40:
            raise _StopAfterMasks

    monkeypatch.setattr(physics_remove, "ensure_package", lambda *a, **k: True)
    monkeypatch.setattr(physics_remove, "get_video_info", lambda _p: {"fps": 30.0, "width": 40, "height": 30})
    monkeypatch.setattr(physics_remove, "get_ffmpeg_path", lambda: "ffmpeg")
    monkeypatch.setattr(physics_remove, "run_ffmpeg", fake_ffmpeg)
    monkeypatch.setattr(physics_remove.tempfile, "mkdtemp", lambda prefix="": str(work))

    with pytest.raises(_StopAfterMasks):
        physics_remove.remove_with_physics(
            str(video),
            object_bbox=(10, 8, 6, 5),
            detect_shadows=False,
            detect_reflections=False,
            on_progress=stop_at_inpaint,
        )

    # The box covers rows 8-12 and columns 10-15. Dilating by 8 pixels with a
    # square element grows it to rows 0-20 and columns 2-23.
    expected = np.zeros((30, 40), dtype=np.uint8)
    expected[0:21, 2:24] = 255
    masks = sorted(os.listdir(work / "masks"))
    assert masks == ["mask_000000.png", "mask_000001.png"]
    for name in masks:
        assert _png(work / "masks" / name) == ("L", (40, 30), expected.tobytes())


# ---------------------------------------------------------------------------
# text_segment: SAM2 refinement masks (mode="L")
# ---------------------------------------------------------------------------
def test_sam2_refinement_masks_threshold_at_half(tmp_path, monkeypatch):
    from opencut.core import text_segment

    probs = np.array([[0.2, 0.6], [0.51, 0.5]], dtype=np.float32)

    class FakePredictor:
        def init_state(self, video_path):
            return {"video_path": video_path}

        def add_new_points_or_box(self, **_kwargs):
            return None

        def propagate_in_video(self, _state):
            yield 0, [1], [_FakeTensor(probs)]
            yield 1, [1], []

    fake_torch = types.ModuleType("torch")
    fake_torch.inference_mode = contextlib.nullcontext
    fake_build = types.ModuleType("sam2.build_sam")
    fake_build.build_sam2_video_predictor = lambda _model_id: FakePredictor()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "sam2", types.ModuleType("sam2"))
    monkeypatch.setitem(sys.modules, "sam2.build_sam", fake_build)
    monkeypatch.setattr(text_segment, "ensure_package", lambda *a, **k: True)

    frames = tmp_path / "frames"
    frames.mkdir()
    Image.new("RGB", (2, 2)).save(frames / "frame_000000.png")
    masks = tmp_path / "masks"

    count = text_segment._refine_with_sam2(str(frames), (1, 1), str(masks))

    assert count == 1
    assert sorted(os.listdir(masks)) == ["mask_000000.png"]
    # Strictly above 0.5 is foreground.
    assert _png(masks / "mask_000000.png") == ("L", (2, 2), bytes([0, 255, 255, 0]))


# ---------------------------------------------------------------------------
# video_ai: Robust Video Matting alpha (mode="L") and RGBA (mode="RGBA") frames
# ---------------------------------------------------------------------------
_BGR_FRAME = np.array(
    [
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
    ],
    dtype=np.uint8,
)
_ALPHA = np.array([[0.0, 0.5, 1.0], [0.25, 0.75, 0.1]], dtype=np.float32)
# alpha * 255, truncated
_ALPHA_U8 = [0, 127, 255, 63, 191, 25]


def _run_rvm(tmp_path, monkeypatch, **kwargs):
    import cv2

    from opencut.core import video_ai

    class FakeModel:
        def to(self, _device):
            return self

        def eval(self):
            return self

        def __call__(self, src, *rec, downsample_ratio=None):
            return src, _FakeTensor(_ALPHA), None, None, None, None

    class FakeCapture:
        def __init__(self, _path):
            self.frames = [_BGR_FRAME.copy()]

        def isOpened(self):
            return True

        def get(self, prop):
            return {
                cv2.CAP_PROP_FPS: 30.0,
                cv2.CAP_PROP_FRAME_WIDTH: 3,
                cv2.CAP_PROP_FRAME_HEIGHT: 2,
                cv2.CAP_PROP_FRAME_COUNT: 1,
            }.get(prop, 0)

        def read(self):
            if self.frames:
                return True, self.frames.pop(0)
            return False, None

        def release(self):
            pass

    fake_torch = types.ModuleType("torch")
    fake_torch.hub = types.SimpleNamespace(load=lambda *a, **k: FakeModel())
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch.no_grad = contextlib.nullcontext
    fake_torch.from_numpy = _FakeTensor
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(video_ai, "get_ffmpeg_path", lambda: "ffmpeg")

    written = {}

    def fake_ffmpeg(cmd, *a, **k):
        frames_out = os.path.dirname(cmd[cmd.index("-i") + 1])
        for name in sorted(os.listdir(frames_out)):
            written[name] = _png(os.path.join(frames_out, name))

    monkeypatch.setattr(video_ai, "run_ffmpeg", fake_ffmpeg)
    video_ai._remove_background_rvm(
        str(tmp_path / "in.mp4"), output_path=str(tmp_path / "out.mov"), **kwargs
    )
    return written


def test_rvm_alpha_frames_keep_their_pixels(tmp_path, monkeypatch):
    written = _run_rvm(tmp_path, monkeypatch, alpha_only=True)
    assert written == {"frame_000000.png": ("L", (3, 2), bytes(_ALPHA_U8))}


def test_rvm_rgba_frames_keep_their_pixels(tmp_path, monkeypatch):
    written = _run_rvm(tmp_path, monkeypatch)
    # BGR reversed to RGB, then alpha appended.
    expected = bytes([
        3, 2, 1, 0, 6, 5, 4, 127, 9, 8, 7, 255,
        12, 11, 10, 63, 15, 14, 13, 191, 18, 17, 16, 25,
    ])
    assert written == {"frame_000000.png": ("RGBA", (3, 2), expected)}


# ---------------------------------------------------------------------------
# ONNX frame writers: DDColor, neural deflicker and VRT pass "RGB" positionally
# ---------------------------------------------------------------------------
# The model's C x H x W output for one row of two pixels, in 0..1.
_CHW_OUT = np.array([[[0.0, 0.25]], [[0.5, 0.75]], [[1.0, 0.1]]], dtype=np.float32)
# value * 255, truncated, per pixel: (0, 127, 255) and (63, 191, 25)
_RGB_OUT = bytes([0, 127, 255, 63, 191, 25])


def _fake_onnx(monkeypatch, env_var, output):
    import opencut.gpu

    class FakeInput:
        name = "frames"

    class FakeSession:
        def __init__(self, _path, providers=None):
            pass

        def get_inputs(self):
            return [FakeInput()]

        def run(self, _names, _feeds):
            return [output]

    fake_ort = types.ModuleType("onnxruntime")
    fake_ort.InferenceSession = FakeSession
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setattr(opencut.gpu, "selected_onnx_providers", lambda *a, **k: ["CPUExecutionProvider"])
    monkeypatch.setenv(env_var, "model.onnx")


def _one_frame(tmp_path):
    path = tmp_path / "frame_000000.png"
    Image.new("RGB", (2, 1), (9, 9, 9)).save(path)
    return [str(path)]


def _window_output():
    # 1 x T x C x H x W with only the centre of the 7-frame window set.
    out = np.zeros((1, 7, 3, 1, 2), dtype=np.float32)
    out[0, 3] = _CHW_OUT
    return out


def test_ddcolor_onnx_frames_keep_their_pixels(tmp_path, monkeypatch):
    from opencut.core import colorize_ddcolor

    _fake_onnx(monkeypatch, "OPENCUT_DDCOLOR_ONNX", _CHW_OUT[None])
    frames = _one_frame(tmp_path)

    colorize_ddcolor._colourise_onnx(frames, None)

    assert _png(frames[0]) == ("RGB", (2, 1), _RGB_OUT)


def test_neural_deflicker_frames_keep_their_pixels(tmp_path, monkeypatch):
    from opencut.core import deflicker_neural

    _fake_onnx(monkeypatch, "OPENCUT_DEFLICKER_ONNX", _window_output())
    frames = _one_frame(tmp_path)

    deflicker_neural._deflicker_neural_onnx(frames, None)

    assert _png(frames[0]) == ("RGB", (2, 1), _RGB_OUT)


def test_vrt_restored_frames_keep_their_pixels(tmp_path, monkeypatch):
    from opencut.core import restore_vrt

    _fake_onnx(monkeypatch, "OPENCUT_VRT_ONNX", _window_output())
    frames = _one_frame(tmp_path)

    restore_vrt._infer_onnx(frames, 7, None)

    assert _png(frames[0]) == ("RGB", (2, 1), _RGB_OUT)
