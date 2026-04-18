"""
BiRefNet matting for stills and keyframes.

BiRefNet (CVPR'24, MIT — https://github.com/ZhengPeng7/BiRefNet) is a
bilateral-reference segmentation network that dominates the DIS-5K and
HRSOD leaderboards as of 2024.  It produces noticeably cleaner edges
than RVM on still images and keyframes — at the cost of temporal
stability, which is why OpenCut keeps RVM for *video* matting and
reaches for BiRefNet only when edge quality on a single frame is the
priority (title-card art, thumbnails, key-frame mattes for compositing
handoff).

Backend preferences in priority order:
1. ``onnxruntime`` with an external ONNX checkpoint — fastest, zero
   torch requirement. Path via ``OPENCUT_BIREFNET_ONNX`` env var.
2. Torch via HuggingFace ``ZhengPeng7/BiRefNet`` snapshot + transformers.
3. Graceful error with install guidance.

Output modes:
- ``alpha``: raw α-matte PNG (single channel, 0..255).
- ``rgba``:  input RGB + α-matte composited as RGBA PNG.
- ``cutout``: foreground premultiplied over transparent — good for
  drop-in compositing on a title background.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger("opencut")


OUTPUT_MODES = ("alpha", "rgba", "cutout")

# Target inference resolution — BiRefNet is trained at 1024² and upstream
# inference code agrees.  Lower resolutions still work but hurt edge
# quality, which is the whole reason callers reach for BiRefNet.
TARGET_SIZE = 1024


@dataclass
class MatteResult:
    """Structured return for a single-image matte."""
    output: str = ""
    mode: str = "rgba"
    backend: str = ""
    source_size: Tuple[int, int] = (0, 0)
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, key):
        return getattr(self, key)

    def __contains__(self, key):
        return key in self.__dataclass_fields__

    def keys(self):
        return self.__dataclass_fields__.keys()


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

def check_birefnet_available() -> bool:
    """True when any supported BiRefNet backend can run."""
    onnx_path = os.environ.get("OPENCUT_BIREFNET_ONNX", "")
    if onnx_path and os.path.isfile(onnx_path):
        try:
            import onnxruntime  # noqa: F401
            return True
        except ImportError:
            pass
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def available_backends() -> List[str]:
    """Return the names of usable backends in priority order."""
    out: List[str] = []
    onnx_path = os.environ.get("OPENCUT_BIREFNET_ONNX", "")
    if onnx_path and os.path.isfile(onnx_path):
        try:
            import onnxruntime  # noqa: F401
            out.append("onnx")
        except ImportError:
            pass
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        out.append("torch_hf")
    except ImportError:
        pass
    return out


# ---------------------------------------------------------------------------
# Preprocessing / postprocessing
# ---------------------------------------------------------------------------

def _load_image(path: str):
    from PIL import Image
    return Image.open(path).convert("RGB")


def _preprocess_for_inference(img, target: int = TARGET_SIZE):
    """Letterbox-resize RGB image to ``target×target`` with ImageNet norm.

    Returns a (1, 3, H, W) ``float32`` ndarray ready for ONNX / torch
    inference plus the original ``(w, h)`` size for postprocess resize.
    """
    import numpy as np
    from PIL import Image

    w0, h0 = img.size
    resized = img.resize((target, target), Image.BILINEAR)
    arr = np.asarray(resized).astype("float32") / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype="float32").reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype="float32").reshape(1, 1, 3)
    arr = (arr - mean) / std
    chw = arr.transpose(2, 0, 1)[None]  # NCHW
    return chw, (w0, h0)


def _postprocess_alpha(mask, src_size):
    """Resize an α-mask tensor back to the source size.

    Accepts torch tensors, numpy arrays, or Python-typed nested lists.
    Handles arbitrary leading dims (some BiRefNet checkpoints emit a
    single tensor, others a tuple / list where the main mask is the
    final head).  Applies a sigmoid when the values look unbounded.
    """
    import numpy as np
    from PIL import Image

    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    arr = np.asarray(mask)
    while arr.ndim > 2:
        arr = arr[0]
    if arr.min() < 0 or arr.max() > 1:
        arr = 1.0 / (1.0 + np.exp(-arr))
    arr = (arr * 255.0).clip(0, 255).astype("uint8")
    img = Image.fromarray(arr, mode="L")
    return img.resize(src_size, Image.BILINEAR)


# ---------------------------------------------------------------------------
# Inference backends
# ---------------------------------------------------------------------------

def _infer_onnx(onnx_path: str, preprocessed):
    import onnxruntime as ort
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)
    ort_inputs = {sess.get_inputs()[0].name: preprocessed}
    outs = sess.run(None, ort_inputs)
    # BiRefNet typically emits multiple heads; main mask is the final.
    return outs[-1]


def _infer_torch_hf(preprocessed, model_name: str):
    import torch
    from transformers import AutoModelForImageSegmentation

    model = AutoModelForImageSegmentation.from_pretrained(
        model_name, trust_remote_code=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    try:
        x = torch.from_numpy(preprocessed).to(device)
        with torch.no_grad():
            out = model(x)
            if isinstance(out, (list, tuple)):
                out = out[-1]
            return out
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Output composition
# ---------------------------------------------------------------------------

def _compose_output(
    src_img,
    alpha_img,
    output_path: str,
    mode: str,
) -> None:
    """Write the final PNG for the requested ``mode``.

    - ``alpha`` — single-channel α only.
    - ``rgba``  — source RGB with α channel, RGB preserved.
    - ``cutout`` — source RGB premultiplied by α so viewers that
      composite RGB without respecting α (thumbnail generators, older
      chat clients) still show the cut-out shape.
    """
    from PIL import Image

    if mode == "alpha":
        alpha_img.save(output_path, "PNG", optimize=True)
        return

    rgba = Image.new("RGBA", src_img.size)
    rgba.paste(src_img.convert("RGBA"), (0, 0))
    rgba.putalpha(alpha_img)

    if mode == "cutout":
        import numpy as np
        arr = np.asarray(rgba).copy()
        a = arr[..., 3:4] / 255.0
        arr[..., :3] = (arr[..., :3].astype("float32") * a).astype("uint8")
        rgba = Image.fromarray(arr, "RGBA")

    rgba.save(output_path, "PNG", optimize=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def matte_image(
    input_path: str,
    output_path: Optional[str] = None,
    mode: str = "rgba",
    backend: str = "auto",
    hf_model: str = "ZhengPeng7/BiRefNet",
    on_progress: Optional[Callable] = None,
) -> MatteResult:
    """Matte a single image (still) with BiRefNet.

    Args:
        input_path: Source image (PNG/JPG/WEBP).
        output_path: Output path. Defaults to ``<input>_matte_{mode}.png``.
        mode: ``"alpha"`` / ``"rgba"`` / ``"cutout"``.
        backend: ``"auto"`` / ``"onnx"`` / ``"torch_hf"``.
        hf_model: HuggingFace repo ID when backend is ``torch_hf``.
        on_progress: ``(pct, msg)`` callback.

    Raises:
        RuntimeError: no backend is usable.
        ValueError: invalid arguments.
        FileNotFoundError: ``input_path`` missing.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")
    if mode not in OUTPUT_MODES:
        raise ValueError(f"mode must be one of {OUTPUT_MODES}, got {mode!r}")
    if not check_birefnet_available():
        raise RuntimeError(
            "BiRefNet not installed. Install one of: "
            "(1) set OPENCUT_BIREFNET_ONNX to an ONNX checkpoint path and "
            "`pip install onnxruntime`; or "
            "(2) `pip install transformers torch`."
        )

    if backend == "auto":
        backends = available_backends()
        backend = backends[0] if backends else "torch_hf"

    if on_progress:
        on_progress(5, f"Loading image (backend={backend})…")

    src_img = _load_image(input_path)
    pre, src_size = _preprocess_for_inference(src_img)

    if on_progress:
        on_progress(25, "Running BiRefNet…")

    if backend == "onnx":
        onnx_path = os.environ.get("OPENCUT_BIREFNET_ONNX", "")
        if not onnx_path or not os.path.isfile(onnx_path):
            raise RuntimeError(
                "OPENCUT_BIREFNET_ONNX not set — point it at a .onnx checkpoint."
            )
        mask = _infer_onnx(onnx_path, pre)
    elif backend == "torch_hf":
        mask = _infer_torch_hf(pre, hf_model)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    if on_progress:
        on_progress(80, "Resizing alpha back to source size…")

    alpha_img = _postprocess_alpha(mask, src_size)

    if output_path is None:
        base, _ext = os.path.splitext(input_path)
        output_path = f"{base}_matte_{mode}.png"

    _compose_output(src_img, alpha_img, output_path, mode)

    if on_progress:
        on_progress(100, "Matte complete")

    return MatteResult(
        output=output_path,
        mode=mode,
        backend=backend,
        source_size=src_size,
        notes=[f"target_size={TARGET_SIZE}"],
    )
