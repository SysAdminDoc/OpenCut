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
2. Torch via the ``BiRefNet`` pip package (if anyone publishes it) or
   via HuggingFace ``ZhengPeng7/BiRefNet`` snapshot + transformers.
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
    # Backend 1: ONNX via env-var-specified checkpoint
    onnx_path = os.environ.get("OPENCUT_BIREFNET_ONNX", "")
    if onnx_path and os.path.isfile(onnx_path):
        try:
            import onnxruntime  # noqa: F401
            return True
        except ImportError:
            pass
    # Backend 2: torch + transformers (HF snapshot)
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def available_backends() -> List[str]:
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

TARGET_SIZE = 1024  # BiRefNet's training resolution; upstream inference code agrees


def _load_image(path: str):
    from PIL import Image
    img = Image.open(path).convert("RGB")
    return img


def _preprocess_for_inference(img, target: int = TARGET_SIZE):
    """Letterbox-resize RGB image to ``target×target`` with value range [0, 1]."""
    import numpy as np
    from PIL import Image
    w0, h0 = img.size
    resized = img.resize((target, target), Image.BILINEAR)
    arr = np.asarray(resized).astype("float32") / 255.0
    # CHW + normalise as per BiRefNet training (ImageNet stats)
    mean = np.array([0.485, 0.456, 0.406], dtype="float32").reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype="float32").reshape(1, 1, 3)
    arr = (arr - mean) / std
    chw = arr.transpose(2, 0, 1)[None]  # NCHW
    return chw, (w0, h0)


def _postprocess_alpha(mask, src_size):
    """Resize a (H, W) or (1, 1, H, W) alpha tensor back to source size."""
    import numpy as np
    from PIL import Image

    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    arr = np.asarray(mask)
    # Squeeze leading dims → (H, W)
    while arr.ndim > 2:
        arr = arr[0]
    # Sigmoid the logits if they look unbounded
    if arr.min() < 0 or arr.max() > 1:
        arr = 1.0 / (1.0 + np.exp(-arr))
    arr = (arr * 255.0).clip(0, 255).astype("uint8")
    img = Image.fromarray(arr, mode="L")
    img = img.resize(src_size, Image.BILINEAR)
    return img


# ---------------------------------------------------------------------------
# Inference backends
# ---------------------------------------------------------------------------

def _infer_onnx(onnx_path: str, preprocessed):
    import onnxruntime as ort
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)
    ort_inputs = {sess.get_inputs()[0].name: preprocessed}
    outs = sess.run(None, ort_inputs)
    # BiRefNet emits multiple heads; the final one is the main mask.
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
            # Model output may be a tensor or a tuple/list of tensors
            if isinstance(out, (list, tuple)):
                out = out[-1]
            return out
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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
        output_path: Output path. If None, writes next to source with
            ``_matte_{mode}.png`` suffix.
        mode: ``"alpha"`` (α only), ``"rgba"`` (RGB + α), ``"cutout"``
            (premultiplied RGBA, background cleared).
        backend: ``"auto"`` / ``"onnx"`` / ``"torch_hf"``.
        hf_model: HuggingFace repo ID when backend is ``torch_hf``.
        on_progress: ``(pct, msg)`` callback.

    Returns:
        :class:`MatteResult`.

    Raises:
        RuntimeError: no backend is usable.
        ValueError: invalid arguments.
        FileNotFoundError: ``input_path`` missing.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")
    if mode not in OUTPUT_MODES:
        raise ValueError(
            f"mode must be one of {OUTPUT_MODES}, got {mode!r}"
        )
    if not check_birefnet_available():
        raise RuntimeError(
            "BiRefNet not installed. Install one of: "
            "(1) set OPENCUT_BIREFNET_ONNX to an ONNX checkpoint path "
            "and `pip install onnxruntime`; or "
            "(2) `pip install transformers torch`."
        )

    if backend == "auto":
        backends = available_backends()
        backend = backends[0] if backends else "torch_hf"

    if on_progress:
        on_progress(5, f"Loading image (backend={backend})…")

    img = _load_image(input_path)
    pre, src_size = _preprocess_for_inference(img)

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
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_matte_{mode}.png"

    if mode == "alpha":
        alpha_img.save(output_path, "PNG", optimize=True)
    else:
        # Compose RGBA
        from PIL import Image
        rgba = Image.new("RGBA", img.size)
        rgba.paste(img.convert("RGBA"), (0, 0))
        rgba.putalpha(alpha_img)
        if mode == "cutout":
            # Zero out RGB where alpha is 0 (strictly speaking optional
            # since PNG viewers ignore RGB when A=0, but keeps file smaller)
            import numpy as np
            arr = __import__("numpy").asarray(rgba).copy()
            a = arr[..., 3:4] / 255.0
            arr[..., :3] = (arr[..., :3].astype("float32") * a).astype("uint8")
            rgba = __import__("PIL.Image").Image.fromarray(arr, "RGBA")
            _ = np  # silence unused-import noise on some linters
        rgba.save(output_path, "PNG", optimize=True)

    if on_progress:
        on_progress(100, "Matte complete")

    return MatteResult(
        output=output_path,
        mode=mode,
        backend=backend,
        source_size=src_size,
        notes=[f"target_size={TARGET_SIZE}"],
    )
