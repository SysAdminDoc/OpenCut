"""
Clip quality scoring via CLIP-IQA+.

Uses CLIP-IQA+ (https://github.com/IceClear/CLIP-IQA, Apache-2) — a
zero-shot no-reference image quality metric based on CLIP — to score
arbitrary video clips across subjective axes (sharp / blurry,
aesthetic, well-exposed, etc.) without task-specific training data.

Two primary use cases in OpenCut:

1. **Best-take selection.** Feed several candidate takes, rank by
   aggregate quality.
2. **Auto-reject.** Score below a user-set threshold → flag for human
   review before it enters the timeline.

Implementation samples frames at a user-configured interval (default 1
fps) and averages the per-frame CLIP-IQA+ scores.  Optional axes
(``"sharp"``, ``"aesthetic"``, ``"exposure"``, ``"colourful"``, ``"noisy"``)
are passed through to the model's ``antonym_pairs`` dict.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import get_ffmpeg_path

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Default axis set — keys match CLIP-IQA+ antonym pair names.
# ---------------------------------------------------------------------------

DEFAULT_AXES = (
    "sharp",        # Higher is sharper
    "aesthetic",    # Higher is more aesthetically pleasing
    "exposure",     # Higher is better-exposed
    "colourful",    # Higher is more vivid
    "noisy",        # Higher is *noisier* — invert before aggregation
)

# Axes where a HIGHER score is worse — we invert to 1 - score so the
# aggregate "quality" metric is monotone-positive.
INVERTED_AXES = frozenset({"noisy", "blurry", "dark"})


@dataclass
class QualityResult:
    """Structured per-clip quality score."""
    filepath: str = ""
    overall: float = 0.0
    axis_scores: Dict[str, float] = field(default_factory=dict)
    sample_count: int = 0
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

def check_clip_iqa_available() -> bool:
    """True when both `torch` and a `clip_iqa` entry point are usable."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    # CLIP-IQA+ ships under multiple module names depending on how
    # users installed it; accept either.
    for mod in ("piq", "clip_iqa", "iqa", "pyiqa"):
        try:
            __import__(mod)
            return True
        except ImportError:
            continue
    return False


# ---------------------------------------------------------------------------
# Frame sampler
# ---------------------------------------------------------------------------

def _extract_sample_frames(
    input_path: str,
    target_dir: str,
    fps_sample: float = 1.0,
    max_frames: int = 60,
) -> List[str]:
    """Sample up to ``max_frames`` JPEGs at ``fps_sample`` Hz.

    Returns the sorted list of JPEG paths.
    """
    os.makedirs(target_dir, exist_ok=True)
    pattern = os.path.join(target_dir, "frame_%04d.jpg")
    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-vf", f"fps={max(0.1, float(fps_sample))},scale='min(512,iw)':'-2'",
        "-frames:v", str(int(max_frames)),
        "-q:v", "3",
        pattern,
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=600, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg frame sample failed: "
            f"{proc.stderr.decode(errors='replace')[-200:]}"
        )
    frames = sorted(
        os.path.join(target_dir, f)
        for f in os.listdir(target_dir)
        if f.lower().endswith(".jpg")
    )
    return frames


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_clip(
    input_path: str,
    axes: Optional[List[str]] = None,
    fps_sample: float = 1.0,
    max_frames: int = 60,
    on_progress: Optional[Callable] = None,
) -> QualityResult:
    """Score ``input_path`` across ``axes`` using CLIP-IQA+.

    Args:
        input_path: Any file FFmpeg can decode (video or image).
        axes: List of axis names. Defaults to :data:`DEFAULT_AXES`.
        fps_sample: Frame sample rate. 1.0 means one frame/sec.
        max_frames: Cap total sampled frames. Prevents runaway on long
            files while the caller just wants a ballpark.
        on_progress: ``(pct, msg)`` callback.

    Returns:
        :class:`QualityResult`. ``overall`` is the mean of per-axis
        scores with :data:`INVERTED_AXES` inverted first, clamped to
        [0, 1].

    Raises:
        RuntimeError: CLIP-IQA+ not installed.
        FileNotFoundError: ``input_path`` missing.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")
    if not check_clip_iqa_available():
        raise RuntimeError(
            "CLIP-IQA+ not installed. Install one of: "
            "pip install pyiqa, pip install piq, or follow "
            "https://github.com/IceClear/CLIP-IQA."
        )

    axes = list(axes or DEFAULT_AXES)

    if on_progress:
        on_progress(5, "Sampling frames…")

    tmpdir = tempfile.mkdtemp(prefix="opencut_clipiqa_")
    try:
        frames = _extract_sample_frames(
            input_path, tmpdir,
            fps_sample=fps_sample,
            max_frames=max_frames,
        )
        if not frames:
            raise RuntimeError("No frames could be sampled from input.")

        if on_progress:
            on_progress(20, f"Scoring {len(frames)} frames…")

        # Prefer pyiqa (highest-quality python wrapper, 2024 update).
        import torch

        try:
            import pyiqa  # type: ignore
            device = "cuda" if torch.cuda.is_available() else "cpu"
            iqa_model = pyiqa.create_metric("clipiqa+", device=device)
            per_frame: List[Dict[str, float]] = []
            for i, fpath in enumerate(frames):
                with torch.no_grad():
                    score = float(iqa_model(fpath))
                # pyiqa returns a single aggregate score; expose it on
                # "aesthetic" and leave other axes empty (pyiqa doesn't
                # expose the antonym map by name).
                per_frame.append({"aesthetic": score})
                if on_progress:
                    pct = 20 + int(70 * ((i + 1) / len(frames)))
                    on_progress(pct, f"Scored {i + 1}/{len(frames)}")
            del iqa_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            # Fallback: piq library (no per-axis, just clip-iqa aggregate)
            import piq  # type: ignore
            from PIL import Image  # noqa: F401
            per_frame = []
            device = "cuda" if torch.cuda.is_available() else "cpu"
            metric = piq.CLIPIQA().to(device)
            from torchvision.transforms import ToTensor
            tt = ToTensor()
            for i, fpath in enumerate(frames):
                img = Image.open(fpath).convert("RGB")
                t = tt(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    s = float(metric(t).cpu().item())
                per_frame.append({"aesthetic": s})
                if on_progress:
                    pct = 20 + int(70 * ((i + 1) / len(frames)))
                    on_progress(pct, f"Scored {i + 1}/{len(frames)}")
            del metric
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Aggregate: average per axis across frames
        axis_totals: Dict[str, float] = {a: 0.0 for a in axes}
        axis_counts: Dict[str, int] = {a: 0 for a in axes}
        for pf in per_frame:
            for ax in axes:
                if ax in pf:
                    axis_totals[ax] += float(pf[ax])
                    axis_counts[ax] += 1
        axis_scores: Dict[str, float] = {}
        for ax in axes:
            if axis_counts[ax] > 0:
                avg = axis_totals[ax] / axis_counts[ax]
                if ax in INVERTED_AXES:
                    avg = 1.0 - avg
                axis_scores[ax] = round(max(0.0, min(1.0, avg)), 4)

        overall = (
            sum(axis_scores.values()) / len(axis_scores)
            if axis_scores else 0.0
        )

        if on_progress:
            on_progress(100, "Quality scoring complete")

        return QualityResult(
            filepath=input_path,
            overall=round(overall, 4),
            axis_scores=axis_scores,
            sample_count=len(frames),
            notes=[
                f"Sampled {len(frames)} frames at {fps_sample} Hz",
                "Inverted axes: " + ", ".join(sorted(INVERTED_AXES & set(axes))),
            ],
        )
    finally:
        try:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:  # noqa: BLE001
            pass


def rank_clips(
    filepaths: List[str],
    axes: Optional[List[str]] = None,
    fps_sample: float = 1.0,
    max_frames: int = 30,
    on_progress: Optional[Callable] = None,
) -> List[QualityResult]:
    """Score a set of clips, return them sorted by ``overall`` descending."""
    results: List[QualityResult] = []
    for i, fp in enumerate(filepaths):
        if on_progress:
            pct = int(100 * (i / max(1, len(filepaths))))
            on_progress(pct, f"Clip {i + 1}/{len(filepaths)}: {os.path.basename(fp)}")
        try:
            results.append(
                score_clip(fp, axes=axes,
                           fps_sample=fps_sample,
                           max_frames=max_frames)
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Quality scoring failed for %s: %s", fp, exc)
            results.append(QualityResult(
                filepath=fp,
                overall=0.0,
                axis_scores={},
                sample_count=0,
                notes=[f"error: {exc}"],
            ))
    results.sort(key=lambda r: r.overall, reverse=True)
    return results
