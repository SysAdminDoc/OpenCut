"""
A/B variant generator for the shorts pipeline (RESEARCH_FEATURE_PLAN_2026-05-25 Q8).

Opus Clip and competitors ship "generate N variants of the same short
with different hooks, captions, and focal points" so creators can A/B
test before posting. OpenCut's `shorts_pipeline.generate_shorts` is
one-shot — there is no variant surface.

This module provides the variant generator. Two entry points:

  * ``plan_variants(clip_path, start, end, n_variants=3, ...)`` — return
    the list of variant descriptors without doing any work. Used by
    ``POST /shorts/variants/dry-run``.
  * ``generate_variants(clip_path, start, end, n_variants=3, ...)`` —
    actually render each variant. Used by ``POST /shorts/variants``.

Variants differ along three axes:

  1. **hook tightness** — variant ``i`` trims the first ``i * 0.5`` s
     so a faster hook is tested against a slower one. The original
     ``[start, end]`` window is variant 0.
  2. **caption style** — cycles through the in-house style catalogue
     (``default`` / ``bold_yellow`` / ``boxed_dark`` / ``neon_cyan`` /
     ``cinematic_serif`` / ``top_center``).
  3. **focal point** — alternates ``face_track=True`` (subject-locked
     center-crop) with ``face_track=False`` (fixed top-third crop) so
     creators can compare AI-tracked vs static framing.

The module reuses the existing reframe + caption-burn-in machinery
that ``shorts_pipeline`` already calls into.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from opencut.helpers import get_video_info

logger = logging.getLogger("opencut")


CAPTION_STYLES: tuple[str, ...] = (
    "default",
    "bold_yellow",
    "boxed_dark",
    "neon_cyan",
    "cinematic_serif",
    "top_center",
)

DEFAULT_VARIANTS = 3
MIN_VARIANTS = 2
MAX_VARIANTS = 6


@dataclass
class ShortsVariant:
    variant_id: int
    output: str
    start: float
    end: float
    hook_offset: float       # seconds trimmed off the start vs the source window
    caption_style: str
    face_track: bool
    width: int
    height: int
    duration_ms: int = 0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "variant_id": self.variant_id,
            "output": self.output,
            "start": self.start,
            "end": self.end,
            "hook_offset": self.hook_offset,
            "caption_style": self.caption_style,
            "face_track": self.face_track,
            "width": self.width,
            "height": self.height,
            "duration_ms": self.duration_ms,
            "notes": list(self.notes),
        }


@dataclass
class VariantsResult:
    input_path: str = ""
    start: float = 0.0
    end: float = 0.0
    variants: List[ShortsVariant] = field(default_factory=list)
    dry_run: bool = False
    notes: List[str] = field(default_factory=list)

    # Flask jsonify protocol.
    def __getitem__(self, key: str) -> Any:
        if key == "variants":
            return [v.to_dict() for v in self.variants]
        return getattr(self, key)

    def keys(self):
        return ("input_path", "start", "end", "variants", "dry_run", "notes")

    def __contains__(self, key: str) -> bool:
        return key in self.keys()


def check_shorts_variants_available() -> bool:
    """Always True — the generator depends only on bundled FFmpeg and the
    existing reframe + caption modules. Burn-in caption styles need
    Pillow; absent, the variant ships without burn-in."""
    return True


INSTALL_HINT = (
    "Variant generator orchestrates existing modules; install Pillow + "
    "opencv-python for the full caption-burn-in surface."
)


def _validate_window(input_path: str, start: float, end: float) -> tuple[float, float, float]:
    """Clamp ``[start, end]`` to the source duration; raise on inverted window."""
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")
    if not (start >= 0 and end > start):
        raise ValueError(
            f"Invalid window: start={start} end={end} (start must be >= 0 and end > start)"
        )
    info = get_video_info(input_path) or {}
    src_dur = float(info.get("duration") or 0.0)
    if src_dur > 0:
        start = max(0.0, min(start, src_dur - 0.1))
        end = max(start + 0.1, min(end, src_dur))
    return start, end, src_dur


def _variant_descriptor(
    base_start: float,
    base_end: float,
    n_variants: int,
    width: int,
    height: int,
) -> List[dict]:
    """Build the per-variant config descriptor list."""
    n = max(MIN_VARIANTS, min(MAX_VARIANTS, int(n_variants)))
    descriptors: List[dict] = []
    duration = base_end - base_start
    for i in range(n):
        hook_offset = round(min(i * 0.5, max(0.0, duration - 0.5)), 2)
        descriptors.append({
            "variant_id": i,
            "start": round(base_start + hook_offset, 2),
            "end": round(base_end, 2),
            "hook_offset": hook_offset,
            "caption_style": CAPTION_STYLES[i % len(CAPTION_STYLES)],
            "face_track": (i % 2) == 0,
            "width": width,
            "height": height,
        })
    return descriptors


def plan_variants(
    input_path: str,
    start: float,
    end: float,
    n_variants: int = DEFAULT_VARIANTS,
    width: int = 1080,
    height: int = 1920,
) -> VariantsResult:
    """Return the planned variant set without doing any work.

    Mirrors :func:`opencut.core.enhance_auto.enhance` ``dry_run=True``.
    """
    s, e, src_dur = _validate_window(input_path, start, end)
    descriptors = _variant_descriptor(s, e, n_variants, width, height)
    result = VariantsResult(
        input_path=input_path,
        start=s,
        end=e,
        dry_run=True,
    )
    for desc in descriptors:
        result.variants.append(ShortsVariant(
            variant_id=desc["variant_id"],
            output="",  # not generated yet
            start=desc["start"],
            end=desc["end"],
            hook_offset=desc["hook_offset"],
            caption_style=desc["caption_style"],
            face_track=desc["face_track"],
            width=desc["width"],
            height=desc["height"],
        ))
    if src_dur > 0:
        result.notes.append(f"source duration {src_dur:.1f}s")
    return result


def _render_one_variant(
    input_path: str,
    desc: dict,
    output_dir: str,
    transcript_segments: Optional[List[dict]],
    burn_captions: bool,
    on_progress: Optional[Callable[[int, str], None]],
    pct_floor: int,
    pct_ceil: int,
) -> ShortsVariant:
    """Trim → reframe → optionally burn captions for a single variant."""
    import time
    start_ts = time.perf_counter()
    base = os.path.splitext(os.path.basename(input_path))[0]
    final_path = os.path.join(
        output_dir,
        f"{base}_variant{desc['variant_id']}_{desc['caption_style']}.mp4",
    )

    # 1) Trim — use shorts_pipeline._trim_clip if available, else inline FFmpeg.
    trim_dir = tempfile.mkdtemp(prefix="opencut_variant_")
    trim_out = os.path.join(trim_dir, f"trim_{desc['variant_id']}.mp4")
    try:
        from opencut.core.shorts_pipeline import _trim_clip
        _trim_clip(input_path, float(desc["start"]), float(desc["end"]), trim_out)
    except Exception as exc:
        shutil.rmtree(trim_dir, ignore_errors=True)
        raise RuntimeError(f"trim failed: {exc}") from exc
    if on_progress:
        on_progress(pct_floor + (pct_ceil - pct_floor) // 4, f"variant {desc['variant_id']} trimmed")

    # 2) Reframe — call face_reframe.reframe_face when face_track=True,
    #    otherwise a fixed scale+crop via FFmpeg.
    reframe_out = os.path.join(trim_dir, f"reframe_{desc['variant_id']}.mp4")
    try:
        if desc["face_track"]:
            from opencut.core.face_reframe import reframe_face
            reframe_face(
                trim_out,
                output_path=reframe_out,
                target_width=int(desc["width"]),
                target_height=int(desc["height"]),
            )
        else:
            _fixed_crop(trim_out, reframe_out, int(desc["width"]), int(desc["height"]))
    except Exception as exc:
        # Fall back to a fixed crop if face_track wasn't available — never
        # let the variant pipeline die on one missing optional dep.
        try:
            _fixed_crop(trim_out, reframe_out, int(desc["width"]), int(desc["height"]))
        except Exception as exc2:
            shutil.rmtree(trim_dir, ignore_errors=True)
            raise RuntimeError(f"reframe failed: {exc2} (initial: {exc})") from exc2
    if on_progress:
        on_progress(pct_floor + (pct_ceil - pct_floor) // 2, f"variant {desc['variant_id']} reframed")

    # 3) Burn captions (optional) — uses the existing styled_captions surface.
    notes: List[str] = []
    if burn_captions and transcript_segments:
        try:
            _burn_captions(reframe_out, final_path, transcript_segments,
                           desc["caption_style"], desc["start"], desc["end"])
        except Exception as exc:
            logger.warning("variant %d caption burn-in failed: %s", desc["variant_id"], exc)
            notes.append("caption burn-in failed — output is uncaptioned")
            shutil.copyfile(reframe_out, final_path)
    else:
        shutil.copyfile(reframe_out, final_path)
        if not transcript_segments:
            notes.append("no transcript segments supplied — uncaptioned")

    # Clean intermediates.
    shutil.rmtree(trim_dir, ignore_errors=True)

    duration_ms = int((time.perf_counter() - start_ts) * 1000)
    if on_progress:
        on_progress(pct_ceil, f"variant {desc['variant_id']} rendered")

    return ShortsVariant(
        variant_id=desc["variant_id"],
        output=final_path,
        start=desc["start"],
        end=desc["end"],
        hook_offset=desc["hook_offset"],
        caption_style=desc["caption_style"],
        face_track=desc["face_track"],
        width=desc["width"],
        height=desc["height"],
        duration_ms=duration_ms,
        notes=notes,
    )


def _fixed_crop(input_path: str, output_path: str, w: int, h: int) -> None:
    """Static scale-and-crop (no face tracking)."""
    from opencut.helpers import get_ffmpeg_path, run_ffmpeg
    run_ffmpeg([
        get_ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-vf", f"scale=-2:'max({h},ih*{w}/iw)',crop={w}:{h}:(in_w-{w})/2:(in_h-{h})/3",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "copy",
        output_path,
    ], timeout=1800)


def _burn_captions(
    input_path: str,
    output_path: str,
    transcript_segments: List[dict],
    caption_style: str,
    window_start: float,
    window_end: float,
) -> None:
    """Burn the transcript subset into ``[window_start, window_end]``."""
    from opencut.core.styled_captions import burn_styled_captions
    # Clip the transcript to the variant window and shift to local times.
    window: List[dict] = []
    for seg in transcript_segments:
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", seg_start))
        if seg_end < window_start or seg_start > window_end:
            continue
        window.append({
            "start": max(0.0, seg_start - window_start),
            "end": max(0.0, seg_end - window_start),
            "text": str(seg.get("text", "")).strip(),
        })
    if not window:
        # Nothing to burn; copy through.
        shutil.copyfile(input_path, output_path)
        return
    burn_styled_captions(
        input_path,
        output_path,
        window,
        style=caption_style,
    )


def generate_variants(
    input_path: str,
    start: float,
    end: float,
    n_variants: int = DEFAULT_VARIANTS,
    width: int = 1080,
    height: int = 1920,
    transcript_segments: Optional[List[dict]] = None,
    burn_captions: bool = True,
    output_dir: str = "",
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> VariantsResult:
    """Render ``n_variants`` short-form variants of ``input_path[start:end]``.

    Args:
        input_path: Source video.
        start, end: Window in seconds within the source.
        n_variants: 2..6 variants (clamped).
        width, height: Target dimensions (default 1080×1920 vertical).
        transcript_segments: Optional list of ``{start, end, text}`` dicts
            in source-timeline seconds. When provided AND ``burn_captions``
            is True, the matching window is burned per variant.
        burn_captions: When False, the caption_style axis still varies in
            the descriptor but no burn-in happens.
        output_dir: Where to write the variant MP4s. Default = ``<src_dir>/variants``.
        on_progress: ``(pct, msg)`` callback. Pct ranges 0..100.

    Returns:
        :class:`VariantsResult` with one :class:`ShortsVariant` per variant.
    """
    s, e, _ = _validate_window(input_path, start, end)
    descriptors = _variant_descriptor(s, e, n_variants, width, height)

    out_dir = output_dir or os.path.join(os.path.dirname(input_path), "variants")
    os.makedirs(out_dir, exist_ok=True)

    if on_progress:
        on_progress(2, f"planning {len(descriptors)} variant(s)")

    result = VariantsResult(input_path=input_path, start=s, end=e, dry_run=False)

    span = max(1, 95 // len(descriptors))
    for i, desc in enumerate(descriptors):
        floor = 5 + i * span
        ceil = min(95, floor + span)
        variant = _render_one_variant(
            input_path=input_path,
            desc=desc,
            output_dir=out_dir,
            transcript_segments=transcript_segments,
            burn_captions=burn_captions,
            on_progress=on_progress,
            pct_floor=floor,
            pct_ceil=ceil,
        )
        result.variants.append(variant)

    if on_progress:
        on_progress(100, f"{len(result.variants)} variant(s) ready")

    return result


__all__ = [
    "CAPTION_STYLES",
    "DEFAULT_VARIANTS",
    "MIN_VARIANTS",
    "MAX_VARIANTS",
    "ShortsVariant",
    "VariantsResult",
    "INSTALL_HINT",
    "check_shorts_variants_available",
    "plan_variants",
    "generate_variants",
]
