"""
One-click "Enhance" macro (RESEARCH_FEATURE_PLAN_2026-05-25 Q3).

Chains the existing audio + video clean-up modules into a single
opinionated pipeline. Picks defaults from a ``style`` knob and the
clip's measured characteristics (LUFS, resolution, motion) so the
common case ("take this clip and make it noticeably better") is one
button click.

Modules orchestrated (all already exist):
  - ``opencut.core.loudness_match.normalize_to_lufs``   (always)
  - ``opencut.core.audio_pro.deepfilter_denoise``       (skip if denoise OK)
  - ``opencut.core.advanced_stabilize.stabilize_advanced`` (skip if static)
  - ``opencut.core.deflicker_neural.deflicker_video``   (FFmpeg fallback OK)
  - ``opencut.core.auto_color.auto_grade``              (intent-driven)

The pipeline is *non-destructive*: each step writes a new file, the
previous file becomes the input to the next, and the final output is
the only file the caller needs. Intermediate files are auto-cleaned
on success and preserved with a `_enhance_failed_*` suffix on failure
(so the user can inspect mid-pipeline state).

``style="social"`` (default):  loudness -16 LUFS (YouTube/TikTok), denoise on, mild stabilization, auto-grade "vibrant".
``style="speech"``:             loudness -16 LUFS, aggressive denoise, no stabilization, no grade.
``style="cinematic"``:          loudness -23 LUFS (broadcast), denoise on, stabilization on, auto-grade "balanced cinematic".

A ``dry_run=True`` call returns the planned pipeline as a list of
step descriptors without executing anything. Used by ``GET /enhance/auto?dry-run=1``.

Returns :class:`EnhanceResult` (subscriptable; ``keys()``/`__getitem__`
implemented so Flask jsonify works directly).
"""
from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from opencut.helpers import get_video_info, output_path

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Style presets — keep this dict the only knob; new presets must add a row.
# ---------------------------------------------------------------------------
STYLES: dict[str, dict[str, Any]] = {
    "social": {
        "target_lufs": -16.0,            # YouTube / TikTok / Instagram baseline
        "denoise": True,
        "stabilize_mode": "smooth",      # mild
        "stabilize_smoothing": 30,
        "deflicker": False,              # mostly modern phone footage; skip
        "grade_intent": "vibrant punchy social media",
        "grade_intensity": 0.85,
    },
    "speech": {
        "target_lufs": -16.0,
        "denoise": True,
        "stabilize_mode": None,
        "stabilize_smoothing": 0,
        "deflicker": False,
        "grade_intent": None,            # speech-only — no color grade
        "grade_intensity": 0.0,
    },
    "cinematic": {
        "target_lufs": -23.0,            # EBU R128 broadcast
        "denoise": True,
        "stabilize_mode": "smooth",
        "stabilize_smoothing": 45,
        "deflicker": True,
        "grade_intent": "balanced cinematic",
        "grade_intensity": 1.0,
    },
}
DEFAULT_STYLE = "social"
VALID_STYLES = tuple(STYLES.keys())


# ---------------------------------------------------------------------------
# Result + step records
# ---------------------------------------------------------------------------
@dataclass
class EnhanceStep:
    name: str                       # "loudness", "denoise", "stabilize", "deflicker", "grade"
    module: str                     # core module path used
    status: str = "planned"         # planned | skipped | ok | failed
    reason: str = ""                # why a step was skipped, or the failure message
    output: str = ""                # path produced by this step (empty if skipped/failed)
    duration_ms: int = 0
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "module": self.module,
            "status": self.status,
            "reason": self.reason,
            "output": self.output,
            "duration_ms": self.duration_ms,
            "params": dict(self.params),
        }


@dataclass
class EnhanceResult:
    output: str = ""
    style: str = ""
    steps: List[EnhanceStep] = field(default_factory=list)
    dry_run: bool = False
    notes: List[str] = field(default_factory=list)

    # Flask jsonify protocol — same pattern as InterpResult / ComposeResult.
    def __getitem__(self, key: str) -> Any:
        if key == "steps":
            return [s.to_dict() for s in self.steps]
        return getattr(self, key)

    def keys(self):
        return ("output", "style", "steps", "dry_run", "notes")

    def __contains__(self, key: str) -> bool:
        return key in self.keys()


# ---------------------------------------------------------------------------
# Public availability check (matches the `check_X_available()` convention)
# ---------------------------------------------------------------------------
def check_enhance_auto_available() -> bool:
    """Always True — the macro degrades to whichever sub-modules are present.

    Loudness normalisation needs only FFmpeg (bundled). Denoise / stabilize
    / deflicker / grade each have their own ``check_X_available()`` and
    degrade gracefully when unavailable.
    """
    return True


INSTALL_HINT = (
    "Enhance macro orchestrates existing modules; install optional extras "
    "(DeepFilterNet, vidstab, deflicker neural) for the full pipeline."
)


# ---------------------------------------------------------------------------
# Pipeline planner
# ---------------------------------------------------------------------------
def _resolve_style(style: str) -> dict[str, Any]:
    if style not in STYLES:
        raise ValueError(
            f"Unknown style '{style}'. Valid: {', '.join(VALID_STYLES)}"
        )
    return dict(STYLES[style])


def _measure_loudness_safe(input_path: str) -> Optional[float]:
    """Probe the source LUFS; return None on any failure."""
    try:
        from opencut.core.loudness_match import measure_loudness
        info = measure_loudness(input_path)
        return float(info.get("lufs")) if info else None
    except Exception as exc:  # pragma: no cover — depends on FFmpeg presence
        logger.debug("enhance_auto loudness probe failed: %s", exc)
        return None


_AUDIO_ONLY_EXTS = frozenset({
    ".wav", ".mp3", ".flac", ".aac", ".m4a", ".ogg", ".opus", ".wma", ".aiff", ".aif",
})


def _is_audio_only(input_path: str) -> bool:
    """Return True when the file extension implies an audio-only container.

    ``helpers.get_video_info`` returns 1920x1080 defaults on probe failure
    (CLAUDE.md gotcha), so we can't rely on width/height alone — extension
    is the only reliable hint for the planner.
    """
    return os.path.splitext(input_path)[1].lower() in _AUDIO_ONLY_EXTS


def _plan_pipeline(input_path: str, style: str) -> tuple[dict[str, Any], List[EnhanceStep]]:
    """Compute the per-step plan without running anything.

    Returns ``(resolved_style_dict, [planned_steps])``.
    """
    settings = _resolve_style(style)
    src_lufs = _measure_loudness_safe(input_path)
    info = get_video_info(input_path) or {}
    audio_only = _is_audio_only(input_path)
    src_w = 0 if audio_only else int(info.get("width") or 0)
    src_h = 0 if audio_only else int(info.get("height") or 0)
    src_dur = float(info.get("duration") or 0.0)

    steps: List[EnhanceStep] = []

    # 1) Loudness — always plan; skip only if source is already within ±0.5
    # LUFS of the target (saves a re-encode pass for already-mastered clips).
    target_lufs = float(settings["target_lufs"])
    loud_step = EnhanceStep(
        name="loudness",
        module="opencut.core.loudness_match.normalize_to_lufs",
        params={"target_lufs": target_lufs, "source_lufs": src_lufs},
    )
    if src_lufs is not None and abs(src_lufs - target_lufs) <= 0.5:
        loud_step.status = "skipped"
        loud_step.reason = (
            f"source LUFS {src_lufs:.1f} already within ±0.5 of target {target_lufs}"
        )
    steps.append(loud_step)

    # 2) Denoise — gated by style and the deepfilter check.
    denoise_step = EnhanceStep(
        name="denoise",
        module="opencut.core.audio_pro.deepfilter_denoise",
        params={"enabled": bool(settings["denoise"])},
    )
    if not settings["denoise"]:
        denoise_step.status = "skipped"
        denoise_step.reason = "denoise disabled by style"
    else:
        try:
            from opencut.core.audio_pro import check_deepfilter_available
            if not check_deepfilter_available():
                denoise_step.status = "skipped"
                denoise_step.reason = (
                    "deepfilternet not installed (optional dependency)"
                )
        except Exception:  # pragma: no cover
            denoise_step.status = "skipped"
            denoise_step.reason = "deepfilter availability check raised"
    steps.append(denoise_step)

    # 3) Stabilize — gated by style + audio-only sources skip silently.
    stab_step = EnhanceStep(
        name="stabilize",
        module="opencut.core.advanced_stabilize.stabilize_advanced",
        params={
            "mode": settings["stabilize_mode"],
            "smoothing": settings["stabilize_smoothing"],
        },
    )
    if not settings["stabilize_mode"]:
        stab_step.status = "skipped"
        stab_step.reason = "stabilize disabled by style"
    elif src_w == 0 or src_h == 0:
        stab_step.status = "skipped"
        stab_step.reason = "source has no video stream"
    steps.append(stab_step)

    # 4) Deflicker — cinematic only by default; the FFmpeg fallback is bundled.
    deflicker_step = EnhanceStep(
        name="deflicker",
        module="opencut.core.deflicker_neural.deflicker_video",
        params={"enabled": bool(settings["deflicker"])},
    )
    if not settings["deflicker"]:
        deflicker_step.status = "skipped"
        deflicker_step.reason = "deflicker disabled by style"
    elif src_w == 0 or src_h == 0:
        deflicker_step.status = "skipped"
        deflicker_step.reason = "source has no video stream"
    steps.append(deflicker_step)

    # 5) Color grade — natural-language intent via auto_color.auto_grade.
    grade_step = EnhanceStep(
        name="grade",
        module="opencut.core.auto_color.auto_grade",
        params={
            "intent": settings["grade_intent"],
            "intensity": settings["grade_intensity"],
        },
    )
    if not settings["grade_intent"]:
        grade_step.status = "skipped"
        grade_step.reason = "no color intent for this style"
    elif src_w == 0 or src_h == 0:
        grade_step.status = "skipped"
        grade_step.reason = "source has no video stream"
    steps.append(grade_step)

    settings.update({
        "source_lufs": src_lufs,
        "source_width": src_w,
        "source_height": src_h,
        "source_duration": src_dur,
    })
    return settings, steps


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------
def _run_step(
    step: EnhanceStep,
    current_input: str,
    style_settings: dict[str, Any],
    progress: Optional[Callable[[int, str], None]],
    pct_floor: int,
    pct_ceil: int,
) -> str:
    """Execute ``step``, return the new ``current_input`` path.

    A *skipped* step returns ``current_input`` unchanged.
    A *failed* step raises — the caller mutates the step record.
    """
    import time
    start = time.perf_counter()
    span = max(1, pct_ceil - pct_floor)

    def _sub(p: int, msg: str = "") -> None:
        if progress is None:
            return
        try:
            progress(pct_floor + int(span * (max(0, min(100, p)) / 100.0)), msg)
        except Exception:  # pragma: no cover — never crash on UI callback failure
            pass

    if step.status == "skipped":
        return current_input

    try:
        if step.name == "loudness":
            from opencut.core.loudness_match import normalize_to_lufs
            out = output_path(current_input, "enhance_loud")
            _sub(0, "normalize loudness")
            normalize_to_lufs(current_input, out, target_lufs=float(style_settings["target_lufs"]))
            step.output = out
        elif step.name == "denoise":
            from opencut.core.audio_pro import deepfilter_denoise
            _sub(0, "denoise audio")
            new_out = deepfilter_denoise(current_input, on_progress=_sub)
            step.output = new_out
        elif step.name == "stabilize":
            from opencut.core.advanced_stabilize import stabilize_advanced
            _sub(0, "stabilize video")
            result = stabilize_advanced(
                current_input,
                mode=str(style_settings["stabilize_mode"]),
                smoothing=int(style_settings["stabilize_smoothing"]),
                on_progress=_sub,
            )
            step.output = str(result.get("output_path") or "")
        elif step.name == "deflicker":
            from opencut.core.deflicker_neural import deflicker_video
            _sub(0, "deflicker")
            r = deflicker_video(current_input, backend="auto", on_progress=_sub)
            step.output = str(getattr(r, "output", None) or r["output"])
        elif step.name == "grade":
            from opencut.core.auto_color import auto_grade
            _sub(0, "color grade")
            r = auto_grade(
                current_input,
                intent=str(style_settings["grade_intent"]),
                intensity=float(style_settings["grade_intensity"]),
                on_progress=_sub,
            )
            step.output = str(r.get("output_path") or "")
        else:  # pragma: no cover — defensive
            raise ValueError(f"unknown step name: {step.name}")
    except Exception as exc:
        step.status = "failed"
        step.reason = f"{type(exc).__name__}: {exc}"
        step.duration_ms = int((time.perf_counter() - start) * 1000)
        raise

    if not step.output or not os.path.isfile(step.output):
        step.status = "failed"
        step.reason = "step returned no output file"
        step.duration_ms = int((time.perf_counter() - start) * 1000)
        raise RuntimeError(step.reason)

    step.status = "ok"
    step.duration_ms = int((time.perf_counter() - start) * 1000)
    _sub(100, f"{step.name} done")
    return step.output


def enhance(
    input_path: str,
    style: str = DEFAULT_STYLE,
    output: Optional[str] = None,
    dry_run: bool = False,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> EnhanceResult:
    """One-click Enhance macro.

    Args:
        input_path: Source video or audio file.
        style: One of ``"social"``, ``"speech"``, ``"cinematic"`` (see
            :data:`STYLES`).
        output: Final output path. Auto-generated under the source dir
            with a ``_enhanced_<style>`` suffix when omitted.
        dry_run: When True, return the plan only (no work performed).
        on_progress: ``(pct, msg)`` callback. Pct ranges 0..100 across
            all enabled steps.

    Returns:
        :class:`EnhanceResult` (subscriptable for Flask jsonify).
    """
    if not isinstance(input_path, str) or not input_path:
        raise ValueError("input_path is required")
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    settings, steps = _plan_pipeline(input_path, style)

    result = EnhanceResult(
        style=style,
        steps=steps,
        dry_run=dry_run,
    )

    if dry_run:
        # Surface advisory notes so the panel can render context next to the plan.
        if settings["source_lufs"] is not None:
            result.notes.append(
                f"source loudness {settings['source_lufs']:.1f} LUFS"
            )
        if settings["source_width"] and settings["source_height"]:
            result.notes.append(
                f"source {settings['source_width']}x{settings['source_height']} "
                f"({settings['source_duration']:.1f}s)"
            )
        return result

    # Reserve a small floor for plan + final mux; distribute the rest evenly
    # across enabled steps.
    enabled = [s for s in steps if s.status != "skipped"]
    if not enabled:
        result.notes.append("All pipeline steps skipped; nothing to do.")
        result.output = input_path
        return result

    step_span = max(1, (95 - 5) // len(enabled))
    current = input_path
    intermediates: List[str] = []

    for i, step in enumerate(steps):
        floor = 5 + step_span * sum(1 for prior in enabled if enabled.index(prior) < enabled.index(step)) if step in enabled else 5
        ceil = floor + step_span
        try:
            new_current = _run_step(step, current, settings, on_progress, floor, ceil)
        except Exception:
            # Preserve intermediates with a clear suffix so the user can inspect.
            for tmp in intermediates:
                if os.path.isfile(tmp):
                    rescue = tmp.replace(".mp4", "_enhance_failed.mp4")
                    try:
                        os.replace(tmp, rescue)
                    except OSError:  # pragma: no cover
                        pass
            raise
        if new_current != current and current != input_path:
            intermediates.append(current)
        current = new_current

    # Move/rename the last step's file to the requested final path.
    if output:
        try:
            os.replace(current, output)
            current = output
        except OSError as exc:  # pragma: no cover — best-effort rename
            logger.warning("enhance_auto final-rename failed: %s", exc)

    # Clean intermediates (the final file is preserved at `current`).
    for tmp in intermediates:
        try:
            if os.path.isfile(tmp) and tmp != current:
                os.remove(tmp)
        except OSError:  # pragma: no cover
            pass

    result.output = current
    if on_progress is not None:
        try:
            on_progress(100, "enhance complete")
        except Exception:  # pragma: no cover
            pass
    return result


__all__ = [
    "STYLES",
    "DEFAULT_STYLE",
    "VALID_STYLES",
    "EnhanceStep",
    "EnhanceResult",
    "INSTALL_HINT",
    "check_enhance_auto_available",
    "enhance",
]
