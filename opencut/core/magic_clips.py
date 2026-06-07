"""
Plan-first Magic Clips conductor.

This module builds a deterministic dry-run graph for the long-to-shorts
workflow without running transcription, FFmpeg probes, reframe, caption burn-in,
or render work. It turns cached transcript/highlight hints plus platform preset
choices into reviewable candidates and render steps.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from opencut.core.export_presets import EXPORT_PRESETS
from opencut.core.shorts_variants import CAPTION_STYLES

SCHEMA_VERSION = "opencut.magic_clips.plan.v1"
DEFAULT_PLATFORM_PRESETS = ("youtube_shorts",)
SUPPORTED_PLATFORM_PRESETS = (
    "youtube_shorts",
    "tiktok",
    "instagram_reels",
    "instagram_feed",
)


def check_magic_clips_available() -> bool:
    """Always True for dry-run planning; rendering remains in other modules."""
    return True


INSTALL_HINT = "Magic Clips planning is dry-run only and uses no optional dependencies."


@dataclass(frozen=True)
class MagicClipsConfig:
    max_candidates: int = 3
    min_duration: float = 15.0
    max_duration: float = 60.0
    platform_presets: tuple[str, ...] = DEFAULT_PLATFORM_PRESETS
    caption_style: str = "default"
    face_track: bool = True
    burn_captions: bool = True
    variants_per_candidate: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_candidates": self.max_candidates,
            "min_duration": self.min_duration,
            "max_duration": self.max_duration,
            "platform_presets": list(self.platform_presets),
            "caption_style": self.caption_style,
            "face_track": self.face_track,
            "burn_captions": self.burn_captions,
            "variants_per_candidate": self.variants_per_candidate,
        }


@dataclass
class MagicClipStep:
    step_id: str
    step_type: str
    status: str
    candidate_id: str = ""
    depends_on: list[str] = field(default_factory=list)
    estimated_outputs: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "step_type": self.step_type,
            "status": self.status,
            "candidate_id": self.candidate_id,
            "depends_on": list(self.depends_on),
            "estimated_outputs": [dict(item) for item in self.estimated_outputs],
            "reason": self.reason,
        }


@dataclass
class MagicClipCandidate:
    candidate_id: str
    index: int
    start: float
    end: float
    duration: float
    title: str = ""
    transcript_excerpt: str = ""
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    selection_reason: str = ""
    score_breakdown: dict[str, float] = field(default_factory=dict)
    fallback_mode: str = "heuristic_no_llm"
    source: str = "cached_transcript"
    platform_presets: list[dict[str, Any]] = field(default_factory=list)
    estimated_outputs: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "title": self.title,
            "transcript_excerpt": self.transcript_excerpt,
            "score": self.score,
            "reasons": list(self.reasons),
            "selection_reason": self.selection_reason,
            "score_breakdown": dict(self.score_breakdown),
            "fallback_mode": self.fallback_mode,
            "source": self.source,
            "platform_presets": [dict(item) for item in self.platform_presets],
            "estimated_outputs": [dict(item) for item in self.estimated_outputs],
        }


@dataclass
class MagicClipsPlan:
    schema_version: str = SCHEMA_VERSION
    plan_id: str = ""
    source_path: str = ""
    source_path_hash: str = ""
    config_hash: str = ""
    dry_run: bool = True
    requires_analysis: bool = False
    candidates: list[MagicClipCandidate] = field(default_factory=list)
    rejected_candidates: list[dict[str, Any]] = field(default_factory=list)
    fallback_mode: str = "heuristic_no_llm"
    steps: list[MagicClipStep] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def __getitem__(self, key: str) -> Any:
        if key == "candidates":
            return [candidate.to_dict() for candidate in self.candidates]
        if key == "rejected_candidates":
            return [dict(candidate) for candidate in self.rejected_candidates]
        if key == "steps":
            return [step.to_dict() for step in self.steps]
        return getattr(self, key)

    def keys(self):
        return (
            "schema_version",
            "plan_id",
            "source_path",
            "source_path_hash",
            "config_hash",
            "dry_run",
            "requires_analysis",
            "candidates",
            "rejected_candidates",
            "fallback_mode",
            "steps",
            "notes",
        )

    def __contains__(self, key: str) -> bool:
        return key in self.keys()


def _stable_hash(payload: Any, prefix: str = "") -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()
    return f"{prefix}{digest}" if prefix else digest


def _slug(value: str, fallback: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_")
    return slug[:64] or fallback


def _coerce_float(value: Any, default: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if value == value and value not in (float("inf"), float("-inf")) else default


def _coerce_int(value: Any, default: int, *, min_value: int, max_value: int) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default
    return max(min_value, min(max_value, value))


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _coerce_platform_presets(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, str):
        values: Iterable[Any] = [raw]
    elif isinstance(raw, Iterable):
        values = raw
    else:
        values = DEFAULT_PLATFORM_PRESETS

    presets: list[str] = []
    for value in values:
        preset = str(value or "").strip()
        if not preset:
            continue
        if preset not in SUPPORTED_PLATFORM_PRESETS:
            raise ValueError(
                f"Unsupported platform preset '{preset}'. "
                f"Valid: {', '.join(SUPPORTED_PLATFORM_PRESETS)}"
            )
        if preset not in presets:
            presets.append(preset)
    return tuple(presets or DEFAULT_PLATFORM_PRESETS)


def config_from_payload(payload: Mapping[str, Any] | None = None) -> MagicClipsConfig:
    payload = payload or {}
    min_duration = _coerce_float(payload.get("min_duration"), 15.0)
    max_duration = _coerce_float(payload.get("max_duration"), 60.0)
    if min_duration <= 0:
        raise ValueError("min_duration must be positive")
    if max_duration < min_duration:
        raise ValueError("max_duration must be greater than or equal to min_duration")

    caption_style = str(payload.get("caption_style") or "default")
    if caption_style not in CAPTION_STYLES:
        raise ValueError(f"Unsupported caption_style '{caption_style}'")

    raw_presets = payload.get("platform_presets", payload.get("platform_preset"))
    return MagicClipsConfig(
        max_candidates=_coerce_int(
            payload.get("max_candidates", payload.get("max_shorts")),
            3,
            min_value=1,
            max_value=20,
        ),
        min_duration=round(min_duration, 3),
        max_duration=round(max_duration, 3),
        platform_presets=_coerce_platform_presets(raw_presets),
        caption_style=caption_style,
        face_track=_coerce_bool(payload.get("face_track"), True),
        burn_captions=_coerce_bool(payload.get("burn_captions"), True),
        variants_per_candidate=_coerce_int(
            payload.get("variants_per_candidate"),
            1,
            min_value=1,
            max_value=6,
        ),
    )


def _normalise_segment(raw: Any, index: int) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not isinstance(raw, Mapping):
        return None, {"input_index": index, "reason": "malformed_transcript_segment"}
    start = _coerce_float(raw.get("start", raw.get("start_s")), 0.0)
    end = _coerce_float(raw.get("end", raw.get("end_s")), start)
    if start < 0 or end <= start:
        return None, {
            "input_index": index,
            "reason": "malformed_transcript_segment",
            "start": start,
            "end": end,
        }
    text = str(raw.get("text") or raw.get("transcript") or "").strip()
    speaker = str(raw.get("speaker") or raw.get("speaker_label") or "").strip()
    return {"start": round(start, 3), "end": round(end, 3), "text": text, "speaker": speaker}, None


def _normalise_segments(raw: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(raw, list):
        return [], []
    segments: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        segment, rejection = _normalise_segment(item, index)
        if segment:
            segments.append(segment)
        if rejection:
            rejected.append(rejection)
    return sorted(segments, key=lambda item: (item["start"], item["end"])), rejected


def _normalise_highlight(raw: Any, index: int) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not isinstance(raw, Mapping):
        return None, {"input_index": index, "reason": "malformed_highlight"}
    start = _coerce_float(raw.get("start", raw.get("start_s")), 0.0)
    end = _coerce_float(raw.get("end", raw.get("end_s")), start)
    if start < 0 or end <= start:
        return None, {
            "input_index": index,
            "reason": "malformed_highlight",
            "start": start,
            "end": end,
        }
    title = str(raw.get("title") or raw.get("name") or f"Candidate {index + 1}").strip()
    score = _coerce_float(raw.get("score"), 0.0)
    reason = str(raw.get("reason") or raw.get("selection_reason") or "cached highlight").strip()
    return {
        "input_index": index,
        "start": round(start, 3),
        "end": round(end, 3),
        "duration": round(end - start, 3),
        "title": title,
        "score": round(max(0.0, min(100.0, score)), 3),
        "reasons": [reason],
        "source": "cached_highlight",
    }, None


def _normalise_highlights(raw: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(raw, list):
        return [], []
    highlights: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        highlight, rejection = _normalise_highlight(item, index)
        if highlight:
            highlights.append(highlight)
        if rejection:
            rejected.append(rejection)
    return sorted(highlights, key=lambda item: (item["start"], item["end"])), rejected


def _text_for_window(segments: list[dict[str, Any]], start: float, end: float) -> str:
    texts: list[str] = []
    for segment in segments:
        if segment["end"] < start or segment["start"] > end:
            continue
        if segment["text"]:
            texts.append(segment["text"])
    return " ".join(texts).strip()


def _candidate_windows_from_segments(
    segments: list[dict[str, Any]],
    config: MagicClipsConfig,
) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for segment in segments:
        current.append(segment)
        start = current[0]["start"]
        end = current[-1]["end"]
        if end - start >= config.min_duration or len(current) >= 4:
            windows.append(_window_from_segments(current, len(windows)))
            current = []
        if len(windows) >= config.max_candidates:
            break
    if current and len(windows) < config.max_candidates:
        windows.append(_window_from_segments(current, len(windows)))
    return windows


def _window_from_segments(segments: list[dict[str, Any]], index: int) -> dict[str, Any]:
    start = segments[0]["start"]
    end = segments[-1]["end"]
    text = " ".join(segment["text"] for segment in segments if segment["text"]).strip()
    score = min(100.0, 40.0 + len(text.split()) * 1.5)
    return {
        "start": round(start, 3),
        "end": round(end, 3),
        "duration": round(end - start, 3),
        "title": f"Candidate {index + 1}",
        "score": round(score, 3),
        "reasons": ["cached transcript window"],
        "source": "cached_transcript",
        "input_index": index,
    }


def _duration_fit_score(duration: float, config: MagicClipsConfig) -> float:
    if duration < config.min_duration or duration > config.max_duration:
        return 0.0
    target = min(config.max_duration, max(config.min_duration, 30.0))
    span = max(config.max_duration - config.min_duration, 1.0)
    return round(max(70.0, 100.0 - (abs(duration - target) / span) * 30.0), 3)


def _speaker_continuity_score(segments: list[dict[str, Any]], start: float, end: float) -> float:
    speakers = [
        segment.get("speaker", "")
        for segment in segments
        if segment.get("speaker")
        and not (segment["end"] < start or segment["start"] > end)
    ]
    if not speakers:
        return 70.0
    transitions = sum(1 for prev, curr in zip(speakers, speakers[1:]) if prev != curr)
    return round(max(40.0, 100.0 - transitions * 15.0), 3)


def _hook_score(text: str) -> tuple[float, str]:
    try:
        from opencut.core.virality_score import _hook_lexicon_score

        score, phrase = _hook_lexicon_score(text)
        return round(float(score), 3), str(phrase or "")
    except Exception:
        words = len(text.split())
        return round(min(100.0, words * 2.0), 3), ""


def _score_candidate(item: dict[str, Any], segments: list[dict[str, Any]], config: MagicClipsConfig) -> dict[str, Any]:
    excerpt = _text_for_window(segments, item["start"], item["end"])
    hook_score, hook_phrase = _hook_score(excerpt)
    duration_fit = _duration_fit_score(item["duration"], config)
    speaker_continuity = _speaker_continuity_score(segments, item["start"], item["end"])
    highlight_score = round(float(item.get("score") or 0.0), 3)
    if highlight_score <= 0:
        highlight_score = 50.0
    score_breakdown = {
        "highlight_score": highlight_score,
        "transcript_hook": hook_score,
        "duration_fit": duration_fit,
        "speaker_continuity": speaker_continuity,
    }
    total = round(
        highlight_score * 0.4
        + hook_score * 0.3
        + duration_fit * 0.2
        + speaker_continuity * 0.1,
        3,
    )
    reasons = list(item.get("reasons") or [])
    if hook_phrase:
        reasons.append(f"hook phrase: {hook_phrase}")
    if duration_fit >= 90:
        reasons.append("duration fits platform window")
    if speaker_continuity >= 90:
        reasons.append("speaker continuity is strong")
    item = dict(item)
    item.update({
        "score": total,
        "score_breakdown": score_breakdown,
        "selection_reason": "; ".join(reasons[:3]) or "heuristic candidate score",
        "reasons": reasons or ["heuristic candidate score"],
        "fallback_mode": "heuristic_no_llm",
        "transcript_excerpt": excerpt,
    })
    return item


def _reject_duration_outliers(items: list[dict[str, Any]], config: MagicClipsConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for item in items:
        if item["duration"] < config.min_duration:
            rejected.append(_rejected_from_item(item, "too_short"))
        elif item["duration"] > config.max_duration:
            rejected.append(_rejected_from_item(item, "too_long"))
        else:
            accepted.append(item)
    return accepted, rejected


def _select_non_overlapping(
    items: list[dict[str, Any]],
    config: MagicClipsConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(
        items,
        key=lambda item: (-float(item.get("score") or 0.0), item["start"], int(item.get("input_index", 0))),
    )
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for item in ordered:
        if len(selected) >= config.max_candidates:
            rejected.append(_rejected_from_item(item, "below_selection_cutoff"))
            continue
        overlaps = next(
            (
                chosen
                for chosen in selected
                if not (item["end"] <= chosen["start"] or item["start"] >= chosen["end"])
            ),
            None,
        )
        if overlaps:
            rejected_item = _rejected_from_item(item, "overlap_with_selected")
            rejected_item["overlaps_candidate_start"] = overlaps["start"]
            rejected_item["overlaps_candidate_end"] = overlaps["end"]
            rejected.append(rejected_item)
            continue
        selected.append(item)
    return selected, rejected


def _rejected_from_item(item: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "input_index": int(item.get("input_index", 0)),
        "start": item.get("start"),
        "end": item.get("end"),
        "duration": item.get("duration"),
        "title": item.get("title", ""),
        "reason": reason,
        "score": item.get("score", 0.0),
        "score_breakdown": dict(item.get("score_breakdown") or {}),
    }


def _platform_contracts(config: MagicClipsConfig) -> list[dict[str, Any]]:
    contracts: list[dict[str, Any]] = []
    for preset_id in config.platform_presets:
        preset = EXPORT_PRESETS[preset_id]
        contracts.append({
            "preset_id": preset_id,
            "label": str(preset.get("label") or preset_id),
            "width": int(preset.get("width") or 0),
            "height": int(preset.get("height") or 0),
            "max_duration": preset.get("max_duration"),
            "ext": str(preset.get("ext") or ".mp4"),
        })
    return contracts


def _estimated_outputs(
    source_path: str,
    candidate_id: str,
    candidate_title: str,
    platforms: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source_stem = _slug(os.path.splitext(os.path.basename(source_path))[0], "source")
    title_slug = _slug(candidate_title, "candidate")
    safe_candidate = candidate_id.replace(":", "_")
    outputs: list[dict[str, Any]] = []
    for platform in platforms:
        ext = platform["ext"]
        outputs.append({
            "preset_id": platform["preset_id"],
            "filename_template": f"{source_stem}_{safe_candidate}_{platform['preset_id']}_{title_slug}{ext}",
            "width": platform["width"],
            "height": platform["height"],
            "max_duration": platform["max_duration"],
        })
    return outputs


def _build_steps(plan_id: str, candidates: list[MagicClipCandidate], config: MagicClipsConfig) -> list[MagicClipStep]:
    steps: list[MagicClipStep] = []
    for candidate in candidates:
        base = f"{plan_id}:{candidate.candidate_id}"
        trim_id = _stable_hash({"base": base, "step": "trim"}, "step:")
        steps.append(MagicClipStep(
            step_id=trim_id,
            step_type="trim",
            status="planned",
            candidate_id=candidate.candidate_id,
            reason="trim approved candidate window",
        ))
        reframe_id = _stable_hash({"base": base, "step": "reframe"}, "step:")
        steps.append(MagicClipStep(
            step_id=reframe_id,
            step_type="reframe",
            status="planned",
            candidate_id=candidate.candidate_id,
            depends_on=[trim_id],
            reason="apply target platform framing" if config.face_track else "apply fixed crop framing",
        ))
        previous = reframe_id
        if config.burn_captions:
            captions_id = _stable_hash({"base": base, "step": "captions", "style": config.caption_style}, "step:")
            steps.append(MagicClipStep(
                step_id=captions_id,
                step_type="captions",
                status="planned",
                candidate_id=candidate.candidate_id,
                depends_on=[previous],
                reason=f"burn captions with {config.caption_style} style",
            ))
            previous = captions_id
        export_id = _stable_hash({"base": base, "step": "export"}, "step:")
        steps.append(MagicClipStep(
            step_id=export_id,
            step_type="export",
            status="planned",
            candidate_id=candidate.candidate_id,
            depends_on=[previous],
            estimated_outputs=candidate.estimated_outputs,
            reason="write platform-specific exports",
        ))
    return steps


def _analysis_steps() -> list[MagicClipStep]:
    transcribe_id = "analysis:transcribe"
    highlights_id = "analysis:highlights"
    return [
        MagicClipStep(
            step_id=transcribe_id,
            step_type="transcribe",
            status="requires_analysis",
            reason="no cached transcript_segments supplied",
        ),
        MagicClipStep(
            step_id=highlights_id,
            step_type="highlight_extract",
            status="requires_analysis",
            depends_on=[transcribe_id],
            reason="no cached highlights supplied",
        ),
    ]


def build_magic_clips_plan(
    source_path: str,
    payload: Mapping[str, Any] | None = None,
    *,
    config: MagicClipsConfig | None = None,
) -> MagicClipsPlan:
    """Build a dry-run Magic Clips plan from cached hints and config."""
    payload = payload or {}
    source_path = os.path.abspath(source_path)
    config = config or config_from_payload(payload)
    source_path_hash = _stable_hash({"path": os.path.normcase(source_path)}, "src:")
    config_hash = _stable_hash(config.to_dict(), "cfg:")
    plan_id = _stable_hash({"source": source_path_hash, "config": config_hash}, "mcplan:")

    segments, rejected_candidates = _normalise_segments(payload.get("transcript_segments"))
    highlights, rejected_highlights = _normalise_highlights(payload.get("highlights", payload.get("candidates")))
    rejected_candidates.extend(rejected_highlights)
    if not highlights and segments:
        highlights = _candidate_windows_from_segments(segments, config)

    highlights, duration_rejections = _reject_duration_outliers(highlights, config)
    rejected_candidates.extend(duration_rejections)
    scored_highlights = [
        _score_candidate(item, segments, config)
        for item in highlights
    ]
    highlights, selection_rejections = _select_non_overlapping(scored_highlights, config)
    rejected_candidates.extend(selection_rejections)

    platforms = _platform_contracts(config)
    candidates: list[MagicClipCandidate] = []
    for index, item in enumerate(highlights):
        duration = round(item["end"] - item["start"], 3)
        if duration <= 0:
            continue
        candidate_payload = {
            "plan_id": plan_id,
            "index": index,
            "start": item["start"],
            "end": item["end"],
            "title": item["title"],
        }
        candidate_id = _stable_hash(candidate_payload, "mcc:")
        excerpt = str(item.get("transcript_excerpt") or "")
        if len(excerpt) > 240:
            excerpt = excerpt[:239].rstrip() + "..."
        estimated = _estimated_outputs(source_path, candidate_id, item["title"], platforms)
        candidates.append(MagicClipCandidate(
            candidate_id=candidate_id,
            index=index + 1,
            start=item["start"],
            end=item["end"],
            duration=duration,
            title=item["title"],
            transcript_excerpt=excerpt,
            score=item["score"],
            reasons=list(item["reasons"]),
            selection_reason=str(item.get("selection_reason") or ""),
            score_breakdown=dict(item.get("score_breakdown") or {}),
            fallback_mode=str(item.get("fallback_mode") or "heuristic_no_llm"),
            source=item["source"],
            platform_presets=platforms,
            estimated_outputs=estimated,
        ))

    requires_analysis = not candidates
    steps = _analysis_steps() if requires_analysis else _build_steps(plan_id, candidates, config)
    notes: list[str] = []
    if requires_analysis:
        notes.append("cached transcript_segments or highlights are required for dry-run candidates")
    else:
        notes.append("dry-run plan only; no media was rendered")

    return MagicClipsPlan(
        plan_id=plan_id,
        source_path=source_path,
        source_path_hash=source_path_hash,
        config_hash=config_hash,
        dry_run=True,
        requires_analysis=requires_analysis,
        candidates=candidates,
        rejected_candidates=rejected_candidates,
        fallback_mode="heuristic_no_llm",
        steps=steps,
        notes=notes,
    )


__all__ = [
    "DEFAULT_PLATFORM_PRESETS",
    "INSTALL_HINT",
    "MagicClipCandidate",
    "MagicClipStep",
    "MagicClipsConfig",
    "MagicClipsPlan",
    "SCHEMA_VERSION",
    "SUPPORTED_PLATFORM_PRESETS",
    "build_magic_clips_plan",
    "check_magic_clips_available",
    "config_from_payload",
]
