"""
OpenCut Workflow Engine

Executes a sequence of processing steps, chaining each step's output
as the next step's input.  Reports progress per-step and stops on
first failure.
"""

import copy
import hashlib
import hmac
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

logger = logging.getLogger("opencut")
ROUTE_MANIFEST_PATH = Path(__file__).resolve().parents[1] / "_generated" / "route_manifest.json"

# The cleanup verb is intentionally a plan contract rather than another
# opaque workflow preset.  The preview is serialized into the client and the
# apply request must present the same plan id before any artifact is written.
CLEANUP_PLAN_SCHEMA_VERSION = 1
CLEANUP_CHAIN_ID = "standard-cleanup"
CLEANUP_STEP_IDS = ("silence_trim", "denoise", "loudness", "captions")

# Workflow execution is deliberately split into a deterministic compile phase
# and a side-effecting run phase.  The plan hash excludes only the mutable
# approval/checkpoint envelopes; every source, parameter, readiness, media,
# disk, output, and network decision remains bound to the hash.
WORKFLOW_PLAN_SCHEMA_VERSION = 1
WORKFLOW_PLAN_MAX_STEPS = 50
WORKFLOW_PLAN_MAX_PARAMS_BYTES = 64 * 1024

_WORKFLOW_MEDIA_EXTENSIONS = frozenset({
    ".3gp", ".avi", ".flac", ".m4a", ".mkv", ".mov", ".mp3", ".mp4",
    ".mpeg", ".mpg", ".ogg", ".wav", ".webm", ".wmv",
})

# The route handlers intentionally accept a broad parameter surface.  These
# common fields are the values whose type changes execution semantics and can
# therefore be rejected before a long-running job starts.  Unknown fields are
# preserved for forward-compatible route-specific options.
_WORKFLOW_PARAMETER_TYPES: Dict[str, str] = {
    "threshold": "number",
    "min_duration": "number",
    "min_speech": "number",
    "padding_before": "number",
    "padding_after": "number",
    "strength": "number",
    "noise_floor": "number",
    "target_lufs": "number",
    "speed": "number",
    "duration": "number",
    "start": "number",
    "end": "number",
    "fps": "number",
    "width": "integer",
    "height": "integer",
    "channels": "integer",
    "sample_rate": "integer",
    "overwrite": "boolean",
    "smart_pause": "boolean",
    "captions": "boolean",
    "no_input": "boolean",
    "aspect": "string",
    "preset": "string",
    "method": "string",
    "model": "string",
    "format": "string",
    "sequence_name": "string",
    "output_dir": "string",
    "output": "string",
    "output_path": "string",
    "output_file": "string",
}

_WORKFLOW_ROUTE_PARAMETER_TYPES: Dict[str, Dict[str, str]] = {
    "/silence": {key: _WORKFLOW_PARAMETER_TYPES[key] for key in (
        "threshold", "min_duration", "min_speech", "padding_before",
        "padding_after", "smart_pause", "method", "preset", "sequence_name",
        "output_dir",
    )},
    "/audio/denoise": {key: _WORKFLOW_PARAMETER_TYPES[key] for key in (
        "method", "strength", "noise_floor", "output_dir",
    )},
    "/audio/loudness-match": {key: _WORKFLOW_PARAMETER_TYPES[key] for key in (
        "target_lufs", "preset", "output_dir",
    )},
    "/video/reframe": {key: _WORKFLOW_PARAMETER_TYPES[key] for key in (
        "aspect", "width", "height", "output_dir",
    )},
    "/export-video": {key: _WORKFLOW_PARAMETER_TYPES[key] for key in (
        "format", "output", "output_path", "output_dir", "overwrite",
    )},
}

_WORKFLOW_CAPABILITY_KEYS: Dict[str, str] = {
    "/silence": "ffmpeg",
    "/audio/denoise": "ffmpeg",
    "/audio/normalize": "ffmpeg",
    "/audio/loudness-match": "ffmpeg",
    "/audio/separate": "separation",
    "/audio/pro/deepfilter": "deepfilter",
    "/video/ai/upscale": "video_ai",
    "/video/ai/denoise": "video_ai",
    "/video/depth/map": "depth_effects",
    "/video/depth/bokeh": "depth_effects",
    "/video/depth/parallax": "depth_effects",
    "/video/face/blur": "face_tools",
    "/video/face/enhance": "face_tools",
    "/video/face/swap": "face_tools",
    "/captions/whisperx": "whisperx",
    "/captions/translate": "nllb",
}

_WORKFLOW_CLOUD_PREFIXES = (
    "/cloud/", "/generate/cloud/", "/social/", "/delivery/",
)
_WORKFLOW_DESTRUCTIVE_MARKERS = (
    "/delete", "/remove/", "/watermark", "/clear", "/uninstall",
)

# ---------------------------------------------------------------------------
# Workflowable route markers.
# ---------------------------------------------------------------------------
def workflow_step(label: str):
    """Mark an async POST route as safe for sequential workflow execution."""
    clean_label = str(label or "").strip()
    if not clean_label:
        raise ValueError("workflow_step label is required")

    def decorator(func):
        setattr(func, "_opencut_workflow_step", {"label": clean_label})
        return func

    return decorator


def get_workflow_step_metadata(view_func: Any) -> Optional[Dict[str, str]]:
    """Return committed manifest metadata for a workflowable route."""
    if view_func is None or not getattr(view_func, "_opencut_async_job", False):
        return None
    raw = getattr(view_func, "_opencut_workflow_step", None)
    if isinstance(raw, Mapping):
        label = str(raw.get("label") or "").strip()
    elif isinstance(raw, str):
        label = raw.strip()
    else:
        label = ""
    if not label:
        return None
    return {"label": label}


# Labels retained as an additive compatibility fallback for older manifests.
_FALLBACK_WORKFLOW_ENDPOINTS: Dict[str, str] = {
    # Audio
    "/silence": "Detecting silence",
    "/fillers": "Removing filler words",
    "/audio/denoise": "Denoising audio",
    "/audio/isolate": "Isolating vocals",
    "/audio/separate": "Separating audio stems",
    "/audio/normalize": "Normalizing audio",
    "/audio/effects/apply": "Applying audio effects",
    "/audio/pro/apply": "Applying pro audio effects",
    "/audio/pro/deepfilter": "Running DeepFilter denoise",
    "/audio/tts/generate": "Generating TTS",
    "/audio/duck": "Ducking audio",
    "/audio/loudness-match": "Matching loudness",
    # Video
    "/video/scenes": "Detecting scenes",
    "/video/auto-edit": "Auto-editing",
    "/video/reframe": "Reframing video",
    "/video/reframe/face": "Reframing to face",
    "/video/trim": "Trimming video",
    "/video/merge": "Merging clips",
    "/video/speed/change": "Changing speed",
    "/video/speed/reverse": "Reversing video",
    "/video/speed/ramp": "Applying speed ramp",
    "/video/chromakey": "Applying chroma key",
    "/video/watermark": "Adding watermark",
    "/video/fx/apply": "Applying video FX",
    "/video/ai/upscale": "Upscaling video",
    "/video/ai/rembg": "Removing background",
    "/video/ai/interpolate": "Interpolating frames",
    "/video/ai/denoise": "Denoising video",
    "/video/face/blur": "Blurring faces",
    "/video/face/enhance": "Enhancing faces",
    "/video/face/swap": "Swapping faces",
    "/video/style/apply": "Applying style transfer",
    "/video/lut/apply": "Applying LUT",
    "/video/color/correct": "Correcting color",
    "/video/color-match": "Matching colors",
    "/video/auto-zoom": "Applying auto-zoom",
    "/video/highlights": "Extracting highlights",
    "/video/shorts-pipeline": "Running shorts pipeline",
    "/video/pip": "Adding picture-in-picture",
    "/video/blend": "Blending videos",
    "/video/transitions/apply": "Applying transitions",
    "/video/particles/apply": "Adding particles",
    "/video/title/render": "Rendering title",
    "/video/title/overlay": "Overlaying title",
    "/video/upscale/run": "Upscaling video",
    "/export-video": "Exporting video",
    # Captions
    "/captions": "Generating captions",
    "/styled-captions": "Generating styled captions",
    "/transcript": "Transcribing",
    "/captions/burnin/file": "Burning in captions",
    "/captions/animated/render": "Rendering animated captions",
    "/captions/chapters": "Generating chapters",
    "/captions/repeat-detect": "Detecting repeats",
}


def workflow_endpoints_from_manifest(
    manifest: Mapping[str, Any],
    *,
    include_fallback: bool = True,
) -> Dict[str, str]:
    """Return ``{route_rule: progress_label}`` from route manifest metadata."""
    endpoints: Dict[str, str] = {}
    routes = manifest.get("routes", [])
    if not isinstance(routes, list):
        return endpoints

    for route in routes:
        if not isinstance(route, Mapping):
            continue
        rule = str(route.get("rule") or "").strip()
        if not rule.startswith("/") or "<" in rule:
            continue
        methods = {str(method).upper() for method in (route.get("methods") or [])}
        if "POST" not in methods:
            continue

        label = ""
        workflow_meta = route.get("workflow")
        if isinstance(workflow_meta, Mapping):
            label = str(workflow_meta.get("label") or "").strip()
        if not label and include_fallback:
            label = _FALLBACK_WORKFLOW_ENDPOINTS.get(rule, "")
        if label:
            endpoints[rule] = label

    return dict(sorted(endpoints.items()))


def load_workflow_endpoints(path: Path = ROUTE_MANIFEST_PATH) -> Dict[str, str]:
    """Load workflowable endpoints from the generated route manifest."""
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Cannot load route manifest for workflow validation: %s", exc)
        return dict(_FALLBACK_WORKFLOW_ENDPOINTS)
    endpoints = workflow_endpoints_from_manifest(manifest)
    if not endpoints:
        return dict(_FALLBACK_WORKFLOW_ENDPOINTS)
    return endpoints


# Public compatibility name used by tests and workflow execution.
KNOWN_ENDPOINTS: Dict[str, str] = load_workflow_endpoints()


# ---------------------------------------------------------------------------
# Reviewable cleanup-chain plans.
# ---------------------------------------------------------------------------
def _cleanup_bool(value: Any, default: bool) -> bool:
    """Coerce the small set of boolean values accepted by the plan API."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value == value:
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", "", "none", "null"}:
            return False
    return default


def _cleanup_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number or number in (float("inf"), float("-inf")):
        return default
    return number


def _cleanup_segments(raw_segments: Any, duration: float) -> List[Dict[str, Any]]:
    """Normalize speech segments for a deterministic, JSON-safe plan."""
    if not isinstance(raw_segments, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for item in raw_segments:
        if isinstance(item, Mapping):
            raw_start = item.get("start", 0.0)
            raw_end = item.get("end", 0.0)
            label = str(item.get("label") or "speech")
        else:
            raw_start = getattr(item, "start", 0.0)
            raw_end = getattr(item, "end", 0.0)
            label = str(getattr(item, "label", "speech") or "speech")
        start = _cleanup_float(raw_start)
        end = _cleanup_float(raw_end)
        if start is None or end is None or end <= start:
            continue
        start = max(0.0, min(float(duration), start))
        end = max(0.0, min(float(duration), end))
        if end - start <= 1e-6:
            continue
        normalized.append({
            "start": round(start, 4),
            "end": round(end, 4),
            "label": label,
        })

    normalized.sort(key=lambda segment: (segment["start"], segment["end"]))
    merged: List[Dict[str, Any]] = []
    for segment in normalized:
        if merged and segment["start"] <= merged[-1]["end"] + 1e-6:
            merged[-1]["end"] = round(max(merged[-1]["end"], segment["end"]), 4)
        else:
            merged.append(segment)
    return merged


def _cleanup_removed_ranges(
    kept_segments: List[Dict[str, Any]], duration: float
) -> List[Dict[str, float]]:
    """Return the source ranges that the silence step proposes removing."""
    removed: List[Dict[str, float]] = []
    cursor = 0.0
    for segment in kept_segments:
        start = float(segment["start"])
        end = float(segment["end"])
        if start > cursor + 1e-6:
            removed.append({"start": round(cursor, 4), "end": round(start, 4)})
        cursor = max(cursor, end)
    if cursor < duration - 1e-6:
        removed.append({"start": round(cursor, 4), "end": round(duration, 4)})
    return removed


def cleanup_plan_id(plan: Mapping[str, Any]) -> str:
    """Return the content hash used to bind an apply request to its preview."""
    canonical = copy.deepcopy(dict(plan))
    canonical.pop("plan_id", None)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_cleanup_plan(
    *,
    filepath: str,
    duration: float,
    speech_segments: Any,
    options: Optional[Mapping[str, Any]] = None,
    capabilities: Optional[Mapping[str, Any]] = None,
    output_dir: str = "",
    source_loudness: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a reviewable silence → denoise → loudness → captions plan.

    This function is deliberately side-effect free.  It only normalizes
    analysis data and declares the artifacts that the apply job may write.
    Keeping it in the workflow core gives CEP, UXP, and future resumable
    workflows one stable plan shape to render and validate.
    """
    duration_value = _cleanup_float(duration, 0.0) or 0.0
    if duration_value <= 0:
        raise ValueError("Cleanup plans require a positive source duration")
    raw_source_path = str(filepath or "").strip()
    if not raw_source_path:
        raise ValueError("Cleanup plans require a source filepath")
    source_path = os.path.abspath(raw_source_path)

    raw_options = dict(options or {})
    preset = str(raw_options.get("preset") or "podcast").strip().lower() or "podcast"
    denoise_method = str(raw_options.get("denoise_method") or "afftdn").strip().lower()
    if denoise_method not in {"afftdn", "highpass", "gate"}:
        denoise_method = "afftdn"
    denoise_strength = _cleanup_float(raw_options.get("denoise_strength"), 0.7)
    denoise_strength = max(0.0, min(1.0, denoise_strength if denoise_strength is not None else 0.7))
    requested_target = _cleanup_float(raw_options.get("target_lufs"))
    if requested_target is None:
        from opencut.core.loudness_standards import get_loudness_preset

        requested_target = _cleanup_float(
            get_loudness_preset(preset).get("i"),
            -16.0,
        )
    requested_target = max(-70.0, min(0.0, requested_target or -16.0))
    captions_requested = _cleanup_bool(raw_options.get("captions", True), True)
    denoise_requested = _cleanup_bool(raw_options.get("denoise", True), True)
    loudness_requested = _cleanup_bool(raw_options.get("loudness", True), True)

    kept_segments = _cleanup_segments(speech_segments, duration_value)
    if not kept_segments:
        raise ValueError("Cleanup analysis did not produce any keepable speech segments")
    removed_ranges = _cleanup_removed_ranges(kept_segments, duration_value)
    removed_seconds = sum(item["end"] - item["start"] for item in removed_ranges)

    caps = dict(capabilities or {})
    ffmpeg_available = _cleanup_bool(caps.get("ffmpeg", True), True)
    captions_available = _cleanup_bool(caps.get("captions_available", False), False)
    captions_backend = str(caps.get("captions_backend") or "none")
    captions_reason = str(caps.get("captions_reason") or "")
    if not captions_reason and not captions_available:
        captions_reason = "No caption backend is installed"

    artifact_dir = os.path.abspath(str(output_dir or os.path.dirname(source_path)))
    base_name, extension = os.path.splitext(os.path.basename(source_path))
    artifacts = {
        "output_dir": artifact_dir,
        "denoised_path": os.path.join(artifact_dir, f"{base_name}_cleanup_denoised{extension}"),
        "normalized_path": os.path.join(artifact_dir, f"{base_name}_cleanup{extension}"),
        "xml_path": os.path.join(artifact_dir, f"{base_name}_cleanup.xml"),
        "srt_path": os.path.join(artifact_dir, f"{base_name}_cleanup.srt"),
    }

    def _step(step_id: str, label: str, state: str, **details: Any) -> Dict[str, Any]:
        return {
            "id": step_id,
            "label": label,
            "state": state,
            "status": state,
            **details,
        }

    denoise_details: Dict[str, Any] = {
        "requested": denoise_requested,
        "method": denoise_method,
        "strength": round(denoise_strength, 4),
        "output_path": artifacts["denoised_path"],
    }
    if denoise_requested and ffmpeg_available:
        denoise_details["capability"] = "ffmpeg"
    else:
        denoise_details["reason"] = (
            "Disabled by user" if not denoise_requested else "FFmpeg is unavailable"
        )

    loudness_details: Dict[str, Any] = {
        "requested": loudness_requested,
        "preset": preset,
        "target_lufs": requested_target,
        "output_path": artifacts["normalized_path"],
    }
    if loudness_requested and ffmpeg_available:
        loudness_details["capability"] = "ffmpeg"
    else:
        loudness_details["reason"] = (
            "Disabled by user" if not loudness_requested else "FFmpeg is unavailable"
        )

    captions_details: Dict[str, Any] = {
        "requested": captions_requested,
        "backend": captions_backend,
        "output_path": artifacts["srt_path"],
    }
    if captions_requested and captions_available:
        captions_details["capability"] = captions_backend
    else:
        captions_details["reason"] = (
            "Disabled by user" if not captions_requested else captions_reason
        )

    steps = [
        _step(
            "silence_trim",
            "Trim silence",
            "ready",
            proposed_changes=removed_ranges,
            kept_segments=len(kept_segments),
            removed_seconds=round(removed_seconds, 4),
        ),
        _step(
            "denoise",
            "Remove background noise",
            "ready" if denoise_requested and ffmpeg_available else "skipped",
            **denoise_details,
        ),
        _step(
            "loudness",
            "Normalize loudness",
            "ready" if loudness_requested and ffmpeg_available else "skipped",
            **loudness_details,
        ),
        _step(
            "captions",
            "Generate captions",
            "ready" if captions_requested and captions_available else "skipped",
            **captions_details,
        ),
    ]

    normalized_options = {
        "preset": preset,
        "denoise": denoise_requested,
        "denoise_method": denoise_method,
        "denoise_strength": round(denoise_strength, 4),
        "loudness": loudness_requested,
        "target_lufs": requested_target,
        "captions": captions_requested,
    }
    plan: Dict[str, Any] = {
        "schema_version": CLEANUP_PLAN_SCHEMA_VERSION,
        "chain": CLEANUP_CHAIN_ID,
        "verb": "cleanup",
        "preview_only": True,
        "requires_confirmation": True,
        "reversible": True,
        "source": {
            "filepath": source_path,
            "duration": round(duration_value, 4),
        },
        "options": normalized_options,
        "steps": steps,
        "segments_data": kept_segments,
        "removed_ranges": removed_ranges,
        "artifacts": artifacts,
        "summary": {
            "removed_seconds": round(removed_seconds, 4),
            "kept_segments": len(kept_segments),
            "removed_ranges": len(removed_ranges),
            "ready_steps": sum(step["state"] == "ready" for step in steps),
            "skipped_steps": sum(step["state"] == "skipped" for step in steps),
        },
    }
    if source_loudness:
        plan["source_loudness"] = dict(source_loudness)
    plan["plan_id"] = cleanup_plan_id(plan)
    return plan


def validate_cleanup_plan(
    plan: Any,
    *,
    filepath: str = "",
) -> Tuple[bool, str]:
    """Validate a client-returned cleanup plan before the apply job writes."""
    if not isinstance(plan, Mapping):
        return False, "Cleanup plan must be an object"
    if plan.get("schema_version") != CLEANUP_PLAN_SCHEMA_VERSION:
        return False, "Unsupported cleanup plan schema version"
    if plan.get("chain") != CLEANUP_CHAIN_ID or plan.get("verb") != "cleanup":
        return False, "Invalid cleanup plan chain"
    if not plan.get("preview_only"):
        return False, "Cleanup apply requires a preview plan"
    if not isinstance(plan.get("steps"), list) or len(plan["steps"]) != len(CLEANUP_STEP_IDS):
        return False, "Cleanup plan has an invalid step list"
    step_ids = [step.get("id") for step in plan["steps"] if isinstance(step, Mapping)]
    if step_ids != list(CLEANUP_STEP_IDS):
        return False, "Cleanup plan steps are out of order"
    source = plan.get("source")
    if not isinstance(source, Mapping) or not str(source.get("filepath") or "").strip():
        return False, "Cleanup plan is missing its source filepath"
    if filepath:
        expected = os.path.normcase(os.path.abspath(str(filepath)))
        actual = os.path.normcase(os.path.abspath(str(source.get("filepath"))))
        if expected != actual:
            return False, "Cleanup plan source does not match the requested file"
    if not isinstance(plan.get("segments_data"), list) or not plan["segments_data"]:
        return False, "Cleanup plan has no keepable speech segments"
    supplied_id = str(plan.get("plan_id") or "")
    if not supplied_id or supplied_id != cleanup_plan_id(plan):
        return False, "Cleanup plan id does not match its contents"
    return True, ""


def _workflow_json(value: Any) -> str:
    """Return the canonical JSON representation used by workflow hashes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _workflow_normalize_steps(steps: Any) -> List[Dict[str, Any]]:
    """Normalize a workflow definition without mutating the caller payload."""
    if not isinstance(steps, list):
        raise ValueError("Workflow must contain a list of steps")
    if not steps:
        raise ValueError("Workflow must contain at least one step")
    if len(steps) > WORKFLOW_PLAN_MAX_STEPS:
        raise ValueError(
            f"Workflow contains too many steps (max {WORKFLOW_PLAN_MAX_STEPS})"
        )

    normalized: List[Dict[str, Any]] = []
    for index, raw_step in enumerate(steps):
        if not isinstance(raw_step, Mapping):
            raise ValueError("Step %d is not a valid object" % (index + 1))
        endpoint = str(raw_step.get("endpoint") or "").strip()
        if not endpoint:
            raise ValueError("Step %d is missing an endpoint" % (index + 1))
        params = raw_step.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, Mapping):
            raise ValueError("Step %d params must be an object" % (index + 1))
        params = copy.deepcopy(dict(params))
        if "filepath" in params:
            raise ValueError(
                "Step %d must not override filepath; the workflow source is injected"
                % (index + 1)
            )
        try:
            encoded = _workflow_json(params).encode("utf-8")
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Step %d params must be JSON serializable" % (index + 1)) from exc
        if len(encoded) > WORKFLOW_PLAN_MAX_PARAMS_BYTES:
            raise ValueError(
                "Step %d params exceed the %d-byte limit"
                % (index + 1, WORKFLOW_PLAN_MAX_PARAMS_BYTES)
            )
        normalized.append({"endpoint": endpoint, "params": params})
    return normalized


def workflow_definition_id(steps: Any) -> str:
    """Hash the source-independent workflow definition used by saved templates."""
    normalized = _workflow_normalize_steps(steps)
    return hashlib.sha256(_workflow_json(normalized).encode("utf-8")).hexdigest()


def workflow_plan_id(plan: Mapping[str, Any]) -> str:
    """Return the immutable content hash for a compiled workflow plan."""
    canonical = copy.deepcopy(dict(plan))
    canonical.pop("plan_id", None)
    # Approval and checkpoint state are deliberately mutable envelopes.  The
    # source, steps, and all preflight evidence stay part of the immutable hash.
    canonical.pop("approval", None)
    canonical.pop("resume", None)
    encoded = _workflow_json(canonical)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def workflow_approval_token(plan_id: str, csrf_token: str) -> str:
    """Bind explicit approval to both the exact plan and the CSRF session."""
    secret = str(csrf_token or "").encode("utf-8")
    message = ("opencut-workflow-approval:" + str(plan_id or "")).encode("utf-8")
    return hmac.new(secret, message, hashlib.sha256).hexdigest()


def _workflow_parameter_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    if value is None:
        return "null"
    return type(value).__name__


def _validate_workflow_params(endpoint: str, params: Mapping[str, Any], step_number: int) -> None:
    """Reject parameter values that would be silently coerced by route helpers."""
    schema = dict(_WORKFLOW_PARAMETER_TYPES)
    schema.update(_WORKFLOW_ROUTE_PARAMETER_TYPES.get(endpoint, {}))
    for key, expected in schema.items():
        if key not in params:
            continue
        value = params[key]
        actual = _workflow_parameter_type(value)
        valid = (
            expected == actual
            or expected == "number" and actual == "integer"
        )
        if expected == "number" and actual in {"integer", "number"}:
            try:
                valid = math.isfinite(float(value))
            except (TypeError, ValueError, OverflowError):
                valid = False
        if not valid:
            raise ValueError(
                "Step %d parameter '%s' must be %s, got %s"
                % (step_number, key, expected, actual)
            )


def _workflow_route_metadata(path: Path = ROUTE_MANIFEST_PATH) -> Dict[str, Dict[str, Any]]:
    """Load readiness and labels for workflowable routes from the manifest."""
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        # The fallback endpoint list is a safety net for routing, but readiness
        # is a claim about what can actually run. Reporting "implemented" for
        # all 53 because the manifest was missing invented an answer, and a
        # packaged build with no opencut/_generated hit this every time.
        logger.error(
            "Route manifest unreadable (%s); reporting workflow readiness as unknown. "
            "A packaged build missing opencut/_generated causes this.", exc,
        )
        return {
            endpoint: {"label": label, "readiness": "unknown"}
            for endpoint, label in KNOWN_ENDPOINTS.items()
        }
    metadata: Dict[str, Dict[str, Any]] = {}
    for route in manifest.get("routes", []):
        if not isinstance(route, Mapping):
            continue
        endpoint = str(route.get("rule") or "").strip()
        if endpoint not in KNOWN_ENDPOINTS:
            continue
        workflow = route.get("workflow")
        metadata[endpoint] = {
            "label": str((workflow or {}).get("label") or KNOWN_ENDPOINTS[endpoint]),
            "readiness": str(route.get("readiness") or "implemented"),
        }
    for endpoint, label in KNOWN_ENDPOINTS.items():
        metadata.setdefault(endpoint, {"label": label, "readiness": "implemented"})
    return metadata


def _workflow_feature_readiness(endpoint: str) -> Optional[Dict[str, Any]]:
    """Return the live registry row that owns *endpoint*, when one exists."""
    try:
        from opencut.registry import list_features

        for record in list_features():
            if endpoint in (record.routes or []):
                return record.as_dict()
    except Exception as exc:  # pragma: no cover - registry is optional in old installs
        logger.debug("Workflow readiness lookup failed for %s: %s", endpoint, exc)
    return None


def _workflow_media_dict(info: Any) -> Dict[str, Any]:
    if isinstance(info, Mapping):
        return {
            "duration": float(info.get("duration") or 0.0),
            "format_name": str(info.get("format_name") or ""),
            "has_video": bool(info.get("has_video", info.get("video") is not None)),
            "has_audio": bool(info.get("has_audio", info.get("audio") is not None)),
            "video": copy.deepcopy(info.get("video")) if isinstance(info.get("video"), Mapping) else {},
            "audio": copy.deepcopy(info.get("audio")) if isinstance(info.get("audio"), Mapping) else {},
        }
    return {
        "duration": float(getattr(info, "duration", 0.0) or 0.0),
        "format_name": str(getattr(info, "format_name", "") or ""),
        "has_video": bool(getattr(info, "has_video", False)),
        "has_audio": bool(getattr(info, "has_audio", False)),
        "video": {
            "width": int(getattr(getattr(info, "video", None), "width", 0) or 0),
            "height": int(getattr(getattr(info, "video", None), "height", 0) or 0),
            "codec": str(getattr(getattr(info, "video", None), "codec", "") or ""),
            "fps": float(getattr(getattr(info, "video", None), "fps", 0.0) or 0.0),
        } if getattr(info, "video", None) is not None else {},
        "audio": {
            "sample_rate": int(getattr(getattr(info, "audio", None), "sample_rate", 0) or 0),
            "channels": int(getattr(getattr(info, "audio", None), "channels", 0) or 0),
            "codec": str(getattr(getattr(info, "audio", None), "codec", "") or ""),
        } if getattr(info, "audio", None) is not None else {},
    }


def _workflow_probe_media(filepath: str, media_probe: Optional[Callable] = None) -> Tuple[Dict[str, Any], str]:
    if media_probe is None:
        from opencut.utils.media import probe as media_probe
    try:
        return _workflow_media_dict(media_probe(filepath)), "probed"
    except Exception as exc:
        extension = os.path.splitext(filepath)[1].lower()
        state = "failed" if extension in _WORKFLOW_MEDIA_EXTENSIONS else "unknown"
        return {
            "duration": 0.0,
            "format_name": "",
            "has_video": False,
            "has_audio": False,
            "video": {},
            "audio": {},
            "error": str(exc)[:300],
        }, state


def _workflow_capability_value(capabilities: Mapping[str, Any], key: str) -> Optional[bool]:
    if key not in capabilities:
        return None
    value = capabilities.get(key)
    if isinstance(value, Mapping):
        if "available" in value:
            return bool(value.get("available"))
        if "ok" in value:
            return bool(value.get("ok"))
    return bool(value)


def _workflow_media_requirement(endpoint: str) -> str:
    lowered = endpoint.lower()
    if lowered.startswith("/video/") or lowered in {"/export-video", "/captions/burnin/file"}:
        return "video"
    if lowered.startswith("/audio/") or lowered in {"/silence", "/fillers", "/transcript", "/captions"}:
        return "audio"
    return "any"


def _workflow_side_effect(endpoint: str, params: Mapping[str, Any]) -> Tuple[str, bool, str]:
    lowered = endpoint.lower()
    has_external_url = any(
        isinstance(params.get(key), str)
        and params.get(key, "").strip().lower().startswith(("http://", "https://"))
        for key in ("url", "source_url", "reference_url", "webhook_url")
    )
    if has_external_url or lowered.startswith(_WORKFLOW_CLOUD_PREFIXES):
        return "cloud", False, "external network or remote service"
    if any(marker in lowered for marker in _WORKFLOW_DESTRUCTIVE_MARKERS) or bool(params.get("overwrite")):
        return "destructive", False, "overwrites or removes existing data"
    return "artifact_write", True, "writes a derived local artifact"


def _workflow_source_fingerprint(filepath: str) -> Dict[str, Any]:
    try:
        stat = os.stat(filepath)
    except OSError:
        return {}
    return {
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _workflow_explicit_outputs(
    steps: Iterable[Mapping[str, Any]], source_path: str, output_dir: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    outputs: List[Dict[str, Any]] = []
    collisions: List[Dict[str, Any]] = []
    seen: Dict[str, int] = {}
    for index, step in enumerate(steps):
        params = step.get("params") or {}
        candidate = ""
        for key in ("output_path", "output_file", "output"):
            value = params.get(key)
            if isinstance(value, str) and value.strip():
                candidate = value.strip()
                break
        if not candidate:
            continue
        try:
            from opencut.security import validate_output_path

            resolved = validate_output_path(candidate)
        except ValueError as exc:
            collisions.append({"step": index + 1, "path": candidate, "reason": str(exc)})
            continue
        normalized = os.path.normcase(os.path.abspath(resolved))
        entry = {
            "step": index + 1,
            "path": resolved,
            "exists": os.path.isfile(resolved),
            "overwrite": bool(params.get("overwrite")),
        }
        outputs.append(entry)
        if normalized == os.path.normcase(os.path.abspath(source_path)):
            collisions.append({"step": index + 1, "path": resolved, "reason": "output matches source"})
        if normalized in seen:
            collisions.append({"step": index + 1, "path": resolved, "reason": "duplicate output path"})
        seen[normalized] = index + 1
        if entry["exists"] and not entry["overwrite"]:
            collisions.append({"step": index + 1, "path": resolved, "reason": "output already exists"})
    if output_dir:
        try:
            from opencut.security import validate_path

            resolved_dir = validate_path(output_dir)
            if not os.path.isdir(resolved_dir):
                collisions.append({"path": resolved_dir, "reason": "output directory does not exist"})
        except ValueError as exc:
            collisions.append({"path": output_dir, "reason": str(exc)})
    return outputs, collisions


def compile_workflow_plan(
    filepath: str,
    steps: Any,
    *,
    output_dir: str = "",
    capabilities: Optional[Mapping[str, Any]] = None,
    media_info: Any = None,
    media_probe: Optional[Callable] = None,
    check_disk: bool = True,
) -> Dict[str, Any]:
    """Compile a typed, readiness-aware, resumable workflow plan.

    Compilation is side-effect free apart from read-only media, capability,
    and disk probes.  A plan with blocked checks is returned for review rather
    than being executable; callers can render the exact reasons before asking
    for approval.
    """
    normalized_steps = _workflow_normalize_steps(steps)
    for index, step in enumerate(normalized_steps):
        if step["endpoint"] not in KNOWN_ENDPOINTS:
            raise ValueError("Step %d has unknown endpoint: %s" % (index + 1, step["endpoint"]))
        _validate_workflow_params(step["endpoint"], step["params"], index + 1)
        if output_dir and "output_dir" not in step["params"]:
            step["params"]["output_dir"] = str(output_dir)

    source_path = os.path.abspath(str(filepath).strip()) if str(filepath or "").strip() else ""
    checks: List[Dict[str, Any]] = []
    blocked_reasons: List[str] = []
    approval_reasons: List[str] = []

    if source_path and not os.path.isfile(source_path):
        checks.append({"id": "source", "state": "block", "message": "Source file does not exist"})
        blocked_reasons.append("Source file does not exist")
    elif source_path:
        checks.append({"id": "source", "state": "pass", "message": "Source file is readable"})

    if media_info is not None:
        media = _workflow_media_dict(media_info)
        media_state = "provided"
    elif source_path:
        media, media_state = _workflow_probe_media(source_path, media_probe)
    else:
        media = {"duration": 0.0, "format_name": "", "has_video": False, "has_audio": False, "video": {}, "audio": {}}
        media_state = "not_requested"
    if media_state == "probed":
        checks.append({"id": "media_probe", "state": "pass", "message": "Media streams are available"})
    elif media_state == "failed":
        checks.append({"id": "media_probe", "state": "block", "message": media.get("error", "Media probe failed")})
        blocked_reasons.append("Media probe failed")
    elif media_state == "unknown":
        checks.append({"id": "media_probe", "state": "warn", "message": "Media type could not be probed; stream checks are deferred"})

    caps = dict(capabilities or {})
    route_metadata = _workflow_route_metadata()
    step_plans: List[Dict[str, Any]] = []
    all_network_local = True
    for index, step in enumerate(normalized_steps):
        endpoint = step["endpoint"]
        params = step["params"]
        metadata = route_metadata.get(endpoint, {})
        readiness = str(metadata.get("readiness") or "implemented")
        feature = _workflow_feature_readiness(endpoint)
        feature_state = str(feature.get("state") or "") if feature else ""
        if readiness == "stub" or feature_state == "stub":
            message = "Route is not implemented"
            checks.append({"id": f"readiness-{index + 1}", "state": "block", "message": message, "endpoint": endpoint})
            blocked_reasons.append(f"{endpoint}: {message}")
        elif readiness == "dependency-gated" or feature_state == "missing_dependency":
            message = str((feature or {}).get("state_reason") or "Required dependency is unavailable")
            checks.append({"id": f"readiness-{index + 1}", "state": "block", "message": message, "endpoint": endpoint})
            blocked_reasons.append(f"{endpoint}: {message}")
        elif feature_state == "experimental":
            checks.append({"id": f"readiness-{index + 1}", "state": "warn", "message": "Route is experimental", "endpoint": endpoint})

        requirement = _workflow_media_requirement(endpoint)
        if media_state in {"probed", "provided"} and requirement != "any":
            has_stream = bool(media.get("has_video" if requirement == "video" else "has_audio"))
            if not has_stream:
                message = f"Source has no {requirement} stream required by {endpoint}"
                checks.append({"id": f"stream-{index + 1}", "state": "block", "message": message, "endpoint": endpoint})
                blocked_reasons.append(message)
            else:
                checks.append({"id": f"stream-{index + 1}", "state": "pass", "message": f"Source has a {requirement} stream", "endpoint": endpoint})
        elif requirement != "any":
            checks.append({"id": f"stream-{index + 1}", "state": "warn", "message": f"{requirement.title()} stream check deferred", "endpoint": endpoint})

        capability_key = _WORKFLOW_CAPABILITY_KEYS.get(endpoint)
        capability_value = _workflow_capability_value(caps, capability_key) if capability_key else None
        if capability_key and capability_value is False:
            if media_state in {"probed", "provided"}:
                message = f"Required capability unavailable: {capability_key}"
                checks.append({"id": f"capability-{index + 1}", "state": "block", "message": message, "endpoint": endpoint})
                blocked_reasons.append(message)
            else:
                checks.append({"id": f"capability-{index + 1}", "state": "warn", "message": f"Capability will be checked at run time: {capability_key}", "endpoint": endpoint})
        elif capability_key:
            checks.append({"id": f"capability-{index + 1}", "state": "pass" if capability_value is True else "unknown", "message": capability_key, "endpoint": endpoint})

        side_effect, idempotent, side_effect_reason = _workflow_side_effect(endpoint, params)
        network = "external" if side_effect == "cloud" else "local"
        if network == "external":
            all_network_local = False
            try:
                from opencut.config import is_local_only

                local_only = bool(is_local_only())
            except Exception:
                local_only = False
            if local_only:
                message = "Local-only mode blocks external network access"
                checks.append({"id": f"network-{index + 1}", "state": "block", "message": message, "endpoint": endpoint})
                blocked_reasons.append(f"{endpoint}: {message}")
            else:
                approval_reasons.append(f"{endpoint}: {side_effect_reason}")
                checks.append({"id": f"network-{index + 1}", "state": "approval_required", "message": "External network access requires explicit approval", "endpoint": endpoint})
        else:
            checks.append({"id": f"network-{index + 1}", "state": "pass", "message": "Local-only operation", "endpoint": endpoint})
        if side_effect == "destructive":
            approval_reasons.append(f"{endpoint}: {side_effect_reason}")
            checks.append({"id": f"side-effect-{index + 1}", "state": "approval_required", "message": "Destructive step requires explicit approval", "endpoint": endpoint})

        step_plans.append({
            "index": index,
            "endpoint": endpoint,
            "label": str(metadata.get("label") or KNOWN_ENDPOINTS.get(endpoint, endpoint)),
            "params": copy.deepcopy(params),
            "parameter_types": {
                key: _WORKFLOW_PARAMETER_TYPES.get(key, _workflow_parameter_type(value))
                for key, value in params.items()
            },
            "readiness": readiness,
            "media_requirement": requirement,
            "capability": capability_key or "",
            "side_effect": side_effect,
            "side_effect_reason": side_effect_reason,
            "network": network,
            "idempotent": idempotent,
            "checkpoint": "artifact_checksum" if idempotent else "manual_review",
        })

    explicit_outputs, collisions = _workflow_explicit_outputs(normalized_steps, source_path, output_dir)
    for collision in collisions:
        message = f"Output collision: {collision.get('path', '')} ({collision.get('reason', 'invalid path')})"
        state = "approval_required" if collision.get("reason") == "output already exists" else "block"
        checks.append({"id": "output", "state": state, "message": message})
        if state == "block":
            blocked_reasons.append(message)
        else:
            approval_reasons.append(message)

    disk = {}
    if check_disk and source_path:
        try:
            from opencut.core.preflight import ensure_disk_for

            disk = ensure_disk_for("workflow", source_path, {"output_dir": output_dir} if output_dir else {})
            if not disk.get("ok", True):
                message = "Insufficient disk space for workflow outputs"
                checks.append({"id": "disk", "state": "block", "message": message, "details": disk})
                blocked_reasons.append(message)
            else:
                checks.append({"id": "disk", "state": "pass", "message": "Output volume has enough free space", "details": disk})
        except (OSError, ValueError) as exc:
            message = f"Disk preflight failed: {exc}"
            checks.append({"id": "disk", "state": "block", "message": message})
            blocked_reasons.append(message)

    try:
        definition_id = workflow_definition_id(normalized_steps)
    except ValueError:
        raise
    plan: Dict[str, Any] = {
        "schema_version": WORKFLOW_PLAN_SCHEMA_VERSION,
        "definition_id": definition_id,
        "source": {
            "filepath": source_path,
            "fingerprint": _workflow_source_fingerprint(source_path) if source_path else {},
            "media": media,
        },
        "steps": step_plans,
        "preflight": {
            "status": "blocked" if blocked_reasons else "ready",
            "checks": checks,
            "blocked_reasons": blocked_reasons,
            "approval_reasons": approval_reasons,
            "media": media,
            "capabilities": {
                key: value for key, value in caps.items()
                if key in {"ffmpeg", "ffprobe", "gpu", "separation", "deepfilter", "video_ai", "depth_effects", "face_tools", "whisperx", "nllb"}
            },
            "disk": disk,
            "outputs": explicit_outputs,
            "output_policy": {
                "mode": "explicit-paths-or-route-unique-output",
                "allow_existing": False,
                "collisions": collisions,
            },
            "network": "local" if all_network_local else "external",
        },
        "approval": {
            "required": bool(approval_reasons),
            "approved": False,
            "plan_id": "",
            "token": "",
        },
        "resume": {
            "enabled": True,
            "strategy": "idempotent-artifact-checksum",
            "completed_steps": 0,
        },
    }
    plan["plan_id"] = workflow_plan_id(plan)
    plan["approval"]["plan_id"] = plan["plan_id"]
    return plan


def compile_workflow_template(steps: Any) -> Dict[str, Any]:
    """Compile the source-independent portion persisted with saved workflows."""
    return compile_workflow_plan("", steps, check_disk=False)


def validate_workflow_plan(
    plan: Any,
    *,
    filepath: str = "",
    steps: Any = None,
) -> Tuple[bool, str]:
    """Validate a client-returned plan without re-running its mutable probes."""
    if not isinstance(plan, Mapping):
        return False, "Workflow plan must be an object"
    if plan.get("schema_version") != WORKFLOW_PLAN_SCHEMA_VERSION:
        return False, "Unsupported workflow plan schema version"
    supplied_id = str(plan.get("plan_id") or "")
    if not supplied_id or supplied_id != workflow_plan_id(plan):
        return False, "Workflow plan id does not match its contents"
    source = plan.get("source")
    if not isinstance(source, Mapping):
        return False, "Workflow plan is missing its source"
    if filepath:
        expected = os.path.normcase(os.path.abspath(str(filepath)))
        actual = os.path.normcase(os.path.abspath(str(source.get("filepath") or "")))
        if not actual or expected != actual:
            return False, "Workflow plan source does not match the requested file"
    plan_steps = plan.get("steps")
    if not isinstance(plan_steps, list) or not plan_steps:
        return False, "Workflow plan has no steps"
    try:
        plan_definition = [
            {"endpoint": item.get("endpoint"), "params": item.get("params", {})}
            for item in plan_steps
            if isinstance(item, Mapping)
        ]
        if len(plan_definition) != len(plan_steps):
            return False, "Workflow plan contains an invalid step"
        normalized_definition = _workflow_normalize_steps(plan_definition)
        for index, step in enumerate(normalized_definition):
            endpoint = step["endpoint"]
            if endpoint not in KNOWN_ENDPOINTS:
                return False, "Workflow plan contains unknown endpoint: %s" % endpoint
            _validate_workflow_params(endpoint, step["params"], index + 1)
        if workflow_definition_id(plan_definition) != str(plan.get("definition_id") or ""):
            return False, "Workflow plan definition id does not match its steps"
        if steps is not None and workflow_definition_id(steps) != str(plan.get("definition_id") or ""):
            return False, "Workflow plan does not match the requested workflow steps"
    except (TypeError, ValueError) as exc:
        return False, str(exc)
    preflight = plan.get("preflight")
    if not isinstance(preflight, Mapping):
        return False, "Workflow plan is missing preflight results"
    if preflight.get("status") == "blocked":
        return False, "Workflow plan is blocked: " + "; ".join(
            str(reason) for reason in preflight.get("blocked_reasons", [])
        )
    return True, ""


def workflow_plan_requires_approval(plan: Mapping[str, Any]) -> bool:
    return bool((plan.get("approval") or {}).get("required"))


def validate_workflow_approval(plan: Mapping[str, Any], token: str, csrf_token: str) -> bool:
    approval = plan.get("approval") or {}
    if not workflow_plan_requires_approval(plan):
        return True
    if not approval.get("approved") or approval.get("plan_id") != plan.get("plan_id"):
        return False
    expected = workflow_approval_token(str(plan.get("plan_id") or ""), csrf_token)
    supplied = str(token or approval.get("token") or "")
    return bool(supplied) and hmac.compare_digest(supplied, expected)


def validate_workflow_steps(steps: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """Validate that all steps reference known endpoints.

    Returns ``(True, "")`` on success or ``(False, error_message)`` on
    failure.
    """
    if not steps or not isinstance(steps, list):
        return False, "Workflow must contain at least one step"

    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            return False, "Step %d is not a valid object" % (i + 1)
        endpoint = step.get("endpoint", "")
        if not endpoint:
            return False, "Step %d is missing an endpoint" % (i + 1)
        if endpoint not in KNOWN_ENDPOINTS:
            return False, "Step %d has unknown endpoint: %s" % (i + 1, endpoint)
    return True, ""


def run_workflow(
    app,
    filepath: str,
    steps: List[Dict[str, Any]],
    csrf_token: str,
    on_progress: Optional[Callable[[int, str], None]] = None,
    parent_job_id: str = "",
    *,
    plan: Optional[Mapping[str, Any]] = None,
    resume_state: Optional[Mapping[str, Any]] = None,
    on_checkpoint: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Execute a workflow — a sequential chain of processing steps.

    Parameters
    ----------
    app : Flask
        The Flask application instance (used to create a test client).
    filepath : str
        The path to the initial input file.
    steps : list[dict]
        Each dict has ``endpoint`` (str) and optional ``params`` (dict).
    csrf_token : str
        A valid CSRF token for internal requests.
    on_progress : callable, optional
        ``on_progress(percentage, message)`` called after each step.

    Returns
    -------
    dict
        ``{"success": True/False, "steps_completed": N, "output": path,
          "step_results": [...], "error": optional_str}``
    """
    if plan is not None:
        plan_steps = plan.get("steps") if isinstance(plan, Mapping) else None
        if not isinstance(plan_steps, list) or not plan_steps:
            return {
                "success": False,
                "steps_completed": 0,
                "output": filepath,
                "step_results": [],
                "error": "Workflow plan has no executable steps",
            }
        steps = [
            {"endpoint": item.get("endpoint"), "params": copy.deepcopy(item.get("params") or {})}
            for item in plan_steps
        ]

    total = len(steps)
    step_results = []  # type: List[Dict[str, Any]]
    current_input = filepath
    resume_payload = resume_state if isinstance(resume_state, Mapping) else {}
    if isinstance(resume_payload.get("result"), Mapping):
        resume_payload = resume_payload["result"]
    prior_results = resume_payload.get("step_results") if isinstance(resume_payload, Mapping) else None
    if not isinstance(prior_results, list):
        prior_results = []
    try:
        resume_count = max(0, min(total, int(resume_payload.get("steps_completed", 0))))
    except (TypeError, ValueError):
        resume_count = 0

    def _checkpoint() -> None:
        if on_checkpoint is None:
            return
        payload = {
            "success": True,
            "steps_completed": len([item for item in step_results if item.get("success")]),
            "total_steps": total,
            "output": current_input,
            "step_results": copy.deepcopy(step_results),
        }
        if plan and plan.get("plan_id"):
            payload["plan_id"] = plan["plan_id"]
        try:
            on_checkpoint(payload)
        except Exception as exc:  # checkpoint persistence must not break the active step
            logger.warning("Workflow checkpoint callback failed: %s", exc)

    def _artifact_checksum(path: str) -> str:
        digest = hashlib.sha256()
        try:
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError:
            return ""
        return digest.hexdigest()

    def _can_resume(previous: Any) -> bool:
        if not isinstance(previous, Mapping) or not previous.get("success"):
            return False
        if previous.get("idempotent") is False:
            return False
        artifact_path = str(previous.get("artifact_path") or "")
        checksum = str(previous.get("artifact_checksum") or "")
        if not artifact_path or not checksum or not os.path.isfile(artifact_path):
            return False
        return _artifact_checksum(artifact_path) == checksum

    for i, step in enumerate(steps):
        # Check if the parent workflow job was cancelled between steps
        if parent_job_id:
            from opencut.jobs import _is_cancelled
            if _is_cancelled(parent_job_id):
                logger.info("Workflow cancelled after step %d/%d", i, total)
                return {
                    "success": False,
                    "steps_completed": i,
                    "output": current_input,
                    "step_results": step_results,
                    "error": "Workflow cancelled by user",
                }

        step_num = i + 1
        endpoint = step["endpoint"]
        params = step.get("params", {})
        label = KNOWN_ENDPOINTS.get(endpoint, endpoint)

        # A restarted workflow may reuse only an idempotent artifact whose
        # checksum still matches the durable checkpoint.  Missing or changed
        # artifacts deliberately restart at the first unsafe step.
        if i < resume_count and i < len(prior_results) and _can_resume(prior_results[i]):
            previous = copy.deepcopy(prior_results[i])
            current_input = str(previous.get("artifact_path") or current_input)
            previous["resumed"] = True
            step_results.append(previous)
            if on_progress:
                on_progress(int((step_num / total) * 100), "step %d/%d — %s (resumed)" % (step_num, total, label))
            _checkpoint()
            continue
        if i < resume_count:
            resume_count = i

        if on_progress:
            pct = int((i / total) * 100)
            on_progress(pct, "step %d/%d \u2014 %s" % (step_num, total, label))

        logger.info("Workflow step %d/%d: %s on %s", step_num, total, endpoint, current_input)

        # Build the request payload — always inject the current file
        payload = copy.deepcopy(params)
        payload["filepath"] = current_input

        # Use Flask test client to invoke the endpoint internally
        try:
            with app.test_client() as tc:
                resp = tc.post(
                    endpoint,
                    data=json.dumps(payload),
                    content_type="application/json",
                    headers={"X-OpenCut-Token": csrf_token},
                )
        except Exception as exc:
            error_msg = "Step %d (%s) request failed: %s" % (step_num, label, exc)
            logger.error(error_msg)
            step_results.append({"step": step_num, "endpoint": endpoint, "success": False, "error": str(exc)})
            return {
                "success": False,
                "steps_completed": i,
                "output": current_input,
                "step_results": step_results,
                "error": error_msg,
            }

        resp_data = resp.get_json() or {}

        # Most async endpoints return {"job_id": "..."} with 200.
        # For the workflow engine we treat a 2xx response as step-success
        # and look for an output path in the result.
        if resp.status_code >= 400:
            error_msg = "Step %d (%s) failed (HTTP %d): %s" % (
                step_num, label, resp.status_code, resp_data.get("error", "unknown error"),
            )
            logger.error(error_msg)
            step_results.append({
                "step": step_num,
                "endpoint": endpoint,
                "success": False,
                "error": resp_data.get("error", "HTTP %d" % resp.status_code),
            })
            return {
                "success": False,
                "steps_completed": i,
                "output": current_input,
                "step_results": step_results,
                "error": error_msg,
            }

        # If the endpoint returned a job_id, we need to poll for completion.
        job_id = resp_data.get("job_id")
        if job_id:
            result = _wait_for_job(
                app,
                job_id,
                csrf_token,
                step_num,
                label,
                on_progress,
                total,
                parent_job_id=parent_job_id,
            )
            if result is None:
                if parent_job_id:
                    from opencut.jobs import _cancel_job

                    _cancel_job(
                        job_id,
                        message="Workflow step timed out",
                        persist_sync=True,
                    )
                error_msg = "Step %d (%s) job timed out" % (step_num, label)
                step_results.append({"step": step_num, "endpoint": endpoint, "success": False, "error": error_msg})
                return {
                    "success": False,
                    "steps_completed": i,
                    "output": current_input,
                    "step_results": step_results,
                    "error": error_msg,
                }
            if result.get("status") == "error":
                error_msg = "Step %d (%s) failed: %s" % (step_num, label, result.get("error", "unknown"))
                step_results.append({"step": step_num, "endpoint": endpoint, "success": False, "error": result.get("error", "unknown")})
                return {
                    "success": False,
                    "steps_completed": i,
                    "output": current_input,
                    "step_results": step_results,
                    "error": error_msg,
                }
            if result.get("status") == "cancelled":
                error_msg = "Step %d (%s) was cancelled" % (step_num, label)
                step_results.append({"step": step_num, "endpoint": endpoint, "success": False, "error": "Cancelled"})
                return {
                    "success": False,
                    "steps_completed": i,
                    "output": current_input,
                    "step_results": step_results,
                    "error": error_msg,
                }
            resp_data = result.get("result", {}) or {}

        # Determine output file for chaining.
        # Different endpoints use different result keys.
        output = _extract_output_path(resp_data, current_input)
        artifact_path = ""
        artifact_checksum = ""
        if output and os.path.isfile(output):
            # A fallback to the current input is not a new artifact and is not
            # safe to use as a resume checkpoint for a side-effecting step.
            if os.path.normcase(os.path.abspath(output)) != os.path.normcase(os.path.abspath(current_input)):
                artifact_path = output
                artifact_checksum = _artifact_checksum(output)
        step_result = {
            "step": step_num,
            "endpoint": endpoint,
            "success": True,
            "output": output,
            "job_id": job_id,
        }
        if plan:
            step_result.update({
                "artifact_path": artifact_path,
                "artifact_checksum": artifact_checksum,
                "idempotent": bool(plan.get("steps", [{}] * total)[i].get("idempotent", True)),
            })
        step_results.append(step_result)

        if output and os.path.isfile(output):
            current_input = output
        _checkpoint()

    if on_progress:
        on_progress(100, "Workflow complete")

    result = {
        "success": True,
        "steps_completed": total,
        "output": current_input,
        "step_results": step_results,
    }
    if plan and plan.get("plan_id"):
        result["plan_id"] = plan["plan_id"]
    return result


def _extract_output_path(result: Any, fallback: str) -> str:
    """Try to extract an output file path from a step result dict."""
    if not isinstance(result, dict):
        return fallback

    # Common result keys across OpenCut endpoints
    for key in ("output", "output_path", "output_file", "file", "path",
                "trimmed", "merged", "exported"):
        val = result.get(key, "")
        if val and isinstance(val, str) and os.path.isfile(val):
            return val

    # Some endpoints return a list of outputs
    outputs = result.get("outputs", [])
    if isinstance(outputs, list) and outputs:
        last = outputs[-1]
        if isinstance(last, str) and os.path.isfile(last):
            return last
        if isinstance(last, dict):
            for key in ("output", "output_path", "path", "file"):
                val = last.get(key, "")
                if val and isinstance(val, str) and os.path.isfile(val):
                    return val

    return fallback


def _wait_for_job(app, job_id: str, csrf_token: str, step_num: int,
                   label: str, on_progress, total: int,
                   timeout: float = 3600, poll_interval: float = 0.5,
                   parent_job_id: str = "") -> Optional[Dict]:
    """Poll the /jobs/<job_id> endpoint until the job completes or times out.

    Instead of HTTP polling, we read from the in-memory job store directly
    for efficiency.
    """
    from opencut.jobs import _cancel_job, _get_job_copy, _is_cancelled

    deadline = time.time() + timeout
    none_count = 0
    while time.time() < deadline:
        if parent_job_id and _is_cancelled(parent_job_id):
            _cancel_job(
                job_id,
                message="Cancelled because the parent workflow was cancelled",
                persist_sync=True,
            )
            cancelled_job = _get_job_copy(job_id)
            if cancelled_job is not None:
                return cancelled_job
            return {
                "id": job_id,
                "status": "cancelled",
                "error": "Cancelled because the parent workflow was cancelled",
            }

        job = _get_job_copy(job_id)
        if job is None:
            none_count += 1
            # If job disappears after being seen, it was likely cleaned up
            if none_count > 20:
                logger.warning("Workflow step %d job %s disappeared from memory", step_num, job_id)
                return None
            time.sleep(poll_interval)
            continue
        none_count = 0  # Reset once we see the job

        status = job.get("status", "")
        if status == "complete":
            return job
        if status in ("error", "cancelled", "interrupted"):
            return job

        # Update progress within the step
        if on_progress:
            sub_pct = job.get("progress", 0)
            overall = int(((step_num - 1) / total) * 100 + (sub_pct / total))
            on_progress(min(overall, 99), "step %d/%d \u2014 %s (%d%%)" % (step_num, total, label, sub_pct))

        time.sleep(poll_interval)

    return None  # Timed out
