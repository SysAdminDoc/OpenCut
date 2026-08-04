"""
OpenCut Workflow Engine

Executes a sequence of processing steps, chaining each step's output
as the next step's input.  Reports progress per-step and stops on
first failure.
"""

import copy
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

logger = logging.getLogger("opencut")
ROUTE_MANIFEST_PATH = Path(__file__).resolve().parents[1] / "_generated" / "route_manifest.json"

# The cleanup verb is intentionally a plan contract rather than another
# opaque workflow preset.  The preview is serialized into the client and the
# apply request must present the same plan id before any artifact is written.
CLEANUP_PLAN_SCHEMA_VERSION = 1
CLEANUP_CHAIN_ID = "standard-cleanup"
CLEANUP_STEP_IDS = ("silence_trim", "denoise", "loudness", "captions")

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
    total = len(steps)
    step_results = []  # type: List[Dict[str, Any]]
    current_input = filepath

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
        step_results.append({
            "step": step_num,
            "endpoint": endpoint,
            "success": True,
            "output": output,
            "job_id": job_id,
        })

        if output and os.path.isfile(output):
            current_input = output

    if on_progress:
        on_progress(100, "Workflow complete")

    return {
        "success": True,
        "steps_completed": total,
        "output": current_input,
        "step_results": step_results,
    }


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
