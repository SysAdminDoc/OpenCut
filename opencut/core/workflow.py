"""
OpenCut Workflow Engine

Executes a sequence of processing steps, chaining each step's output
as the next step's input.  Reports progress per-step and stops on
first failure.
"""

import copy
import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Known endpoints — maps route path to a human-readable label for progress
# messages.  Only POST endpoints that accept {filepath, ...} are valid
# workflow steps.
# ---------------------------------------------------------------------------
KNOWN_ENDPOINTS: Dict[str, str] = {
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
            result = _wait_for_job(app, job_id, csrf_token, step_num, label, on_progress, total)
            if result is None:
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
                  timeout: float = 3600, poll_interval: float = 0.5) -> Optional[Dict]:
    """Poll the /jobs/<job_id> endpoint until the job completes or times out.

    Instead of HTTP polling, we read from the in-memory job store directly
    for efficiency.
    """
    from opencut.jobs import _get_job_copy

    deadline = time.time() + timeout
    none_count = 0
    while time.time() < deadline:
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
        if status in ("error", "cancelled"):
            return job

        # Update progress within the step
        if on_progress:
            sub_pct = job.get("progress", 0)
            overall = int(((step_num - 1) / total) * 100 + (sub_pct / total))
            on_progress(min(overall, 99), "step %d/%d \u2014 %s (%d%%)" % (step_num, total, label, sub_pct))

        time.sleep(poll_interval)

    return None  # Timed out
