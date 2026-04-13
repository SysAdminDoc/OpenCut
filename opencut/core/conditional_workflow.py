"""
OpenCut Conditional Workflow Steps

Evaluate simple conditions against clip metadata and run workflow steps
conditionally based on media properties.
"""

import logging
import operator
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from opencut.helpers import get_video_info

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Condition parser
# ---------------------------------------------------------------------------
_OPERATORS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}

# Pattern: "field op value" e.g. "duration > 60", "width >= 1920"
_CONDITION_RE = re.compile(
    r"^(\w+)\s*(<=|>=|!=|==|<|>)\s*(-?\d+(?:\.\d+)?)$"
)

# Boolean flags -- no operator/value needed
_BOOLEAN_FLAGS = {"has_video", "has_audio"}


def evaluate_condition(condition_str: str, clip_metadata: Dict[str, Any]) -> bool:
    """Evaluate a simple condition string against clip metadata.

    Supported forms:
        - ``"duration > 60"`` -- numeric comparison
        - ``"loudness_lufs < -20"`` -- numeric comparison
        - ``"width >= 1920"`` -- numeric comparison
        - ``"has_video"`` -- boolean flag
        - ``"has_audio"`` -- boolean flag

    Args:
        condition_str: Condition expression string.
        clip_metadata: Dict with keys like width, height, fps, duration,
            loudness_lufs, has_video, has_audio, etc.

    Returns:
        True if condition is satisfied, False otherwise.
    """
    condition_str = condition_str.strip()
    if not condition_str:
        return True  # empty condition = always run

    # Boolean flags
    if condition_str in _BOOLEAN_FLAGS:
        return bool(clip_metadata.get(condition_str, False))

    # Negated boolean flags
    if condition_str.startswith("!") and condition_str[1:].strip() in _BOOLEAN_FLAGS:
        return not bool(clip_metadata.get(condition_str[1:].strip(), False))

    # Numeric comparisons
    match = _CONDITION_RE.match(condition_str)
    if not match:
        logger.warning("Unparseable condition: %r -- treating as False", condition_str)
        return False

    field_name, op_str, value_str = match.groups()
    op_fn = _OPERATORS[op_str]

    actual = clip_metadata.get(field_name)
    if actual is None:
        logger.debug("Condition field %r not in metadata -- False", field_name)
        return False

    try:
        actual_num = float(actual)
        target_num = float(value_str)
    except (ValueError, TypeError):
        logger.warning("Cannot compare %r=%r with %s -- False",
                       field_name, actual, value_str)
        return False

    result = op_fn(actual_num, target_num)
    logger.debug("Condition %r => %s %s %s = %s",
                 condition_str, actual_num, op_str, target_num, result)
    return result


# ---------------------------------------------------------------------------
# Metadata builder
# ---------------------------------------------------------------------------
def build_clip_metadata(input_path: str) -> Dict[str, Any]:
    """Build metadata dict for condition evaluation.

    Combines ffprobe info with optional loudness measurement.
    """
    info = get_video_info(input_path)
    meta = {
        "width": info.get("width", 0),
        "height": info.get("height", 0),
        "fps": info.get("fps", 0),
        "duration": info.get("duration", 0),
        "has_video": info.get("width", 0) > 0,
        "has_audio": True,  # assume True; refined below
    }

    # Attempt loudness measurement via ebur128
    try:
        from opencut.helpers import get_ffmpeg_path, run_ffmpeg
        cmd = [
            get_ffmpeg_path(), "-hide_banner", "-nostats",
            "-i", input_path,
            "-filter:a", "ebur128=peak=true",
            "-f", "null", "-",
        ]
        stderr = run_ffmpeg(cmd, timeout=120, stderr_cap=8000)
        # Parse integrated loudness from ebur128 summary
        import re as _re
        lufs_match = _re.search(r"I:\s*(-?\d+\.?\d*)\s*LUFS", stderr)
        if lufs_match:
            meta["loudness_lufs"] = float(lufs_match.group(1))
    except Exception as e:
        logger.debug("Loudness measurement failed for %s: %s", input_path, e)

    # Check for audio stream presence
    try:
        import json as _json
        import subprocess as _sp

        from opencut.helpers import get_ffprobe_path
        cmd = [
            get_ffprobe_path(), "-v", "quiet",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_type",
            "-of", "json", input_path,
        ]
        result = _sp.run(cmd, capture_output=True, timeout=15)
        if result.returncode == 0:
            data = _json.loads(result.stdout.decode())
            streams = data.get("streams", [])
            meta["has_audio"] = len(streams) > 0
    except Exception:
        pass

    return meta


# ---------------------------------------------------------------------------
# Conditional workflow runner
# ---------------------------------------------------------------------------
# Map of action names to core module functions
_ACTION_REGISTRY = {
    "export": "opencut.core.export_presets:export_with_preset",
    "normalize": "opencut.core.audio:normalize_audio",
    "denoise": "opencut.core.audio_enhance:denoise_audio",
    "silence_remove": "opencut.core.silence:remove_silence",
    "loudness_match": "opencut.core.loudness_match:match_loudness",
    "auto_edit": "opencut.core.auto_edit:auto_edit",
    "scene_detect": "opencut.core.scene_detect:detect_scenes",
}


def _resolve_action(action_name: str) -> Optional[Callable]:
    """Resolve an action name to its callable."""
    entry = _ACTION_REGISTRY.get(action_name)
    if not entry:
        return None
    module_path, func_name = entry.rsplit(":", 1)
    try:
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, func_name, None)
    except (ImportError, AttributeError) as e:
        logger.debug("Cannot resolve action %r: %s", action_name, e)
        return None


@dataclass
class StepResult:
    """Result of a single conditional workflow step."""
    action: str
    skipped: bool = False
    condition: str = ""
    condition_met: bool = True
    result: Any = None
    error: str = ""


def run_conditional_workflow(
    input_path: str,
    steps: List[Dict],
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> Dict:
    """Execute a workflow with conditional steps.

    Args:
        input_path: Path to input media file.
        steps: List of step dicts, each with:
            - ``"action"``: action name from registry
            - ``"params"``: dict of parameters (optional)
            - ``"condition"``: condition string (optional)
        on_progress: Progress callback ``(percent, message)``.

    Returns:
        Dict with ``"steps"`` list of per-step results and ``"completed"`` count.
    """
    if not steps:
        return {"steps": [], "completed": 0, "total": 0}

    # Build metadata once for all conditions
    if on_progress:
        on_progress(5, "Analyzing clip metadata...")
    metadata = build_clip_metadata(input_path)

    results = []
    total = len(steps)
    current_input = input_path
    completed = 0

    for i, step in enumerate(steps):
        action_name = step.get("action", "")
        params = step.get("params", {}) or {}
        condition = step.get("condition", "")

        base_pct = int(10 + (80 * i / total))
        if on_progress:
            on_progress(base_pct, f"Step {i+1}/{total}: {action_name}")

        step_result = StepResult(action=action_name, condition=condition)

        # Evaluate condition
        if condition:
            met = evaluate_condition(condition, metadata)
            step_result.condition_met = met
            if not met:
                step_result.skipped = True
                logger.info("Conditional workflow: skipping step %d (%s) -- condition %r not met",
                            i + 1, action_name, condition)
                results.append(step_result)
                continue

        # Resolve and execute action
        action_fn = _resolve_action(action_name)
        if action_fn is None:
            step_result.error = f"Unknown action: {action_name}"
            results.append(step_result)
            logger.warning("Conditional workflow: unknown action %r in step %d",
                           action_name, i + 1)
            continue

        try:
            # Build kwargs -- always pass input_path, merge user params
            kwargs = {"input_path": current_input}
            kwargs.update(params)

            # Add progress callback if supported
            def _step_progress(pct, msg=""):
                if on_progress:
                    overall = base_pct + int(pct * 0.8 / total)
                    on_progress(overall, f"Step {i+1}: {msg}")

            kwargs["on_progress"] = _step_progress

            result = action_fn(**kwargs)
            step_result.result = result

            # Chain output: if result is a string path, use as next input
            if isinstance(result, str) and os.path.isfile(result):
                current_input = result
            elif isinstance(result, dict) and result.get("output_path"):
                out = result["output_path"]
                if os.path.isfile(out):
                    current_input = out

            completed += 1
        except Exception as e:
            step_result.error = str(e)
            logger.error("Conditional workflow step %d (%s) error: %s",
                         i + 1, action_name, e)

        results.append(step_result)

    if on_progress:
        on_progress(100, "Conditional workflow complete")

    return {
        "steps": [
            {
                "action": r.action,
                "skipped": r.skipped,
                "condition": r.condition,
                "condition_met": r.condition_met,
                "result": r.result if not isinstance(r.result, Exception) else str(r.result),
                "error": r.error,
            }
            for r in results
        ],
        "completed": completed,
        "total": total,
    }


# Need os for path checks in run_conditional_workflow
import os  # noqa: E402
