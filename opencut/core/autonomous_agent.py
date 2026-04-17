"""
OpenCut Multi-Step Autonomous Editing Agent

LLM-powered agent that plans and executes multi-step editing workflows
from natural language instructions. Uses a tool-use pattern where the
LLM generates API calls and the agent executes them sequentially.

Extends the NL intent matching from timeline_copilot with multi-step
planning and autonomous execution.

Requires: Configured LLM provider (via opencut.core.llm)
"""

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import get_video_info

logger = logging.getLogger("opencut")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class AgentGoal:
    """Description of what the user wants to achieve."""
    description: str = ""
    constraints: List[str] = field(default_factory=list)
    style: str = ""
    duration_target: Optional[float] = None


@dataclass
class AgentStep:
    """A single step in an agent plan."""
    action: str = ""
    endpoint: str = ""
    params: Dict = field(default_factory=dict)
    status: str = "pending"  # pending, running, complete, error, skipped
    result: Optional[Dict] = None
    error: str = ""
    retry_count: int = 0
    duration_seconds: float = 0.0


@dataclass
class AgentPlan:
    """Complete execution plan."""
    plan_id: str = ""
    goal: AgentGoal = field(default_factory=AgentGoal)
    steps: List[AgentStep] = field(default_factory=list)
    current_step_idx: int = 0
    status: str = "created"  # created, running, complete, error, cancelled
    execution_log: List[str] = field(default_factory=list)
    created_at: float = 0.0
    input_files: List[str] = field(default_factory=list)


@dataclass
class AgentResult:
    """Result from agent execution."""
    plan: AgentPlan = field(default_factory=AgentPlan)
    output_paths: List[str] = field(default_factory=list)
    summary: str = ""
    total_duration: float = 0.0
    steps_completed: int = 0
    steps_failed: int = 0


# ---------------------------------------------------------------------------
# Tool registry — maps tool names to internal functions
# ---------------------------------------------------------------------------
AGENT_TOOLS = {
    "silence_removal": {
        "description": "Remove silent segments from video/audio",
        "params": ["filepath", "threshold", "min_duration", "padding"],
        "module": "opencut.core.silence",
        "function": "remove_silence",
    },
    "caption_generation": {
        "description": "Generate captions/subtitles from speech",
        "params": ["filepath", "model", "language"],
        "module": "opencut.core.captions",
        "function": "generate_captions",
    },
    "audio_normalization": {
        "description": "Normalize audio levels (loudness, peak, or RMS)",
        "params": ["filepath", "target_level", "method"],
        "module": "opencut.core.audio",
        "function": "normalize_audio",
    },
    "color_correction": {
        "description": "Apply automatic color correction",
        "params": ["filepath", "preset", "intensity"],
        "module": "opencut.core.auto_color",
        "function": "auto_color_correct",
    },
    "scene_detection": {
        "description": "Detect scene boundaries in video",
        "params": ["filepath", "threshold", "min_scene_length"],
        "module": "opencut.core.scene_detect",
        "function": "detect_scenes",
    },
    "highlight_extraction": {
        "description": "Extract highlights from video based on audio/visual energy",
        "params": ["filepath", "target_duration", "min_clip_duration"],
        "module": "opencut.core.highlights",
        "function": "extract_highlights",
    },
    "music_generation": {
        "description": "Generate background music for video",
        "params": ["filepath", "genre", "duration", "mood"],
        "module": "opencut.core.music_gen",
        "function": "generate_music",
    },
    "export": {
        "description": "Export/render video with specified settings",
        "params": ["filepath", "output_path", "codec", "quality"],
        "module": "opencut.core.export_presets",
        "function": "export_video",
    },
    "speed_ramp": {
        "description": "Apply speed ramping effects to video",
        "params": ["filepath", "segments", "curve"],
        "module": "opencut.core.speed_ramp",
        "function": "apply_speed_ramp",
    },
    "noise_reduction": {
        "description": "Reduce background noise from audio",
        "params": ["filepath", "strength", "method"],
        "module": "opencut.core.audio_enhance",
        "function": "reduce_noise",
    },
    "auto_zoom": {
        "description": "Apply automatic Ken Burns / zoom effects",
        "params": ["filepath", "style", "intensity"],
        "module": "opencut.core.auto_zoom",
        "function": "apply_auto_zoom",
    },
    "filler_removal": {
        "description": "Remove filler words (um, uh, like) from speech",
        "params": ["filepath", "words", "padding"],
        "module": "opencut.core.fillers",
        "function": "remove_fillers",
    },
}

# In-memory plan store
_plans: Dict[str, AgentPlan] = {}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _get_llm_response(prompt, system_prompt="", on_progress=None):
    """Query the LLM and return response text."""
    from opencut.core.llm import query_llm

    response = query_llm(prompt, system_prompt=system_prompt, on_progress=on_progress)
    return response.text


def _build_plan_prompt(goal_text, available_files, file_info):
    """Build the LLM prompt for plan generation."""
    tool_descriptions = []
    for name, tool in AGENT_TOOLS.items():
        params_str = ", ".join(tool["params"])
        tool_descriptions.append(f"  - {name}: {tool['description']} (params: {params_str})")

    tools_list = "\n".join(tool_descriptions)

    files_desc = []
    for fp in available_files:
        info = file_info.get(fp, {})
        dur = info.get("duration", "unknown")
        res = f"{info.get('width', '?')}x{info.get('height', '?')}"
        files_desc.append(f"  - {fp} (duration: {dur}s, resolution: {res})")

    files_list = "\n".join(files_desc) if files_desc else "  (no files provided)"

    return f"""You are a video editing assistant. Create an ordered step-by-step plan to achieve the user's goal.

Available tools:
{tools_list}

Available files:
{files_list}

User goal: {goal_text}

Respond with a JSON array of steps. Each step must have:
- "action": tool name from the list above
- "description": what this step does
- "params": object with parameter values

Rules:
- Each step's filepath should reference the output of previous steps where appropriate (use "{{prev_output}}" as placeholder)
- First step should use the actual input file path
- Order steps logically (e.g., noise reduction before caption generation)
- Only use tools from the available list
- Keep plans concise — no more than 10 steps

Respond ONLY with valid JSON array, no explanation."""


def _parse_plan_response(llm_text):
    """Parse LLM response into list of AgentStep objects."""
    # Try to extract JSON from the response
    text = llm_text.strip()

    # Strip markdown code blocks if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        )

    try:
        steps_data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON array in the text
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            try:
                steps_data = json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                raise ValueError(f"Could not parse LLM plan response as JSON: {text[:200]}")
        else:
            raise ValueError(f"No JSON array found in LLM response: {text[:200]}")

    if not isinstance(steps_data, list):
        raise ValueError("LLM response is not a JSON array")

    steps = []
    for item in steps_data:
        if not isinstance(item, dict):
            continue
        action = item.get("action", "").strip()
        if action not in AGENT_TOOLS:
            logger.warning("Unknown agent tool '%s', skipping step", action)
            continue
        steps.append(AgentStep(
            action=action,
            endpoint=AGENT_TOOLS[action].get("module", ""),
            params=item.get("params", {}),
        ))

    return steps


def _resolve_step_params(step, previous_output, input_files):
    """Replace placeholders in step params with actual values."""
    params = dict(step.params)

    for key, value in params.items():
        if isinstance(value, str):
            if "{prev_output}" in value and previous_output:
                params[key] = value.replace("{prev_output}", previous_output)
            elif key == "filepath" and not value and input_files:
                params[key] = input_files[0]

    # Ensure filepath is set
    if "filepath" not in params or not params.get("filepath"):
        if previous_output:
            params["filepath"] = previous_output
        elif input_files:
            params["filepath"] = input_files[0]

    step.params = params
    return step


def _execute_tool(tool_name, params):
    """Execute a tool by importing its module and calling the function."""
    if tool_name not in AGENT_TOOLS:
        raise ValueError(f"Unknown tool: {tool_name}")

    tool_info = AGENT_TOOLS[tool_name]
    module_name = tool_info["module"]
    function_name = tool_info["function"]

    try:
        import importlib
        module = importlib.import_module(module_name)
        func = getattr(module, function_name)
    except (ImportError, AttributeError) as e:
        raise RuntimeError(f"Cannot load tool {tool_name}: {e}")

    # Filter params to only those the function accepts
    import inspect
    sig = inspect.signature(func)
    valid_params = {}
    for k, v in params.items():
        if k in sig.parameters:
            valid_params[k] = v

    # Map common parameter names
    if "filepath" in params and "input_path" in sig.parameters and "input_path" not in valid_params:
        valid_params["input_path"] = params["filepath"]

    result = func(**valid_params)

    # Normalize result to dict
    if hasattr(result, "__dataclass_fields__"):
        from dataclasses import asdict
        return asdict(result)
    elif isinstance(result, dict):
        return result
    else:
        return {"result": str(result)}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def create_plan(
    goal_text: str,
    available_files: Optional[List[str]] = None,
    on_progress: Optional[Callable] = None,
) -> AgentPlan:
    """
    Create an editing plan from natural language goal.

    Uses LLM to analyze the goal and available files, then generates
    an ordered list of editing steps.

    Args:
        goal_text: Natural language description of desired edits.
        available_files: List of input file paths.
        on_progress: Progress callback(pct, msg).

    Returns:
        AgentPlan with generated steps.
    """
    if not goal_text or not goal_text.strip():
        raise ValueError("Goal text cannot be empty")

    available_files = available_files or []

    if on_progress:
        on_progress(5, "Analyzing goal and files")

    # Gather file info
    file_info = {}
    for fp in available_files:
        try:
            file_info[fp] = get_video_info(fp)
        except Exception:
            file_info[fp] = {}

    # Build prompt and query LLM
    prompt = _build_plan_prompt(goal_text, available_files, file_info)
    system_prompt = (
        "You are an expert video editor. Generate precise, efficient editing "
        "plans as JSON arrays. Be concise and practical."
    )

    if on_progress:
        on_progress(20, "Querying LLM for plan generation")

    llm_text = _get_llm_response(prompt, system_prompt, on_progress)

    if on_progress:
        on_progress(60, "Parsing plan from LLM response")

    steps = _parse_plan_response(llm_text)

    if not steps:
        raise ValueError("LLM did not generate any valid steps")

    plan = AgentPlan(
        plan_id=str(uuid.uuid4()),
        goal=AgentGoal(description=goal_text),
        steps=steps,
        status="created",
        created_at=time.time(),
        input_files=available_files,
        execution_log=[f"Plan created with {len(steps)} steps"],
    )

    # Store plan
    _plans[plan.plan_id] = plan

    if on_progress:
        on_progress(100, f"Plan created: {len(steps)} steps")

    return plan


def execute_step(step: AgentStep) -> AgentStep:
    """
    Execute a single agent step.

    Args:
        step: AgentStep to execute.

    Returns:
        Updated AgentStep with result or error.
    """
    step.status = "running"
    # Use perf_counter for elapsed timing — time.time() resolution on
    # Windows is ~15.6 ms, so fast/mocked tools record duration 0.0.
    start_time = time.perf_counter()

    try:
        result = _execute_tool(step.action, step.params)
        step.status = "complete"
        step.result = result
    except Exception as e:
        step.status = "error"
        step.error = str(e)
        logger.warning("Step '%s' failed: %s", step.action, e)

    step.duration_seconds = max(time.perf_counter() - start_time, 1e-6)
    return step


def validate_step_result(step: AgentStep) -> bool:
    """
    Validate that a step's result is valid and usable.

    Args:
        step: Completed AgentStep.

    Returns:
        True if result is valid.
    """
    if step.status != "complete":
        return False

    if step.result is None:
        return False

    # Check for output file existence if result has output_path
    out = step.result.get("output_path")
    if out and isinstance(out, str):
        return os.path.isfile(out)

    # If no output_path, check the result isn't an error
    if step.result.get("error"):
        return False

    return True


def recover_from_failure(
    plan: AgentPlan,
    failed_step: AgentStep,
    on_progress: Optional[Callable] = None,
) -> AgentStep:
    """
    Use LLM to generate an alternative approach for a failed step.

    Args:
        plan: Current execution plan.
        failed_step: The step that failed.
        on_progress: Progress callback(pct, msg).

    Returns:
        New AgentStep as alternative, or the original step marked skipped.
    """
    tool_names = ", ".join(AGENT_TOOLS.keys())

    prompt = f"""A video editing step failed. Suggest an alternative approach.

Failed step: {failed_step.action}
Error: {failed_step.error}
Parameters: {json.dumps(failed_step.params)}

Available tools: {tool_names}

Goal: {plan.goal.description}

Respond with a single JSON object with keys: "action", "params", "description".
If no alternative is possible, respond with {{"action": "skip", "description": "No alternative available"}}."""

    try:
        if on_progress:
            on_progress(50, "Consulting LLM for recovery strategy")

        llm_text = _get_llm_response(prompt, on_progress=on_progress)
        text = llm_text.strip()

        # Strip markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            )

        # Find JSON object
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            data = json.loads(text[start:end + 1])
        else:
            data = {"action": "skip"}

        action = data.get("action", "skip")
        if action == "skip" or action not in AGENT_TOOLS:
            failed_step.status = "skipped"
            plan.execution_log.append(
                f"Step '{failed_step.action}' skipped: no alternative found"
            )
            return failed_step

        alternative = AgentStep(
            action=action,
            endpoint=AGENT_TOOLS[action]["module"],
            params=data.get("params", failed_step.params),
        )

        plan.execution_log.append(
            f"Recovery: replacing '{failed_step.action}' with '{action}'"
        )
        return alternative

    except Exception as e:
        logger.warning("Recovery failed: %s", e)
        failed_step.status = "skipped"
        plan.execution_log.append(
            f"Step '{failed_step.action}' skipped after recovery failure: {e}"
        )
        return failed_step


def execute_plan(
    plan: AgentPlan,
    max_retries: int = 1,
    on_progress: Optional[Callable] = None,
) -> AgentResult:
    """
    Execute all steps in a plan sequentially.

    Each step's output is fed as input to the next step.
    Failed steps trigger LLM-based recovery.

    Args:
        plan: AgentPlan to execute.
        max_retries: Max retry attempts per step.
        on_progress: Progress callback(pct, msg).

    Returns:
        AgentResult with execution summary.
    """
    plan.status = "running"
    start_time = time.perf_counter()
    output_paths = []
    previous_output = None
    steps_completed = 0
    steps_failed = 0

    total_steps = len(plan.steps)

    if on_progress:
        on_progress(1, f"Executing plan: {total_steps} steps")

    for idx, step in enumerate(plan.steps):
        plan.current_step_idx = idx
        step_pct_base = int(5 + 90 * idx / total_steps)

        if on_progress:
            on_progress(step_pct_base, f"Step {idx + 1}/{total_steps}: {step.action}")

        # Resolve parameter placeholders
        _resolve_step_params(step, previous_output, plan.input_files)

        plan.execution_log.append(
            f"Executing step {idx + 1}: {step.action} with {list(step.params.keys())}"
        )

        # Execute with retry logic
        executed = execute_step(step)
        attempts = 0

        while executed.status == "error" and attempts < max_retries:
            attempts += 1
            plan.execution_log.append(
                f"Step {idx + 1} failed (attempt {attempts}): {executed.error}"
            )

            recovered = recover_from_failure(plan, executed, on_progress)
            if recovered.status == "skipped":
                executed = recovered
                break

            if recovered.action != executed.action:
                # Try alternative step
                _resolve_step_params(recovered, previous_output, plan.input_files)
                executed = execute_step(recovered)
                plan.steps[idx] = executed
            else:
                break

        # Track results
        if executed.status == "complete":
            steps_completed += 1
            out = executed.result.get("output_path") if executed.result else None
            if out:
                output_paths.append(out)
                previous_output = out
            plan.execution_log.append(f"Step {idx + 1} complete: {executed.action}")
        elif executed.status == "error":
            steps_failed += 1
            plan.execution_log.append(
                f"Step {idx + 1} failed permanently: {executed.error}"
            )
        elif executed.status == "skipped":
            plan.execution_log.append(f"Step {idx + 1} skipped: {executed.action}")

    total_duration = max(time.perf_counter() - start_time, 1e-6)
    plan.status = "complete" if steps_failed == 0 else "error"

    if on_progress:
        on_progress(100, "Plan execution complete")

    plan.execution_log.append(
        f"Finished: {steps_completed} complete, {steps_failed} failed, "
        f"{total_duration:.1f}s total"
    )

    result = AgentResult(
        plan=plan,
        output_paths=output_paths,
        summary=f"Executed {steps_completed}/{total_steps} steps successfully",
        total_duration=total_duration,
        steps_completed=steps_completed,
        steps_failed=steps_failed,
    )

    # Update stored plan
    _plans[plan.plan_id] = plan

    return result


def agent_edit(
    goal_text: str,
    file_paths: Optional[List[str]] = None,
    out_path: Optional[str] = None,
    on_progress: Optional[Callable] = None,
) -> AgentResult:
    """
    Full autonomous editing pipeline: plan + execute from natural language.

    Args:
        goal_text: Natural language description of desired edits.
        file_paths: List of input file paths.
        out_path: Desired output path (optional).
        on_progress: Progress callback(pct, msg).

    Returns:
        AgentResult with all outputs.
    """
    if not goal_text or not goal_text.strip():
        raise ValueError("Goal text cannot be empty")

    file_paths = file_paths or []

    if on_progress:
        on_progress(1, "Starting autonomous edit")

    # Create plan
    plan = create_plan(goal_text, file_paths, on_progress=on_progress)

    # If output path requested, inject into last step
    if out_path and plan.steps:
        plan.steps[-1].params["output_path"] = out_path

    # Execute plan
    result = execute_plan(plan, on_progress=on_progress)

    return result


def get_plan(plan_id: str) -> Optional[AgentPlan]:
    """Retrieve a stored plan by ID."""
    return _plans.get(plan_id)


def list_tools() -> Dict:
    """Return available agent tools with descriptions."""
    return {
        name: {
            "description": tool["description"],
            "params": tool["params"],
        }
        for name, tool in AGENT_TOOLS.items()
    }
