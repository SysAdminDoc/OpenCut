"""
OpenCut MCP Server Expansion (Feature 7.7)

Maps major API routes to Model Context Protocol (MCP) tool definitions,
provides compound tools for common multi-step workflows, and exposes
resource endpoints for listing available operations.
"""

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MCPToolParameter:
    """A single parameter in an MCP tool definition."""
    name: str
    type: str = "string"
    description: str = ""
    required: bool = False
    default: Any = None

    def to_dict(self) -> dict:
        d = {"name": self.name, "type": self.type, "description": self.description,
             "required": self.required}
        if self.default is not None:
            d["default"] = self.default
        return d


@dataclass
class MCPTool:
    """Definition of a single MCP tool."""
    name: str
    description: str = ""
    category: str = "general"
    parameters: List[MCPToolParameter] = field(default_factory=list)
    endpoint: str = ""
    method: str = "POST"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "parameters": [p.to_dict() for p in self.parameters],
            "endpoint": self.endpoint,
            "method": self.method,
        }


@dataclass
class CompoundTool:
    """A compound tool that chains multiple MCP tools."""
    name: str
    description: str = ""
    steps: List[Dict] = field(default_factory=list)
    parameters: List[MCPToolParameter] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "steps": self.steps,
            "parameters": [p.to_dict() for p in self.parameters],
        }


@dataclass
class CompoundToolResult:
    """Result of executing a compound tool."""
    success: bool = False
    tool_name: str = ""
    steps_completed: int = 0
    total_steps: int = 0
    results: List[Dict] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Core MCP tool definitions
# ---------------------------------------------------------------------------
_MCP_TOOLS: List[MCPTool] = [
    # Audio tools
    MCPTool(
        name="remove_silence",
        description="Detect and remove silent segments from audio/video.",
        category="audio",
        parameters=[
            MCPToolParameter("file_path", "string", "Path to input file", required=True),
            MCPToolParameter("threshold_db", "number", "Silence threshold in dB", default=-35),
            MCPToolParameter("min_duration", "number", "Minimum silence duration (seconds)", default=0.5),
        ],
        endpoint="/api/audio/silence",
        method="POST",
    ),
    MCPTool(
        name="enhance_audio",
        description="Enhance audio quality with noise reduction, EQ, and normalization.",
        category="audio",
        parameters=[
            MCPToolParameter("file_path", "string", "Path to input file", required=True),
            MCPToolParameter("noise_reduce", "boolean", "Apply noise reduction", default=True),
            MCPToolParameter("normalize", "boolean", "Normalize loudness", default=True),
        ],
        endpoint="/api/audio/enhance",
        method="POST",
    ),
    MCPTool(
        name="generate_captions",
        description="Generate captions/subtitles using Whisper transcription.",
        category="captions",
        parameters=[
            MCPToolParameter("file_path", "string", "Path to input media file", required=True),
            MCPToolParameter("model", "string", "Whisper model size", default="base"),
            MCPToolParameter("language", "string", "Language code", default="en"),
        ],
        endpoint="/api/captions/generate",
        method="POST",
    ),
    MCPTool(
        name="burn_captions",
        description="Burn captions/subtitles into video as hardcoded text.",
        category="captions",
        parameters=[
            MCPToolParameter("video_path", "string", "Path to video file", required=True),
            MCPToolParameter("srt_path", "string", "Path to SRT subtitle file", required=True),
            MCPToolParameter("style", "string", "Caption style preset", default="default"),
        ],
        endpoint="/api/captions/burn",
        method="POST",
    ),
    # Video editing tools
    MCPTool(
        name="trim_video",
        description="Trim video to specified start and end times.",
        category="video",
        parameters=[
            MCPToolParameter("file_path", "string", "Path to video file", required=True),
            MCPToolParameter("start", "number", "Start time in seconds", required=True),
            MCPToolParameter("end", "number", "End time in seconds", required=True),
        ],
        endpoint="/api/video/trim",
        method="POST",
    ),
    MCPTool(
        name="smart_reframe",
        description="Automatically reframe video for different aspect ratios.",
        category="video",
        parameters=[
            MCPToolParameter("file_path", "string", "Path to video file", required=True),
            MCPToolParameter("target_ratio", "string", "Target aspect ratio (e.g. '9:16')", required=True),
        ],
        endpoint="/api/video/reframe",
        method="POST",
    ),
    MCPTool(
        name="scene_detect",
        description="Detect scene boundaries in a video.",
        category="analysis",
        parameters=[
            MCPToolParameter("file_path", "string", "Path to video file", required=True),
            MCPToolParameter("threshold", "number", "Detection sensitivity", default=0.3),
        ],
        endpoint="/api/video/scenes",
        method="POST",
    ),
    MCPTool(
        name="color_match",
        description="Match color grading between reference and target video.",
        category="color",
        parameters=[
            MCPToolParameter("source_path", "string", "Path to source video", required=True),
            MCPToolParameter("reference_path", "string", "Path to reference video", required=True),
        ],
        endpoint="/api/video/color-match",
        method="POST",
    ),
    MCPTool(
        name="generate_thumbnail",
        description="Generate a thumbnail image from a video.",
        category="export",
        parameters=[
            MCPToolParameter("file_path", "string", "Path to video file", required=True),
            MCPToolParameter("timestamp", "number", "Timestamp in seconds", default=0),
        ],
        endpoint="/api/video/thumbnail",
        method="POST",
    ),
    MCPTool(
        name="export_video",
        description="Export video with specified encoding settings.",
        category="export",
        parameters=[
            MCPToolParameter("file_path", "string", "Path to video file", required=True),
            MCPToolParameter("format", "string", "Output format", default="mp4"),
            MCPToolParameter("quality", "string", "Quality preset", default="high"),
        ],
        endpoint="/api/video/export",
        method="POST",
    ),
]


# ---------------------------------------------------------------------------
# Compound tool definitions
# ---------------------------------------------------------------------------
_COMPOUND_TOOLS: Dict[str, CompoundTool] = {
    "clean_interview": CompoundTool(
        name="clean_interview",
        description="Full interview cleanup: remove silence, enhance audio, generate captions.",
        steps=[
            {"tool": "remove_silence", "map": {"file_path": "input.file_path",
                                                 "threshold_db": "input.silence_threshold"}},
            {"tool": "enhance_audio", "map": {"file_path": "step_0.output_path"}},
            {"tool": "generate_captions", "map": {"file_path": "step_1.output_path",
                                                    "model": "input.whisper_model"}},
        ],
        parameters=[
            MCPToolParameter("file_path", "string", "Path to interview video", required=True),
            MCPToolParameter("silence_threshold", "number", "Silence threshold dB", default=-35),
            MCPToolParameter("whisper_model", "string", "Whisper model size", default="base"),
        ],
    ),
    "prepare_for_youtube": CompoundTool(
        name="prepare_for_youtube",
        description="Prepare video for YouTube: enhance audio, generate captions, generate thumbnail, export.",
        steps=[
            {"tool": "enhance_audio", "map": {"file_path": "input.file_path"}},
            {"tool": "generate_captions", "map": {"file_path": "step_0.output_path"}},
            {"tool": "generate_thumbnail", "map": {"file_path": "step_0.output_path"}},
            {"tool": "export_video", "map": {"file_path": "step_0.output_path",
                                               "format": "input.format",
                                               "quality": "input.quality"}},
        ],
        parameters=[
            MCPToolParameter("file_path", "string", "Path to video file", required=True),
            MCPToolParameter("format", "string", "Output format", default="mp4"),
            MCPToolParameter("quality", "string", "Quality preset", default="high"),
        ],
    ),
    "podcast_polish": CompoundTool(
        name="podcast_polish",
        description="Polish a podcast recording: remove silence, enhance audio, detect scenes, generate captions.",
        steps=[
            {"tool": "remove_silence", "map": {"file_path": "input.file_path",
                                                 "min_duration": "input.min_silence"}},
            {"tool": "enhance_audio", "map": {"file_path": "step_0.output_path",
                                                "normalize": "input.normalize"}},
            {"tool": "scene_detect", "map": {"file_path": "step_1.output_path"}},
            {"tool": "generate_captions", "map": {"file_path": "step_1.output_path",
                                                    "model": "input.whisper_model"}},
        ],
        parameters=[
            MCPToolParameter("file_path", "string", "Path to podcast recording", required=True),
            MCPToolParameter("min_silence", "number", "Minimum silence duration", default=1.0),
            MCPToolParameter("normalize", "boolean", "Normalize loudness", default=True),
            MCPToolParameter("whisper_model", "string", "Whisper model size", default="base"),
        ],
    ),
}


# ---------------------------------------------------------------------------
# Tool resolution helpers
# ---------------------------------------------------------------------------
def _get_tool_by_name(name: str) -> Optional[MCPTool]:
    """Look up an MCP tool by name."""
    for tool in _MCP_TOOLS:
        if tool.name == name:
            return tool
    return None


def _resolve_param(mapping: str, input_params: Dict, step_results: List[Dict]) -> Any:
    """Resolve a parameter mapping like ``'input.file_path'`` or ``'step_0.output_path'``.

    Returns the resolved value, or None if the mapping cannot be resolved.
    """
    parts = mapping.split(".", 1)
    if len(parts) != 2:
        return mapping  # literal value

    source, key = parts
    if source == "input":
        return input_params.get(key)
    elif source.startswith("step_"):
        try:
            idx = int(source.split("_", 1)[1])
            if idx < len(step_results):
                return step_results[idx].get(key)
        except (ValueError, IndexError):
            pass
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_mcp_tools() -> List[Dict]:
    """Return all registered MCP tool definitions.

    Returns:
        List of tool definition dicts in MCP-compatible format.
    """
    return [tool.to_dict() for tool in _MCP_TOOLS]


def get_compound_tools() -> List[Dict]:
    """Return all registered compound tool definitions.

    Returns:
        List of compound tool definition dicts.
    """
    return [ct.to_dict() for ct in _COMPOUND_TOOLS.values()]


def list_available_operations() -> Dict[str, List[str]]:
    """Return a categorized listing of all available operations.

    Returns:
        Dict mapping category names to lists of tool names.
    """
    categories: Dict[str, List[str]] = {}
    for tool in _MCP_TOOLS:
        categories.setdefault(tool.category, []).append(tool.name)
    # Add compound tools under their own category
    categories["compound"] = list(_COMPOUND_TOOLS.keys())
    return categories


def execute_compound_tool(
    tool_name: str,
    params: Dict[str, Any],
    on_progress: Optional[Callable] = None,
) -> CompoundToolResult:
    """Execute a compound tool by running its steps sequentially.

    Each step's output is made available to subsequent steps via parameter
    mapping.  The executor simulates tool execution (actual HTTP calls
    are not made; this is for MCP server orchestration).

    Args:
        tool_name: Name of the compound tool.
        params: Input parameters.
        on_progress: Optional progress callback.

    Returns:
        CompoundToolResult with step-by-step results.
    """
    compound = _COMPOUND_TOOLS.get(tool_name)
    if compound is None:
        return CompoundToolResult(
            success=False,
            tool_name=tool_name,
            error=f"Unknown compound tool: {tool_name!r}. "
                  f"Available: {', '.join(_COMPOUND_TOOLS.keys())}",
        )

    # Validate required parameters
    for p in compound.parameters:
        if p.required and p.name not in params:
            return CompoundToolResult(
                success=False,
                tool_name=tool_name,
                error=f"Missing required parameter: {p.name}",
            )

    # Fill in defaults
    full_params = {}
    for p in compound.parameters:
        if p.name in params:
            full_params[p.name] = params[p.name]
        elif p.default is not None:
            full_params[p.name] = p.default

    start_time = time.time()
    step_results: List[Dict] = []
    total = len(compound.steps)

    for i, step in enumerate(compound.steps):
        tool_ref = _get_tool_by_name(step["tool"])
        if tool_ref is None:
            return CompoundToolResult(
                success=False,
                tool_name=tool_name,
                steps_completed=i,
                total_steps=total,
                results=step_results,
                error=f"Step {i}: unknown tool {step['tool']!r}",
            )

        if on_progress:
            on_progress({
                "step": i + 1,
                "total": total,
                "tool": step["tool"],
                "status": "running",
            })

        # Resolve parameter mappings
        resolved_params = {}
        for param_name, mapping in step.get("map", {}).items():
            val = _resolve_param(mapping, full_params, step_results)
            if val is not None:
                resolved_params[param_name] = val

        # Record the resolved step invocation
        step_result = {
            "step": i,
            "tool": step["tool"],
            "endpoint": tool_ref.endpoint,
            "method": tool_ref.method,
            "params": resolved_params,
            "status": "completed",
            # Simulated output path for chaining
            "output_path": resolved_params.get("file_path", full_params.get("file_path", "")),
        }
        step_results.append(step_result)

    elapsed = time.time() - start_time

    if on_progress:
        on_progress({"step": "done", "elapsed": elapsed})

    return CompoundToolResult(
        success=True,
        tool_name=tool_name,
        steps_completed=total,
        total_steps=total,
        results=step_results,
        elapsed_seconds=round(elapsed, 3),
    )
