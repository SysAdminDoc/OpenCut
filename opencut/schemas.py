"""Typed response schemas for OpenCut API routes.

Prevents field-name mismatch bugs between routes, frontends (CEP/UXP),
and core modules by providing a single source of truth for result shapes.

Usage in routes::

    from opencut.schemas import JobResponse, ColorMatchResult, JobResult

    return jsonify(JobResponse(job_id=job_id).to_dict())

    _update_job(job_id, status="complete", progress=100,
                result=ColorMatchResult(output=out_path).to_dict())
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_none(d: dict) -> dict:
    """Remove keys with None values from a dict."""
    return {k: v for k, v in d.items() if v is not None}


# ---------------------------------------------------------------------------
# Job lifecycle
# ---------------------------------------------------------------------------

@dataclass
class JobResponse:
    """Returned immediately when an async route starts a job."""
    job_id: str
    status: str = "running"

    def to_dict(self) -> dict:
        return _strip_none(asdict(self))


@dataclass
class JobResult:
    """Wrapper written into _update_job() on completion or error."""
    status: str = "complete"
    progress: int = 100
    message: str = "Done"
    result: Optional[dict] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return _strip_none(asdict(self))


# ---------------------------------------------------------------------------
# Common API envelopes
# ---------------------------------------------------------------------------

@dataclass
class HealthResult:
    """Result from /health."""
    status: str = "ok"
    version: str = ""
    csrf_token: str = ""
    capabilities: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return _strip_none(asdict(self))


@dataclass
class ActionResult:
    """Generic acknowledgement for routes that perform one action."""
    success: bool = True
    ok: bool = True
    message: str = ""
    output: Optional[str] = None
    output_path: Optional[str] = None
    job_id: Optional[str] = None

    def to_dict(self) -> dict:
        return _strip_none(asdict(self))


@dataclass
class SettingsResult:
    """Generic settings payload for GET/POST settings routes."""
    settings: Dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    message: str = ""

    def to_dict(self) -> dict:
        return _strip_none(asdict(self))


@dataclass
class ListEnvelopeResult:
    """Generic list payload used by catalogue routes."""
    items: List[dict] = field(default_factory=list)
    total: int = 0

    def to_dict(self) -> dict:
        return _strip_none(asdict(self))


@dataclass
class CapabilityResult:
    """Availability / capability payload for backend-info routes."""
    available: bool = False
    status: str = ""
    reason: str = ""
    capabilities: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return _strip_none(asdict(self))


@dataclass
class FileOutputResult:
    """Payload for routes that write one or more files."""
    output: str = ""
    output_path: str = ""
    outputs: List[str] = field(default_factory=list)
    count: int = 0

    def to_dict(self) -> dict:
        return _strip_none(asdict(self))


@dataclass
class JobStatusResult:
    """Result from job status and diagnostics routes."""
    job_id: str = ""
    status: str = ""
    progress: int = 0
    message: str = ""
    result: Optional[dict] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return _strip_none(asdict(self))


@dataclass
class JobListResult:
    """Result from job listing routes."""
    jobs: List[dict] = field(default_factory=list)
    total: int = 0

    def to_dict(self) -> dict:
        return _strip_none(asdict(self))


@dataclass
class JobStatsResult:
    """Result from /jobs/stats."""
    total: int = 0
    running: int = 0
    complete: int = 0
    cancelled: int = 0
    error: int = 0

    def to_dict(self) -> dict:
        return _strip_none(asdict(self))


@dataclass
class ModelListResult:
    """Result from model catalogue / installed-model routes."""
    models: List[dict] = field(default_factory=list)
    installed: List[dict] = field(default_factory=list)
    available: List[dict] = field(default_factory=list)
    total: int = 0

    def to_dict(self) -> dict:
        return _strip_none(asdict(self))


@dataclass
class ModelProgressResult:
    """Result from model-download progress routes."""
    downloads: List[dict] = field(default_factory=list)
    active: int = 0
    completed: int = 0

    def to_dict(self) -> dict:
        return _strip_none(asdict(self))


@dataclass
class GpuStatusResult:
    """Result from GPU status routes."""
    available: bool = False
    device: str = ""
    devices: List[dict] = field(default_factory=list)
    models: List[dict] = field(default_factory=list)
    vram_total_mb: int = 0
    vram_free_mb: int = 0

    def to_dict(self) -> dict:
        return _strip_none(asdict(self))


@dataclass
class ToolListResult:
    """Result from tool-catalogue routes."""
    tools: List[dict] = field(default_factory=list)
    count: int = 0

    def to_dict(self) -> dict:
        return _strip_none(asdict(self))


@dataclass
class CaptionPreviewResult:
    """Result from caption/style preview routes."""
    css: str = ""
    ass: str = ""
    preview: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return _strip_none(asdict(self))


@dataclass
class CaptionProfilesResult:
    """Result from caption profile / token catalogue routes."""
    profiles: List[dict] = field(default_factory=list)
    tokens: Dict[str, Any] = field(default_factory=dict)
    defaults: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return _strip_none(asdict(self))


@dataclass
class CaptionTranslationResult:
    """Result from /captions/translate."""
    segments: List[dict] = field(default_factory=list)
    srt: str = ""
    output_path: Optional[str] = None
    language: str = ""

    def to_dict(self) -> dict:
        return _strip_none(asdict(self))


@dataclass
class CaptionQcResult:
    """Result from caption quality-control routes."""
    issues: List[dict] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    passed: bool = False

    def to_dict(self) -> dict:
        return _strip_none(asdict(self))


@dataclass
class VoiceListResult:
    """Result from TTS voice listing routes."""
    voices: List[dict] = field(default_factory=list)
    backends: Dict[str, Any] = field(default_factory=dict)
    total: int = 0

    def to_dict(self) -> dict:
        return _strip_none(asdict(self))


@dataclass
class AnalyticsResult:
    """Result from analytics summary/report routes."""
    events: List[dict] = field(default_factory=list)
    report: Dict[str, Any] = field(default_factory=dict)
    usage: Dict[str, Any] = field(default_factory=dict)
    total: int = 0

    def to_dict(self) -> dict:
        return _strip_none(asdict(self))


# ---------------------------------------------------------------------------
# Audio results
# ---------------------------------------------------------------------------

@dataclass
class SilenceResult:
    """Result from /silence (silence detection / removal)."""
    cuts: List[dict] = field(default_factory=list)
    total_removed_seconds: float = 0.0
    output_path: Optional[str] = None

    def to_dict(self) -> dict:
        return _strip_none(asdict(self))


@dataclass
class LoudnessMatchResult:
    """Result from /audio/loudness-match."""
    outputs: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return _strip_none(asdict(self))


@dataclass
class BeatMarkersResult:
    """Result from /audio/beat-markers."""
    beats: List[float] = field(default_factory=list)
    bpm: float = 0.0
    total_beats: int = 0

    def to_dict(self) -> dict:
        return _strip_none(asdict(self))


# ---------------------------------------------------------------------------
# Video results
# ---------------------------------------------------------------------------

@dataclass
class ColorMatchResult:
    """Result from /video/color-match."""
    output: str = ""

    def to_dict(self) -> dict:
        return _strip_none(asdict(self))


@dataclass
class AutoZoomResult:
    """Result from /video/auto-zoom."""
    keyframes: List[dict] = field(default_factory=list)
    output: Optional[str] = None
    fps: float = 0.0
    duration: float = 0.0

    def to_dict(self) -> dict:
        return _strip_none(asdict(self))


@dataclass
class MulticamResult:
    """Result from /video/multicam-cuts."""
    cuts: List[dict] = field(default_factory=list)
    total_cuts: int = 0
    speakers: List[str] = field(default_factory=list)
    speaker_map: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return _strip_none(asdict(self))


# ---------------------------------------------------------------------------
# Caption / transcript results
# ---------------------------------------------------------------------------

@dataclass
class ChaptersResult:
    """Result from /captions/chapters."""
    chapters: List[dict] = field(default_factory=list)
    description_block: str = ""
    source: str = "heuristic"

    def to_dict(self) -> dict:
        return _strip_none(asdict(self))


@dataclass
class RepeatDetectResult:
    """Result from /captions/repeat-detect."""
    repeats: List[dict] = field(default_factory=list)
    clean_ranges: List[dict] = field(default_factory=list)
    total_removed_seconds: float = 0.0

    def to_dict(self) -> dict:
        return _strip_none(asdict(self))


# ---------------------------------------------------------------------------
# Search / index results
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """Result from /search/footage."""
    results: List[dict] = field(default_factory=list)
    query: str = ""
    total_matches: int = 0

    def to_dict(self) -> dict:
        return _strip_none(asdict(self))


@dataclass
class IndexResult:
    """Result from /search/index."""
    indexed: int = 0
    total: int = 0
    errors: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return _strip_none(asdict(self))


# ---------------------------------------------------------------------------
# Deliverables results
# ---------------------------------------------------------------------------

@dataclass
class DeliverableResult:
    """Result from /deliverables/* (vfx-sheet, adr-list, etc.)."""
    output: str = ""
    rows: int = 0

    def to_dict(self) -> dict:
        return _strip_none(asdict(self))


# ---------------------------------------------------------------------------
# Timeline results
# ---------------------------------------------------------------------------

@dataclass
class ExportMarkersResult:
    """Result from /timeline/export-from-markers."""
    outputs: List[str] = field(default_factory=list)
    count: int = 0

    def to_dict(self) -> dict:
        return _strip_none(asdict(self))


# ---------------------------------------------------------------------------
# Workflow results
# ---------------------------------------------------------------------------

@dataclass
class WorkflowResult:
    """Result from /workflow/run."""
    workflow: str = ""
    steps_completed: int = 0
    total_steps: int = 0
    output_path: Optional[str] = None
    step_results: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return _strip_none(asdict(self))


# ---------------------------------------------------------------------------
# Context awareness results
# ---------------------------------------------------------------------------

@dataclass
class ContextAnalysisResult:
    """Result from /context/analyze."""
    tags: List[str] = field(default_factory=list)
    features: List[dict] = field(default_factory=list)
    guidance: str = ""
    tab_scores: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return _strip_none(asdict(self))


# ---------------------------------------------------------------------------
# Video AI results
# ---------------------------------------------------------------------------

@dataclass
class VideoAIResult:
    """Generic result for /video/ai/* routes (upscale, rembg, denoise, interpolate)."""
    output_path: str = ""

    def to_dict(self) -> dict:
        return _strip_none(asdict(self))


@dataclass
class ShortsPipelineResult:
    """Result from /video/shorts-pipeline."""
    clips: List[dict] = field(default_factory=list)
    total_clips: int = 0

    def to_dict(self) -> dict:
        return _strip_none(asdict(self))


@dataclass
class DepthMapResult:
    """Result from /video/depth/* routes."""
    output_path: str = ""

    def to_dict(self) -> dict:
        return _strip_none(asdict(self))


@dataclass
class BrollPlanResult:
    """Result from /video/broll-plan."""
    windows: List[dict] = field(default_factory=list)
    total_windows: int = 0
    total_broll_time: float = 0.0
    keywords_used: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return _strip_none(asdict(self))


# ---------------------------------------------------------------------------
# Batch processing results
# ---------------------------------------------------------------------------

@dataclass
class BatchResult:
    """Result from batch processing routes."""
    batch_id: str = ""
    total: int = 0
    completed: int = 0
    failed: int = 0
    outputs: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return _strip_none(asdict(self))


# ---------------------------------------------------------------------------
# Plugin results
# ---------------------------------------------------------------------------

@dataclass
class PluginListResult:
    """Result from /plugins/list."""
    plugins: List[dict] = field(default_factory=list)
    plugins_dir: str = ""
    total: int = 0
    loaded: int = 0

    def to_dict(self) -> dict:
        return _strip_none(asdict(self))


# ---------------------------------------------------------------------------
# System results
# ---------------------------------------------------------------------------

@dataclass
class UpdateCheckResult:
    """Result from update check."""
    current_version: str = ""
    latest_version: str = ""
    update_available: bool = False
    release_url: str = ""

    def to_dict(self) -> dict:
        return _strip_none(asdict(self))
