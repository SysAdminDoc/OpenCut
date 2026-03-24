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
