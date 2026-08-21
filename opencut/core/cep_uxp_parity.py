"""CEP host-function to UXP migration catalogue.

F198 tracks the small surface that still depends on the CEP ExtendScript host
bridge. Keep this catalogue in code so the prose migration matrix cannot drift
away from ``extension/com.opencut.panel/host/index.jsx``.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

CATALOGUE_VERSION = 1
SOURCE_HOST_FILE = "extension/com.opencut.panel/host/index.jsx"
UXP_TYPINGS = "@adobe/premierepro@26.3.0-beta.67"
ROUTE_CATALOGUE_VERSION = 1
ROUTE_CEP_SOURCE = "extension/com.opencut.panel/client/main.js"
ROUTE_UXP_SOURCES = (
    "extension/com.opencut.uxp/main.js",
    "extension/com.opencut.uxp/uxp-utils.js",
)

# These are the user-facing capability routes that must be present in UXP
# before the CEP sunset. Keep the list small and explicit so the dashboard
# cannot call a transport or an incidental backend string a migrated feature.
PRIORITY_ROUTE_COVERAGE = (
    "/install-whisper",
    "/audio/separate",
    "/captions/translate",
    "/audio/enhance",
    "/captions/animated/render",
    "/export/preset",
    "/full",
)

# ``/jobs`` is the CEP panel's polling transport, not a capability that needs
# a feature-level UXP button. It is retained as a visible, justified
# exclusion instead of being silently dropped from the inventory.
ROUTE_EXCLUSIONS: Mapping[str, str] = {
    "/jobs": "CEP job-status polling transport; UXP uses JobPoller internally.",
}

# Every CEP route that has no UXP equivalent yet. These are not exclusions:
# each one is a feature the UXP panel is expected to gain before the CEP
# sunset, and calling them "excluded" would quietly retire work nobody
# decided to drop. The gate below fails on a CEP route that appears in
# neither map, so a new route cannot ship unclassified, and the floor stops
# this list growing without someone raising it on purpose. Porting a route
# to UXP lowers the count; re-record the floor in the same change.
UXP_PENDING_REASONS: Mapping[str, str] = {
    "analysis": (
        "Virality, assistant, and context analysis surfaces. Read-only "
        "scoring the UXP panel has no view for yet."
    ),
    "audio": (
        "Audio generation, repair, and effects. Needs the UXP audio surface, "
        "which is behind the cut and caption work in the migration order."
    ),
    "automation": (
        "Workflow, template, preset, and batch surfaces. These compose other "
        "routes, so they port after the routes they call."
    ),
    "delivery": (
        "Social platform metadata, export presets, and thumbnails. Blocked "
        "with the rest of the export surface on UXP encoder work."
    ),
    "plumbing": (
        "Settings, diagnostics, job history, and process control. Mostly "
        "served in UXP by panel-local state or the typed host APIs rather "
        "than by porting the CEP endpoint one for one; each still needs a "
        "decision, so they are tracked rather than excluded."
    ),
    "timeline": (
        "Timeline organisation helpers (renaming, smart bins, beat cuts). "
        "Depend on host write paths still being validated on 26.3."
    ),
    "transcript": (
        "Caption styling, burn-in segments, and alternate ASR backends. The "
        "core transcribe path is already covered; these are the extras."
    ),
    "video-effects": (
        "Compositing, colour, titles, transitions, and speed. The largest "
        "single block and the least started on the UXP side."
    ),
    "video-restore": (
        "Face, upscale, watermark, and interpolation models. Long-running "
        "model jobs that need the UXP durable-job surface first."
    ),
}

# route -> family key in UXP_PENDING_REASONS
ROUTE_UXP_PENDING: Mapping[str, str] = {
    # analysis
    "/analyze/virality": "analysis",
    "/analyze/virality/rank": "analysis",
    "/assistant/dismiss": "analysis",
    "/assistant/suggest": "analysis",
    "/context/analyze": "analysis",
    "/interview-polish": "analysis",
    "/interview-polish/state": "analysis",
    # audio
    "/audio/beats": "audio",
    "/audio/duck-video": "audio",
    "/audio/effects/apply": "audio",
    "/audio/gen/sfx": "audio",
    "/audio/gen/tone": "audio",
    "/audio/music-ai/generate": "audio",
    "/audio/pro/apply": "audio",
    "/audio/pro/deepfilter": "audio",
    "/audio/pro/effects": "audio",
    "/audio/pro/install": "audio",
    "/audio/tts/generate": "audio",
    "/audio/tts/install": "audio",
    # automation
    "/batch": "automation",
    "/batch/create": "automation",
    "/deliverables": "automation",
    "/favorites": "automation",
    "/favorites/save": "automation",
    "/presets": "automation",
    "/presets/save": "automation",
    "/templates/apply": "automation",
    "/templates/list": "automation",
    "/templates/save": "automation",
    "/workflow/approve": "automation",
    "/workflow/compile": "automation",
    "/workflow/presets": "automation",
    "/workflow/run": "automation",
    "/workflow/save": "automation",
    "/workflows": "automation",
    "/workflows/list": "automation",
    # delivery
    "/export/presets": "delivery",
    "/export/thumbnails": "delivery",
    "/social/platforms": "delivery",
    # plumbing
    "/cancel": "plumbing",
    "/info": "plumbing",
    "/jobs/history": "plumbing",
    "/journal/clear": "plumbing",
    "/journal/list": "plumbing",
    "/llm/test": "plumbing",
    "/models/list": "plumbing",
    "/outputs/recent": "plumbing",
    "/preflight": "plumbing",
    "/queue/list": "plumbing",
    "/queue/replay": "plumbing",
    "/settings/auto-zoom": "plumbing",
    "/settings/export": "plumbing",
    "/settings/gist/pull": "plumbing",
    "/settings/gist/push": "plumbing",
    "/settings/import": "plumbing",
    "/settings/loudness-target": "plumbing",
    "/settings/onboarding": "plumbing",
    "/shutdown": "plumbing",
    "/status": "plumbing",
    "/system/changelog/mark-seen": "plumbing",
    "/system/demo/sample": "plumbing",
    "/system/dependencies": "plumbing",
    "/system/estimate-time": "plumbing",
    "/system/gpu-recommend": "plumbing",
    "/system/issue-report/bundle": "plumbing",
    "/system/open-path": "plumbing",
    "/system/qe-reflect": "plumbing",
    "/system/status": "plumbing",
    # timeline
    "/timeline/batch-rename": "timeline",
    "/timeline/beat-cut": "timeline",
    "/timeline/smart-bins": "timeline",
    # transcript
    "/caption-styles": "transcript",
    "/captions/burnin/segments": "transcript",
    "/captions/enhanced/capabilities": "transcript",
    "/captions/whisperx": "transcript",
    "/transcript": "transcript",
    "/transcript/export": "transcript",
    "/transcript/summarize": "transcript",
    "/whisper/clear-cache": "transcript",
    "/whisper/reinstall": "transcript",
    "/whisper/settings": "transcript",
    # video-effects
    "/silence/speed-up": "video-effects",
    "/video/auto-edit": "video-effects",
    "/video/blend": "video-effects",
    "/video/chromakey": "video-effects",
    "/video/color/correct": "video-effects",
    "/video/cursor-zoom/resolve": "video-effects",
    "/video/highlights": "video-effects",
    "/video/lut/apply": "video-effects",
    "/video/lut/generate-from-ref": "video-effects",
    "/video/merge": "video-effects",
    "/video/particles/apply": "video-effects",
    "/video/pip": "video-effects",
    "/video/speed/ramp": "video-effects",
    "/video/speed/reverse": "video-effects",
    "/video/title/overlay": "video-effects",
    "/video/title/render": "video-effects",
    "/video/transitions/apply": "video-effects",
    "/video/trim": "video-effects",
    "/video/watermark": "video-effects",
    # video-restore
    "/video/ai/capabilities": "video-restore",
    "/video/ai/denoise": "video-restore",
    "/video/ai/install": "video-restore",
    "/video/ai/interpolate": "video-restore",
    "/video/ai/rembg": "video-restore",
    "/video/auto-detect-watermark": "video-restore",
    "/video/face/blur": "video-restore",
    "/video/face/enhance": "video-restore",
    "/video/face/install": "video-restore",
    "/video/face/swap": "video-restore",
    "/video/reframe/face": "video-restore",
    "/video/remove/watermark": "video-restore",
    "/video/upscale/run": "video-restore",
}

# The gate refuses to let this grow. Lower it whenever a route is ported.
UXP_PENDING_FLOOR = 113


_CEP_ROUTE_PATTERNS = (
    re.compile(r"\bstartJob\s*\(\s*[\"'](/[^\"']+)"),
    re.compile(
        r"\bapi\s*\(\s*[\"'](?:GET|POST|PUT|PATCH|DELETE)[\"']\s*,\s*[\"'](/[^\"']+)"
    ),
)
_UXP_ROUTE_PATTERNS = (
    re.compile(
        r"\b(?:BackendClient\.(?:get|post|del)|JobPoller\.start)\s*\(\s*[\"'](/[^\"']+)"
    ),
    re.compile(r"CHAT_ACTION_ENDPOINTS\s*=\s*Object\.freeze\(\{(?P<body>.*?)\}\s*\)", re.S),
)
_ROUTE_LITERAL_PATTERN = re.compile(r"[\"'](/[^\"']+)[\"']")


def _repo_root() -> Path:
    """Return the repository root for source-derived route inspection."""

    return Path(__file__).resolve().parents[2]


def _normalize_route(route: str) -> str:
    """Normalize a literal endpoint while preserving dynamic route intent."""

    normalized = route.split("?", 1)[0].strip()
    if normalized != "/":
        normalized = normalized.rstrip("/")
    return normalized or "/"


def _read_route_source(relative_path: str) -> str:
    try:
        return (_repo_root() / relative_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _extract_routes(text: str, patterns: Sequence[re.Pattern[str]]) -> set[str]:
    routes: set[str] = set()
    for pattern in patterns:
        for match in pattern.finditer(text):
            if "body" in match.groupdict():
                routes.update(_normalize_route(value) for value in _ROUTE_LITERAL_PATTERN.findall(match.group("body")))
            else:
                routes.add(_normalize_route(match.group(1)))
    return {route for route in routes if route}


def _route_sources(relative_paths: Sequence[str], patterns: Sequence[re.Pattern[str]]) -> dict[str, list[str]]:
    sources: dict[str, list[str]] = {}
    for relative_path in relative_paths:
        text = _read_route_source(relative_path)
        for route in _extract_routes(text, patterns):
            sources.setdefault(route, []).append(relative_path)
    return {route: sorted(paths) for route, paths in sources.items()}


def _route_gate_errors(
    cep_routes: Iterable[str],
    uxp_routes: Iterable[str],
    exclusions: Mapping[str, str] = ROUTE_EXCLUSIONS,
    pending: Mapping[str, str] = ROUTE_UXP_PENDING,
    pending_floor: int | None = None,
) -> list[str]:
    cep = {_normalize_route(route) for route in cep_routes}
    uxp = {_normalize_route(route) for route in uxp_routes}
    excluded = {_normalize_route(route) for route in exclusions}
    deferred = {_normalize_route(route) for route in pending}
    unclassified = sorted(cep - uxp - excluded - deferred)
    priority_missing = sorted(set(PRIORITY_ROUTE_COVERAGE) - uxp - excluded)
    floor = UXP_PENDING_FLOOR if pending_floor is None else pending_floor
    errors: list[str] = []
    if unclassified:
        errors.append(
            "CEP routes classified as neither covered, excluded, nor pending: "
            + ", ".join(unclassified)
        )
    unknown_family = sorted(
        {family for family in pending.values() if family not in UXP_PENDING_REASONS}
    )
    if unknown_family:
        errors.append(
            "pending routes name a family with no recorded reason: "
            + ", ".join(unknown_family)
        )
    # The map must be pruned in the same change that ports or removes a route,
    # or it overstates the backlog and turns the floor mushy: a ported route's
    # entry would linger while the row reads "covered", and a deleted route's
    # entry would silently stop counting without anyone deciding that.
    ported = sorted(deferred & uxp)
    if ported:
        errors.append(
            "pending routes that already have a UXP path; remove them from "
            "ROUTE_UXP_PENDING and lower the floor: " + ", ".join(ported)
        )
    vanished = sorted(deferred - cep)
    if vanished:
        errors.append(
            "pending routes the CEP panel no longer calls; remove them from "
            "ROUTE_UXP_PENDING: " + ", ".join(vanished)
        )
    # Only routes the CEP panel actually calls count against the floor, so
    # deleting a CEP route lowers it rather than leaving a stale allowance.
    live_pending = deferred & cep
    if len(live_pending) > floor:
        errors.append(
            f"UXP-pending routes grew to {len(live_pending)} against a floor of "
            f"{floor}; port the route or raise the floor deliberately"
        )
    if priority_missing:
        errors.append(f"priority routes without a UXP path: {', '.join(priority_missing)}")
    return errors


def validate_route_coverage(
    cep_routes: Iterable[str],
    uxp_routes: Iterable[str],
    exclusions: Mapping[str, str] = ROUTE_EXCLUSIONS,
    pending: Mapping[str, str] = ROUTE_UXP_PENDING,
    pending_floor: int | None = None,
) -> list[str]:
    """Return route-parity gate errors for a CEP/UXP route inventory."""

    return _route_gate_errors(cep_routes, uxp_routes, exclusions, pending, pending_floor)


def build_route_coverage_manifest() -> dict:
    """Build a source-derived route coverage inventory for the two panels."""

    cep_sources = _route_sources((ROUTE_CEP_SOURCE,), _CEP_ROUTE_PATTERNS)
    uxp_sources = _route_sources(ROUTE_UXP_SOURCES, _UXP_ROUTE_PATTERNS)
    cep_routes = sorted(cep_sources)
    uxp_routes = sorted(uxp_sources)
    excluded = {_normalize_route(route): reason for route, reason in ROUTE_EXCLUSIONS.items()}
    pending = {_normalize_route(route): family for route, family in ROUTE_UXP_PENDING.items()}
    rows: list[dict] = []
    for route in cep_routes:
        family = ""
        if route in uxp_sources:
            status = "covered"
            justification = ""
        elif route in excluded:
            status = "excluded"
            justification = excluded[route]
        elif route in pending:
            status = "pending"
            family = pending[route]
            justification = UXP_PENDING_REASONS.get(family, "")
        else:
            status = "missing"
            justification = ""
        rows.append(
            {
                "route": route,
                "status": status,
                "priority": route in PRIORITY_ROUTE_COVERAGE,
                "cep_sources": cep_sources[route],
                "uxp_sources": uxp_sources.get(route, []),
                "justification": justification,
                "pending_family": family,
            }
        )

    errors = _route_gate_errors(cep_routes, uxp_routes, excluded, pending)
    covered_count = sum(row["status"] == "covered" for row in rows)
    excluded_count = sum(row["status"] == "excluded" for row in rows)
    pending_rows = [row for row in rows if row["status"] == "pending"]
    pending_families: dict[str, int] = {}
    for row in pending_rows:
        pending_families[row["pending_family"]] = pending_families.get(row["pending_family"], 0) + 1
    missing_routes = [row["route"] for row in rows if row["status"] == "missing"]
    priority_missing = [
        route
        for route in PRIORITY_ROUTE_COVERAGE
        if route not in uxp_sources and route not in excluded
    ]
    total = len(rows)
    return {
        "version": ROUTE_CATALOGUE_VERSION,
        "cep_source": ROUTE_CEP_SOURCE,
        "uxp_sources": list(ROUTE_UXP_SOURCES),
        "priority_routes": list(PRIORITY_ROUTE_COVERAGE),
        "exclusions": excluded,
        "pending_reasons": dict(UXP_PENDING_REASONS),
        "pending_floor": UXP_PENDING_FLOOR,
        "summary": {
            "cep_route_count": total,
            "uxp_route_count": len(uxp_routes),
            "covered": covered_count,
            "excluded": excluded_count,
            "pending": len(pending_rows),
            "pending_by_family": dict(sorted(pending_families.items())),
            "missing": len(missing_routes),
            "coverage_percent": round(((covered_count + excluded_count) / total) * 100, 1) if total else 100.0,
            "priority_count": len(PRIORITY_ROUTE_COVERAGE),
            "priority_covered": len(PRIORITY_ROUTE_COVERAGE) - len(priority_missing),
            "priority_missing": len(priority_missing),
        },
        "gate": {
            "passes": not errors,
            "errors": errors,
            "missing_routes": missing_routes,
            "priority_missing": priority_missing,
        },
        "rows": rows,
    }


@dataclass(frozen=True)
class CepUxpParityEntry:
    """One CEP JSX host function and its post-CEP disposition."""

    name: str
    role: str
    status: str
    risk: str
    uxp_path: str
    replacement_plan: str
    cep_only: bool = False
    f_numbers: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        data = asdict(self)
        data["f_numbers"] = list(self.f_numbers)
        return data


CEP_UXP_PARITY: tuple[CepUxpParityEntry, ...] = (
    CepUxpParityEntry(
        name="ocPing",
        role="Synchronous health probe for the CEP host bridge.",
        status="direct_uxp",
        risk="low",
        uxp_path="Return a direct health response from the UXP host module.",
        replacement_plan="Inline in the UXP panel bridge; no CEP fallback required.",
        f_numbers=("F198", "F252"),
    ),
    CepUxpParityEntry(
        name="ocGetSequenceInfo",
        role="Read the active sequence identity, dimensions, frame rate, and duration.",
        status="direct_uxp",
        risk="low",
        uxp_path="PProBridge.getSequenceInfo() via Sequence settings and active project state.",
        replacement_plan="Keep the UXP bridge implementation as the canonical path.",
        f_numbers=("F198", "F252"),
    ),
    CepUxpParityEntry(
        name="ocAddSequenceMarkers",
        role="Create timeline markers from backend-generated marker payloads.",
        status="direct_uxp",
        risk="low",
        uxp_path="PProBridge.addMarkers() / Sequence marker list APIs.",
        replacement_plan="Port callers to the UXP bridge and retain CEP only for legacy hosts.",
        f_numbers=("F198", "F252"),
    ),
    CepUxpParityEntry(
        name="ocGetSequenceMarkers",
        role="Read active-sequence markers for round-trip metadata workflows.",
        status="direct_uxp",
        risk="low",
        uxp_path="Sequence.getMarkerList() and marker object traversal.",
        replacement_plan="Use the same shape as CEP so backend marker contracts stay stable.",
        f_numbers=("F198", "F252"),
    ),
    CepUxpParityEntry(
        name="ocApplySequenceCuts",
        role="Remove or trim timeline ranges and ripple-delete where requested.",
        status="partial_uxp",
        risk="medium",
        uxp_path=(
            "SequenceEditor.getEditor(sequence).createRemoveItemsAction("
            "trackItemSelection, ripple, mediaType, shiftOverLapping?) -> Action "
            "(confirmed 2026-08-20 in @adobe/premierepro 26.3.0 typings, "
            "src/premierepro.d.ts:3203-3232). Run through "
            "Project.executeTransaction so the edit is undoable and its effect "
            "is observable — preferred over sequence.rippleDelete(), which "
            "premiere-pro-mcp #21 measured returning success while changing "
            "nothing on 26.3."
        ),
        replacement_plan=(
            "Landed 2026-08-20 (F349): applyCuts builds a TrackItemSelection and "
            "runs createRemoveItemsAction inside Project.executeTransaction, "
            "verified by the existing host-write read-back contract. It stays "
            "partial because the action removes whole track items and the typed "
            "API exposes no razor, so a cut whose range crosses a track-item "
            "boundary — silence inside one clip, the common case — still falls "
            "back to sequence.rippleDelete() and records why. Emulating a razor "
            "with clone plus set-start/set-end actions, and confirming the ripple "
            "accounting, needs F252's live-host lane before the CEP fallback "
            "can be dropped."
        ),
        f_numbers=("F198", "F252", "F267", "F349"),
    ),
    CepUxpParityEntry(
        name="ocApplyClipKeyframes",
        role="Apply opacity, scale, position, or other keyframe payloads to clips.",
        status="direct_uxp",
        risk="low",
        uxp_path="Component and property keyframe APIs on UXP clip items.",
        replacement_plan="Port as a UXP bridge operation once feature-level UDT covers keyframes.",
        f_numbers=("F198", "F252", "F267"),
    ),
    CepUxpParityEntry(
        name="ocBatchRenameProjectItems",
        role="Rename selected project items in bulk.",
        status="direct_uxp",
        risk="low",
        uxp_path="Project item rename APIs.",
        replacement_plan="Treat as low-priority convenience wiring in the UXP bridge.",
        f_numbers=("F198", "F252"),
    ),
    CepUxpParityEntry(
        name="ocCreateSmartBins",
        role="Create bins for organized project-item workflows.",
        status="direct_uxp",
        risk="low",
        uxp_path="Project root item and bin creation APIs.",
        replacement_plan="Move bin creation behind the UXP project bridge.",
        f_numbers=("F198", "F252"),
    ),
    CepUxpParityEntry(
        name="ocAddNativeCaptionTrack",
        role="Create a native Premiere caption track from SRT-style segments.",
        status="cep_only",
        risk="high",
        uxp_path=(
            "No UXP caption-track write API as of 26.3 (audited 2026-08-20 against "
            "@adobe/premierepro 26.3.0 typings). CaptionTrack exposes only "
            "getCaptionTrackCount, getCaptionTrack(index), createSetNameAction, and "
            "setMute — read, rename, and mute. Transcript.createImportTextSegmentsAction "
            "targets a ClipProjectItem transcript, which is a different object from a "
            "sequence caption track and is not a substitute."
        ),
        replacement_plan=(
            "Keep CEP fallback while available; prioritize F253 Hybrid Plugin "
            "caption-track creation or adopt an Adobe UXP API if it ships. Until "
            "then the supported UXP-era path is exporting the SRT sidecar for "
            "manual import."
        ),
        cep_only=True,
        f_numbers=("F186", "F198", "F253", "F266"),
    ),
    CepUxpParityEntry(
        name="ocGetProjectBins",
        role="List project bins for import and organization workflows.",
        status="direct_uxp",
        risk="low",
        uxp_path="Project.getRootItem() traversal.",
        replacement_plan="Use UXP project-tree traversal and keep response shape stable.",
        f_numbers=("F198", "F252"),
    ),
    CepUxpParityEntry(
        name="ocExportSequenceRange",
        role="Export a selected active-sequence range.",
        status="direct_uxp",
        risk="low",
        uxp_path="EncoderManager / export APIs where available.",
        replacement_plan="Route to UXP export APIs as part of F255 encoder integration.",
        f_numbers=("F198", "F255"),
    ),
    CepUxpParityEntry(
        name="ocRemoveSequenceMarkers",
        role="Remove OpenCut-created sequence markers.",
        status="direct_uxp",
        risk="low",
        uxp_path="Sequence marker list APIs.",
        replacement_plan="Use UXP marker deletion for parity with marker import/export routes.",
        f_numbers=("F198", "F252"),
    ),
    CepUxpParityEntry(
        name="ocUnrenameItems",
        role="Undo project-item batch rename payloads.",
        status="direct_uxp",
        risk="low",
        uxp_path="Project item rename APIs.",
        replacement_plan="Share the batch-rename UXP bridge path.",
        f_numbers=("F198", "F252"),
    ),
    CepUxpParityEntry(
        name="ocRemoveImportedSequence",
        role="Remove a sequence imported for preview or interchange workflows.",
        status="direct_uxp",
        risk="low",
        uxp_path="Project item delete/remove APIs.",
        replacement_plan="Use UXP project-item deletion with explicit result reporting.",
        f_numbers=("F198", "F252"),
    ),
    CepUxpParityEntry(
        name="ocSetSequencePlayhead",
        role="Move the active sequence playhead to a target time.",
        status="direct_uxp",
        risk="low",
        uxp_path="Active sequence playhead/time APIs.",
        replacement_plan="Expose as a UXP bridge utility for review and marker navigation.",
        f_numbers=("F198", "F252"),
    ),
    CepUxpParityEntry(
        name="ocRemoveImportedItem",
        role="Remove an imported project item by identity.",
        status="direct_uxp",
        risk="low",
        uxp_path="Project item delete/remove APIs.",
        replacement_plan="Share the imported-sequence removal guardrails.",
        f_numbers=("F198", "F252"),
    ),
    CepUxpParityEntry(
        name="ocQeReflect",
        role="Reflect undocumented QE DOM methods for diagnostics.",
        status="cep_only",
        risk="high",
        uxp_path=(
            "No supported UXP QE DOM equivalent (audited 2026-08-20 against "
            "@adobe/premierepro 26.3.0 typings: no qe, reflect, executeScript, or "
            "evalScript surface exists). QE reflection is a CEP/ExtendScript "
            "debug facility, not a user feature, so it retires rather than ports."
        ),
        replacement_plan=(
            "Retire QE reflection after CEP EOL rather than seeking a successor; "
            "replace any real workflow that depends on it one by one with "
            "documented UXP APIs and F267 UDT evidence."
        ),
        cep_only=True,
        f_numbers=("F187", "F198", "F266", "F267"),
    ),
    CepUxpParityEntry(
        name="ocEmitPingEvent",
        role="Emit a panel acknowledgement event for host/panel round trips.",
        status="different_mechanism",
        risk="low",
        uxp_path="UXP addon events or direct backend callback instead of CSXSEvent.",
        replacement_plan="Replace the event transport while keeping the acknowledgement semantics.",
        f_numbers=("F198", "F252"),
    ),
    CepUxpParityEntry(
        name="ocReadPanelBootstrapToken",
        role=(
            "Read the local CSRF bootstrap secret so a file:// panel can prove "
            "it is host-embedded and not a web page."
        ),
        status="different_mechanism",
        risk="low",
        uxp_path=(
            "UXP reads the same secret through its own localFileSystem "
            "permission; no ExtendScript bridge is involved."
        ),
        replacement_plan=(
            "Keep the secret contract identical and swap only the file-read "
            "transport when CEP retires."
        ),
        f_numbers=("F303",),
    ),
)

_ENTRY_BY_NAME = {entry.name: entry for entry in CEP_UXP_PARITY}


def list_parity_entries() -> tuple[CepUxpParityEntry, ...]:
    """Return all catalogue entries in stable host-order."""

    return CEP_UXP_PARITY


def parity_names() -> tuple[str, ...]:
    """Return all catalogued CEP host function names."""

    return tuple(entry.name for entry in CEP_UXP_PARITY)


def cep_only_names() -> tuple[str, ...]:
    """Return the functions that have no supported UXP replacement today."""

    return tuple(entry.name for entry in CEP_UXP_PARITY if entry.cep_only)


def get_parity_entry(name: str) -> CepUxpParityEntry:
    """Return a catalogue entry or raise ``KeyError``."""

    return _ENTRY_BY_NAME[name]


def build_manifest() -> dict:
    """Return a JSON-safe manifest for the CEP/UXP migration catalogue."""

    entries = [entry.as_dict() for entry in CEP_UXP_PARITY]
    cep_only = [entry.name for entry in CEP_UXP_PARITY if entry.cep_only]
    by_status: dict[str, int] = {}
    for entry in CEP_UXP_PARITY:
        by_status[entry.status] = by_status.get(entry.status, 0) + 1
    return {
        "catalogue_version": CATALOGUE_VERSION,
        "source": SOURCE_HOST_FILE,
        "uxp_typings": UXP_TYPINGS,
        "function_count": len(entries),
        "cep_only_count": len(cep_only),
        "cep_only": cep_only,
        "status_counts": dict(sorted(by_status.items())),
        "functions": entries,
    }


def build_dashboard_manifest() -> dict:
    """Return the F260 UXP migration dashboard derived from live sources."""

    source = build_manifest()
    entries = source["functions"]
    risk_counts: dict[str, int] = {}
    for entry in entries:
        risk_counts[entry["risk"]] = risk_counts.get(entry["risk"], 0) + 1

    hybrid_candidates = [
        entry["name"]
        for entry in entries
        if "F253" in entry.get("f_numbers", ()) or entry["status"] == "cep_only"
    ]
    priority = [
        {
            "name": entry["name"],
            "status": entry["status"],
            "risk": entry["risk"],
            "role": entry["role"],
            "replacement_plan": entry["replacement_plan"],
        }
        for entry in entries
        if entry["risk"] == "high" or entry["status"] in {"cep_only", "partial_uxp"}
    ]
    rows = [
        {
            "name": entry["name"],
            "status": entry["status"],
            "risk": entry["risk"],
            "role": entry["role"],
            "uxp_path": entry["uxp_path"],
            "replacement_plan": entry["replacement_plan"],
            "f_numbers": entry["f_numbers"],
            "needs_hybrid": entry["name"] in hybrid_candidates,
            "cep_only": entry["cep_only"],
        }
        for entry in entries
    ]
    return {
        "dashboard_version": 2,
        "source_catalogue_version": source["catalogue_version"],
        "source": source["source"],
        "uxp_typings": source["uxp_typings"],
        "summary": {
            "function_count": source["function_count"],
            "direct_uxp": source["status_counts"].get("direct_uxp", 0),
            "partial_uxp": source["status_counts"].get("partial_uxp", 0),
            "different_mechanism": source["status_counts"].get("different_mechanism", 0),
            "cep_only": source["cep_only_count"],
            "hybrid_candidates": len(hybrid_candidates),
            "high_risk": risk_counts.get("high", 0),
            "medium_risk": risk_counts.get("medium", 0),
            "low_risk": risk_counts.get("low", 0),
        },
        "status_counts": source["status_counts"],
        "risk_counts": dict(sorted(risk_counts.items())),
        "cep_only": source["cep_only"],
        "hybrid_candidates": hybrid_candidates,
        "priority": priority,
        "rows": rows,
        "route_coverage": build_route_coverage_manifest(),
    }


def validate_catalogue(
    host_function_names: Iterable[str],
    entries: Sequence[CepUxpParityEntry] = CEP_UXP_PARITY,
) -> list[str]:
    """Return human-readable catalogue drift errors."""

    errors: list[str] = []
    host_names = set(host_function_names)
    catalogue_names = [entry.name for entry in entries]
    catalogue_set = set(catalogue_names)
    if len(catalogue_names) != len(catalogue_set):
        errors.append("catalogue contains duplicate function names")
    missing = sorted(host_names - catalogue_set)
    extra = sorted(catalogue_set - host_names)
    if missing:
        errors.append(f"missing catalogue entries: {', '.join(missing)}")
    if extra:
        errors.append(f"catalogue entries absent from host JSX: {', '.join(extra)}")

    for entry in entries:
        if entry.status == "cep_only" and not entry.cep_only:
            errors.append(f"{entry.name}: status cep_only requires cep_only=True")
        if entry.cep_only and entry.status != "cep_only":
            errors.append(f"{entry.name}: cep_only=True requires status cep_only")
        if entry.risk not in {"low", "medium", "high"}:
            errors.append(f"{entry.name}: invalid risk {entry.risk!r}")
        if not entry.replacement_plan.strip():
            errors.append(f"{entry.name}: replacement plan is required")
    if set(cep_only_names()) != {"ocAddNativeCaptionTrack", "ocQeReflect"}:
        errors.append("CEP-only surface must stay pinned to caption-track + QE reflection")
    return errors
