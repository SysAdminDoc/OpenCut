"""Feature readiness registry (F100, F191).

Every optional dependency in OpenCut is gated by a ``check_X_available()``
function in :mod:`opencut.checks`. Historically the panel only discovered
that a feature was a stub when it tried to call the route and received a
503/501. The registry centralises that knowledge: each row carries an
explicit *readiness state* plus the install hint and a docs link so the
panel can grey out an action **before** the user clicks it.

Readiness states
----------------

``available``
    Feature is shippable today. The route returns 2xx on a valid request.

``stub``
    The route exists but always returns ``ROUTE_STUBBED`` (501) — we
    documented the shape so MCP clients keep working, but there is no
    implementation behind it yet.

``missing_dependency``
    Implementation lives in the repo but an optional pip extra or system
    binary must be installed before the route can succeed. The route
    returns ``MISSING_DEPENDENCY`` (503) until the dep is present.

``experimental``
    Works but unstable / not yet covered by the standard test gates. UI
    surfaces it with a warning chip.

The hand-written registry is intentionally a plain Python list of dataclasses
so the file is easy to scan and review during PRs. F191 adds a generated
extension: ``opencut/_generated/feature_readiness.json`` maps route functions
that call known ``check_*`` probes back to feature records. The generated rows
are merged at import time so ``GET /system/feature-state`` sees both curated
metadata and directly discoverable dependency gates without scanning source on
every request.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from opencut import checks as _checks

ReadinessState = str  # one of {available, stub, missing_dependency, experimental}

STATE_AVAILABLE: ReadinessState = "available"
STATE_STUB: ReadinessState = "stub"
STATE_MISSING_DEPENDENCY: ReadinessState = "missing_dependency"
STATE_EXPERIMENTAL: ReadinessState = "experimental"

GENERATED_FEATURE_READINESS_PATH = (
    Path(__file__).resolve().parent / "_generated" / "feature_readiness.json"
)

STATES: tuple = (
    STATE_AVAILABLE,
    STATE_STUB,
    STATE_MISSING_DEPENDENCY,
    STATE_EXPERIMENTAL,
)

# Checks that intentionally do not gate a model/AI dependency. This mirrors the
# F115 model-card allowlist, but lives in the registry so readiness derivation
# can share the same taxonomy without importing model_cards.
NON_AI_CHECKS: tuple = (
    "check_aaf_adapter_available",
    "check_ab_av1_available",
    "check_atheris_available",
    "check_birefnet_available",
    "check_changelog_feed_available",
    "check_color_match_available",
    "check_cursor_zoom_available",
    "check_declarative_compose_available",
    "check_demo_bundle_available",
    "check_deprecation_registry_available",
    "check_disk_monitor_available",
    "check_event_moments_available",
    "check_footage_search_available",
    "check_gist_sync_available",
    "check_gpu_semaphore_available",
    "check_issue_report_available",
    "check_loudness_match_available",
    "check_neural_interp_available",
    "check_obs_bridge_available",
    "check_onboarding_available",
    "check_openapi_available",
    "check_otio_available",
    "check_otio_diff_available",
    "check_pedalboard_available",
    "check_quality_metrics_available",
    "check_rate_limit_categories_available",
    "check_request_correlation_available",
    "check_resolve_available",
    "check_rife_cli_available",
    "check_runpod_available",
    "check_sentry_available",
    "check_shaka_available",
    "check_social_post_available",
    "check_srt_available",
    "check_svtav1_psy_available",
    "check_temp_cleanup_available",
    "check_vmaf_available",
    "check_vvc_available",
    "check_websocket_available",
)


@dataclass
class FeatureRecord:
    """A single optional feature surface."""

    feature_id: str
    label: str
    category: str
    state: ReadinessState
    install_hint: str = ""
    docs: str = ""
    routes: List[str] = field(default_factory=list)
    probe: Optional[Callable[[], bool]] = None
    check_name: str = ""
    source: str = "manual"
    notes: str = ""

    def resolved_state(self) -> ReadinessState:
        """Return the readiness state after probing optional dependencies."""
        if self.state == STATE_AVAILABLE and self.probe is not None:
            try:
                ok = bool(self.probe())
            except Exception:  # pragma: no cover - defensive against probe errors
                ok = False
            return STATE_AVAILABLE if ok else STATE_MISSING_DEPENDENCY
        return self.state

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload.pop("probe", None)
        payload["state"] = self.resolved_state()
        return payload


def _check(name: str) -> Optional[Callable[[], bool]]:
    """Return a callable from :mod:`opencut.checks` if it exists."""
    fn = getattr(_checks, name, None)
    return fn if callable(fn) else None


# ---------------------------------------------------------------------------
# Feature catalogue
# ---------------------------------------------------------------------------
#
# Keep this list deduped and alphabetical within each category. The
# probes use the existing ``opencut.checks`` API so we don't duplicate
# import-detection logic. ``state=STATE_AVAILABLE`` means "stable when
# the probe says yes" — the panel still calls ``resolved_state()`` to
# get the runtime answer.
#
# This curated list remains hand-written: these rows carry labels, install
# hints, docs links, and shipped-vs-stub judgement. The generated F191 rows are
# loaded below and merged with this metadata instead of replacing it.


_FEATURES: List[FeatureRecord] = [
    # ---- Audio --------------------------------------------------------
    FeatureRecord(
        feature_id="audio.demucs",
        label="Demucs stem separation",
        category="audio",
        state=STATE_AVAILABLE,
        install_hint="pip install demucs",
        docs="docs/MODELS.md#demucs",
        routes=["/audio/separate"],
        probe=_check("check_demucs_available"),
    ),
    FeatureRecord(
        feature_id="audio.deepfilter",
        label="DeepFilterNet studio sound",
        category="audio",
        state=STATE_AVAILABLE,
        install_hint="pip install deepfilternet",
        routes=["/audio/pro/deepfilter"],
        probe=_check("check_deepfilternet_available"),
    ),
    FeatureRecord(
        feature_id="audio.pedalboard",
        label="Pedalboard audio effects",
        category="audio",
        state=STATE_AVAILABLE,
        install_hint="pip install pedalboard",
        routes=["/audio/effects/pedalboard"],
        probe=_check("check_pedalboard_available"),
    ),
    FeatureRecord(
        feature_id="audio.audiocraft",
        label="AudioCraft music generation",
        category="audio",
        state=STATE_AVAILABLE,
        install_hint="pip install audiocraft",
        routes=["/audio/music/audiocraft"],
        probe=_check("check_audiocraft_available"),
    ),
    FeatureRecord(
        feature_id="audio.edge-tts",
        label="Edge TTS voice synthesis",
        category="audio",
        state=STATE_AVAILABLE,
        install_hint="pip install edge-tts",
        routes=["/audio/tts/edge"],
        probe=_check("check_edge_tts_available"),
    ),
    FeatureRecord(
        feature_id="audio.loudness-match",
        label="EBU R128 two-pass loudness match",
        category="audio",
        state=STATE_AVAILABLE,
        install_hint="bundled FFmpeg",
        routes=["/audio/loudness-match"],
        probe=_check("check_loudness_match_available"),
    ),
    # ---- Video --------------------------------------------------------
    FeatureRecord(
        feature_id="video.upscale.realesrgan",
        label="Real-ESRGAN AI upscale",
        category="video",
        state=STATE_AVAILABLE,
        install_hint="pip install realesrgan",
        routes=["/video/upscale/realesrgan"],
        probe=_check("check_upscale_available"),
    ),
    FeatureRecord(
        feature_id="video.matte.rvm",
        label="RVM robust video matting",
        category="video",
        state=STATE_AVAILABLE,
        install_hint="pip install robust-video-matting",
        routes=["/video/matte/rvm"],
        probe=_check("check_rvm_available"),
    ),
    FeatureRecord(
        feature_id="video.depth",
        label="Depth Anything depth maps",
        category="video",
        state=STATE_AVAILABLE,
        install_hint="pip install transformers torch",
        routes=["/video/depth"],
        probe=_check("check_depth_available"),
    ),
    FeatureRecord(
        feature_id="video.scenes.detect",
        label="PySceneDetect shot boundary",
        category="video",
        state=STATE_AVAILABLE,
        install_hint="pip install scenedetect",
        routes=["/video/scenes/detect", "/video/scenes/auto"],
        probe=_check("check_scenedetect_available"),
    ),
    FeatureRecord(
        feature_id="video.propainter",
        label="ProPainter video inpainting",
        category="video",
        state=STATE_AVAILABLE,
        install_hint="pip install propainter (heavy GPU model)",
        routes=["/video/inpaint/propainter"],
        probe=_check("check_propainter_available"),
    ),
    FeatureRecord(
        feature_id="video.watermark.remove",
        label="Watermark removal",
        category="video",
        state=STATE_AVAILABLE,
        install_hint="pip install simple-lama-inpainting",
        routes=["/video/watermark/remove"],
        probe=_check("check_watermark_available"),
    ),
    # ---- Captions / Transcripts ---------------------------------------
    FeatureRecord(
        feature_id="captions.whisperx",
        label="WhisperX captions + diarisation",
        category="captions",
        state=STATE_AVAILABLE,
        install_hint="pip install whisperx",
        routes=["/captions/whisperx"],
        probe=_check("check_whisperx_available"),
    ),
    FeatureRecord(
        feature_id="captions.silero-vad",
        label="Silero VAD voice activity",
        category="captions",
        state=STATE_AVAILABLE,
        install_hint="pip install silero-vad",
        routes=["/captions/vad"],
        probe=_check("check_silero_vad_available"),
    ),
    FeatureRecord(
        feature_id="captions.crisper-whisper",
        label="CrisperWhisper precise timings",
        category="captions",
        state=STATE_AVAILABLE,
        install_hint="pip install ctranslate2 + crisper-whisper",
        routes=["/captions/crisper"],
        probe=_check("check_crisper_whisper_available"),
    ),
    # ---- Editing / Automation -----------------------------------------
    FeatureRecord(
        feature_id="editing.auto-editor",
        label="auto-editor automatic cuts",
        category="editing",
        state=STATE_AVAILABLE,
        install_hint="pip install auto-editor",
        routes=["/timeline/auto-editor"],
        probe=_check("check_auto_editor_available"),
    ),
    FeatureRecord(
        feature_id="editing.color-match",
        label="OpenCV color match",
        category="editing",
        state=STATE_AVAILABLE,
        install_hint="pip install opencv-python numpy",
        routes=["/video/color-match"],
        probe=_check("check_color_match_available"),
    ),
    FeatureRecord(
        feature_id="editing.auto-zoom",
        label="Auto-zoom keyframes",
        category="editing",
        state=STATE_AVAILABLE,
        install_hint="pip install opencv-python",
        routes=["/video/auto-zoom"],
        probe=_check("check_auto_zoom_available"),
    ),
    # ---- Interchange / Export ----------------------------------------
    FeatureRecord(
        feature_id="export.otio",
        label="OpenTimelineIO interchange",
        category="export",
        state=STATE_AVAILABLE,
        install_hint="pip install opentimelineio",
        routes=["/timeline/export/otio", "/timeline/import/otio"],
        probe=_check("check_otio_available"),
    ),
    FeatureRecord(
        feature_id="export.aaf",
        label="OTIO → AAF Avid bridge",
        category="export",
        state=STATE_AVAILABLE,
        install_hint="pip install otio-aaf-adapter",
        routes=["/timeline/export/aaf"],
        probe=_check("check_otio_available"),
    ),
    # ---- LLM / NLP ----------------------------------------------------
    FeatureRecord(
        feature_id="llm.local",
        label="Local LLM provider (Ollama / vLLM)",
        category="llm",
        state=STATE_AVAILABLE,
        install_hint="install Ollama or set OPENAI_API_KEY",
        routes=["/llm/chat"],
        probe=_check("check_llm_available"),
    ),
    FeatureRecord(
        feature_id="llm.ollama",
        label="Ollama backend",
        category="llm",
        state=STATE_AVAILABLE,
        install_hint="install Ollama from ollama.ai",
        routes=["/llm/ollama"],
        probe=_check("check_ollama_available"),
    ),
    # ---- Stubs (still on the roadmap) ---------------------------------
    FeatureRecord(
        feature_id="captions.qc",
        label="Caption QC gate",
        category="captions",
        state=STATE_AVAILABLE,
        install_hint="bundled (no extra deps)",
        docs="ROADMAP.md#F111",
        notes="Wraps caption_compliance with stricter defaults + forbidden-glyph + overlap rules.",
        routes=["/captions/qc"],
    ),
    FeatureRecord(
        feature_id="system.capabilities",
        label="Codec + hardware capability probe",
        category="system",
        state=STATE_AVAILABLE,
        install_hint="bundled (no extra deps)",
        docs="ROADMAP.md#F106",
        routes=["/system/capabilities"],
    ),
    FeatureRecord(
        feature_id="markers.import",
        label="CSV / EDL / Premiere marker import",
        category="export",
        state=STATE_AVAILABLE,
        install_hint="bundled (no extra deps)",
        docs="ROADMAP.md#F102",
        routes=["/markers/import"],
    ),
    FeatureRecord(
        feature_id="review.bundle",
        label="Portable review bundle export",
        category="export",
        state=STATE_AVAILABLE,
        install_hint="bundled (no extra deps)",
        docs="ROADMAP.md#F105",
        routes=["/review/bundle"],
    ),
    FeatureRecord(
        feature_id="provenance.c2pa-sidecar",
        label="C2PA provenance sidecar (unsigned by default)",
        category="export",
        state=STATE_AVAILABLE,
        install_hint="bundled; pip install cryptography to enable Ed25519 signing",
        docs="ROADMAP.md#F110",
        routes=["/provenance/c2pa", "/provenance/verify"],
    ),
    FeatureRecord(
        feature_id="system.ai-eval-harness",
        label="AI evaluation harness",
        category="system",
        state=STATE_AVAILABLE,
        install_hint="bundled (no extra deps)",
        docs="ROADMAP.md#F120",
        routes=["/system/ai-eval", "/system/ai-eval/<feature_id>"],
    ),
    FeatureRecord(
        feature_id="system.ocio-validate",
        label="OpenColorIO + ACES validator",
        category="system",
        state=STATE_AVAILABLE,
        install_hint="bundled; pip install PyOpenColorIO for full surface (route returns available=False when missing)",
        docs="ROADMAP.md#F109",
        routes=["/system/ocio"],
    ),
    FeatureRecord(
        feature_id="system.project-health",
        label="Local project + media health report",
        category="system",
        state=STATE_AVAILABLE,
        install_hint="bundled (no extra deps)",
        docs="ROADMAP.md#F011",
        routes=["/system/project-health"],
    ),
    FeatureRecord(
        feature_id="system.crash-packet",
        label="Crash + recovery diagnostic packet",
        category="system",
        state=STATE_AVAILABLE,
        install_hint="bundled (no extra deps)",
        docs="ROADMAP.md#F066",
        routes=["/system/crash-packet"],
    ),
    FeatureRecord(
        feature_id="system.job-diagnostics",
        label="Per-job diagnostic payload with correlation IDs",
        category="system",
        state=STATE_AVAILABLE,
        install_hint="bundled (no extra deps)",
        docs="ROADMAP.md#F010",
        routes=["/jobs/<job_id>/diagnostics"],
    ),
    FeatureRecord(
        feature_id="lipsync.latentsync",
        label="LatentSync diffusion lip-sync",
        category="ai",
        state=STATE_STUB,
        docs="ROADMAP-NEXT.md#B1",
        routes=["/lipsync/latentsync"],
    ),
    FeatureRecord(
        feature_id="lipsync.musetalk",
        label="MuseTalk real-time lip-sync",
        category="ai",
        state=STATE_STUB,
        docs="ROADMAP-NEXT.md#B1",
        routes=["/lipsync/musetalk"],
    ),
    FeatureRecord(
        feature_id="video.upscale.flashvsr",
        label="FlashVSR streaming VSR",
        category="ai",
        state=STATE_STUB,
        docs="ROADMAP-NEXT.md#H2.1",
        routes=["/video/upscale/flashvsr"],
    ),
    FeatureRecord(
        feature_id="video.inpaint.rose",
        label="ROSE shadow-aware inpaint",
        category="ai",
        state=STATE_STUB,
        docs="ROADMAP-NEXT.md#H2.2",
        routes=["/video/inpaint/rose"],
    ),
    # ---- Experimental -------------------------------------------------
    FeatureRecord(
        feature_id="ai.video.cogvideox",
        label="CogVideoX text-to-video",
        category="ai",
        state=STATE_EXPERIMENTAL,
        install_hint="pip install diffusers + heavy GPU model",
        routes=["/generate/cogvideox"],
    ),
    FeatureRecord(
        feature_id="ai.video.ltx",
        label="LTX-Video generation",
        category="ai",
        state=STATE_EXPERIMENTAL,
        install_hint="pip install ltx-video (Apache-2)",
        routes=["/generate/ltx"],
    ),
]


def _unique_routes(routes: Iterable[str]) -> List[str]:
    return sorted({route for route in routes if route})


def _probe_name(record: FeatureRecord) -> str:
    if record.check_name:
        return record.check_name
    if record.probe is not None:
        return getattr(record.probe, "__name__", "")
    return ""


def _record_from_generated(payload: dict) -> FeatureRecord:
    check_name = str(payload.get("check_name") or "")
    return FeatureRecord(
        feature_id=str(payload["feature_id"]),
        label=str(payload.get("label") or payload["feature_id"]),
        category=str(payload.get("category") or "generated"),
        state=str(payload.get("state") or STATE_AVAILABLE),
        install_hint=str(payload.get("install_hint") or ""),
        docs=str(payload.get("docs") or ""),
        routes=_unique_routes(payload.get("routes") or []),
        probe=_check(check_name),
        check_name=check_name,
        source=str(payload.get("source") or "generated"),
        notes=str(payload.get("notes") or ""),
    )


def load_generated_feature_records(
    path: Path = GENERATED_FEATURE_READINESS_PATH,
) -> List[FeatureRecord]:
    """Load F191 generated feature records from disk.

    Missing files are tolerated so editable installs created before F191 still
    import cleanly; ``dump_feature_readiness --check`` enforces the committed
    artifact for releases.
    """
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    records = payload.get("records") or []
    out: List[FeatureRecord] = []
    for record in records:
        try:
            out.append(_record_from_generated(record))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _merge_generated_records(
    manual_records: Iterable[FeatureRecord],
    generated_records: Iterable[FeatureRecord],
) -> List[FeatureRecord]:
    records = list(manual_records)
    by_probe: Dict[str, FeatureRecord] = {}
    for record in records:
        probe_name = _probe_name(record)
        if probe_name and probe_name not in by_probe:
            by_probe[probe_name] = record

    existing_ids = {record.feature_id for record in records}
    for generated in generated_records:
        probe_name = _probe_name(generated)
        target = by_probe.get(probe_name)
        if target is not None:
            target.routes = _unique_routes([*target.routes, *generated.routes])
            if target.check_name == "":
                target.check_name = probe_name
            continue
        if generated.feature_id in existing_ids:
            continue
        records.append(generated)
        existing_ids.add(generated.feature_id)
        if probe_name and probe_name not in by_probe:
            by_probe[probe_name] = generated
    return records


def _build_index(records: Iterable[FeatureRecord]) -> Dict[str, FeatureRecord]:
    out: Dict[str, FeatureRecord] = {}
    for record in records:
        if record.feature_id in out:
            raise ValueError(f"duplicate feature_id: {record.feature_id!r}")
        out[record.feature_id] = record
    return out


_GENERATED_FEATURES = load_generated_feature_records()


# Public, immutable view of the catalogue.
FEATURES: Dict[str, FeatureRecord] = _build_index(
    _merge_generated_records(_FEATURES, _GENERATED_FEATURES)
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_features() -> List[FeatureRecord]:
    """Return every registered feature in insertion order."""
    return list(FEATURES.values())


def get_feature(feature_id: str) -> Optional[FeatureRecord]:
    return FEATURES.get(feature_id)


def feature_states() -> Dict[str, ReadinessState]:
    """Return ``{feature_id: resolved_state}`` for fast manifest dumps."""
    return {fid: rec.resolved_state() for fid, rec in FEATURES.items()}


def feature_manifest() -> dict:
    """Build the JSON payload served by ``GET /system/feature-state``."""
    payload = [record.as_dict() for record in FEATURES.values()]
    counts: Dict[str, int] = {state: 0 for state in STATES}
    for record in payload:
        counts[record["state"]] = counts.get(record["state"], 0) + 1
    return {
        "version": 1,
        "states": list(STATES),
        "counts": counts,
        "generated": {
            "source": "opencut/_generated/feature_readiness.json",
            "record_count": len(_GENERATED_FEATURES),
            "route_count": sum(len(record.routes) for record in _GENERATED_FEATURES),
        },
        "features": payload,
    }


def assert_states_valid(records: Iterable[FeatureRecord] = ()) -> None:
    """Raise ``ValueError`` if any record uses an unknown state."""
    sample = list(records) or list(FEATURES.values())
    for record in sample:
        if record.state not in STATES:
            raise ValueError(
                f"feature {record.feature_id!r} uses unknown state {record.state!r}"
            )
