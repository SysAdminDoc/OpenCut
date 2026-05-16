"""Feature readiness registry (F100).

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

The registry is intentionally a plain Python list of dataclasses so the
file is easy to scan and review during PRs. ``available`` rows resolve
their state by calling the ``probe`` callable; everything else is
declared up-front so the panel can render the catalogue without
importing the heavy AI extras.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Iterable, List, Optional

from opencut import checks as _checks

ReadinessState = str  # one of {available, stub, missing_dependency, experimental}

STATE_AVAILABLE: ReadinessState = "available"
STATE_STUB: ReadinessState = "stub"
STATE_MISSING_DEPENDENCY: ReadinessState = "missing_dependency"
STATE_EXPERIMENTAL: ReadinessState = "experimental"

STATES: tuple = (
    STATE_AVAILABLE,
    STATE_STUB,
    STATE_MISSING_DEPENDENCY,
    STATE_EXPERIMENTAL,
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
    """Return ``opencut.checks.check_<name>_available`` if it exists."""
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
# The list is intentionally NOT auto-derived from ``opencut.checks``:
# we want each registry row to carry an install hint and docs link,
# and that metadata must be hand-written.


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


def _build_index(records: Iterable[FeatureRecord]) -> Dict[str, FeatureRecord]:
    out: Dict[str, FeatureRecord] = {}
    for record in records:
        if record.feature_id in out:
            raise ValueError(f"duplicate feature_id: {record.feature_id!r}")
        out[record.feature_id] = record
    return out


# Public, immutable view of the catalogue.
FEATURES: Dict[str, FeatureRecord] = _build_index(_FEATURES)


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
