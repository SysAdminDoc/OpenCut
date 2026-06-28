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
    hardware: str = ""
    requires_gpu: bool = False
    minimum_vram_mb: int = 0
    privacy: str = ""
    license: str = ""
    advisory_notes: List[str] = field(default_factory=list)

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
        hardware="gpu (>= 8 GB VRAM)",
        requires_gpu=True,
        minimum_vram_mb=8192,
    ),
    FeatureRecord(
        feature_id="audio.beatnet",
        label="BeatNet beat/downbeat tracker",
        category="audio",
        state=STATE_AVAILABLE,
        install_hint="pip install BeatNet",
        docs="docs/MODELS.md#beatnet-beatdownbeat-tracker",
        routes=["/audio/beats/beatnet"],
        probe=_check("check_beatnet_available"),
        hardware="cpu",
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
        feature_id="audio.f5-tts",
        label="F5-TTS zero-shot voice clone",
        category="audio",
        state=STATE_AVAILABLE,
        install_hint="pip install f5-tts",
        docs="docs/MODELS.md#f5-tts-zero-shot-voice-clone",
        routes=["/audio/tts/f5", "/audio/tts/f5/models"],
        probe=_check("check_f5_tts_available"),
        hardware="gpu (>= 6 GB VRAM)",
        requires_gpu=True,
        minimum_vram_mb=6144,
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
    FeatureRecord(
        feature_id="audio.voice-grammar",
        label="Voice-command grammar",
        category="audio",
        state=STATE_AVAILABLE,
        install_hint="pip install whisperx + faster-whisper",
        docs="docs/MODELS.md#whisperx-voice-command-grammar",
        routes=["/voice/grammar/catalogue", "/voice/grammar/parse"],
        probe=_check("check_voice_grammar_available"),
        hardware="cpu/gpu",
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
        hardware="gpu",
        requires_gpu=True,
    ),
    FeatureRecord(
        feature_id="video.upscale.seedvr2",
        label="SeedVR2 one-step diffusion VSR",
        category="video",
        state=STATE_AVAILABLE,
        install_hint="pip install diffusers torch",
        docs="docs/MODELS.md#seedvr2-one-step-diffusion-vsr",
        routes=["/video/upscale/smart"],
        probe=_check("check_seedvr2_available"),
        hardware="gpu (>= 8 GB VRAM)",
        requires_gpu=True,
        minimum_vram_mb=8192,
    ),
    FeatureRecord(
        feature_id="video.matte.rvm",
        label="RVM robust video matting",
        category="video",
        state=STATE_AVAILABLE,
        install_hint="pip install robust-video-matting",
        routes=["/video/matte/rvm"],
        probe=_check("check_rvm_available"),
        hardware="gpu",
        requires_gpu=True,
    ),
    FeatureRecord(
        feature_id="video.depth",
        label="Depth Anything depth maps",
        category="video",
        state=STATE_AVAILABLE,
        install_hint="pip install transformers torch",
        routes=["/video/depth"],
        probe=_check("check_depth_available"),
        hardware="gpu",
        requires_gpu=True,
    ),
    FeatureRecord(
        feature_id="video.depth.da3",
        label="Depth Anything 3 Small",
        category="video",
        state=STATE_AVAILABLE,
        install_hint="pip install transformers torch",
        docs="docs/MODELS.md#depth-anything-3-small",
        routes=["/video/depth"],
        probe=_check("check_depth_anything_3_available"),
        hardware="gpu",
        requires_gpu=True,
    ),
    FeatureRecord(
        feature_id="video.scenes.detect",
        label="PySceneDetect shot boundary",
        category="video",
        state=STATE_AVAILABLE,
        install_hint="pip install scenedetect",
        routes=["/video/scenes/detect", "/video/scenes/auto"],
        probe=_check("check_scenedetect_available"),
        hardware="cpu",
    ),
    FeatureRecord(
        feature_id="video.scenes.transnetv2",
        label="TransNetV2 shot-boundary detection",
        category="video",
        state=STATE_AVAILABLE,
        install_hint="pip install transnetv2-pytorch",
        docs="docs/MODELS.md#transnetv2-shot-boundary",
        routes=["/video/scenes/auto"],
        probe=_check("check_transnetv2_available"),
        hardware="gpu",
        requires_gpu=True,
    ),
    FeatureRecord(
        feature_id="video.scenes.autoshot",
        label="AutoShot gradual-transition scene detector",
        category="video",
        state=STATE_AVAILABLE,
        install_hint="git clone https://github.com/wentaozhu/AutoShot && pip install -e AutoShot",
        docs="docs/MODELS.md#autoshot-gradual-transition-scene-detector",
        routes=["/video/scenes/auto"],
        probe=_check("check_autoshot_available"),
        hardware="cpu/gpu",
    ),
    FeatureRecord(
        feature_id="video.propainter",
        label="ProPainter video inpainting",
        category="video",
        state=STATE_AVAILABLE,
        install_hint="pip install propainter (heavy GPU model)",
        routes=["/video/inpaint/propainter"],
        probe=_check("check_propainter_available"),
        hardware="gpu (>= 12 GB VRAM)",
        requires_gpu=True,
        minimum_vram_mb=12288,
    ),
    FeatureRecord(
        feature_id="video.sam2",
        label="Segment Anything 2 object masks",
        category="video",
        state=STATE_AVAILABLE,
        install_hint="pip install sam2",
        docs="docs/MODELS.md#segment-anything-2-sam2",
        routes=[
            "/object-effects/generate-mask",
            "/video/physics-remove",
            "/video/text-segment",
        ],
        probe=_check("check_sam2_available"),
        hardware="gpu",
        requires_gpu=True,
    ),
    FeatureRecord(
        feature_id="video.sam3",
        label="SAM 3 text-prompted video segmentation",
        category="video",
        state=STATE_AVAILABLE,
        install_hint="pip install sam3 torch",
        docs="docs/MODELS.md#sam-3-text-prompted-video-segmentation",
        routes=["/object-effects/generate-mask", "/video/text-segment"],
        probe=_check("check_sam3_available"),
        hardware="gpu",
        requires_gpu=True,
    ),
    FeatureRecord(
        feature_id="video.rembg",
        label="rembg background removal",
        category="video",
        state=STATE_AVAILABLE,
        install_hint="pip install rembg",
        docs="docs/MODELS.md#rembg-u2-net-family",
        routes=["/video/ai/rembg"],
        probe=_check("check_rembg_available"),
        hardware="cpu/gpu",
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
    FeatureRecord(
        feature_id="video.relight.iclight",
        label="IC-Light v1 relight",
        category="video",
        state=STATE_AVAILABLE,
        install_hint="pip install diffusers>=0.32 torch",
        docs="docs/MODELS.md#ic-light-v1-relight",
        routes=["/video/relight/iclight", "/video/relight/iclight/capabilities"],
        probe=_check("check_iclight_available"),
        hardware="gpu (>= 4 GB VRAM)",
        requires_gpu=True,
        minimum_vram_mb=4096,
    ),
    FeatureRecord(
        feature_id="video.quality.clip-iqa",
        label="CLIP-IQA+ quality scoring",
        category="video",
        state=STATE_AVAILABLE,
        install_hint="pip install pyiqa",
        docs="docs/MODELS.md#clip-iqa",
        routes=["/video/quality/rank", "/video/quality/score"],
        probe=_check("check_clip_iqa_available"),
        hardware="gpu",
        requires_gpu=True,
    ),
    FeatureRecord(
        feature_id="video.emotion-arc",
        label="HSEmotion emotion timeline",
        category="video",
        state=STATE_AVAILABLE,
        install_hint="pip install hsemotion",
        docs="docs/MODELS.md#hsemotion",
        routes=["/video/emotion/arc"],
        probe=_check("check_hsemotion_available"),
        hardware="cpu/gpu",
    ),
    FeatureRecord(
        feature_id="video.deepface",
        label="DeepFace emotion analysis",
        category="video",
        state=STATE_AVAILABLE,
        install_hint="pip install deepface",
        docs="docs/MODELS.md#deepface",
        routes=["/video/emotion-highlights", "/video/emotion/arc"],
        probe=_check("check_deepface_available"),
        hardware="cpu/gpu",
    ),
    FeatureRecord(
        feature_id="video.upscale.hub",
        label="HF smart upscaling hub",
        category="video",
        state=STATE_AVAILABLE,
        install_hint="pip install diffusers transformers",
        docs="docs/MODELS.md#hf-upscale-model-hub",
        routes=["/video/upscale/smart", "/video/upscale/smart/info"],
        probe=_check("check_upscale_hub_available"),
        hardware="gpu",
        requires_gpu=True,
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
        hardware="cpu",
    ),
    FeatureRecord(
        feature_id="captions.crisper-whisper",
        label="CrisperWhisper precise timings",
        category="captions",
        state=STATE_AVAILABLE,
        install_hint="pip install ctranslate2 + crisper-whisper",
        routes=["/captions/crisper"],
        probe=_check("check_crisper_whisper_available"),
        hardware="gpu",
        requires_gpu=True,
    ),
    FeatureRecord(
        feature_id="captions.translate.nllb",
        label="NLLB-200 distilled caption translation",
        category="captions",
        state=STATE_AVAILABLE,
        install_hint="pip install ctranslate2 sentencepiece huggingface-hub",
        docs="docs/MODELS.md#nllb-200-distilled-caption-translation",
        routes=["/captions/translate", "/captions/enhanced/install"],
        probe=_check("check_nllb_available"),
        hardware="cpu/gpu",
    ),
    FeatureRecord(
        feature_id="captions.translate.seamless-m4t",
        label="SeamlessM4T v2 caption translation",
        category="captions",
        state=STATE_AVAILABLE,
        install_hint="pip install transformers torch sentencepiece",
        docs="docs/MODELS.md#seamlessm4t-v2-caption-translation",
        routes=["/captions/translate", "/captions/enhanced/install"],
        probe=_check("check_seamless_m4t_available"),
        hardware="gpu (recommended)",
        requires_gpu=True,
    ),
    FeatureRecord(
        feature_id="captions.nemo-asr",
        label="NVIDIA NeMo ASR",
        category="captions",
        state=STATE_AVAILABLE,
        install_hint='pip install "nemo_toolkit[asr]"',
        docs="docs/MODELS.md#nvidia-nemo-asr-parakeet--canary",
        routes=["/transcribe", "/captions/generate"],
        probe=_check("check_nemo_asr_available"),
        hardware="cpu/gpu",
    ),
    FeatureRecord(
        feature_id="captions.pyannote-diarization",
        label="pyannote speaker diarization",
        category="captions",
        state=STATE_AVAILABLE,
        install_hint="pip install pyannote.audio",
        docs="docs/MODELS.md#pyannote-speaker-diarization",
        routes=["/captions/generate", "/video/multimodal-diarize"],
        probe=_check("check_diarization_available"),
        hardware="cpu/gpu",
    ),
    FeatureRecord(
        feature_id="captions.sortformer",
        label="NVIDIA Sortformer diarization",
        category="captions",
        state=STATE_AVAILABLE,
        install_hint='pip install "nemo_toolkit[asr]"',
        docs="docs/MODELS.md#nvidia-sortformer-diarization",
        routes=["/captions/generate", "/video/multimodal-diarize"],
        probe=_check("check_sortformer_available"),
        hardware="gpu",
        requires_gpu=True,
    ),
    FeatureRecord(
        feature_id="captions.multimodal-diarise",
        label="Multimodal diarisation pipeline",
        category="captions",
        state=STATE_AVAILABLE,
        install_hint="pip install whisperx + pyannote.audio",
        docs="docs/MODELS.md#multimodal-diarisation-pipeline",
        routes=["/video/multimodal-diarize"],
        probe=_check("check_multimodal_diarize_available"),
        hardware="gpu",
        requires_gpu=True,
    ),
    # ---- Editing / Automation -----------------------------------------
    FeatureRecord(
        feature_id="editing.auto-editor",
        label="auto-editor automatic cuts",
        category="editing",
        state=STATE_AVAILABLE,
        install_hint="Native v30+ binary: https://github.com/WyattBlue/auto-editor/releases — or legacy pip: pip install auto-editor",
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
    FeatureRecord(
        feature_id="video.multicam.visual-cues",
        label="MediaPipe visual multicam cues",
        category="video",
        state=STATE_AVAILABLE,
        install_hint="pip install mediapipe opencv-python-headless",
        docs="docs/MODELS.md#mediapipe-visual-multicam-cues",
        routes=["/editing/active-speaker-switch"],
        probe=_check("check_visual_multicam_available"),
        hardware="cpu/gpu",
    ),
    FeatureRecord(
        feature_id="vision.ocr",
        label="pytesseract / EasyOCR text extraction",
        category="vision",
        state=STATE_AVAILABLE,
        install_hint="pip install pytesseract easyocr",
        docs="docs/MODELS.md#pytesseract--easyocr-text-extraction",
        routes=["/api/ai/ocr"],
        probe=_check("check_ocr_available"),
        hardware="cpu/gpu",
    ),
    FeatureRecord(
        feature_id="editing.broll-generate",
        label="AI B-roll generation",
        category="editing",
        state=STATE_AVAILABLE,
        install_hint="pip install transformers torch",
        docs="docs/MODELS.md#b-roll-suggestion-pipeline",
        routes=[
            "/ai/generate-broll",
            "/ai/generate-broll/batch",
            "/video/broll-generate",
        ],
        probe=_check("check_broll_generate_available"),
        hardware="cpu/gpu",
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
        hardware="gpu (recommended)",
        requires_gpu=True,
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
    FeatureRecord(
        feature_id="analyze.virality",
        label="Virality / hook scorer",
        category="llm",
        state=STATE_AVAILABLE,
        install_hint="ships with core; optional LLM dependency",
        docs="docs/MODELS.md#virality--hook-scorer",
        routes=["/analyze/virality", "/analyze/virality/rank"],
        probe=_check("check_virality_score_available"),
        hardware="cpu",
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
        category="lipsync",
        state=STATE_STUB,
        install_hint="pip install diffusers torch torchvision",
        docs="docs/MODELS.md#latentsync-diffusion-lip-sync",
        routes=["/lipsync/latentsync"],
        probe=_check("check_latentsync_available"),
        hardware="gpu (>= 6 GB VRAM)",
        requires_gpu=True,
        minimum_vram_mb=6144,
    ),
    FeatureRecord(
        feature_id="lipsync.musetalk",
        label="MuseTalk real-time lip-sync",
        category="ai",
        state=STATE_STUB,
        docs="ROADMAP.md",
        routes=["/lipsync/musetalk"],
    ),
    FeatureRecord(
        feature_id="lipsync.advanced",
        label="GaussianHeadTalk / FantasyTalking2 lip-sync",
        category="lipsync",
        state=STATE_STUB,
        install_hint="pip install gaussian-head-talk + fantasytalking2",
        docs="docs/MODELS.md#gaussianheadtalk--fantasytalking2",
        routes=[
            "/lipsync/advanced/backends",
            "/lipsync/fantasy2",
            "/lipsync/gaussian",
        ],
        probe=_check("check_lipsync_advanced_available"),
        hardware="gpu (>= 12 GB VRAM)",
        requires_gpu=True,
        minimum_vram_mb=12288,
    ),
    FeatureRecord(
        feature_id="video.upscale.flashvsr",
        label="FlashVSR streaming VSR",
        category="ai",
        state=STATE_STUB,
        docs="ROADMAP.md",
        routes=["/video/upscale/flashvsr"],
    ),
    FeatureRecord(
        feature_id="video.inpaint.rose",
        label="ROSE shadow-aware inpaint",
        category="ai",
        state=STATE_STUB,
        docs="ROADMAP.md",
        routes=["/video/inpaint/rose"],
    ),
    FeatureRecord(
        feature_id="ai.video-agent",
        label="Semantic search + storyboard generation",
        category="llm",
        state=STATE_AVAILABLE,
        install_hint="pip install transformers torch (for CLIP search); core storyboard is bundled",
        docs="docs/MODELS.md#videoagent--vimax-agentic-search",
        routes=["/agent/search-footage", "/agent/storyboard"],
        probe=_check("check_video_agent_available"),
        hardware="cpu/gpu",
    ),
    FeatureRecord(
        feature_id="ai.video.cloud",
        label="Cloud gen-video",
        category="generation",
        state=STATE_STUB,
        install_hint="API key required - opt-in only",
        docs="docs/MODELS.md#cloud-gen-video-hailuo--seedance",
        routes=[
            "/generate/cloud/backends",
            "/generate/cloud/status/<eid>",
            "/generate/cloud/submit",
        ],
        probe=_check("check_gen_video_cloud_available"),
        hardware="cpu (client)",
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
    # ---- Wave-L generative models -------------------------------------
    FeatureRecord(
        feature_id="audio.acestep",
        label="ACE-Step music generation",
        category="audio",
        state=STATE_AVAILABLE,
        install_hint="pip install acestep",
        routes=["/audio/music/acestep"],
        probe=_check("check_acestep_available"),
        hardware="gpu (>= 8 GB VRAM)",
        requires_gpu=True,
        minimum_vram_mb=8192,
    ),
    FeatureRecord(
        feature_id="audio.chatterbox",
        label="Chatterbox TTS (voice clone)",
        category="audio",
        state=STATE_AVAILABLE,
        install_hint="pip install chatterbox-tts",
        routes=["/audio/tts/chatterbox"],
        probe=_check("check_chatterbox_tts_available"),
        hardware="gpu",
        requires_gpu=True,
    ),
    FeatureRecord(
        feature_id="audio.diffrhythm",
        label="DiffRhythm full-song generation",
        category="audio",
        state=STATE_AVAILABLE,
        install_hint="git clone https://github.com/ASLP-lab/DiffRhythm",
        routes=["/audio/music/diffrhythm"],
        probe=_check("check_diffrhythm_available"),
        hardware="gpu (>= 8 GB VRAM)",
        requires_gpu=True,
        minimum_vram_mb=8192,
    ),
    FeatureRecord(
        feature_id="audio.kokoro",
        label="Kokoro TTS (82M, CPU-only)",
        category="audio",
        state=STATE_AVAILABLE,
        install_hint="pip install kokoro",
        routes=["/audio/tts/kokoro"],
        probe=_check("check_kokoro_available"),
        hardware="cpu",
    ),
    FeatureRecord(
        feature_id="audio.sparktts",
        label="Spark-TTS zero-shot voice clone",
        category="audio",
        state=STATE_AVAILABLE,
        install_hint="pip install sparktts",
        routes=["/audio/tts/spark"],
        probe=_check("check_sparktts_available"),
        hardware="cpu",
    ),
    FeatureRecord(
        feature_id="audio.moonshine",
        label="Moonshine ASR (CPU-optimized)",
        category="audio",
        state=STATE_AVAILABLE,
        install_hint="pip install moonshine",
        routes=["/audio/transcribe/moonshine"],
        probe=_check("check_moonshine_available"),
        hardware="cpu",
    ),
    FeatureRecord(
        feature_id="video.framepack",
        label="FramePack image-to-video",
        category="video",
        state=STATE_AVAILABLE,
        install_hint="pip install framepack",
        routes=["/generate/framepack"],
        probe=_check("check_framepack_available"),
        hardware="gpu (>= 6 GB VRAM)",
        requires_gpu=True,
        minimum_vram_mb=6144,
    ),
    FeatureRecord(
        feature_id="image.kontext",
        label="FLUX.1 Kontext image editing",
        category="ai",
        state=STATE_AVAILABLE,
        install_hint="pip install diffusers>=0.29 torch transformers accelerate",
        routes=["/image/edit/kontext"],
        probe=_check("check_kontext_available"),
        hardware="gpu",
        requires_gpu=True,
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


def feature_check_name(record: FeatureRecord) -> str:
    """Return the public check/probe name attached to a feature record."""
    return _probe_name(record)


def features_by_check_name(
    records: Iterable[FeatureRecord] = (),
) -> Dict[str, FeatureRecord]:
    """Return feature records keyed by their public check/probe name."""
    sample = list(records) or list(FEATURES.values())
    out: Dict[str, FeatureRecord] = {}
    for record in sample:
        check_name = _probe_name(record)
        if check_name:
            out.setdefault(check_name, record)
    return out


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
        hardware=str(payload.get("hardware") or ""),
        requires_gpu=bool(payload.get("requires_gpu") or False),
        minimum_vram_mb=max(0, int(payload.get("minimum_vram_mb") or 0)),
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

    by_id = {record.feature_id: record for record in records}
    existing_ids = set(by_id)
    for generated in generated_records:
        probe_name = _probe_name(generated)
        target = by_probe.get(probe_name) or by_id.get(generated.feature_id)
        if target is not None:
            target.routes = _unique_routes([*target.routes, *generated.routes])
            if target.check_name == "":
                target.check_name = probe_name
            if not target.hardware and generated.hardware:
                target.hardware = generated.hardware
            if not target.requires_gpu and generated.requires_gpu:
                target.requires_gpu = generated.requires_gpu
            if not target.minimum_vram_mb and generated.minimum_vram_mb:
                target.minimum_vram_mb = generated.minimum_vram_mb
            continue
        if generated.feature_id in existing_ids:
            continue
        records.append(generated)
        existing_ids.add(generated.feature_id)
        by_id[generated.feature_id] = generated
        if probe_name and probe_name not in by_probe:
            by_probe[probe_name] = generated
    return records


def _enrich_from_model_cards(records: Iterable[FeatureRecord]) -> List[FeatureRecord]:
    """Fill privacy/license/advisory_notes from model cards where available."""
    from opencut.model_cards import CARDS

    cards_by_check = {card.check_name: card for card in CARDS}
    cards_by_fid = {card.feature_id: card for card in CARDS}
    out = list(records)
    for record in out:
        card = cards_by_check.get(_probe_name(record)) or cards_by_fid.get(
            record.feature_id
        )
        if card is None:
            continue
        if not record.privacy:
            record.privacy = card.privacy
        if not record.license:
            record.license = card.license
        if not record.advisory_notes and card.advisory_notes:
            record.advisory_notes = list(card.advisory_notes)
    return out


def _build_index(records: Iterable[FeatureRecord]) -> Dict[str, FeatureRecord]:
    out: Dict[str, FeatureRecord] = {}
    for record in records:
        if record.feature_id in out:
            raise ValueError(f"duplicate feature_id: {record.feature_id!r}")
        out[record.feature_id] = record
    return out


_GENERATED_FEATURES = load_generated_feature_records()

_FEATURES_MERGED = _merge_generated_records(_FEATURES, _GENERATED_FEATURES)

_ENRICHED = False


def _ensure_enriched() -> None:
    global _ENRICHED
    if _ENRICHED:
        return
    _ENRICHED = True
    _enrich_from_model_cards(list(FEATURES.values()))


# Public, immutable view of the catalogue.
FEATURES: Dict[str, FeatureRecord] = _build_index(_FEATURES_MERGED)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_features() -> List[FeatureRecord]:
    """Return every registered feature in insertion order."""
    _ensure_enriched()
    return list(FEATURES.values())


def get_feature(feature_id: str) -> Optional[FeatureRecord]:
    _ensure_enriched()
    return FEATURES.get(feature_id)


def feature_states() -> Dict[str, ReadinessState]:
    """Return ``{feature_id: resolved_state}`` for fast manifest dumps."""
    _ensure_enriched()
    return {fid: rec.resolved_state() for fid, rec in FEATURES.items()}


def feature_manifest() -> dict:
    """Build the JSON payload served by ``GET /system/feature-state``."""
    _ensure_enriched()
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
