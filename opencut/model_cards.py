"""Model + license cards for OpenCut's optional AI extras (F115).

Why hand-written: each row needs deliberate metadata (license, hardware
requirement, privacy posture, install hint) that nobody but a human can
derive from the codebase. The list lives in this module so it's
reviewable in PRs, then ``opencut.tools.dump_model_cards`` renders it
to JSON and Markdown.

A test (``tests/test_model_cards.py``) asserts:

* Every ``check_X_available()`` in ``opencut.checks`` either has a card
  here or is on the explicit ``NON_AI_CHECKS`` list (infrastructure
  guards that don't ship an AI/model dependency).
* No two cards share a ``check_name``.
* Every card declares a license + privacy field.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, List, Optional

from opencut.registry import NON_AI_CHECKS


@dataclass
class ModelCard:
    """A single optional AI / model dependency."""

    check_name: str                  # ``check_<name>_available`` matched in opencut.checks
    feature_id: str                  # matches a row in opencut.registry where applicable
    label: str
    category: str                    # "audio" | "video" | "captions" | "vision" | "llm" | "lipsync" | "generation" | "interchange"
    license: str                     # short SPDX-friendly token
    upstream: str                    # canonical URL
    hardware: str                    # "cpu" | "gpu" | "cpu/gpu" | "gpu (>= NN GB VRAM)"
    install_hint: str                # one-line pip / system instruction
    privacy: str                     # "local-only" | "local + optional cloud" | "cloud"
    latency: str = "n/a"             # rough order-of-magnitude per minute of media
    quality_notes: str = ""
    requires_checkpoint_env: Optional[str] = None
    advisory_notes: List[str] = field(default_factory=list)
    # When non-empty, downgrades a denied-license error to a warning. Must
    # contain a written justification (reviewed by the maintainer).
    license_waiver: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


# Checks that don't gate a model — infrastructure guards, codecs, dev tools.
# These are deliberately excluded from the model-card surface. The canonical
# allowlist lives in ``opencut.registry`` so F191 readiness derivation and F115
# model-card validation share the same taxonomy.


# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------
#
# Kept short and dense — each row is read in PRs to confirm the license
# and privacy posture before merging. Add new rows alphabetically within
# their category.


CARDS: List[ModelCard] = [
    # ---- Audio --------------------------------------------------------
    ModelCard(
        check_name="check_demucs_available",
        feature_id="audio.demucs",
        label="Demucs (htdemucs / hdemucs)",
        category="audio",
        license="MIT",
        upstream="https://github.com/facebookresearch/demucs",
        hardware="cpu/gpu",
        install_hint="pip install demucs",
        privacy="local-only",
        latency="~0.3x realtime on CPU; near-real on RTX 3060",
        quality_notes="Strong baseline; superseded by BS-RoFormer in newer benches.",
    ),
    ModelCard(
        check_name="check_audiocraft_available",
        feature_id="audio.audiocraft",
        label="AudioCraft / MusicGen",
        category="audio",
        license="MIT",
        upstream="https://github.com/facebookresearch/audiocraft",
        hardware="gpu (>= 8 GB VRAM)",
        install_hint='pip install "opencut[music]" (Python 3.11; Torch 2.1 stack)',
        privacy="local-only",
        latency="~1x realtime on RTX 4060+ for 30 s clips",
    ),
    ModelCard(
        check_name="check_edge_tts_available",
        feature_id="audio.edge-tts",
        label="Edge TTS (Microsoft cloud voices)",
        category="audio",
        license="MIT (client)",
        upstream="https://github.com/rany2/edge-tts",
        hardware="cpu",
        install_hint="pip install edge-tts",
        privacy="cloud — text is sent to Microsoft's Speech API",
        latency="seconds per sentence",
        advisory_notes=[
            "Sends synthesis text to Microsoft over HTTPS. Disable for offline-only deployments.",
        ],
    ),
    ModelCard(
        check_name="check_f5_tts_available",
        feature_id="audio.f5-tts",
        label="F5-TTS (zero-shot voice clone)",
        category="audio",
        license="MIT",
        upstream="https://github.com/SWivid/F5-TTS",
        hardware="gpu (>= 6 GB VRAM)",
        install_hint="pip install f5-tts",
        privacy="local-only",
        latency="~1-2x realtime",
    ),
    ModelCard(
        check_name="check_resemble_enhance_available",
        feature_id="audio.resemble-enhance",
        label="Resemble Enhance",
        category="audio",
        license="MIT",
        upstream="https://github.com/resemble-ai/resemble-enhance",
        hardware="gpu",
        install_hint='pip install "opencut[enhance]" (Python 3.11; Torch 2.1 stack)',
        privacy="local-only",
    ),
    ModelCard(
        check_name="check_beatnet_available",
        feature_id="audio.beatnet",
        label="BeatNet beat/downbeat tracker",
        category="audio",
        license="MIT",
        upstream="https://github.com/mjhydri/BeatNet",
        hardware="cpu",
        install_hint="pip install BeatNet",
        privacy="local-only",
    ),
    ModelCard(
        check_name="check_silero_vad_available",
        feature_id="captions.silero-vad",
        label="Silero VAD",
        category="audio",
        license="MIT",
        upstream="https://github.com/snakers4/silero-vad",
        hardware="cpu",
        install_hint="pip install silero-vad",
        privacy="local-only",
        latency="real-time on CPU",
    ),
    # ---- Captions / Transcripts ---------------------------------------
    ModelCard(
        check_name="check_crisper_whisper_available",
        feature_id="captions.crisper-whisper",
        label="CrisperWhisper",
        category="captions",
        license="Apache-2.0",
        upstream="https://github.com/nyrahealth/CrisperWhisper",
        hardware="gpu",
        install_hint="pip install ctranslate2 + crisper-whisper",
        privacy="local-only",
    ),
    ModelCard(
        check_name="check_multimodal_diarize_available",
        feature_id="captions.multimodal-diarise",
        label="Multimodal diarisation pipeline",
        category="captions",
        license="MIT",
        upstream="https://github.com/m-bain/whisperX",
        hardware="gpu",
        install_hint="pip install whisperx + pyannote.audio (HF token required)",
        privacy="local-only (HF token used for weights download only)",
        advisory_notes=[
            "pyannote checkpoints are gated by Hugging Face acceptance — set HUGGINGFACE_HUB_TOKEN.",
        ],
    ),
    # ---- Video --------------------------------------------------------
    ModelCard(
        check_name="check_upscale_available",
        feature_id="video.upscale.realesrgan",
        label="Real-ESRGAN",
        category="video",
        license="BSD-3-Clause",
        upstream="https://github.com/xinntao/Real-ESRGAN",
        hardware="gpu",
        install_hint="pip install realesrgan",
        privacy="local-only",
        latency="~0.2-0.4x realtime on RTX 3060",
    ),
    ModelCard(
        check_name="check_rvm_available",
        feature_id="video.matte.rvm",
        label="Robust Video Matting (RVM)",
        category="video",
        license="GPL-3.0",
        upstream="https://github.com/PeterL1n/RobustVideoMatting",
        hardware="gpu",
        install_hint="pip install robust-video-matting",
        privacy="local-only",
        advisory_notes=[
            "GPL-3.0 — bundling the model with proprietary distributions requires legal review.",
        ],
        license_waiver=(
            "User installs separately via `pip install robust-video-matting`; OpenCut "
            "(MIT) invokes the GPL-3.0 model at runtime only, does not redistribute the "
            "weights or library, and the feature is explicitly opt-in. Downstream "
            "redistributors must perform their own license review."
        ),
    ),
    ModelCard(
        check_name="check_depth_available",
        feature_id="video.depth",
        label="Depth Anything V2",
        category="video",
        license="Apache-2.0",
        upstream="https://github.com/DepthAnything/Depth-Anything-V2",
        hardware="gpu",
        install_hint="pip install transformers torch",
        privacy="local-only",
    ),
    ModelCard(
        check_name="check_propainter_available",
        feature_id="video.propainter",
        label="ProPainter",
        category="video",
        license="NTU S-Lab License 1.0",
        upstream="https://github.com/sczhou/ProPainter",
        hardware="gpu (>= 12 GB VRAM)",
        install_hint="pip install propainter",
        privacy="local-only",
        advisory_notes=[
            "Custom S-Lab license — non-commercial use only. Verify before shipping in paid distributions.",
        ],
        license_waiver=(
            "Non-commercial source — guarded behind an opt-in check_propainter_available() "
            "gate. The default OpenCut install does not include ProPainter; users must "
            "explicitly `pip install propainter` and accept the NTU S-Lab license at that "
            "point. Maintainers must remove this waiver before any commercial distribution."
        ),
    ),
    ModelCard(
        check_name="check_sam2_available",
        feature_id="video.sam2",
        label="Segment Anything 2 (SAM2)",
        category="video",
        license="Apache-2.0",
        upstream="https://github.com/facebookresearch/sam2",
        hardware="gpu",
        install_hint="pip install sam2",
        privacy="local-only",
    ),
    ModelCard(
        check_name="check_rembg_available",
        feature_id="video.rembg",
        label="rembg (U^2-Net family)",
        category="video",
        license="MIT",
        upstream="https://github.com/danielgatis/rembg",
        hardware="cpu/gpu",
        install_hint="pip install rembg",
        privacy="local-only",
    ),
    ModelCard(
        check_name="check_watermark_available",
        feature_id="video.watermark.remove",
        label="LaMa inpainting watermark removal",
        category="video",
        license="Apache-2.0",
        upstream="https://github.com/advimman/lama",
        hardware="gpu",
        install_hint="pip install simple-lama-inpainting",
        privacy="local-only",
    ),
    ModelCard(
        check_name="check_scenedetect_available",
        feature_id="video.scenes.detect",
        label="PySceneDetect",
        category="video",
        license="BSD-3-Clause",
        upstream="https://github.com/Breakthrough/PySceneDetect",
        hardware="cpu",
        install_hint="pip install scenedetect",
        privacy="local-only",
    ),
    ModelCard(
        check_name="check_transnetv2_available",
        feature_id="video.scenes.transnetv2",
        label="TransNetV2 shot-boundary",
        category="video",
        license="MIT",
        upstream="https://github.com/allenday/transnetv2_pytorch",
        hardware="gpu",
        install_hint="pip install transnetv2-pytorch",
        privacy="local-only",
    ),
    ModelCard(
        check_name="check_clip_iqa_available",
        feature_id="video.quality.clip-iqa",
        label="CLIP-IQA+",
        category="video",
        license="Apache-2.0",
        upstream="https://github.com/IceClear/CLIP-IQA",
        hardware="gpu",
        install_hint="pip install pyiqa",
        privacy="local-only",
    ),
    ModelCard(
        check_name="check_hsemotion_available",
        feature_id="video.emotion-arc",
        label="HSEmotion",
        category="video",
        license="Apache-2.0",
        upstream="https://github.com/av-savchenko/face-emotion-recognition",
        hardware="cpu/gpu",
        install_hint="pip install hsemotion",
        privacy="local-only",
    ),
    ModelCard(
        check_name="check_deepface_available",
        feature_id="video.deepface",
        label="DeepFace",
        category="video",
        license="MIT",
        upstream="https://github.com/serengil/deepface",
        hardware="cpu/gpu",
        install_hint="pip install deepface",
        privacy="local-only",
    ),
    ModelCard(
        check_name="check_mediapipe_available",
        feature_id="video.mediapipe",
        label="MediaPipe (face mesh, hands, pose)",
        category="video",
        license="Apache-2.0",
        upstream="https://github.com/google-ai-edge/mediapipe",
        hardware="cpu/gpu",
        install_hint="pip install mediapipe",
        privacy="local-only",
    ),
    ModelCard(
        check_name="check_auto_editor_available",
        feature_id="editing.auto-editor",
        label="auto-editor",
        category="editing",
        license="Unlicense (public domain)",
        upstream="https://github.com/WyattBlue/auto-editor",
        hardware="cpu",
        install_hint="pip install auto-editor",
        privacy="local-only",
    ),
    ModelCard(
        check_name="check_auto_zoom_available",
        feature_id="editing.auto-zoom",
        label="OpenCV / face-tracker auto-zoom",
        category="editing",
        license="Apache-2.0",
        upstream="https://github.com/opencv/opencv-python",
        hardware="cpu",
        install_hint="pip install opencv-python",
        privacy="local-only",
    ),
    ModelCard(
        check_name="check_broll_generate_available",
        feature_id="editing.broll-generate",
        label="B-roll suggestion pipeline",
        category="editing",
        license="MIT (orchestration; underlying providers vary)",
        upstream="https://github.com/openai/whisper",
        hardware="cpu/gpu",
        install_hint="pip install transformers torch",
        privacy="local + optional cloud (LLM dependency)",
    ),
    # ---- LLM ----------------------------------------------------------
    ModelCard(
        check_name="check_llm_available",
        feature_id="llm.local",
        label="Local LLM provider (any of: Ollama, vLLM, OpenAI-compatible)",
        category="llm",
        license="varies per backend",
        upstream="https://github.com/ollama/ollama",
        hardware="gpu (recommended)",
        install_hint="install Ollama or set OPENAI_API_KEY",
        privacy="local-only when using Ollama / vLLM; cloud otherwise",
        advisory_notes=[
            "Configure OPENCUT_LLM_PROVIDER=ollama to force local; otherwise cloud providers can be hit.",
        ],
    ),
    ModelCard(
        check_name="check_ollama_available",
        feature_id="llm.ollama",
        label="Ollama backend",
        category="llm",
        license="MIT",
        upstream="https://github.com/ollama/ollama",
        hardware="cpu/gpu",
        install_hint="install Ollama from https://ollama.ai",
        privacy="local-only",
    ),
    # ---- TTS / Voice (extras beyond audio bucket) --------------------
    ModelCard(
        check_name="check_elevenlabs_available",
        feature_id="audio.elevenlabs",
        label="ElevenLabs cloud TTS",
        category="audio",
        license="proprietary client SDK; cloud service",
        upstream="https://github.com/elevenlabs/elevenlabs-python",
        hardware="cpu (client)",
        install_hint="pip install elevenlabs + ELEVENLABS_API_KEY",
        privacy="cloud — text is sent to ElevenLabs",
        advisory_notes=[
            "Requires an ElevenLabs account. Synthesised audio + reference voices may be retained per their TOS.",
        ],
    ),
    ModelCard(
        check_name="check_omnivoice_available",
        feature_id="audio.omnivoice",
        label="OmniVoice TTS",
        category="audio",
        license="Apache-2.0",
        upstream="https://github.com/k2-fsa/OmniVoice",
        hardware="gpu",
        install_hint="pip install omnivoice",
        privacy="local-only",
    ),
    ModelCard(
        check_name="check_voice_grammar_available",
        feature_id="audio.voice-grammar",
        label="WhisperX voice-command grammar",
        category="audio",
        license="MIT",
        upstream="https://github.com/linto-ai/whisper-timestamped",
        hardware="cpu/gpu",
        install_hint="pip install whisperx + faster-whisper",
        privacy="local-only",
    ),
    # ---- Restoration / Enhancement -----------------------------------
    ModelCard(
        check_name="check_ddcolor_available",
        feature_id="video.colorize.ddcolor",
        label="DDColor",
        category="video",
        license="Apache-2.0",
        upstream="https://github.com/piddnad/DDColor",
        hardware="gpu",
        install_hint="pip install ddcolor",
        privacy="local-only",
    ),
    ModelCard(
        check_name="check_vrt_available",
        feature_id="video.restore.vrt",
        label="VRT / RVRT video restoration",
        category="video",
        license="Apache-2.0",
        upstream="https://github.com/JingyunLiang/VRT",
        hardware="gpu (>= 16 GB VRAM for VRT)",
        install_hint="pip install vrt-restoration",
        privacy="local-only",
    ),
    ModelCard(
        check_name="check_neural_deflicker_available",
        feature_id="video.restore.deflicker",
        label="All-In-One-Deflicker",
        category="video",
        license="MIT",
        upstream="https://github.com/ChenyangLEI/All-In-One-Deflicker",
        hardware="gpu",
        install_hint="pip install all-in-one-deflicker",
        privacy="local-only",
    ),
    # ---- Roadmap stubs (still placeholder until implemented) ---------
    ModelCard(
        check_name="check_flashvsr_available",
        feature_id="video.upscale.flashvsr",
        label="FlashVSR (streaming VSR)",
        category="video",
        license="Apache-2.0",
        upstream="https://github.com/OpenImagingLab/FlashVSR",
        hardware="gpu (>= 12 GB VRAM)",
        install_hint="pip install flashvsr (stub — roadmap H2.1)",
        privacy="local-only",
        advisory_notes=["Roadmap stub; check returns False until the implementation lands."],
    ),
    ModelCard(
        check_name="check_rose_available",
        feature_id="video.inpaint.rose",
        label="ROSE shadow-aware inpaint",
        category="video",
        license="Apache-2.0",
        upstream="https://rose2025-inpaint.github.io/",
        hardware="gpu",
        install_hint="pip install rose-inpaint (stub — roadmap H2.2)",
        privacy="local-only",
        advisory_notes=["Roadmap stub."],
    ),
    ModelCard(
        check_name="check_sammie_available",
        feature_id="video.matte.sammie",
        label="Sammie-Roto-2 (VideoMaMa)",
        category="video",
        license="MIT",
        upstream="https://github.com/Zarxrax/Sammie-Roto-2",
        hardware="gpu",
        install_hint="pip install sammie-roto (stub — roadmap H2.3)",
        privacy="local-only",
        advisory_notes=["Roadmap stub."],
    ),
    ModelCard(
        check_name="check_reezsynth_available",
        feature_id="video.style.reezsynth",
        label="ReEzSynth (Ebsynth successor)",
        category="video",
        license="Apache-2.0",
        upstream="https://github.com/FuouM/ReEzSynth",
        hardware="gpu",
        install_hint="pip install reezsynth (stub — roadmap H2.5)",
        privacy="local-only",
        advisory_notes=["Roadmap stub."],
    ),
    ModelCard(
        check_name="check_vidmuse_available",
        feature_id="audio.music.vidmuse",
        label="VidMuse video-to-music",
        category="audio",
        license="Apache-2.0",
        upstream="https://vidmuse.github.io/",
        hardware="gpu",
        install_hint="pip install vidmuse (stub — roadmap H2.6)",
        privacy="local-only",
        advisory_notes=["Roadmap stub."],
    ),
    ModelCard(
        check_name="check_video_agent_available",
        feature_id="ai.video-agent",
        label="VideoAgent / ViMax agentic search",
        category="llm",
        license="MIT",
        upstream="https://github.com/HKUDS/VideoAgent",
        hardware="cpu/gpu",
        install_hint="pip install videoagent (stub — roadmap H3.1)",
        privacy="local + optional cloud (depends on configured LLM)",
        advisory_notes=["Roadmap stub. LLM dependency may route through a cloud provider."],
    ),
    ModelCard(
        check_name="check_gen_video_cloud_available",
        feature_id="ai.video.cloud",
        label="Cloud gen-video (Hailuo / Seedance)",
        category="generation",
        license="proprietary cloud service",
        upstream="https://hailuo-02.com",
        hardware="cpu (client)",
        install_hint="API key required — opt-in only",
        privacy="cloud — prompts + reference frames leave the machine",
        advisory_notes=[
            "Disabled by default. Set HAILUO_API_KEY / SEEDANCE_API_KEY to enable.",
        ],
    ),
    ModelCard(
        check_name="check_lipsync_advanced_available",
        feature_id="lipsync.advanced",
        label="GaussianHeadTalk / FantasyTalking2",
        category="lipsync",
        license="Apache-2.0",
        upstream="https://wacv.thecvf.com/",
        hardware="gpu (>= 12 GB VRAM)",
        install_hint="pip install gaussian-head-talk + fantasytalking2 (stubs)",
        privacy="local-only",
        advisory_notes=["Roadmap stub (H3.3)."],
    ),
    ModelCard(
        check_name="check_pyonfx_available",
        feature_id="captions.karaoke-adv",
        label="pyonfx ASS karaoke",
        category="captions",
        license="LGPL-3.0",
        upstream="https://github.com/CoffeeStraw/PyonFX",
        hardware="cpu",
        install_hint="pip install pyonfx",
        privacy="local-only",
        advisory_notes=[
            "LGPL — using as a library inside a proprietary distribution requires legal review.",
        ],
    ),
    # ---- Vision niche --------------------------------------------------
    ModelCard(
        check_name="check_colour_science_available",
        feature_id="video.colour-science",
        label="colour-science (CIE / vectorscope math)",
        category="video",
        license="BSD-3-Clause",
        upstream="https://github.com/colour-science/colour",
        hardware="cpu",
        install_hint="pip install colour-science",
        privacy="local-only",
    ),
    ModelCard(
        check_name="check_face_reshape_available",
        feature_id="video.face-reshape",
        label="Face reshape / beauty filter",
        category="video",
        license="MIT",
        upstream="https://github.com/google-ai-edge/mediapipe",
        hardware="cpu/gpu",
        install_hint="pip install mediapipe + opencv-python",
        privacy="local-only",
    ),
    ModelCard(
        check_name="check_skin_retouch_available",
        feature_id="video.skin-retouch",
        label="Skin retouch (bilateral / GAN)",
        category="video",
        license="MIT",
        upstream="https://github.com/opencv/opencv-python",
        hardware="cpu/gpu",
        install_hint="pip install opencv-python (optional onnx model)",
        privacy="local-only",
    ),
    ModelCard(
        check_name="check_upscale_hub_available",
        feature_id="video.upscale.hub",
        label="HF upscale model hub",
        category="video",
        license="varies per model",
        upstream="https://huggingface.co/models?other=image-super-resolution",
        hardware="gpu",
        install_hint="pip install diffusers transformers",
        privacy="local-only (model downloads only)",
        advisory_notes=["Each Hugging Face model has its own license — check before redistributing weights."],
    ),
    ModelCard(
        check_name="check_virality_score_available",
        feature_id="analyze.virality",
        label="Virality / hook scorer",
        category="llm",
        license="MIT (orchestration)",
        upstream="https://github.com/SysAdminDoc/OpenCut",
        hardware="cpu",
        install_hint="ships with core; optional LLM dependency",
        privacy="local + optional cloud LLM",
        advisory_notes=[
            "Falls back to a keyword lexicon when no LLM provider is configured.",
        ],
    ),
    # ---- Wave-L generative models -------------------------------------
    ModelCard(
        check_name="check_acestep_available",
        feature_id="audio.acestep",
        label="ACE-Step (full-song music with lyrics)",
        category="audio",
        license="Apache-2.0",
        upstream="https://github.com/ACEStudio/ACE-Step",
        hardware="gpu (>= 8 GB VRAM)",
        install_hint="pip install acestep",
        privacy="local-only",
        quality_notes="3.5B model; ~1 min of music in seconds on an RTX 3090.",
    ),
    ModelCard(
        check_name="check_chatterbox_tts_available",
        feature_id="audio.chatterbox",
        label="Chatterbox TTS (emotional voice clone)",
        category="audio",
        license="MIT",
        upstream="https://github.com/resemble-ai/chatterbox",
        hardware="gpu",
        install_hint="pip install chatterbox-tts",
        privacy="local-only",
        quality_notes="Zero-shot voice cloning from a 10 s clip; multilingual variant covers 23 languages.",
    ),
    ModelCard(
        check_name="check_diffrhythm_available",
        feature_id="audio.diffrhythm",
        label="DiffRhythm (diffusion full-song generator)",
        category="audio",
        license="Apache-2.0",
        upstream="https://github.com/ASLP-lab/DiffRhythm",
        hardware="gpu (>= 8 GB VRAM)",
        install_hint="git clone https://github.com/ASLP-lab/DiffRhythm && pip install -r DiffRhythm/requirements.txt",
        privacy="local-only",
        quality_notes="Accepts LRC lyrics + optional style audio; base model up to 1m35s, full up to 4m45s.",
    ),
    ModelCard(
        check_name="check_kokoro_available",
        feature_id="audio.kokoro",
        label="Kokoro TTS (82M, CPU-only)",
        category="audio",
        license="Apache-2.0",
        upstream="https://github.com/hexgrad/kokoro",
        hardware="cpu",
        install_hint="pip install kokoro (needs espeak-ng for some languages)",
        privacy="local-only",
        latency="real-time on CPU",
        quality_notes="Ultralight last-resort TTS fallback for machines without a GPU.",
    ),
    ModelCard(
        check_name="check_sparktts_available",
        feature_id="audio.sparktts",
        label="Spark-TTS (CPU-native zero-shot)",
        category="audio",
        license="Apache-2.0",
        upstream="https://github.com/SparkAudio/Spark-TTS",
        hardware="cpu",
        install_hint="pip install sparktts",
        privacy="local-only",
    ),
    ModelCard(
        check_name="check_moonshine_available",
        feature_id="audio.moonshine",
        label="Moonshine ASR (CPU-optimized STT)",
        category="captions",
        license="MIT (English models)",
        upstream="https://github.com/usefulsensors/moonshine",
        hardware="cpu",
        install_hint="pip install moonshine",
        privacy="local-only",
        advisory_notes=[
            "Multilingual models use a community (non-commercial) license and are gated separately.",
        ],
    ),
    ModelCard(
        check_name="check_framepack_available",
        feature_id="video.framepack",
        label="FramePack (image-to-video diffusion)",
        category="generation",
        license="Apache-2.0",
        upstream="https://github.com/lllyasviel/FramePack",
        hardware="gpu (>= 6 GB VRAM)",
        install_hint="pip install framepack",
        privacy="local-only",
    ),
    ModelCard(
        check_name="check_kontext_available",
        feature_id="image.kontext",
        label="FLUX.1 Kontext-dev (image editing)",
        category="generation",
        license="Apache-2.0 (dev variant)",
        upstream="https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev",
        hardware="gpu",
        install_hint="pip install diffusers>=0.29 torch transformers accelerate",
        privacy="local-only",
        advisory_notes=[
            "Kontext-dev weights (~24 GB) download on first use; the pro variant is commercial and is not used.",
        ],
    ),
]


def list_cards() -> List[ModelCard]:
    return list(CARDS)


def cards_by_check_name() -> dict:
    return {card.check_name: card for card in CARDS}


def manifest() -> dict:
    """Build the JSON manifest written to ``opencut/_generated/model_cards.json``."""
    payload = sorted([card.as_dict() for card in CARDS], key=lambda c: c["check_name"])
    return {
        "version": 1,
        "total": len(payload),
        "cards": payload,
        "non_ai_checks": sorted(NON_AI_CHECKS),
    }


def assert_card_invariants(cards: Iterable[ModelCard] = ()) -> None:
    """Catch the most common authoring mistakes before they ship."""
    sample = list(cards) or list(CARDS)
    seen = set()
    for card in sample:
        if not card.check_name:
            raise ValueError(f"card {card.label!r} has empty check_name")
        if card.check_name in seen:
            raise ValueError(f"duplicate check_name in CARDS: {card.check_name!r}")
        seen.add(card.check_name)
        if not card.license:
            raise ValueError(f"card {card.check_name!r} missing license")
        if not card.privacy:
            raise ValueError(f"card {card.check_name!r} missing privacy posture")
        if not card.upstream.startswith("http"):
            raise ValueError(f"card {card.check_name!r} upstream is not a URL")
