"""
OpenCut MCP Server v1.30.0

Exposes OpenCut's API as a Model Context Protocol (MCP) server,
allowing AI clients (Claude Code, Cursor, etc.) to drive video
editing operations programmatically.

Usage:
    opencut-mcp-server                          # stdio JSON-RPC (default)
    opencut-mcp-server --http                   # HTTP transport on port 5681
    opencut-mcp-server --port 5679              # OpenCut backend on custom port
    opencut-mcp-server --list-tools             # Print all tools and exit
    python -m opencut.mcp_server                # same as stdio

The MCP server proxies requests to the running OpenCut Flask backend
on localhost:5679. Start the backend first:  opencut server

Protocol: JSON-RPC 2.0 over stdio (one JSON object per line).
Supports: initialize, tools/list, tools/call, resources/list, prompts/list.
"""

import argparse
import ipaddress
import json
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

from opencut import __version__, mcp_extended_tools
from opencut import auth as _auth

logger = logging.getLogger("opencut.mcp")

BACKEND_URL = "http://127.0.0.1:5679"
_csrf_token = ""
_csrf_fetched_at: float = 0.0
# CSRF tokens have a 1-hour TTL on the backend; refresh proactively 5 minutes
# before expiry so long-running MCP sessions never hit a 403 mid-operation.
_CSRF_TTL_SECONDS = 3300  # 55 minutes
_LOOPBACK_BINDS = {"localhost"}


def _csrf_is_fresh() -> bool:
    """Return True when the cached token is non-empty and not near expiry."""
    return bool(_csrf_token) and (time.time() - _csrf_fetched_at < _CSRF_TTL_SECONDS)


def _refresh_csrf():
    """Fetch a fresh CSRF token from the backend and record the fetch time."""
    global _csrf_token, _csrf_fetched_at
    try:
        req = urllib.request.Request(f"{BACKEND_URL}/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = json.loads(resp.read())
            token = body.get("csrf_token", "")
            if token:
                _csrf_token = token
                _csrf_fetched_at = time.time()
    except Exception as exc:
        logger.warning("CSRF refresh failed: %s", exc)


def _backend_requires_auth() -> bool:
    parsed = urllib.parse.urlsplit(BACKEND_URL)
    host = (parsed.hostname or "").strip().lower()
    if host == "localhost":
        return False
    try:
        return not ipaddress.ip_address(host).is_loopback
    except ValueError:
        return True


def _apply_backend_auth(headers: dict[str, str]) -> None:
    """Attach the current secret without caching it across file rotations."""
    if not _backend_requires_auth():
        return
    token = _auth.current_token()
    if token is not None:
        headers[_auth.AUTH_HEADER] = token.token


def _api(method, path, data=None):
    """Call the OpenCut Flask backend."""
    global _csrf_token

    if not _csrf_is_fresh():
        _refresh_csrf()

    url = f"{BACKEND_URL}{path}"
    headers = {"Content-Type": "application/json"}
    if _csrf_token:
        headers["X-OpenCut-Token"] = _csrf_token
    _apply_backend_auth(headers)

    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        # Retry once on 403 (stale CSRF token after backend restart)
        if e.code == 403 and _csrf_token:
            _refresh_csrf()
            headers["X-OpenCut-Token"] = _csrf_token
            _apply_backend_auth(headers)
            req2 = urllib.request.Request(url, data=body, headers=headers, method=method)
            try:
                with urllib.request.urlopen(req2, timeout=120) as resp2:
                    return json.loads(resp2.read())
            except Exception as retry_exc:
                return {"error": f"Retry failed after CSRF refresh: {retry_exc}"}
        error_body = e.read().decode(errors="replace")
        try:
            return json.loads(error_body)
        except json.JSONDecodeError:
            return {"error": f"HTTP {e.code}: {error_body[:200]}"}
    except Exception as e:
        return {"error": str(e)}


def _mcp_http_bind_requires_auth(bind: str) -> bool:
    addr = (bind or "").strip().lower().strip("[]")
    if not addr:
        return True
    if addr in _LOOPBACK_BINDS:
        return False
    try:
        return not ipaddress.ip_address(addr.split("%", 1)[0]).is_loopback
    except ValueError:
        return True


def _mcp_http_request_is_authorized(headers, *, auth_required: bool) -> bool:
    # Tokens are accepted from the X-OpenCut-Auth header only. A ?auth=
    # query fallback used to exist but leaked tokens into access logs and
    # history; MCP HTTP clients are plain JSON-RPC callers (no SSE), so
    # they can always set headers.
    if not auth_required:
        return True
    return _auth.is_token_valid(_auth.extract_request_token(headers))


# ---------------------------------------------------------------------------
# MCP Tool Definitions
# ---------------------------------------------------------------------------
MCP_TOOLS = [
    {
        "name": "opencut_transcribe",
        "description": "Transcribe video/audio to text with timestamps using Whisper AI",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to video/audio file"},
                "model": {"type": "string", "description": "Whisper model (tiny/base/small/medium/turbo/large-v3/distil-large-v3.5)", "default": "turbo"},
                "language": {"type": "string", "description": "Language code or empty for auto-detect", "default": ""},
            },
            "required": ["filepath"],
        },
    },
    {
        "name": "opencut_silence_remove",
        "description": "Detect and remove silent segments from video/audio",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to video/audio file"},
                "threshold": {"type": "number", "description": "Silence threshold in dB (-60 to 0)", "default": -30},
                "min_duration": {"type": "number", "description": "Minimum silence duration in seconds", "default": 0.5},
            },
            "required": ["filepath"],
        },
    },
    {
        "name": "opencut_export_video",
        "description": "Export/render video with specified codec and quality settings",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to source video"},
                "segments": {"type": "array", "description": "Segments to export [{start, end}]"},
                "video_codec": {"type": "string", "default": "libx264"},
                "quality": {"type": "string", "default": "medium"},
                "output_format": {"type": "string", "default": "mp4"},
            },
            "required": ["filepath", "segments"],
        },
    },
    {
        "name": "opencut_highlights",
        "description": "Extract highlight/viral clips using LLM analysis of transcript",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to video file"},
                "max_highlights": {"type": "integer", "default": 5},
                "use_vision": {"type": "boolean", "description": "Use vision-augmented analysis", "default": False},
            },
            "required": ["filepath"],
        },
    },
    {
        "name": "opencut_separate_audio",
        "description": "Separate audio into stems (vocals, drums, bass, etc.) using AI",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to audio/video file"},
                "backend": {"type": "string", "description": "Backend: demucs or audio-separator", "default": "audio-separator"},
                "stems": {"type": "array", "description": "Stems to extract", "default": ["vocals", "no_vocals"]},
            },
            "required": ["filepath"],
        },
    },
    {
        "name": "opencut_tts",
        "description": "Generate speech from text (edge-tts, Kokoro, or Chatterbox with voice cloning)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to synthesize"},
                "engine": {"type": "string", "description": "TTS engine: edge, kokoro, chatterbox", "default": "kokoro"},
                "voice": {"type": "string", "description": "Voice name/preset", "default": "af_heart"},
                "voice_ref": {"type": "string", "description": "Path to reference audio for Chatterbox voice cloning"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "opencut_style_transfer",
        "description": "Apply style transfer to video (preset styles or arbitrary reference image)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to video file"},
                "style": {"type": "string", "description": "Preset style name or 'arbitrary'"},
                "style_image": {"type": "string", "description": "Path to reference image (for arbitrary style)"},
                "intensity": {"type": "number", "default": 1.0},
            },
            "required": ["filepath"],
        },
    },
    {
        "name": "opencut_face_enhance",
        "description": "Enhance/restore faces in video using GFPGAN or CodeFormer",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to video file"},
                "model": {"type": "string", "description": "Model: gfpgan or codeformer", "default": "codeformer"},
                "fidelity": {"type": "number", "description": "CodeFormer fidelity (0=quality, 1=identity)", "default": 0.5},
            },
            "required": ["filepath"],
        },
    },
    {
        "name": "opencut_generate_music",
        "description": "Generate music with AI (MusicGen instrumental or ACE-Step with vocals+lyrics)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Music style/genre description"},
                "engine": {"type": "string", "description": "Engine: musicgen or ace-step", "default": "ace-step"},
                "lyrics": {"type": "string", "description": "Lyrics for ACE-Step vocal generation"},
                "duration": {"type": "number", "default": 30},
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "opencut_job_status",
        "description": "Check the status of a running OpenCut job",
        "inputSchema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "description": "Job ID to check"},
            },
            "required": ["job_id"],
        },
    },
    {
        "name": "opencut_repeat_detect",
        "description": "Detect and identify repeated/fumbled takes in a video or audio file using transcript similarity analysis.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file": {"type": "string", "description": "Path to media file"},
                "model": {"type": "string", "default": "base", "description": "Whisper model size"},
                "threshold": {"type": "number", "default": 0.6, "description": "Similarity threshold 0-1"},
            },
            "required": ["file"],
        },
    },
    {
        "name": "opencut_chapters",
        "description": "Generate YouTube chapter timestamps from a video transcript using LLM analysis.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file": {"type": "string", "description": "Path to media file"},
                "llm_provider": {"type": "string", "default": "ollama"},
                "llm_model": {"type": "string", "default": "llama3"},
                "api_key": {"type": "string", "default": ""},
                "max_chapters": {"type": "integer", "default": 15},
            },
            "required": ["file"],
        },
    },
    {
        "name": "opencut_footage_search",
        "description": "Search indexed media library by spoken content. Returns matching clips with timestamps.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "top_k": {"type": "integer", "default": 10, "description": "Max results"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "opencut_index_footage",
        "description": "Index media files for footage search by transcribing their spoken content.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "files": {"type": "array", "items": {"type": "string"}, "description": "List of file paths to index"},
                "model": {"type": "string", "default": "base"},
            },
            "required": ["files"],
        },
    },
    {
        "name": "opencut_color_match",
        "description": "Match the color profile of a source video clip to a reference clip using histogram matching.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Path to source video"},
                "reference": {"type": "string", "description": "Path to reference video"},
                "output_dir": {"type": "string", "description": "Output directory"},
                "strength": {"type": "number", "default": 1.0, "description": "Matching strength 0-1"},
            },
            "required": ["source", "reference"],
        },
    },
    {
        "name": "opencut_loudness_match",
        "description": "Normalize audio loudness (LUFS) across multiple clips to a target level.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "files": {"type": "array", "items": {"type": "string"}},
                "target_lufs": {"type": "number", "default": -14.0},
                "output_dir": {"type": "string"},
            },
            "required": ["files"],
        },
    },
    {
        "name": "opencut_auto_zoom",
        "description": "Generate face-detected zoom keyframes for a push-in zoom effect on talking-head clips.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file": {"type": "string"},
                "zoom_amount": {"type": "number", "default": 1.15},
                "easing": {"type": "string", "default": "ease_in_out", "enum": ["linear", "ease_in", "ease_out", "ease_in_out"]},
                "output_dir": {"type": "string"},
                "apply_to_file": {"type": "boolean", "default": False},
            },
            "required": ["file"],
        },
    },
    {
        "name": "opencut_multicam_cuts",
        "description": "Generate multicam cut decisions from speaker diarization — cuts to the camera angle assigned to whoever is speaking.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file": {"type": "string", "description": "Audio/video file with multiple speakers"},
                "speaker_count": {"type": "integer", "default": 2},
                "min_cut_duration": {"type": "number", "default": 1.0},
            },
            "required": ["file"],
        },
    },
    {
        "name": "opencut_denoise_audio",
        "description": "Reduce noise from audio/video using FFmpeg filters (afftdn, highpass, gate)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to audio/video file"},
                "method": {"type": "string", "description": "Denoise method: afftdn, highpass, gate", "default": "afftdn"},
                "strength": {"type": "number", "description": "Denoise strength 0-1", "default": 0.5},
            },
            "required": ["filepath"],
        },
    },
    {
        "name": "opencut_upscale",
        "description": "AI upscale video using Real-ESRGAN (2x or 4x resolution)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to video file"},
                "scale": {"type": "integer", "description": "Upscale factor: 1-4", "default": 2},
                "model": {"type": "string", "description": "Model: realesrgan-x4plus, realesrgan-x2plus", "default": "realesrgan-x4plus"},
            },
            "required": ["filepath"],
        },
    },
    {
        "name": "opencut_scene_detect",
        "description": "Detect scene boundaries/cuts in a video file",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to video file"},
                "method": {"type": "string", "description": "Detection method: ffmpeg, ml, pyscenedetect", "default": "ffmpeg"},
                "threshold": {"type": "number", "description": "Detection sensitivity 0-1", "default": 0.3},
            },
            "required": ["filepath"],
        },
    },
    {
        "name": "opencut_depth_map",
        "description": "Generate a depth map video using Depth Anything V2",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to video file"},
                "model_size": {"type": "string", "description": "Model size: small, base, large", "default": "small"},
            },
            "required": ["filepath"],
        },
    },
    {
        "name": "opencut_shorts_pipeline",
        "description": "One-click pipeline: transcribe + highlight + face-reframe + captions + export short-form clips",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to long-form video"},
                "max_shorts": {"type": "integer", "default": 5},
                "min_duration": {"type": "number", "default": 15},
                "max_duration": {"type": "number", "default": 60},
                "face_track": {"type": "boolean", "default": True},
                "burn_captions": {"type": "boolean", "default": True},
            },
            "required": ["filepath"],
        },
    },
    # Wave M additions (v1.30.0)
    {
        "name": "opencut_dub_video",
        "description": (
            "Automatically dub a video into a target language. "
            "Transcribes speech with Whisper, translates, synthesises dubbed "
            "voices (with optional voice cloning), and can apply lip-sync."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to source video"},
                "target_language": {"type": "string", "description": "BCP-47 code (e.g. 'es', 'fr', 'ja')", "default": "es"},
                "whisper_model": {"type": "string", "description": "Whisper model: tiny/base/small/medium/large", "default": "base"},
                "voice_clone": {"type": "boolean", "description": "Clone original speaker voice", "default": True},
                "lip_sync": {"type": "boolean", "description": "Apply lip-sync to dubbed video", "default": False},
                "preserve_music": {"type": "boolean", "description": "Keep background music track", "default": True},
                "tts_engine": {"type": "string", "description": "TTS engine: auto / kokoro / chatterbox / edge", "default": "auto"},
            },
            "required": ["filepath"],
        },
    },
    {
        "name": "opencut_sports_highlights",
        "description": (
            "Detect and extract highlight moments from sports or action video "
            "using optical-flow motion analysis combined with audio energy scoring."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to source video"},
                "genre": {"type": "string", "description": "Content genre: sports/concert/reaction/gaming/news", "default": "sports"},
                "top_n": {"type": "integer", "description": "Max highlights to return", "default": 5},
                "window_sec": {"type": "number", "description": "Highlight window duration in seconds", "default": 3.0},
                "min_score": {"type": "number", "description": "Minimum composite score threshold [0-1]", "default": 0.4},
            },
            "required": ["filepath"],
        },
    },
    {
        "name": "opencut_lipsync_echomimic",
        "description": (
            "Animate a portrait image or video to match spoken audio "
            "using EchoMimic diffusion-based lip-sync."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to portrait image or video"},
                "audio_path": {"type": "string", "description": "Path to audio file (WAV/MP3)"},
                "steps": {"type": "integer", "description": "Diffusion inference steps", "default": 20},
                "cfg_scale": {"type": "number", "description": "Classifier-free guidance scale", "default": 7.5},
            },
            "required": ["filepath", "audio_path"],
        },
    },
    {
        "name": "opencut_chat_edit",
        "description": (
            "Send a natural-language editing instruction to the OpenCut AI agent. "
            "The agent interprets the prompt and returns structured edit actions."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Natural-language editing instruction"},
                "session_id": {"type": "string", "description": "Conversation session ID for multi-turn context", "default": ""},
            },
            "required": ["message"],
        },
    },
    # F195 post-Wave-M shipped surface expansion
    {
        "name": "opencut_face_reshape",
        "description": "Apply AI face reshaping operations such as slim face, enlarge eyes, or smooth jaw.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to source video"},
                "operation": {"type": "string", "default": "slim_face"},
                "strength": {"type": "number", "default": 0.5},
                "output": {"type": "string", "description": "Optional output path"},
            },
            "required": ["filepath"],
        },
    },
    {
        "name": "opencut_skin_retouch",
        "description": "Retouch skin and blemishes in video using bilateral or GFPGAN-backed processing.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to source video"},
                "intensity": {"type": "number", "default": 0.6},
                "mode": {"type": "string", "default": "bilateral", "enum": ["bilateral", "gfpgan"]},
                "radiance": {"type": "number", "default": 0.0},
                "output": {"type": "string", "description": "Optional output path"},
            },
            "required": ["filepath"],
        },
    },
    {
        "name": "opencut_smart_upscale",
        "description": "Upscale video through OpenCut's smart backend selector with Lanczos fallback.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to source video"},
                "scale": {"type": "integer", "default": 2},
                "hint": {"type": "string", "default": "auto"},
                "backend": {"type": "string", "default": "auto"},
                "output": {"type": "string", "description": "Optional output path"},
            },
            "required": ["filepath"],
        },
    },
    {
        "name": "opencut_elevenlabs_tts",
        "description": "Generate speech through the optional ElevenLabs cloud TTS integration.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to synthesize"},
                "voice": {"type": "string", "default": "Rachel"},
                "model": {"type": "string", "default": "eleven_multilingual_v2"},
                "stability": {"type": "number", "default": 0.5},
                "similarity_boost": {"type": "number", "default": 0.75},
                "style": {"type": "number", "default": 0.0},
                "use_speaker_boost": {"type": "boolean", "default": True},
                "output": {"type": "string", "description": "Optional output path"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "opencut_caption_qc",
        "description": "Run the caption QC gate over SRT text or an SRT file.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "srt_path": {"type": "string", "description": "Path to SRT file"},
                "srt_text": {"type": "string", "description": "Inline SRT text"},
                "standard": {"type": "string", "default": "accessibility"},
                "mode": {"type": "string", "default": "strict", "enum": ["strict", "advisory"]},
            },
        },
    },
    {
        "name": "opencut_review_bundle",
        "description": "Build a portable review bundle zip with media, captions, markers, notes, and extra files.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "output_path": {"type": "string", "description": "Destination .zip path"},
                "job_label": {"type": "string", "description": "Human label for the bundle"},
                "media_path": {"type": "string", "description": "Optional rendered media path"},
                "captions_path": {"type": "string", "description": "Optional captions path"},
                "markers_payload": {"type": "object", "description": "Optional marker/comment payload"},
                "notes": {"type": "string", "description": "Free-form reviewer notes"},
                "extra_files": {"type": "array", "items": {"type": "string"}},
                "include_media": {"type": "boolean", "default": True},
            },
            "required": ["output_path"],
        },
    },
    {
        "name": "opencut_c2pa_provenance",
        "description": "Write a lightweight C2PA provenance sidecar next to a rendered asset.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "asset_path": {"type": "string", "description": "Rendered asset path"},
                "title": {"type": "string", "description": "Optional manifest title"},
                "ingredients": {"type": "array", "items": {"type": "object"}},
                "actions": {"type": "array", "items": {"type": "object"}},
                "claim_generator": {"type": "string", "description": "Optional claim-generator override"},
            },
            "required": ["asset_path"],
        },
    },
    {
        "name": "opencut_marker_import",
        "description": "Parse CSV, Premiere CSV, or EDL markers into OpenCut's normalized marker format.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to marker file"},
                "text": {"type": "string", "description": "Inline marker text"},
                "format": {"type": "string", "description": "csv, premiere_csv, or edl"},
                "fps": {"type": "number", "default": 30.0},
            },
        },
    },
    {
        "name": "opencut_capability_probe",
        "description": "Return OpenCut's FFmpeg, ffprobe, GPU, disk, and Python capability profile.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "opencut_brand_kit",
        "description": "Get, save, delete, or preview the local Brand Kit settings.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "default": "get", "enum": ["get", "save", "delete", "preview"]},
                "filepath": {"type": "string", "description": "Source media path for preview action"},
                "brand_kit": {"type": "object", "description": "Brand Kit settings for save or preview"},
                "output": {"type": "string", "description": "Optional preview output path"},
            },
        },
    },
    {
        "name": "opencut_semantic_search",
        "description": "Search media paths by semantic text query or index media for semantic search.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "default": "search", "enum": ["search", "index", "status"]},
                "query": {"type": "string", "description": "Search query"},
                "media_paths": {"type": "array", "items": {"type": "string"}},
                "mode": {"type": "string", "default": "all"},
                "top_k": {"type": "integer", "default": 10},
            },
        },
    },
    {
        "name": "opencut_spectral_match",
        "description": "Match a clip's spectral profile to a reference audio or video file.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to source media"},
                "reference_path": {"type": "string", "description": "Reference media path"},
                "strength": {"type": "number", "default": 1.0},
                "output": {"type": "string", "description": "Optional output path"},
            },
            "required": ["filepath", "reference_path"],
        },
    },
    # ------------------------------------------------------------------
    # Video editing tools
    # ------------------------------------------------------------------
    {
        "name": "opencut_trim_video",
        "description": "Trim a video to specified start and end times.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to video file"},
                "start": {"type": "number", "description": "Start time in seconds"},
                "end": {"type": "number", "description": "End time in seconds"},
            },
            "required": ["filepath", "start", "end"],
        },
    },
    {
        "name": "opencut_merge_videos",
        "description": "Merge multiple video files with optional format conversion.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "files": {"type": "array", "items": {"type": "string"}, "description": "Paths to video files"},
                "output_format": {"type": "string", "description": "Output container format", "default": "mp4"},
            },
            "required": ["files"],
        },
    },
    {
        "name": "opencut_concat_videos",
        "description": "Concatenate video files sequentially without re-encoding when possible.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "files": {"type": "array", "items": {"type": "string"}, "description": "Paths to video files in order"},
            },
            "required": ["files"],
        },
    },
    {
        "name": "opencut_reframe_video",
        "description": "Reframe video to a new aspect ratio with smart cropping or padding.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to video file"},
                "aspect_ratio": {"type": "string", "description": "Target aspect ratio, e.g. 9:16, 1:1, 4:3"},
                "mode": {"type": "string", "description": "Reframe mode: crop, pad, or smart", "default": "smart"},
            },
            "required": ["filepath", "aspect_ratio"],
        },
    },
    {
        "name": "opencut_stabilize_video",
        "description": "Stabilize shaky video footage using motion analysis.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to video file"},
                "smoothing": {"type": "number", "description": "Smoothing factor (higher = smoother)", "default": 10},
            },
            "required": ["filepath"],
        },
    },
    {
        "name": "opencut_speed_change",
        "description": "Change playback speed of a video uniformly.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to video file"},
                "speed": {"type": "number", "description": "Speed multiplier (e.g. 0.5 for half, 2.0 for double)"},
            },
            "required": ["filepath", "speed"],
        },
    },
    {
        "name": "opencut_speed_ramp",
        "description": "Apply speed ramping with keyframes for variable playback speed.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to video file"},
                "keyframes": {"type": "array", "items": {"type": "object"}, "description": "Keyframe array [{time, speed}]"},
            },
            "required": ["filepath", "keyframes"],
        },
    },
    {
        "name": "opencut_pip",
        "description": "Create a picture-in-picture composite with an overlay video.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to base video"},
                "overlay": {"type": "string", "description": "Path to overlay video"},
                "position": {"type": "string", "description": "Position: top-left, top-right, bottom-left, bottom-right, center", "default": "bottom-right"},
            },
            "required": ["filepath", "overlay"],
        },
    },
    {
        "name": "opencut_blend_videos",
        "description": "Blend two video layers with a compositing mode.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to base video"},
                "overlay": {"type": "string", "description": "Path to overlay video"},
                "mode": {"type": "string", "description": "Blend mode: add, multiply, screen, overlay", "default": "overlay"},
                "opacity": {"type": "number", "description": "Overlay opacity 0.0-1.0", "default": 0.5},
            },
            "required": ["filepath", "overlay"],
        },
    },
    {
        "name": "opencut_letterbox",
        "description": "Add letterbox/pillarbox bars to fit a target aspect ratio.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to video file"},
                "aspect_ratio": {"type": "string", "description": "Target aspect ratio, e.g. 16:9, 2.39:1"},
                "color": {"type": "string", "description": "Bar color as hex or name", "default": "black"},
            },
            "required": ["filepath", "aspect_ratio"],
        },
    },
    {
        "name": "opencut_preview_frame",
        "description": "Extract a single preview frame from a video at a given timestamp.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to video file"},
                "time": {"type": "number", "description": "Timestamp in seconds to extract"},
            },
            "required": ["filepath", "time"],
        },
    },
    # ------------------------------------------------------------------
    # Video effects tools
    # ------------------------------------------------------------------
    {
        "name": "opencut_chromakey",
        "description": "Apply chroma key (green/blue screen) removal to a video.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to video file"},
                "color": {"type": "string", "description": "Key color: green, blue, or hex value", "default": "green"},
                "tolerance": {"type": "number", "description": "Color tolerance 0.0-1.0", "default": 0.3},
            },
            "required": ["filepath"],
        },
    },
    {
        "name": "opencut_lut_apply",
        "description": "Apply a LUT (Look-Up Table) color grade to a video.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to video file"},
                "lut_path": {"type": "string", "description": "Path to .cube or .3dl LUT file"},
            },
            "required": ["filepath", "lut_path"],
        },
    },
    {
        "name": "opencut_lut_generate",
        "description": "Generate a LUT from a reference image or video for consistent color grading.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to source video"},
                "reference": {"type": "string", "description": "Path to reference image or video"},
            },
            "required": ["filepath", "reference"],
        },
    },
    {
        "name": "opencut_video_fx",
        "description": "Apply a video effect (blur, sharpen, vignette, glitch, film grain, etc.).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to video file"},
                "effect": {"type": "string", "description": "Effect name: blur, sharpen, vignette, glitch, grain, glow"},
                "params": {"type": "object", "description": "Effect-specific parameters"},
            },
            "required": ["filepath", "effect"],
        },
    },
    {
        "name": "opencut_denoise_video",
        "description": "Reduce visual noise in a video using spatial/temporal filtering.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to video file"},
                "strength": {"type": "number", "description": "Denoise strength 0.0-1.0", "default": 0.5},
            },
            "required": ["filepath"],
        },
    },
    {
        "name": "opencut_interpolate",
        "description": "Interpolate frames using neural methods to increase video frame rate.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to video file"},
                "target_fps": {"type": "number", "description": "Target frame rate"},
            },
            "required": ["filepath", "target_fps"],
        },
    },
    {
        "name": "opencut_transitions",
        "description": "Apply a transition effect between clips or at cut points.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to video file"},
                "type": {"type": "string", "description": "Transition type: crossfade, wipe, slide, zoom, dip-to-black"},
                "duration": {"type": "number", "description": "Transition duration in seconds", "default": 0.5},
            },
            "required": ["filepath", "type"],
        },
    },
    # ------------------------------------------------------------------
    # Caption tools
    # ------------------------------------------------------------------
    {
        "name": "opencut_caption_burnin",
        "description": "Burn caption segments into video with customizable styling.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to video file"},
                "segments": {"type": "array", "items": {"type": "object"}, "description": "Caption segments [{start, end, text}]"},
                "style": {"type": "object", "description": "Caption style: font, size, color, position, bg"},
            },
            "required": ["filepath", "segments"],
        },
    },
    {
        "name": "opencut_caption_translate",
        "description": "Translate existing captions or SRT to a target language.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to video or SRT file"},
                "target_language": {"type": "string", "description": "Target language code (e.g. es, fr, de, ja)"},
            },
            "required": ["filepath", "target_language"],
        },
    },
    {
        "name": "opencut_caption_styled",
        "description": "Generate and render styled captions (TikTok, YouTube, cinematic, etc.).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to video file"},
                "style": {"type": "string", "description": "Caption style preset name"},
            },
            "required": ["filepath", "style"],
        },
    },
    {
        "name": "opencut_caption_animated",
        "description": "Render animated captions with word-level timing and motion effects.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to video file"},
                "style": {"type": "string", "description": "Animation style preset"},
                "segments": {"type": "array", "items": {"type": "object"}, "description": "Caption segments with word timing"},
            },
            "required": ["filepath", "style", "segments"],
        },
    },
    {
        "name": "opencut_caption_srt_import",
        "description": "Import an SRT file and convert to timeline caption segments.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to video file for alignment"},
                "srt_path": {"type": "string", "description": "Path to SRT subtitle file"},
            },
            "required": ["filepath", "srt_path"],
        },
    },
    {
        "name": "opencut_caption_karaoke",
        "description": "Generate karaoke-style word-by-word highlight captions from audio.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to video or audio file"},
            },
            "required": ["filepath"],
        },
    },
    # ------------------------------------------------------------------
    # Audio tools
    # ------------------------------------------------------------------
    {
        "name": "opencut_audio_effects",
        "description": "Apply a chain of audio effects (EQ, reverb, compression, delay, etc.).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to audio or video file"},
                "effects": {"type": "array", "items": {"type": "object"}, "description": "Effects chain [{name, params}]"},
            },
            "required": ["filepath", "effects"],
        },
    },
    {
        "name": "opencut_audio_duck",
        "description": "Auto-duck music volume under speech segments.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to mixed audio/video file"},
                "music_path": {"type": "string", "description": "Path to music track"},
                "speech_path": {"type": "string", "description": "Path to speech track"},
            },
            "required": ["filepath", "music_path", "speech_path"],
        },
    },
    {
        "name": "opencut_audio_isolate",
        "description": "Isolate dialogue/vocals from background noise and music.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to audio or video file"},
            },
            "required": ["filepath"],
        },
    },
    {
        "name": "opencut_audio_normalize",
        "description": "Normalize audio loudness to a target LUFS level.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to audio or video file"},
                "target_lufs": {"type": "number", "description": "Target loudness in LUFS", "default": -14},
            },
            "required": ["filepath"],
        },
    },
    {
        "name": "opencut_filler_remove",
        "description": "Detect and remove filler words (um, uh, like, etc.) from audio.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to audio or video file"},
                "words": {"type": "array", "items": {"type": "string"}, "description": "Filler words to detect"},
            },
            "required": ["filepath"],
        },
    },
    {
        "name": "opencut_beat_markers",
        "description": "Detect beats in audio and return marker timestamps.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to audio or video file"},
            },
            "required": ["filepath"],
        },
    },
    {
        "name": "opencut_audio_enhance",
        "description": "Enhance audio quality using a preset profile (podcast, music, dialogue).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to audio or video file"},
                "preset": {"type": "string", "description": "Enhancement preset: podcast, music, dialogue, broadcast", "default": "podcast"},
            },
            "required": ["filepath"],
        },
    },
    # ------------------------------------------------------------------
    # Timeline & workflow tools
    # ------------------------------------------------------------------
    {
        "name": "opencut_workflow_run",
        "description": "Execute a multi-step editing workflow from a preset or custom steps.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "preset": {"type": "string", "description": "Named workflow preset"},
                "steps": {"type": "array", "items": {"type": "object"}, "description": "Custom workflow steps [{tool, args}]"},
            },
        },
    },
    {
        "name": "opencut_workflow_presets",
        "description": "List available workflow presets and their descriptions.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "opencut_timeline_batch_rename",
        "description": "Batch-rename clips or segments on the timeline.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "renames": {"type": "array", "items": {"type": "object"}, "description": "Rename operations [{old_name, new_name}]"},
            },
            "required": ["renames"],
        },
    },
    {
        "name": "opencut_timeline_smart_bins",
        "description": "Organize timeline clips into smart bins based on rules.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "rules": {"type": "array", "items": {"type": "object"}, "description": "Bin rules [{field, operator, value, bin_name}]"},
            },
            "required": ["rules"],
        },
    },
    {
        "name": "opencut_timeline_beat_cut",
        "description": "Auto-cut a video to the beat of its audio track.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to video file"},
                "mode": {"type": "string", "description": "Cut mode: on-beat, off-beat, every-other", "default": "on-beat"},
            },
            "required": ["filepath"],
        },
    },
    {
        "name": "opencut_batch_export",
        "description": "Export video using a named export preset (social, broadcast, archive, etc.).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to source video"},
                "preset": {"type": "string", "description": "Export preset name"},
            },
            "required": ["filepath", "preset"],
        },
    },
    # ------------------------------------------------------------------
    # Deliverables & search tools
    # ------------------------------------------------------------------
    {
        "name": "opencut_vfx_sheet",
        "description": "Generate a VFX breakdown sheet from sequence data.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "sequence_data": {"type": "object", "description": "Sequence/timeline data with VFX annotations"},
            },
            "required": ["sequence_data"],
        },
    },
    {
        "name": "opencut_adr_list",
        "description": "Generate an ADR (Automated Dialogue Replacement) cue list from sequence data.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "sequence_data": {"type": "object", "description": "Sequence/timeline data with dialogue markers"},
            },
            "required": ["sequence_data"],
        },
    },
    {
        "name": "opencut_music_cue_sheet",
        "description": "Generate a music cue sheet with timing, rights, and usage data.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "sequence_data": {"type": "object", "description": "Sequence/timeline data with music cues"},
            },
            "required": ["sequence_data"],
        },
    },
    {
        "name": "opencut_nlp_command",
        "description": "Parse a natural-language editing instruction into tool calls.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Natural-language editing command"},
                "llm_config": {"type": "object", "description": "Optional LLM configuration overrides"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "opencut_ingest_url",
        "description": "Ingest media from a URL for search indexing and processing.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to media resource"},
            },
            "required": ["url"],
        },
    },
    # ------------------------------------------------------------------
    # System & settings tools
    # ------------------------------------------------------------------
    {
        "name": "opencut_system_info",
        "description": "Return system information including OS, CPU, RAM, and disk stats.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "opencut_gpu_status",
        "description": "Return GPU availability, driver version, VRAM, and compute capability.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "opencut_dependencies",
        "description": "List installed dependencies and their versions (FFmpeg, Python packages, etc.).",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "opencut_feature_state",
        "description": "Return the enabled/disabled state of all feature flags.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "opencut_social_upload",
        "description": "Upload a rendered video to a social platform (YouTube, TikTok, Instagram, etc.).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Path to rendered video"},
                "platform": {"type": "string", "description": "Target platform: youtube, tiktok, instagram, twitter"},
                "credentials": {"type": "object", "description": "Platform auth credentials or token reference"},
            },
            "required": ["filepath", "platform", "credentials"],
        },
    },
]

# Route mapping for tool execution
_TOOL_ROUTES = {
    "opencut_transcribe": ("POST", "/captions"),
    "opencut_silence_remove": ("POST", "/silence"),
    "opencut_export_video": ("POST", "/export-video"),
    "opencut_highlights": ("POST", "/video/highlights"),
    "opencut_separate_audio": ("POST", "/audio/separate"),
    "opencut_tts": ("POST", "/audio/tts/generate"),
    "opencut_style_transfer": ("POST", "/video/style/apply"),
    "opencut_face_enhance": ("POST", "/video/face/enhance"),
    "opencut_generate_music": ("POST", "/audio/music-ai/generate"),
    # ``/status/{job_id}`` placeholder stays for documentation; the handler
    # below builds the real path from validated args so the literal curly
    # braces never hit the wire.
    "opencut_job_status": ("GET", "/status/{job_id}"),
    "opencut_repeat_detect": ("POST", "/captions/repeat-detect"),
    "opencut_chapters": ("POST", "/captions/chapters"),
    "opencut_footage_search": ("POST", "/search/footage"),
    "opencut_index_footage": ("POST", "/search/index"),
    "opencut_color_match": ("POST", "/video/color-match"),
    "opencut_loudness_match": ("POST", "/audio/loudness-match"),
    "opencut_auto_zoom": ("POST", "/video/auto-zoom"),
    "opencut_multicam_cuts": ("POST", "/video/multicam-cuts"),
    "opencut_denoise_audio": ("POST", "/audio/denoise"),
    "opencut_upscale": ("POST", "/video/ai/upscale"),
    "opencut_scene_detect": ("POST", "/video/scenes"),
    "opencut_depth_map": ("POST", "/video/depth/map"),
    "opencut_shorts_pipeline": ("POST", "/video/shorts-pipeline"),
    # Wave M additions (v1.30.0)
    "opencut_dub_video": ("POST", "/video/dub"),
    "opencut_sports_highlights": ("POST", "/video/highlights/sports"),
    "opencut_lipsync_echomimic": ("POST", "/video/lipsync/echomimic"),
    "opencut_chat_edit": ("POST", "/chat"),
    # F195 post-Wave-M shipped surface expansion
    "opencut_face_reshape": ("POST", "/video/face/reshape"),
    "opencut_skin_retouch": ("POST", "/video/face/retouch"),
    "opencut_smart_upscale": ("POST", "/video/upscale/smart"),
    "opencut_elevenlabs_tts": ("POST", "/audio/tts/elevenlabs"),
    "opencut_caption_qc": ("POST", "/captions/qc"),
    "opencut_review_bundle": ("POST", "/review/bundle"),
    "opencut_c2pa_provenance": ("POST", "/provenance/c2pa"),
    "opencut_marker_import": ("POST", "/markers/import"),
    "opencut_capability_probe": ("GET", "/system/capabilities"),
    "opencut_brand_kit": ("GET", "/settings/brand-kit"),
    "opencut_semantic_search": ("POST", "/search/ai"),
    "opencut_spectral_match": ("POST", "/audio/spectral-match"),
    # Video editing tools
    "opencut_trim_video": ("POST", "/video/trim"),
    "opencut_merge_videos": ("POST", "/video/merge"),
    "opencut_concat_videos": ("POST", "/video/merge"),
    "opencut_reframe_video": ("POST", "/video/reframe"),
    "opencut_stabilize_video": ("POST", "/video/stabilize-advanced"),
    "opencut_speed_change": ("POST", "/video/speed/change"),
    "opencut_speed_ramp": ("POST", "/video/speed/ramp"),
    "opencut_pip": ("POST", "/video/pip"),
    "opencut_blend_videos": ("POST", "/video/blend"),
    "opencut_letterbox": ("POST", "/video/fx/apply"),
    "opencut_preview_frame": ("POST", "/video/preview-frame"),
    # Video effects tools
    "opencut_chromakey": ("POST", "/video/chromakey"),
    "opencut_lut_apply": ("POST", "/video/lut/apply"),
    "opencut_lut_generate": ("POST", "/video/lut/generate-from-ref"),
    "opencut_video_fx": ("POST", "/video/fx/apply"),
    "opencut_denoise_video": ("POST", "/video/ai/denoise"),
    "opencut_interpolate": ("POST", "/video/interpolate/neural"),
    "opencut_transitions": ("POST", "/video/transitions/apply"),
    # Caption tools
    "opencut_caption_burnin": ("POST", "/captions/burnin/segments"),
    "opencut_caption_translate": ("POST", "/captions/translate"),
    "opencut_caption_styled": ("POST", "/styled-captions"),
    "opencut_caption_animated": ("POST", "/captions/animated/render"),
    "opencut_caption_srt_import": ("POST", "/timeline/srt-to-captions"),
    "opencut_caption_karaoke": ("POST", "/captions/karaoke"),
    # Audio tools
    "opencut_audio_effects": ("POST", "/audio/effects/apply"),
    "opencut_audio_duck": ("POST", "/audio/duck"),
    "opencut_audio_isolate": ("POST", "/audio/isolate"),
    "opencut_audio_normalize": ("POST", "/audio/normalize"),
    "opencut_filler_remove": ("POST", "/fillers"),
    "opencut_beat_markers": ("POST", "/audio/beat-markers"),
    "opencut_audio_enhance": ("POST", "/audio/enhance"),
    # Timeline & workflow tools
    "opencut_workflow_run": ("POST", "/workflow/run"),
    "opencut_workflow_presets": ("GET", "/workflow/presets"),
    "opencut_timeline_batch_rename": ("POST", "/timeline/batch-rename"),
    "opencut_timeline_smart_bins": ("POST", "/timeline/smart-bins"),
    "opencut_timeline_beat_cut": ("POST", "/timeline/beat-cut"),
    "opencut_batch_export": ("POST", "/export/preset"),
    # Deliverables & search tools
    "opencut_vfx_sheet": ("POST", "/deliverables/vfx-sheet"),
    "opencut_adr_list": ("POST", "/deliverables/adr-list"),
    "opencut_music_cue_sheet": ("POST", "/deliverables/music-cue-sheet"),
    "opencut_nlp_command": ("POST", "/nlp/command"),
    "opencut_ingest_url": ("POST", "/search/ingest"),
    # System & settings tools
    "opencut_system_info": ("GET", "/system/status"),
    "opencut_gpu_status": ("GET", "/api/gpu/status"),
    "opencut_dependencies": ("GET", "/system/dependencies"),
    "opencut_feature_state": ("GET", "/system/feature-state"),
    "opencut_social_upload": ("POST", "/social/upload"),
}


def get_mcp_tools(include_extended=None):
    """Return curated MCP tools, optionally with generated route-level tools."""
    if include_extended is None:
        include_extended = mcp_extended_tools.extended_tools_enabled()
    tools = list(MCP_TOOLS)
    if include_extended:
        tools.extend(mcp_extended_tools.get_extended_tools())
    return tools


_WIN_RESERVED_STEMS = frozenset({"CON", "PRN", "AUX", "NUL"})
_WIN_RESERVED_RE = re.compile(r"^(COM|LPT)\d$")
_REQUIRED_MCP_PATH_KEYS = (
    "filepath",
    "file",
    "source",
    "reference",
    "audio_path",
    "asset_path",
    "media_path",
    "captions_path",
    "srt_path",
    "path",
    "reference_path",
    "overlay",
    "lut_path",
    "music_path",
    "speech_path",
)
_OPTIONAL_MCP_PATH_KEYS = (
    "style_image",
    "voice_ref",
    "output_dir",
    "output",
    "output_path",
    "sidecar_path",
)
_MCP_PATH_ARRAY_KEYS = ("files", "media_paths", "extra_files")
_NESTED_MCP_PATH_CONTAINERS = ("body", "query")


def _validate_mcp_filepath(args, key="filepath", *, allow_empty=False):
    """Validate filepath arguments at MCP layer (defense-in-depth).

    Mirrors ``opencut.security.validate_path`` checks so the MCP surface
    can reject bad inputs *before* they hit the backend. The backend still
    re-validates — this just lets us fail fast with a clearer error and
    avoids burning a round-trip for obvious garbage.

    ``allow_empty=True`` lets optional keys (e.g. ``voice_ref``) omit the
    value by passing an empty string — the server route treats empty as
    "not provided" and skips processing. Without this, legitimate empty
    values were rejected as invalid.
    """
    path = args.get(key, "")
    if not isinstance(path, str):
        return False
    if not path:
        return allow_empty
    # Reject whitespace-only paths (would collapse to "." on normpath).
    if not path.strip():
        return False
    if "\x00" in path:
        return False
    # Reject other ASCII control characters that can confuse shells/logs.
    if any(ord(ch) < 0x20 for ch in path):
        return False
    # Reject UNC / Win32 device paths (\\server\share, \\?\..., \\.\...).
    # Cover both slash conventions because Windows accepts forward slashes.
    if path.startswith("\\\\") or path.startswith("//"):
        return False
    # Reject Windows reserved device names as the filename stem.
    try:
        import os as _os
        stem = _os.path.splitext(_os.path.basename(path))[0].upper()
    except Exception:
        stem = ""
    if stem in _WIN_RESERVED_STEMS or _WIN_RESERVED_RE.match(stem):
        return False
    # Reject `..` path components (substrings like ``my..file.mp4`` are OK).
    parts = path.replace("\\", "/").split("/")
    if ".." in parts:
        return False
    return True


def _mcp_path_key(prefix, key):
    return f"{prefix}.{key}" if prefix else key


def _validate_mcp_path_arguments(arguments, *, prefix=""):
    """Return a validation error for unsafe path-like MCP arguments."""
    for key in _REQUIRED_MCP_PATH_KEYS:
        if key in arguments and not _validate_mcp_filepath(arguments, key, allow_empty=False):
            label = _mcp_path_key(prefix, key)
            return f"Invalid {label}: path traversal, null byte, or UNC path detected"
    for key in _OPTIONAL_MCP_PATH_KEYS:
        if key in arguments and not _validate_mcp_filepath(arguments, key, allow_empty=True):
            label = _mcp_path_key(prefix, key)
            return f"Invalid {label}: path traversal, null byte, or UNC path detected"

    # Validate filepath arrays (e.g. "files" in index_footage / loudness_match).
    for key in _MCP_PATH_ARRAY_KEYS:
        if key not in arguments or not isinstance(arguments[key], list):
            continue
        label = _mcp_path_key(prefix, key)
        for i, item in enumerate(arguments[key]):
            # Route each item through the scalar validator so arrays get the
            # same control-character and reserved-device-name checks as
            # scalar filepath arguments (not just null-byte/UNC/traversal).
            if not _validate_mcp_filepath({key: item}, key, allow_empty=False):
                return (
                    f"Invalid path in {label}[{i}]: empty, path traversal, "
                    "null byte, control character, UNC path, or reserved "
                    "device name detected"
                )

    for key in _NESTED_MCP_PATH_CONTAINERS:
        value = arguments.get(key)
        if isinstance(value, dict):
            error = _validate_mcp_path_arguments(value, prefix=_mcp_path_key(prefix, key))
            if error:
                return error
    return None


def handle_tool_call(tool_name, arguments):
    """Execute an MCP tool call by proxying to the Flask backend."""
    is_extended_tool = mcp_extended_tools.is_extended_tool(tool_name)
    if is_extended_tool:
        if not mcp_extended_tools.extended_tools_enabled():
            return {
                "error": (
                    "Extended MCP tools are disabled. Set "
                    f"{mcp_extended_tools.EXTENDED_MCP_ENV}=1 or start "
                    "opencut-mcp-server with --extended-tools."
                )
            }
    elif tool_name not in _TOOL_ROUTES:
        return {"error": f"Unknown tool: {tool_name}"}

    # Defensive: the HTTP transport passes ``arguments`` through without
    # type-checking, and the stdio transport's guard only catches None/non-
    # dict at the top level. Callers can still send ``arguments=[]`` or
    # ``arguments="foo"``. Treat anything non-dict as empty so we return a
    # clear validation error instead of an AttributeError stack trace.
    if not isinstance(arguments, dict):
        return {"error": "`arguments` must be a JSON object"}

    path_error = _validate_mcp_path_arguments(arguments)
    if path_error:
        return {"error": path_error}

    if is_extended_tool:
        return mcp_extended_tools.invoke_extended_tool(tool_name, arguments, _api)

    method, path = _TOOL_ROUTES[tool_name]

    # Handle special routing
    if tool_name == "opencut_job_status":
        job_id = arguments.get("job_id", "")
        # Validate job_id format (UUID hex + hyphens only)
        if not re.match(r'^[a-f0-9-]+$', job_id):
            return {"error": "Invalid job_id format"}
        path = f"/status/{job_id}"
        return _api("GET", path)

    if tool_name == "opencut_generate_music":
        engine = arguments.get("engine", "ace-step")
        if engine == "ace-step":
            path = "/audio/music-ai/ace-step"

    if tool_name == "opencut_style_transfer":
        style = arguments.get("style", "")
        if style == "arbitrary":
            path = "/video/style/arbitrary"

    if tool_name == "opencut_brand_kit":
        action = str(arguments.get("action") or "get").strip().lower()
        if action == "get":
            return _api("GET", "/settings/brand-kit")
        if action == "delete":
            return _api("DELETE", "/settings/brand-kit")
        if action == "preview":
            return _api("POST", "/settings/brand-kit/preview", arguments)
        if action == "save":
            payload = {k: v for k, v in arguments.items() if k != "action"}
            if "brand_kit" in payload and isinstance(payload["brand_kit"], dict):
                payload = payload["brand_kit"]
            return _api("POST", "/settings/brand-kit", payload)
        return {"error": "Invalid action for opencut_brand_kit"}

    if tool_name == "opencut_semantic_search":
        action = str(arguments.get("action") or "search").strip().lower()
        if action == "index":
            path = "/search/ai/index"
        elif action == "status":
            return _api("GET", "/search/ai/index/status")
        elif action != "search":
            return {"error": "Invalid action for opencut_semantic_search"}

    return _api(method, path, arguments)


# ---------------------------------------------------------------------------
# MCP Protocol (stdio JSON-RPC)
# ---------------------------------------------------------------------------
_MAX_STDIO_LINE_BYTES = 4 * 1024 * 1024  # 4 MB — generous for huge JSON-RPC payloads


def run_mcp_stdio():
    """Run MCP server over stdio (JSON-RPC 2.0).

    Each input line is capped at ``_MAX_STDIO_LINE_BYTES`` to stop a
    misbehaving client from blowing up memory with an unterminated line.
    Lines beyond the cap are rejected with JSON-RPC parse error.
    """
    logger.info("OpenCut MCP server starting (stdio)")

    while True:
        line = sys.stdin.readline(_MAX_STDIO_LINE_BYTES + 1)
        if line == "":
            # EOF — client closed stdin.
            return
        oversize = len(line) > _MAX_STDIO_LINE_BYTES and not line.endswith("\n")
        line = line.strip()
        if not line:
            continue

        if oversize:
            # Drain the rest of the oversized line so we resync on the next \n.
            while True:
                chunk = sys.stdin.readline(_MAX_STDIO_LINE_BYTES + 1)
                if chunk == "" or chunk.endswith("\n"):
                    break
            err = {"jsonrpc": "2.0", "id": None,
                   "error": {"code": -32700,
                             "message": f"Input line exceeded {_MAX_STDIO_LINE_BYTES} bytes"}}
            sys.stdout.write(json.dumps(err) + "\n")
            sys.stdout.flush()
            continue

        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            err = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
            sys.stdout.write(json.dumps(err) + "\n")
            sys.stdout.flush()
            continue

        msg_id = msg.get("id")
        method = msg.get("method", "")
        params = msg.get("params", {})

        # JSON-RPC 2.0: messages without an ``id`` field are notifications
        # and must not produce a response, regardless of method name.
        if "id" not in msg:
            continue

        if method == "initialize":
            response = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": "opencut",
                        "version": __version__,
                    },
                },
            }
        elif method == "tools/list":
            response = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"tools": get_mcp_tools()},
            }
        elif method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            if not isinstance(arguments, dict):
                arguments = {}
            result = handle_tool_call(tool_name, arguments)
            response = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result, indent=2)}],
                },
            }
        elif method == "resources/list":
            response = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"resources": []},
            }
        elif method == "prompts/list":
            response = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"prompts": []},
            }
        elif method == "notifications/initialized":
            continue  # No response needed for notifications
        else:
            response = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }

        # Fall back to an error response if the result contained a
        # non-JSON-serializable value — never leave the client hanging
        # without a reply for a request that has an id.
        try:
            payload = json.dumps(response, default=str)
        except (TypeError, ValueError) as e:
            payload = json.dumps({
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32603, "message": f"Serialization error: {e}"},
            })
        sys.stdout.write(payload + "\n")
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# HTTP transport (JSON-RPC 2.0 over HTTP on port 5681)
# ---------------------------------------------------------------------------
def run_http_server(
    backend_url: str = BACKEND_URL,
    port: int = 5681,
    bind: str = "127.0.0.1",
) -> None:
    """Run a JSON-RPC 2.0 HTTP server that proxies to the OpenCut backend.

    Defaults to ``127.0.0.1`` so the MCP sidecar matches the Flask backend's
    loopback-only default. Operators can still opt into LAN exposure with
    ``--http-bind 0.0.0.0``. Non-loopback binds require the same persistent
    ``X-OpenCut-Auth`` token used by the main OpenCut HTTP server.
    """
    auth_required = _mcp_http_bind_requires_auth(bind)
    if auth_required:
        _auth.ensure_token(label="mcp-http")

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            logger.debug("MCP HTTP: " + fmt, *args)

        def _send_json(self, data: dict, status: int = 200) -> None:
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b""
            return json.loads(raw) if raw else {}

        def _request_path(self) -> str:
            return urllib.parse.urlsplit(self.path).path

        def _authorize(self) -> bool:
            if _mcp_http_request_is_authorized(
                self.headers,
                auth_required=auth_required,
            ):
                return True
            self._send_json(
                {
                    "error": "Missing or invalid X-OpenCut-Auth token",
                    "code": "AUTH_REQUIRED",
                },
                status=401,
            )
            return False

        def do_GET(self):
            if not self._authorize():
                return

            path = self._request_path()
            if path in ("/health", "/"):
                self._send_json({"status": "ok", "server": "opencut-mcp",
                                  "version": __version__, "tools": len(get_mcp_tools())})
            elif path == "/tools":
                self._send_json({"tools": get_mcp_tools()})
            else:
                self._send_json({"error": "Not found"}, status=404)

        def do_POST(self):
            if not self._authorize():
                return

            try:
                body = self._read_json()
            except Exception:
                self._send_json({"error": "Invalid JSON"}, status=400)
                return

            method = body.get("method", "")
            params = body.get("params") or {}
            rpc_id = body.get("id")

            if method == "initialize":
                self._send_json({"jsonrpc": "2.0", "id": rpc_id, "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "opencut", "version": __version__},
                }})
            elif method == "tools/list":
                self._send_json({"jsonrpc": "2.0", "id": rpc_id,
                                  "result": {"tools": get_mcp_tools()}})
            elif method == "tools/call":
                name = params.get("name", "")
                arguments = params.get("arguments") or {}
                result = handle_tool_call(name, arguments)
                self._send_json({"jsonrpc": "2.0", "id": rpc_id,
                                  "result": {"content": [{"type": "text",
                                                          "text": json.dumps(result, indent=2)}]}})
            elif method in ("resources/list", "prompts/list"):
                key = method.split("/")[0]
                self._send_json({"jsonrpc": "2.0", "id": rpc_id,
                                  "result": {key: []}})
            else:
                self._send_json({"jsonrpc": "2.0", "id": rpc_id,
                                  "error": {"code": -32601,
                                            "message": f"Method not found: {method}"}}, status=404)

    httpd = HTTPServer((bind, port), _Handler)
    print(f"OpenCut MCP HTTP server on http://{bind}:{port}", file=sys.stderr)
    if auth_required:
        print(
            "  Non-loopback bind: requests must include the X-OpenCut-Auth "
            "header with the token shown by opencut-server --print-auth.",
            file=sys.stderr,
        )
    print(f"Proxying to OpenCut backend at {backend_url}", file=sys.stderr)
    print(f"Tools registered: {len(get_mcp_tools())}", file=sys.stderr)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nMCP server stopped.", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Entry point for the ``opencut-mcp-server`` console script."""
    parser = argparse.ArgumentParser(
        prog="opencut-mcp-server",
        description="OpenCut MCP sidecar — exposes OpenCut as MCP tools.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5679,
        help="OpenCut backend port (default: 5679)",
    )
    parser.add_argument(
        "--http",
        action="store_true",
        help="Use HTTP transport on port 5681 instead of stdio",
    )
    parser.add_argument(
        "--http-port",
        type=int,
        default=5681,
        help="Port for HTTP transport (default: 5681)",
    )
    parser.add_argument(
        "--http-bind",
        default="127.0.0.1",
        help=(
            "Bind address for HTTP transport (default: 127.0.0.1). "
            "Pass 0.0.0.0 to expose on the LAN; non-loopback binds "
            "require X-OpenCut-Auth."
        ),
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="Print all registered MCP tools and exit",
    )
    parser.add_argument(
        "--extended-tools",
        action="store_true",
        help=(
            "Expose the auto-generated lower-priority route-level MCP tools "
            f"(equivalent to {mcp_extended_tools.EXTENDED_MCP_ENV}=1)."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    # An explicit service URL lets the containerized sidecar reach the backend;
    # --port remains the ergonomic local default.
    global BACKEND_URL
    BACKEND_URL = (
        os.environ.get("OPENCUT_MCP_BACKEND_URL", "").strip().rstrip("/")
        or f"http://127.0.0.1:{args.port}"
    )
    if args.extended_tools:
        os.environ[mcp_extended_tools.EXTENDED_MCP_ENV] = "1"

    if args.list_tools:
        tools = get_mcp_tools()
        for t in tools:
            schema = t.get("inputSchema", {})
            props = schema.get("properties", {})
            req = set(schema.get("required", []))
            param_str = ", ".join(
                f"{k}{'*' if k in req else ''}" for k in props
            )
            print(f"  {t['name']:<40}  ({param_str})")
        print(f"\n{len(tools)} tools total.")
        return

    if args.http:
        run_http_server(
            backend_url=BACKEND_URL,
            port=args.http_port,
            bind=args.http_bind,
        )
        return

    run_mcp_stdio()


if __name__ == "__main__":
    main()
