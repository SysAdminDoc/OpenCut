"""
OpenCut MCP Server

Exposes OpenCut's API as a Model Context Protocol (MCP) server,
allowing AI clients (Claude Code, Cursor, etc.) to drive video
editing operations programmatically.

Usage:
    python -m opencut.mcp_server          # stdio JSON-RPC transport

The MCP server proxies requests to the running OpenCut Flask backend
on localhost:5679. Start the backend first: python -m opencut.server

Protocol: JSON-RPC 2.0 over stdio (one JSON object per line).
Supports: initialize, tools/list, tools/call, resources/list.
"""

import json
import logging
import re
import sys
import urllib.error
import urllib.request

from opencut import __version__

logger = logging.getLogger("opencut.mcp")

BACKEND_URL = "http://127.0.0.1:5679"
_csrf_token = ""


def _refresh_csrf():
    """Fetch fresh CSRF token from backend."""
    global _csrf_token
    try:
        req = urllib.request.Request(f"{BACKEND_URL}/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = json.loads(resp.read())
            _csrf_token = body.get("csrf_token", "")
    except Exception as exc:
        logger.warning("CSRF refresh failed: %s", exc)


def _api(method, path, data=None):
    """Call the OpenCut Flask backend."""
    global _csrf_token

    if not _csrf_token:
        _refresh_csrf()

    url = f"{BACKEND_URL}{path}"
    headers = {"Content-Type": "application/json"}
    if _csrf_token:
        headers["X-OpenCut-Token"] = _csrf_token

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
}


def _validate_mcp_filepath(args, key="filepath", *, allow_empty=False):
    """Validate filepath arguments at MCP layer (defense-in-depth).

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
    if "\x00" in path:
        return False
    if path.startswith("\\\\") or path.startswith("//"):
        return False  # Reject UNC paths
    # Check for `..` as a *path component* (not just a substring); names like
    # ``my..file.mp4`` are legal filenames and must not be rejected.
    parts = path.replace("\\", "/").split("/")
    if ".." in parts:
        return False
    return True


def handle_tool_call(tool_name, arguments):
    """Execute an MCP tool call by proxying to the Flask backend."""
    if tool_name not in _TOOL_ROUTES:
        return {"error": f"Unknown tool: {tool_name}"}

    # Required scalar path keys — empty rejected
    _required_path_keys = ("filepath", "file", "source", "reference")
    # Optional scalar path keys — empty allowed (means "not provided")
    _optional_path_keys = ("style_image", "voice_ref", "output_dir")
    for key in _required_path_keys:
        if key in arguments and not _validate_mcp_filepath(arguments, key, allow_empty=False):
            return {"error": f"Invalid {key}: path traversal, null byte, or UNC path detected"}
    for key in _optional_path_keys:
        if key in arguments and not _validate_mcp_filepath(arguments, key, allow_empty=True):
            return {"error": f"Invalid {key}: path traversal, null byte, or UNC path detected"}

    # Validate filepath arrays (e.g. "files" in index_footage / loudness_match)
    for key in ("files",):
        if key in arguments and isinstance(arguments[key], list):
            for i, item in enumerate(arguments[key]):
                if not isinstance(item, str) or not item or "\x00" in item \
                        or item.startswith("\\\\") or item.startswith("//"):
                    return {"error": f"Invalid path in {key}[{i}]: empty, null byte, or UNC path detected"}
                parts = item.replace("\\", "/").split("/")
                if ".." in parts:
                    return {"error": f"Invalid path in {key}[{i}]: path traversal component detected"}

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

    return _api(method, path, arguments)


# ---------------------------------------------------------------------------
# MCP Protocol (stdio JSON-RPC)
# ---------------------------------------------------------------------------
def run_mcp_stdio():
    """Run MCP server over stdio (JSON-RPC 2.0)."""
    logger.info("OpenCut MCP server starting (stdio)")

    for line in sys.stdin:
        line = line.strip()
        if not line:
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
                "result": {"tools": MCP_TOOLS},
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    run_mcp_stdio()
