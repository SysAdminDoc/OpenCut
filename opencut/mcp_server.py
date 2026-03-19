"""
OpenCut MCP Server

Exposes OpenCut's API as a Model Context Protocol (MCP) server,
allowing AI clients (Claude Code, Cursor, etc.) to drive video
editing operations programmatically.

Usage:
    python -m opencut.mcp_server          # stdio transport (default)
    python -m opencut.mcp_server --sse    # SSE transport on port 5680

The MCP server proxies requests to the running OpenCut Flask backend
on localhost:5679. Start the backend first: python -m opencut.server
"""

import json
import logging
import sys
import urllib.error
import urllib.request

logger = logging.getLogger("opencut.mcp")

BACKEND_URL = "http://127.0.0.1:5679"
_csrf_token = ""


def _api(method, path, data=None):
    """Call the OpenCut Flask backend."""
    global _csrf_token

    # Get CSRF token if we don't have one
    if not _csrf_token:
        try:
            req = urllib.request.Request(f"{BACKEND_URL}/health")
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = json.loads(resp.read())
                _csrf_token = body.get("csrf_token", "")
        except Exception:
            pass

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
}


def handle_tool_call(tool_name, arguments):
    """Execute an MCP tool call by proxying to the Flask backend."""
    if tool_name not in _TOOL_ROUTES:
        return {"error": f"Unknown tool: {tool_name}"}

    method, path = _TOOL_ROUTES[tool_name]

    # Handle special routing
    if tool_name == "opencut_job_status":
        job_id = arguments.get("job_id", "")
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
            continue

        msg_id = msg.get("id")
        method = msg.get("method", "")
        params = msg.get("params", {})

        if method == "initialize":
            response = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": "opencut",
                        "version": "1.3.1",
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
            result = handle_tool_call(tool_name, arguments)
            response = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result, indent=2)}],
                },
            }
        elif method == "notifications/initialized":
            continue  # No response needed for notifications
        else:
            response = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }

        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    run_mcp_stdio()
