"""
OpenCut Standalone Web UI Backend (9.2)

Backend support for a standalone web UI served at localhost:5679.
Handles file uploads, session management, and provides an operation
catalog for the SPA shell.

Sessions are stored in memory with automatic cleanup after 24 hours.
Uploaded files go to temporary directories under ``~/.opencut/web_uploads/``.
"""

import logging
import mimetypes
import os
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from opencut.helpers import OPENCUT_DIR

logger = logging.getLogger("opencut")

WEB_UPLOADS_DIR = os.path.join(OPENCUT_DIR, "web_uploads")

# In-memory session store  {session_id: WebSession}
_sessions: Dict[str, "WebSession"] = {}
_session_lock = threading.Lock()

SESSION_MAX_AGE = 86400  # 24 hours


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class UploadedFile:
    """Metadata for a file uploaded through the web UI."""
    session_id: str
    filename: str
    path: str
    size: int
    mime_type: str
    uploaded_at: float = field(default_factory=time.time)


@dataclass
class WebSession:
    """A web UI session with uploaded files and recent operations."""
    session_id: str
    uploaded_files: List[UploadedFile] = field(default_factory=list)
    recent_operations: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


@dataclass
class OperationCard:
    """Describes an available operation for the web UI catalog."""
    id: str
    name: str
    category: str
    description: str
    params_schema: Dict = field(default_factory=dict)
    endpoint: str = ""


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

def _cleanup_expired_sessions() -> int:
    """Remove sessions older than SESSION_MAX_AGE.  Returns count removed."""
    now = time.time()
    expired = [
        sid for sid, s in _sessions.items()
        if now - s.created_at > SESSION_MAX_AGE
    ]
    for sid in expired:
        _remove_session_files(sid)
        del _sessions[sid]
    return len(expired)


def _remove_session_files(session_id: str) -> None:
    """Delete the upload directory for a session."""
    upload_dir = os.path.join(WEB_UPLOADS_DIR, session_id)
    if os.path.isdir(upload_dir):
        shutil.rmtree(upload_dir, ignore_errors=True)


def create_session() -> WebSession:
    """Create a new web UI session and return it."""
    with _session_lock:
        _cleanup_expired_sessions()
        session_id = uuid.uuid4().hex[:16]
        session = WebSession(session_id=session_id)
        _sessions[session_id] = session
        logger.info("Web UI session created: %s", session_id)
        return session


def get_session(session_id: str) -> Optional[WebSession]:
    """Return a session by ID, or None if not found / expired."""
    with _session_lock:
        session = _sessions.get(session_id)
        if session is None:
            return None
        if time.time() - session.created_at > SESSION_MAX_AGE:
            _remove_session_files(session_id)
            del _sessions[session_id]
            return None
        return session


def cleanup_session(session_id: str) -> bool:
    """Delete a session and its uploaded files.  Returns True if found."""
    with _session_lock:
        if session_id not in _sessions:
            return False
        _remove_session_files(session_id)
        del _sessions[session_id]
        logger.info("Web UI session cleaned up: %s", session_id)
        return True


# ---------------------------------------------------------------------------
# File uploads
# ---------------------------------------------------------------------------

def upload_file(session_id: str, filename: str, file_data: bytes) -> UploadedFile:
    """Save uploaded file data to the session's temp directory."""
    session = get_session(session_id)
    if session is None:
        raise ValueError(f"Session not found: {session_id}")

    # Sanitize filename
    safe_name = os.path.basename(filename).replace("..", "").strip()
    if not safe_name:
        safe_name = f"upload_{uuid.uuid4().hex[:8]}"

    upload_dir = os.path.join(WEB_UPLOADS_DIR, session_id)
    os.makedirs(upload_dir, exist_ok=True)

    dest = os.path.join(upload_dir, safe_name)
    # Avoid overwrites by appending counter
    base, ext = os.path.splitext(dest)
    counter = 1
    while os.path.exists(dest):
        dest = f"{base}_{counter}{ext}"
        counter += 1

    with open(dest, "wb") as fh:
        fh.write(file_data)

    mime_type = mimetypes.guess_type(safe_name)[0] or "application/octet-stream"

    uploaded = UploadedFile(
        session_id=session_id,
        filename=safe_name,
        path=dest,
        size=len(file_data),
        mime_type=mime_type,
    )
    with _session_lock:
        s = _sessions.get(session_id)
        if s is not None:
            s.uploaded_files.append(uploaded)
    logger.info("File uploaded: %s (%d bytes) to session %s",
                safe_name, len(file_data), session_id)
    return uploaded


def list_uploads(session_id: str) -> List[UploadedFile]:
    """List all uploaded files for a session."""
    session = get_session(session_id)
    if session is None:
        return []
    return list(session.uploaded_files)


# ---------------------------------------------------------------------------
# Operation catalog
# ---------------------------------------------------------------------------

_OPERATION_CATALOG: List[OperationCard] = [
    # -- Video Processing --
    OperationCard(
        id="silence_remove", name="Silence Removal",
        category="Audio", description="Detect and remove silent segments",
        params_schema={"threshold": "float", "min_duration": "float"},
        endpoint="/silence",
    ),
    OperationCard(
        id="bg_removal", name="Background Removal",
        category="Video AI", description="Remove or replace video background using AI",
        params_schema={"model": "string", "bg_color": "string"},
        endpoint="/video-ai/bg-removal",
    ),
    OperationCard(
        id="upscale", name="Video Upscale",
        category="Video AI", description="Upscale video resolution using AI models",
        params_schema={"scale": "int", "model": "string"},
        endpoint="/video-ai/upscale",
    ),
    OperationCard(
        id="style_transfer", name="Style Transfer",
        category="Video AI", description="Apply artistic style transfer to video",
        params_schema={"style": "string", "strength": "float"},
        endpoint="/video-ai/style-transfer",
    ),
    OperationCard(
        id="denoise", name="Video Denoise",
        category="Video Processing", description="Remove noise from video footage",
        params_schema={"strength": "float", "method": "string"},
        endpoint="/video-ai/denoise",
    ),
    OperationCard(
        id="captions", name="Auto Captions",
        category="Captions", description="Generate captions using speech recognition",
        params_schema={"model": "string", "language": "string"},
        endpoint="/captions/generate",
    ),
    OperationCard(
        id="color_match", name="Color Match",
        category="Color", description="Match color grading between clips",
        params_schema={"reference": "string", "method": "string"},
        endpoint="/color/match",
    ),
    OperationCard(
        id="stabilize", name="Video Stabilization",
        category="Video Processing", description="Stabilize shaky footage",
        params_schema={"smoothing": "float", "crop": "string"},
        endpoint="/stabilize",
    ),
    OperationCard(
        id="scene_detect", name="Scene Detection",
        category="Analysis", description="Detect scene changes in video",
        params_schema={"threshold": "float", "method": "string"},
        endpoint="/scene-detect",
    ),
    OperationCard(
        id="speed_ramp", name="Speed Ramp",
        category="Effects", description="Create smooth speed ramp transitions",
        params_schema={"segments": "array"},
        endpoint="/speed-ramp",
    ),
    OperationCard(
        id="object_removal", name="Object Removal",
        category="Video AI", description="Remove unwanted objects from video",
        params_schema={"mask": "string", "method": "string"},
        endpoint="/video-ai/object-removal",
    ),
    OperationCard(
        id="audio_enhance", name="Audio Enhancement",
        category="Audio", description="Enhance audio quality with AI",
        params_schema={"model": "string", "strength": "float"},
        endpoint="/audio/enhance",
    ),
    OperationCard(
        id="face_enhance", name="Face Enhancement",
        category="Video AI", description="Enhance and restore faces in video",
        params_schema={"model": "string", "strength": "float"},
        endpoint="/video-ai/face-enhance",
    ),
    OperationCard(
        id="depth_effects", name="Depth Effects",
        category="Video AI", description="Apply depth-based effects using AI depth maps",
        params_schema={"effect": "string", "strength": "float"},
        endpoint="/depth-effects",
    ),
    OperationCard(
        id="export", name="Export Video",
        category="Export", description="Export video with preset or custom settings",
        params_schema={"preset": "string", "format": "string"},
        endpoint="/export",
    ),
]


def get_operation_catalog() -> Dict[str, List[Dict]]:
    """Return all available operations grouped by category."""
    catalog: Dict[str, List[Dict]] = {}
    for op in _OPERATION_CATALOG:
        cat = op.category
        if cat not in catalog:
            catalog[cat] = []
        catalog[cat].append(asdict(op))
    return catalog


# ---------------------------------------------------------------------------
# SPA shell
# ---------------------------------------------------------------------------

_SPA_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OpenCut Web UI</title>
<style>
  :root { --bg: #1a1a2e; --surface: #16213e; --accent: #0f3460; --text: #e0e0e0; --primary: #e94560; }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background:var(--bg); color:var(--text); }
  header { background:var(--surface); padding:1rem 2rem; display:flex; align-items:center; gap:1rem; border-bottom:2px solid var(--accent); }
  header h1 { font-size:1.4rem; color:var(--primary); }
  .container { max-width:1200px; margin:2rem auto; padding:0 1rem; }
  .upload-zone { border:2px dashed var(--accent); border-radius:12px; padding:3rem; text-align:center; cursor:pointer; transition:border-color .2s; }
  .upload-zone:hover { border-color:var(--primary); }
  .upload-zone input { display:none; }
  .file-list { margin:1rem 0; }
  .file-item { background:var(--surface); padding:0.5rem 1rem; border-radius:6px; margin:0.25rem 0; display:flex; justify-content:space-between; }
  .ops-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:1rem; margin:2rem 0; }
  .op-card { background:var(--surface); border-radius:10px; padding:1.2rem; border:1px solid var(--accent); transition:transform .15s; }
  .op-card:hover { transform:translateY(-2px); }
  .op-card h3 { color:var(--primary); margin-bottom:0.5rem; font-size:1rem; }
  .op-card .cat { font-size:0.75rem; opacity:0.7; text-transform:uppercase; }
  .op-card p { font-size:0.85rem; margin-top:0.4rem; opacity:0.85; }
  .status-bar { background:var(--surface); padding:0.75rem 1rem; border-radius:8px; margin:1rem 0; font-size:0.85rem; }
  section h2 { margin-bottom:1rem; font-size:1.1rem; }
  #job-status { min-height:40px; }
</style>
</head>
<body>
<header>
  <h1>OpenCut</h1>
  <span style="opacity:0.6;font-size:0.85rem;">Standalone Web Editor</span>
</header>
<div class="container">
  <section>
    <h2>Upload Media</h2>
    <div class="upload-zone" id="dropZone">
      <p>Drop files here or click to upload</p>
      <input type="file" id="fileInput" multiple accept="video/*,audio/*,image/*">
    </div>
    <div class="file-list" id="fileList"></div>
  </section>
  <section>
    <h2>Operations</h2>
    <div class="ops-grid" id="opsGrid"></div>
  </section>
  <section>
    <h2>Job Status</h2>
    <div class="status-bar" id="job-status">No active jobs.</div>
  </section>
</div>
<script>
(function(){
  let sessionId = null;
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
  const fileList = document.getElementById('fileList');
  const opsGrid = document.getElementById('opsGrid');

  async function initSession() {
    const r = await fetch('/web-ui/session/create', {method:'POST', headers:{'Content-Type':'application/json'}, body:'{}'});
    const d = await r.json(); sessionId = d.session_id;
  }

  async function loadOps() {
    const r = await fetch('/web-ui/operations');
    const d = await r.json();
    for (const [cat, ops] of Object.entries(d.catalog || {})) {
      ops.forEach(op => {
        const card = document.createElement('div'); card.className='op-card';
        card.innerHTML = `<div class="cat">${cat}</div><h3>${op.name}</h3><p>${op.description}</p>`;
        opsGrid.appendChild(card);
      });
    }
  }

  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.style.borderColor='var(--primary)'; });
  dropZone.addEventListener('dragleave', () => { dropZone.style.borderColor=''; });
  dropZone.addEventListener('drop', e => { e.preventDefault(); dropZone.style.borderColor=''; handleFiles(e.dataTransfer.files); });
  fileInput.addEventListener('change', e => handleFiles(e.target.files));

  async function handleFiles(files) {
    if (!sessionId) await initSession();
    for (const file of files) {
      const fd = new FormData(); fd.append('file', file); fd.append('session_id', sessionId);
      const r = await fetch('/web-ui/upload', {method:'POST', body:fd});
      const d = await r.json();
      const item = document.createElement('div'); item.className='file-item';
      item.innerHTML = `<span>${d.filename || file.name}</span><span>${(file.size/1024).toFixed(1)} KB</span>`;
      fileList.appendChild(item);
    }
  }

  initSession(); loadOps();
})();
</script>
</body>
</html>
"""


def serve_web_ui() -> str:
    """Return the HTML for the SPA shell."""
    return _SPA_HTML
