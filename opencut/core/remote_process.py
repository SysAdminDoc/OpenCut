"""
OpenCut Remote Processing / Render Node (Feature 7.5)

Offload heavy processing to remote OpenCut server instances.
Send job + source file via HTTP multipart, poll for completion,
download result.  Uses urllib.request (no requests dependency).

Node registry persisted at ~/.opencut/remote_nodes.json.
"""

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OPENCUT_DIR = os.path.join(os.path.expanduser("~"), ".opencut")
REGISTRY_PATH = os.path.join(OPENCUT_DIR, "remote_nodes.json")
DEFAULT_TIMEOUT = 30          # seconds for control-plane requests
UPLOAD_TIMEOUT = 600          # seconds for file uploads
POLL_INTERVAL = 2.0           # seconds between status polls
MAX_POLL_DURATION = 7200      # seconds — give up after 2 hours
USER_AGENT = "OpenCut-RemoteNode/1.0"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class RemoteNode:
    """A registered remote OpenCut server."""
    url: str = ""
    api_key: str = ""
    name: str = ""
    capabilities: List[str] = field(default_factory=list)
    status: str = "unknown"       # online | offline | unknown
    last_ping: float = 0.0        # epoch timestamp

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RemoteJob:
    """Tracks a job submitted to a remote node."""
    node_url: str = ""
    local_file: str = ""
    remote_job_id: str = ""
    status: str = "pending"       # pending | running | complete | error
    progress: int = 0
    result_path: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class NodeRegistry:
    """In-memory registry of remote nodes."""
    nodes: Dict[str, RemoteNode] = field(default_factory=dict)
    default_node: str = ""        # URL of the preferred node

    def to_dict(self) -> dict:
        return {
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "default_node": self.default_node,
        }


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_registry = NodeRegistry()
_registry_lock = threading.Lock()
_remote_jobs: Dict[str, RemoteJob] = {}
_jobs_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def _ensure_dir():
    os.makedirs(OPENCUT_DIR, exist_ok=True)


def _save_registry():
    """Persist registry to disk.  Caller must hold ``_registry_lock``."""
    _ensure_dir()
    try:
        payload = {
            "default_node": _registry.default_node,
            "nodes": {},
        }
        for url, node in _registry.nodes.items():
            payload["nodes"][url] = node.to_dict()
        with open(REGISTRY_PATH, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    except Exception as exc:
        logger.warning("Failed to save node registry: %s", exc)


def _load_registry():
    """Load registry from disk into module state."""
    global _registry
    if not os.path.isfile(REGISTRY_PATH):
        return
    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        _registry.default_node = data.get("default_node", "")
        for url, nd in data.get("nodes", {}).items():
            _registry.nodes[url] = RemoteNode(
                url=nd.get("url", url),
                api_key=nd.get("api_key", ""),
                name=nd.get("name", ""),
                capabilities=nd.get("capabilities", []),
                status=nd.get("status", "unknown"),
                last_ping=nd.get("last_ping", 0.0),
            )
    except Exception as exc:
        logger.warning("Failed to load node registry: %s", exc)


# Load on import
_load_registry()


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def _make_request(url: str, api_key: str = "", method: str = "GET",
                  data: bytes = None, headers: dict = None,
                  timeout: int = DEFAULT_TIMEOUT) -> dict:
    """Issue an HTTP request and return parsed JSON response.

    Raises ``RuntimeError`` on HTTP/network errors.
    """
    hdrs = {
        "User-Agent": USER_AGENT,
    }
    if api_key:
        hdrs["Authorization"] = f"Bearer {api_key}"
    if headers:
        hdrs.update(headers)

    req = Request(url, data=data, headers=hdrs, method=method)
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            if not body:
                return {}
            return json.loads(body)
    except HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(
            f"Remote server returned HTTP {exc.code}: {body[:500]}"
        )
    except URLError as exc:
        raise RuntimeError(f"Cannot reach remote node: {exc.reason}")
    except json.JSONDecodeError:
        raise RuntimeError("Remote server returned invalid JSON")


def _build_multipart(file_path: str, params: dict) -> tuple:
    """Build a multipart/form-data request body.

    Returns ``(body_bytes, content_type)``.
    """
    boundary = uuid.uuid4().hex
    lines: list = []
    crlf = b"\r\n"

    # Add form fields
    for key, value in params.items():
        lines.append(f"--{boundary}".encode())
        lines.append(f'Content-Disposition: form-data; name="{key}"'.encode())
        lines.append(b"")
        lines.append(str(value).encode("utf-8"))

    # Add file
    filename = os.path.basename(file_path)
    lines.append(f"--{boundary}".encode())
    lines.append(
        f'Content-Disposition: form-data; name="file"; filename="{filename}"'.encode()
    )
    lines.append(b"Content-Type: application/octet-stream")
    lines.append(b"")
    with open(file_path, "rb") as fh:
        lines.append(fh.read())

    lines.append(f"--{boundary}--".encode())
    lines.append(b"")

    body = crlf.join(lines)
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def register_node(url: str, api_key: str = "", name: str = "",
                  on_progress: Optional[Callable] = None) -> RemoteNode:
    """Test connection to a remote node and register it.

    Pings ``GET <url>/health`` to verify the node is an OpenCut server.
    Returns the ``RemoteNode`` dataclass on success.
    Raises ``RuntimeError`` if the node is unreachable.
    """
    if on_progress:
        on_progress(10, f"Testing connection to {url}")

    # Normalise URL (strip trailing slash)
    url = url.rstrip("/")

    # Ping the node
    health = ping_node(url, api_key=api_key)

    capabilities = health.get("capabilities", [])
    if not capabilities:
        # Assume basic capabilities if not reported
        capabilities = ["video", "audio", "export"]

    node = RemoteNode(
        url=url,
        api_key=api_key,
        name=name or health.get("name", url),
        capabilities=capabilities,
        status="online",
        last_ping=time.time(),
    )

    with _registry_lock:
        _registry.nodes[url] = node
        if not _registry.default_node:
            _registry.default_node = url
        _save_registry()

    if on_progress:
        on_progress(100, f"Node registered: {node.name}")

    return node


def list_nodes() -> List[RemoteNode]:
    """Return all registered nodes."""
    with _registry_lock:
        return list(_registry.nodes.values())


def get_node(url: str) -> Optional[RemoteNode]:
    """Return a single node by URL, or None."""
    with _registry_lock:
        return _registry.nodes.get(url.rstrip("/"))


def remove_node(url: str) -> bool:
    """Unregister a node.  Returns True if it existed."""
    url = url.rstrip("/")
    with _registry_lock:
        if url in _registry.nodes:
            del _registry.nodes[url]
            if _registry.default_node == url:
                _registry.default_node = next(iter(_registry.nodes), "")
            _save_registry()
            return True
    return False


def ping_node(url: str, api_key: str = "",
              on_progress: Optional[Callable] = None) -> dict:
    """Check health of a remote node via ``GET /health``.

    Returns the parsed JSON health response.
    Raises ``RuntimeError`` on failure.
    """
    url = url.rstrip("/")
    if on_progress:
        on_progress(20, f"Pinging {url}")

    result = _make_request(f"{url}/health", api_key=api_key, timeout=DEFAULT_TIMEOUT)

    # Update cached status
    with _registry_lock:
        if url in _registry.nodes:
            _registry.nodes[url].status = "online"
            _registry.nodes[url].last_ping = time.time()
            _save_registry()

    if on_progress:
        on_progress(100, "Node is online")

    return result


def submit_remote_job(node_url: str, file_path: str, endpoint: str,
                      params: Optional[dict] = None,
                      on_progress: Optional[Callable] = None) -> RemoteJob:
    """Upload file and submit a job to a remote node.

    Steps:
      1. Upload source file via multipart POST to ``<node>/api/upload``.
      2. Submit job via POST to ``<node>/<endpoint>``.
      3. Poll ``<node>/jobs/<remote_job_id>`` until completion.

    Args:
        node_url: Base URL of the remote OpenCut server.
        file_path: Local path to the source file.
        endpoint: API endpoint on the remote node (e.g. ``/silence``).
        params: Additional parameters for the job request.
        on_progress: Optional callback ``(pct, msg)``.

    Returns:
        ``RemoteJob`` with final status and result path.
    """
    node_url = node_url.rstrip("/")
    params = params or {}

    # Resolve node for API key
    node = get_node(node_url)
    api_key = node.api_key if node else ""

    if on_progress:
        on_progress(5, "Preparing upload")

    # Validate local file exists
    if not os.path.isfile(file_path):
        raise ValueError(f"Local file not found: {file_path}")

    # Build multipart body with the file + params
    job_params = dict(params)
    job_params["endpoint"] = endpoint
    body, content_type = _build_multipart(file_path, job_params)

    if on_progress:
        on_progress(15, "Uploading file to remote node")

    # Submit via multipart POST
    submit_url = f"{node_url}/api/remote/receive-job"
    result = _make_request(
        submit_url,
        api_key=api_key,
        method="POST",
        data=body,
        headers={"Content-Type": content_type},
        timeout=UPLOAD_TIMEOUT,
    )

    remote_job_id = result.get("job_id", "")
    if not remote_job_id:
        raise RuntimeError("Remote node did not return a job ID")

    rjob = RemoteJob(
        node_url=node_url,
        local_file=file_path,
        remote_job_id=remote_job_id,
        status="running",
    )

    # Track locally
    local_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _remote_jobs[local_id] = rjob

    if on_progress:
        on_progress(25, f"Job submitted: {remote_job_id}")

    # Poll for completion
    poll_start = time.time()
    status_url = f"{node_url}/jobs/{remote_job_id}"
    while time.time() - poll_start < MAX_POLL_DURATION:
        try:
            status = _make_request(status_url, api_key=api_key, timeout=DEFAULT_TIMEOUT)
        except RuntimeError:
            # Transient failure — retry
            time.sleep(POLL_INTERVAL * 2)
            continue

        rjob.status = status.get("status", "running")
        rjob.progress = status.get("progress", rjob.progress)

        if on_progress:
            pct = 25 + int(rjob.progress * 0.65)  # scale 0-100 -> 25-90
            on_progress(min(pct, 90), status.get("message", "Processing on remote node"))

        if rjob.status == "complete":
            rjob.result_path = status.get("result", {}).get("output", "")
            break
        elif rjob.status == "error":
            rjob.error = status.get("error", "Remote job failed")
            break

        time.sleep(POLL_INTERVAL)
    else:
        rjob.status = "error"
        rjob.error = "Remote job timed out after polling"

    if on_progress:
        on_progress(95, f"Remote job {rjob.status}")

    # Update tracked job
    with _jobs_lock:
        _remote_jobs[local_id] = rjob

    return rjob


def get_remote_job(local_id: str) -> Optional[RemoteJob]:
    """Retrieve a tracked remote job by local ID."""
    with _jobs_lock:
        return _remote_jobs.get(local_id)


def list_remote_jobs() -> List[RemoteJob]:
    """Return all tracked remote jobs."""
    with _jobs_lock:
        return list(_remote_jobs.values())


def download_result(node_url: str, remote_job_id: str, local_path: str,
                    on_progress: Optional[Callable] = None) -> str:
    """Download the result file from a completed remote job.

    Args:
        node_url: Base URL of the remote node.
        remote_job_id: The job ID on the remote node.
        local_path: Local path where the result should be saved.

    Returns:
        The local path of the downloaded file.
    """
    node_url = node_url.rstrip("/")
    node = get_node(node_url)
    api_key = node.api_key if node else ""

    if on_progress:
        on_progress(10, "Requesting result download")

    download_url = f"{node_url}/api/remote/download-result/{remote_job_id}"
    hdrs = {"User-Agent": USER_AGENT}
    if api_key:
        hdrs["Authorization"] = f"Bearer {api_key}"

    req = Request(download_url, headers=hdrs)
    try:
        with urlopen(req, timeout=UPLOAD_TIMEOUT) as resp:
            os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
            with open(local_path, "wb") as fh:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if on_progress and total > 0:
                        pct = 10 + int((downloaded / total) * 85)
                        on_progress(min(pct, 95), "Downloading result")
    except HTTPError as exc:
        raise RuntimeError(f"Download failed: HTTP {exc.code}")
    except URLError as exc:
        raise RuntimeError(f"Download failed: {exc.reason}")

    if on_progress:
        on_progress(100, "Download complete")

    return local_path


def auto_select_node(job_type: str = "",
                     on_progress: Optional[Callable] = None) -> Optional[RemoteNode]:
    """Select the best available node for the given job type.

    Selection criteria:
      1. Node must be online (status != "offline").
      2. If *job_type* specified, node must list it in capabilities.
      3. Prefer the default node if it qualifies.
      4. Fall back to the most recently pinged node.

    Returns ``None`` if no suitable node is found.
    """
    if on_progress:
        on_progress(20, "Selecting best node")

    with _registry_lock:
        candidates = []
        for url, node in _registry.nodes.items():
            if node.status == "offline":
                continue
            if job_type and node.capabilities and job_type not in node.capabilities:
                continue
            candidates.append(node)

        if not candidates:
            return None

        # Prefer default node
        if _registry.default_node:
            for c in candidates:
                if c.url == _registry.default_node:
                    if on_progress:
                        on_progress(100, f"Selected default node: {c.name}")
                    return c

        # Fall back to most recently pinged
        best = max(candidates, key=lambda n: n.last_ping)
        if on_progress:
            on_progress(100, f"Selected node: {best.name}")
        return best


def set_default_node(url: str) -> bool:
    """Set the default node URL.  Returns True if the node exists."""
    url = url.rstrip("/")
    with _registry_lock:
        if url in _registry.nodes:
            _registry.default_node = url
            _save_registry()
            return True
    return False


def get_registry() -> NodeRegistry:
    """Return the current node registry."""
    with _registry_lock:
        return _registry
