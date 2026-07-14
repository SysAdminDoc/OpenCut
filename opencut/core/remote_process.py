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
import re
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from opencut.core.url_safety import (
    DEFAULT_MAX_DOWNLOAD_BYTES,
    transactional_download,
    validate_media_download,
)
from opencut.credential_store import (
    load_and_migrate_secrets,
    persist_secret_changes,
    secret_id,
)
from opencut.security import validate_path

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
REDACTED_API_KEY = "[REDACTED]"
MAX_REMOTE_RESULT_BYTES = DEFAULT_MAX_DOWNLOAD_BYTES
_REMOTE_JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


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
        data = asdict(self)
        data["api_key_set"] = bool(data.get("api_key"))
        if data.get("api_key"):
            data["api_key"] = REDACTED_API_KEY
        return data

    def to_storage_dict(self, *, include_secret: bool = False) -> dict:
        data = asdict(self)
        api_key = data.pop("api_key", "")
        data["api_key_set"] = bool(api_key)
        if include_secret:
            data["api_key"] = api_key
        return data


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


def _read_registry_metadata() -> dict:
    if not os.path.isfile(REGISTRY_PATH):
        return {"default_node": "", "nodes": {}}
    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict) or not isinstance(data.get("nodes", {}), dict):
            return {"default_node": "", "nodes": {}}
        return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load node registry metadata: %s", exc)
        return {"default_node": "", "nodes": {}}


def _write_registry_metadata(payload: dict) -> None:
    _ensure_dir()
    directory = os.path.dirname(REGISTRY_PATH) or "."
    fd, tmp_path = tempfile.mkstemp(
        dir=directory,
        prefix=os.path.basename(REGISTRY_PATH) + ".",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, REGISTRY_PATH)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _save_registry() -> None:
    """Persist registry after vault verification. Caller holds the lock."""
    old_nodes = _read_registry_metadata().get("nodes", {})
    changes: dict[str, Optional[str]] = {}
    for url, node in _registry.nodes.items():
        old = old_nodes.get(url, {})
        if node.api_key or old.get("api_key") or old.get("api_key_set"):
            changes[secret_id("remote-node/api-key", url)] = node.api_key or None
    for url, old in old_nodes.items():
        if (
            url not in _registry.nodes
            and (old.get("api_key") or old.get("api_key_set"))
        ):
            changes[secret_id("remote-node/api-key", url)] = None

    def persist_metadata(secure: bool) -> None:
        payload = {
            "_credential_storage": "os_vault" if secure else "plaintext-opt-in",
            "default_node": _registry.default_node,
            "nodes": {
                url: node.to_storage_dict(include_secret=not secure)
                for url, node in _registry.nodes.items()
            },
        }
        _write_registry_metadata(payload)

    persist_secret_changes(changes, persist_metadata)


def _load_registry(*, migrate: bool = True) -> NodeRegistry:
    """Load registry metadata and optionally migrate legacy API keys."""
    global _registry
    data = _read_registry_metadata()
    stored_nodes = data.get("nodes", {})
    identifiers = {
        url: secret_id("remote-node/api-key", url)
        for url in stored_nodes
    }
    legacy = {
        url: str(metadata.get("api_key") or "")
        for url, metadata in stored_nodes.items()
    }

    def persist_sanitized() -> None:
        nodes = {}
        for url, metadata in stored_nodes.items():
            item = dict(metadata)
            value = str(item.pop("api_key", "") or "")
            item["api_key_set"] = bool(value or item.get("api_key_set"))
            nodes[url] = item
        _write_registry_metadata(
            {
                "_credential_storage": "os_vault",
                "default_node": data.get("default_node", ""),
                "nodes": nodes,
            }
        )

    if migrate:
        secrets = load_and_migrate_secrets(identifiers, legacy, persist_sanitized)
    else:
        # Import-time loading must never prompt or touch the user's OS vault.
        # create_app() calls reload_registry() during startup migrations.
        secrets = dict(legacy)

    registry = NodeRegistry(default_node=data.get("default_node", ""))
    for url, metadata in stored_nodes.items():
        registry.nodes[url] = RemoteNode(
            url=metadata.get("url", url),
            api_key=secrets.get(url, ""),
            name=metadata.get("name", ""),
            capabilities=metadata.get("capabilities", []),
            status=metadata.get("status", "unknown"),
            last_ping=metadata.get("last_ping", 0.0),
        )
    _registry = registry
    return _registry


def reload_registry() -> NodeRegistry:
    """Reload and migrate the persisted registry at application startup."""
    with _registry_lock:
        return _load_registry(migrate=True)


# Load on import
_load_registry(migrate=False)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def normalize_node_url(url: str) -> str:
    """Return a canonical remote-node base URL after SSRF-oriented validation."""
    from opencut.core.url_safety import validate_public_http_url

    cleaned = validate_public_http_url(url, label="Remote node URL")
    parsed = urlparse(cleaned)
    if parsed.query or parsed.fragment:
        raise ValueError("Remote node URL must not include query or fragment")
    return cleaned.rstrip("/")


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
    url = normalize_node_url(url)

    with _registry_lock:
        previous = _registry.nodes.get(url)
        effective_api_key = api_key or (previous.api_key if previous else "")

    # Ping the node
    health = ping_node(url, api_key=effective_api_key)

    capabilities = health.get("capabilities", [])
    if not capabilities:
        # Assume basic capabilities if not reported
        capabilities = ["video", "audio", "export"]

    node = RemoteNode(
        url=url,
        api_key=effective_api_key,
        name=name or health.get("name", url),
        capabilities=capabilities,
        status="online",
        last_ping=time.time(),
    )

    with _registry_lock:
        previous = _registry.nodes.get(url)
        previous_default = _registry.default_node
        _registry.nodes[url] = node
        if not _registry.default_node:
            _registry.default_node = url
        try:
            _save_registry()
        except Exception:
            if previous is None:
                _registry.nodes.pop(url, None)
            else:
                _registry.nodes[url] = previous
            _registry.default_node = previous_default
            raise

    if on_progress:
        on_progress(100, f"Node registered: {node.name}")

    return node


def list_nodes() -> List[RemoteNode]:
    """Return all registered nodes."""
    with _registry_lock:
        return list(_registry.nodes.values())


def get_node(url: str) -> Optional[RemoteNode]:
    """Return a single node by URL, or None."""
    url = normalize_node_url(url)
    with _registry_lock:
        return _registry.nodes.get(url)


def remove_node(url: str) -> bool:
    """Unregister a node.  Returns True if it existed."""
    url = normalize_node_url(url)
    with _registry_lock:
        if url in _registry.nodes:
            previous = _registry.nodes.pop(url)
            previous_default = _registry.default_node
            if _registry.default_node == url:
                _registry.default_node = next(iter(_registry.nodes), "")
            try:
                _save_registry()
            except Exception:
                _registry.nodes[url] = previous
                _registry.default_node = previous_default
                raise
            return True
    return False


def ping_node(url: str, api_key: str = "",
              on_progress: Optional[Callable] = None) -> dict:
    """Check health of a remote node via ``GET /health``.

    Returns the parsed JSON health response.
    Raises ``RuntimeError`` on failure.
    """
    url = normalize_node_url(url)
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
    node_url = normalize_node_url(node_url)
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


def _remote_result_root() -> str:
    """Return the only directory where remote artifacts may be promoted."""
    configured = (os.environ.get("OPENCUT_OUTPUT_DIR") or "").strip()
    root = configured or os.path.join(OPENCUT_DIR, "remote_results")
    return validate_path(os.path.expanduser(root))


def _resolve_remote_result_path(local_path: str) -> str:
    """Resolve a caller-selected result path inside the approved output root."""
    if not isinstance(local_path, str) or not local_path.strip():
        raise ValueError("Remote result path is required")
    try:
        root = _remote_result_root()
        os.makedirs(root, exist_ok=True)
        requested = os.path.expanduser(local_path.strip())
        candidate = requested if os.path.isabs(requested) else os.path.join(root, requested)
        resolved = validate_path(candidate, allowed_base=root)
        os.makedirs(os.path.dirname(resolved), exist_ok=True)
        # Re-resolve after directory creation so a pre-existing symlink cannot
        # become an escape only when its parent materializes.
        return validate_path(resolved, allowed_base=root)
    except ValueError as exc:
        raise PermissionError(
            "Remote result path is outside the approved output directory"
        ) from exc


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
    node_url = normalize_node_url(node_url)
    if not isinstance(remote_job_id, str) or not _REMOTE_JOB_ID_RE.fullmatch(remote_job_id):
        raise PermissionError("Invalid remote job ID")
    final_path = _resolve_remote_result_path(local_path)
    node = get_node(node_url)
    api_key = node.api_key if node else ""

    if on_progress:
        on_progress(10, "Requesting result download")

    download_url = f"{node_url}/api/remote/download-result/{remote_job_id}"
    hdrs = {"User-Agent": USER_AGENT}
    if api_key:
        hdrs["Authorization"] = f"Bearer {api_key}"

    def _report_progress(written: int, total: int) -> None:
        if on_progress and total > 0:
            pct = 10 + int((written / total) * 85)
            on_progress(min(pct, 95), "Downloading result")

    result = transactional_download(
        download_url,
        final_path,
        max_bytes=MAX_REMOTE_RESULT_BYTES,
        timeout=UPLOAD_TIMEOUT,
        validator=validate_media_download,
        allowed_content_types=(
            "video/",
            "audio/",
            "image/",
            "application/octet-stream",
        ),
        headers=hdrs,
        label="remote result download",
        local_alternative="local processing",
        on_chunk=_report_progress,
    )

    if on_progress:
        on_progress(100, "Download complete")

    return result.path


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
    url = normalize_node_url(url)
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
