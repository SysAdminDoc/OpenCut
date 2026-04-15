"""
OpenCut Cloud / Remote Render Dispatch

Define render nodes (name, host, port, capabilities), health-check via
HTTP ping, dispatch render jobs with load balancing (least-loaded first),
track per-node job state, retry on failure (max 2), aggregate progress,
and fall back to local render when all nodes are unavailable.
"""

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from opencut.helpers import OPENCUT_DIR, get_ffmpeg_path, run_ffmpeg

logger = logging.getLogger("opencut")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_NODES_FILE = os.path.join(OPENCUT_DIR, "render_nodes.json")
_nodes_lock = threading.Lock()

MAX_RETRIES = 2
HEALTH_CHECK_TIMEOUT = 5  # seconds
JOB_SUBMIT_TIMEOUT = 30
JOB_POLL_TIMEOUT = 10
JOB_POLL_INTERVAL = 2.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class RenderNode:
    """A single remote render node."""
    name: str
    host: str
    port: int = 9090
    capabilities: List[str] = field(default_factory=lambda: ["cpu"])
    max_concurrent: int = 2
    enabled: bool = True
    priority: int = 0
    auth_token: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "host": self.host,
            "port": self.port,
            "capabilities": list(self.capabilities),
            "max_concurrent": self.max_concurrent,
            "enabled": self.enabled,
            "priority": self.priority,
            "auth_token": self.auth_token,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RenderNode":
        return cls(
            name=d.get("name", ""),
            host=d.get("host", ""),
            port=int(d.get("port", 9090)),
            capabilities=d.get("capabilities", ["cpu"]),
            max_concurrent=int(d.get("max_concurrent", 2)),
            enabled=d.get("enabled", True),
            priority=int(d.get("priority", 0)),
            auth_token=d.get("auth_token", ""),
        )

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class NodeStatus:
    """Health/status snapshot for a render node."""
    name: str
    online: bool = False
    active_jobs: int = 0
    max_concurrent: int = 2
    capabilities: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    last_check: float = 0.0
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "online": self.online,
            "active_jobs": self.active_jobs,
            "max_concurrent": self.max_concurrent,
            "capabilities": list(self.capabilities),
            "latency_ms": round(self.latency_ms, 1),
            "last_check": self.last_check,
            "error": self.error,
        }

    @property
    def load_ratio(self) -> float:
        if self.max_concurrent <= 0:
            return 1.0
        return self.active_jobs / self.max_concurrent


@dataclass
class CloudRenderResult:
    """Result from a cloud render dispatch."""
    node_used: str = ""
    job_id: str = ""
    status: str = ""
    output_url: str = ""
    output_path: str = ""
    render_time_ms: float = 0.0
    retries: int = 0
    fallback_local: bool = False

    def to_dict(self) -> dict:
        return {
            "node_used": self.node_used,
            "job_id": self.job_id,
            "status": self.status,
            "output_url": self.output_url,
            "output_path": self.output_path,
            "render_time_ms": round(self.render_time_ms, 1),
            "retries": self.retries,
            "fallback_local": self.fallback_local,
        }


@dataclass
class RemoteJobState:
    """Tracking state for a job dispatched to a remote node."""
    job_id: str
    node_name: str
    status: str = "submitted"  # submitted, rendering, complete, failed
    progress: int = 0
    output_url: str = ""
    error: str = ""
    submitted_at: float = 0.0
    completed_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "node_name": self.node_name,
            "status": self.status,
            "progress": self.progress,
            "output_url": self.output_url,
            "error": self.error,
            "submitted_at": self.submitted_at,
            "completed_at": self.completed_at,
        }


# ---------------------------------------------------------------------------
# In-memory node status cache
# ---------------------------------------------------------------------------
_node_statuses: Dict[str, NodeStatus] = {}
_node_status_lock = threading.Lock()

# In-memory tracking of dispatched jobs
_remote_jobs: Dict[str, RemoteJobState] = {}
_remote_jobs_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Node config persistence
# ---------------------------------------------------------------------------
def _ensure_config_dir():
    os.makedirs(OPENCUT_DIR, exist_ok=True)


def load_nodes() -> List[RenderNode]:
    """Load render nodes from persistent config file."""
    with _nodes_lock:
        if not os.path.isfile(_NODES_FILE):
            return []
        try:
            with open(_NODES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [RenderNode.from_dict(d) for d in data if d.get("name")]
        except Exception as e:
            logger.warning("Failed to load render nodes config: %s", e)
            return []


def save_nodes(nodes: List[RenderNode]):
    """Save render nodes to persistent config file."""
    _ensure_config_dir()
    with _nodes_lock:
        data = [n.to_dict() for n in nodes]
        tmp = _NODES_FILE + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, _NODES_FILE)
        except Exception as e:
            logger.error("Failed to save render nodes config: %s", e)
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


def add_node(node: RenderNode) -> List[RenderNode]:
    """Add or update a render node. Returns updated node list."""
    nodes = load_nodes()
    existing = [n for n in nodes if n.name != node.name]
    existing.append(node)
    existing.sort(key=lambda n: n.name)
    save_nodes(existing)
    return existing


def remove_node(name: str) -> List[RenderNode]:
    """Remove a render node by name. Returns updated node list."""
    nodes = load_nodes()
    filtered = [n for n in nodes if n.name != name]
    if len(filtered) == len(nodes):
        raise ValueError(f"Node not found: {name}")
    save_nodes(filtered)
    with _node_status_lock:
        _node_statuses.pop(name, None)
    return filtered


def get_node(name: str) -> Optional[RenderNode]:
    """Get a specific node by name."""
    for n in load_nodes():
        if n.name == name:
            return n
    return None


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def _http_request(url: str, method: str = "GET", data: Optional[bytes] = None,
                  headers: Optional[dict] = None,
                  timeout: int = 10) -> dict:
    """Make an HTTP request and return parsed JSON response."""
    hdrs = {"Content-Type": "application/json", "Accept": "application/json"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            if body.strip():
                return json.loads(body)
            return {"status": "ok"}
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        raise RuntimeError(f"HTTP {e.code} from {url}: {body}") from e
    except urllib.error.URLError as e:
        raise ConnectionError(f"Cannot reach {url}: {e.reason}") from e
    except Exception as e:
        raise ConnectionError(f"Request to {url} failed: {e}") from e


def _build_auth_headers(node: RenderNode) -> dict:
    """Build auth headers if the node has an auth token."""
    if node.auth_token:
        return {"Authorization": f"Bearer {node.auth_token}"}
    return {}


# ---------------------------------------------------------------------------
# Health checking
# ---------------------------------------------------------------------------
def check_node_health(node: RenderNode) -> NodeStatus:
    """Ping a render node and return its status."""
    status = NodeStatus(
        name=node.name,
        capabilities=list(node.capabilities),
        max_concurrent=node.max_concurrent,
        last_check=time.time(),
    )
    try:
        start = time.time()
        url = f"{node.base_url}/health"
        resp = _http_request(url, headers=_build_auth_headers(node),
                             timeout=HEALTH_CHECK_TIMEOUT)
        elapsed_ms = (time.time() - start) * 1000
        status.online = True
        status.latency_ms = elapsed_ms
        status.active_jobs = int(resp.get("active_jobs", 0))
        caps = resp.get("capabilities", None)
        if caps and isinstance(caps, list):
            status.capabilities = caps
    except Exception as e:
        status.online = False
        status.error = str(e)[:200]
        logger.debug("Node %s health check failed: %s", node.name, e)

    with _node_status_lock:
        _node_statuses[node.name] = status
    return status


def check_all_nodes(on_progress: Optional[Callable] = None) -> List[NodeStatus]:
    """Health-check all configured nodes. Returns list of NodeStatus."""
    nodes = load_nodes()
    results = []
    for i, node in enumerate(nodes):
        if not node.enabled:
            st = NodeStatus(name=node.name, online=False, error="disabled")
            with _node_status_lock:
                _node_statuses[node.name] = st
            results.append(st)
            continue
        status = check_node_health(node)
        results.append(status)
        if on_progress:
            pct = int(((i + 1) / max(len(nodes), 1)) * 100)
            on_progress(pct)
    return results


def get_cached_statuses() -> Dict[str, NodeStatus]:
    """Return a copy of the cached node statuses."""
    with _node_status_lock:
        return dict(_node_statuses)


# ---------------------------------------------------------------------------
# Node selection (least-loaded first)
# ---------------------------------------------------------------------------
def _select_node(required_capabilities: Optional[List[str]] = None,
                 exclude_nodes: Optional[List[str]] = None) -> Optional[RenderNode]:
    """Select the best available node based on load and capabilities."""
    nodes = load_nodes()
    exclude = set(exclude_nodes or [])
    with _node_status_lock:
        statuses = dict(_node_statuses)

    candidates = []
    for node in nodes:
        if not node.enabled or node.name in exclude:
            continue
        status = statuses.get(node.name)
        if not status or not status.online:
            continue
        if status.active_jobs >= node.max_concurrent:
            continue
        if required_capabilities:
            node_caps = set(status.capabilities or node.capabilities)
            if not all(cap in node_caps for cap in required_capabilities):
                continue
        candidates.append((node, status))

    if not candidates:
        return None

    # Sort: priority DESC, load_ratio ASC, latency ASC
    candidates.sort(key=lambda x: (-x[0].priority, x[1].load_ratio, x[1].latency_ms))
    return candidates[0][0]


# ---------------------------------------------------------------------------
# Job dispatch
# ---------------------------------------------------------------------------
def _submit_to_node(node: RenderNode, render_config: dict) -> str:
    """Submit a render job to a remote node. Returns remote job ID."""
    url = f"{node.base_url}/render/submit"
    payload = json.dumps(render_config).encode("utf-8")
    headers = _build_auth_headers(node)
    resp = _http_request(url, method="POST", data=payload,
                         headers=headers, timeout=JOB_SUBMIT_TIMEOUT)
    remote_job_id = resp.get("job_id", "")
    if not remote_job_id:
        raise RuntimeError("Remote node did not return a job_id")
    return remote_job_id


def _poll_remote_job(node: RenderNode, remote_job_id: str) -> dict:
    """Poll a remote node for job status."""
    url = f"{node.base_url}/render/status/{remote_job_id}"
    headers = _build_auth_headers(node)
    return _http_request(url, headers=headers, timeout=JOB_POLL_TIMEOUT)


def _wait_for_remote_job(node: RenderNode, remote_job_id: str,
                         on_progress: Optional[Callable] = None,
                         timeout: int = 3600) -> dict:
    """Poll a remote job until completion or timeout."""
    start = time.time()
    last_progress = 0
    while time.time() - start < timeout:
        try:
            resp = _poll_remote_job(node, remote_job_id)
            status = resp.get("status", "unknown")
            progress = int(resp.get("progress", 0))

            with _remote_jobs_lock:
                if remote_job_id in _remote_jobs:
                    _remote_jobs[remote_job_id].status = status
                    _remote_jobs[remote_job_id].progress = progress

            if progress > last_progress and on_progress:
                on_progress(min(progress, 99))
                last_progress = progress

            if status == "complete":
                return resp
            if status in ("failed", "error"):
                error = resp.get("error", "Remote render failed")
                raise RuntimeError(error)

        except ConnectionError:
            logger.warning("Lost connection to node %s while polling job %s",
                           node.name, remote_job_id)
            raise
        except RuntimeError:
            raise
        except Exception as e:
            logger.debug("Poll error for %s on %s: %s", remote_job_id, node.name, e)

        time.sleep(JOB_POLL_INTERVAL)

    raise TimeoutError(f"Remote job {remote_job_id} timed out after {timeout}s")


# ---------------------------------------------------------------------------
# Local fallback render
# ---------------------------------------------------------------------------
def _render_local(input_path: str, render_config: dict,
                  on_progress: Optional[Callable] = None) -> dict:
    """Render locally using FFmpeg as a fallback."""
    from opencut.helpers import output_path as make_output_path

    codec = render_config.get("codec", "libx264")
    crf = render_config.get("crf", 18)
    preset = render_config.get("preset", "medium")
    output_dir = render_config.get("output_dir", "")
    out = render_config.get("output_path", "")

    if not out:
        suffix = "cloud_render"
        out = make_output_path(input_path, suffix, output_dir)

    if on_progress:
        on_progress(10)

    cmd = [
        get_ffmpeg_path(), "-hide_banner", "-y",
        "-i", input_path,
        "-c:v", codec, "-crf", str(crf), "-preset", preset,
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        out,
    ]

    if on_progress:
        on_progress(20)

    run_ffmpeg(cmd)

    if on_progress:
        on_progress(100)

    return {
        "output_path": out,
        "status": "complete",
        "fallback": True,
    }


# ---------------------------------------------------------------------------
# Main dispatch function
# ---------------------------------------------------------------------------
def dispatch_render(input_path: str,
                    render_config: Optional[dict] = None,
                    required_capabilities: Optional[List[str]] = None,
                    on_progress: Optional[Callable] = None,
                    timeout: int = 3600) -> CloudRenderResult:
    """
    Dispatch a render job to the best available cloud node.

    Falls back to local rendering if no nodes are available.
    Retries on different nodes up to MAX_RETRIES times.

    Args:
        input_path: Path to input video file.
        render_config: Dict with render settings (codec, crf, preset, etc).
        required_capabilities: List of required capabilities (e.g. ["gpu"]).
        on_progress: Progress callback taking int (0-100).
        timeout: Max wait time for remote job in seconds.

    Returns:
        CloudRenderResult with render outcome details.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    config = dict(render_config or {})
    config.setdefault("input_path", input_path)
    result = CloudRenderResult()
    start_time = time.time()

    # Refresh node health
    if on_progress:
        on_progress(5)
    check_all_nodes()

    excluded_nodes: List[str] = []
    attempts = 0

    while attempts <= MAX_RETRIES:
        node = _select_node(required_capabilities, excluded_nodes)
        if node is None:
            break

        try:
            if on_progress:
                on_progress(10)

            remote_job_id = _submit_to_node(node, config)

            # Track the job
            job_state = RemoteJobState(
                job_id=remote_job_id,
                node_name=node.name,
                status="submitted",
                submitted_at=time.time(),
            )
            with _remote_jobs_lock:
                _remote_jobs[remote_job_id] = job_state

            if on_progress:
                on_progress(15)

            # Wait for completion
            resp = _wait_for_remote_job(node, remote_job_id,
                                        on_progress=on_progress,
                                        timeout=timeout)

            # Success
            elapsed_ms = (time.time() - start_time) * 1000
            result.node_used = node.name
            result.job_id = remote_job_id
            result.status = "complete"
            result.output_url = resp.get("output_url", "")
            result.output_path = resp.get("output_path", "")
            result.render_time_ms = elapsed_ms
            result.retries = attempts

            with _remote_jobs_lock:
                if remote_job_id in _remote_jobs:
                    _remote_jobs[remote_job_id].status = "complete"
                    _remote_jobs[remote_job_id].output_url = result.output_url
                    _remote_jobs[remote_job_id].completed_at = time.time()

            if on_progress:
                on_progress(100)

            return result

        except Exception as e:
            logger.warning("Render failed on node %s (attempt %d): %s",
                           node.name, attempts + 1, e)
            excluded_nodes.append(node.name)
            attempts += 1

    # All nodes failed or none available -- fallback to local
    logger.info("All cloud nodes unavailable or failed, falling back to local render")
    if on_progress:
        on_progress(10)

    try:
        local_result = _render_local(input_path, config, on_progress=on_progress)
        elapsed_ms = (time.time() - start_time) * 1000
        result.node_used = "local"
        result.status = "complete"
        result.output_path = local_result.get("output_path", "")
        result.render_time_ms = elapsed_ms
        result.retries = attempts
        result.fallback_local = True
        return result
    except Exception as e:
        elapsed_ms = (time.time() - start_time) * 1000
        result.node_used = "local"
        result.status = "failed"
        result.render_time_ms = elapsed_ms
        result.retries = attempts
        result.fallback_local = True
        raise RuntimeError(f"Local fallback render also failed: {e}") from e


# ---------------------------------------------------------------------------
# Batch dispatch
# ---------------------------------------------------------------------------
def dispatch_batch(input_paths: List[str],
                   render_config: Optional[dict] = None,
                   required_capabilities: Optional[List[str]] = None,
                   on_progress: Optional[Callable] = None) -> List[CloudRenderResult]:
    """Dispatch multiple render jobs across available nodes."""
    results = []
    total = len(input_paths)
    for i, path in enumerate(input_paths):
        def _inner_progress(pct):
            if on_progress:
                base = int((i / max(total, 1)) * 100)
                step = int((1 / max(total, 1)) * 100)
                overall = base + int(pct * step / 100)
                on_progress(min(overall, 99))

        try:
            r = dispatch_render(path, render_config, required_capabilities,
                                on_progress=_inner_progress)
            results.append(r)
        except Exception as e:
            logger.error("Batch render failed for %s: %s", path, e)
            r = CloudRenderResult(status="failed", output_path="",
                                  node_used="", fallback_local=False)
            results.append(r)

    if on_progress:
        on_progress(100)
    return results


# ---------------------------------------------------------------------------
# Remote job tracking
# ---------------------------------------------------------------------------
def get_remote_jobs() -> List[dict]:
    """Return list of all tracked remote jobs."""
    with _remote_jobs_lock:
        return [j.to_dict() for j in _remote_jobs.values()]


def get_remote_job(job_id: str) -> Optional[dict]:
    """Get a specific remote job by ID."""
    with _remote_jobs_lock:
        job = _remote_jobs.get(job_id)
        return job.to_dict() if job else None


def clear_completed_jobs():
    """Remove completed/failed jobs from tracking."""
    with _remote_jobs_lock:
        to_remove = [
            jid for jid, j in _remote_jobs.items()
            if j.status in ("complete", "failed")
        ]
        for jid in to_remove:
            del _remote_jobs[jid]
    return len(to_remove)


# ---------------------------------------------------------------------------
# Node stats aggregation
# ---------------------------------------------------------------------------
def get_node_summary() -> dict:
    """Get summary of all nodes and their statuses."""
    nodes = load_nodes()
    with _node_status_lock:
        statuses = dict(_node_statuses)

    total = len(nodes)
    online = sum(1 for n in nodes if statuses.get(n.name, NodeStatus(name=n.name)).online)
    total_capacity = sum(n.max_concurrent for n in nodes if n.enabled)
    total_active = sum(
        statuses.get(n.name, NodeStatus(name=n.name)).active_jobs
        for n in nodes if n.enabled
    )

    gpu_nodes = sum(
        1 for n in nodes
        if "gpu" in (statuses.get(n.name, NodeStatus(name=n.name)).capabilities or n.capabilities)
    )

    return {
        "total_nodes": total,
        "online": online,
        "offline": total - online,
        "total_capacity": total_capacity,
        "total_active_jobs": total_active,
        "gpu_nodes": gpu_nodes,
        "utilization_pct": round(total_active / max(total_capacity, 1) * 100, 1),
    }
