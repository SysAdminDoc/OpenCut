"""
OBS Studio WebSocket v5 bridge.

Connects to an OBS Studio instance exposing the built-in
``obs-websocket`` (v5 protocol, GPL-2, shipped with OBS 28+) so
OpenCut can:

- enumerate scenes / sources,
- switch scenes programmatically (scene transitions for auto-record),
- start / stop recording,
- pull the current scene snapshot (for thumbnail generation or
  continuous-recording split points),
- subscribe to events (scene-changed, recording-started, item-added).

Primary use cases:
1. **Gaming vertical** — Twitch VOD clipping: watch for chat-triggered
   scene changes or hotkey events, record timestamps, hand them to
   OpenCut's highlight reel pipeline.
2. **Live production** — signal scene transitions from an external
   cue sheet.
3. **Tutorial capture** — split a long screen-record at each scene
   change so the resulting clips line up with lesson chapters.

The v5 protocol is JSON-over-WebSocket with a SHA-256 auth challenge.
We use the ``websockets`` pip package (already used elsewhere in
OpenCut — see ``checks.check_websocket_available``) and implement the
auth + message/request frames inline so we don't add a dedicated
obs-websocket-py dependency.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("opencut")


# obs-websocket v5 opcodes — mirrored from the protocol spec.
OP_HELLO = 0
OP_IDENTIFY = 1
OP_IDENTIFIED = 2
OP_REIDENTIFY = 3
OP_EVENT = 5
OP_REQUEST = 6
OP_REQUEST_RESPONSE = 7
OP_REQUEST_BATCH = 8
OP_REQUEST_BATCH_RESPONSE = 9

DEFAULT_TIMEOUT = 5.0   # seconds — request/response timeout
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4455


@dataclass
class ObsStatus:
    """High-level OBS connection + recording status."""
    connected: bool = False
    obs_version: str = ""
    websocket_version: str = ""
    recording: bool = False
    streaming: bool = False
    current_scene: str = ""
    scenes: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def __getitem__(self, key):
        return getattr(self, key)

    def __contains__(self, key):
        return key in self.__dataclass_fields__

    def keys(self):
        return self.__dataclass_fields__.keys()


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

def check_obs_bridge_available() -> bool:
    """True when the ``websockets`` pip package is importable.

    OBS itself running is a runtime question answered by
    :func:`ping` / :func:`connect`; this only checks that we can
    speak the WebSocket protocol.
    """
    try:
        import websockets  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def _compute_auth(password: str, salt: str, challenge: str) -> str:
    """Compute the obs-websocket v5 auth string from a Hello frame."""
    secret_bytes = hashlib.sha256((password + salt).encode("utf-8")).digest()
    secret = base64.b64encode(secret_bytes).decode("ascii")
    auth_bytes = hashlib.sha256((secret + challenge).encode("utf-8")).digest()
    return base64.b64encode(auth_bytes).decode("ascii")


# ---------------------------------------------------------------------------
# Sync wrapper around asyncio websockets client
# ---------------------------------------------------------------------------

class ObsClient:
    """Blocking WebSocket client for obs-websocket v5.

    Not thread-safe — one OBS connection per thread. The lifetime
    contract is a context-manager style: ``connect()`` → call
    methods → ``close()``.  A single background event-loop thread
    runs the async websocket client and marshals messages through
    blocking ``Queue`` objects.
    """

    def __init__(self, host: str = DEFAULT_HOST,
                 port: int = DEFAULT_PORT,
                 password: Optional[str] = None,
                 timeout: float = DEFAULT_TIMEOUT):
        self.host = host
        self.port = int(port)
        self.password = password or ""
        self.timeout = float(timeout)

        self._ws = None
        self._loop = None
        self._thread: Optional[threading.Thread] = None
        self._send_lock = threading.Lock()
        self._pending: Dict[str, "_FutureSlot"] = {}
        self._events: "list[Dict[str, Any]]" = []
        self._events_lock = threading.Lock()
        self._connected_event = threading.Event()
        self._close_event = threading.Event()
        self._connect_error: Optional[str] = None
        self._identified: bool = False

    # -- lifecycle ---------------------------------------------------------

    def connect(self) -> None:
        """Open the WebSocket + complete the Hello/Identify handshake.

        Blocks up to ``self.timeout * 3`` seconds; raises
        ``RuntimeError`` on timeout or auth failure.
        """
        if not check_obs_bridge_available():
            raise RuntimeError(
                "obs-websocket bridge requires the `websockets` pip package. "
                "Install: pip install websockets"
            )

        self._thread = threading.Thread(
            target=self._run_loop, name="obs-ws", daemon=True,
        )
        self._thread.start()

        # Wait for the handshake to complete OR for an error to be set.
        if not self._connected_event.wait(timeout=self.timeout * 3):
            self.close()
            raise RuntimeError(
                f"OBS WebSocket handshake timed out after {self.timeout * 3:.1f}s"
            )
        if self._connect_error:
            self.close()
            raise RuntimeError(self._connect_error)

    def close(self) -> None:
        self._close_event.set()
        if self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except RuntimeError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        self._ws = None
        self._loop = None

    def __enter__(self) -> "ObsClient":
        self.connect()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- request plumbing --------------------------------------------------

    def call(self, request_type: str, request_data: Optional[Dict[str, Any]] = None,
             timeout: Optional[float] = None) -> Dict[str, Any]:
        """Send a request frame and block for the matching response."""
        if self._loop is None or self._ws is None:
            raise RuntimeError("ObsClient not connected")
        if timeout is None:
            timeout = self.timeout

        rid = str(uuid.uuid4())
        payload = {
            "op": OP_REQUEST,
            "d": {
                "requestType": request_type,
                "requestId": rid,
                "requestData": request_data or {},
            },
        }

        slot = _FutureSlot()
        self._pending[rid] = slot

        with self._send_lock:
            self._schedule_send(payload)

        if not slot.event.wait(timeout=timeout):
            self._pending.pop(rid, None)
            raise RuntimeError(
                f"OBS request {request_type} timed out after {timeout:.1f}s"
            )

        self._pending.pop(rid, None)
        resp = slot.value or {}
        d = resp.get("d") or {}
        status = d.get("requestStatus") or {}
        if not status.get("result", False):
            code = status.get("code")
            comment = status.get("comment") or "Unknown error"
            raise RuntimeError(
                f"OBS request {request_type} failed (code={code}): {comment}"
            )
        return d.get("responseData") or {}

    def drain_events(self) -> List[Dict[str, Any]]:
        """Return + clear the queued event list."""
        with self._events_lock:
            out = list(self._events)
            self._events.clear()
        return out

    # -- internal coroutine machinery --------------------------------------

    def _run_loop(self) -> None:
        import asyncio
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run_client())
        except Exception as exc:  # noqa: BLE001
            self._connect_error = str(exc)
            self._connected_event.set()
        finally:
            try:
                self._loop.close()
            except Exception:  # noqa: BLE001
                pass

    async def _run_client(self) -> None:
        import websockets
        url = f"ws://{self.host}:{self.port}"
        try:
            # Let websockets negotiate the subprotocol.  OBS v5 uses
            # "obswebsocket.json" as the subprotocol identifier.
            async with websockets.connect(
                url,
                subprotocols=["obswebsocket.json"],
                open_timeout=self.timeout,
                ping_interval=20.0,
                ping_timeout=10.0,
            ) as ws:
                self._ws = ws
                await self._handshake(ws)
                await self._receive_loop(ws)
        except Exception as exc:  # noqa: BLE001
            self._connect_error = f"OBS connect error: {exc}"
            self._connected_event.set()

    async def _handshake(self, ws) -> None:
        import asyncio
        hello_raw = await asyncio.wait_for(ws.recv(), timeout=self.timeout)
        hello = json.loads(hello_raw)
        if hello.get("op") != OP_HELLO:
            raise RuntimeError(f"Unexpected first frame op={hello.get('op')}")

        d = hello.get("d") or {}
        auth_block = d.get("authentication")
        identify: Dict[str, Any] = {
            "op": OP_IDENTIFY,
            "d": {
                "rpcVersion": 1,
                "eventSubscriptions": 511,  # all Non-High-Volume
            },
        }
        if auth_block:
            if not self.password:
                raise RuntimeError(
                    "OBS requires authentication but no password was supplied"
                )
            identify["d"]["authentication"] = _compute_auth(
                self.password,
                str(auth_block.get("salt") or ""),
                str(auth_block.get("challenge") or ""),
            )

        await ws.send(json.dumps(identify))

        ident_raw = await asyncio.wait_for(ws.recv(), timeout=self.timeout)
        ident = json.loads(ident_raw)
        if ident.get("op") != OP_IDENTIFIED:
            raise RuntimeError(
                f"Expected Identified frame, got op={ident.get('op')}"
            )
        self._identified = True
        self._connected_event.set()

    async def _receive_loop(self, ws) -> None:
        import asyncio
        while not self._close_event.is_set():
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            except Exception:  # noqa: BLE001
                break
            try:
                msg = json.loads(raw)
            except Exception:  # noqa: BLE001
                continue

            op = msg.get("op")
            if op == OP_EVENT:
                with self._events_lock:
                    self._events.append({
                        "timestamp": time.time(),
                        "data": msg.get("d") or {},
                    })
            elif op == OP_REQUEST_RESPONSE:
                d = msg.get("d") or {}
                rid = d.get("requestId")
                if rid and rid in self._pending:
                    slot = self._pending[rid]
                    slot.value = msg
                    slot.event.set()

    def _schedule_send(self, payload: Dict[str, Any]) -> None:
        """Thread-safe send through the event loop."""
        if self._loop is None or self._ws is None:
            raise RuntimeError("OBS WebSocket not connected")
        msg = json.dumps(payload)

        async def _send():
            await self._ws.send(msg)

        try:
            future = __import__("asyncio").run_coroutine_threadsafe(_send(), self._loop)
            future.result(timeout=self.timeout)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"OBS send failed: {exc}") from exc


@dataclass
class _FutureSlot:
    event: threading.Event = field(default_factory=threading.Event)
    value: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# High-level operations
# ---------------------------------------------------------------------------

def ping(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
         password: Optional[str] = None, timeout: float = 2.0) -> bool:
    """Cheap reachability probe: connect + immediately disconnect.

    Returns ``True`` when OBS is reachable + identify succeeds.
    """
    try:
        with ObsClient(host=host, port=port, password=password, timeout=timeout):
            return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("OBS ping failed: %s", exc)
        return False


def status(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
           password: Optional[str] = None,
           timeout: float = DEFAULT_TIMEOUT) -> ObsStatus:
    """Return a one-shot :class:`ObsStatus` snapshot."""
    try:
        with ObsClient(host, port, password, timeout) as cli:
            version_info = cli.call("GetVersion") or {}
            record_info = cli.call("GetRecordStatus") or {}
            stream_info = cli.call("GetStreamStatus") or {}
            scene_info = cli.call("GetSceneList") or {}
            current = scene_info.get("currentProgramSceneName") or ""
            scenes = [
                s.get("sceneName", "")
                for s in (scene_info.get("scenes") or [])
                if s.get("sceneName")
            ]
            return ObsStatus(
                connected=True,
                obs_version=str(version_info.get("obsVersion") or ""),
                websocket_version=str(version_info.get("obsWebSocketVersion") or ""),
                recording=bool(record_info.get("outputActive")),
                streaming=bool(stream_info.get("outputActive")),
                current_scene=current,
                scenes=scenes,
            )
    except Exception as exc:  # noqa: BLE001
        return ObsStatus(connected=False, notes=[f"error: {exc}"])


def switch_scene(scene_name: str,
                 host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                 password: Optional[str] = None,
                 timeout: float = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """Switch OBS's program scene. Raises on failure."""
    if not scene_name or not scene_name.strip():
        raise ValueError("scene_name is required")
    with ObsClient(host, port, password, timeout) as cli:
        cli.call("SetCurrentProgramScene", {"sceneName": scene_name})
        return {"scene": scene_name, "ok": True}


def recording(action: str,
              host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
              password: Optional[str] = None,
              timeout: float = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """Control OBS recording.

    ``action`` = ``"start"`` / ``"stop"`` / ``"toggle"`` / ``"status"``.
    """
    with ObsClient(host, port, password, timeout) as cli:
        if action == "start":
            cli.call("StartRecord")
        elif action == "stop":
            resp = cli.call("StopRecord") or {}
            return {"stopped": True, "output_path": resp.get("outputPath", "")}
        elif action == "toggle":
            cli.call("ToggleRecord")
        elif action == "status":
            pass
        else:
            raise ValueError(
                "action must be one of: start, stop, toggle, status"
            )
        status_resp = cli.call("GetRecordStatus") or {}
        return {
            "active": bool(status_resp.get("outputActive")),
            "paused": bool(status_resp.get("outputPaused")),
            "duration": int(status_resp.get("outputDuration") or 0),
            "timecode": str(status_resp.get("outputTimecode") or ""),
        }


def take_screenshot(scene_name: Optional[str],
                    output_path: str,
                    host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                    password: Optional[str] = None,
                    width: int = 1280, height: int = 720,
                    timeout: float = DEFAULT_TIMEOUT) -> str:
    """Snapshot a scene (or the current program scene) to ``output_path``."""
    if not output_path or not output_path.strip():
        raise ValueError("output_path is required")
    out = os.path.abspath(output_path)
    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fmt = os.path.splitext(out)[1].lstrip(".").lower() or "png"
    if fmt not in ("png", "jpg", "jpeg", "bmp"):
        raise ValueError("output must end in .png / .jpg / .jpeg / .bmp")
    with ObsClient(host, port, password, timeout) as cli:
        if not scene_name:
            scene_info = cli.call("GetSceneList") or {}
            scene_name = scene_info.get("currentProgramSceneName") or ""
        if not scene_name:
            raise RuntimeError("Could not resolve a current program scene")
        cli.call(
            "SaveSourceScreenshot",
            {
                "sourceName": scene_name,
                "imageFormat": "jpeg" if fmt in ("jpg", "jpeg") else fmt,
                "imageFilePath": out,
                "imageWidth": max(64, min(7680, int(width))),
                "imageHeight": max(64, min(4320, int(height))),
            },
        )
    return out
