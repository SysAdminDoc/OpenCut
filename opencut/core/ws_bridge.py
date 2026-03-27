"""
WebSocket Bridge for Premiere Pro

Real-time bidirectional communication between the OpenCut backend and the
CEP/UXP extension panel, replacing evalScript polling for live operations.

Based on the PremiereRemote pattern:
- Backend runs a WebSocket server alongside Flask
- Extension panel connects as a client
- Supports: progress streaming, live preview, timeline sync, remote commands

Protocol:
  Client -> Server: {"type": "command", "action": "...", "params": {...}, "id": "..."}
  Server -> Client: {"type": "response", "id": "...", "data": {...}}
  Server -> Client: {"type": "event", "event": "...", "data": {...}}
  Server -> Client: {"type": "progress", "job_id": "...", "percent": N, "message": "..."}
"""

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger("opencut")

# Global bridge instance
_bridge: Optional["WebSocketBridge"] = None
_bridge_lock = threading.Lock()


@dataclass
class WSClient:
    """A connected WebSocket client."""
    client_id: str
    ws: Any  # websocket connection object
    connected_at: float
    client_type: str = "unknown"  # "cep", "uxp", "external"
    subscriptions: Set[str] = field(default_factory=set)


class WebSocketBridge:
    """
    WebSocket server for real-time Premiere Pro communication.

    Runs in a background thread alongside the Flask HTTP server.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5680):
        self.host = host
        self.port = port
        self._clients: Dict[str, WSClient] = {}
        self._clients_lock = threading.Lock()
        self._server = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._handlers: Dict[str, Callable] = {}
        self._event_listeners: Dict[str, List[Callable]] = {}

        # Register built-in handlers
        self._register_builtin_handlers()

    def _register_builtin_handlers(self):
        """Register default command handlers."""
        self.register_handler("ping", self._handle_ping)
        self.register_handler("subscribe", self._handle_subscribe)
        self.register_handler("unsubscribe", self._handle_unsubscribe)
        self.register_handler("get_status", self._handle_get_status)
        self.register_handler("get_jobs", self._handle_get_jobs)

    def register_handler(self, action: str, handler: Callable):
        """Register a handler for a specific command action."""
        self._handlers[action] = handler
        logger.debug("Registered WS handler: %s", action)

    def on_event(self, event_name: str, listener: Callable):
        """Register a listener for a specific event type."""
        if event_name not in self._event_listeners:
            self._event_listeners[event_name] = []
        self._event_listeners[event_name].append(listener)

    def start(self):
        """Start the WebSocket server in a background thread."""
        if self._running:
            logger.warning("WebSocket bridge already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_server, daemon=True, name="ws-bridge")
        self._thread.start()
        logger.info("WebSocket bridge starting on ws://%s:%d", self.host, self.port)

    def stop(self):
        """Stop the WebSocket server."""
        self._running = False
        if self._server:
            try:
                self._server.shutdown()
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("WebSocket bridge stopped")

    def _run_server(self):
        """Run the WebSocket server (blocking, called in thread)."""
        try:
            import asyncio

            import websockets  # noqa: F401
            import websockets.server  # noqa: F401

            async def _handler(websocket):
                client_id = str(uuid.uuid4())[:8]
                client = WSClient(
                    client_id=client_id,
                    ws=websocket,
                    connected_at=time.monotonic(),
                )

                with self._clients_lock:
                    self._clients[client_id] = client

                logger.info("WS client connected: %s", client_id)
                self._emit_event("client_connected", {"client_id": client_id})

                try:
                    async for message in websocket:
                        await self._handle_message(client, message)
                except Exception as e:
                    if self._running:
                        logger.debug("WS client %s disconnected: %s", client_id, e)
                finally:
                    with self._clients_lock:
                        self._clients.pop(client_id, None)
                    logger.info("WS client disconnected: %s", client_id)
                    self._emit_event("client_disconnected", {"client_id": client_id})

            async def _serve():
                async with websockets.serve(
                    _handler,
                    self.host,
                    self.port,
                    ping_interval=20,
                    ping_timeout=30,
                    max_size=10 * 1024 * 1024,  # 10MB
                ) as server:
                    self._server = server
                    logger.info("WebSocket bridge listening on ws://%s:%d", self.host, self.port)
                    # Run until stopped
                    while self._running:
                        await asyncio.sleep(0.5)

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            try:
                loop.run_until_complete(_serve())
            finally:
                self._loop = None
                loop.close()

        except ImportError:
            logger.warning(
                "websockets package not installed. WebSocket bridge disabled.\n"
                "Install with: pip install websockets"
            )
        except OSError as e:
            logger.error("WebSocket bridge failed to start: %s", e)
        except Exception as e:
            if self._running:
                logger.error("WebSocket bridge error: %s", e)

    async def _handle_message(self, client: WSClient, raw: str):
        """Handle an incoming WebSocket message."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await self._send(client, {"type": "error", "error": "Invalid JSON"})
            return

        msg_type = msg.get("type", "")
        msg_id = msg.get("id", "")

        if msg_type == "identify":
            # Client identifying itself
            client.client_type = msg.get("client_type", "unknown")
            await self._send(client, {
                "type": "response",
                "id": msg_id,
                "data": {"client_id": client.client_id, "status": "identified"},
            })

        elif msg_type == "command":
            action = msg.get("action", "")
            params = msg.get("params", {})
            handler = self._handlers.get(action)

            if handler:
                try:
                    result = handler(client, params)
                    await self._send(client, {
                        "type": "response",
                        "id": msg_id,
                        "data": result if result else {},
                    })
                except Exception as e:
                    await self._send(client, {
                        "type": "error",
                        "id": msg_id,
                        "error": str(e),
                    })
            else:
                await self._send(client, {
                    "type": "error",
                    "id": msg_id,
                    "error": f"Unknown action: {action}",
                })

        else:
            # Emit as generic event
            self._emit_event(msg_type, msg)

    async def _send(self, client: WSClient, data: dict):
        """Send a message to a specific client."""
        try:
            await client.ws.send(json.dumps(data))
        except Exception as e:
            logger.debug("Failed to send to %s: %s", client.client_id, e)

    def broadcast(self, data: dict, event_type: Optional[str] = None):
        """
        Broadcast a message to all connected clients.
        If event_type is set, only send to clients subscribed to that event.
        Thread-safe — can be called from any thread (Flask routes, job threads, etc.).
        """
        import asyncio

        loop = getattr(self, "_loop", None)
        if not loop or loop.is_closed():
            return

        msg = json.dumps(data)

        with self._clients_lock:
            targets = list(self._clients.values())

        if event_type:
            targets = [c for c in targets if event_type in c.subscriptions or not c.subscriptions]

        for client in targets:
            try:
                asyncio.run_coroutine_threadsafe(client.ws.send(msg), loop)
            except Exception:
                pass

    def send_progress(self, job_id: str, percent: int, message: str = ""):
        """Send job progress to all connected clients."""
        self.broadcast({
            "type": "progress",
            "job_id": job_id,
            "percent": percent,
            "message": message,
        })

    def send_job_complete(self, job_id: str, result: dict):
        """Notify clients that a job completed."""
        self.broadcast({
            "type": "event",
            "event": "job_complete",
            "data": {"job_id": job_id, "result": result},
        })

    def send_job_error(self, job_id: str, error: str):
        """Notify clients that a job failed."""
        self.broadcast({
            "type": "event",
            "event": "job_error",
            "data": {"job_id": job_id, "error": error},
        })

    def send_timeline_update(self, data: dict):
        """Notify clients of a timeline change."""
        self.broadcast({
            "type": "event",
            "event": "timeline_update",
            "data": data,
        }, event_type="timeline")

    def _emit_event(self, event_name: str, data: dict):
        """Emit an event to registered listeners."""
        listeners = self._event_listeners.get(event_name, [])
        for listener in listeners:
            try:
                listener(data)
            except Exception as e:
                logger.debug("Event listener error (%s): %s", event_name, e)

    @property
    def client_count(self) -> int:
        with self._clients_lock:
            return len(self._clients)

    @property
    def is_running(self) -> bool:
        return self._running

    # -----------------------------------------------------------------------
    # Built-in command handlers
    # -----------------------------------------------------------------------

    def _handle_ping(self, client: WSClient, params: dict) -> dict:
        return {"pong": True, "time": time.time()}

    def _handle_subscribe(self, client: WSClient, params: dict) -> dict:
        events = params.get("events", [])
        if isinstance(events, list):
            client.subscriptions.update(events)
        return {"subscribed": list(client.subscriptions)}

    def _handle_unsubscribe(self, client: WSClient, params: dict) -> dict:
        events = params.get("events", [])
        if isinstance(events, list):
            client.subscriptions.difference_update(events)
        return {"subscribed": list(client.subscriptions)}

    def _handle_get_status(self, client: WSClient, params: dict) -> dict:
        with self._clients_lock:
            clients = [
                {"id": c.client_id, "type": c.client_type}
                for c in self._clients.values()
            ]
        return {"clients": clients, "count": len(clients)}

    def _handle_get_jobs(self, client: WSClient, params: dict) -> dict:
        try:
            from opencut.jobs import _list_jobs_copy
            return {"jobs": _list_jobs_copy()}
        except Exception:
            return {"jobs": []}


# ---------------------------------------------------------------------------
# Module-level convenience API
# ---------------------------------------------------------------------------

def get_bridge() -> Optional[WebSocketBridge]:
    """Get the global WebSocket bridge instance."""
    return _bridge


def init_bridge(host: str = "127.0.0.1", port: int = 5680) -> WebSocketBridge:
    """Initialize and start the global WebSocket bridge."""
    global _bridge
    with _bridge_lock:
        if _bridge and _bridge.is_running:
            return _bridge
        _bridge = WebSocketBridge(host=host, port=port)
        _bridge.start()
        return _bridge


def stop_bridge():
    """Stop the global WebSocket bridge."""
    global _bridge
    with _bridge_lock:
        if _bridge:
            _bridge.stop()
            _bridge = None


def check_websocket_available() -> bool:
    """Check if websockets package is installed."""
    try:
        import websockets  # noqa: F401
        return True
    except ImportError:
        return False
