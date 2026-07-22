"""System realtime routes registered on the shared system blueprint."""

from .system import (
    get_json_dict,
    jsonify,
    request,
    require_csrf,
    safe_error,
    safe_int,
    socket,
    system_bp,
)

# ---------------------------------------------------------------------------
# WebSocket Bridge Control
# ---------------------------------------------------------------------------
WS_BRIDGE_HOST = "127.0.0.1"
WS_BRIDGE_DEFAULT_PORT = 5680
WS_BRIDGE_MAX_PORT = 5689


def _ws_bridge_url(port: int, host: str = WS_BRIDGE_HOST) -> str:
    return f"ws://{host}:{int(port)}"


def _request_port() -> int | None:
    try:
        return int(request.host.rsplit(":", 1)[1])
    except (IndexError, TypeError, ValueError):
        return None


def _port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, int(port)))
        except OSError:
            return False
    return True


def _select_ws_bridge_port(preferred: int) -> int:
    http_port = _request_port()
    candidates = [preferred] + list(range(WS_BRIDGE_DEFAULT_PORT, WS_BRIDGE_MAX_PORT + 1))
    seen: set[int] = set()
    for candidate in candidates:
        if candidate in seen or candidate == http_port:
            continue
        seen.add(candidate)
        if _port_is_free(WS_BRIDGE_HOST, candidate):
            return candidate
    raise RuntimeError("No available WebSocket bridge port in 5680-5689")


def _ws_bridge_payload(*, running: bool, clients: int, port: int = WS_BRIDGE_DEFAULT_PORT) -> dict:
    return {
        "running": bool(running),
        "clients": int(clients),
        "host": WS_BRIDGE_HOST,
        "port": int(port),
        "url": _ws_bridge_url(port),
    }


@system_bp.route("/ws/status", methods=["GET"])
def ws_status():
    """Get WebSocket bridge status."""
    try:
        from opencut.core.ws_bridge import get_bridge
        bridge = get_bridge()
        if bridge and bridge.is_running:
            return jsonify(_ws_bridge_payload(
                running=True,
                clients=bridge.client_count,
                port=bridge.port,
            ))
        return jsonify(_ws_bridge_payload(running=False, clients=0))
    except Exception:
        return jsonify(_ws_bridge_payload(running=False, clients=0))


@system_bp.route("/ws/start", methods=["POST"])
@require_csrf
def ws_start():
    """Start the WebSocket bridge."""
    try:
        from opencut.core.ws_bridge import check_websocket_available, get_bridge, init_bridge
        if not check_websocket_available():
            return jsonify({"error": "websockets package not installed. pip install websockets"}), 400
        bridge = get_bridge()
        if bridge and bridge.is_running:
            return jsonify(_ws_bridge_payload(
                running=True,
                clients=bridge.client_count,
                port=bridge.port,
            ) | {"success": True, "message": f"WebSocket bridge already running on port {bridge.port}"})
        data = get_json_dict() if request.is_json else {}
        requested_port = safe_int(data.get("port", WS_BRIDGE_DEFAULT_PORT), WS_BRIDGE_DEFAULT_PORT, min_val=1024, max_val=65535)
        port = _select_ws_bridge_port(requested_port)
        init_bridge(port=port)
        return jsonify(_ws_bridge_payload(running=True, clients=0, port=port) | {
            "success": True,
            "message": f"WebSocket bridge started on port {port}",
        })
    except Exception as e:
        return safe_error(e, "ws_start")


@system_bp.route("/ws/stop", methods=["POST"])
@require_csrf
def ws_stop():
    """Stop the WebSocket bridge."""
    try:
        from opencut.core.ws_bridge import stop_bridge
        stop_bridge()
        return jsonify({"success": True, "message": "WebSocket bridge stopped"})
    except Exception as e:
        return safe_error(e, "ws_stop")


# ---------------------------------------------------------------------------
# Engine Registry (Multi-Engine Backend)
# ---------------------------------------------------------------------------
@system_bp.route("/engines", methods=["GET"])
def engine_list():
    """List all registered AI engine backends and their status."""
    try:
        from opencut.core.engine_registry import get_registry
        reg = get_registry()
        return jsonify({"engines": reg.get_status()})
    except Exception as e:
        return safe_error(e, "engine_list")


@system_bp.route("/engines/preference", methods=["POST"])
@require_csrf
def engine_set_preference():
    """Set the preferred engine for a feature domain."""
    data = get_json_dict()
    domain = data.get("domain", "").strip()
    engine = str(data.get("engine", "") or "").strip()

    if not domain:
        return jsonify({"error": "domain required"}), 400

    try:
        from opencut.core.engine_registry import get_registry
        reg = get_registry()

        if not engine or engine.lower() == "auto":
            reg.clear_preference(domain)
            return jsonify({
                "success": True,
                "domain": domain,
                "engine": "",
                "mode": "auto",
            })

        # Validate domain and engine exist
        info = reg.get_engine(domain, engine)
        if not info:
            available = [e.name for e in reg.get_engines(domain)]
            return jsonify({
                "error": f"Engine '{engine}' not found in domain '{domain}'",
                "available": available,
            }), 400

        reg.set_preference(domain, engine)
        return jsonify({
            "success": True,
            "domain": domain,
            "engine": engine,
            "available": info.is_available,
        })
    except Exception as e:
        return safe_error(e, "engine_set_preference")


@system_bp.route("/engines/preferences", methods=["GET"])
def engine_get_preferences():
    """Get all engine preferences."""
    try:
        from opencut.core.engine_registry import get_registry
        reg = get_registry()
        return jsonify({"preferences": reg.export_preferences()})
    except Exception as e:
        return safe_error(e, "engine_get_preferences")


@system_bp.route("/engines/resolve", methods=["POST"])
@require_csrf
def engine_resolve():
    """Resolve which engine to use for a domain, considering availability and preferences."""
    data = get_json_dict()
    domain = data.get("domain", "").strip()
    requested = data.get("engine", "")

    if not domain:
        return jsonify({"error": "domain required"}), 400

    try:
        from opencut.core.engine_registry import get_registry
        reg = get_registry()
        engine = reg.resolve_engine(domain, requested if requested else None)

        if engine:
            return jsonify({
                "domain": domain,
                "engine": engine.name,
                "display_name": engine.display_name,
                "available": engine.is_available,
                "vram_mb": engine.vram_mb,
                "speed": engine.speed_rating,
                "quality": engine.quality_rating,
            })
        else:
            return jsonify({"error": f"No available engine for domain: {domain}"}), 404
    except Exception as e:
        return safe_error(e, "engine_resolve")


@system_bp.route("/auth/info", methods=["GET"])
def auth_info():
    """Return non-sensitive metadata about the local auth token (F112).

    The actual token is **never** included in this response. Operators use
    ``opencut-server --print-auth`` to retrieve it from their OS credential
    vault. Secret-file deployments never expose their configured path.
    """
    try:
        from opencut import auth as _auth

        remote_required = _auth.is_remote_bind_enabled()
        token = _auth.current_token()
        return jsonify(
            {
                "remote_bind_enabled": remote_required,
                "auth_required_for_remote": remote_required,
                "token_issued": token is not None,
                "token_issued_at": token.issued_at if token else None,
                "token_label": token.label if token else None,
                "token_file": None if _auth.using_token_file() else str(_auth.AUTH_FILE),
                "metadata_file": None if _auth.using_token_file() else str(_auth.AUTH_FILE),
                "credential_storage": _auth.credential_storage(),
                "header": _auth.AUTH_HEADER,
            }
        )
    except Exception as exc:
        return safe_error(exc, "auth_info")


@system_bp.route("/auth/rotate", methods=["POST"])
@require_csrf
def auth_rotate():
    """Issue a fresh local auth token (F112)."""
    try:
        from opencut import auth as _auth

        token = _auth.rotate_token()
        return jsonify(
            {
                "ok": True,
                "issued_at": token.issued_at,
                "label": token.label,
                "token_file": None if _auth.using_token_file() else str(_auth.AUTH_FILE),
                "credential_storage": _auth.credential_storage(),
                "note": (
                    "Token rotated in the configured secret file; its value is not returned."
                    if _auth.using_token_file()
                    else "Token rotated in the OS vault; reveal it with opencut-server --print-auth."
                ),
            }
        )
    except Exception as exc:
        return safe_error(exc, "auth_rotate")
