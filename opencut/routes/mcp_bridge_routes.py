"""
F146 — UXP-native MCP bridge.

Every competing Premiere-Pro MCP server today is CEP-bound (HTTP +
ExtendScript) and will break with Adobe's ~Sept-2026 CEP EOL. UXP
panels can't easily talk JSON-RPC to a sidecar process, but they can
hit the existing Flask app on :5679 over HTTPS/HTTP just like any
other route. This module bridges the MCP tool surface onto that same
HTTP server so UXP keeps the 39 curated tools (and the 1,325 opt-in
extended tools) usable post-EOL — no transport surgery required.

Three routes:

  GET  /mcp/tools     — list the available tools (the same payload
                        the sidecar exposes over JSON-RPC).
  POST /mcp/call      — invoke a tool: {tool, arguments} → {result}.
                        Wraps ``opencut.mcp_server.handle_tool_call``;
                        rate-limited per-tool via the existing
                        ``rate_limit`` machinery so the bridge can't
                        be used to bypass per-key throttles.
  GET  /mcp/info      — capability report (count, extended-enabled,
                        version, base-url).

Design notes:
  * The bridge stays in-process — no socket round-trip — by calling
    ``handle_tool_call`` directly. ``mcp_server._api`` will still
    re-hit ``:5679`` for the underlying REST calls, but the bridge
    itself adds no extra hop.
  * CSRF is required on ``POST /mcp/call`` (mutations); ``GET /mcp/*``
    is read-only.
  * Tool-name allowlist is enforced server-side via
    ``mcp_server.get_mcp_tools`` so a malicious UXP panel can't
    invoke arbitrary string names.
"""
from __future__ import annotations

import logging
import time

from flask import Blueprint, jsonify, request

from opencut.errors import safe_error
from opencut.security import rate_limit, rate_limit_release, require_csrf, safe_bool

logger = logging.getLogger("opencut")
mcp_bridge_bp = Blueprint("mcp_bridge", __name__)


def _tool_index() -> dict:
    """Return ``{tool_name: tool_def}`` for fast allowlist lookups."""
    from opencut import mcp_server
    out: dict = {}
    for tool in mcp_server.get_mcp_tools(include_extended=True):
        if isinstance(tool, dict) and tool.get("name"):
            out[str(tool["name"])] = tool
    return out


@mcp_bridge_bp.route("/mcp/tools", methods=["GET"])
def route_mcp_tools():
    """Return the tool catalogue.

    Query params:
      include_extended  bool  default true — include the 1,325 opt-in
                              auto-generated route tools.
    """
    try:
        from opencut import mcp_server
        include_extended = safe_bool(request.args.get("include_extended", "true"), True)
        tools = mcp_server.get_mcp_tools(include_extended=include_extended)
        return jsonify({
            "tools": tools,
            "count": len(tools),
            "include_extended": include_extended,
        })
    except Exception as exc:  # pragma: no cover
        return safe_error(exc, "mcp_bridge_tools")


@mcp_bridge_bp.route("/mcp/call", methods=["POST"])
@require_csrf
def route_mcp_call():
    """Invoke an MCP tool.

    Body params:
      tool        str   required, must be in the bridge allowlist
      arguments   dict  required (use ``{}`` for no-arg tools)
    """
    acquired_key: str | None = None
    try:
        from opencut import mcp_server

        data = request.get_json(silent=True) or {}
        tool = str(data.get("tool") or "").strip()
        if not tool:
            raise ValueError("'tool' is required")
        arguments = data.get("arguments")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise ValueError("'arguments' must be an object")

        # Allowlist guard — refuse unknown tool names BEFORE invoking.
        # Keeps the bridge from being used to probe arbitrary strings.
        idx = _tool_index()
        if tool not in idx:
            return jsonify({"error": f"unknown tool: {tool}"}), 400

        # Per-tool rate limit to keep one UXP panel from starving others.
        # Uses a deterministic key per tool name so concurrent identical
        # calls queue (rather than fan out and overload the backend).
        rl_key = f"mcp_bridge::{tool}"
        if not rate_limit(rl_key):
            return jsonify({
                "error": "rate limit exceeded for tool",
                "tool": tool,
                "retry_after_seconds": 1,
            }), 429
        acquired_key = rl_key

        start = time.perf_counter()
        result = mcp_server.handle_tool_call(tool, arguments)
        duration_ms = int((time.perf_counter() - start) * 1000)

        return jsonify({
            "tool": tool,
            "result": result,
            "duration_ms": duration_ms,
        })
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover
        return safe_error(exc, "mcp_bridge_call")
    finally:
        if acquired_key:
            try:
                rate_limit_release(acquired_key)
            except Exception:  # pragma: no cover
                pass


@mcp_bridge_bp.route("/mcp/info", methods=["GET"])
def route_mcp_info():
    """Capability report — how many tools, extended-mode flag, version."""
    try:
        from opencut import __version__, mcp_server
        from opencut.mcp_extended_tools import extended_tools_enabled
        curated = mcp_server.get_mcp_tools(include_extended=False)
        extended = mcp_server.get_mcp_tools(include_extended=True)
        return jsonify({
            "version": __version__,
            "curated_count": len(curated),
            "extended_count": len(extended) - len(curated),
            "extended_enabled_by_default": extended_tools_enabled(),
            "transport": "uxp-bridge",  # vs "json-rpc-stdio" or "json-rpc-http"
            "endpoints": ["/mcp/tools", "/mcp/call", "/mcp/info"],
        })
    except Exception as exc:  # pragma: no cover
        return safe_error(exc, "mcp_bridge_info")
