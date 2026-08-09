"""Small, dependency-free MCP Apps review/progress surface.

The MCP server is intentionally stdlib-only, so the app is a versioned inline
HTML resource instead of a runtime npm dependency.  It receives sanitized
tool data from the host and can request only the fixed review-action tool.
"""

from __future__ import annotations

import copy
import re
from typing import Any

APP_VERSION = "1"
RESOURCE_URI = f"ui://opencut/review-progress/v{APP_VERSION}/index.html"
RESOURCE_MIME_TYPE = "text/html;profile=mcp-app"
EXTENSION_ID = "io.modelcontextprotocol/ui"

_PATH_KEYS = frozenset(
    {
        "path",
        "filepath",
        "file_path",
        "output",
        "output_path",
        "media_path",
        "video_path",
        "audio_path",
        "image_path",
        "captions_path",
        "thumbnail_path",
        "root_path",
        "source_path",
        "asset_path",
        "render_path",
        "bundle_path",
        "zip_path",
        "reference_path",
        "sidecar_path",
        "voice_ref",
        "extra_files",
        "files",
        "file",
        "source",
        "reference",
        "folder",
        "directory",
    }
)
_ABSOLUTE_PATH_RE = re.compile(r"^(?:[A-Za-z]:[\\/]|/|\\\\|//)")


def resource_ui_metadata() -> dict:
    """Return the resource's restrictive, host-enforced UI metadata."""
    return {
        "ui": {
            "csp": {
                "baseUriDomains": [],
                "connectDomains": [],
                "frameDomains": [],
                "resourceDomains": [],
            },
            "permissions": {},
            "prefersBorder": True,
        }
    }


def resource_listing() -> dict:
    """Return the static ``resources/list`` record for capable clients."""
    return {
        "uri": RESOURCE_URI,
        "name": "OpenCut review and progress",
        "description": (
            "Sandboxed, local-only review/progress surface for OpenCut jobs "
            "and capability-scoped search results."
        ),
        "mimeType": RESOURCE_MIME_TYPE,
        "_meta": resource_ui_metadata(),
    }


def resource_contents() -> dict:
    """Return the static ``resources/read`` payload."""
    return {
        "contents": [
            {
                "uri": RESOURCE_URI,
                "mimeType": RESOURCE_MIME_TYPE,
                "text": APP_HTML,
                "_meta": resource_ui_metadata(),
            }
        ]
    }


def _redact_path_value(value: Any) -> Any:
    if isinstance(value, list):
        return {"redacted": True, "count": len(value)}
    if isinstance(value, str):
        # The app never needs a local path.  Preserve a safe display label only
        # for relative names, and redact absolute values completely.
        if _ABSOLUTE_PATH_RE.match(value.strip()):
            return "[redacted local path]"
        return "[redacted local path]"
    return "[redacted local path]"


def redact_payload(value: Any) -> Any:
    """Copy tool data while removing local paths and executable capabilities."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            key_text = str(key)
            lowered = key_text.casefold()
            if (
                lowered in _PATH_KEYS
                or lowered.endswith("_path")
                or lowered.endswith("filepath")
            ):
                result[key_text] = _redact_path_value(item)
            else:
                result[key_text] = redact_payload(item)
        return result
    if isinstance(value, list):
        return [redact_payload(item) for item in value]
    if isinstance(value, tuple):
        return [redact_payload(item) for item in value]
    return value


def build_surface_payload(tool_name: str, payload: Any) -> dict:
    """Wrap a sanitized tool result for ``structuredContent``."""
    return {
        "surface": "review-progress",
        "version": APP_VERSION,
        "tool": str(tool_name),
        "capabilities": {
            "local_paths": False,
            "network": False,
            "actions": ["refresh", "cancel", "approve", "reject", "request_changes"],
        },
        "data": redact_payload(copy.deepcopy(payload)),
    }


APP_HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src data:;">
<title>OpenCut review and progress</title>
<style>
:root { color-scheme: light dark; font: 14px system-ui, sans-serif; }
body { margin: 0; padding: 14px; background: Canvas; color: CanvasText; }
header { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
h1 { font-size: 16px; margin: 0; }
#state { color: GrayText; font-size: 12px; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; max-height: 260px; overflow: auto; border: 1px solid GrayText; border-radius: 6px; padding: 10px; }
button { margin: 4px 4px 0 0; padding: 6px 10px; border: 1px solid GrayText; border-radius: 5px; background: ButtonFace; color: ButtonText; }
</style>
</head>
<body>
<header><h1>OpenCut review and progress</h1><span id="state">Waiting for tool data</span></header>
<pre id="payload">No data received.</pre>
<div id="actions" aria-label="Review actions"></div>
<script>
(function () {
  "use strict";
  var actionTool = "opencut_review_action";
  var allowed = ["refresh", "cancel", "approve", "reject", "request_changes"];
  var sequence = 0;
  var state = {};
  var output = document.getElementById("payload");
  var status = document.getElementById("state");
  var actions = document.getElementById("actions");

  function safeText(value) {
    try { return JSON.stringify(value, null, 2); }
    catch (_) { return "Unavailable"; }
  }
  function render(value) {
    state = (value && value.data) ? value.data : (value || {});
    output.textContent = safeText(value);
    status.textContent = "Data received; paths and network access are disabled";
  }
  function callAction(action) {
    if (allowed.indexOf(action) === -1) return;
    var args = { action: action };
    if (typeof state.job_id === "string") args.job_id = state.job_id;
    if (typeof state.workflow_id === "string") args.workflow_id = state.workflow_id;
    window.parent.postMessage({
      jsonrpc: "2.0",
      id: "opencut-app-" + (++sequence),
      method: "tools/call",
      params: { name: actionTool, arguments: args }
    }, "*");
    status.textContent = "Requested " + action;
  }
  allowed.forEach(function (action) {
    var button = document.createElement("button");
    button.type = "button";
    button.textContent = action.replace("_", " ");
    button.addEventListener("click", function () { callAction(action); });
    actions.appendChild(button);
  });
  window.addEventListener("message", function (event) {
    if (!event || !event.data) return;
    var message = event.data;
    var value = message.result || (message.params && (message.params.result || message.params));
    if (value) render(value);
  });
  window.parent.postMessage({
    jsonrpc: "2.0",
    id: "opencut-app-init",
    method: "ui/initialize",
    params: { protocolVersion: "2026-01-26", capabilities: {} }
  }, "*");
}());
</script>
</body>
</html>'''
