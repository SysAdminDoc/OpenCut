"""Deny-by-default outbound network policy for local-only mode.

The guard uses Python's process-wide audit hook so direct ``socket`` calls,
HTTP libraries, third-party SDKs, and network-capable subprocesses cannot
bypass a forgotten feature-level check. Only explicit loopback destinations
(``localhost``, ``127.0.0.0/8``, and ``::1``) are permitted.
"""

from __future__ import annotations

import ast
import ipaddress
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ERROR_CODE = "LOCAL_ONLY_NETWORK_BLOCKED"
DEFAULT_LOCAL_ALTERNATIVE = (
    "Use a local file or a service bound to localhost (127.0.0.1 or ::1), "
    "or disable local-only mode explicitly."
)

_AUDIT_EVENTS_WITH_HOST = {
    "socket.getaddrinfo": 0,
    "socket.gethostbyaddr": 0,
    "socket.gethostbyname": 0,
    "socket.gethostbyname_ex": 0,
}
_URL_RE = re.compile(
    r"(?i)\b(?:https?|ftp|srt|rtmp|rtsp|tcp|udp|ssh)://[^\s\"']+"
)
_ALWAYS_NETWORK_EXECUTABLES = {
    "croc",
    "ftp",
    "nc",
    "ncat",
    "rclone",
    "scp",
    "sftp",
    "ssh",
    "telnet",
}
_URL_NETWORK_EXECUTABLES = {"curl", "wget", "youtube-dl", "yt-dlp"}
_NETWORK_GIT_VERBS = {
    "clone",
    "fetch",
    "ls-remote",
    "pull",
    "push",
    "remote-fd",
    "submodule",
}
_NETWORK_PIP_VERBS = {"download", "index", "install", "search", "wheel"}
_DIRECT_NETWORK_IMPORTS = {
    "aiohttp",
    "http.client",
    "httpx",
    "requests",
    "socket",
    "urllib.request",
    "websocket",
    "websockets",
}

_hook_installed = False


class LocalOnlyNetworkError(RuntimeError):
    """Stable policy error raised before non-loopback outbound I/O."""

    code = ERROR_CODE
    status = 403
    suggestion = DEFAULT_LOCAL_ALTERNATIVE

    def __init__(
        self,
        target: str,
        *,
        feature: str = "Outbound network access",
        local_alternative: str = "",
    ) -> None:
        self.target = str(target or "unknown destination")
        self.feature = str(feature or "Outbound network access")
        self.suggestion = str(local_alternative or DEFAULT_LOCAL_ALTERNATIVE)
        super().__init__(
            f"{ERROR_CODE}: Local-only mode blocked {self.feature.lower()} "
            f"to {self.target}."
        )


def _local_only_active() -> bool:
    from opencut.config import is_local_only

    return is_local_only()


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value or "")


def host_from_address(address: Any) -> str:
    """Extract the target host from a socket address or audit argument."""
    if isinstance(address, (tuple, list)):
        return _text(address[0]) if address else ""
    if isinstance(address, (str, bytes)):
        value = _text(address)
        # A bare filesystem-like string is an AF_UNIX socket, not a host.
        if (
            value.startswith("\x00")
            or os.path.isabs(value)
            or value.startswith(("./", "../", "\\\\"))
        ):
            return ""
        return value
    return ""


def is_loopback_host(host: Any) -> bool:
    """Return true only for explicit loopback hostnames and addresses."""
    value = _text(host).strip().lower().rstrip(".")
    if not value:
        return True
    if value.startswith("[") and "]" in value:
        value = value[1:value.index("]")]
    if value == "localhost" or value.endswith(".localhost"):
        return True
    if "%" in value:
        value = value.split("%", 1)[0]
    try:
        address = ipaddress.ip_address(value)
        if address.is_loopback:
            return True
        return bool(
            isinstance(address, ipaddress.IPv6Address)
            and address.ipv4_mapped
            and address.ipv4_mapped.is_loopback
        )
    except ValueError:
        return False


def require_outbound_host_allowed(
    host: Any,
    *,
    feature: str = "Outbound network access",
    local_alternative: str = "",
) -> None:
    """Raise before resolving or connecting to a non-loopback destination."""
    target = host_from_address(host)
    if not target or not _local_only_active() or is_loopback_host(target):
        return
    raise LocalOnlyNetworkError(
        target,
        feature=feature,
        local_alternative=local_alternative,
    )


def _command_tokens(executable: Any, arguments: Any) -> list[str]:
    if isinstance(arguments, (list, tuple)):
        tokens = [_text(item) for item in arguments]
    elif isinstance(arguments, (str, bytes)):
        try:
            tokens = [
                token.strip('"\'')
                for token in shlex.split(_text(arguments), posix=os.name != "nt")
            ]
        except ValueError:
            tokens = [_text(arguments)]
    else:
        tokens = [_text(arguments)] if arguments else []
    if not tokens and executable:
        tokens = [_text(executable)]
    return tokens


def _url_host(value: str) -> str:
    try:
        return str(urlsplit(value).hostname or "")
    except ValueError:
        return ""


def _require_command_allowed(executable: Any, arguments: Any) -> None:
    if not _local_only_active():
        return
    tokens = _command_tokens(executable, arguments)
    joined = " ".join(tokens)
    urls = _URL_RE.findall(joined)
    for url in urls:
        host = _url_host(url.rstrip(".,);]"))
        if host and not is_loopback_host(host):
            raise LocalOnlyNetworkError(host, feature="External process network access")

    raw_executable = _text(executable) or (tokens[0] if tokens else "")
    command = os.path.splitext(os.path.basename(raw_executable).lower())[0]
    lowered = [token.lower() for token in tokens]
    if command in _ALWAYS_NETWORK_EXECUTABLES:
        raise LocalOnlyNetworkError(command, feature="External process network access")
    if command in _URL_NETWORK_EXECUTABLES and not urls:
        raise LocalOnlyNetworkError(command, feature="External process network access")
    if command in {"pip", "pip3"} and any(
        token in _NETWORK_PIP_VERBS for token in lowered[1:]
    ):
        raise LocalOnlyNetworkError(command, feature="Package download")
    if command == "npx" and not any(
        token in {"--no-install", "--offline"} for token in lowered[1:]
    ):
        raise LocalOnlyNetworkError(command, feature="Package download")
    if command.startswith("python") and "-m" in lowered:
        module_index = lowered.index("-m") + 1
        if (
            module_index < len(lowered)
            and lowered[module_index] == "pip"
            and any(token in _NETWORK_PIP_VERBS for token in lowered[module_index + 1:])
        ):
            raise LocalOnlyNetworkError("pip", feature="Package download")
    if command == "git" and any(token in _NETWORK_GIT_VERBS for token in lowered[1:]):
        raise LocalOnlyNetworkError("git remote", feature="Git network access")


def _audit_hook(event: str, args: tuple[Any, ...]) -> None:
    if event in _AUDIT_EVENTS_WITH_HOST and args:
        require_outbound_host_allowed(args[_AUDIT_EVENTS_WITH_HOST[event]])
        return
    if event in {"socket.connect", "socket.sendto"} and len(args) > 1:
        require_outbound_host_allowed(args[1])
        return
    if event == "subprocess.Popen" and args:
        _require_command_allowed(args[0], args[1] if len(args) > 1 else ())
        return
    if event in {"os.system", "os.startfile", "os.startfile/2"} and args:
        value = _text(args[0])
        urls = _URL_RE.findall(value)
        if urls:
            for url in urls:
                host = _url_host(url.rstrip(".,);]"))
                require_outbound_host_allowed(
                    host, feature="External application network access"
                )
        elif event == "os.system":
            _require_command_allowed("", value)


def install_egress_guard() -> None:
    """Install the process-wide audit hook exactly once."""
    global _hook_installed
    if _hook_installed:
        return
    sys.addaudithook(_audit_hook)
    _hook_installed = True


def discover_direct_egress_modules(root: Path) -> set[str]:
    """Generate the modules that import low-level network clients.

    The checked inventory test uses this AST scan so a new direct HTTP/socket
    callsite must be classified deliberately. Third-party SDK traffic is still
    covered by the runtime audit hook even when no low-level import is visible.
    """
    discovered: set[str] = set()
    for path in sorted(root.rglob("*.py")):
        if "_generated" in path.parts or "data" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeError):
            continue
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
        if any(
            imported == candidate or imported.startswith(candidate + ".")
            for imported in imports
            for candidate in _DIRECT_NETWORK_IMPORTS
        ):
            discovered.add(path.relative_to(root.parent).as_posix())
    return discovered


# Every direct low-level network client has an explicit role. The runtime guard
# applies to all roles; these classifications document why the import exists.
EGRESS_INVENTORY: dict[str, str] = {
    "opencut/checks.py": "mixed-client-health",
    "opencut/cli.py": "mixed-cli-client",
    "opencut/core/ai_storyboard.py": "external-integration",
    "opencut/core/auto_update.py": "external-integration",
    "opencut/core/broll_ai_gen.py": "external-integration",
    "opencut/core/changelog_feed.py": "external-integration",
    "opencut/core/cloud_render.py": "external-integration",
    "opencut/core/demo_bundle.py": "external-integration",
    "opencut/core/frameio_integration.py": "external-integration",
    "opencut/core/gist_sync.py": "external-integration",
    "opencut/core/google_fonts.py": "external-integration",
    "opencut/core/llm.py": "mixed-loopback-and-external-client",
    "opencut/core/model_manager.py": "external-integration",
    "opencut/core/notion_sync.py": "external-integration",
    "opencut/core/obs_bridge.py": "loopback-client",
    "opencut/core/plugin_marketplace.py": "external-integration",
    "opencut/core/remote_process.py": "external-or-lan-integration",
    "opencut/core/runpod_render.py": "external-integration",
    "opencut/core/sfx_library.py": "external-integration",
    "opencut/core/slack_notify.py": "external-integration",
    "opencut/core/social_post.py": "mixed-loopback-and-external-client",
    "opencut/core/stock_search.py": "external-integration",
    "opencut/core/style_transfer.py": "external-integration",
    "opencut/core/telemetry_aptabase.py": "external-integration",
    "opencut/core/telemetry_plausible.py": "external-integration",
    "opencut/core/tts_gptsovits.py": "loopback-client",
    "opencut/core/url_safety.py": "egress-guard",
    "opencut/core/video_llm.py": "external-integration",
    "opencut/core/voice_overdub.py": "external-integration",
    "opencut/core/webhook_integrations.py": "external-integration",
    "opencut/core/webhook_system.py": "external-integration",
    "opencut/core/webhooks.py": "external-integration",
    "opencut/core/ws_bridge.py": "loopback-server",
    "opencut/mcp_server.py": "loopback-server",
    "opencut/pid.py": "loopback-client-health",
    "opencut/routes/system.py": "mixed-route-client",
    "opencut/tools/adobe_premierepro_versions.py": "operator-tool",
    "opencut/tools/download_eval_dataset.py": "operator-tool",
}


def unclassified_egress_modules(root: Path) -> set[str]:
    return discover_direct_egress_modules(root) - set(EGRESS_INVENTORY)


def stale_egress_inventory(root: Path) -> set[str]:
    return set(EGRESS_INVENTORY) - discover_direct_egress_modules(root)


def inventory_by_role() -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for module, role in sorted(EGRESS_INVENTORY.items()):
        grouped.setdefault(role, []).append(module)
    return grouped
