"""
OpenCut Issue Report Bundle (Wave H1.5, v1.25.0)

Bundles recent logs + crash excerpt into a pre-filled GitHub issue URL.
The panel shows the URL; the user reviews and submits manually — we
never post on their behalf.

Security: home paths and common credential formats are redacted before
the report is truncated or URL-encoded. This is defence in depth against
private data leaking into a public bug tracker. Callers MUST still prompt
the user to review the generated report before opening the URL.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import urllib.parse
from typing import Any, Dict, List

from opencut.server_log import server_log_tail

logger = logging.getLogger("opencut")

GITHUB_REPO = "SysAdminDoc/OpenCut"
ISSUE_URL = f"https://github.com/{GITHUB_REPO}/issues/new"


def check_issue_report_available() -> bool:
    """Always True — stdlib only."""
    return True


_HOME = os.path.expanduser("~")
# Any run of separators between segments, so a path inside JSON (where
# every backslash is doubled) is caught too.
_HOME_PATTERNS = [
    re.compile(r"[\\/]+".join(re.escape(part) for part in re.split(r"[\\/]+", _HOME)), re.IGNORECASE),
    re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\r\n\"']+", re.IGNORECASE),
]

REDACTION_MARKER = "[REDACTED]"

_SECRET_NAME = (
    r"(?:api[_-]?key|access[_-]?token|auth(?:entication)?[_-]?token|"
    r"bearer[_-]?token|client[_-]?secret|credential|passphrase|password|passwd|"
    r"private[_-]?key|refresh[_-]?token|secret(?:[_-]?key)?|"
    r"session[_-]?token|signature)"
)
_KNOWN_SECRET_ENV = (
    r"(?:ANTHROPIC_API_KEY|AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY|"
    r"AWS_SESSION_TOKEN|AZURE_OPENAI_API_KEY|COHERE_API_KEY|DEEPGRAM_API_KEY|"
    r"ELEVENLABS_API_KEY|GEMINI_API_KEY|GH_TOKEN|GITHUB_TOKEN|GOOGLE_API_KEY|"
    r"HF_TOKEN|HUGGING_FACE_HUB_TOKEN|MISTRAL_API_KEY|OPENAI_API_KEY|"
    r"OPENCUT_API_TOKEN|OPENCUT_REMOTE_AUTH_TOKEN|REPLICATE_API_TOKEN|"
    r"SLACK_TOKEN|STRIPE_SECRET_KEY)"
)
_QUERY_SECRET_NAME = (
    rf"(?:{_SECRET_NAME}|key|sig|token|x-amz-credential|x-amz-security-token|"
    r"x-amz-signature|x-goog-signature)"
)
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN(?: [A-Z0-9]+)? PRIVATE KEY-----.*?"
    r"-----END(?: [A-Z0-9]+)? PRIVATE KEY-----",
    re.IGNORECASE | re.DOTALL,
)
_SECRET_HEADER_RE = re.compile(
    r"(?P<prefix>(?<![\w-])(?:authorization|proxy-authorization|x-api-key|"
    r"x-auth-token|x-opencut-token|x-amz-security-token|cookie|set-cookie)"
    r"[\"']?[ \t]*:[ \t]*(?P<vq>[\"'])?)"
    r"(?P<value>(?(vq)(?:(?!(?P=vq))[^\r\n])*|[^\r\n]*))",
    re.IGNORECASE,
)
_URL_USERINFO_RE = re.compile(
    # The user may be empty and the password may hold "@" or "/" (an
    # escaped one is decoded first), so run to the last "@" before a host.
    r"(?P<prefix>\b[a-z][a-z0-9+.-]*://)[^\s/@:]*:\S*?@"
    r"(?=[^\s/@?#]*(?:[/?#\s\"'<>]|$))",
    re.IGNORECASE | re.MULTILINE,
)
# Besides query strings, a bare ``token=`` after whitespace, a quote or a
# bracket counts, and so does ``:key=`` right behind a hex key id, which is
# Shaka Packager's ``--keys label=X:key_id=...:key=...`` list.
_QUERY_SECRET_RE = re.compile(
    r"(?P<prefix>(?:[?&;]|&amp;|(?<![^\s\"'(\[,])|key_id=[0-9A-Fa-f]+:)"
    rf"{_QUERY_SECRET_NAME}=)"
    r"(?P<value>[^&#\s\"'<>\r\n]*)",
    re.IGNORECASE,
)
_ASSIGNED_SECRET_RE = re.compile(
    rf"(?P<prefix>(?P<kq>[\"']?)\b(?:[A-Za-z0-9]+_)*(?:{_KNOWN_SECRET_ENV}|{_SECRET_NAME})\b"
    r"(?P=kq)[ \t]*(?:(?P<colon>:)|=)[ \t]*)"
    rf"(?P<value>(?P<vq>[\"'])(?!{re.escape(REDACTION_MARKER)})"
    r"(?:(?!(?P=vq))[^\r\n])*(?P=vq)"
    rf"|(?![\"']?{re.escape(REDACTION_MARKER)})"
    r"(?(colon)[^,;\r\n}\]\"']+|[^\s,;\r\n}\]]+))",
    re.IGNORECASE,
)
_CLI_SECRET_RE = re.compile(
    rf"(?P<prefix>(?<![\w-])--?(?:[A-Za-z0-9]+[_-])*{_SECRET_NAME}"
    r"(?:=|[ \t]+|[\"'],[ \t]*[\"']))(?P<value>[^\s\"']+)",
    re.IGNORECASE,
)
_AUTH_SCHEME_RE = re.compile(
    r"(?P<prefix>\b(?:basic|bearer)(?:[ \t]+|\+))(?P<value>[A-Za-z0-9._~+/=-]{8,})",
    re.IGNORECASE,
)
# Keys whose whole value is a secret when a structure is walked.
_SECRET_KEY_RE = re.compile(
    rf"(?:^|[_\-\s.])(?:{_KNOWN_SECRET_ENV}|{_SECRET_NAME}|authorization|bearer|token)(?:[_\-\s.]|$)",
    re.IGNORECASE,
)
_PERCENT_ESCAPE_RE = re.compile(r"%[0-9A-Fa-f]{2}")
_PROVIDER_SECRET_PATTERNS = (
    re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b", re.IGNORECASE),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b", re.IGNORECASE),
    re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b", re.IGNORECASE),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b", re.IGNORECASE),
    re.compile(r"\bnpm_[A-Za-z0-9]{20,}\b", re.IGNORECASE),
    re.compile(r"\bpypi-[A-Za-z0-9_-]{32,}\b", re.IGNORECASE),
    re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b", re.IGNORECASE),
    re.compile(r"\b(?:sk|rk)_(?:live|test)_[0-9A-Za-z]{16,}\b", re.IGNORECASE),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
)


def _scrub_paths(text: str) -> str:
    """Replace absolute HOME paths with ``~``.  Does not touch relative paths."""
    if not text:
        return text
    for pat in _HOME_PATTERNS:
        text = pat.sub("~", text)
    return text


def _redact_match_value(match: re.Match[str]) -> str:
    """Keep a credential label or URL prefix while removing its value."""
    return f"{match.group('prefix')}{REDACTION_MARKER}"


def _redact_assigned_value(match: re.Match[str]) -> str:
    """Redact an assigned secret while preserving quotes around the value."""
    quote = match.group("vq") or ""
    return f"{match.group('prefix')}{quote}{REDACTION_MARKER}{quote}"


def _redact_sensitive_text(text: str) -> str:
    """Redact common secrets and home paths from diagnostic text.

    The substitutions intentionally retain header, variable, query-parameter,
    and command-line option names so the resulting issue preview remains useful.
    This is a best-effort guardrail, not a replacement for user review.

    server.log writes one JSON object per line, where quotes and backslashes
    are escaped and a header is never at the start of a line. Each such line
    is decoded, its strings are redacted as plain text, and, when anything
    changed, it is encoded again in the line's own JSON style.
    """
    if not text:
        return text

    redacted = _PRIVATE_KEY_RE.sub(REDACTION_MARKER, str(text))
    return "\n".join(_redact_line(line) for line in redacted.split("\n"))


def _redact_line(line: str) -> str:
    body = line.rstrip("\r")
    stripped = body.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            record = json.loads(stripped)
        except ValueError:
            record = None
        if isinstance(record, dict):
            redacted = _redact_structure(record)
            if redacted == record:
                return line
            start = body.index(stripped)
            return (
                body[:start]
                + _encode_like(redacted, record, stripped)
                + body[start + len(stripped):]
                + line[len(body):]
            )
    return _redact_plain_text(line)


def _encode_like(value: Any, original: Any, encoded: str) -> str:
    """Encode ``value`` with the separators and escaping that turn
    ``original`` back into ``encoded``, so compact JSON stays compact."""
    for separators in ((", ", ": "), (",", ":")):
        for ensure_ascii in (False, True):
            if json.dumps(original, ensure_ascii=ensure_ascii, separators=separators) == encoded:
                return json.dumps(value, ensure_ascii=ensure_ascii, separators=separators)
    return json.dumps(value, ensure_ascii=False)


def _redact_structure(value: Any) -> Any:
    """Redact a decoded JSON value: secret-named keys lose their value,
    and every string is redacted as plain text."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            name = _redact_plain_text(key) if isinstance(key, str) else key
            if isinstance(item, str) and item and _SECRET_KEY_RE.search(str(key)):
                out[name] = REDACTION_MARKER
            else:
                out[name] = _redact_structure(item)
        return out
    if isinstance(value, list):
        return [_redact_structure(item) for item in value]
    if isinstance(value, str):
        return _redact_plain_text(value)
    return value


def _redact_plain_text(text: str) -> str:
    if not text:
        return text
    redacted = text
    if _PERCENT_ESCAPE_RE.search(redacted):
        # A URL nested in a query string hides its userinfo and an
        # ``auth=Bearer%20...`` value behind percent escapes.
        redacted = urllib.parse.unquote(redacted)
    redacted = _scrub_paths(redacted)
    redacted = _PRIVATE_KEY_RE.sub(REDACTION_MARKER, redacted)
    redacted = _SECRET_HEADER_RE.sub(_redact_match_value, redacted)
    redacted = _URL_USERINFO_RE.sub(
        lambda match: f"{match.group('prefix')}{REDACTION_MARKER}@",
        redacted,
    )
    redacted = _QUERY_SECRET_RE.sub(_redact_match_value, redacted)
    redacted = _ASSIGNED_SECRET_RE.sub(_redact_assigned_value, redacted)
    redacted = _CLI_SECRET_RE.sub(_redact_match_value, redacted)
    redacted = _AUTH_SCHEME_RE.sub(_redact_match_value, redacted)
    for pattern in _PROVIDER_SECRET_PATTERNS:
        redacted = pattern.sub(REDACTION_MARKER, redacted)
    return redacted


def _tail_file(path: str, max_bytes: int = 20_000) -> str:
    """Return the last ``max_bytes`` of ``path`` with best-effort decoding."""
    if not os.path.isfile(path):
        return ""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - max_bytes))
            data = fh.read()
    except OSError as exc:
        return f"[could not read {os.path.basename(path)}: {exc}]"
    return data.decode("utf-8", errors="replace")


def bundle(
    title: str = "",
    description: str = "",
    log_tail_lines: int = 200,
    include_crash: bool = True,
    include_logs: bool = True,
    host_diagnostics: Any = None,
) -> Dict[str, Any]:
    """Assemble a report bundle. Returns ``{title, body, url, size_bytes}``.

    Title and body are scrubbed of HOME paths and common secrets before the
    body is capped at ~60 KB and both fields are URL-encoded.
    """
    log_tail_lines = max(10, min(int(log_tail_lines or 200), 2000))
    data_dir = os.path.join(_HOME, ".opencut")
    crash_path = os.path.join(data_dir, "crash.log")

    try:
        from opencut import __version__ as ocv
    except Exception:  # noqa: BLE001
        ocv = "?"

    parts: List[str] = []
    if description:
        parts.append(
            "## Description\n\n" + _redact_sensitive_text(str(description)) + "\n"
        )
    parts.append(
        "## Environment\n\n"
        f"- OpenCut: {ocv}\n"
        f"- Platform: {platform.system()} {platform.release()}\n"
        f"- Python: {platform.python_version()}\n"
    )

    if host_diagnostics:
        try:
            diagnostic_text = json.dumps(
                host_diagnostics,
                indent=2,
                sort_keys=True,
                default=str,
            )
        except (TypeError, ValueError):
            diagnostic_text = str(host_diagnostics)
        # Host results can carry project-tree samples. Keep the public issue
        # body bounded while preserving the fields F319 needs: host version,
        # attempted/reported/verified counts, status, and read-back method.
        diagnostic_text = _redact_sensitive_text(diagnostic_text[:12_000])
        parts.append(
            "## Premiere host-write verification\n\n```json\n"
            + diagnostic_text
            + "\n```\n"
        )

    if include_crash:
        tail = _tail_file(crash_path, max_bytes=20_000)
        if tail:
            parts.append("## Crash log (tail)\n\n```\n"
                         + _redact_sensitive_text(tail) + "\n```\n")

    if include_logs:
        tail = "\n".join(server_log_tail(max_lines=log_tail_lines, home=_HOME))
        if tail:
            parts.append("## Recent logs\n\n```\n"
                         + _redact_sensitive_text(tail) + "\n```\n")

    body = "\n".join(parts)
    if len(body) > 60_000:
        body = body[:60_000] + "\n\n_[truncated]_\n"

    effective_title = _redact_sensitive_text(
        (title or f"OpenCut v{ocv} issue report").strip()
    )
    q = urllib.parse.urlencode({
        "title": effective_title,
        "body": body,
        "labels": "bug,from-panel",
    })
    url = f"{ISSUE_URL}?{q}"
    return {
        "title": effective_title,
        "body": body,
        "url": url,
        "size_bytes": len(body.encode("utf-8")),
    }


__all__ = [
    "check_issue_report_available",
    "bundle",
]
