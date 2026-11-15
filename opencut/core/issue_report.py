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
    r"session[_-]?token|signature|(?:en|de)cryption[_-]?key)"
)
_KNOWN_SECRET_ENV = (
    r"(?:ANTHROPIC_API_KEY|AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY|"
    r"AWS_SESSION_TOKEN|AZURE_OPENAI_API_KEY|COHERE_API_KEY|DEEPGRAM_API_KEY|"
    r"ELEVENLABS_API_KEY|GEMINI_API_KEY|GH_TOKEN|GITHUB_TOKEN|GOOGLE_API_KEY|"
    r"HF_TOKEN|HUGGING_FACE_HUB_TOKEN|MISTRAL_API_KEY|OPENAI_API_KEY|"
    r"OPENCUT_API_TOKEN|OPENCUT_REMOTE_AUTH_TOKEN|REPLICATE_API_TOKEN|"
    r"SLACK_TOKEN|STRIPE_SECRET_KEY)"
)
# Names that end in "token" but hold no credential: tokenizer specials
# (``bos_token``, ``pad_token``), stop words and pagination cursors.
_PLAIN_TOKEN_NAME = (
    r"(?:bos|eos|pad|unk|sep|cls|mask|stop|start|end|eot|special|"
    r"(?:next[_-]?)?page|continuation|cancel(?:lation)?)[_-]?token"
)
_PLAIN_TOKEN_KEY_RE = re.compile(_PLAIN_TOKEN_NAME, re.IGNORECASE)
# A query name may carry a prefix (``api_token``, ``oauth_token``), but a
# bare ``key`` or ``sig`` may not, so ``cache_key`` stays readable.
_QUERY_SECRET_NAME = (
    rf"(?:(?!{_PLAIN_TOKEN_NAME}(?![\w-]))(?:[A-Za-z0-9]+[_-])*(?:{_SECRET_NAME}|token)|"
    r"authorization|key|sig|x-amz-credential|"
    r"x-amz-security-token|x-amz-signature|x-api-key|x-auth-token|"
    r"x-goog-signature|x-opencut-token)"
)
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN(?: [A-Z0-9]+)? PRIVATE KEY-----.*?"
    r"-----END(?: [A-Z0-9]+)? PRIVATE KEY-----",
    re.IGNORECASE | re.DOTALL,
)
# ``name: value`` and ``NAME=value`` (a WSGI environ), plus the
# ``('Name', 'value')`` pair of a header list, with ASGI's bytes too. The
# pair form needs the opening bracket, so ``['Authorization',
# 'Content-Type']`` stays. An environ ``HTTP_NAME=`` value runs to the
# end of the line. A bare ``name=`` value stops where a query value
# would, but keeps its auth scheme (a Digest or OAuth parameter list
# runs to the end of the line), a ``name=value`` cookie list joined by
# ``;``, ``,`` or spaces, and a dict or call repr. A flag that is the
# whole value (``cookie=on``, ``cookie=off: disabled``) is left alone,
# but not when anything other than a space, quote or closing bracket
# follows its punctuation (``cookie=on.;sid=...``). A list item whose
# ``=`` is followed by a space and then a word with no ``=`` ends the list
# (``;password= ...``), so the assignment pattern still sees that name,
# but an empty item before another pair (``;theme= sid=...``) doesn't.
_SECRET_HEADER_RE = re.compile(
    r"(?P<prefix>(?:(?P<open>\([ \t]*b?[\"'])|(?<![\w-]))(?P<env>http_)?(?:authorization|"
    r"proxy[-_]authorization|x[-_]api[-_]key|x[-_]auth[-_]token|x[-_]opencut[-_]token|"
    r"x[-_]amz[-_]security[-_]token|cookie|set[-_]cookie)"
    r"(?:[\"']?[ \t]*(?::|(?P<eq>=))|(?(open)[\"'],|(?!)))[ \t]*+(?:b?(?P<vq>[\"']))?)"
    r"(?P<value>(?(vq)(?:\\.|(?!(?P=vq))[^\r\n\\])*"
    r"|(?(eq)(?(env)[^\r\n]*"
    rf"|(?!{re.escape(REDACTION_MARKER)})(?!(?:true|false|none|null|on|off|yes|no)"
    r"(?:(?:[.:!?]|/(?:true|false|none|null|on|off|yes|no))(?=[\s\"')\]}]|$)|(?![^\s&,;\"')\]}])))"
    r"(?:(?:digest|oauth|hawk|aws4-hmac-sha256|vapid|mutual|hoba|scram-sha-(?:1|256))[ \t]+[^\r\n]*"
    r"|(?:[{\[]|[A-Za-z_][\w.]*\()[^\r\n]*"
    r"|(?:(?:basic|bearer|token|negotiate|ntlm|apikey|api-key|dpop|kerberos|sharedkey(?:lite)?)[ \t]+)?"
    r"(?:[^\s&,;=]++=[^\s&,;]*+(?:[ \t]*[;,][ \t]*[^\s&,;=]++(?!=[ \t]++[^\s&,;=]++(?!=))(?:=[^\s&,;]*+)?"
    r"|[ \t]++[^\s&,;=]++=(?![ \t]++[^\s&,;=]++(?!=))[^\s&,;]*+)*+"
    r"|[^\s&,;]++(?:;[ \t]*[^\s&,;=]+=(?![ \t]++[^\s&,;=]++(?!=))[^\s&,;]*)*)))"
    r"|[^\r\n]*)))",
    re.IGNORECASE,
)
_URL_USERINFO_RE = re.compile(
    # The user may be empty and the password may hold "@" or "/", so the
    # match takes the whole run and _redact_url_userinfo cuts at its last
    # "@", or earlier only where a plain host and path follow. Taking the
    # run whole, with no "@" required, means a long line
    # is read once however many "://" it holds. Digits after the first
    # ":" are a port when a query or the end follows, or a path does after
    # a host (a dot or ``localhost``: "cdn.example.com:8443/logo@2x.png"),
    # but not when the name holds an encoded "@" ("first.last%40corp.com").
    # After a bare name, "/" and then an "@" in that segment read as a
    # password ("admin:2024/pw@cam"), so "nas:445/a@b" is redacted too.
    # The user part never holds "[", so an IPv6 host's colons don't
    # start one.
    r"(?P<prefix>(?<![a-z0-9+.-])[a-z][a-z0-9+.-]*://)"
    r"(?>(?P<host>(?![^\s/@:\[\]\"'<>]*%40)(?:localhost(?=:)|[^\s/@:\[\]\"'<>.]*\.))?)"
    r"[^\s/@:\[\]\"'<>]*+:"
    r"(?!\d+(?:[?#\s\"'<>]|$|(?(host)/|/(?![^\s/?#\"'<>@]+@))))"
    r"(?P<rest>[^\s\"'<>]*)",
    re.IGNORECASE | re.MULTILINE,
)
# Besides query strings, a bare ``token=`` after whitespace, a quote or a
# bracket counts, and so does ``:key=`` right behind a hex key id, which is
# Shaka Packager's ``--keys label=X:key_id=...:key=...`` list. An empty
# value is left for the assignment pattern, which reads ``key= value``.
_QUERY_SECRET_RE = re.compile(
    r"(?P<prefix>(?:[?&;]|&amp;|(?<![^\s\"'(\[,])|key_id=[0-9A-Fa-f]+:|"
    r"label=[^:=&\s\"']*:)"
    rf"{_QUERY_SECRET_NAME}=)"
    r"(?P<value>(?:[bru]{1,2})?(?P<vq>[\"'])(?:\\.|(?!(?P=vq))[^\r\n\\])*(?P=vq)"
    r"|[^&#\s\"'<>\r\n]+)",
    re.IGNORECASE,
)
# ``token`` alone is too common a word, so it counts as a secret name when
# the key is quoted (``'api_token': ...``), the value is quoted
# (``token = 'x'``), or it stands alone before a colon and a value that
# is 8 or more characters after a space (``token: 9f2c...``) or mixes
# letters and digits in 4 or more (``token:Ab3x``). A logging prefix
# (``WARNING:root:token: ...``) or a path (``/run/secrets/token: ...``)
# doesn't hide it. So a parser's ``Unexpected token: punc``, drawtext's
# ``Token: 5a`` and a namespaced ``app:token:refresh-window`` stay.
_TOKEN_NAME = (
    r"(?:(?(kq)token|(?!))|token(?=[ \t]*[:=][ \t]*(?:[bru]{1,2})?[\"'])"
    r"|(?<![A-Za-z0-9_-])token(?=[ \t]*:(?:[ \t]+[^\s,;}\]\"']{8}"
    r"|[ \t]*+(?=[^\s,;}\]\"':]{4})(?=[^\s,;}\]\"':]*[0-9])(?=[^\s,;}\]\"':]*[A-Za-z]))))"
)
_ASSIGNED_SECRET_RE = re.compile(
    rf"(?P<prefix>(?P<kq>[\"'])?(?<![A-Za-z0-9_-])_*(?!{_PLAIN_TOKEN_NAME}(?![\w-]))(?:[A-Za-z0-9]+[_-])*"
    r"(?-i:(?:[a-z0-9]+(?:[A-Z][a-z0-9]+)*?(?=[A-Z]))?)"
    rf"(?:{_KNOWN_SECRET_ENV}|{_SECRET_NAME}|{_TOKEN_NAME})\b"
    r"(?(kq)(?P=kq))[ \t]*(?:(?P<colon>:)|=)[ \t]*+)"
    # A list value goes whole. A bytes or raw string keeps its quotes.
    # An unquoted true/false/none/null is a flag (``hasPassword=true``).
    r"(?P<value>\[[^\[\]\r\n]*\]"
    rf"|(?:[bru]{{1,2}})?(?P<vq>[\"'])(?!{re.escape(REDACTION_MARKER)})"
    r"(?:\\.|(?!(?P=vq))[^\r\n\\])*(?P=vq)"
    rf"|(?![\"']?{re.escape(REDACTION_MARKER)})(?!(?:true|false|none|null)\b)"
    r"(?(colon)[^,;\r\n}\]\"']+|[^\s,;\r\n}\]]+))",
    re.IGNORECASE,
)
# A quoted value (``-passphrase "correct horse"``) runs to its closing
# quote, which stays. In an argument list (``"--api-key", "..."``) the
# value is the next item, and an empty one is left alone.
_CLI_SECRET_RE = re.compile(
    rf"(?P<prefix>(?<![\w-])--?(?:[A-Za-z0-9]+[_-])*{_SECRET_NAME}"
    r"(?:(?:=|[ \t]+)(?P<vq>[\"'])?|[\"'],[ \t]*[\"']))"
    r"(?P<value>(?(vq)(?:\\.|(?!(?P=vq))[^\r\n\\])+|[^\s\"']+))",
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
# Every run between separators. The callback skips runs with no escape;
# a pattern that looked for the escape itself rescanned long runs.
_PERCENT_TOKEN_RE = re.compile(r"[^\s\"'<>]+")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
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
    # A start right after "-" reads at most 1,024 characters of the first
    # segment, so a run of ``eyJ-`` can't make every start rescan the rest.
    re.compile(
        r"(?:(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8,}+|(?<=-)eyJ[A-Za-z0-9_-]{8,1024}+)"
        r"\.[A-Za-z0-9_-]{8,}+\.[A-Za-z0-9_-]{8,}\b"
    ),
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


# What may follow the ``@`` that ends a userinfo cut short of the run's
# last ``@``: a dotted host, ``localhost`` or an IP, a port, then a path.
# A ``?`` or ``#`` there could still be a password's (``P@ss.w0rd#1``).
_URL_HOST_PATH_RE = re.compile(
    r"(?:localhost|[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+|\[[0-9A-Fa-f:.]+\])(?::\d+)?/"
)


def _redact_url_userinfo(match: re.Match[str]) -> str:
    """Redact from ``scheme://`` to the ``@`` that ends the userinfo.

    An earlier ``@`` is used only when a plain host and path follow it
    and no other URL does, which keeps ``?to=ops@example.com`` and
    ``/media/@user/clip.mp4`` readable. Otherwise the run's last ``@``
    ends the userinfo or comes after it, so cutting there can't leak the
    password. But with an ``=`` before it, that ``@`` may sit inside a
    named value (``?password=Pa@ss``) whose name the cut would drop, so
    the whole run goes.
    """
    rest = match.group("rest")
    last = rest.rfind("@")
    if last < 0:
        return match.group(0)
    at = _readable_userinfo_end(rest, _userinfo_stop(rest))
    if at is None:
        if "=" in rest[rest.find("@"):last]:
            return f"{match.group('prefix')}{REDACTION_MARKER}"
        at = last
    return f"{match.group('prefix')}{REDACTION_MARKER}{rest[at:]}"


def _userinfo_stop(rest: str) -> int:
    """Where the query or fragment after the run's first ``@`` starts."""
    first = rest.find("@")
    stop = len(rest)
    for mark in "?#":
        found = rest.find(mark, first)
        if found >= 0:
            stop = min(stop, found)
    return stop


def _readable_userinfo_end(rest: str, stop: int) -> int | None:
    """The last ``@`` before ``stop``, stepping back past an ``@`` that
    starts a path segment, or ``None`` when the text after it isn't
    plainly a host.
    """
    at = rest.rfind("@", 0, stop)
    while at > 0 and rest[at - 1] == "/":
        earlier = rest.rfind("@", 0, at)
        if earlier < 0:
            break
        at = earlier
    if at == rest.rfind("@"):
        return at
    if "://" in rest[at:] or not _URL_HOST_PATH_RE.match(rest, at + 1):
        return None
    return at


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
        except (ValueError, RecursionError):
            pass
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
            snake = _CAMEL_BOUNDARY_RE.sub("_", str(key))
            secret = bool(_SECRET_KEY_RE.search(snake)) and not _PLAIN_TOKEN_KEY_RE.fullmatch(snake)
            if secret and isinstance(item, str) and item:
                out[name] = REDACTION_MARKER
            elif secret and isinstance(item, list):
                # A list of secrets (``"password": ["..."]``) loses each string.
                out[name] = [
                    REDACTION_MARKER if isinstance(entry, str) and entry else _redact_structure(entry)
                    for entry in item
                ]
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
    redacted = _redact_unescaped(text)
    if _PERCENT_ESCAPE_RE.search(redacted):
        redacted = _PERCENT_TOKEN_RE.sub(_redact_escaped_token, redacted)
    return redacted


def _redact_escaped_token(match: re.Match[str]) -> str:
    """Redact a secret that only shows once its token is percent-decoded
    (a URL nested in a query string, ``auth=Bearer%20...``).

    The token is replaced by its decoded, redacted form only when that
    finds something, so ``frame_%04d.png`` and ``my%20clip.mp4`` keep
    their escapes. It is decoded up to three times, for a URL encoded
    again each time it was nested (``%253Ftoken%253D...``), and redacted
    after each round, before the next one can turn an escaped ``&``,
    ``#`` or space inside a secret into a separator that splits it.
    """
    token = match.group(0)
    current = token
    found = False
    for _ in range(3):
        if not _PERCENT_ESCAPE_RE.search(current):
            break
        decoded = urllib.parse.unquote(current)
        current = _redact_unescaped(decoded)
        found = found or current != decoded
    return current if found else token


def _redact_unescaped(text: str) -> str:
    redacted = _scrub_paths(text)
    redacted = _PRIVATE_KEY_RE.sub(REDACTION_MARKER, redacted)
    redacted = _SECRET_HEADER_RE.sub(_redact_match_value, redacted)
    redacted = _URL_USERINFO_RE.sub(_redact_url_userinfo, redacted)
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
