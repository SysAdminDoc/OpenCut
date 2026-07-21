# Security Policy

## Supported Versions

OpenCut ships rapidly. We actively support the **latest minor** (`1.40.x`) and the one immediately preceding it (`1.39.x`). Older minors receive security-only backports for 90 days after they're superseded.

| Version | Supported         | Security fixes until |
|---------|-------------------|----------------------|
| 1.40.x  | ✅ Active         | —                    |
| 1.39.x  | ✅ Previous       | +90 days after 1.40  |
| 1.38.x  | ⚠️ Critical only  | +30 days after 1.40  |
| ≤ 1.37  | ❌ End of life    | n/a                  |

Version numbers ship in [`opencut/__init__.py`](opencut/__init__.py) and are kept in sync by [`scripts/sync_version.py`](scripts/sync_version.py).

## Reporting a Vulnerability

**Please do not open public GitHub issues for security problems.**

Email [matt@mavenimaging.com](mailto:matt@mavenimaging.com) with:

1. A description of the issue — what you see, what you expected.
2. Reproducer steps, ideally as a minimal request / script / config.
3. The commit SHA + `__version__` you tested against.
4. Your assessment of severity (low / medium / high / critical) and why.

We acknowledge reports within **72 hours** and aim to land a fix or mitigation within:

- **Critical** — 72 hours (RCE, auth bypass, data exfiltration, sandbox escape)
- **High** — 7 days (privilege escalation, unauthenticated DoS on a production endpoint)
- **Medium** — 30 days (authenticated DoS, information disclosure, supply-chain risk)
- **Low** — 90 days (hardening suggestions, theoretical concerns)

We don't run a paid bounty programme, but we credit reporters in `CHANGELOG.md` and the release notes unless you prefer to remain anonymous.

## In Scope

- `opencut/` backend (Flask API, job system, CLI, MCP server)
- `extension/com.opencut.panel/` CEP panel (HTML/JS/ExtendScript)
- `extension/com.opencut.uxp/` UXP panel (HTML/JS)
- `installer/` (C# WPF Windows installer)
- `scripts/` build + utility scripts
- `tests/fuzz/` harness targets

## Out of Scope

- Third-party dependencies (report upstream). We audit `pyproject.toml`,
  `requirements.txt`, `requirements-lock.txt`, and panel dependencies locally
  with `python scripts/release_smoke.py --json`, `pip-audit`, npm advisory
  checks, and the declared SBOM generator.
- User-supplied plugins loaded via `~/.opencut/plugins/` — plugins run with the host's trust, so audit before installing.
- Social-engineering / phishing attacks against maintainers.
- Reports that require pre-existing local code execution (e.g. "an attacker with shell access can edit `~/.opencut/settings.json`").

## Known-Safe-By-Design Surface

OpenCut's security model leans on a handful of intentional choices:

- **CSRF on every mutation.** `@require_csrf` decorator on all `POST`/`PUT`/`PATCH`/`DELETE` routes. Token rotates per server start, delivered via `GET /health`, sent as `X-OpenCut-Token` header.
- **Path validation.** All file-accepting routes pass user-supplied paths through `security.validate_path()` / `validate_filepath()` / `validate_output_path()`. Realpath resolution, null-byte rejection, symlink-out-of-allowlist defence.
- **SSRF defence.** Outbound URL validators (`_validate_webhook_url`, `_validate_download_url`) reject localhost, loopback, private IPs, link-local, reserved ranges.
- **Rate-limit categories.** Four-way classification (`gpu_heavy` / `cpu_heavy` / `io_bound` / `light`) bounds concurrent work per category — see `core/rate_limit_categories.py`.
- **Scripting console sandbox.** Dunder builtins stripped, `__import__` / `exec` / `eval` / `compile` / `open` / `os` / `sys` / `subprocess` blocked in AST. Context keys containing `__` rejected.
- **Fuzz harness** for parsers (`tests/fuzz/`) — SRT / VTT / `.cube` / voice-grammar parsers are expected to be total.
- **Atomic writes** for user-data files via `tempfile + os.replace`.

## Hardening Recommendations

Operators running OpenCut in a shared-network environment should:

1. Bind to `127.0.0.1` only (default) — the service is single-user. Non-loopback binds require `OPENCUT_ALLOW_REMOTE=1`.
2. **Use the persistent local auth token when binding non-loopback.**
   Setting `OPENCUT_ALLOW_REMOTE=1` automatically issues a token under
   `~/.opencut/auth.json` (POSIX: mode `0600`). Every non-loopback
   request must include `X-OpenCut-Auth: <token>`. Loopback peers
   (`127.0.0.1`, `::1`) still bypass the token to keep the single-user
   workflow snappy.
3. Read or rotate the token explicitly:
   ```bash
   opencut-server --print-auth     # print the persisted token
   opencut-server --rotate-auth    # generate a fresh token, then exit
   ```
   The `GET /auth/info` endpoint returns *metadata only* — it never
   includes the token value. Treat `~/.opencut/auth.json` like an SSH
   private key; never check it into source control.
4. Set `SENTRY_DSN` so crashes route to a tracker you control.
5. Set `PLAUSIBLE_HOST` + `PLAUSIBLE_DOMAIN` (optional) for usage telemetry.
6. Configure `OPENCUT_TEMP_CLEANUP_*` to fit the expected workload.
7. Use the bundled FFmpeg or build FFmpeg explicitly — distro builds can lag on CVE fixes.
8. Keep `~/.opencut/plugins/` empty until you've audited each plugin manifest.

### Threat model for non-loopback binds

Default deployment: a single user, on their workstation, talking to
`127.0.0.1:5679`. The Adobe Premiere CEP/UXP panel sits on the same
machine. We deliberately do **not** require an API key in that path
because the token would be visible to anything that can read the panel
preferences anyway.

When the operator opts into `OPENCUT_ALLOW_REMOTE=1` (e.g. remote
render host on a private VLAN), the threat surface changes:

- Anyone who can hit the bind address can issue render jobs, read media
  paths, or call shell-adjacent endpoints (FFmpeg invocations, OS
  shell-out for ``open``/Finder integration).
- CSRF alone is not enough — CSRF protects browser sessions, not API
  clients on the same network.

The local auth token closes that gap: non-loopback callers must include
the token in the `X-OpenCut-Auth` header. Query-string tokens are not
accepted — URLs leak into access logs, browser history, and `Referer`
headers, so the header is the only credential channel. `/health` and
`/auth/info` remain exempt so panels can bootstrap connectivity and
render a "Authentication required" hint.

## Software Bill of Materials (SBOM)

Generate a declared-dependency CycloneDX SBOM from `pyproject.toml`,
`requirements.txt`, and OpenCut model-card metadata:

```bash
python scripts/sbom.py
```

The script writes `dist/opencut-declared-sbom.cyclonedx.json` (or `.xml` with
`--format xml`) and marks the CycloneDX metadata as `declared-only`. The
committed `requirements-lock.txt` is audited separately by the pip-audit release
gate; it is not presented as a resolved installed-package SBOM inventory.
