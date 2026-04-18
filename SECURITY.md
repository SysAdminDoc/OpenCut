# Security Policy

## Supported Versions

OpenCut ships rapidly. We actively support the **latest minor** (`1.24.x`) and the one immediately preceding it (`1.23.x`). Older minors receive security-only backports for 90 days after they're superseded.

| Version | Supported         | Security fixes until |
|---------|-------------------|----------------------|
| 1.24.x  | ✅ Active         | —                    |
| 1.23.x  | ✅ Previous       | +90 days after 1.25  |
| 1.22.x  | ⚠️ Critical only  | +30 days after 1.24  |
| ≤ 1.21  | ❌ End of life    | n/a                  |

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

- Third-party dependencies (report upstream). We monitor `pyproject.toml` / `requirements.txt` via Dependabot.
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

1. Bind to `127.0.0.1` only (default) — the service is single-user.
2. Set `SENTRY_DSN` so crashes route to a tracker you control.
3. Set `PLAUSIBLE_HOST` + `PLAUSIBLE_DOMAIN` (optional) for usage telemetry.
4. Configure `OPENCUT_TEMP_CLEANUP_*` to fit the expected workload.
5. Use the bundled FFmpeg or build FFmpeg explicitly — distro builds can lag on CVE fixes.
6. Keep `~/.opencut/plugins/` empty until you've audited each plugin manifest.

## Software Bill of Materials (SBOM)

Generate a CycloneDX SBOM from the pinned dependencies:

```bash
python scripts/sbom.py
```

The script writes `dist/opencut-sbom.cyclonedx.json` (or `.xml` with `--format xml`).
