# OpenCut Telemetry

OpenCut telemetry is disabled on fresh installs. F250 makes Aptabase the
default opt-in analytics provider for desktop usage signals, but core editing,
captioning, export, and local AI workflows still run without telemetry, cloud
accounts, or analytics keys.

## Provider

Aptabase was chosen because it is built for app analytics rather than web-page
analytics, is open source, publishes EU/US data-residency app-key prefixes, and
documents anonymous sessions without cookies, fingerprinting, or long-term user
identifiers.

Primary references:

- Aptabase privacy/product page: https://aptabase.com
- Aptabase SDK contract: https://github.com/aptabase/aptabase/wiki/How-to-build-your-own-SDK
- Aptabase Python SDK: https://github.com/aptabase/aptabase-python

## Opt-In Controls

Local settings live in `~/.opencut/telemetry_settings.json` and are accessed
through `opencut.user_data` atomic JSON helpers.

API surface:

- `GET /telemetry/aptabase/info` returns provider status, queue depth, masked
  app-key state, and privacy guarantees.
- `GET /telemetry/aptabase/settings` returns the local opt-in settings with the
  app key masked.
- `POST /telemetry/aptabase/settings` enables, disables, or updates settings.
  It requires the normal OpenCut CSRF header.
- `POST /telemetry/aptabase/track` queues a best-effort event only when
  telemetry is enabled and configured.

Environment overrides:

- `OPENCUT_TELEMETRY_ENABLED=1` explicitly enables telemetry for the process.
- `OPENCUT_APTABASE_APP_KEY` or `APTABASE_APP_KEY` provides the app key.
- `OPENCUT_APTABASE_BASE_URL` or `APTABASE_BASE_URL` provides a self-hosted
  server URL for `A-SH-*` app keys.
- `OPENCUT_TELEMETRY_DEBUG=1` marks events as debug events.

## Wire Contract

OpenCut posts batches of up to 25 events to:

```text
{host}/api/v0/events
```

It sends the `App-Key` header and an event list containing `timestamp`,
`sessionId`, `eventName`, `systemProps`, and scrubbed `props`, matching
Aptabase's documented SDK contract.

App-key host resolution:

- `A-EU-*` uses `https://eu.aptabase.com`.
- `A-US-*` uses `https://us.aptabase.com`.
- `A-SH-*` requires an explicit public `base_url`.

Self-hosted URLs are validated with the same public-HTTP SSRF guard used by
OpenCut webhook/download integrations. Localhost and private-network targets
are rejected from route-provided settings; use a public reverse proxy for a
self-hosted Aptabase instance.

## Privacy Boundary

The client drops or truncates event properties before network dispatch:

- Drops keys that look like file paths, filenames, media paths, transcripts,
  prompts, API keys, tokens, cookies, passwords, secrets, or email fields.
- Drops values that look like URLs, email addresses, Windows paths, UNC paths,
  or common Unix/macOS user-media paths.
- Limits each event to 30 primitive properties and caps string lengths.
- Sends no raw clip names, project paths, transcript text, prompt text, API
  tokens, or user email fields.
- Uses anonymous Aptabase-style session IDs with a one-hour rotation window.

Telemetry failures never fail the user operation. Events are queued in memory,
sent on a daemon worker thread, and dropped on process exit or unreachable
servers.

## Legacy Plausible Path

`/telemetry/plausible/*` remains available for older self-hosted deployments,
but new OpenCut telemetry should use Aptabase. Plausible still requires
`PLAUSIBLE_HOST` and `PLAUSIBLE_DOMAIN` environment variables and remains
disabled when those variables are absent.
