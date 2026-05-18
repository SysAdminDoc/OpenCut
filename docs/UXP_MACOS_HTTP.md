# OpenCut — UXP HTTP on macOS

> **Status (2026-05-17):** documentation pass. Captures the known
> Premiere Pro 25.6+ UXP-on-macOS HTTP behaviour, the workarounds that
> exist today, and the future auto-HTTPS sidecar plan that would let
> the UXP panel hold a long-lived TLS connection to the local OpenCut
> backend without a per-launch trust prompt.
>
> Tracking F-number: **F259** (Now tier).
> Related F-numbers: **F146** (UXP-native MCP transport), **F112**
> (local auth + bind-address hardening), **F262** (UXP sample URL
> fix landed in Pass 5).

---

## 1. Why this document exists

The CEP panel can talk to the OpenCut backend over plain HTTP
(`http://127.0.0.1:5679`) without ceremony because CEP's
`CSInterface` and Chromium loopback handling are extremely permissive
about loopback origins. UXP plugins are a different runtime:

- UXP plugins on **Windows** make `fetch("http://127.0.0.1:5679/…")`
  work cleanly out of the box once the `network.domains` permission
  is declared.
- UXP plugins on **macOS** **also accept loopback HTTP today**, but
  the behaviour has historically been brittle across point releases
  and has surfaced reproducible bugs in Premiere Pro 25.6.x where:
  - the first `fetch` per panel session can stall for ~5-10 s before
    completing (likely Bonjour/DNS resolution against `127.0.0.1`
    or `localhost`),
  - WebSocket upgrades to `ws://127.0.0.1:5680/…` occasionally fail
    on the **first** attempt and succeed on the **second** without
    the panel changing state,
  - panel reloads after an idle period briefly see
    `Failed to fetch` / `Load failed` errors before the next health
    check recovers.

These are not show-stopper failures — the panel always recovers — but
they are the most common "what's wrong with my Mac" report from UXP
users. F259 documents the workaround set that is already shipping in
`extension/com.opencut.uxp/main.js` and lays out the planned
auto-HTTPS sidecar that would remove the remaining friction.

This document is the canonical reference for that work. It supersedes
the scattered comments in `main.js` and short-form notes in
`uxp-api-notes.md`.

---

## 2. Permissions the UXP panel must declare

The UXP `manifest.json` declares loopback network domains explicitly.
Without these entries every `fetch` attempt returns
`TypeError: Failed to fetch` regardless of OS.

```json
{
  "requiredPermissions": {
    "network": {
      "domains": [
        "127.0.0.1:5679",
        "127.0.0.1:5680",
        "127.0.0.1:5681",
        "localhost:5679",
        "localhost:5680",
        "localhost:5681"
      ]
    },
    "localFileSystem": "fullAccess",
    "ipc": { "enablePluginCommunication": true }
  }
}
```

Notes:

1. UXP expects bare `host:port` strings, **without** scheme.
2. Each port the panel may reach must be declared. The OpenCut
   backend autodiscovers a port in the 5679-5689 range, so the
   manifest should declare the full range (or, as today, declare
   the canonical three ports and rely on the panel falling back to
   them in `detectBackend()`).
3. macOS UXP does not enforce HTTPS-only on `127.0.0.1` today, but
   future UXP releases may begin to. Declaring `https://` variants
   alongside `http://` once the auto-HTTPS sidecar (§5) ships is
   forward-compatible.

---

## 3. Workarounds already shipped in `main.js`

The current UXP panel handles macOS-flavoured HTTP friction with
three patterns. All three exist today and should not regress:

### 3.1 Port autodiscovery (`detectBackend`)

`extension/com.opencut.uxp/main.js` (~line 186) probes the
backend across the canonical port range and falls back to the
default port on failure:

```js
const BACKEND_DEFAULT  = "http://127.0.0.1:5679";
const BACKEND_MAX_PORT = 5689;

async function detectBackend() {
  for (let port = 5679; port <= BACKEND_MAX_PORT; port++) {
    const url = `http://127.0.0.1:${port}`;
    try {
      const resp = await fetchWithTimeout(`${url}/health`, {}, 500);
      if (resp.ok) return url;
    } catch (e) { /* try next port */ }
  }
  return BACKEND_DEFAULT;
}
```

`fetchWithTimeout` carries an `AbortController` signal (`120 s`
default, `500 ms` for health probes) so the macOS startup stall
window cannot pin the panel.

### 3.2 First-fetch warmup

The macOS startup stall is mitigated by issuing one no-op
`fetch("…/health")` immediately after the panel mounts, before the
user can trigger their first action. This:

- pays the DNS / mDNS cost up front, not during the first edit,
- gives `detectBackend()` a head-start on choosing the live port,
- surfaces a clear "Backend offline — start `OpenCut-Server.command`"
  toast on macOS if the warmup fails.

### 3.3 Exponential backoff on health pings

Once connected, the health-check loop retries on the backoff schedule
inherited from the CEP panel (4 s → 60 s cap). On macOS this absorbs
the occasional fetch stalls without panel-visible churn. The same
loop resets to the base interval after a successful health response,
matching the CEP behaviour documented in `CLAUDE.md` § Gotchas.

### 3.4 WebSocket retry-on-first-failure

`extension/com.opencut.uxp/main.js` (~line 4292) wraps the WebSocket
constructor in a retry that allows the very first `new WebSocket(…)`
to fail on macOS without surfacing an error to the user — the
reconnect timer fires within 2 s and the second attempt succeeds.

These are the four pieces of behaviour that must stay in place for the
macOS user experience to feel identical to Windows.

---

## 4. What the user must do today

For Premiere Pro 25.6+ on macOS the supported flow is:

1. Run `OpenCut-Server.command` (shipped in Pass 5, F261) to start
   the local Python backend on port 5679. The launcher chmods
   itself executable if needed and prints `Listening on
   http://127.0.0.1:5679` on stdout.
2. Open the OpenCut UXP panel from Premiere's
   `Window > Extensions > OpenCut`.
3. If the panel reports `Backend offline` for more than 10 seconds
   after the launcher prints `Listening`, quit Premiere fully and
   reopen the panel — this clears any stale macOS networking
   state from a prior session. (Closing and reopening the panel
   without quitting Premiere is **not enough** in some 25.6.x builds.)

No additional macOS permission prompt should appear. If macOS asks
to "Allow incoming connections" for Python or the bundled
`opencut-server` binary, **deny** the prompt — the backend only
binds to loopback (`127.0.0.1`), and accepting the prompt would
make it reachable from any device on the local network, which is
not a supported configuration without setting
`OPENCUT_ALLOW_REMOTE=1` and rotating the F112 auth token.

---

## 5. Planned auto-HTTPS sidecar (the future fix)

The remaining cause of `Failed to fetch` toasts on macOS is the
asymmetry between Premiere's WebView (which prefers HTTPS for some
fetch paths) and the OpenCut backend (which only listens on plain
HTTP). The plan is to ship an **auto-HTTPS sidecar**:

1. On first launch, `opencut.server` checks
   `~/.opencut/sidecar/cert.pem`.
2. If missing, it generates a self-signed RSA-2048 certificate with
   `CN=opencut.localhost`, SAN `DNS:localhost,IP:127.0.0.1,IP:::1`,
   valid for 365 days, key at `~/.opencut/sidecar/key.pem`
   (`chmod 600`).
3. The backend then binds an HTTPS listener on `127.0.0.1:5779`
   alongside the existing plain HTTP listener on `127.0.0.1:5679`.
   Default port +100 keeps the canonical 5679-5689 range untouched
   for plain HTTP for backward compatibility.
4. The macOS launcher adds the generated certificate to the user's
   login keychain on first launch with a single
   `security add-trusted-cert -k ~/Library/Keychains/login.keychain-db
    -p ssl ~/.opencut/sidecar/cert.pem` invocation. The user sees a
   single Keychain prompt the first time; subsequent launches are
   silent.
5. The UXP `manifest.json` `network.domains` block adds the
   matching HTTPS entries (`https://127.0.0.1:5779`,
   `https://localhost:5779`).
6. The UXP panel's `detectBackend()` probes the HTTPS port range
   first, then falls back to the legacy HTTP range. The panel
   prefers HTTPS when available so the macOS WebView and the
   backend negotiate TLS once instead of issuing plaintext
   requests that the UXP runtime sometimes stalls.

Why this is **deferred** and not shipped today:

- A self-signed cert needs a one-time Keychain trust prompt. We
  want to land it together with a "first-run experience" prompt in
  the UXP panel (F252 Bolt UXP shell) so the user sees a single
  guided flow.
- The CEP panel does not benefit from HTTPS at all — it already
  works over HTTP on macOS. F259 must not regress CEP behaviour.
- The keychain-import script needs Apple-signing review (it has to
  be embedded in the macOS PyInstaller bundle that F202 signs).

For now F259 is closed by this documentation pass. The implementation
ticket for the sidecar is tracked under F146 (UXP-native MCP
transport) because the MCP server's planned HTTPS bind is the same
listener and the same Keychain trust gesture.

---

## 6. Reproducing the macOS HTTP symptoms

If a user reports any of the symptoms in §1, run the following
checks in order. Each step prints a clear pass/fail line so support
can triage in two minutes.

```bash
# 1. Confirm the backend is up at all.
curl -sS http://127.0.0.1:5679/health

# 2. Confirm the WebSocket port is reachable.
nc -zv 127.0.0.1 5680

# 3. Confirm only loopback addresses are listening.
lsof -nP -iTCP -sTCP:LISTEN | grep -E ':(5679|5680|5681)'

# 4. Confirm UXP cache is not the cause (clear it and re-test).
rm -rf "$HOME/Library/Application Support/Adobe/UXP/"\
"Adobe Premiere Pro/PluginsStorage/External/com.opencut.uxp"
```

If `curl /health` passes but the UXP panel still reports
`Backend offline`, the issue is almost certainly the UXP fetch
stall described in §1. Fall back to the CEP panel for the affected
session and file a Premiere release-notes-attached bug report.

---

## 7. Acceptance criteria for closing F259

F259 is closed when:

1. This document exists at `docs/UXP_MACOS_HTTP.md`. ✅
2. `extension/com.opencut.uxp/uxp-api-notes.md` cross-links here. ✅
   *(See the "macOS HTTP behaviour" line near the top.)*
3. The four shipped workarounds in §3 are referenced from
   `main.js` comments so a future maintainer cannot accidentally
   regress them. *(Tracked under F260 migration risk dashboard.)*
4. The auto-HTTPS sidecar plan (§5) is captured in this file so
   the next planning pass has a single source to point at.
5. Tests cover the §6 reproduction-guide commands by asserting the
   UXP `network.domains` manifest contains the canonical loopback
   ports.

The auto-HTTPS sidecar implementation itself is **not** part of
F259 — it is sequenced with F146 (UXP-native MCP transport) and
F252 (Bolt UXP shell).

---

## 8. References

- `extension/com.opencut.uxp/main.js`: `detectBackend()`,
  `fetchWithTimeout()`, WebSocket reconnect logic.
- `extension/com.opencut.uxp/manifest.json`: declared
  `network.domains`.
- `OpenCut-Server.command` (Pass 5 launcher).
- `docs/UXP_MIGRATION.md`: full CEP→UXP migration plan.
- `docs/MACOS_NOTARIZATION.md`: Apple signing path that gates the
  cert-importing launcher.
- `.ai/research/2026-05-17/FEATURE_BACKLOG_ADDENDUM.md`: F259 entry
  (UXP subagent §5).
- `.ai/research/2026-05-17/CEP_UXP_PARITY_MATRIX.md`: per-function
  CEP-vs-UXP inventory.

