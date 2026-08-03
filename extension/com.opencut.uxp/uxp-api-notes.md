# UXP API Notes

## Status

UXP for Premiere Pro became GA in Premiere Pro 25.6 (late 2025).
Timeline manipulation APIs are still being added incrementally.
The `premierepro` module name and exact API surface are subject to change — always check
the official Adobe docs before shipping.

## Available in Premiere Pro 25.6+

- Project panel access (`app.getProjectList()`)
- Basic sequence info (`sequence.getSettings()`, `getName()`, `getEnd()`)
- Marker read/write via `MarkerList` API (`createMarker`, `setName`, `setColorIndex`)
- Project item enumeration
- Application events (`app.on("afterActivateSequence", ...)`)
- `localFileSystem` for native file/folder dialogs
- `fetch()` natively (no CEP/CSInterface bridge required)
- ES module imports (`type="module"` in `<script>`)

## Timeline Write-Back

Full clip insertion/deletion API (ripple delete, clip move, track management):
currently in preview/incremental rollout as of Premiere Pro 25.6. Check
`sequence.rippleDelete()` availability in the installed build.

**Workaround:** The CEP panel (`com.opencut.panel`) handles all write-back for
Premiere < 25.6 and as a fallback if the UXP timeline API is not yet available
on a given build. The UXP panel's `PProBridge.applyCuts()` and
`PProBridge.addMarkers()` degrade gracefully with a user-facing warning when
the API call fails.

## Key Module Names (25.6)

```js
import("premierepro")       // main Premiere Pro UXP module
import("uxp")               // UXP platform module (localFileSystem, etc.)
```

If either import throws, the panel falls back to HTTP-only mode against the
Python backend and displays an info notice on the Timeline tab.

## CSRF Tokens

The Python backend (`opencut.server`) issues CSRF tokens via `GET /csrf`.
The UXP panel fetches this on init and attaches it as `X-CSRF-Token` on every
mutating request. The backend returns a refreshed token in response headers.

## Permissions

The `manifest.json` `requiredPermissions` block controls what the plugin can
access:

| Permission | Value | Purpose |
|---|---|---|
| `network.domains` | `127.0.0.1:5679`, `localhost:5679` | Backend HTTP calls |
| `localFileSystem` | `request` | Picker-scoped file/folder browse dialogs |
| `clipboard` | `readAndWrite` | Generated-output copy button writes text to the system clipboard |
| `launchProcess.schemes` | `https` | OAuth authorization pages opened in the system browser |
| `ipc.enablePluginCommunication` | `true` | Inter-plugin messaging |

## Manifest Schema

The shipped UXP panel declares `manifestVersion: 5`, which is the Premiere
supported manifest schema. The dormant Bolt/WebView scaffold keeps
`manifestVersion: 6` as a future cutover template and is not the live entrypoint.

## Deprecated API Sentinel

The live UXP/WebView source path uses `navigator.clipboard.writeText(...)` with
a string value and does not use deprecated `Clipboard.setContent`,
`Clipboard.getContent`, `Clipboard.clearContent`, object-form clipboard writes,
or legacy `uxpvideo*` events. `tests/test_uxp_deprecation_sentinel.py` pins that
contract for the F252 cutover path.

## Clipboard Permission

Adobe's Premiere UXP manifest contract defaults clipboard access to unavailable.
OpenCut declares `requiredPermissions.clipboard: "readAndWrite"` in both the live
manifest and dormant WebView scaffold because the output copy button writes text
to the system clipboard. `copyTextToClipboard()` centralizes the runtime path and
falls back to a manual-copy warning when clipboard access is unsupported or
permission is denied. `tests/test_uxp_clipboard_permission.py` guards this.

## Filesystem Permission

OpenCut's live UXP file access is picker-scoped: `browseFile()` calls
`localFileSystem.getFileForOpening(...)`, and `browseFolder()` calls
`localFileSystem.getFolder()`. The live manifest and dormant WebView scaffold
therefore declare `requiredPermissions.localFileSystem: "request"` instead of
`"fullAccess"`. `tests/test_uxp_filesystem_permission.py` guards the narrowed
manifest value and blocks direct file APIs that would require a new permission
review.

## External Launch Permission

OpenCut only opens external URLs for social-account OAuth authorization. The
live manifest and dormant WebView scaffold declare
`requiredPermissions.launchProcess.schemes: ["https"]` with an empty extension
allowlist. `openHttpsExternalUrl()` rejects non-HTTPS auth URLs before calling
`shell.openExternal(...)`, and the fallback shows the validated HTTPS URL for
manual browser entry if UXP denies the launch. `tests/test_uxp_external_launch_permission.py`
guards the manifest, helper, WebView scaffold, and no-`openPath()` contract.

## Confirmation Policy

The live manifest does not enable beta UXP browser alerts. Search-index clearing
uses an inline second-click panel confirmation with an 8-second window instead
of `window.confirm`. `tests/test_uxp_confirmation_guard.py` blocks raw
`window.alert`, `window.prompt`, `window.confirm`, and bare dialog calls in UXP
source.

## Differences from CEP Panel

| Aspect | CEP (`com.opencut.panel`) | UXP (`com.opencut.uxp`) |
|---|---|---|
| Host API | ExtendScript `.jsx` via `CSInterface.evalScript()` | `premierepro` UXP module (`async/await`) |
| HTTP | Via Node.js `http` module (CEP Node context) | Native `fetch()` |
| File dialogs | `openDialog()` from CSInterface | `localFileSystem.getFileForOpening()` |
| JS standard | ES5/ES6, CommonJS | ES2022+, ES modules |
| Min Premiere | 13.0 (CC 2019) | 25.6 |
| Theme sync | Reads `__adobe_cep__` host data | CSS variables only |

## macOS HTTP behaviour

UXP on Premiere Pro 25.6.x has known first-fetch stalls and WebSocket
retry-on-first-failure behaviour against `http://127.0.0.1:5679`. The
panel already handles this in `main.js` (port autodiscovery,
`fetchWithTimeout`, exponential health-check backoff, WS reconnect).
See [`docs/UXP_MACOS_HTTP.md`](../../docs/UXP_MACOS_HTTP.md) for the
full workaround set and the planned auto-HTTPS sidecar (F259).

## References

- Adobe UXP Developer Docs: https://developer.adobe.com/premiere-pro/uxp/
- UXP Plugin Samples: https://github.com/AdobeDocs/uxp-premiere-pro-samples
- UXP Community Forum: https://community.adobe.com/t5/premiere-pro/ct-p/ct-premiere-pro
