# OpenCut Bolt/WebView Scaffold

This directory is the F252.1 scaffold for moving the UXP panel toward the Bolt
UXP WebView architecture. It is intentionally dormant: the shipped UXP panel
still loads `../index.html` from `../manifest.json` until the WebView UI has a
host-API smoke test inside Premiere.

The live UXP panel declares `manifestVersion: 5`, matching Premiere's supported
manifest schema. This scaffold keeps `manifestVersion: 6` only for the future
WebView/hybrid-style manifest generated during the F252 cutover.

Both the live manifest and this scaffold declare clipboard `readAndWrite`
permission because the panel writes generated output text to the system
clipboard and handles denied access with a manual-copy fallback.

Both manifest surfaces also declare the external-launch contract as HTTPS-only:
`launchProcess.schemes` is `["https"]`, `launchProcess.extensions` is empty,
and WebView `openURL()` rejects non-HTTPS URLs before calling the UXP shell.

The scaffold follows the current Bolt UXP WebView split:

- `uxp.config.ts` is the least-privilege manifest/config template.
- `src/api/uxp.ts` exposes generic UXP host functions to the WebView context.
- `src/api/premierepro.ts` exposes Premiere-specific host functions.
- `webview-ui/src/webview-setup.ts` owns the `window.uxpHost.postMessage`
  bridge used by the browser-like WebView context.
- `webview-ui/src/webview-api.ts` contains functions the UXP host can call back
  into the WebView.

Cutover sequence:

1. Move the existing panel HTML/CSS/JS into `webview-ui/` without rewriting the
   UI in Spectrum widgets.
2. Route all former `CSInterface.evalScript()` calls through `src/api/*`.
3. Enable the generated WebView manifest after `tests/test_uxp_webview_scaffold.py`
   and an in-Premiere UDT smoke pass both succeed.
4. Keep CEP only for `ocAddNativeCaptionTrack` and `ocQeReflect` until F253 or
   Adobe replacement APIs land.
