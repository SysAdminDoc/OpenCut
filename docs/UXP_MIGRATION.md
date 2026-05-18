# OpenCut — CEP to UXP Migration Plan

> **Deadline:** CEP support removed from Premiere Pro ~September 2026
> **Current state:** Dual CEP + UXP panels. Pass-3 audit found 16/18 JSX host functions have a UXP path; 2 remain CEP-only.
> **Last updated:** 2026-05-18
> **Machine-readable catalogue:** `opencut/_generated/cep_uxp_parity.json` (generated from `opencut/core/cep_uxp_parity.py`)

## Migration Strategy

### Phase 1: WebView UI Wrapper (Weeks 1-2)
Use Bolt UXP's WebView UI feature (available March 2026) to embed the existing
CEP HTML/CSS/JS inside a UXP WebView. This gives us immediate feature parity
without rewriting UI code.

**Steps:**
1. Set up bolt-uxp project: `npx create-bolt-uxp`
2. Configure WebView UI in manifest
3. Copy `client/index.html`, `client/main.js`, `client/style.css` into WebView
4. Replace `CSInterface` calls with Comlink `postMessage()` to UXP host
5. Replace `evalScript()` ExtendScript calls with UXP Premiere API

### Phase 2: Replace ExtendScript with UXP API (Weeks 3-5)
Map each ExtendScript function in `host/index.jsx` to its UXP equivalent.

| ExtendScript Function | UXP Equivalent | Status |
|----------------------|----------------|--------|
| `getProjectMedia()` | `Project.getRootItem()` + walk | Implemented in PProBridge |
| `getSelectedClips()` | `Sequence.getSelection()` | Implemented in PProBridge |
| `browseForFile()` | `localFileSystem.getFileForOpening()` | Implemented |
| `importXMLToProject()` | `Project.importFiles()` | Implemented in PProBridge |
| `importFileToProject()` | `Project.importFiles()` | Implemented in PProBridge |
| `importCaptions()` | `Project.importFiles()` + caption track | Implemented (via importFiles) |
| `getSequenceClips()` | `Sequence.getVideoTrackList()` | Partial in PProBridge |
| `applyEditsToTimeline()` | `Sequence` clip insertion API | Partial (via applyCuts) |
| `ocAddSequenceMarkers()` | `Sequence.getMarkerList()` | Implemented in PProBridge |
| `ocApplySequenceCuts()` | Clip removal API | Partial (`applyCuts`) |
| `ocGetSequenceInfo()` | `Sequence.getSettings()` | Implemented in PProBridge |
| `ocBatchRenameProjectItems()` | Item rename API | Low priority (CEP ExtendScript works) |
| `ocCreateSmartBins()` | Bin creation API | Low priority (CEP ExtendScript works) |
| `ocAddNativeCaptionTrack()` | No UXP write API yet | CEP-only until Adobe ships `createCaptionTrack()` or F253 Hybrid Plugin adds it |
| `ocQeReflect()` | No supported UXP API | Retire after CEP EOL; replace individual QE use cases with documented UXP APIs |
| `startOpenCutBackend()` | UXP `shell.openExternal()` | N/A (backend started externally in UXP) |

## F266 — CEP Residual and Drop-QE Plan

The 2026-05-17 parity audit walked all 18 `ocXxx` functions in
`extension/com.opencut.panel/host/index.jsx` against the `@adobe/premierepro`
26.3 beta typings and `extension/com.opencut.uxp/main.js`.

| Function | Current role | Post-CEP plan |
|---|---|---|
| `ocAddNativeCaptionTrack(srtJSON)` | Creates a native Premiere caption track from SRT-style segments. | Keep CEP fallback while CEP exists. Long-term target is F253 Hybrid Plugin `createCaptionTrack(sequence, srtBytes)` or an Adobe UXP `createCaptionTrack()` API. Until then, UXP keeps SRT validation, project import, manual timeline placement, and burn-in captions as the fallback paths. |
| `ocQeReflect()` | Diagnostic probe for undocumented QE DOM methods, cached for `/system/qe-reflect`. | Do not port wholesale. Treat QE reflection as a research aid and retire it after CEP EOL. Replace real user workflows one by one with documented UXP APIs: `SequenceEditor.createRemoveItemsAction(ripple=true)` for most ripple-delete cases, `VideoFilterFactory` / `AudioFilterFactory` for effect creation, and explicit adjacent-clip recompute for advanced trim cases. |

Rules for new migration work:

1. No new user-facing feature should depend on QE reflection.
2. Hybrid Plugin scope should prioritize native caption-track creation before any QE-like wrapper.
3. A QE replacement only gets built if UDT testing proves a shipped workflow still lacks a documented UXP route.
4. The UXP panel should phrase unsupported native-caption and QE paths as capability limitations, not generic "UXP failed" errors.

### Phase 3: TypeScript + Framework Migration (Weeks 6-8)
1. Convert vanilla JS to TypeScript (prevents null-reference bugs)
2. Choose framework: Svelte (smallest bundle, best DX) or React (largest ecosystem)
3. Component-ize the UI (currently monolithic main.js)
4. Add proper state management

### Phase 4: Testing + Distribution (Weeks 9-10)
1. Test on Premiere Pro 25.6+ with UXP GA
2. Package as CCX for Adobe Marketplace
3. Maintain CEP version for older Premiere until deprecation
4. Set up GitHub Actions for automated CCX builds (bolt-uxp includes this)

## Architecture Comparison

```
CEP Architecture (current):
  Panel HTML/JS <-> CSInterface <-> ExtendScript (ES3) <-> Premiere API
       |
       v
  XHR/fetch <-> Python Backend (Flask)

UXP Architecture (target):
  WebView HTML/JS <-> Comlink postMessage <-> UXP Host JS <-> Premiere UXP API
       |
       v
  fetch <-> Python Backend (Flask)
```

## Key Differences to Handle
- UXP is NOT a browser — limited DOM, no `window.open()`, no `localStorage` (use UXP storage API)
- WebView UI restores full browser capabilities inside UXP
- UXP uses `import` instead of `require` for modules
- Node.js APIs (child_process, fs) are NOT available in UXP — use UXP file system API
- Theme: UXP provides host app theme info via `host.uiTheme`

## Implementation Progress

### Phase 1 Status: ~80% Ready
- [x] UXP panel with full feature parity (settings, engine registry, WebSocket, all video features)
- [x] `csinterface-shim.js` — Drop-in CSInterface replacement for WebView mode
- [x] Backend communication works natively (fetch in UXP/WebView)
- [x] F252.1 Bolt/WebView scaffold setup at `extension/com.opencut.uxp/bolt-webview/`
- [x] F252.1 WebView manifest/config template in `extension/com.opencut.uxp/bolt-webview/uxp.config.ts`
- [ ] Live manifest switch to the WebView entrypoint after an in-Premiere UDT smoke pass
- [ ] Test CSInterface shim with CEP main.js in WebView
- [ ] Replace `cep_node.require("child_process")` calls with UXP alternatives

### Phase 2 Status: ~75% Ready
- [x] `PProBridge` wraps UXP Premiere API (project items, markers, sequence info, cuts, selected clips, import)
- [x] `getSelectedClips()` — selection via `Sequence.getSelection()`
- [x] `importFiles()` — import via `Project.importFiles()` with optional bin
- [ ] Remaining UXP ports: `ocBatchRenameProjectItems()`, `ocCreateSmartBins()` (low priority, UI-only convenience)
- [ ] Residual CEP-only paths: native caption-track creation and QE reflection (documented in F266 above)
- [ ] Full timeline write-back without ExtendScript for advanced trim edge cases (blocked on UXP API maturity)

### Key Files
- `extension/com.opencut.uxp/csinterface-shim.js` — CSInterface→postMessage bridge for WebView
- `extension/com.opencut.uxp/main.js` — Native UXP panel (2000+ lines, full feature parity)
- `extension/com.opencut.uxp/index.html` — UXP panel UI with all tabs including Settings
- `extension/com.opencut.uxp/bolt-webview/` — F252.1 dormant Bolt/WebView scaffold with host API wrappers, WebView message bridge, and least-privilege config template
- `tests/test_uxp_webview_scaffold.py` — Static guardrails for the scaffold contract

## Risk Assessment
- **Low risk:** Backend communication (fetch works natively in UXP)
- **Medium risk:** File system access (UXP requires explicit permissions)
- **High risk:** ExtendScript replacement (UXP Premiere API is still evolving)
- **Mitigated:** WebView UI gives us full DOM when needed
