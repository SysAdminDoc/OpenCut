# OpenCut — CEP to UXP Migration Plan

> **Deadline:** CEP support removed from Premiere Pro ~September 2026
> **Current state:** Dual CEP + UXP panels, UXP has ~98% feature parity (March 2026)
> **Last updated:** 2026-03-26

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
| `startOpenCutBackend()` | UXP `shell.openExternal()` | N/A (backend started externally in UXP) |

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
- [ ] Bolt UXP project skeleton setup (`npx create-bolt-uxp`)
- [ ] WebView manifest configuration
- [ ] Test CSInterface shim with CEP main.js in WebView
- [ ] Replace `cep_node.require("child_process")` calls with UXP alternatives

### Phase 2 Status: ~75% Ready
- [x] `PProBridge` wraps UXP Premiere API (project items, markers, sequence info, cuts, selected clips, import)
- [x] `getSelectedClips()` — selection via `Sequence.getSelection()`
- [x] `importFiles()` — import via `Project.importFiles()` with optional bin
- [ ] Remaining: `ocBatchRenameProjectItems()`, `ocCreateSmartBins()` (low priority, UI-only convenience)
- [ ] Full timeline write-back without ExtendScript (blocked on UXP API maturity)

### Key Files
- `extension/com.opencut.uxp/csinterface-shim.js` — CSInterface→postMessage bridge for WebView
- `extension/com.opencut.uxp/main.js` — Native UXP panel (2000+ lines, full feature parity)
- `extension/com.opencut.uxp/index.html` — UXP panel UI with all tabs including Settings

## Risk Assessment
- **Low risk:** Backend communication (fetch works natively in UXP)
- **Medium risk:** File system access (UXP requires explicit permissions)
- **High risk:** ExtendScript replacement (UXP Premiere API is still evolving)
- **Mitigated:** WebView UI gives us full DOM when needed
