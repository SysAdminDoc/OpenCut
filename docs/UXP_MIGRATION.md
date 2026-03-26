# OpenCut — CEP to UXP Migration Plan

> **Deadline:** CEP support removed from Premiere Pro ~September 2026
> **Current state:** Dual CEP + UXP panels, UXP has ~40% feature parity

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
| `getSelectedClips()` | `Sequence.getSelection()` | TODO |
| `browseForFile()` | `localFileSystem.getFileForOpening()` | Implemented |
| `importXMLToProject()` | `Project.importFiles()` | TODO |
| `importFileToProject()` | `Project.importFiles()` | TODO |
| `importCaptions()` | `Project.importFiles()` + caption track | TODO |
| `getSequenceClips()` | `Sequence.getVideoTrackList()` | Partial in PProBridge |
| `applyEditsToTimeline()` | `Sequence` clip insertion API | TODO |
| `ocAddSequenceMarkers()` | `Sequence.getMarkerList()` | Implemented in PProBridge |
| `ocApplySequenceCuts()` | Clip removal API | Partial (`applyCuts`) |
| `ocGetSequenceInfo()` | `Sequence.getSettings()` | Implemented in PProBridge |
| `ocBatchRenameProjectItems()` | Item rename API | TODO |
| `ocCreateSmartBins()` | Bin creation API | TODO |
| `startOpenCutBackend()` | UXP `shell.openExternal()` | TODO |

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

## Risk Assessment
- **Low risk:** Backend communication (fetch works natively in UXP)
- **Medium risk:** File system access (UXP requires explicit permissions)
- **High risk:** ExtendScript replacement (UXP Premiere API is still evolving)
- **Mitigated:** WebView UI gives us full DOM when needed
