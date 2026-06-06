# OpenCut — CEP to UXP Migration Plan

> **Deadline:** CEP support removed from Premiere Pro ~September 2026
> **Current state:** Dual CEP + UXP panels. Pass-3 audit found 16/18 JSX host functions have a UXP path; 2 remain CEP-only.
> **Last updated:** 2026-05-19
> **Machine-readable catalogue:** `opencut/_generated/cep_uxp_parity.json` (generated from `opencut/core/cep_uxp_parity.py`)
> **UDT smoke harness:** `opencut/_generated/uxp_udt_harness.json` and bundled panel copy `extension/com.opencut.uxp/uxp-udt-harness.json`
> **UDT result validation:** `python -m opencut.tools.validate_uxp_udt_results <capture.json> --json`

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
- [x] RA-17 live manifest schema guard: `extension/com.opencut.uxp/manifest.json`
      declares Premiere-supported `manifestVersion: 5`; the dormant Bolt/WebView
      scaffold keeps `manifestVersion: 6` until its separate UDT cutover.
- [x] F252.1 Bolt/WebView scaffold setup at `extension/com.opencut.uxp/bolt-webview/`
- [x] F252.1 WebView manifest/config template in `extension/com.opencut.uxp/bolt-webview/uxp.config.ts`
- [x] F252.2 UXP host-action dispatcher maps the 14 direct-UXP `ocXxx` actions into `PProBridge.executeHostAction`
- [x] F254 `Sequence.createSubsequence(ignoreTrackTargeting?)` range integration behind `ocExportSequenceRange`
- [x] F255 `EncoderManager.launchEncoder` / `exportSequence` / `startBatchEncode` handoff for range exports
- [x] F256 `Transcript.hasTranscript` / `querySupportedLanguages` helpers for caption-QC host context
- [x] F257 `ObjectMaskUtils.hasObjectMask` helpers for active-sequence and project-level mask detection
- [x] F258 `ProjectConverter.exportAAF` active-sequence export helper with `AAFExportOptions`
- [x] F260 generated UXP migration risk dashboard in Settings, sourced from the F198 CEP/UXP parity catalogue
- [x] F267 UDT smoke harness for the 14 direct-UXP `ocXxx` actions, exposed as `window.OpenCutUXPUdtHarness`
- [x] F252.3 UDT result-capture validator for strict WebView cutover readiness checks
- [ ] Live manifest switch to the WebView entrypoint after an in-Premiere `window.OpenCutUXPUdtHarness.run({ includeMutating: true })` capture passes `python -m opencut.tools.validate_uxp_udt_results`
- [ ] Test CSInterface shim with CEP main.js in WebView
- [ ] Replace `cep_node.require("child_process")` calls with UXP alternatives

### Phase 2 Status: ~75% Ready
- [x] `PProBridge` wraps UXP Premiere API (project items, markers, sequence info, cuts, selected clips, import)
- [x] `getSelectedClips()` — selection via `Sequence.getSelection()`
- [x] `importFiles()` — import via `Project.importFiles()` with optional bin
- [x] Generated UDT smoke coverage for direct UXP host actions (`ocBatchRenameProjectItems()`, `ocCreateSmartBins()`, marker operations, range export, playhead, and import cleanup)
- [x] Repository-side validator for captured F267 harness results (`validate_uxp_udt_results`)
- [ ] Capture the F267 harness results in Premiere UDT and pass the strict validator before treating the direct-action set as live-verified
- [ ] Residual CEP-only paths: native caption-track creation and QE reflection (documented in F266 above)
- [ ] Full timeline write-back without ExtendScript for advanced trim edge cases (blocked on UXP API maturity)

### Key Files
- `extension/com.opencut.uxp/csinterface-shim.js` — CSInterface→postMessage bridge for WebView
- `extension/com.opencut.uxp/main.js` — Native UXP panel (2000+ lines, full feature parity)
- `extension/com.opencut.uxp/index.html` — UXP panel UI with all tabs including Settings
- `extension/com.opencut.uxp/bolt-webview/` — F252.1 dormant Bolt/WebView scaffold with host API wrappers, WebView message bridge, and least-privilege config template
- `tests/test_uxp_webview_scaffold.py` — Static guardrails for the scaffold contract
- `tests/test_uxp_host_action_dispatch.py` — Static guardrails that keep the F252.2 host-action dispatcher aligned with the CEP/UXP parity catalogue
- `tests/test_uxp_create_subsequence_integration.py` — Static guardrails for the F254 subsequence range handoff and F255 encoder boundary
- `tests/test_uxp_encoder_manager_integration.py` — Static guardrails for the F255 EncoderManager export handoff
- `tests/test_uxp_transcript_api_integration.py` — Static guardrails for the F256 Transcript API helper contract
- `tests/test_uxp_object_mask_api_integration.py` — Static guardrails for the F257 ObjectMaskUtils helper contract
- `tests/test_uxp_aaf_export_integration.py` — Static guardrails for the F258 ProjectConverter AAF export contract
- `opencut/_generated/uxp_migration_dashboard.json` and `extension/com.opencut.uxp/uxp-migration-dashboard.json` — F260 generated migration dashboard artifacts
- `tests/test_uxp_migration_dashboard.py` — Static guardrails for the F260 dashboard generator, bundled panel JSON, and Settings UI surface
- `opencut/_generated/uxp_udt_harness.json` and `extension/com.opencut.uxp/uxp-udt-harness.json` — F267 generated UDT smoke harness artifacts for the 14 direct-UXP actions
- `extension/com.opencut.uxp/udt-smoke.js` — Panel-side UDT runner exposed as `window.OpenCutUXPUdtHarness`; read-only actions run by default, project-changing cases require `includeMutating: true`
- `tests/test_uxp_udt_harness.py` — Static guardrails for the F267 generator, bundled JSON, panel runner, and release-smoke wiring
- `opencut/core/uxp_udt_results.py` and `opencut/tools/validate_uxp_udt_results.py` — F252.3 capture-template and strict result validator for WebView cutover readiness
- `tests/test_uxp_udt_results.py` — Static guardrails for the F252.3 capture validator and release-smoke wiring

## Risk Assessment
- **Low risk:** Backend communication (fetch works natively in UXP)
- **Medium risk:** File system access (UXP requires explicit permissions)
- **High risk:** ExtendScript replacement (UXP Premiere API is still evolving)
- **Mitigated:** WebView UI gives us full DOM when needed
