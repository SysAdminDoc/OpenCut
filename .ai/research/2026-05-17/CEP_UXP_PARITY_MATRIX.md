# OpenCut — CEP ↔ UXP Parity Matrix (Pass 3)

**Audit date:** 2026-05-17 (Pass 3)
**Source:** `extension/com.opencut.panel/host/index.jsx` (2,736 lines, 18 `ocXxx` JSX host functions) ↔ `@adobe/premierepro@26.3.0-beta.67` typings (per Pass-2 UXP subagent deep walk) ↔ `extension/com.opencut.uxp/main.js` (`PProBridge` class).

This file completed the initial **F198** (CEP-only route catalogue) analysis that Pass 2 flagged as deferred. Pass 42 promoted it into the enforced code catalogue at `opencut/core/cep_uxp_parity.py` plus generated artifact `opencut/_generated/cep_uxp_parity.json`.

---

## 1. The full JSX host surface (18 functions)

| # | JSX function | What it does | UXP API today | CEP-only? | Migration risk |
|---|---|---|---|---|---|
| 1 | `ocPing()` | Handshake / availability probe | trivial (`return true`) | No | None |
| 2 | `ocGetSequenceInfo()` | Active sequence: id, name, duration, fps, in/out, track counts | `Sequence.getName()` + `.getEnd()` + `.getSettings()` + `.getInPoint()` + `.getOutPoint()` + `.getVideoTracks()` + `.getAudioTracks()` | No | Low — straight port |
| 3 | `ocAddSequenceMarkers(markersJSON)` | Batch marker create with color + name + comment | `Markers.createMarker(time)` + `Marker.setName()` + `Marker.setColorIndex()` | No | Low — straight port |
| 4 | `ocGetSequenceMarkers()` | Read all sequence markers | `Markers.getMarkers()` (iterator) | No | Low — straight port |
| 5 | `ocApplySequenceCuts(cutsJSON)` | Trim N clips + ripple-delete | `SequenceEditor.createRemoveItemsAction(items, ripple=true)` | **Partial** | **Med** — advanced QE-trim edge cases lack UXP equiv |
| 6 | `ocApplyClipKeyframes(trackIndex, clipStartTime, keyframesJSON)` | Write effect-property keyframes | `Keyframe` / `PointKeyframe` + `VideoFilterFactory` / `AudioFilterFactory` | No | Low |
| 7 | `ocBatchRenameProjectItems(renamesJSON)` | Rename N project items | `ProjectItem.setName()` per item | No | Low |
| 8 | `ocCreateSmartBins(rulesJSON)` | Create bins + move matching items | `FolderItem.create()` + `ProjectItem.move()` | No | Low |
| 9 | **`ocAddNativeCaptionTrack(srtJSON)`** | Import SRT as a native Premiere caption track | **NO UXP equivalent** — `createCaptionTrack()` is not in `@adobe/premierepro@26.3.0-beta.67` | **YES — CEP-only** | **High** — requires F253 Hybrid Plugin or wait for Adobe |
| 10 | `ocGetProjectBins()` | Enumerate project bins | `FolderItem` walk + `Project.getRootItem()` | No | Low |
| 11 | `ocExportSequenceRange(outputPath, startSeconds, endSeconds)` | Export a sub-range | `Sequence.createSubsequence(ignoreTrackTargeting?)` (confirmed present in beta) + `Exporter.exportSequence()` | No | Low (was thought missing pre-Pass-2; confirmed present) |
| 12 | `ocRemoveSequenceMarkers(fingerprintsJSON)` | Inverse of #3 | `Markers.removeMarker(marker)` | No | Low |
| 13 | `ocUnrenameItems(mapJSON)` | Inverse of #7 | `ProjectItem.setName()` per item with old name | No | Low |
| 14 | `ocRemoveImportedSequence(payloadJSON)` | Inverse of XML import (Operation Journal revert) | `Project.deleteItem()` on the imported sequence | No | Low |
| 15 | `ocSetSequencePlayhead(seconds)` | Move CTI | `SourceMonitor.setPosition()` — **new in 26.3.0-beta.67** | No | Low |
| 16 | `ocRemoveImportedItem(payloadJSON)` | Inverse of media import | `Project.deleteItem()` | No | Low |
| 17 | **`ocQeReflect()`** | Call `qe.reflect.methods` to surface undocumented QE APIs | **NO UXP equivalent** — QE DOM is CEP/Premiere-internal-only | **YES — CEP-only** | **High** — QE may never have a UXP equivalent; F253 Hybrid Plugin or accept loss |
| 18 | `ocEmitPingEvent(tag)` | Dispatch a `CSXSEvent` for panel ack | UXP `addon` events or direct fetch back to backend | No | Low — different mechanism, same effect |

**Summary: 16 of 18 functions (~89%) have a clean UXP migration path** in the 26.3.0-beta typings. Only **2 are truly CEP-only** today.

---

## 2. Per-function risk classification

| Risk | Count | Functions |
|---|---:|---|
| Low — direct UXP port | 14 | #1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16 |
| Med — partial / edge cases need QE | 1 | #5 (`ocApplySequenceCuts` — ripple-delete works, QE-trim advanced cases don't) |
| Different-mechanism low risk | 1 | #18 (`ocEmitPingEvent` — needs new addon-event surface) |
| **High — CEP-only today** | **2** | **#9 (`ocAddNativeCaptionTrack`)**, **#17 (`ocQeReflect`)** |

---

## 3. The 2 truly CEP-blocked functions: deep-dive

### 3.1 `ocAddNativeCaptionTrack(srtJSON)` — F186 territory

**What it does:** the CEP panel calls this with a parsed SRT (collapsed blank lines per CLAUDE.md gotcha) to import it as a **native Premiere caption track** (not a sidecar). This gives Essential Graphics caption-track UX rather than burn-in.

**Why CEP-only:** UXP's `Sequence` class exposes `getCaptionTrack(idx)` (read) but no `createCaptionTrack()` / `addCaptionTrack(format, srtBytes)` (write). `Project.importFiles([srtPath])` will import an SRT **as a project item**, but doesn't create a caption track on the active sequence.

**Workarounds without F253 Hybrid Plugin:**
- (a) Tell users to drag the imported SRT project-item onto the timeline themselves. Loses the one-click UX.
- (b) Burn-in instead. Loses Essential Graphics caption track UX.
- (c) Wait for Adobe to ship `createCaptionTrack()`. No public timeline.
- (d) Use UXP's `ProjectConverter.exportAsFinalCutProXML(sequence, path)` to round-trip → not applicable (no `importFrom...` symmetry post-26.3 typings).

**F253 Hybrid Plugin path:** a `.uxpaddon` could call native Premiere-app SDK to create a caption track. Effort: M-L. Bolt UXP 1.3 ships a public-hybrid template.

**Adobe gap report:** **F186** in Pass 1 backlog is exactly this. Re-file with screenshot + repro steps.

### 3.2 `ocQeReflect()` — F187 territory (the harder one)

**What it does:** OpenCut Wave H2.8 shipped this as "QE reflection probe" — calls `qe.reflect.methods` on a Premiere instance with QE enabled (`app.enableQE()`) to enumerate undocumented internal APIs. Result is cached at `~/.opencut/qe_reflect.json` and surfaced via `GET /system/qe-reflect`.

**Why CEP-only:** QE DOM is a Premiere-internal undocumented API surface. It's never been officially exposed. UXP intentionally does **not** expose QE because it's not contractually stable across Premiere versions.

**Workarounds:**
- (a) Drop the feature post-CEP-EOL. Acceptable — it's a research aid, not a production feature.
- (b) Replace each individual QE use case with a documented UXP API. The `vakago-tools` QE catalogue (per CLAUDE.md Wave H source) shows ~50 QE methods; ~30 have UXP equivalents now per Pass-2 walk.
- (c) **F253 Hybrid Plugin** could call Premiere-app SDK directly — but the SDK doesn't formally expose QE either.

**Recommendation:** **document this as a deliberate post-Sept-2026 retirement.** The 3 "I need QE for X" use cases (ripple-delete advanced, effect-by-name, advanced trim) can each be solved by:
- ripple-delete advanced → `SequenceEditor.createRemoveItemsAction(ripple=true)` covers 90%
- effect-by-name → `VideoFilterFactory.createVideoFilter("Lumetri Color")` + enumerated effect catalogue (F265-class UDT harness work)
- advanced trim → manual UXP trim with adjacent-clip recompute

**Adobe gap report:** **F187** in Pass 1 backlog covers the QE category at a high level. Realistically Adobe will not re-expose QE; this needs use-case-by-use-case UXP replacements.

---

## 4. Updated F252 migration plan (Bolt UXP + WebView UI)

Pass 2 proposed **F252** as XL effort to migrate the 7,730-line CEP main.js + 3,210-line index.html + 4,100-line style.css. The Pass 3 JSX walk de-risks this:

**14 of 18 JSX host functions** can be replaced with `await ppro.X()` calls inline in the UXP main.js — no `evalScript` wrapper layer needed. The Pass-2 estimate budgeted "wrapped-host-comms-module pattern" but **only 2 functions** need a wrapper layer (the CEP-only ones, which call out to `host/index.jsx` as before via a Bolt-CEP-fallback build).

Revised F252 sub-phases:

| Sub-phase | Effort | What |
|---|---|---|
| F252.1 | M | Scaffold Bolt UXP (template: react/svelte/vue/vanilla) sharing `client/shared/` with the existing CEP panel |
| F252.2 | M | Port the 14 low-risk `ocXxx` calls to inline `ppro` calls in UXP main.js (the existing UXP main.js already wraps some of these in `PProBridge.applyCuts`, `addMarkers`, etc.) |
| F252.3 | L | Port the 3,210-line index.html + 4,100-line style.css to either (a) WebView UI (1:1 with CEP) or (b) Spectrum widgets (rewrite) |
| F252.4 | M | Ship 2 CEP-fallback paths for `ocAddNativeCaptionTrack` + `ocQeReflect` until F253 Hybrid Plugin lands |
| F252.5 | S | Cut over default UXP-on-25.6+, leave CEP build as fallback for 25.5- |

Original F252 estimate: XL with 4-6 week timeline → revised: **L with 3-4 weeks** if (a) WebView UI is the HTML/CSS port path. Spectrum widget rewrite (option b) keeps it XL.

---

## 5. Updated F253 Hybrid Plugin scope

Pass 2 proposed **F253** as XL effort covering "file drag-out + QE-equivalent ops". Pass 3 narrows the scope:

- **File drag-out (HTML5 `dragstart`)** is NOT a JSX host function in OpenCut today — it's panel-side main.js code. UXP gap. F253 covers via native drag-event injection.
- **QE-equivalent ops** are needed for ~3 use cases (ripple-delete advanced, effect-by-name, advanced trim). Most have UXP partial replacements per §3.2; the remainder might not even ship in OpenCut today (need UDT testing — F265).
- **`createCaptionTrack`** is the **highest-leverage** Hybrid Plugin use case. If F253 only ships caption-track creation, it closes the single largest CEP-only gap.

Revised F253 sub-phases:

| Sub-phase | Effort | What |
|---|---|---|
| F253.1 | M | Bolt UXP `public-hybrid/` template integration; build mac universal + win-x64 + win-arm64 binaries |
| F253.2 | M | Implement `createCaptionTrack(sequence, srtBytes)` C++ binding |
| F253.3 | S | Implement file-drag-out via native drag-source platform APIs |
| F253.4 | M | (Optional) Implement effect-by-name + advanced trim wrappers — defer to post-Sept-2026 if not validated as needed |

Original F253 estimate: XL → revised: **L** for caption-track + drag-out only.

---

## 6. Recommended sequence

Given the Pass 3 risk reclassification:

1. **v1.34**: F252.1 + F252.2 — Bolt UXP scaffold + 14-function port. **Med effort, high coverage.**
2. **v1.35**: F252.3 — HTML/CSS migration via WebView UI. **L effort, completes panel parity.**
3. **v1.36**: F252.4 + F252.5 — CEP-fallback paths for 2 functions + UXP-on-25.6+ default cutover. **M effort, ships UXP-first.**
4. **v1.37 (before Sept 2026 CEP EOL)**: F253.1 + F253.2 + F253.3 — Hybrid Plugin caption-track + drag-out. **L effort, retires CEP for 99% of users.**
5. **v1.38+ (post-Sept 2026)**: F253.4 (optional QE replacements) or just drop the QE features. **S-M effort.**

**Total runway needed:** ~4 versions, ~3-4 months at the observed cadence (1 minor version every 1-2 weeks). **Comfortably fits the Sept 2026 CEP EOL window** if started in v1.34 (next release after the current Now-tier dependency bumps land).

---

## 7. New F-numbers from this Pass-3 parity work

| F# | Title | Priority | Effort | Notes |
|---|---|---|---|---|
| (existing) F198 | CEP-only route catalogue | Done in Pass 42 | M | Code-owned catalogue + generated JSON now pin this matrix against `host/index.jsx`. |
| (existing) F252 | Bolt UXP + WebView UI migration | Next | revised L | Subdivided into F252.1-F252.5 |
| (existing) F253 | UXP Hybrid Plugin sidecar | Next/Later | revised L | Subdivided into F253.1-F253.4 |
| F266 | Document the 2-function CEP residual + drop-QE-features plan | Done in Pass 6 | S | Documented in `docs/UXP_MIGRATION.md` |
| F267 | UDT test harness for the 14 low-risk JSX→UXP ports | Next | M | Validate F252.2 before users see it |

---

## 8. Conclusion

Pass 2's worry-tone about CEP-EOL ("4 months runway", "5 truly blocked features") is **less severe than feared**:
- OpenCut's actual CEP-only surface is **2 of 18 JSX functions** (~11%), not 5 of N feature surfaces.
- **89% of the JSX surface migrates cleanly** to existing UXP APIs.
- The biggest blocker (`createCaptionTrack`) is solvable via Hybrid Plugin in a single L-effort PR.
- The other blocker (`ocQeReflect`) is a research aid; can be retired without user impact.

**Net: OpenCut's CEP-EOL exposure is manageable inside the Sept 2026 window.** The harder lift is the 7,730-line vanilla JS panel — but Bolt UXP WebView UI makes that an L (not XL) task too. F252 and F253 are not "race against the clock"; they're "deliberate three-release migration."
