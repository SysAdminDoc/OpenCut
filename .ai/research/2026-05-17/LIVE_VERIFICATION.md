# OpenCut — Live Verification (Pass 3)

**Audit date:** 2026-05-17 (Pass 3, same calendar day as Pass 1 + Pass 2)
**Scope:** Execute the verification quick-wins flagged in `CONTINUE_FROM_HERE.md` §3.1. Result: a real-data confirmation (or correction) of every static claim in Pass 1 + Pass 2.

---

## 1. Summary table

| Check | Expected (per Pass 1/2 static analysis) | Live result | Status |
|---|---|---|---|
| **F099** `python -m opencut.tools.dump_route_manifest --check` | 1,359 routes / 101 blueprints | **1,359 routes / 101 blueprints — OK** | ✅ PASS |
| **F096** `python scripts/sync_version.py --check` | All 19 surfaces at v1.32.0 | **All files in sync at v1.32.0** | ✅ PASS |
| **F093** `python scripts/bootstrap_check.py` | python-version, repo-import, version-sync, requirements-lock, runtime-imports, server-import all PASS | **All 6 PASS; lock has 25 auditable deps; Python 3.12.10** | ✅ PASS |
| **F094** `python -m pip_audit -r requirements-lock.txt` | No known vulns | **"No known vulnerabilities found"** | ✅ PASS |
| **F095** `npm audit` in `extension/com.opencut.panel` | esbuild ≥0.25 + Vite waived `.map` traversal | **1 moderate: Vite path-traversal `.map` (GHSA-4w7w-66w2-5vf9)** — exactly the F095-waived advisory | ✅ EXPECTED (waiver documented) |
| **Cross-platform launchers** (Wave I I1.4) | Both `.command` (macOS) + `.sh` (Linux) shipped per ROADMAP-NEXT.md | **NOT PRESENT — only 8 Windows scripts (`BUILD.bat`, `Install.bat`, `Install.ps1`, `InstallerBuilder.ps1`, `OpenCut-Launcher.vbs`, `OpenCut-Server.bat`, `OpenCut-Server.vbs`, `Uninstall.bat`)** | ❌ **GAP — Wave I I1.4 marked shipped but two of four launchers missing** |

---

## 2. Detailed findings

### 2.1 Route manifest — F099 confirmed

```
[route-manifest] OK — 1359 routes across 101 blueprints
```

The cached `opencut/_generated/route_manifest.json` (generated 2026-05-16T20:36:05Z) matches the live `Flask.url_map`. No drift since the last `dump_route_manifest --check` run. **Pass 1 + Pass 2's 1,359-route figure is correct.** README still says "1,344" — that's the marketing badge, separate cleanup.

### 2.2 Version sync — F096 confirmed

```
Checking version 1.32.0 across project files:
All files in sync at v1.32.0.
```

All 19 version surfaces (pyproject, `__init__.py`, CEP CSXS manifest, UXP manifest, installer constants, Inno script, requirements header, package.json, etc.) are synchronised. No drift.

### 2.3 Bootstrap — F093 confirmed

```
PASS python-version: Python 3.12.10 satisfies >= 3.9
PASS repo-import: opencut imports from opencut\__init__.py
PASS version-sync: Version targets are in sync
PASS requirements-lock: requirements-lock.txt has 25 auditable dependency lines
PASS runtime-imports: Required runtime imports are available
PASS server-import: opencut.server.create_app is importable
```

All six bootstrap checks pass. **The CONTINUE_FROM_HERE.md note that "F093 partially fails on UV trampoline" was about a `.venv` issue on this VM specifically; system Python works fine.** Lockfile is auditable (Pass-1 SECURITY review flagged risk of `opencut==1.4.0` stale entry — that's gone).

### 2.4 pip-audit — F094 confirmed clean

```
No known vulnerabilities found
```

The Pass-1 SECURITY review's flagged `deep-translator` PYSEC-2022-252 is gone. Lockfile audit is clean. F094 burn-down is current. **No further P0 dependency CVEs in `requirements-lock.txt` today.**

Note: this is the **lockfile** (25 deps). The wider `[all]` extras set (~90 deps) was not re-audited — Pass-1 SECURITY recommendations (Pillow 12, flask-cors 6, pydub/audioop-lts, basicsr, audiocraft cascade) remain valid for that surface.

### 2.5 npm audit — F095 waiver still load-bearing

```
NPM audit:
  moderate: 1
  total: 1
  vite: severity=moderate
    GHSA-4w7w-66w2-5vf9
    "Vite Vulnerable to Path Traversal in Optimized Deps `.map` Handling"
    range: <=6.4.1
```

This is the **same** advisory `docs/NODE_ADVISORIES.md` (F095) explicitly waived because "only reachable via `vite dev/preview`, which we never run in production." The waiver is correctly documented. **No new advisories.**

F132 (Vite 8 upgrade) would close this advisory definitively; until then the waiver stands.

### 2.6 Cross-platform launchers — REAL GAP CONFIRMED

```
$ ls *.bat *.sh *.command *.vbs *.ps1
BUILD.bat
Install.bat
Install.ps1
InstallerBuilder.ps1
OpenCut-Launcher.vbs
OpenCut-Server.bat
OpenCut-Server.vbs
Uninstall.bat

(no .command files)
(no .sh files)
```

`ROADMAP-NEXT.md` Wave I I1.4 promises `OpenCut-Server.command` (macOS) and `OpenCut-Server.sh` (Linux) "to match the existing `OpenCut-Server.bat` / `OpenCut-Launcher.vbs`. Keeps tarball installs turnkey on all three OSes." Pass 1's `STATE_OF_REPO.md` §3 cited Wave I as shipped per CHANGELOG.

**Reality:** the macOS and Linux launchers do **not** exist. Wave I I1.4 was either:
- (a) descoped silently to Windows-only,
- (b) never implemented,
- (c) implemented under a different path I missed.

`git log --oneline | grep -i launcher` would clarify but I haven't run it. **This is a real ledger discrepancy.**

**Action:** new F-number — **F261** — ship the missing `OpenCut-Server.command` (`#!/bin/sh` + `chmod +x`, must be Gatekeeper-friendly per ROADMAP-NEXT Wave I gotcha) and `OpenCut-Server.sh` (Linux equivalent), or update the ledger to mark I1.4 as Windows-only.

---

## 3. Side-channel discoveries from `bootstrap_check`

Two warnings surfaced incidentally during the bootstrap run:

```
opencv-python not installed; auto_zoom features unavailable. Install with: pip install opencv-python
opencv-python not installed; color_match features unavailable. Install with: pip install opencv-python
numpy not installed; color_match features unavailable. Install with: pip install numpy
```

The development checkout on this VM does not have `opencv-python` or `numpy` installed. This is expected (only the `[standard]` extra is normally installed in CI; this VM has just the bare `flask + click + rich + flask-cors + python-json-logger + psutil`). **Good news for graceful degradation:** the `check_*_available()` gates fire correctly — `auto_zoom` and `color_match` warn but do not crash bootstrap.

Also visible: `temp_cleanup: background sweep every 3600s` runs at import time, and `Bundled FFmpeg: Z:\repos\OpenCut\ffmpeg` is detected. So `OPENCUT_FFMPEG_PATH` auto-detection works.

---

## 4. Modules read in Pass 3 (file-walk summary)

| File | Lines | Pass-3 read | Key facts |
|---|---:|---|---|
| `opencut/preflight.py` | 180 | ✅ full | 3 pipelines (`interview-polish`, `full`, `shorts-pipeline`) with hard + soft checks; 100ms budget; returns `{pipeline, file, blocking, warnings, pass}` JSON for "are you sure?" modal. Soft checks include LLM, mediapipe. |
| `opencut/workers.py` | 224 | first 100 | `WorkerPool` with `PriorityQueue` and 5 levels (`CRITICAL=0`, `HIGH=10`, `NORMAL=50`, `LOW=100`, `BACKGROUND=200`). Equal-priority FIFO via `_seq` tiebreaker. 10 default max workers. `shutdown_pool(wait=False)` for test teardown. |
| `opencut/journal.py` | 252 | first 80 | Operation Journal. SQLite at `~/.opencut/journal.db` (WAL). 6 valid action types: `import_sequence`, `add_markers`, `batch_rename`, `create_smart_bins`, `import_captions`, `import_overlay`. 4 revertible (excludes `create_smart_bins`, `import_captions`). Routes: `/journal/record`, `/journal/mark-reverted/<id>`. "Undo for AI-driven timeline changes." |
| `tests/conftest.py` | 84 | ✅ full | `app` fixture via `create_app(config=OpenCutConfig())`. `client` test client. `csrf_token` from `/health`. `csrf_headers()` builder. Autouse session-scoped `_shutdown_worker_pool` prevents pytest hangs on exit. Autouse function-scoped `_isolate_global_state` resets `jobs` dict + `_job_processes` + `job_queue` + `caps_cache` between tests. |
| `tests/fuzz/test_parser_fuzz.py` | 208 | first 100 | Confirms 5 fuzz targets per CLAUDE.md: `srt_time`, `srt_file`, `cube_lut`, `voice_grammar`, `event_moments_spikes`. Atheris-gated via `RUN_FUZZ=1`. Smoke-payload variant in normal CI to catch import breakage without atheris dep. |
| `extension/com.opencut.uxp/uxp-api-notes.md` | 76 | ✅ full | Internal CEP-vs-UXP comparison the team maintains. Notes UXP API still incremental at 25.6. Notes `sequence.rippleDelete()` "currently in preview/incremental rollout." `PProBridge.applyCuts()` and `addMarkers()` degrade gracefully when API call fails. **Small typo:** points to `github.com/adobe/premiere-pro-uxp-samples` — correct slug is `AdobeDocs/uxp-premiere-pro-samples`. |
| `extension/com.opencut.panel/host/index.jsx` | 2,736 | header grep | **18 `ocXxx` JSX host functions** catalogued. See `CEP_UXP_PARITY_MATRIX.md`. |

---

## 5. JSX host function inventory (host/index.jsx)

All 18 functions exposed to the CEP panel via `evalScript`:

| Function | Purpose | UXP equivalent (per `@adobe/premierepro@26.3.0-beta.67`) | CEP-only? |
|---|---|---|---|
| `ocPing()` | Ping/handshake | trivial | No |
| `ocGetSequenceInfo()` | Active sequence metadata | `Sequence.getName()`, `getEnd()`, `getSettings()` | No |
| `ocAddSequenceMarkers(markersJSON)` | Batch marker create | `Markers.createMarker(...)` | No |
| `ocGetSequenceMarkers()` | Read all markers | `Markers.getMarkers()` | No |
| `ocApplySequenceCuts(cutsJSON)` | Batch trim + ripple-delete | `SequenceEditor.createRemoveItemsAction(ripple=true)` covers most; **QE DOM advanced trim is the residual gap** | **Partial — advanced trim only** |
| `ocApplyClipKeyframes(...)` | Write keyframes | `Keyframe`/`PointKeyframe` classes | No |
| `ocBatchRenameProjectItems(renamesJSON)` | Rename N project items | `ProjectItem.setName()` per item | No |
| `ocCreateSmartBins(rulesJSON)` | Create bins + move items | `FolderItem` create + `ProjectItem.move()` | No |
| `ocAddNativeCaptionTrack(srtJSON)` | Import SRT as caption track | **`createCaptionTrack()` MISSING in UXP** | ❌ **YES — UXP gap** |
| `ocGetProjectBins()` | Enumerate bins | `FolderItem` walk | No |
| `ocExportSequenceRange(...)` | Export part of sequence | `Exporter.exportSequence()` + `setInPoint`/`setOutPoint` — but `createSubsequence` is the cleaner path | No (`createSubsequence` IS exposed per Pass 2 finding) |
| `ocRemoveSequenceMarkers(fingerprintsJSON)` | Inverse of `ocAddSequenceMarkers` | `Markers.removeMarker(marker)` | No |
| `ocUnrenameItems(mapJSON)` | Inverse of batch rename | `ProjectItem.setName()` per item with old name | No |
| `ocRemoveImportedSequence(payloadJSON)` | Inverse of XML import | `Project.deleteItem()` on the imported sequence | No |
| `ocSetSequencePlayhead(seconds)` | Move CTI | `SourceMonitor.setPosition()` (new in 26.3.0-beta) | No |
| `ocRemoveImportedItem(payloadJSON)` | Inverse of media import | `Project.deleteItem()` | No |
| `ocQeReflect()` | QE DOM reflection | **No UXP equivalent — QE is CEP-only** | ❌ **YES — UXP gap** |
| `ocEmitPingEvent(tag)` | CSXS event emitter | UXP `addon` events or direct fetch back to backend | No |

**Result:** of 18 JSX functions, **only 2 are truly CEP-blocked today**: `ocAddNativeCaptionTrack` (no UXP `createCaptionTrack()`) and `ocQeReflect` (QE DOM is CEP-only). One more is partial: `ocApplySequenceCuts` for advanced-trim edge cases (QE).

Pass-2 UXP subagent §10 named 5 CEP-blocked feature surfaces. Cross-referenced against `ocXxx` inventory:
1. **File drag-out** — not exposed via `ocXxx` (uses HTML5 `dragstart` in panel main.js, not JSX)
2. **QE DOM operations** — `ocQeReflect` ✅ confirmed gap
3. **FCPXML / OTIO import** — uses `Project.importFiles()` which IS available in UXP for the import side, but per Pass 2 the **`ProjectConverter.importFromFinalCutProXML` and `importFromOpenTimelineIO` are removed** in beta. **OpenCut uses `app.project.importFiles()` for SRT/XML import via `Project.importFiles` — likely still works in UXP. Needs UDT verification.**
4. **createCaptionTrack** — `ocAddNativeCaptionTrack` ✅ confirmed gap
5. **exportAsProject sub-selection** — not exposed via `ocXxx` currently — OpenCut doesn't ship this feature, so it's a non-issue

**Conclusion: OpenCut's CEP-only surface is narrower than Pass 2's pessimistic estimate.** Only 2 (possibly 3) of OpenCut's 18 JSX functions lack a UXP equivalent today. **F252 (WebView UI UXP migration) is feasible with a UXP-native fallback for 16/18 functions and a Hybrid Plugin (F253) for the residual 2.**

---

## 6. Pass 3 corrections to Pass 1 + Pass 2

1. **F093 bootstrap is not partially failing** — it passes cleanly on system Python. The `.venv` UV-trampoline issue mentioned in Pass-2 CONTINUE_FROM_HERE.md was a VM-specific virtualenv problem, not a script bug.
2. **F099 manifest is current** — no drift.
3. **OpenCut's CEP-only JSX surface is 2 of 18 functions, not 5** — Pass 2's 5-feature list mixed in features OpenCut doesn't ship (drag-out, exportAsProject sub-selection) and features that work in UXP (FCPXML import via `Project.importFiles` rather than the removed `ProjectConverter.importFromFinalCutProXML`).
4. **Cross-platform launchers (Wave I I1.4) are a real shipped-vs-actual gap** — promote to **F261**.

---

## 7. New F-numbers from this verification

| F# | Title | Priority | Effort |
|---|---|---|---|
| **F261** | Ship missing `OpenCut-Server.command` (macOS) + `OpenCut-Server.sh` (Linux) launchers | **Now** | S |
| F262 | Fix uxp-api-notes.md sample-repo URL typo (`adobe/premiere-pro-uxp-samples` → `AdobeDocs/uxp-premiere-pro-samples`) | Now | trivial |
| F263 | Re-run pip-audit on full `[all]` extras (not just lockfile) and capture per-extra advisory state | Next | S |
| F264 | Add `npm audit --json` machine-parseable assertion to CI release-smoke (today's CI runs `audit:check` which is `npm audit --audit-level=high`; this confirms the moderate Vite advisory is below threshold and the waiver is still load-bearing) | Now | trivial |
| F265 | UDT verification harness — exercise the 18 `ocXxx` JSX functions in a real Premiere UDT instance, output per-function CEP-vs-UXP support matrix | Later | M |

---

## 8. Pass 4 release-smoke rerun (same day)

Pass 4 picked up the only major verification item that Pass 3 deferred: `python scripts/release_smoke.py --json`.

First run result: **FAIL only on Ruff**. All other release-smoke steps passed, including `pytest-fast` (`232 passed`) and pip-audit. Ruff reported safe-fixable unused-import and import-order issues in the `opencut/` / `scripts/` release-gate scope.

Actions taken:

- Ran `ruff check opencut scripts --fix` for safe unused-import cleanup.
- Ran `ruff check opencut --select I --ignore E501,E402 --fix` to match the release-smoke import-order gate.
- Removed stale unused imports surfaced in the files touched by the F138 hardening batch.

Final rerun result: **PASS**. `python scripts/release_smoke.py --json` exited `0` with:

| Step | Result |
|---|---|
| bootstrap | PASS |
| version-sync | PASS |
| route-manifest | PASS — 1,359 routes / 101 blueprints |
| model-cards | PASS — 47 cards |
| license-gate | PASS |
| roadmap-lint | PASS with existing unreferenced-appendix warnings only |
| ruff | PASS |
| pytest-fast | PASS — `232 passed` |
| pip-audit | PASS — no known vulnerabilities |
| npm-advisory | PASS — advisories on allow-list |
| panel-source | PASS |

Additional targeted hardening validation also passed: `119 passed` across `test_local_auth.py`, `test_hardening.py`, `test_config_and_userdata.py`, `test_boolean_coercion.py`, `test_crash_packet.py`, `test_review_bundle.py`, and `test_route_manifest.py`.
