# OpenCut

## Overview
Free, open-source Adobe Premiere Pro CEP extension — AI-powered silence removal, caption generation, audio processing, voice lab, and visual effects. Runs locally.

## Tech Stack
- CEP panel extension: HTML/JS frontend + ExtendScript backend
- Python/Flask server for heavy processing (AI models, FFmpeg pipelines)
- ExtendScript communicates via `csInterface.evalScript()`
- Single IIFE in main.js (~3200 lines), ES3-compatible (no let/const/arrow functions)

## Key Files
- `extension/com.opencut.panel/client/main.js` — All frontend logic
- `extension/com.opencut.panel/client/index.html` — UI layout (~1990 lines)
- `extension/com.opencut.panel/client/style.css` — Themed styles (~2400 lines)
- `extension/com.opencut.panel/host/index.jsx` — ExtendScript backend (1082 lines)
- `opencut/server.py` — Flask backend (~248KB, 116 API routes)

## Development Setup
- Set `PlayerDebugMode = 1` registry key for unsigned extension development
- Flask server handles AI inference and FFmpeg operations on localhost:5679-5689
- CEP panel communicates with Flask server via localhost HTTP

## Architecture
- **Job system**: `startJob()` -> SSE streaming + polling fallback -> `onJobDone()`
- **Event listeners**: `addJobDoneListener(fn)` — listeners return `true` to handle a job and skip default processing
- **Job chaining**: Pending flags (`pendingBurnin`, `pendingAnimCap`, `pendingTranslate`) auto-chain transcript -> processing
- **Workflow queue**: `runWorkflow(steps)` for arbitrary multi-step job pipelines
- **Step indicator**: `jobStepCurrent`/`jobStepTotal` shown in progress messages during chained jobs
- **Custom dropdowns**: All `<select>` replaced with div-based dropdowns via `initCustomDropdowns()` — supports keyboard nav (arrows, Enter, Escape, type-to-search)
- **Single-job enforcement**: UI lockout via `job-active` class on body
- **Theme system**: 6 dark themes via CSS custom properties
- **localStorage**: Transcript segments cached per file path, recent files list (max 10)

## Conventions
- CEP extensions use HTML/JS for UI, ExtendScript for Premiere Pro automation
- Python/Flask for any processing that ExtendScript can't handle (AI, FFmpeg)
- All processing runs locally - no cloud dependencies
- ES3 syntax only in main.js (CEP Chromium limitation)
- `esc()` for all user-facing server strings (XSS prevention)
- `textContent` over `innerHTML` where possible
- `apiWithSpinner(btn, ...)` for direct API calls with button feedback
- `.text-input` class on all themed text inputs
- `.path-input-row` + `.btn-browse` for file path inputs with browse buttons

## Gotchas
- ExtendScript is ES3 — no let/const, no arrow functions, no template literals
- CEP panels can't use ES modules — everything is in a single IIFE
- `browseForFile()` in ExtendScript returns "null" string on cancel, not actual null
- Transcript chaining: `currentJob` must be null before calling `startJob()` in chain handlers
- MSVC raw string limit: not applicable here but relevant for VaultBox
