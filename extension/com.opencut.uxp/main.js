/**
 * OpenCut UXP Panel — main.js
 * Requires Premiere Pro 25.6+ (UXP GA)
 *
 * Key differences from the CEP panel:
 *  - Uses UXP APIs instead of ExtendScript (async/await throughout)
 *  - fetch() works natively — no CSInterface bridge needed
 *  - Direct Premiere Pro access via the `premierepro` UXP module (when available)
 *  - ES module; import/export syntax is supported in UXP
 *
 * Architecture:
 *  - BackendClient  — wraps all HTTP calls to the Python backend
 *  - PProBridge     — wraps UXP premierepro module with graceful degradation
 *  - JobPoller      — handles async job polling + progress
 *  - UIController   — tab switching, toast, processing banner, sliders
 *  - Feature fns    — one async function per feature
 */

// ─────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────
const BACKEND_DEFAULT  = "http://127.0.0.1:5679";
const BACKEND_MAX_PORT = 5689;
const POLL_INTERVAL_MS = 1200;
const HEALTH_CHECK_MS  = 8000;
const HEALTH_MAX_MS    = 60000;
const MEDIA_SCAN_MS    = 30000;
const SSE_AVAILABLE    = typeof EventSource !== "undefined";
const VERSION          = "1.9.19";

async function detectBackend() {
  // Try ports 5679-5689 like CEP panel does
  for (let port = 5679; port <= BACKEND_MAX_PORT; port++) {
    const url = `http://127.0.0.1:${port}`;
    try {
      // AbortSignal.timeout() may not exist in older UXP runtimes
      const opts = {};
      if (typeof AbortSignal !== "undefined" && AbortSignal.timeout) {
        opts.signal = AbortSignal.timeout(500);
      } else {
        const ac = new AbortController();
        setTimeout(() => ac.abort(), 500);
        opts.signal = ac.signal;
      }
      const resp = await fetch(`${url}/health`, opts);
      if (resp.ok) return url;
    } catch (e) { /* try next port */ }
  }
  return BACKEND_DEFAULT;
}

let BACKEND = BACKEND_DEFAULT;

// ─────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────
let csrfToken     = null;
let activeJobId   = null;
let elapsedTimer  = null;
let elapsedSec    = 0;
let lastCuts      = [];     // cuts array from last silence/filler run
let lastMarkers   = [];     // marker array from last beat detection

// ---- SSE / health-check state ----
let _activeSSE       = null;  // current EventSource instance
let _healthBackoff   = HEALTH_CHECK_MS;
let _mediaScanTimer  = null;

// ---- Premiere Pro state cache (reduces UXP API round-trips) ----
const _pproCache = { seq: null, ts: 0 };
const PPRO_CACHE_TTL = 8000; // 8 seconds

// ─────────────────────────────────────────────────────────────
// PProBridge — gracefully degrades when UXP module unavailable
// ─────────────────────────────────────────────────────────────
const PProBridge = (() => {
  let ppro = null;
  let available = false;

  async function init() {
    try {
      // The official UXP module name for Premiere Pro (25.6+).
      // Module name may still be subject to Adobe revision — check
      // developer.adobe.com/premiere-pro/uxp/ for latest.
      const mod = await import("premierepro");
      ppro = mod;
      available = true;
      console.log("[PProBridge] premierepro module loaded successfully.");
    } catch (e) {
      available = false;
      console.warn("[PProBridge] premierepro module not available:", e.message);
    }
  }

  /**
   * Returns active sequence object or null.
   */
  async function getActiveSequence() {
    if (!available || !ppro) return null;
    try {
      const proj = await ppro.app.getProjectList();
      if (!proj || proj.length === 0) return null;
      return await proj[0].getActiveSequence();
    } catch (e) {
      console.warn("[PProBridge] getActiveSequence failed:", e.message);
      return null;
    }
  }

  /**
   * Returns basic info about the active sequence as a plain object.
   * Results are cached for PPRO_CACHE_TTL ms to reduce UXP API round-trips.
   */
  async function getSequenceInfo() {
    // Return cached data if still fresh
    if (_pproCache.seq && (Date.now() - _pproCache.ts < PPRO_CACHE_TTL)) {
      return _pproCache.seq;
    }
    const seq = await getActiveSequence();
    if (!seq) return null;
    try {
      const settings = await seq.getSettings();
      const info = {
        name:        await seq.getName(),
        duration:    await seq.getEnd(),
        framerate:   settings ? settings.videoFrameRate : "unknown",
        width:       settings ? settings.videoFrameWidth : "unknown",
        height:      settings ? settings.videoFrameHeight : "unknown",
        audioTracks: (await seq.getAudioTrackList())?.length ?? "unknown",
        videoTracks: (await seq.getVideoTrackList())?.length ?? "unknown",
      };
      _pproCache.seq = info;
      _pproCache.ts = Date.now();
      return info;
    } catch (e) {
      console.warn("[PProBridge] getSequenceInfo failed:", e.message);
      return null;
    }
  }

  /**
   * Invalidates the sequence info cache, forcing a fresh UXP API call next time.
   */
  function invalidateCache() {
    _pproCache.seq = null;
  }

  /**
   * Adds an array of sequence markers.
   * @param {Array<{time: number, label: string, color: string}>} markers
   */
  async function addMarkers(markers) {
    const seq = await getActiveSequence();
    if (!seq) {
      return { ok: false, reason: "No active sequence or UXP API unavailable." };
    }
    try {
      const markerList = await seq.getMarkerList();
      for (const m of markers) {
        await markerList.createMarker(m.time);
        // Marker label/color API may vary — wrap each in try/catch
        try {
          const created = markerList.getFirstMarkerAtTime(m.time);
          if (created) {
            await created.setName(m.label || "Beat");
            if (m.color) await created.setColorIndex(colorNameToIndex(m.color));
          }
        } catch (_) { /* color/name not critical */ }
      }
      return { ok: true, count: markers.length };
    } catch (e) {
      console.warn("[PProBridge] addMarkers failed:", e.message);
      return { ok: false, reason: e.message };
    }
  }

  /**
   * Applies silence-removal cuts by ripple-deleting regions on V1/A1.
   * @param {Array<{start: number, end: number}>} cuts  — times in seconds
   */
  async function applyCuts(cuts) {
    const seq = await getActiveSequence();
    if (!seq) {
      return { ok: false, reason: "No active sequence or UXP API unavailable." };
    }
    try {
      // Sort descending so removal doesn't shift earlier cut points
      const sorted = [...cuts].sort((a, b) => b.start - a.start);
      for (const cut of sorted) {
        // UXP uses ticks internally; 254016000000 ticks/sec in Premiere
        const startTick = Math.round(cut.start * 254016000000);
        const endTick   = Math.round(cut.end   * 254016000000);
        await seq.rippleDelete(startTick, endTick);
      }
      return { ok: true, applied: sorted.length };
    } catch (e) {
      console.warn("[PProBridge] applyCuts failed:", e.message);
      return { ok: false, reason: e.message };
    }
  }

  function colorNameToIndex(name) {
    const map = { green: 1, red: 2, blue: 4, yellow: 3, purple: 6, cyan: 7, pink: 8 };
    return map[name.toLowerCase()] ?? 1;
  }

  /**
   * Returns all media items in the open project as an array of {name, path, duration, type}.
   * Falls back to empty array if the UXP module is unavailable.
   */
  async function getProjectItems() {
    if (!available || !ppro) return [];
    try {
      // Timeout the entire walk to prevent hanging if UXP API stalls
      const result = await Promise.race([
        (async () => {
          const projList = await ppro.app.getProjectList();
          if (!projList || projList.length === 0) return [];
          const proj = projList[0];
          const rootItem = await proj.getRootItem();
          if (!rootItem) return [];
          return await _walkItems(rootItem, 0);
        })(),
        new Promise(resolve => setTimeout(() => resolve([]), 5000))
      ]);
      return result;
    } catch (e) {
      console.warn("[PProBridge] getProjectItems failed:", e.message);
      return [];
    }
  }

  async function _walkItems(parent, depth) {
    if (depth > 20) return [];
    const items = [];
    try {
      const children = await parent.getItems();
      if (!children) return [];
      for (const child of children) {
        try {
          const isFolder = await child.isFolder?.() ?? false;
          if (isFolder) {
            const subItems = await _walkItems(child, depth + 1);
            items.push(...subItems);
          } else {
            const mediaPath = await child.getMediaPath?.() ?? "";
            if (mediaPath) {
              const name = await child.getName?.() ?? "";
              const duration = await child.getOutPoint?.()?.seconds ?? 0;
              items.push({ name, path: mediaPath, duration });
            }
          }
        } catch (_) { /* skip inaccessible items */ }
      }
    } catch (_) { /* parent has no children */ }
    return items;
  }

  /**
   * Returns the selected clips in the active sequence.
   * @returns {Promise<Array<{name: string, start: number, end: number, trackIndex: number}>>}
   */
  async function getSelectedClips() {
    const seq = await getActiveSequence();
    if (!seq) return [];
    try {
      const selection = await seq.getSelection();
      if (!selection || selection.length === 0) return [];
      const clips = [];
      for (const item of selection) {
        try {
          clips.push({
            name: await item.getName?.() ?? "",
            start: (await item.getInPoint?.())?.seconds ?? 0,
            end: (await item.getOutPoint?.())?.seconds ?? 0,
            trackIndex: await item.getTrackIndex?.() ?? 0,
          });
        } catch (_) { /* skip inaccessible */ }
      }
      return clips;
    } catch (e) {
      console.warn("[PProBridge] getSelectedClips failed:", e.message);
      return [];
    }
  }

  /**
   * Import files into the project media pool.
   * @param {string[]} filePaths — array of absolute file paths
   * @param {string} [binName] — optional target bin name
   * @returns {Promise<{ok: boolean, imported?: number, reason?: string}>}
   */
  async function importFiles(filePaths, binName) {
    if (!available || !ppro) return { ok: false, reason: "UXP API unavailable" };
    try {
      const projList = await ppro.app.getProjectList();
      if (!projList || projList.length === 0) return { ok: false, reason: "No open project" };
      const proj = projList[0];

      // If binName specified, find or create it
      if (binName) {
        try {
          const root = await proj.getRootItem();
          const children = await root.getItems();
          let targetBin = null;
          for (const child of (children || [])) {
            if ((await child.isFolder?.()) && (await child.getName?.()) === binName) {
              targetBin = child;
              break;
            }
          }
          if (!targetBin) {
            targetBin = await proj.createBin(binName);
          }
        } catch (_) { /* bin creation is best-effort */ }
      }

      const imported = await proj.importFiles(filePaths);
      return { ok: true, imported: imported ? filePaths.length : 0 };
    } catch (e) {
      console.warn("[PProBridge] importFiles failed:", e.message);
      return { ok: false, reason: e.message };
    }
  }

  return { init, available: () => available, getActiveSequence, getSequenceInfo, addMarkers, applyCuts, invalidateCache, getProjectItems, getSelectedClips, importFiles };
})();

// ─────────────────────────────────────────────────────────────
// BackendClient — all HTTP communication with Python server
// ─────────────────────────────────────────────────────────────
const BackendClient = (() => {
  /**
   * Core fetch wrapper with CSRF token handling and error normalisation.
   * @param {"GET"|"POST"|"PUT"|"DELETE"} method
   * @param {string} endpoint  — e.g. "/silence"
   * @param {object|null} body — JSON body for POST/PUT
   * @returns {Promise<{ok: boolean, data?: any, error?: string, status?: number}>}
   */
  async function call(method, endpoint, body = null) {
    const url = BACKEND + endpoint;
    const headers = { "Content-Type": "application/json" };
    if (csrfToken) headers["X-OpenCut-Token"] = csrfToken;

    const opts = { method, headers };
    if (body && method !== "GET") opts.body = JSON.stringify(body);

    try {
      const resp = await fetch(url, opts);

      // Refresh CSRF token if provided in response headers
      const newToken = resp.headers.get("X-OpenCut-Token");
      if (newToken) csrfToken = newToken;

      let data;
      const ct = resp.headers.get("Content-Type") || "";
      if (ct.includes("application/json")) {
        data = await resp.json();
      } else {
        data = await resp.text();
      }

      if (!resp.ok) {
        const msg = (data && data.error) ? data.error : `HTTP ${resp.status}`;
        return { ok: false, error: msg, status: resp.status, data };
      }
      return { ok: true, data, status: resp.status };
    } catch (err) {
      return { ok: false, error: err.message ?? "Network error" };
    }
  }

  async function get(endpoint)             { return call("GET",    endpoint); }
  async function post(endpoint, body)      { return call("POST",   endpoint, body); }
  async function del(endpoint)             { return call("DELETE", endpoint); }

  /**
   * Ping /health endpoint.
   * @returns {Promise<boolean>}
   */
  let _capabilities = {};

  async function checkHealth() {
    const r = await get("/health");
    if (r.ok && r.data?.capabilities) {
      _capabilities = r.data.capabilities;
      _updateCapabilityHints();
    }
    return r.ok;
  }

  function getCapabilities() { return _capabilities; }

  function _updateCapabilityHints() {
    // Show/hide install hints based on backend capabilities
    const hints = {
      depthHintUxp: _capabilities.depth_effects !== false,
    };
    for (const [id, available] of Object.entries(hints)) {
      const el = document.getElementById(id);
      if (el) el.style.display = available ? "none" : "block";
    }
  }

  /**
   * Fetch CSRF token from /csrf or /api/csrf.
   */
  async function fetchCsrf() {
    const r = await get("/health");
    if (r.ok && r.data && r.data.csrf_token) {
      csrfToken = r.data.csrf_token;
    }
  }

  return { call, get, post, del: del, checkHealth, fetchCsrf, getCapabilities };
})();

// ─────────────────────────────────────────────────────────────
// JobPoller — submits job and polls until done
// ─────────────────────────────────────────────────────────────
const JobPoller = (() => {
  /**
   * Start a backend job and poll until completion.
   * @param {string}   endpoint     — POST endpoint to start the job
   * @param {object}   body         — request body
   * @param {Function} onProgress   — (pct: number, msg: string) => void
   * @param {Function} onComplete   — (result: object) => void
   * @param {Function} onError      — (msg: string) => void
   */
  async function start(endpoint, body, onProgress, onComplete, onError) {
    const r = await BackendClient.post(endpoint, body);
    if (!r.ok) {
      onError(r.error ?? "Failed to start job");
      return;
    }

    const jobId = r.data?.job_id ?? r.data?.id ?? null;
    if (!jobId) {
      // Synchronous response — job completed inline
      onProgress(100, "Done");
      onComplete(r.data);
      return;
    }

    activeJobId = jobId;

    // Prefer SSE for real-time progress; fall back to polling
    if (SSE_AVAILABLE) {
      trackJobSSE(jobId, onProgress, onComplete, onError);
    } else {
      pollJob(jobId, onProgress, onComplete, onError);
    }
  }

  /**
   * Track job progress via Server-Sent Events (SSE).
   * Falls back to polling on connection error.
   */
  function trackJobSSE(jobId, onProgress, onComplete, onError) {
    if (_activeSSE) { _activeSSE.close(); _activeSSE = null; }

    const es = new EventSource(`${BACKEND}/stream/${jobId}`);
    _activeSSE = es;

    es.onmessage = (event) => {
      try {
        const job = JSON.parse(event.data);
        const status = job.status ?? "running";
        const pct    = typeof job.progress === "number" ? job.progress : 0;
        const msg    = job.message ?? job.msg ?? "Processing...";

        onProgress(pct, msg);

        if (status === "done" || status === "complete" || status === "success") {
          es.close();
          _activeSSE = null;
          activeJobId = null;
          onComplete(job.result ?? job);
          _fireCompletionHooks();
        } else if (status === "error" || status === "failed" || status === "cancelled") {
          es.close();
          _activeSSE = null;
          activeJobId = null;
          onError(job.error ?? job.message ?? "Job failed");
          _fireCompletionHooks();
        }
      } catch (_) { /* ignore parse errors in SSE stream */ }
    };

    es.onerror = () => {
      if (!_activeSSE) return;
      es.close();
      _activeSSE = null;
      // Fallback to polling on SSE failure
      pollJob(jobId, onProgress, onComplete, onError);
    };
  }

  async function pollJob(jobId, onProgress, onComplete, onError) {
    const r = await BackendClient.get(`/status/${jobId}`);
    if (!r.ok) {
      onError(r.error ?? "Polling error");
      activeJobId = null;
      return;
    }

    const job = r.data;
    const status  = job.status ?? "running";
    const pct     = typeof job.progress === "number" ? job.progress : 0;
    const msg     = job.message ?? job.msg ?? "Processing...";

    onProgress(pct, msg);

    if (status === "done" || status === "complete" || status === "success") {
      activeJobId = null;
      onComplete(job.result ?? job);
      _fireCompletionHooks();
      return;
    }

    if (status === "error" || status === "failed" || status === "cancelled") {
      activeJobId = null;
      onError(job.error ?? job.message ?? "Job failed");
      _fireCompletionHooks();
      return;
    }

    // Still running — schedule next poll
    setTimeout(() => {
      if (activeJobId === jobId) {
        pollJob(jobId, onProgress, onComplete, onError);
      }
    }, POLL_INTERVAL_MS);
  }

  // Post-completion hooks — called after every successful or failed job
  const _completionHooks = [];

  function onJobFinished(hook) { _completionHooks.push(hook); }

  function _fireCompletionHooks() {
    for (const hook of _completionHooks) {
      try { hook(); } catch (_) {}
    }
  }

  async function cancel() {
    if (!activeJobId) return;
    // Close SSE stream first to prevent stale events after cancel
    if (_activeSSE) { _activeSSE.close(); _activeSSE = null; }
    await BackendClient.post(`/cancel/${activeJobId}`, {});
    activeJobId = null;
    _fireCompletionHooks();
  }

  /**
   * Poll an already-started job by ID until completion.
   * Returns the final result object or throws.
   */
  function poll(jobId) {
    return new Promise((resolve, reject) => {
      activeJobId = jobId;
      const onProgress = () => {};
      const onComplete = (result) => resolve(result);
      const onError = (msg) => reject(new Error(msg));
      if (SSE_AVAILABLE) {
        trackJobSSE(jobId, onProgress, onComplete, onError);
      } else {
        pollJob(jobId, onProgress, onComplete, onError);
      }
    });
  }

  return { start, poll, cancel, onJobFinished };
})();

// ─────────────────────────────────────────────────────────────
// UIController — DOM manipulation, tabs, toasts, sliders
// ─────────────────────────────────────────────────────────────
const UIController = (() => {
  // ── Tab switching ──
  function switchTab(tabId) {
    // Invalidate Premiere state cache on tab switch
    PProBridge.invalidateCache();
    document.querySelectorAll(".oc-tab").forEach(btn => {
      const active = btn.dataset.tab === tabId;
      btn.classList.toggle("active", active);
      btn.setAttribute("aria-selected", active ? "true" : "false");
    });
    document.querySelectorAll(".oc-tab-panel").forEach(panel => {
      panel.classList.toggle("active", panel.id === `tab-${tabId}`);
    });
    setStatus(`${tabId.charAt(0).toUpperCase() + tabId.slice(1)} tab`);
  }

  // ── Processing banner ──
  function showProcessing(msg = "Processing...") {
    const banner = document.getElementById("processingBanner");
    if (banner) banner.classList.remove("hidden");
    setProcessingMsg(msg);
    setProgress(0);
    startElapsedTimer();
  }

  function hideProcessing() {
    const banner = document.getElementById("processingBanner");
    if (banner) banner.classList.add("hidden");
    stopElapsedTimer();
  }

  function setProcessingMsg(msg) {
    const el = document.getElementById("processingMsg");
    if (el) el.textContent = msg;
  }

  function setProgress(pct) {
    const fill = document.getElementById("progressFill");
    if (fill) {
      const clamped = Math.min(100, Math.max(0, pct));
      fill.style.width = `${clamped}%`;
      fill.setAttribute("aria-valuenow", String(Math.round(clamped)));
    }
  }

  function startElapsedTimer() {
    stopElapsedTimer();
    elapsedSec = 0;
    updateElapsedDisplay();
    elapsedTimer = setInterval(() => {
      elapsedSec++;
      updateElapsedDisplay();
    }, 1000);
  }

  function stopElapsedTimer() {
    if (elapsedTimer) { clearInterval(elapsedTimer); elapsedTimer = null; }
    elapsedSec = 0;
    updateElapsedDisplay();
  }

  function updateElapsedDisplay() {
    const el = document.getElementById("processingElapsed");
    if (!el) return;
    const m = Math.floor(elapsedSec / 60);
    const s = elapsedSec % 60;
    el.textContent = m > 0 ? `${m}m ${s}s` : `${s}s`;
  }

  // ── Status bar ──
  function setStatus(msg) {
    const el = document.getElementById("statusText");
    if (el) el.textContent = msg;
  }

  function setStatusRight(msg) {
    const el = document.getElementById("statusRight");
    if (el) el.textContent = msg;
  }

  // ── Connection indicator ──
  function setConnection(state) {
    // state: "connected" | "connecting" | "disconnected"
    const dot   = document.getElementById("connDot");
    const label = document.getElementById("connLabel");
    if (!dot || !label) return;
    dot.className = `oc-conn-dot ${state}`;
    const labels = { connected: "Online", connecting: "Connecting...", disconnected: "Offline" };
    label.textContent = labels[state] ?? state;
  }

  // ── Toast notifications ──
  function showToast(message, type = "info", duration = 4000) {
    const area = document.getElementById("toastArea");
    if (!area) return;

    const icons = {
      success: `<svg width="13" height="13" viewBox="0 0 16 16" fill="currentColor"><path d="M13.854 3.646a.5.5 0 010 .708l-7 7a.5.5 0 01-.708 0l-3.5-3.5a.5.5 0 11.708-.708L6.5 10.293l6.646-6.647a.5.5 0 01.708 0z"/></svg>`,
      error:   `<svg width="13" height="13" viewBox="0 0 16 16" fill="currentColor"><path d="M4.646 4.646a.5.5 0 000 .708L7.293 8l-2.647 2.646a.5.5 0 00.708.708L8 8.707l2.646 2.647a.5.5 0 00.708-.708L8.707 8l2.647-2.646a.5.5 0 00-.708-.708L8 7.293 5.354 4.646a.5.5 0 00-.708 0z"/></svg>`,
      warning: `<svg width="13" height="13" viewBox="0 0 16 16" fill="currentColor"><path d="M8 1a7 7 0 100 14A7 7 0 008 1zm0 3a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 018 4zm0 8a1 1 0 110-2 1 1 0 010 2z"/></svg>`,
      info:    `<svg width="13" height="13" viewBox="0 0 16 16" fill="currentColor"><path d="M8 1a7 7 0 100 14A7 7 0 008 1zm.93 6.588l-2.29.287-.082.38.45.083c.294.07.352.176.288.469l-.738 3.468c-.194.897.105 1.319.808 1.319.545 0 1.178-.252 1.465-.598l.088-.416c-.2.176-.492.246-.686.246-.275 0-.375-.193-.304-.533L8.93 7.588z"/></svg>`,
    };

    const toast = document.createElement("div");
    toast.className = `oc-toast ${type}`;
    toast.innerHTML = `
      <span class="oc-toast-icon">${icons[type] ?? icons.info}</span>
      <span class="oc-toast-msg">${escapeHtml(message)}</span>`;

    area.appendChild(toast);

    setTimeout(() => {
      toast.classList.add("fade-out");
      setTimeout(() => toast.remove(), 320);
    }, duration);
  }

  // ── Slider live value display ──
  function bindSlider(sliderId, valId, formatter) {
    const slider = document.getElementById(sliderId);
    const valEl  = document.getElementById(valId);
    if (!slider || !valEl) return;
    const update = () => { valEl.textContent = formatter(parseFloat(slider.value)); };
    slider.addEventListener("input", update);
    update();
  }

  // ── Collapsible cards ──
  function initCollapsibles() {
    document.querySelectorAll(".oc-card-header.collapsible").forEach(header => {
      header.addEventListener("click", () => {
        const targetId = header.dataset.target;
        const body = document.getElementById(targetId);
        if (!body) return;
        const collapsed = body.classList.toggle("collapsed");
        header.classList.toggle("collapsed", collapsed);
      });
    });
  }

  // ── Button loading state ──
  function setButtonLoading(btnId, loading) {
    const btn = document.getElementById(btnId);
    if (!btn) return;
    btn.classList.toggle("loading", loading);
    btn.disabled = loading;
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  return {
    switchTab, showProcessing, hideProcessing, setProcessingMsg, setProgress,
    setStatus, setStatusRight, setConnection, showToast,
    bindSlider, initCollapsibles, setButtonLoading, escapeHtml,
  };
})();

// ─────────────────────────────────────────────────────────────
// File browse helper (UXP localFileSystem)
// ─────────────────────────────────────────────────────────────
async function browseFile(inputId, options = {}) {
  try {
    const { localFileSystem } = await import("uxp").catch(() => null) ?? {};
    if (!localFileSystem) throw new Error("UXP localFileSystem not available");

    const entry = await localFileSystem.getFileForOpening({
      allowMultiple: false,
      types: options.types ?? ["*"],
      ...options,
    });

    if (entry) {
      const input = document.getElementById(inputId);
      if (input) input.value = entry.nativePath ?? entry.name ?? "";
      return entry.nativePath ?? entry.name ?? null;
    }
  } catch (e) {
    // In non-UXP environments or if unsupported, fall back silently
    console.warn("[browseFile]", e.message);
    UIController.showToast("File browser not available in this environment.", "warning");
  }
  return null;
}

async function browseFolder(inputId) {
  try {
    const { localFileSystem } = await import("uxp").catch(() => null) ?? {};
    if (!localFileSystem) throw new Error("UXP localFileSystem not available");

    const entry = await localFileSystem.getFolder();
    if (entry) {
      const input = document.getElementById(inputId);
      if (input) input.value = entry.nativePath ?? entry.name ?? "";
      return entry.nativePath ?? entry.name ?? null;
    }
  } catch (e) {
    console.warn("[browseFolder]", e.message);
    UIController.showToast("Folder browser not available in this environment.", "warning");
  }
  return null;
}

// ─────────────────────────────────────────────────────────────
// Project media discovery — populates clip path inputs
// ─────────────────────────────────────────────────────────────
let _projectClips = [];
let _clipScanTimer = null;

/**
 * Scan project media via UXP bridge (or backend fallback) and populate
 * a shared <datalist> so all clip path inputs offer autocomplete.
 * Also updates any <select id="clipSelect"> if present.
 */
async function scanProjectClips() {
  let items = [];

  // Try UXP bridge first (direct Premiere access)
  if (PProBridge.available()) {
    items = await PProBridge.getProjectItems();
  }

  // Fallback: ask the backend (it can query via its own bridge)
  if (items.length === 0) {
    const r = await BackendClient.get("/project/media");
    if (r.ok && Array.isArray(r.data?.media)) {
      items = r.data.media;
    }
  }

  if (items.length === 0) return;

  _projectClips = items;

  // Build or update a shared <datalist> for clip path inputs
  let datalist = document.getElementById("projectClipList");
  if (!datalist) {
    datalist = document.createElement("datalist");
    datalist.id = "projectClipList";
    document.body.appendChild(datalist);
  }
  datalist.innerHTML = items.map(c =>
    `<option value="${UIController.escapeHtml(c.path)}" label="${UIController.escapeHtml(c.name)}">`
  ).join("");

  // Attach datalist to all clip path inputs so they get autocomplete
  const clipInputIds = ["clipPathCut", "clipPathCaptions", "clipPathAudio", "clipPathVideo"];
  for (const id of clipInputIds) {
    const input = document.getElementById(id);
    if (input && !input.getAttribute("list")) {
      input.setAttribute("list", "projectClipList");
    }
  }

  // Populate <select id="clipSelect"> if it exists in the UXP panel
  const clipSelect = document.getElementById("clipSelect");
  if (clipSelect) {
    const currentVal = clipSelect.value;
    clipSelect.innerHTML = `<option value="">-- Select a clip --</option>` +
      items.map(c =>
        `<option value="${UIController.escapeHtml(c.path)}">${UIController.escapeHtml(c.name)}</option>`
      ).join("");
    if (currentVal) clipSelect.value = currentVal;
  }

  UIController.setStatusRight(`${items.length} clip(s)`);
}

// ─────────────────────────────────────────────────────────────
// Feature functions
// ─────────────────────────────────────────────────────────────

/** ── SILENCE REMOVAL ── */
async function runSilenceRemoval() {
  const clipPath = document.getElementById("clipPathCut")?.value?.trim();
  if (!clipPath) { UIController.showToast("Please select a clip first.", "warning"); return; }

  const threshold = parseFloat(document.getElementById("silenceThreshold")?.value ?? -35);
  const minSilence = parseFloat(document.getElementById("minSilence")?.value ?? 0.5);
  const padding = parseInt(document.getElementById("silencePadding")?.value ?? 80);
  const mode = document.getElementById("silenceMode")?.value ?? "remove";
  const detectMethod = document.getElementById("silenceDetectMethod")?.value ?? "auto";

  UIController.setButtonLoading("runSilenceBtn", true);
  UIController.showProcessing("Detecting silences...");
  UIController.setStatus("Running silence removal...");

  await JobPoller.start(
    "/silence",
    { filepath: clipPath, threshold: threshold, min_duration: minSilence, padding_before: padding / 1000, padding_after: padding / 1000, mode, method: detectMethod },
    (pct, msg) => {
      UIController.setProgress(pct);
      UIController.setProcessingMsg(msg || "Processing...");
    },
    (result) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runSilenceBtn", false);

      if (result.cuts && result.cuts.length > 0) {
        lastCuts = result.cuts;
        showCutResult(result);
        UIController.showToast(`Removed ${result.cuts.length} silence region(s).`, "success");
        UIController.setStatus(`Done — ${result.cuts.length} cuts`);
      } else if (result.output_path) {
        UIController.showToast(`Output: ${result.output_path}`, "success");
        UIController.setStatus("Silence removal complete.");
      } else {
        UIController.showToast("No silences found with current settings.", "info");
        UIController.setStatus("No silences detected.");
      }
    },
    (err) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runSilenceBtn", false);
      UIController.showToast(`Error: ${err}`, "error");
      UIController.setStatus("Error during silence removal.");
    }
  );
}

function showCutResult(result) {
  const area = document.getElementById("cutResultArea");
  const body = document.getElementById("cutResultBody");
  if (!area || !body) return;
  area.classList.remove("hidden");
  const lines = (result.cuts ?? []).map((c, i) =>
    `Cut ${i + 1}: ${Number(c.start).toFixed(3)}s → ${Number(c.end).toFixed(3)}s (${((Number(c.end) - Number(c.start)) * 1000).toFixed(0)} ms)`
  );
  body.textContent = lines.join("\n") || "No cuts.";
}

/** ── FILLER WORD DETECTION ── */
async function runFillerDetection() {
  const clipPath = document.getElementById("clipPathCut")?.value?.trim();
  if (!clipPath) { UIController.showToast("Please select a clip first.", "warning"); return; }

  const words   = document.getElementById("fillerWords")?.value ?? "um,uh,like";
  const padding = parseInt(document.getElementById("fillerPadding")?.value ?? 50);
  const fillerBackend = document.getElementById("fillerBackend")?.value ?? "whisper";

  UIController.setButtonLoading("runFillerBtn", true);
  UIController.showProcessing(fillerBackend === "crisper" ? "Detecting fillers with CrisperWhisper..." : "Detecting filler words...");

  await JobPoller.start(
    "/fillers",
    { filepath: clipPath, custom_words: words.split(",").map(w => w.trim()), filler_backend: fillerBackend },
    (pct, msg) => { UIController.setProgress(pct); UIController.setProcessingMsg(msg || "Transcribing..."); },
    (result) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runFillerBtn", false);
      if (result.cuts) lastCuts = result.cuts;
      const count = result.count ?? result.cuts?.length ?? 0;
      UIController.showToast(`Detected ${count} filler word(s).`, "success");
      UIController.setStatus(`Filler detection done — ${count} removed.`);
      if (result.cuts?.length) showCutResult(result);
    },
    (err) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runFillerBtn", false);
      UIController.showToast(`Error: ${err}`, "error");
    }
  );
}

/** ── TRANSCRIBE ── */
async function runTranscribe() {
  const clipPath = document.getElementById("clipPathCaptions")?.value?.trim();
  if (!clipPath) { UIController.showToast("Please select a clip first.", "warning"); return; }

  const model    = document.getElementById("whisperModel")?.value ?? "medium";
  const lang     = document.getElementById("transcribeLang")?.value ?? "auto";
  const style    = document.getElementById("captionStyle")?.value ?? "youtube_bold";
  const diarize  = document.getElementById("enableDiarization")?.checked ?? false;
  const wordLevel = document.getElementById("enableWordLevel")?.checked ?? true;

  UIController.setButtonLoading("runTranscribeBtn", true);
  UIController.showProcessing("Transcribing — this may take a while...");

  await JobPoller.start(
    "/captions",
    { filepath: clipPath, model, language: lang === "auto" ? null : lang,
      format: style, diarize, word_timestamps: wordLevel },
    (pct, msg) => { UIController.setProgress(pct); UIController.setProcessingMsg(msg || "Transcribing..."); },
    (result) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runTranscribeBtn", false);
      showCaptionsResult(result);
      UIController.showToast("Transcription complete.", "success");
      UIController.setStatus("Transcription done.");
    },
    (err) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runTranscribeBtn", false);
      UIController.showToast(`Transcription error: ${err}`, "error");
    }
  );
}

function showCaptionsResult(result) {
  const area = document.getElementById("captionsResultArea");
  const body = document.getElementById("captionsResultBody");
  if (!area || !body) return;
  area.classList.remove("hidden");
  // Prefer SRT content, then plain text, then JSON
  body.value = result.srt ?? result.text ?? JSON.stringify(result, null, 2);
}

/** ── CHAPTER GENERATION ── */
async function runChapterGeneration() {
  const llmProvider = window._llmSettings?.provider || "ollama";
  const llmModel = window._llmSettings?.model || "llama3";
  const provider = document.getElementById("llmProvider")?.value ?? llmProvider;
  const model    = document.getElementById("llmModel")?.value ?? llmModel;
  const clipPath = document.getElementById("clipPathCaptions")?.value?.trim();
  if (!clipPath) { UIController.showToast("Please transcribe a clip first, or select one.", "warning"); return; }

  UIController.setButtonLoading("runChaptersBtn", true);
  UIController.showProcessing("Generating chapters with AI...");

  await JobPoller.start(
    "/captions/chapters",
    { filepath: clipPath, llm_provider: provider, llm_model: model },
    (pct, msg) => { UIController.setProgress(pct); UIController.setProcessingMsg(msg || "Generating..."); },
    (result) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runChaptersBtn", false);
      const count = result.chapters?.length ?? 0;
      UIController.showToast(`Generated ${count} chapter(s).`, "success");
      UIController.setStatus(`Chapter generation complete — ${count} chapters.`);
      if (result.chapters) {
        const area = document.getElementById("captionsResultArea");
        const body = document.getElementById("captionsResultBody");
        if (area && body) {
          area.classList.remove("hidden");
          body.value = result.chapters.map((c, i) =>
            `${formatTimecode(c.seconds ?? c.start ?? 0)} — ${c.title ?? `Chapter ${i + 1}`}`
          ).join("\n");
        }
      }
    },
    (err) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runChaptersBtn", false);
      UIController.showToast(`Chapter generation error: ${err}`, "error");
    }
  );
}

/** ── REPEAT DETECTION ── */
async function runRepeatDetection() {
  const clipPath = document.getElementById("clipPathCaptions")?.value?.trim();
  if (!clipPath) { UIController.showToast("Please select a clip first.", "warning"); return; }

  const threshold = parseFloat(document.getElementById("repeatSimilarity")?.value ?? 0.85);
  const keepBest  = document.getElementById("keepBestRepeat")?.checked ?? true;

  UIController.setButtonLoading("runRepeatBtn", true);
  UIController.showProcessing("Detecting repeated segments...");

  await JobPoller.start(
    "/captions/repeat-detect",
    { filepath: clipPath, threshold, keep_best: keepBest },
    (pct, msg) => { UIController.setProgress(pct); UIController.setProcessingMsg(msg || "Analysing..."); },
    (result) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runRepeatBtn", false);
      const count = result.repeats?.length ?? result.removed_count ?? 0;
      UIController.showToast(`Detected ${count} repeat(s).`, "success");
      UIController.setStatus(`Repeat detection done — ${count} found.`);
    },
    (err) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runRepeatBtn", false);
      UIController.showToast(`Error: ${err}`, "error");
    }
  );
}

/** ── DENOISE ── */
async function runDenoise() {
  const clipPath = document.getElementById("clipPathAudio")?.value?.trim();
  if (!clipPath) { UIController.showToast("Please select a clip first.", "warning"); return; }

  const method   = document.getElementById("denoiseMethod")?.value ?? "noisereduce";
  const strength = parseInt(document.getElementById("denoiseStrength")?.value ?? 75) / 100;

  UIController.setButtonLoading("runDenoiseBtn", true);
  UIController.showProcessing("Applying noise reduction...");

  await JobPoller.start(
    "/audio/denoise",
    { filepath: clipPath, method, strength },
    (pct, msg) => { UIController.setProgress(pct); UIController.setProcessingMsg(msg || "Processing..."); },
    (result) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runDenoiseBtn", false);
      UIController.showToast(`Denoise complete. Output: ${result.output ?? result.output_path ?? "saved"}`, "success");
      UIController.setStatus("Denoise complete.");
    },
    (err) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runDenoiseBtn", false);
      UIController.showToast(`Denoise error: ${err}`, "error");
    }
  );
}

/** ── NORMALIZE ── */
async function runNormalize() {
  const clipPath = document.getElementById("clipPathAudio")?.value?.trim();
  if (!clipPath) { UIController.showToast("Please select a clip first.", "warning"); return; }

  const targetLufs = parseFloat(document.getElementById("targetLufs")?.value ?? -14);
  const truePeak   = document.getElementById("normalizeTruePeak")?.checked ?? true;

  UIController.setButtonLoading("runNormalizeBtn", true);
  UIController.showProcessing("Normalizing audio...");

  await JobPoller.start(
    "/audio/normalize",
    { filepath: clipPath, target_lufs: targetLufs, true_peak: truePeak ? -1.0 : null },
    (pct, msg) => { UIController.setProgress(pct); UIController.setProcessingMsg(msg || "Normalizing..."); },
    (result) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runNormalizeBtn", false);
      UIController.showToast(`Normalization complete. Output: ${result.output ?? result.output_path ?? "saved"}`, "success");
      UIController.setStatus("Normalization complete.");
    },
    (err) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runNormalizeBtn", false);
      UIController.showToast(`Normalization error: ${err}`, "error");
    }
  );
}

/** ── LOUDNESS MATCH ── */
async function runLoudnessMatch() {
  const clipPath = document.getElementById("clipPathAudio")?.value?.trim();
  const refPath  = document.getElementById("refClipLoudness")?.value?.trim();
  if (!clipPath || !refPath) {
    UIController.showToast("Please select both input and reference clips.", "warning");
    return;
  }

  UIController.setButtonLoading("runLoudnessBtn", true);
  UIController.showProcessing("Matching loudness to reference...");

  await JobPoller.start(
    "/audio/loudness-match",
    { files: [clipPath, refPath], target_lufs: -14.0 },
    (pct, msg) => { UIController.setProgress(pct); UIController.setProcessingMsg(msg || "Matching..."); },
    (result) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runLoudnessBtn", false);
      UIController.showToast("Loudness match complete.", "success");
      UIController.setStatus("Loudness match done.");
    },
    (err) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runLoudnessBtn", false);
      UIController.showToast(`Error: ${err}`, "error");
    }
  );
}

/** ── BEAT MARKERS ── */
async function runBeatMarkers() {
  const trackPath = document.getElementById("beatTrackPath")?.value?.trim();
  if (!trackPath) { UIController.showToast("Please select an audio/music file.", "warning"); return; }

  const sensitivity = parseInt(document.getElementById("beatSensitivity")?.value ?? 60) / 100;

  UIController.setButtonLoading("runBeatMarkersBtn", true);
  UIController.showProcessing("Detecting beats...");

  await JobPoller.start(
    "/audio/beat-markers",
    { filepath: trackPath, subdivisions: Math.max(1, Math.round(sensitivity * 4)) },
    (pct, msg) => { UIController.setProgress(pct); UIController.setProcessingMsg(msg || "Analysing tempo..."); },
    async (result) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runBeatMarkersBtn", false);

      const beats = result.beats ?? result.markers ?? [];
      lastMarkers = beats;
      UIController.showToast(`Detected ${beats.length} beats. Adding markers to timeline...`, "success");
      UIController.setStatus(`Beat detection done — ${beats.length} beats.`);

      // Attempt direct UXP marker insertion
      await addSequenceMarkers(beats, "green");
    },
    (err) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runBeatMarkersBtn", false);
      UIController.showToast(`Beat detection error: ${err}`, "error");
    }
  );
}

/** ── COLOR MATCH ── */
async function runColorMatch() {
  const clipPath = document.getElementById("clipPathVideo")?.value?.trim();
  const refPath  = document.getElementById("colorRefPath")?.value?.trim();
  if (!clipPath || !refPath) {
    UIController.showToast("Please select both input and reference clips.", "warning");
    return;
  }

  const strength = parseInt(document.getElementById("colorMatchStrength")?.value ?? 80) / 100;

  UIController.setButtonLoading("runColorMatchBtn", true);
  UIController.showProcessing("Matching color grading...");

  await JobPoller.start(
    "/video/color-match",
    { source: clipPath, reference: refPath, strength },
    (pct, msg) => { UIController.setProgress(pct); UIController.setProcessingMsg(msg || "Grading..."); },
    (result) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runColorMatchBtn", false);
      UIController.showToast(`Color match complete. Output: ${result.output ?? result.output_path ?? "saved"}`, "success");
      UIController.setStatus("Color match complete.");
    },
    (err) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runColorMatchBtn", false);
      UIController.showToast(`Color match error: ${err}`, "error");
    }
  );
}

/** ── AUTO ZOOM ── */
async function runAutoZoom() {
  const clipPath = document.getElementById("clipPathVideo")?.value?.trim();
  if (!clipPath) { UIController.showToast("Please select a clip first.", "warning"); return; }

  const aspect    = document.getElementById("zoomAspect")?.value ?? "9:16";
  const maxZoom   = parseFloat(document.getElementById("zoomFactor")?.value ?? 1.4);

  UIController.setButtonLoading("runAutoZoomBtn", true);
  UIController.showProcessing("Applying auto zoom / reframe...");

  await JobPoller.start(
    "/video/auto-zoom",
    { filepath: clipPath, zoom_amount: maxZoom, easing: "ease_in_out" },
    (pct, msg) => { UIController.setProgress(pct); UIController.setProcessingMsg(msg || "Reframing..."); },
    (result) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runAutoZoomBtn", false);
      UIController.showToast(`Auto zoom complete. Output: ${result.output ?? result.output_path ?? "saved"}`, "success");
      UIController.setStatus("Auto zoom complete.");
    },
    (err) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runAutoZoomBtn", false);
      UIController.showToast(`Auto zoom error: ${err}`, "error");
    }
  );
}

/** ── MULTICAM CUTS ── */
async function runMulticamCuts() {
  const cam1Path = document.getElementById("clipPathVideo")?.value?.trim();
  const cam2Path = document.getElementById("cam2Path")?.value?.trim();
  if (!cam1Path || !cam2Path) {
    UIController.showToast("Please select both camera files.", "warning");
    return;
  }

  const strategy = document.getElementById("multicamStrategy")?.value ?? "activity";

  UIController.setButtonLoading("runMulticamBtn", true);
  UIController.showProcessing("Generating multicam cuts...");

  await JobPoller.start(
    "/video/multicam-cuts",
    { filepath: cam1Path, diarization_file: cam2Path, min_cut_duration: 1.0 },
    (pct, msg) => { UIController.setProgress(pct); UIController.setProcessingMsg(msg || "Analysing cameras..."); },
    async (result) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runMulticamBtn", false);
      const cuts = result.cuts ?? [];
      lastCuts = cuts;
      UIController.showToast(`Generated ${cuts.length} multicam cut point(s).`, "success");
      UIController.setStatus(`Multicam cuts ready — ${cuts.length} cuts.`);
      // Attempt to apply directly to timeline
      await applyTimelineCuts(cuts);
    },
    (err) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runMulticamBtn", false);
      UIController.showToast(`Multicam error: ${err}`, "error");
    }
  );
}

/** ── APPLY TIMELINE CUTS (UXP + fallback) ── */
async function applyTimelineCuts(cuts) {
  const cutsToApply = cuts ?? lastCuts;
  if (!cutsToApply || cutsToApply.length === 0) {
    UIController.showToast("No cuts to apply. Run silence removal or filler detection first.", "warning");
    return;
  }

  if (PProBridge.available()) {
    UIController.setStatus("Applying cuts to timeline via UXP...");
    const result = await PProBridge.applyCuts(cutsToApply);
    if (result.ok) {
      UIController.showToast(`Applied ${result.applied} cut(s) to active sequence.`, "success");
      UIController.setStatus(`Applied ${result.applied} cut(s).`);
    } else {
      UIController.showToast(
        `UXP timeline write failed: ${result.reason}. Use CEP panel for Premiere < 25.6.`,
        "warning"
      );
      UIController.setStatus("Timeline write failed — see CEP panel.");
    }
  } else {
    UIController.showToast(
      "Connect via CEP panel for timeline operations (UXP timeline API in preview).",
      "info"
    );
    UIController.setStatus("UXP timeline API unavailable — use CEP panel.");
  }
}

/** ── ADD SEQUENCE MARKERS (UXP + fallback) ── */
async function addSequenceMarkers(markers, color) {
  const markersToAdd = markers ?? lastMarkers;
  if (!markersToAdd || markersToAdd.length === 0) {
    UIController.showToast("No markers to add. Run beat detection first.", "warning");
    return;
  }

  const markerColor = color ?? document.getElementById("beatMarkerColor")?.value ?? "green";
  const formatted = markersToAdd.map(m => ({
    time:  typeof m === "number" ? m : (m.time ?? m.t ?? 0),
    label: m.label ?? "Beat",
    color: markerColor,
  }));

  if (PProBridge.available()) {
    UIController.setStatus("Adding markers to sequence via UXP...");
    const result = await PProBridge.addMarkers(formatted);
    if (result.ok) {
      UIController.showToast(`Added ${result.count} marker(s) to active sequence.`, "success");
      UIController.setStatus(`Added ${result.count} marker(s).`);
    } else {
      UIController.showToast(
        `UXP marker insertion failed: ${result.reason}. Use CEP panel as fallback.`,
        "warning"
      );
    }
  } else {
    UIController.showToast(
      "Connect via CEP panel for timeline operations (UXP timeline API in preview).",
      "info"
    );
  }
}

/** ── BATCH EXPORT ── */
async function runBatchExport() {
  const preset    = document.getElementById("exportPreset")?.value ?? "youtube";
  const outputDir = document.getElementById("exportDir")?.value?.trim();
  if (!outputDir) { UIController.showToast("Please select an output folder.", "warning"); return; }

  const clipPath = document.getElementById("clipPathVideo")?.value?.trim() ?? "";
  const markersToExport = lastMarkers ?? lastCuts ?? [];
  if (!clipPath) { UIController.showToast("Please select a clip first.", "warning"); return; }
  if (markersToExport.length === 0) { UIController.showToast("No markers or cuts to export. Run beat detection or silence removal first.", "warning"); return; }

  UIController.setButtonLoading("runBatchExportBtn", true);
  UIController.showProcessing("Starting batch export from markers...");

  await JobPoller.start(
    "/timeline/export-from-markers",
    { input_file: clipPath, markers: markersToExport, output_dir: outputDir, format: preset },
    (pct, msg) => { UIController.setProgress(pct); UIController.setProcessingMsg(msg || "Exporting..."); },
    (result) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runBatchExportBtn", false);
      const count = result.exported ?? result.count ?? 0;
      UIController.showToast(`Exported ${count} segment(s).`, "success");
      UIController.setStatus(`Batch export done — ${count} files.`);
    },
    (err) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runBatchExportBtn", false);
      UIController.showToast(`Export error: ${err}`, "error");
    }
  );
}

/** ── BATCH RENAME ── */
async function runBatchRename() {
  const pattern = document.getElementById("renamePattern")?.value?.trim() ?? "{name}_{index:03d}";
  const scope   = document.getElementById("renameScope")?.value ?? "selected";

  UIController.setButtonLoading("runBatchRenameBtn", true);
  UIController.showProcessing("Renaming project items...");

  await JobPoller.start(
    "/timeline/batch-rename",
    { pattern, scope },
    (pct, msg) => { UIController.setProgress(pct); UIController.setProcessingMsg(msg || "Renaming..."); },
    (result) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runBatchRenameBtn", false);
      const count = result.renamed ?? 0;
      UIController.showToast(`Renamed ${count} item(s).`, "success");
      UIController.setStatus(`Batch rename done — ${count} items.`);
    },
    (err) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runBatchRenameBtn", false);
      UIController.showToast(`Rename error: ${err}`, "error");
    }
  );
}

/** ── SMART BINS ── */
async function runSmartBins() {
  const strategy = document.getElementById("binStrategy")?.value ?? "type";

  UIController.setButtonLoading("runSmartBinsBtn", true);
  UIController.showProcessing("Organising project bins...");

  await JobPoller.start(
    "/timeline/smart-bins",
    { strategy },
    (pct, msg) => { UIController.setProgress(pct); UIController.setProcessingMsg(msg || "Organising..."); },
    (result) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runSmartBinsBtn", false);
      const bins = result.bins_created ?? result.count ?? 0;
      UIController.showToast(`Created ${bins} smart bin(s).`, "success");
      UIController.setStatus(`Smart bins done — ${bins} bins.`);
    },
    (err) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runSmartBinsBtn", false);
      UIController.showToast(`Smart bins error: ${err}`, "error");
    }
  );
}

/** ── SRT IMPORT ── */
async function runSrtImport() {
  const srtPath    = document.getElementById("srtFilePath")?.value?.trim();
  const trackIndex = parseInt(document.getElementById("srtTrackIndex")?.value ?? 1);
  if (!srtPath) { UIController.showToast("Please select an SRT file.", "warning"); return; }

  UIController.setButtonLoading("runSrtImportBtn", true);
  UIController.showProcessing("Importing SRT to timeline...");

  await JobPoller.start(
    "/timeline/srt-to-captions",
    { srt_path: srtPath, track_index: trackIndex },
    (pct, msg) => { UIController.setProgress(pct); UIController.setProcessingMsg(msg || "Importing..."); },
    (result) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runSrtImportBtn", false);
      const count = result.captions_imported ?? result.count ?? 0;
      UIController.showToast(`Imported ${count} caption(s) to track ${trackIndex}.`, "success");
      UIController.setStatus(`SRT import done — ${count} captions.`);
    },
    (err) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runSrtImportBtn", false);
      UIController.showToast(`SRT import error: ${err}`, "error");
    }
  );
}

/** ── INDEX LIBRARY ── */
async function runIndexLibrary() {
  const folder       = document.getElementById("indexFolder")?.value?.trim();
  const transcripts  = document.getElementById("indexTranscripts")?.checked ?? true;
  const frames       = document.getElementById("indexFrames")?.checked ?? false;
  if (!folder) { UIController.showToast("Please select a media folder to index.", "warning"); return; }

  const statusLine = document.getElementById("indexStatus");
  UIController.setButtonLoading("runIndexLibBtn", true);
  UIController.showProcessing("Indexing media library...");
  if (statusLine) statusLine.textContent = "Indexing...";

  await JobPoller.start(
    "/search/index",
    { folder, model: "base" },
    (pct, msg) => {
      UIController.setProgress(pct);
      UIController.setProcessingMsg(msg || "Scanning...");
      if (statusLine) statusLine.textContent = msg || "Scanning...";
    },
    (result) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runIndexLibBtn", false);
      const count = result.indexed ?? result.files ?? 0;
      if (statusLine) statusLine.textContent = `Indexed ${count} file(s).`;
      UIController.showToast(`Library indexed — ${count} files.`, "success");
      UIController.setStatus(`Library indexed — ${count} files.`);
    },
    (err) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runIndexLibBtn", false);
      if (statusLine) statusLine.textContent = `Error: ${err}`;
      UIController.showToast(`Index error: ${err}`, "error");
    }
  );
}

/** ── FOOTAGE SEARCH ── */
async function runFootageSearch() {
  const query = document.getElementById("searchQuery")?.value?.trim();
  const limit = parseInt(document.getElementById("searchResultCount")?.value ?? 10);
  if (!query) { UIController.showToast("Please enter a search query.", "warning"); return; }

  UIController.setButtonLoading("runFootageSearchBtn", true);
  UIController.setStatus("Searching footage...");

  const r = await BackendClient.post("/search/footage", { query, top_k: limit });

  UIController.setButtonLoading("runFootageSearchBtn", false);

  if (!r.ok) {
    UIController.showToast(`Search error: ${r.error}`, "error");
    return;
  }

  const results = r.data?.results ?? r.data ?? [];
  const list    = document.getElementById("searchResultList");
  if (list) {
    list.innerHTML = "";
    if (results.length === 0) {
      list.innerHTML = `<div class="oc-hint">No results found.</div>`;
    } else {
      results.forEach(item => {
        const el = document.createElement("div");
        el.className = "oc-result-list-item";
        el.textContent = item.path ?? item.file ?? JSON.stringify(item);
        el.title = item.score != null ? `Score: ${item.score.toFixed(3)}` : "";
        el.addEventListener("click", () => {
          const input = document.getElementById("clipPathVideo");
          if (input) input.value = item.path ?? item.file ?? "";
          UIController.showToast(`Selected: ${item.path ?? item.file}`, "info");
        });
        list.appendChild(el);
      });
    }
  }

  UIController.setStatus(`Search done — ${results.length} result(s).`);
}

/** ── NLP COMMAND ── */
async function runNlpCommand() {
  const llmProvider = window._llmSettings?.provider || "ollama";
  const command  = document.getElementById("nlpCommand")?.value?.trim();
  const provider = document.getElementById("nlpLlmProvider")?.value ?? llmProvider;
  if (!command) { UIController.showToast("Please enter a natural language command.", "warning"); return; }

  UIController.setButtonLoading("runNlpBtn", true);
  UIController.showProcessing("Parsing command with AI...");

  await JobPoller.start(
    "/nlp/command",
    { command, llm_provider: provider },
    (pct, msg) => { UIController.setProgress(pct); UIController.setProcessingMsg(msg || "Thinking..."); },
    (result) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runNlpBtn", false);
      showNlpResult(result);
      UIController.showToast("NLP command parsed.", "success");
      UIController.setStatus("NLP command parsed — review and apply.");
    },
    (err) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runNlpBtn", false);
      UIController.showToast(`NLP error: ${err}`, "error");
    }
  );
}

function showNlpResult(result) {
  const area = document.getElementById("nlpResultArea");
  const body = document.getElementById("nlpResultBody");
  if (!area || !body) return;
  area.style.display = "flex";
  body.textContent = JSON.stringify(result.action ?? result, null, 2);
}

/** ── LOAD SEQUENCE INFO ── */
async function loadSequenceInfo() {
  UIController.setButtonLoading("loadSeqInfoBtn", true);
  UIController.setStatus("Loading sequence info...");

  let info = null;

  // Try UXP first
  if (PProBridge.available()) {
    info = await PProBridge.getSequenceInfo();
  }

  // No backend equivalent for sequence-info; PProBridge is the only source
  if (!info) {
    info = null;
  }

  UIController.setButtonLoading("loadSeqInfoBtn", false);

  const grid = document.getElementById("seqInfoGrid");
  if (!grid) return;

  if (!info) {
    grid.innerHTML = `<span class="oc-hint" style="grid-column:1/-1;">No active sequence found.</span>`;
    UIController.setStatus("No active sequence.");
    return;
  }

  const rows = [
    ["Name",        info.name],
    ["Duration",    typeof info.duration === "number" ? formatTimecode(info.duration) : (info.duration ?? "—")],
    ["Frame Rate",  info.framerate],
    ["Resolution",  `${info.width} × ${info.height}`],
    ["Video Tracks",info.videoTracks],
    ["Audio Tracks",info.audioTracks],
  ];

  grid.innerHTML = rows.map(([k, v]) =>
    `<span class="oc-info-key">${UIController.escapeHtml(k)}</span>` +
    `<span class="oc-info-val">${UIController.escapeHtml(String(v ?? "—"))}</span>`
  ).join("");

  UIController.setStatus(`Sequence: ${info.name ?? "—"}`);
}

/** ── DELIVERABLES ── */
async function runDeliverables(type) {
  const outputDir = document.getElementById("delivOutputDir")?.value?.trim();

  const btnMap = {
    vfx_sheet:       "delivVfxSheetBtn",
    adr_list:        "delivAdrListBtn",
    music_cue_sheet: "delivMusicCueBtn",
    asset_list:      "delivAssetListBtn",
  };

  const btnId = btnMap[type];
  if (btnId) UIController.setButtonLoading(btnId, true);
  UIController.showProcessing(`Generating ${type.replace(/_/g, " ")}...`);

  await JobPoller.start(
    `/deliverables/${type.replace(/_/g, "-")}`,
    { sequence_data: { video_tracks: [], audio_tracks: [] }, output_dir: outputDir || null },
    (pct, msg) => { UIController.setProgress(pct); UIController.setProcessingMsg(msg || "Generating..."); },
    (result) => {
      UIController.hideProcessing();
      if (btnId) UIController.setButtonLoading(btnId, false);
      UIController.showToast(
        `${type.replace(/_/g, " ")} saved: ${result.output ?? result.output_path ?? "done"}`,
        "success"
      );
      UIController.setStatus(`${type} generated.`);
    },
    (err) => {
      UIController.hideProcessing();
      if (btnId) UIController.setButtonLoading(btnId, false);
      UIController.showToast(`Deliverable error: ${err}`, "error");
    }
  );
}

/** ── FULL PROJECT REPORT (generates all 4 deliverables) ── */
async function runFullReport() {
  const outputDir = document.getElementById("delivOutputDir")?.value?.trim();
  const types = ["vfx-sheet", "adr-list", "music-cue-sheet", "asset-list"];
  const seqData = { video_tracks: [], audio_tracks: [] };

  // Try to get real sequence data from UXP bridge
  if (PProBridge.available()) {
    const info = await PProBridge.getSequenceInfo();
    if (info) Object.assign(seqData, info);
  }

  UIController.setButtonLoading("runFullReportBtn", true);
  UIController.showProcessing("Generating full project report...");

  let generated = 0;
  let errors = 0;
  for (const type of types) {
    const r = await BackendClient.post(`/deliverables/${type}`, {
      sequence_data: seqData,
      output_dir: outputDir || null,
    });
    if (r.ok) generated++; else errors++;
  }

  UIController.hideProcessing();
  UIController.setButtonLoading("runFullReportBtn", false);
  if (errors === 0) {
    UIController.showToast(`Generated ${generated} deliverable document(s).`, "success");
  } else {
    UIController.showToast(`Generated ${generated}, failed ${errors}.`, "warning");
  }
  UIController.setStatus(`Report: ${generated} docs generated.`);
}

// ─────────────────────────────────────────────────────────────
// AI B-Roll Generation
// ─────────────────────────────────────────────────────────────
async function runBrollGenerate() {
  const prompt = document.getElementById("brollGenPromptUxp")?.value?.trim();
  if (!prompt) { UIController.showToast("Enter a B-roll description.", "warning"); return; }
  const backend = document.getElementById("brollGenBackendUxp")?.value ?? "auto";
  const seedEl = document.getElementById("brollGenSeedUxp");
  const payload = { prompt, backend };
  if (seedEl?.value) payload.seed = parseInt(seedEl.value);
  UIController.setButtonLoading("runBrollGenBtnUxp", true);
  UIController.showProcessing("Generating AI B-roll...");
  const r = await BackendClient.post("/video/broll-generate", payload);
  if (r.ok && r.data?.job_id) {
    const result = await JobPoller.poll(r.data.job_id);
    if (result?.output_path) {
      UIController.showToast(`B-roll generated: ${result.output_path.split(/[/\\]/).pop()}`, "success");
    }
  } else {
    UIController.showToast(`B-roll generation failed: ${r.error}`, "error");
  }
  UIController.hideProcessing();
  UIController.setButtonLoading("runBrollGenBtnUxp", false);
}

// ─────────────────────────────────────────────────────────────
// Multimodal Diarization
// ─────────────────────────────────────────────────────────────
async function runMultimodalDiarize() {
  const clipPath = document.getElementById("clipPathVideo")?.value?.trim();
  if (!clipPath) { UIController.showToast("Select a video clip first.", "warning"); return; }
  const numSpeakers = document.getElementById("mmDiarizeNumSpeakersUxp")?.value || "";
  const payload = { filepath: clipPath, sample_fps: 2.0, min_face_confidence: 0.5 };
  if (numSpeakers) payload.num_speakers = parseInt(numSpeakers);
  UIController.setButtonLoading("runMmDiarizeBtnUxp", true);
  UIController.showProcessing("Running multimodal diarization...");
  const r = await BackendClient.post("/video/multimodal-diarize", payload);
  if (r.ok && r.data?.job_id) {
    const result = await JobPoller.poll(r.data.job_id);
    if (result) {
      const msg = `${result.num_speakers ?? 0} speakers, ${result.num_faces ?? 0} faces, ${(result.mappings ?? []).length} mapped`;
      UIController.showToast(`Diarization complete: ${msg}`, "success");
      UIController.setStatus(msg);
    }
  } else {
    UIController.showToast(`Diarization failed: ${r.error}`, "error");
  }
  UIController.hideProcessing();
  UIController.setButtonLoading("runMmDiarizeBtnUxp", false);
}

// ─────────────────────────────────────────────────────────────
// Social Media Upload
// ─────────────────────────────────────────────────────────────
async function runSocialUpload() {
  const clipPath = document.getElementById("clipPathVideo")?.value?.trim();
  if (!clipPath) { UIController.showToast("Select a video to upload.", "warning"); return; }
  const platform = document.getElementById("socialPlatformUxp")?.value ?? "youtube";
  const title = document.getElementById("socialTitleUxp")?.value ?? "";
  const description = document.getElementById("socialDescriptionUxp")?.value ?? "";
  const privacy = document.getElementById("socialPrivacyUxp")?.value ?? "private";
  UIController.setButtonLoading("socialUploadBtnUxp", true);
  UIController.showProcessing(`Uploading to ${platform}...`);
  const r = await BackendClient.post("/social/upload", {
    filepath: clipPath, platform, title, description, privacy,
  });
  if (r.ok && r.data?.job_id) {
    const result = await JobPoller.poll(r.data.job_id);
    if (result?.url) {
      UIController.showToast(`Uploaded! View at: ${result.url}`, "success");
    } else if (result) {
      UIController.showToast(`Uploaded to ${platform}!`, "success");
    }
  } else {
    UIController.showToast(`Upload failed: ${r.error}`, "error");
  }
  UIController.hideProcessing();
  UIController.setButtonLoading("socialUploadBtnUxp", false);
}

async function socialConnectUxp() {
  const platform = document.getElementById("socialPlatformUxp")?.value ?? "youtube";
  const r = await BackendClient.post("/social/auth-url", { platform });
  if (r.ok && r.data?.auth_url) {
    // Open in system browser
    try {
      const shell = require("uxp").shell;
      await shell.openExternal(r.data.auth_url);
    } catch (_) {
      UIController.showToast("Open this URL in your browser: " + r.data.auth_url, "info", 10000);
    }
  } else {
    UIController.showToast(`OAuth not configured for ${platform}. Set API credentials.`, "warning");
  }
}

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────
function formatTimecode(seconds) {
  if (typeof seconds !== "number" || isNaN(seconds)) return "—";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  const f = Math.floor((seconds % 1) * 100);
  return `${pad(h)}:${pad(m)}:${pad(s)}.${pad(f)}`;
}

function pad(n) { return String(n).padStart(2, "0"); }

// ─────────────────────────────────────────────────────────────
// Health check loop
// ─────────────────────────────────────────────────────────────
let _lastConnectionState = null;

async function checkConnection() {
  UIController.setConnection("connecting");
  const r = await BackendClient.get("/health");
  const alive = r.ok;
  if (alive && r.data?.csrf_token) csrfToken = r.data.csrf_token;
  UIController.setConnection(alive ? "connected" : "disconnected");

  // Visual indicator: toggle a class on the status bar so CSS can style it
  const statusBar = document.getElementById("statusBar");
  if (statusBar) {
    statusBar.classList.toggle("backend-down", !alive);
  }

  if (alive) {
    UIController.setStatus("Server online");
    UIController.setStatusRight(`v${VERSION}`);
  } else {
    UIController.setStatus("Server offline — start the OpenCut backend");
    UIController.setStatusRight("");
  }

  // Toggle all action buttons based on connection state
  const wasAlive = _lastConnectionState;
  if (wasAlive !== alive) {
    _lastConnectionState = alive;
    document.querySelectorAll(".oc-btn-primary").forEach(btn => {
      // Don't override buttons already disabled for other reasons (loading state)
      if (!btn.classList.contains("loading")) {
        btn.disabled = !alive;
      }
    });
    // Show reconnection toast when server comes back
    if (alive && wasAlive === false) {
      UIController.showToast("Server reconnected.", "success");
    }
  }

  return alive;
}

// ─────────────────────────────────────────────────────────────
// Keyboard Shortcuts
// ─────────────────────────────────────────────────────────────
const SHORTCUT_DEFS = {
  "silence-detect":    { keys: "Ctrl+Shift+S", label: "Detect Silence" },
  "caption-generate":  { keys: "Ctrl+Shift+C", label: "Generate Captions" },
  "audio-normalize":   { keys: "Ctrl+Shift+N", label: "Normalize Audio" },
  "audio-denoise":     { keys: "Ctrl+Shift+D", label: "Denoise Audio" },
  "export-video":      { keys: "Ctrl+Shift+E", label: "Export Video" },
  "cancel-job":        { keys: "Escape",        label: "Cancel Current Job" },
};

const _shortcutActions = {
  "silence-detect":   () => { const b = document.getElementById("runSilenceBtn"); if (b && !b.disabled) b.click(); },
  "caption-generate": () => { const b = document.getElementById("runTranscribeBtn"); if (b && !b.disabled) b.click(); },
  "audio-normalize":  () => { const b = document.getElementById("runNormalizeBtn"); if (b && !b.disabled) b.click(); },
  "audio-denoise":    () => { const b = document.getElementById("runDenoiseBtn"); if (b && !b.disabled) b.click(); },
  "export-video":     () => { const b = document.getElementById("runBatchExportBtn"); if (b && !b.disabled) b.click(); },
  "cancel-job":       () => { if (activeJobId) JobPoller.cancel().then(() => {
    UIController.hideProcessing();
    UIController.showToast("Job cancelled.", "warning");
  }); },
};

/**
 * Match a keyboard event against a shortcut string like "Ctrl+Shift+S" or "Escape".
 */
function matchesShortcut(e, keysStr) {
  const parts = keysStr.split("+");
  let needCtrl = false, needShift = false, needAlt = false, needMeta = false;
  let keyPart = "";
  for (const p of parts) {
    const lp = p.trim().toLowerCase();
    if (lp === "ctrl")               needCtrl  = true;
    else if (lp === "shift")         needShift = true;
    else if (lp === "alt")           needAlt   = true;
    else if (lp === "meta" || lp === "cmd") needMeta  = true;
    else                             keyPart   = lp;
  }
  if (e.ctrlKey !== needCtrl)  return false;
  if (e.shiftKey !== needShift) return false;
  if (e.altKey !== needAlt)    return false;
  if (e.metaKey !== needMeta)  return false;
  const eventKey = e.key.toLowerCase();
  if (keyPart === "escape") return eventKey === "escape";
  return eventKey === keyPart;
}

function initKeyboardShortcuts() {
  document.addEventListener("keydown", (e) => {
    // Escape / cancel-job always works, even inside inputs
    if (!e.defaultPrevented && matchesShortcut(e, SHORTCUT_DEFS["cancel-job"].keys) && activeJobId) {
      _shortcutActions["cancel-job"]();
      return;
    }

    // Don't fire other shortcuts when typing in inputs
    const tag = e.target.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || e.target.isContentEditable) return;

    for (const id of Object.keys(SHORTCUT_DEFS)) {
      if (id === "cancel-job") continue;
      if (matchesShortcut(e, SHORTCUT_DEFS[id].keys)) {
        e.preventDefault();
        if (_shortcutActions[id]) _shortcutActions[id]();
        return;
      }
    }

    // Enter to activate the primary button on the visible tab
    if (e.key === "Enter" && !activeJobId) {
      const activePanel = document.querySelector(".oc-tab-panel.active");
      if (!activePanel) return;
      const primaryBtn = activePanel.querySelector(".oc-btn-primary:not([disabled])");
      if (primaryBtn) {
        e.preventDefault();
        primaryBtn.click();
      }
    }

    // Number keys 1-8 to switch tabs (only when focus is on body)
    if (e.key >= "1" && e.key <= "8" && !e.ctrlKey && !e.altKey && !e.metaKey && !e.shiftKey
        && e.target === document.body) {
      const tabBtns = document.querySelectorAll(".oc-tab");
      const idx = parseInt(e.key) - 1;
      if (tabBtns[idx]) {
        e.preventDefault();
        tabBtns[idx].click();
      }
    }
  });
}

// ─────────────────────────────────────────────────────────────
// Periodic media scan — notify backend to re-scan active timeline media
// ─────────────────────────────────────────────────────────────
function startMediaScanInterval() {
  if (_mediaScanTimer) clearInterval(_mediaScanTimer);
  _mediaScanTimer = setInterval(async () => {
    if (activeJobId) return; // skip during active jobs
    try {
      await BackendClient.get("/project/media");
    } catch (_) { /* ignore scan failures */ }
  }, MEDIA_SCAN_MS);
}

function stopMediaScanInterval() {
  if (_mediaScanTimer) { clearInterval(_mediaScanTimer); _mediaScanTimer = null; }
}

// ─────────────────────────────────────────────────────────────
// Event binding
// ─────────────────────────────────────────────────────────────
function bindEvents() {
  // ── Tab navigation ──
  document.querySelectorAll(".oc-tab").forEach(btn => {
    btn.addEventListener("click", () => UIController.switchTab(btn.dataset.tab));
  });

  // ── Refresh button ──
  document.getElementById("refreshBtn")?.addEventListener("click", () => checkConnection());

  // ── Cancel button ──
  document.getElementById("cancelBtn")?.addEventListener("click", async () => {
    await JobPoller.cancel();
    UIController.hideProcessing();
    UIController.showToast("Job cancelled.", "warning");
  });

  // ── Cut & Clean ──
  document.getElementById("browseClipCut")?.addEventListener("click", () => browseFile("clipPathCut"));
  document.getElementById("runSilenceBtn")?.addEventListener("click", runSilenceRemoval);
  document.getElementById("runFillerBtn")?.addEventListener("click", runFillerDetection);
  document.getElementById("applyCutResultBtn")?.addEventListener("click", () => applyTimelineCuts(lastCuts));

  // ── Captions ──
  document.getElementById("browseClipCaptions")?.addEventListener("click", () => browseFile("clipPathCaptions"));
  document.getElementById("runTranscribeBtn")?.addEventListener("click", runTranscribe);
  document.getElementById("runChaptersBtn")?.addEventListener("click", runChapterGeneration);
  document.getElementById("runRepeatBtn")?.addEventListener("click", runRepeatDetection);
  document.getElementById("copySrtBtn")?.addEventListener("click", () => {
    const body = document.getElementById("captionsResultBody");
    if (body?.value) {
      try { navigator.clipboard.writeText(body.value); UIController.showToast("SRT copied to clipboard.", "success"); }
      catch (_) { UIController.showToast("Could not access clipboard.", "warning"); }
    }
  });
  document.getElementById("importSrtBtn")?.addEventListener("click", async () => {
    const body = document.getElementById("captionsResultBody");
    if (!body?.value) { UIController.showToast("No transcript to import.", "warning"); return; }
    UIController.showToast("SRT import from caption result not yet wired — use Timeline > SRT Import.", "info");
  });

  // ── Audio ──
  document.getElementById("browseClipAudio")?.addEventListener("click",   () => browseFile("clipPathAudio"));
  document.getElementById("browseRefLoudness")?.addEventListener("click", () => browseFile("refClipLoudness"));
  document.getElementById("browseBeatTrack")?.addEventListener("click",   () => browseFile("beatTrackPath"));
  document.getElementById("runDenoiseBtn")?.addEventListener("click",     runDenoise);
  document.getElementById("runNormalizeBtn")?.addEventListener("click",   runNormalize);
  document.getElementById("runLoudnessBtn")?.addEventListener("click",    runLoudnessMatch);
  document.getElementById("runBeatMarkersBtn")?.addEventListener("click", runBeatMarkers);

  // ── Video ──
  document.getElementById("browseClipVideo")?.addEventListener("click", () => browseFile("clipPathVideo"));
  document.getElementById("browseColorRef")?.addEventListener("click",  () => browseFile("colorRefPath"));
  document.getElementById("browseCam2")?.addEventListener("click",      () => browseFile("cam2Path"));
  document.getElementById("runColorMatchBtn")?.addEventListener("click", runColorMatch);
  document.getElementById("runAutoZoomBtn")?.addEventListener("click",   runAutoZoom);
  document.getElementById("runMulticamBtn")?.addEventListener("click",   runMulticamCuts);

  // ── Timeline ──
  document.getElementById("browseCutsJson")?.addEventListener("click",   () => browseFile("cutsJsonPath", { types: ["json"] }));
  document.getElementById("browseExportDir")?.addEventListener("click",  () => browseFolder("exportDir"));
  document.getElementById("browseSrtFile")?.addEventListener("click",    () => browseFile("srtFilePath", { types: ["srt"] }));
  document.getElementById("applyTimelineCutsBtn")?.addEventListener("click", () => applyTimelineCuts(lastCuts));
  document.getElementById("addBeatMarkersBtn")?.addEventListener("click", () => addSequenceMarkers(lastMarkers, null));
  document.getElementById("runBatchExportBtn")?.addEventListener("click", runBatchExport);
  document.getElementById("runBatchRenameBtn")?.addEventListener("click", runBatchRename);
  document.getElementById("runSmartBinsBtn")?.addEventListener("click",   runSmartBins);
  document.getElementById("runSrtImportBtn")?.addEventListener("click",   runSrtImport);

  // ── OTIO Export ──
  document.getElementById("exportOtioBtn")?.addEventListener("click", async () => {
    const clipPath = document.getElementById("clipPathCut")?.value?.trim()
                  ?? document.getElementById("clipPathVideo")?.value?.trim() ?? "";
    if (!clipPath) { UIController.showToast("Select a clip first.", "warning"); return; }
    const mode = document.getElementById("otioExportMode")?.value ?? "cuts";
    const payload = { filepath: clipPath, mode };
    if (mode === "cuts") {
      if (!lastCuts || lastCuts.length === 0) {
        UIController.showToast("No cuts available. Run silence removal first.", "warning");
        return;
      }
      payload.cuts = lastCuts;
    } else if (mode === "markers") {
      if (!lastMarkers || lastMarkers.length === 0) {
        UIController.showToast("No markers available. Run beat detection first.", "warning");
        return;
      }
      payload.markers = lastMarkers.map(m => ({
        time: typeof m === "number" ? m : (m.time ?? m.t ?? 0),
        name: m.label ?? "Marker",
      }));
    }
    UIController.setButtonLoading("exportOtioBtn", true);
    const r = await BackendClient.post("/timeline/export-otio", payload);
    UIController.setButtonLoading("exportOtioBtn", false);
    if (r.ok) {
      UIController.showToast(`OTIO exported: ${r.data?.output_path?.split(/[/\\]/).pop() ?? "done"}`, "success");
    } else {
      UIController.showToast(`OTIO export failed: ${r.error}`, "error");
    }
  });

  // ── Search ──
  document.getElementById("browseIndexFolder")?.addEventListener("click", () => browseFolder("indexFolder"));
  document.getElementById("runIndexLibBtn")?.addEventListener("click",    runIndexLibrary);
  document.getElementById("runFootageSearchBtn")?.addEventListener("click", runFootageSearch);
  document.getElementById("runNlpBtn")?.addEventListener("click",         runNlpCommand);
  document.getElementById("applyNlpBtn")?.addEventListener("click", async () => {
    const bodyEl = document.getElementById("nlpResultBody");
    if (!bodyEl?.textContent) { UIController.showToast("No NLP result to apply.", "warning"); return; }
    try {
      const action = JSON.parse(bodyEl.textContent);
      if (action.cuts)    await applyTimelineCuts(action.cuts);
      else if (action.markers) await addSequenceMarkers(action.markers, null);
      else UIController.showToast("Unknown NLP action type. Check result JSON.", "info");
    } catch (_) {
      UIController.showToast("Could not parse NLP result as JSON.", "error");
    }
  });
  document.getElementById("searchQuery")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") runFootageSearch();
  });

  // ── Deliverables ──
  document.getElementById("browseDelivDir")?.addEventListener("click", () => browseFolder("delivOutputDir"));
  document.getElementById("loadSeqInfoBtn")?.addEventListener("click",    loadSequenceInfo);
  document.getElementById("delivVfxSheetBtn")?.addEventListener("click",  () => runDeliverables("vfx_sheet"));
  document.getElementById("delivAdrListBtn")?.addEventListener("click",   () => runDeliverables("adr_list"));
  document.getElementById("delivMusicCueBtn")?.addEventListener("click",  () => runDeliverables("music_cue_sheet"));
  document.getElementById("delivAssetListBtn")?.addEventListener("click", () => runDeliverables("asset_list"));
  document.getElementById("runFullReportBtn")?.addEventListener("click",  runFullReport);

  // ── Depth Effects ──
  document.getElementById("runDepthBtnUxp")?.addEventListener("click", runDepthEffect);
  document.getElementById("installDepthBtnUxp")?.addEventListener("click", async () => {
    const r = await BackendClient.post("/video/depth/install", {});
    if (r.ok) UIController.showToast("Installing Depth Anything V2...", "info");
    else UIController.showToast("Install failed: " + (r.error || "unknown"), "error");
  });

  // ── Emotion Highlights ──
  document.getElementById("runEmotionBtnUxp")?.addEventListener("click", runEmotionHighlights);

  // ── B-Roll Analysis ──
  document.getElementById("runBrollPlanBtnUxp")?.addEventListener("click", runBrollAnalysis);

  // ── Chat Editor ──
  document.getElementById("chatSendBtnUxp")?.addEventListener("click", sendChatMessage);
  document.getElementById("chatInputUxp")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendChatMessage();
  });

  // ── AI B-Roll Generation ──
  document.getElementById("runBrollGenBtnUxp")?.addEventListener("click", runBrollGenerate);

  // ── Multimodal Diarization ──
  document.getElementById("runMmDiarizeBtnUxp")?.addEventListener("click", runMultimodalDiarize);

  // ── AI Upscale ──
  document.getElementById("runUpscaleBtnUxp")?.addEventListener("click", runUpscaleUxp);

  // ── Scene Detection ──
  document.getElementById("runSceneDetectBtnUxp")?.addEventListener("click", runSceneDetectUxp);

  // ── Style Transfer ──
  document.getElementById("runStyleTransferBtnUxp")?.addEventListener("click", runStyleTransferUxp);

  // ── Shorts Pipeline ──
  document.getElementById("runShortsPipelineBtnUxp")?.addEventListener("click", runShortsPipelineUxp);

  // ── Social Media ──
  document.getElementById("socialConnectBtnUxp")?.addEventListener("click", socialConnectUxp);
  document.getElementById("socialUploadBtnUxp")?.addEventListener("click", runSocialUpload);

  // ── Settings: WebSocket ──
  document.getElementById("uxpWsStartBtn")?.addEventListener("click", uxpWsStartBridge);
  document.getElementById("uxpWsStopBtn")?.addEventListener("click", uxpWsStopBridge);
  document.getElementById("uxpWsConnectBtn")?.addEventListener("click", uxpWsConnect);

  // ── Settings: Engine Registry ──
  document.getElementById("uxpRefreshEnginesBtn")?.addEventListener("click", uxpLoadEngines);
}

// ─────────────────────────────────────────────────────────────
// Slider bindings
// ─────────────────────────────────────────────────────────────
function bindSliders() {
  UIController.bindSlider("silenceThreshold", "silenceThresholdVal", v => `${v} dB`);
  UIController.bindSlider("minSilence",        "minSilenceVal",       v => `${v.toFixed(2)} s`);
  UIController.bindSlider("silencePadding",    "silencePaddingVal",   v => `${v} ms`);
  UIController.bindSlider("fillerPadding",     "fillerPaddingVal",    v => `${v} ms`);
  UIController.bindSlider("repeatSimilarity",  "repeatSimilarityVal", v => v.toFixed(2));
  UIController.bindSlider("denoiseStrength",   "denoiseStrengthVal",  v => `${v}%`);
  UIController.bindSlider("targetLufs",        "targetLufsVal",       v => `${v} LUFS`);
  UIController.bindSlider("beatSensitivity",   "beatSensitivityVal",  v => `${v}%`);
  UIController.bindSlider("colorMatchStrength","colorMatchStrengthVal",v => `${v}%`);
  UIController.bindSlider("zoomFactor",        "zoomFactorVal",       v => `${v.toFixed(2)}x`);
  UIController.bindSlider("sceneThresholdUxp", "sceneThresholdValUxp", v => v.toFixed(2));
  UIController.bindSlider("styleIntensityUxp", "styleIntensityValUxp", v => `${v}%`);
}

// ─────────────────────────────────────────────────────────────
// Depth Effects
// ─────────────────────────────────────────────────────────────
async function runDepthEffect() {
  const clipPath = document.getElementById("clipPathVideo")?.value?.trim() ?? "";
  if (!clipPath) { UIController.showToast("Select a clip first.", "warning"); return; }
  const effect = document.getElementById("depthEffectUxp")?.value ?? "depth_map";
  const modelSize = document.getElementById("depthModelSizeUxp")?.value ?? "small";

  const endpointMap = {
    depth_map: "/video/depth/map",
    bokeh: "/video/depth/bokeh",
    parallax: "/video/depth/parallax",
  };
  const endpoint = endpointMap[effect] || "/video/depth/map";
  const payload = { filepath: clipPath, model_size: modelSize };

  UIController.setButtonLoading("runDepthBtnUxp", true);
  const r = await BackendClient.post(endpoint, payload);
  if (r.ok && r.data?.job_id) {
    UIController.showProcessing("Running depth effect...");
    try {
      const result = await JobPoller.poll(r.data.job_id);
      UIController.showToast(`Depth effect complete: ${result?.output_path?.split(/[/\\]/).pop() ?? "done"}`, "success");
    } catch (e) {
      UIController.showToast(`Depth effect failed: ${e.message || "unknown"}`, "error");
    }
    UIController.hideProcessing();
    UIController.setButtonLoading("runDepthBtnUxp", false);
  } else {
    UIController.setButtonLoading("runDepthBtnUxp", false);
    UIController.showToast(`Error: ${r.error || "Failed to start depth effect"}`, "error");
  }
}

// ─────────────────────────────────────────────────────────────
// Emotion Highlights
// ─────────────────────────────────────────────────────────────
async function runEmotionHighlights() {
  const clipPath = document.getElementById("clipPathVideo")?.value?.trim() ?? "";
  if (!clipPath) { UIController.showToast("Select a clip first.", "warning"); return; }

  UIController.setButtonLoading("runEmotionBtnUxp", true);
  const r = await BackendClient.post("/video/emotion-highlights", { filepath: clipPath });
  if (r.ok && r.data?.job_id) {
    UIController.showProcessing("Analyzing emotions...");
    try {
      const result = await JobPoller.poll(r.data.job_id);
      const peaks = result?.peaks?.length ?? 0;
      UIController.showToast(`Emotion analysis complete: ${peaks} emotional peaks found.`, "success");
    } catch (e) {
      UIController.showToast(`Emotion analysis failed: ${e.message || "unknown"}`, "error");
    }
    UIController.hideProcessing();
    UIController.setButtonLoading("runEmotionBtnUxp", false);
  } else {
    UIController.setButtonLoading("runEmotionBtnUxp", false);
    UIController.showToast(`Error: ${r.error || "Failed to start emotion analysis"}`, "error");
  }
}

// ─────────────────────────────────────────────────────────────
// B-Roll Analysis
// ─────────────────────────────────────────────────────────────
async function runBrollAnalysis() {
  const clipPath = document.getElementById("clipPathVideo")?.value?.trim() ?? "";
  if (!clipPath) { UIController.showToast("Select a clip first.", "warning"); return; }

  UIController.setButtonLoading("runBrollPlanBtnUxp", true);
  const r = await BackendClient.post("/video/broll-plan", { filepath: clipPath });
  if (r.ok && r.data?.job_id) {
    UIController.showProcessing("Analyzing B-roll points...");
    try {
      const result = await JobPoller.poll(r.data.job_id);
      const windows = result?.windows?.length ?? 0;
      UIController.showToast(`B-roll analysis complete: ${windows} insertion points found.`, "success");
    } catch (e) {
      UIController.showToast(`B-roll analysis failed: ${e.message || "unknown"}`, "error");
    }
    UIController.hideProcessing();
    UIController.setButtonLoading("runBrollPlanBtnUxp", false);
  } else {
    UIController.setButtonLoading("runBrollPlanBtnUxp", false);
    UIController.showToast(`Error: ${r.error || "Failed to start B-roll analysis"}`, "error");
  }
}

// ─────────────────────────────────────────────────────────────
// Chat Editor
// ─────────────────────────────────────────────────────────────
let _chatSessionId = "";

async function sendChatMessage() {
  const input = document.getElementById("chatInputUxp");
  const message = input?.value?.trim() ?? "";
  if (!message) return;

  const clipPath = document.getElementById("clipPathVideo")?.value?.trim() ?? "";
  const history = document.getElementById("chatHistory");

  // Show user message (text-safe to prevent XSS)
  if (history) {
    const userDiv = document.createElement("div");
    userDiv.className = "oc-chat-user";
    userDiv.textContent = "You: " + message;
    history.appendChild(userDiv);
    history.scrollTop = history.scrollHeight;
  }
  input.value = "";

  if (!_chatSessionId) _chatSessionId = `uxp-${Date.now()}`;

  const r = await BackendClient.post("/chat", {
    session_id: _chatSessionId,
    message: message,
    filepath: clipPath,
  });

  if (r.ok && r.data) {
    const reply = r.data.response || "No response.";
    if (history) {
      const replyDiv = document.createElement("div");
      replyDiv.className = "oc-chat-assistant";
      replyDiv.textContent = "OpenCut: " + reply;
      history.appendChild(replyDiv);
      history.scrollTop = history.scrollHeight;
    }
    // Auto-execute actions if present
    const actions = r.data.actions || [];
    if (actions.length > 0) {
      UIController.showToast(`Executing ${actions.length} action(s)...`, "info");
    }
  } else {
    if (history) {
      const errDiv = document.createElement("div");
      errDiv.className = "oc-chat-error";
      errDiv.textContent = "Error: " + (r.error || "Failed");
      history.appendChild(errDiv);
    }
  }
}

// ─────────────────────────────────────────────────────────────
// WebSocket Client
// ─────────────────────────────────────────────────────────────
let _uxpWs = null;
let _uxpWsReconnectTimer = null;
let _uxpWsConnected = false;

function uxpWsConnect() {
  if (_uxpWs && (_uxpWs.readyState === WebSocket.OPEN || _uxpWs.readyState === WebSocket.CONNECTING)) {
    UIController.showToast("WebSocket already connected.", "info");
    return;
  }
  try {
    _uxpWs = new WebSocket("ws://127.0.0.1:5680");
  } catch (e) {
    UIController.showToast("WebSocket connection failed.", "warning");
    return;
  }

  _uxpWs.onopen = () => {
    _uxpWsConnected = true;
    _uxpWs.send(JSON.stringify({ type: "identify", client_type: "uxp", id: "uxp-1" }));
    _uxpWs.send(JSON.stringify({ type: "command", action: "subscribe", params: { events: ["progress", "job_complete", "job_error"] }, id: "sub-1" }));
    uxpUpdateWsStatus();
    UIController.showToast("WebSocket connected.", "success");
  };

  _uxpWs.onmessage = (evt) => {
    try {
      const msg = JSON.parse(evt.data);
      if (msg.type === "progress" && msg.job_id) {
        const pct = msg.percent || 0;
        const message = msg.message || "";
        const fill = document.getElementById("progressFill");
        const msgEl = document.getElementById("processingMsg");
        if (fill) {
          fill.style.width = `${pct}%`;
          fill.setAttribute("aria-valuenow", String(Math.round(pct)));
        }
        if (msgEl && message) msgEl.textContent = message;
      }
    } catch (e) { /* ignore */ }
  };

  _uxpWs.onclose = () => {
    _uxpWsConnected = false;
    _uxpWs = null;
    uxpUpdateWsStatus();
    if (!_uxpWsReconnectTimer) {
      _uxpWsReconnectTimer = setTimeout(() => {
        _uxpWsReconnectTimer = null;
        uxpWsConnect();
      }, 5000);
    }
  };

  _uxpWs.onerror = () => { _uxpWsConnected = false; };
}

function uxpWsDisconnect() {
  if (_uxpWsReconnectTimer) { clearTimeout(_uxpWsReconnectTimer); _uxpWsReconnectTimer = null; }
  if (_uxpWs) { _uxpWs.close(); _uxpWs = null; }
  _uxpWsConnected = false;
  uxpUpdateWsStatus();
}

async function uxpUpdateWsStatus() {
  const statusEl = document.getElementById("uxpWsStatus");
  const countEl = document.getElementById("uxpWsClients");
  if (_uxpWsConnected) {
    if (statusEl) { statusEl.textContent = "Connected"; statusEl.style.color = "var(--accent)"; }
  } else {
    if (statusEl) { statusEl.textContent = "Disconnected"; statusEl.style.color = ""; }
  }
  const r = await BackendClient.get("/ws/status");
  if (r.ok && r.data) {
    if (countEl) countEl.textContent = r.data.clients || 0;
    if (!r.data.running && statusEl && !_uxpWsConnected) statusEl.textContent = "Server stopped";
  }
}

async function uxpWsStartBridge() {
  const r = await BackendClient.post("/ws/start", {});
  if (r.ok && r.data?.success) {
    UIController.showToast("WebSocket bridge started.", "success");
    setTimeout(() => uxpWsConnect(), 500);
  } else {
    UIController.showToast(r.error || "Failed to start bridge.", "error");
  }
}

async function uxpWsStopBridge() {
  uxpWsDisconnect();
  const r = await BackendClient.post("/ws/stop", {});
  if (r.ok) {
    UIController.showToast("WebSocket bridge stopped.", "success");
    uxpUpdateWsStatus();
  }
}

// ─────────────────────────────────────────────────────────────
// Engine Registry UI
// ─────────────────────────────────────────────────────────────
async function uxpLoadEngines() {
  const grid = document.getElementById("uxpEngineGrid");
  if (!grid) return;
  grid.innerHTML = '<p class="oc-hint">Loading engines...</p>';

  const r = await BackendClient.get("/engines");
  if (!r.ok || !r.data?.engines) {
    grid.innerHTML = '<p class="oc-hint">Could not load engine data.</p>';
    return;
  }

  const engines = r.data.engines;
  const domains = Object.keys(engines).sort();
  let html = "";

  for (const domain of domains) {
    const info = engines[domain];
    const active = info.active || "";
    const preferred = info.preferred || "";

    html += `<div class="oc-field-row"><label class="oc-label">${domain.replace(/_/g, " ")}</label>`;
    html += `<select class="oc-select oc-engine-sel" data-domain="${domain}">`;
    html += `<option value="">Auto (highest priority)</option>`;
    for (const eng of info.engines) {
      const sel = (preferred === eng.name) ? " selected" : "";
      const avail = eng.available ? "" : " (unavailable)";
      const label = `${eng.display_name} — ${eng.quality}/${eng.speed}${avail}${eng.name === active ? " *" : ""}`;
      html += `<option value="${eng.name}"${sel}>${label}</option>`;
    }
    html += `</select></div>`;
  }

  grid.innerHTML = html;

  grid.querySelectorAll(".oc-engine-sel").forEach(sel => {
    sel.addEventListener("change", async () => {
      const dom = sel.dataset.domain;
      const eng = sel.value;
      if (eng) {
        const pr = await BackendClient.post("/engines/preference", { domain: dom, engine: eng });
        if (pr.ok && pr.data?.success) UIController.showToast("Engine preference saved.", "success");
        else UIController.showToast(pr.error || "Failed to save preference.", "error");
      }
    });
  });
}

// ─────────────────────────────────────────────────────────────
// Application init
// ─────────────────────────────────────────────────────────────
async function loadLlmSettings() {
  try {
    const resp = await BackendClient.get("/settings/llm");
    if (resp.ok && resp.data) {
      // Store globally for use in feature functions
      window._llmSettings = resp.data;
      // If there's a provider select in settings tab, populate it
      const provSel = document.getElementById("llmProvider");
      if (provSel && resp.data.provider) provSel.value = resp.data.provider;
      const modInp = document.getElementById("llmModel");
      if (modInp && resp.data.model) modInp.value = resp.data.model;
    }
  } catch (e) {
    console.warn("Could not load LLM settings:", e);
  }
}

// ─────────────────────────────────────────────────────────────
// AI Upscale
// ─────────────────────────────────────────────────────────────
async function runUpscaleUxp() {
  const clipPath = document.getElementById("clipPathVideo")?.value?.trim() ?? "";
  if (!clipPath) { UIController.showToast("Select a clip first.", "warning"); return; }
  const scale = parseInt(document.getElementById("upscaleScaleUxp")?.value ?? "2", 10);
  const model = document.getElementById("upscaleModelUxp")?.value ?? "realesrgan-x4plus";

  UIController.setButtonLoading("runUpscaleBtnUxp", true);
  const r = await BackendClient.post("/video/ai/upscale", { filepath: clipPath, scale, model });
  UIController.setButtonLoading("runUpscaleBtnUxp", false);

  if (r.ok && r.data?.job_id) {
    try {
      const result = await JobPoller.poll(r.data.job_id);
      UIController.showToast(`Upscaled: ${result?.output_path || "done"}`, "success");
    } catch (e) {
      UIController.showToast(`Upscale failed: ${e.message}`, "error");
    }
  } else {
    UIController.showToast(r.data?.error || "Upscale failed.", "error");
  }
}

// ─────────────────────────────────────────────────────────────
// Scene Detection
// ─────────────────────────────────────────────────────────────
async function runSceneDetectUxp() {
  const clipPath = document.getElementById("clipPathVideo")?.value?.trim() ?? "";
  if (!clipPath) { UIController.showToast("Select a clip first.", "warning"); return; }
  const method = document.getElementById("sceneMethodUxp")?.value ?? "ffmpeg";
  const threshold = parseFloat(document.getElementById("sceneThresholdUxp")?.value ?? "0.3");

  UIController.setButtonLoading("runSceneDetectBtnUxp", true);
  const r = await BackendClient.post("/video/scenes", { filepath: clipPath, method, threshold });
  UIController.setButtonLoading("runSceneDetectBtnUxp", false);

  if (r.ok && r.data?.job_id) {
    try {
      const result = await JobPoller.poll(r.data.job_id);
      const count = result?.scenes?.length || result?.total_scenes || 0;
      UIController.showToast(`Found ${count} scene boundaries.`, "success");
    } catch (e) {
      UIController.showToast(`Scene detection failed: ${e.message}`, "error");
    }
  } else {
    UIController.showToast(r.data?.error || "Scene detection failed.", "error");
  }
}

// ─────────────────────────────────────────────────────────────
// Style Transfer
// ─────────────────────────────────────────────────────────────
async function runStyleTransferUxp() {
  const clipPath = document.getElementById("clipPathVideo")?.value?.trim() ?? "";
  if (!clipPath) { UIController.showToast("Select a clip first.", "warning"); return; }
  const style = document.getElementById("styleNameUxp")?.value ?? "candy";
  const intensity = (parseInt(document.getElementById("styleIntensityUxp")?.value ?? "100", 10)) / 100;

  UIController.setButtonLoading("runStyleTransferBtnUxp", true);
  const r = await BackendClient.post("/video/style/apply", { filepath: clipPath, style, intensity });
  UIController.setButtonLoading("runStyleTransferBtnUxp", false);

  if (r.ok && r.data?.job_id) {
    try {
      const result = await JobPoller.poll(r.data.job_id);
      UIController.showToast(`Style applied: ${result?.output_path || "done"}`, "success");
    } catch (e) {
      UIController.showToast(`Style transfer failed: ${e.message}`, "error");
    }
  } else {
    UIController.showToast(r.data?.error || "Style transfer failed.", "error");
  }
}

// ─────────────────────────────────────────────────────────────
// Shorts Pipeline
// ─────────────────────────────────────────────────────────────
async function runShortsPipelineUxp() {
  const clipPath = document.getElementById("clipPathVideo")?.value?.trim() ?? "";
  if (!clipPath) { UIController.showToast("Select a clip first.", "warning"); return; }
  const maxShorts = parseInt(document.getElementById("shortsMaxUxp")?.value ?? "5", 10);
  const faceTrack = document.getElementById("shortsFaceTrackUxp")?.checked ?? true;
  const burnCaptions = document.getElementById("shortsCaptionsUxp")?.checked ?? true;

  UIController.setButtonLoading("runShortsPipelineBtnUxp", true);
  const r = await BackendClient.post("/video/shorts-pipeline", {
    filepath: clipPath,
    max_shorts: maxShorts,
    face_track: faceTrack,
    burn_captions: burnCaptions,
  });
  UIController.setButtonLoading("runShortsPipelineBtnUxp", false);

  if (r.ok && r.data?.job_id) {
    try {
      const result = await JobPoller.poll(r.data.job_id);
      const count = result?.total_clips || result?.clips?.length || 0;
      UIController.showToast(`Generated ${count} short-form clips.`, "success");
    } catch (e) {
      UIController.showToast(`Shorts pipeline failed: ${e.message}`, "error");
    }
  } else {
    UIController.showToast(r.data?.error || "Shorts pipeline failed.", "error");
  }
}

async function initApp() {
  console.log(`[OpenCut UXP] v${VERSION} initialising...`);

  // Detect which port the backend is running on (5679-5689)
  BACKEND = await detectBackend();
  console.log(`[OpenCut UXP] Backend detected at: ${BACKEND}`);

  // Init UXP Premiere Pro bridge (non-blocking)
  PProBridge.init().then(() => {
    if (PProBridge.available()) {
      UIController.showToast("UXP Premiere Pro API available.", "success");
      const notice = document.getElementById("uxpTimelineNotice");
      if (notice) notice.style.display = "none";
    }
  });

  // Wire up all UI interactions
  UIController.initCollapsibles();
  bindSliders();
  bindEvents();
  initKeyboardShortcuts();

  // Initial connection check
  const alive = await checkConnection();
  if (alive) {
    _healthBackoff = HEALTH_CHECK_MS; // reset backoff on initial success
    await BackendClient.fetchCsrf();
    await loadLlmSettings();
    UIController.showToast("OpenCut backend connected.", "success");

    // Auto-connect WebSocket for real-time progress
    uxpWsConnect();

    // Scan project media so clip path inputs have autocomplete
    await scanProjectClips();

    // Start periodic backend media scan
    startMediaScanInterval();

    // One-time update check
    const ur = await BackendClient.get("/system/update-check");
    if (ur.ok && ur.data && ur.data.update_available) {
      UIController.showToast(
        `OpenCut v${ur.data.latest_version} available \u2014 visit GitHub to update`,
        "info",
        6000
      );
    }
  }

  // Re-scan project clips after every job completes (picks up auto-imported outputs)
  JobPoller.onJobFinished(() => {
    setTimeout(() => scanProjectClips(), 1500);
  });

  // Periodic soft re-scan to pick up newly imported media
  setInterval(async () => {
    if (!activeJobId) await scanProjectClips();
  }, 25000);

  // Re-scan when panel regains focus
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) scanProjectClips();
  });

  // Periodic health checks with exponential backoff on failure.
  // Schedule next check only after current one completes to prevent
  // overlapping async calls when backend is slow/down.
  function scheduleHealthCheck() {
    setTimeout(async () => {
      const ok = await checkConnection();
      if (ok) {
        // Reset backoff on success
        if (_healthBackoff !== HEALTH_CHECK_MS) {
          _healthBackoff = HEALTH_CHECK_MS;
          startMediaScanInterval(); // resume media scans on reconnect
        }
      } else {
        // Exponential backoff: double interval on failure, cap at HEALTH_MAX_MS
        _healthBackoff = Math.min(_healthBackoff * 2, HEALTH_MAX_MS);
        stopMediaScanInterval(); // pause media scans while disconnected
      }
      scheduleHealthCheck();
    }, _healthBackoff);
  }
  scheduleHealthCheck();

  // Clean up SSE connections on panel close/navigation
  window.addEventListener("beforeunload", () => {
    if (_activeSSE) { _activeSSE.close(); _activeSSE = null; }
    stopMediaScanInterval();
  });

  console.log("[OpenCut UXP] Ready.");
}

// Bootstrap
initApp().catch(err => {
  console.error("[OpenCut UXP] Init error:", err);
});
