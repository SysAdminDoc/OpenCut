import { escapeHtml as escapeHtmlValue, safeDomIdSegment } from "./uxp-utils.js";

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
const INLINE_CONFIRM_MS = 8000;
const SSE_AVAILABLE    = typeof EventSource !== "undefined";
const VERSION          = "1.32.0";
const PRIMARY_CLIP_INPUT_IDS = ["clipPathCut", "clipPathCaptions", "clipPathAudio", "clipPathVideo"];
const TABS_REQUIRING_SOURCE = new Set(["cut", "captions", "audio", "video"]);
const DELIVERABLE_LABELS = {
  vfx_sheet: "VFX Sheet",
  adr_list: "ADR List",
  music_cue_sheet: "Music Cue Sheet",
  asset_list: "Asset List",
};
const DELIVERABLE_BUTTON_IDS = {
  vfx_sheet: "delivVfxSheetBtn",
  adr_list: "delivAdrListBtn",
  music_cue_sheet: "delivMusicCueBtn",
  asset_list: "delivAssetListBtn",
};
const WORKSPACE_META   = {
  cut: {
    title: "Cut & Clean",
    subtitle: "Trim dead space, fillers, and rough pacing with a tighter review flow.",
    sourceIds: ["clipPathCut"],
  },
  captions: {
    title: "Captions",
    subtitle: "Transcribe, structure, and style subtitles without leaving the panel.",
    sourceIds: ["clipPathCaptions"],
  },
  audio: {
    title: "Audio",
    subtitle: "Denoise, normalize, loudness-match, and cut to rhythm from one focused surface.",
    sourceIds: ["clipPathAudio"],
  },
  video: {
    title: "Video",
    subtitle: "Shape the image, plan coverage, and build short-form versions with a cleaner finishing toolkit.",
    sourceIds: ["clipPathVideo"],
  },
  timeline: {
    title: "Timeline",
    subtitle: "Write changes back into Premiere, export interchange, and run batch production tasks with confidence.",
    sourceIds: ["clipPathCut", "clipPathVideo", "clipPathAudio"],
  },
  search: {
    title: "Search",
    subtitle: "Index the library, search footage, and trigger edit actions from natural-language commands.",
    sourceIds: ["clipPathVideo", "clipPathCaptions", "clipPathCut", "clipPathAudio"],
  },
  deliverables: {
    title: "Deliverables",
    subtitle: "Review sequence context and export reports, documents, and final handoff assets.",
    sourceIds: ["clipPathCut", "clipPathVideo", "clipPathAudio"],
  },
  settings: {
    title: "Settings",
    subtitle: "Tune engine routing, realtime connections, and shared defaults across the studio.",
    sourceIds: [],
  },
};
const WORKSPACE_GUIDES = {
  cut: {
    kicker: "Cut pass",
    title: "Build a cleaner first pass of the edit.",
    text: "Start with silence detection or filler cleanup, then review the suggested cut ranges before writing them back to the timeline.",
    action: "focus-runSilenceBtn",
    actionLabel: "Run Silence Detection",
  },
  captions: {
    kicker: "Transcript",
    title: "Turn the active shot into reviewable text.",
    text: "Transcribe first, then move into chapters, repeat detection, or timeline import once the wording looks right.",
    action: "focus-runTranscribeBtn",
    actionLabel: "Transcribe Clip",
  },
  audio: {
    kicker: "Audio pass",
    title: "Clean the voice bed before the rest of the finish.",
    text: "Start with denoise or normalization, then add rhythm markers if the cut needs to lock to music.",
    action: "focus-runDenoiseBtn",
    actionLabel: "Run Denoise",
  },
  video: {
    kicker: "Finishing",
    title: "Shape the frame and build derivative edits from one source.",
    text: "Use color, reframe, multicam, and short-form tools without repatching the same clip every time.",
    action: "focus-runColorMatchBtn",
    actionLabel: "Match Color",
  },
  timeline: {
    kicker: "Write-back",
    title: "Send approved changes back into Premiere with less friction.",
    text: "Apply the latest cuts or markers, then export OTIO, markers, or captions from the same review session.",
    action: "focus-applyTimelineCutsBtn",
    actionLabel: "Apply Latest Cuts",
  },
  search: {
    kicker: "Discovery",
    title: "Search the library, then reuse the result instantly.",
    text: "Index a folder once, search with natural language, and pull the matching shot back into the rest of the workspace.",
    action: "focus-searchQuery",
    actionLabel: "Start Search",
  },
  deliverables: {
    kicker: "Handoff",
    title: "Pull sequence context before generating delivery docs.",
    text: "Load the active Premiere sequence, choose an output folder, and generate reports with cleaner defaults.",
    action: "focus-loadSeqInfoBtn",
    actionLabel: "Load Sequence Info",
  },
  settings: {
    kicker: "Studio setup",
    title: "Keep routing, engines, and live services healthy.",
    text: "Refresh engine availability, verify the bridge, and make sure the panel is connected to the right backend.",
    action: "focus-uxpRefreshEnginesBtn",
    actionLabel: "Refresh Engines",
  },
};

function isTimeoutError(err) {
  const message = String(err?.message || err || "");
  return err?.name === "AbortError" || /timed out|abort/i.test(message);
}

async function copyTextToClipboard(text, { successLabel = "Output" } = {}) {
  const value = String(text || "");
  if (!value) return false;

  if (
    typeof navigator === "undefined" ||
    !navigator.clipboard ||
    typeof navigator.clipboard.writeText !== "function"
  ) {
    UIController.showToast("Clipboard is unavailable in this UXP environment. Copy the output manually.", "warning");
    return false;
  }

  try {
    await navigator.clipboard.writeText(value);
    UIController.showToast(`${successLabel} copied to clipboard.`, "success");
    return true;
  } catch (err) {
    UIController.showToast("Clipboard permission is unavailable or denied. Copy the output manually.", "warning");
    return false;
  }
}

async function fetchWithTimeout(url, opts = {}, timeoutMs = 120000) {
  const options = { ...opts };
  let timer = null;

  if (typeof AbortController !== "undefined" && !options.signal) {
    const controller = new AbortController();
    timer = setTimeout(() => controller.abort(), timeoutMs);
    options.signal = controller.signal;
    try {
      return await fetch(url, options);
    } finally {
      clearTimeout(timer);
    }
  }

  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => {
      reject(new Error(`Request timed out after ${timeoutMs}ms`));
    }, timeoutMs);
  });

  try {
    return await Promise.race([fetch(url, options), timeout]);
  } finally {
    clearTimeout(timer);
  }
}

async function detectBackend() {
  // Try ports 5679-5689 like CEP panel does
  for (let port = 5679; port <= BACKEND_MAX_PORT; port++) {
    const url = `http://127.0.0.1:${port}`;
    try {
      const resp = await fetchWithTimeout(`${url}/health`, {}, 500);
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
let _lastSequenceInfo = null;
let _lastDeliverableActivity = null;
let _lastIndexStats = { total_files: 0, total_segments: 0, index_size_bytes: 0 };
let _clearIndexConfirmUntil = 0;
let _clearIndexConfirmTimer = null;
let _lastCaptionsResult = null;
let _lastCutsInfo = null;
let _lastMarkersInfo = null;
let _lastTimelineAction = null;

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
  const UXP_DIRECT_HOST_ACTIONS = Object.freeze({
    ocPing: "ping",
    ocGetSequenceInfo: "getSequenceInfo",
    ocAddSequenceMarkers: "addMarkers",
    ocGetSequenceMarkers: "getSequenceMarkers",
    ocApplyClipKeyframes: "applyClipKeyframes",
    ocBatchRenameProjectItems: "batchRenameProjectItems",
    ocCreateSmartBins: "createSmartBins",
    ocGetProjectBins: "getProjectBins",
    ocExportSequenceRange: "exportSequenceRange",
    ocRemoveSequenceMarkers: "removeSequenceMarkers",
    ocUnrenameItems: "batchRenameProjectItems",
    ocRemoveImportedSequence: "removeImportedProjectItem",
    ocSetSequencePlayhead: "setSequencePlayhead",
    ocRemoveImportedItem: "removeImportedProjectItem",
  });
  const CEP_FALLBACK_HOST_ACTIONS = Object.freeze({
    ocAddNativeCaptionTrack: "No UXP caption-track creation API is available in the pinned parity catalogue.",
    ocQeReflect: "QE reflection is CEP-only and should be retired or replaced by documented APIs.",
  });

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

  function ping() {
    return { ok: true, result: "pong", source: "uxp" };
  }

  function _parseHostPayload(payload, fallback = {}) {
    if (payload == null) return fallback;
    if (typeof payload === "string") {
      const trimmed = payload.trim();
      if (!trimmed) return fallback;
      try {
        return JSON.parse(trimmed);
      } catch (e) {
        return { value: payload, parseError: e.message };
      }
    }
    return payload;
  }

  function _arrayPayload(payload, key) {
    const parsed = _parseHostPayload(payload, []);
    if (Array.isArray(parsed)) return parsed;
    if (key && Array.isArray(parsed?.[key])) return parsed[key];
    return [];
  }

  function _secondsToTicks(seconds) {
    return Math.round(Number(seconds || 0) * 254016000000);
  }

  function _tickTimeFromSeconds(seconds) {
    const numeric = Number(seconds || 0);
    if (ppro?.TickTime?.createWithSeconds) return ppro.TickTime.createWithSeconds(numeric);
    if (ppro?.TickTime?.createWithTicks) return ppro.TickTime.createWithTicks(String(_secondsToTicks(numeric)));
    return { seconds: numeric, ticks: String(_secondsToTicks(numeric)) };
  }

  async function _executeProjectActions(actions, undoString) {
    const usableActions = (actions || []).filter(Boolean);
    if (!usableActions.length) return false;
    const context = await _projectRoot();
    if (!context?.proj?.executeTransaction) return false;
    return Boolean(context.proj.executeTransaction((compoundAction) => {
      for (const action of usableActions) {
        compoundAction.addAction(action);
      }
    }, undoString));
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

  async function _markerInfo(marker) {
    const timeValue = await marker.getTime?.() ?? await marker.getStartTime?.() ?? marker.time ?? marker.start ?? 0;
    const nameValue = await marker.getName?.() ?? marker.name ?? "";
    const commentValue = await marker.getComment?.() ?? marker.comment ?? "";
    const colorValue = await marker.getColorIndex?.() ?? marker.colorIndex ?? null;
    const time = typeof timeValue === "object" && timeValue !== null
      ? Number(timeValue.seconds ?? (timeValue.ticks != null ? timeValue.ticks / 254016000000 : 0))
      : Number(timeValue || 0);
    return {
      time,
      ticks: _secondsToTicks(time),
      name: String(nameValue || ""),
      comment: String(commentValue || ""),
      colorIndex: colorValue,
    };
  }

  async function _sequenceMarkerObjects() {
    const seq = await getActiveSequence();
    if (!seq) return { ok: false, reason: "No active sequence or UXP API unavailable.", markers: [] };
    const markerList = await seq.getMarkerList?.();
    if (!markerList) return { ok: false, reason: "Active sequence does not expose marker APIs.", markers: [] };
    const markers = [];
    try {
      const rawMarkers = await markerList.getMarkers?.();
      if (rawMarkers && typeof rawMarkers[Symbol.iterator] === "function") {
        for (const marker of rawMarkers) {
          markers.push({ marker, info: await _markerInfo(marker) });
        }
      } else {
        let marker = await markerList.getFirstMarker?.();
        let guard = 0;
        while (marker && guard < 1000) {
          markers.push({ marker, info: await _markerInfo(marker) });
          marker = await markerList.getNextMarker?.(marker);
          guard += 1;
        }
      }
      return { ok: true, markerList, markers };
    } catch (e) {
      return { ok: false, reason: e.message, markers: [] };
    }
  }

  async function getSequenceMarkers() {
    const result = await _sequenceMarkerObjects();
    if (!result.ok) return result;
    return { ok: true, markers: result.markers.map(item => item.info), count: result.markers.length };
  }

  function _markerMatches(info, fingerprint) {
    if (!fingerprint) return false;
    const expectedTime = Number(fingerprint.time ?? fingerprint.start ?? fingerprint.seconds ?? NaN);
    if (Number.isFinite(expectedTime) && Math.abs(info.time - expectedTime) > 0.01) return false;
    const expectedName = fingerprint.name ?? fingerprint.label;
    if (expectedName != null && String(expectedName) !== info.name) return false;
    if (fingerprint.comment != null && String(fingerprint.comment) !== info.comment) return false;
    return Number.isFinite(expectedTime) || expectedName != null || fingerprint.comment != null;
  }

  async function removeSequenceMarkers(payload) {
    const fingerprints = _arrayPayload(payload, "fingerprints");
    if (!fingerprints.length) return { ok: false, reason: "No marker fingerprints supplied.", removed: 0 };
    const result = await _sequenceMarkerObjects();
    if (!result.ok) return result;
    let removed = 0;
    for (const item of result.markers) {
      if (!fingerprints.some(fp => _markerMatches(item.info, fp))) continue;
      try {
        if (result.markerList?.removeMarker) await result.markerList.removeMarker(item.marker);
        else if (item.marker?.delete) await item.marker.delete();
        else if (item.marker?.remove) await item.marker.remove();
        else continue;
        removed += 1;
      } catch (e) {
        console.warn("[PProBridge] removeSequenceMarkers skipped marker:", e.message);
      }
    }
    return { ok: true, removed };
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

  async function _walkProjectTree(parent, depth = 0, path = "") {
    if (!parent || depth > 20) return [];
    const items = [];
    try {
      const children = await parent.getItems?.();
      if (!children) return items;
      for (const child of children) {
        try {
          const name = await child.getName?.() ?? child.name ?? "";
          const childPath = path ? `${path}/${name}` : name;
          const isFolder = await child.isFolder?.() ?? false;
          const mediaPath = await child.getMediaPath?.() ?? "";
          const nodeId = await child.getNodeId?.() ?? child.nodeId ?? "";
          items.push({ item: child, name, path: childPath, mediaPath, nodeId, isFolder });
          if (isFolder) {
            items.push(...await _walkProjectTree(child, depth + 1, childPath));
          }
        } catch (_) { /* skip inaccessible project item */ }
      }
    } catch (_) { /* parent has no children */ }
    return items;
  }

  async function _projectRoot() {
    if (!available || !ppro) return null;
    const projList = await ppro.app.getProjectList();
    if (!projList || projList.length === 0) return null;
    const proj = projList[0];
    const root = await proj.getRootItem?.();
    return { proj, root };
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

  async function getProjectBins() {
    try {
      const context = await _projectRoot();
      if (!context?.root) return { ok: false, reason: "No open project", bins: [] };
      const tree = await _walkProjectTree(context.root);
      const bins = tree
        .filter(entry => entry.isFolder)
        .map(entry => ({ name: entry.name, path: entry.path, nodeId: entry.nodeId }));
      return { ok: true, bins, count: bins.length };
    } catch (e) {
      console.warn("[PProBridge] getProjectBins failed:", e.message);
      return { ok: false, reason: e.message, bins: [] };
    }
  }

  async function batchRenameProjectItems(payload) {
    const renames = _arrayPayload(payload, "renames");
    if (!renames.length) return { ok: false, reason: "No rename entries supplied.", renamed: 0 };
    try {
      const context = await _projectRoot();
      if (!context?.root) return { ok: false, reason: "No open project", renamed: 0 };
      const tree = await _walkProjectTree(context.root);
      let renamed = 0;
      for (const rename of renames) {
        const newName = rename.newName ?? rename.name ?? rename.to;
        if (!newName) continue;
        const target = tree.find(entry =>
          (rename.nodeId && entry.nodeId === rename.nodeId) ||
          (rename.path && (entry.mediaPath === rename.path || entry.path === rename.path)) ||
          (rename.oldName && entry.name === rename.oldName)
        );
        if (!target?.item?.setName) continue;
        await target.item.setName(String(newName));
        renamed += 1;
      }
      return { ok: true, renamed };
    } catch (e) {
      console.warn("[PProBridge] batchRenameProjectItems failed:", e.message);
      return { ok: false, reason: e.message, renamed: 0 };
    }
  }

  async function createSmartBins(payload) {
    const parsed = _parseHostPayload(payload, {});
    const bins = Array.isArray(parsed) ? parsed : (parsed.bins ?? parsed.rules ?? []);
    if (!Array.isArray(bins) || !bins.length) return { ok: false, reason: "No bin rules supplied.", created: 0 };
    try {
      const context = await _projectRoot();
      if (!context?.root) return { ok: false, reason: "No open project", created: 0 };
      let created = 0;
      for (const rule of bins) {
        const name = String(rule.name ?? rule.bin ?? rule.label ?? "").trim();
        if (!name) continue;
        if (context.root.createBin) await context.root.createBin(name);
        else if (context.proj.createBin) await context.proj.createBin(name);
        else return { ok: false, reason: "Project does not expose bin creation.", created };
        created += 1;
      }
      return { ok: true, created };
    } catch (e) {
      console.warn("[PProBridge] createSmartBins failed:", e.message);
      return { ok: false, reason: e.message, created: 0 };
    }
  }

  async function removeImportedProjectItem(payload) {
    const parsed = _parseHostPayload(payload, {});
    try {
      const context = await _projectRoot();
      if (!context?.root) return { ok: false, reason: "No open project", removed: 0 };
      const tree = await _walkProjectTree(context.root);
      const target = tree.find(entry =>
        (parsed.nodeId && entry.nodeId === parsed.nodeId) ||
        (parsed.path && (entry.mediaPath === parsed.path || entry.path === parsed.path)) ||
        (parsed.name && entry.name === parsed.name)
      );
      if (!target?.item) return { ok: false, reason: "Project item not found.", removed: 0 };
      if (target.item.delete) await target.item.delete();
      else if (target.item.remove) await target.item.remove();
      else if (context.proj.deleteItem) await context.proj.deleteItem(target.item);
      else return { ok: false, reason: "Project item does not expose delete/remove.", removed: 0 };
      return { ok: true, removed: 1 };
    } catch (e) {
      console.warn("[PProBridge] removeImportedProjectItem failed:", e.message);
      return { ok: false, reason: e.message, removed: 0 };
    }
  }

  async function _clipProjectItemFromPayload(payload) {
    const parsed = _parseHostPayload(payload, {});
    const context = await _projectRoot();
    if (!context?.root) return null;
    const tree = await _walkProjectTree(context.root);
    const target = tree.find(entry =>
      !entry.isFolder && (
        (parsed.nodeId && entry.nodeId === parsed.nodeId) ||
        (parsed.path && (entry.mediaPath === parsed.path || entry.path === parsed.path)) ||
        (parsed.name && entry.name === parsed.name)
      )
    );
    if (!target?.item) return null;
    try {
      return ppro?.ClipProjectItem?.cast ? ppro.ClipProjectItem.cast(target.item) : target.item;
    } catch (e) {
      console.warn("[PProBridge] ClipProjectItem.cast failed:", e.message);
      return target.item;
    }
  }

  function querySupportedTranscriptLanguages() {
    if (!available || !ppro?.Transcript?.querySupportedLanguages) {
      return { ok: false, reason: "Premiere Transcript APIs are unavailable.", languages: [] };
    }
    try {
      const languages = ppro.Transcript.querySupportedLanguages();
      return { ok: true, languages: Array.isArray(languages) ? languages : [] };
    } catch (e) {
      console.warn("[PProBridge] querySupportedTranscriptLanguages failed:", e.message);
      return { ok: false, reason: e.message, languages: [] };
    }
  }

  async function getTranscriptState(payload) {
    if (!available || !ppro?.Transcript?.hasTranscript) {
      return { ok: false, reason: "Premiere Transcript APIs are unavailable.", hasTranscript: false };
    }
    const parsed = _parseHostPayload(payload, {});
    const clipProjectItem = await _clipProjectItemFromPayload(parsed);
    if (!clipProjectItem) return { ok: false, reason: "Project item not found.", hasTranscript: false };
    try {
      const hasTranscript = Boolean(ppro.Transcript.hasTranscript(clipProjectItem));
      let transcriptJson = null;
      if (hasTranscript && parsed.includeJson === true && ppro.Transcript.exportToJSON) {
        transcriptJson = await ppro.Transcript.exportToJSON(clipProjectItem);
      }
      return { ok: true, hasTranscript, transcriptJson };
    } catch (e) {
      console.warn("[PProBridge] getTranscriptState failed:", e.message);
      return { ok: false, reason: e.message, hasTranscript: false };
    }
  }

  async function getObjectMaskState(payload) {
    if (!available || !ppro?.ObjectMaskUtils?.hasObjectMask) {
      return { ok: false, reason: "Premiere Object Mask APIs are unavailable.", hasObjectMask: false };
    }
    const parsed = _parseHostPayload(payload, {});
    const requestedTarget = String(parsed.target ?? parsed.scope ?? "sequence").toLowerCase();
    const targetType = requestedTarget === "project" ? "project" : "sequence";
    try {
      let target = null;
      if (targetType === "project") {
        const context = await _projectRoot();
        target = context?.proj ?? null;
      } else {
        target = await getActiveSequence();
      }
      if (!target) return { ok: false, reason: `No active ${targetType}.`, target: targetType, hasObjectMask: false };
      const hasObjectMask = Boolean(ppro.ObjectMaskUtils.hasObjectMask(target));
      return { ok: true, target: targetType, hasObjectMask };
    } catch (e) {
      console.warn("[PProBridge] getObjectMaskState failed:", e.message);
      return { ok: false, reason: e.message, target: targetType, hasObjectMask: false };
    }
  }

  async function setSequencePlayhead(payload) {
    const parsed = _parseHostPayload(payload, {});
    const seconds = Number(parsed.seconds ?? parsed.time ?? parsed.value ?? payload ?? NaN);
    if (!Number.isFinite(seconds)) return { ok: false, reason: "A numeric time in seconds is required." };
    const seq = await getActiveSequence();
    if (!seq) return { ok: false, reason: "No active sequence or UXP API unavailable." };
    try {
      if (seq.setPlayerPosition) await seq.setPlayerPosition(_secondsToTicks(seconds));
      else if (seq.setPlayheadPosition) await seq.setPlayheadPosition(_secondsToTicks(seconds));
      else if (ppro?.SourceMonitor?.setPosition) await ppro.SourceMonitor.setPosition(seconds);
      else return { ok: false, reason: "No supported playhead positioning API is available." };
      return { ok: true, seconds };
    } catch (e) {
      console.warn("[PProBridge] setSequencePlayhead failed:", e.message);
      return { ok: false, reason: e.message };
    }
  }

  async function applyClipKeyframes() {
    return {
      ok: false,
      reason: "UXP keyframe dispatch is registered, but clip-property targeting remains gated on F267 UDT coverage.",
      requiresUdt: true,
    };
  }

  async function createSubsequenceFromRange(payload) {
    const parsed = _parseHostPayload(payload, {});
    const seq = await getActiveSequence();
    if (!seq) return { ok: false, reason: "No active sequence or UXP API unavailable." };
    if (!seq.createSubsequence) {
      return { ok: false, reason: "Sequence.createSubsequence is unavailable in this Premiere UXP runtime." };
    }
    const startSeconds = Number(parsed.startSeconds ?? parsed.start ?? 0);
    const endSeconds = Number(parsed.endSeconds ?? parsed.end ?? 0);
    if (!Number.isFinite(startSeconds) || !Number.isFinite(endSeconds) || endSeconds <= startSeconds) {
      return { ok: false, reason: "A valid start/end range in seconds is required." };
    }
    const ignoreTrackTargeting = parsed.ignoreTrackTargeting !== false;
    const originalIn = await seq.getInPoint?.();
    const originalOut = await seq.getOutPoint?.();
    const startTick = _tickTimeFromSeconds(startSeconds);
    const endTick = _tickTimeFromSeconds(endSeconds);
    const setActions = [
      seq.createSetInPointAction?.(startTick),
      seq.createSetOutPointAction?.(endTick),
    ];
    const rangeSet = await _executeProjectActions(setActions, "OpenCut set subsequence range");
    if (!rangeSet) return { ok: false, reason: "Premiere did not accept the sequence in/out range actions." };

    try {
      const subsequence = await seq.createSubsequence(ignoreTrackTargeting);
      const name = await subsequence?.getName?.() ?? "";
      return {
        ok: true,
        sequence: subsequence,
        sequenceName: name,
        range: { start: startSeconds, end: endSeconds },
        ignoreTrackTargeting,
      };
    } finally {
      const restoreActions = [
        originalIn ? seq.createSetInPointAction?.(originalIn) : null,
        originalOut ? seq.createSetOutPointAction?.(originalOut) : null,
      ];
      await _executeProjectActions(restoreActions, "OpenCut restore sequence range");
    }
  }

  function _encoderManager() {
    const manager = ppro?.EncoderManager?.getManager?.();
    return manager || null;
  }

  function _encoderExportType(queueToAme) {
    if (queueToAme) {
      return ppro?.Constants?.ExportType?.QUEUE_TO_AME ?? ppro?.EncoderManager?.EXPORT_QUEUE_TO_AME;
    }
    return ppro?.Constants?.ExportType?.IMMEDIATELY ?? ppro?.EncoderManager?.EXPORT_IMMEDIATELY;
  }

  async function exportSubsequenceWithEncoder(sequence, payload) {
    const parsed = _parseHostPayload(payload, {});
    const manager = _encoderManager();
    if (!manager) return { ok: false, reason: "Premiere EncoderManager is unavailable in this UXP runtime." };
    if (!sequence) return { ok: false, reason: "No subsequence was created for export." };
    const outputPath = String(parsed.outputPath ?? parsed.path ?? "").trim();
    if (!outputPath) return { ok: false, reason: "An output path is required for UXP encoder export." };

    const queueToAme = parsed.queueToAme !== false && parsed.exportType !== "immediate";
    if (queueToAme && manager.isAMEInstalled === false) {
      return { ok: false, reason: "Adobe Media Encoder is not installed.", outputPath };
    }

    let encoderLaunched = false;
    if (queueToAme && manager.launchEncoder) {
      encoderLaunched = Boolean(await manager.launchEncoder());
    }

    const presetFile = String(parsed.presetFile ?? parsed.presetPath ?? "").trim() || undefined;
    const exportType = _encoderExportType(queueToAme);
    const exportFull = parsed.exportFull !== false;
    const queued = await manager.exportSequence(sequence, exportType, outputPath, presetFile, exportFull);

    let batchStarted = false;
    if (queueToAme && parsed.startBatch !== false && manager.startBatchEncode) {
      batchStarted = Boolean(await manager.startBatchEncode());
    }

    return {
      ok: Boolean(queued),
      outputPath,
      presetFile: presetFile ?? "",
      exportFull,
      queueToAme,
      encoderLaunched,
      batchStarted,
    };
  }

  function _hasOwn(obj, key) {
    return Object.prototype.hasOwnProperty.call(obj ?? {}, key);
  }

  function _aafAudioFormat(value) {
    const normalized = String(value ?? "wav").toLowerCase();
    if (normalized === "aiff") {
      return ppro?.Constants?.AAFExportAudioFormat?.AIFF ?? ppro?.ProjectConverter?.AAF_EXPORT_AUDIO_FORMAT_AIFF ?? 0;
    }
    return ppro?.Constants?.AAFExportAudioFormat?.WAV ?? ppro?.ProjectConverter?.AAF_EXPORT_AUDIO_FORMAT_WAV ?? 1;
  }

  function _createAafExportOptions(settings = {}) {
    const factory = ppro?.AAFExportOptions;
    if (!factory) return null;
    let options = null;
    try {
      options = new factory();
    } catch (_) {
      try { options = factory(); } catch (_) { options = null; }
    }
    if (!options) return null;

    const booleanSetters = [
      ["mixdownVideo", "setMixdownVideo"],
      ["explodeToMono", "setExplodeToMono"],
      ["embedAudio", "setEmbedAudio"],
      ["trimSources", "setTrimSources"],
      ["renderAudioEffects", "setRenderAudioEffects"],
      ["interleaveWithoutEffects", "setInterleaveWithoutEffects"],
      ["preserveParentFolder", "setPreserveParentFolder"],
    ];
    for (const [key, setter] of booleanSetters) {
      if (_hasOwn(settings, key) && options[setter]) options[setter](Boolean(settings[key]));
    }

    const numericSetters = [
      ["sampleRate", "setSampleRate"],
      ["bitsPerSample", "setBitsPerSample"],
      ["handleFrames", "setHandleFrames"],
    ];
    for (const [key, setter] of numericSetters) {
      const value = Number(settings[key]);
      if (_hasOwn(settings, key) && Number.isFinite(value) && options[setter]) options[setter](value);
    }

    if (_hasOwn(settings, "audioFileFormat") && options.setAudioFileFormat) {
      options.setAudioFileFormat(_aafAudioFormat(settings.audioFileFormat));
    }
    const presetPath = String(settings.videoMixdownPresetPath ?? settings.videoPresetPath ?? "").trim();
    if (presetPath && options.setVideoMixdownPresetPath) options.setVideoMixdownPresetPath(presetPath);
    return options;
  }

  async function exportAafSequence(payload) {
    if (!available || !ppro?.ProjectConverter?.exportAAF) {
      return { ok: false, reason: "Premiere ProjectConverter.exportAAF is unavailable in this UXP runtime." };
    }
    const parsed = _parseHostPayload(payload, {});
    const outputPath = String(parsed.outputPath ?? parsed.path ?? parsed.filePath ?? "").trim();
    if (!outputPath) return { ok: false, reason: "An output path is required for UXP AAF export." };
    const seq = await getActiveSequence();
    if (!seq) return { ok: false, reason: "No active sequence or UXP API unavailable." };
    try {
      const optionSettings = parsed.aafOptions ?? parsed.options ?? parsed;
      const aafOptions = _createAafExportOptions(optionSettings);
      const exported = await ppro.ProjectConverter.exportAAF(seq, outputPath, aafOptions || undefined);
      const sequenceName = await seq.getName?.() ?? seq.name ?? "";
      return { ok: Boolean(exported), outputPath, sequenceName, optionsApplied: Boolean(aafOptions) };
    } catch (e) {
      console.warn("[PProBridge] exportAafSequence failed:", e.message);
      return { ok: false, reason: e.message, outputPath };
    }
  }

  async function exportSequenceRange(payload) {
    const parsed = _parseHostPayload(payload, {});
    const subsequence = await createSubsequenceFromRange(parsed);
    if (!subsequence.ok) return subsequence;
    const exportResult = await exportSubsequenceWithEncoder(subsequence.sequence, parsed);
    return {
      ...exportResult,
      subsequence: {
        sequenceName: subsequence.sequenceName,
        range: subsequence.range,
        ignoreTrackTargeting: subsequence.ignoreTrackTargeting,
      },
    };
  }

  function hostActionStatus() {
    return {
      direct: Object.keys(UXP_DIRECT_HOST_ACTIONS),
      partial: ["ocApplySequenceCuts"],
      differentMechanism: ["ocEmitPingEvent"],
      cepFallback: Object.keys(CEP_FALLBACK_HOST_ACTIONS),
      directCount: Object.keys(UXP_DIRECT_HOST_ACTIONS).length,
      cepFallbackCount: Object.keys(CEP_FALLBACK_HOST_ACTIONS).length,
    };
  }

  async function executeHostAction(action, payload = {}) {
    switch (action) {
      case "ocPing": return ping();
      case "ocGetSequenceInfo": return { ok: true, data: await getSequenceInfo() };
      case "ocAddSequenceMarkers": return addMarkers(_arrayPayload(payload, "markers"));
      case "ocGetSequenceMarkers": return getSequenceMarkers();
      case "ocApplySequenceCuts": return applyCuts(_arrayPayload(payload, "cuts"));
      case "ocApplyClipKeyframes": return applyClipKeyframes(payload);
      case "ocBatchRenameProjectItems": return batchRenameProjectItems(payload);
      case "ocCreateSmartBins": return createSmartBins(payload);
      case "ocGetProjectBins": return getProjectBins();
      case "ocExportSequenceRange": return exportSequenceRange(payload);
      case "ocRemoveSequenceMarkers": return removeSequenceMarkers(payload);
      case "ocUnrenameItems": return batchRenameProjectItems(payload);
      case "ocRemoveImportedSequence": return removeImportedProjectItem(payload);
      case "ocSetSequencePlayhead": return setSequencePlayhead(payload);
      case "ocRemoveImportedItem": return removeImportedProjectItem(payload);
      case "ocEmitPingEvent": return { ok: true, emitted: true, tag: _parseHostPayload(payload, {}).tag ?? "" };
      case "ocAddNativeCaptionTrack":
      case "ocQeReflect":
        return { ok: false, cepFallback: true, reason: CEP_FALLBACK_HOST_ACTIONS[action] };
      default:
        return { ok: false, reason: `Unknown host action: ${action}` };
    }
  }

  return {
    init,
    available: () => available,
    getActiveSequence,
    getSequenceInfo,
    addMarkers,
    getSequenceMarkers,
    removeSequenceMarkers,
    applyCuts,
    invalidateCache,
    getProjectItems,
    getProjectBins,
    getSelectedClips,
    importFiles,
    batchRenameProjectItems,
    createSmartBins,
    removeImportedProjectItem,
    querySupportedTranscriptLanguages,
    getTranscriptState,
    getObjectMaskState,
    setSequencePlayhead,
    createSubsequenceFromRange,
    exportSubsequenceWithEncoder,
    exportAafSequence,
    executeHostAction,
    hostActionStatus,
  };
})();

if (typeof window !== "undefined") {
  window.OpenCutUXPHost = Object.freeze({
    executeHostAction: (action, payload) => PProBridge.executeHostAction(action, payload),
    getHostActionStatus: () => PProBridge.hostActionStatus(),
    querySupportedTranscriptLanguages: () => PProBridge.querySupportedTranscriptLanguages(),
    getTranscriptState: (payload) => PProBridge.getTranscriptState(payload),
    getObjectMaskState: (payload) => PProBridge.getObjectMaskState(payload),
    exportAafSequence: (payload) => PProBridge.exportAafSequence(payload),
  });
}

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
  async function _doFetch(method, endpoint, body) {
    const url = BACKEND + endpoint;
    const headers = { "Content-Type": "application/json" };
    if (csrfToken) headers["X-OpenCut-Token"] = csrfToken;

    // 120s default — long enough for synchronous routes (CSRF, health,
    // /info), short enough that a hung backend doesn't pin a button
    // forever. Async job submission returns within seconds; the actual
    // long-running work is observed via SSE/polling, not held in this
    // request.
    const opts = { method, headers };
    if (body && method !== "GET") opts.body = JSON.stringify(body);

    let resp;
    try {
      resp = await fetchWithTimeout(url, opts, 120000);
    } catch (err) {
      if (isTimeoutError(err)) {
        throw new Error("Backend request timed out. Check that OpenCut Server is still running, then try again.");
      }
      throw err;
    }

    // Refresh CSRF token if provided in response headers
    const newToken = resp.headers.get("X-OpenCut-Token");
    if (newToken) csrfToken = newToken;

    let data;
    const ct = resp.headers.get("Content-Type") || "";
    if (ct.includes("application/json")) {
      try {
        data = await resp.json();
      } catch (_) {
        // Server claimed JSON but produced garbage — surface as a string
        // rather than throwing inside the wrapper, which would skip the
        // status-code branch and hide whether this was a 4xx/5xx response.
        data = null;
      }
    } else {
      data = await resp.text();
    }
    return { resp, data };
  }

  async function call(method, endpoint, body = null) {
    try {
      let { resp, data } = await _doFetch(method, endpoint, body);

      // Server-restart recovery: /health issues a new CSRF token on startup,
      // so a stale in-memory token returns 403. Refresh and retry exactly
      // once before surfacing the error to the user.
      if (resp.status === 403 && endpoint !== "/health") {
        await fetchCsrf();
        ({ resp, data } = await _doFetch(method, endpoint, body));
      }

      if (!resp.ok) {
        const msg = (data && data.error) ? data.error : `HTTP ${resp.status}`;
        return { ok: false, error: msg, status: resp.status, data };
      }
      return { ok: true, data, status: resp.status };
    } catch (err) {
      const fallback = isTimeoutError(err)
        ? "Backend request timed out. Check that OpenCut Server is still running, then try again."
        : "Network error";
      return { ok: false, error: err.message ?? fallback };
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
      if (el) el.classList.toggle("oc-hidden", available);
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
        const msg    = job.message ?? job.msg ?? "Processing…";

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

  // Hard cap: 1200ms × 3000 = 60 minutes of polling. If a job is still
  // "running" after an hour the panel gives up rather than spinning closures
  // forever. The CEP panel has the same cap; UXP previously had none.
  const MAX_POLL_ATTEMPTS = 3000;

  async function pollJob(jobId, onProgress, onComplete, onError, attempt = 0) {
    const r = await BackendClient.get(`/status/${jobId}`);
    if (!r.ok) {
      onError(r.error ?? "Polling error");
      activeJobId = null;
      return;
    }

    const job = r.data;
    const status  = job.status ?? "running";
    const pct     = typeof job.progress === "number" ? job.progress : 0;
    const msg     = job.message ?? job.msg ?? "Processing…";

    onProgress(pct, msg);

    if (status === "done" || status === "complete" || status === "success") {
      activeJobId = null;
      onComplete(job.result ?? job);
      _fireCompletionHooks();
      return;
    }

    // 'interrupted' is the terminal state set on server startup for jobs
    // that were running when the server died; treat it like an error so
    // the panel doesn't poll forever for progress that will never arrive.
    if (status === "error" || status === "failed" || status === "cancelled" || status === "interrupted") {
      activeJobId = null;
      onError(job.error ?? job.message ?? "Job failed");
      _fireCompletionHooks();
      return;
    }

    if (attempt >= MAX_POLL_ATTEMPTS) {
      activeJobId = null;
      onError("Polling timed out — the job is still running on the server.");
      _fireCompletionHooks();
      return;
    }

    // Still running — schedule next poll
    setTimeout(() => {
      if (activeJobId === jobId) {
        pollJob(jobId, onProgress, onComplete, onError, attempt + 1);
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
      btn.tabIndex = active ? 0 : -1;
    });
    document.querySelectorAll(".oc-tab-panel").forEach(panel => {
      const active = panel.id === `tab-${tabId}`;
      panel.classList.toggle("active", active);
      panel.hidden = !active;
      panel.setAttribute("aria-hidden", active ? "false" : "true");
    });
    updateWorkspaceOverview(tabId);
    setStatus(`${(WORKSPACE_META[tabId] && WORKSPACE_META[tabId].title) || (tabId.charAt(0).toUpperCase() + tabId.slice(1))} workspace`);
  }

  // ── Processing banner ──
  function showProcessing(msg = "Processing…") {
    const banner = document.getElementById("processingBanner");
    if (banner) banner.classList.remove("hidden");
    document.getElementById("mainContent")?.setAttribute("aria-busy", "true");
    setProcessingMsg(msg);
    setProgress(0);
    startElapsedTimer();
  }

  function hideProcessing() {
    const banner = document.getElementById("processingBanner");
    if (banner) banner.classList.add("hidden");
    document.getElementById("mainContent")?.setAttribute("aria-busy", "false");
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

  function inferStatusTone(msg) {
    const text = String(msg || "").toLowerCase();
    if (!text) return "neutral";
    if (/(error|failed|offline|unavailable|timed out|timeout|could not|stopped)/.test(text)) return "error";
    if (/(connecting|running|processing|loading|refreshing|starting|detecting|indexing|scanning)/.test(text)) return "working";
    if (/(online|connected|saved|ready|done|complete|loaded|updated|synced)/.test(text)) return "success";
    return "neutral";
  }

  // ── Status bar ──
  function setStatus(msg, tone) {
    const el = document.getElementById("statusText");
    if (el) el.textContent = msg || "";
    const bar = document.getElementById("statusBar");
    if (bar) {
      bar.dataset.state = tone || inferStatusTone(msg);
      bar.title = msg || "";
    }
  }

  function setStatusRight(msg) {
    const el = document.getElementById("statusRight");
    if (el) {
      el.textContent = msg || "";
      el.classList.toggle("is-empty", !msg);
    }
  }

  // ── Connection indicator ──
  function setConnection(state) {
    // state: "connected" | "connecting" | "disconnected"
    const dot   = document.getElementById("connDot");
    const label = document.getElementById("connLabel");
    const status = document.getElementById("connectionStatus");
    const statusBar = document.getElementById("statusBar");
    if (!dot || !label) return;
    dot.className = `oc-conn-dot ${state}`;
    if (status) status.dataset.state = state;
    if (statusBar) statusBar.dataset.connection = state;
    const labels = { connected: "Online", connecting: "Connecting…", disconnected: "Offline" };
    label.textContent = labels[state] ?? state;
    updateWorkspaceOverview();
  }

  // ── Toast notifications ──
  function getToastHeading(type) {
    switch (type) {
      case "success": return "Ready";
      case "warning": return "Needs attention";
      case "error": return "Action failed";
      default: return "OpenCut";
    }
  }

  function getToastDuration(type, explicitDuration) {
    if (typeof explicitDuration === "number") return explicitDuration;
    if (type === "error") return 0;
    if (type === "warning") return 5600;
    return 4000;
  }

  function showToast(message, type = "info", duration) {
    const area = document.getElementById("toastArea");
    if (!area) return;

    const payload = (message && typeof message === "object")
      ? message
      : { message };
    const tone = payload.type || type || "info";
    const text = String(payload.message ?? payload.text ?? "").trim()
      || String(message ?? "").trim();
    const title = String(payload.title ?? getToastHeading(tone)).trim() || getToastHeading(tone);
    const detail = String(payload.detail ?? "").trim();

    const icons = {
      success: `<svg width="13" height="13" viewBox="0 0 16 16" fill="currentColor"><path d="M13.854 3.646a.5.5 0 010 .708l-7 7a.5.5 0 01-.708 0l-3.5-3.5a.5.5 0 11.708-.708L6.5 10.293l6.646-6.647a.5.5 0 01.708 0z"/></svg>`,
      error:   `<svg width="13" height="13" viewBox="0 0 16 16" fill="currentColor"><path d="M4.646 4.646a.5.5 0 000 .708L7.293 8l-2.647 2.646a.5.5 0 00.708.708L8 8.707l2.646 2.647a.5.5 0 00.708-.708L8.707 8l2.647-2.646a.5.5 0 00-.708-.708L8 7.293 5.354 4.646a.5.5 0 00-.708 0z"/></svg>`,
      warning: `<svg width="13" height="13" viewBox="0 0 16 16" fill="currentColor"><path d="M8 1a7 7 0 100 14A7 7 0 008 1zm0 3a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 018 4zm0 8a1 1 0 110-2 1 1 0 010 2z"/></svg>`,
      info:    `<svg width="13" height="13" viewBox="0 0 16 16" fill="currentColor"><path d="M8 1a7 7 0 100 14A7 7 0 008 1zm.93 6.588l-2.29.287-.082.38.45.083c.294.07.352.176.288.469l-.738 3.468c-.194.897.105 1.319.808 1.319.545 0 1.178-.252 1.465-.598l.088-.416c-.2.176-.492.246-.686.246-.275 0-.375-.193-.304-.533L8.93 7.588z"/></svg>`,
    };

    const toast = document.createElement("div");
    toast.className = `oc-toast ${tone}`;
    toast.dataset.state = tone;
    toast.setAttribute("role", tone === "error" ? "alert" : "status");
    toast.setAttribute("aria-live", tone === "error" ? "assertive" : "polite");
    toast.innerHTML = `
      <span class="oc-toast-icon" aria-hidden="true">${icons[tone] ?? icons.info}</span>
      <span class="oc-toast-content">
        <span class="oc-toast-title">${escapeHtml(title)}</span>
        <span class="oc-toast-msg">${escapeHtml(text || title)}</span>
        ${detail ? `<span class="oc-toast-detail">${escapeHtml(detail)}</span>` : ""}
      </span>
      <button type="button" class="oc-toast-dismiss" aria-label="Dismiss notification">&times;</button>`;

    area.appendChild(toast);

    const dismiss = () => {
      if (toast.dataset.closing === "true") return;
      toast.dataset.closing = "true";
      toast.classList.add("fade-out");
      setTimeout(() => toast.remove(), 320);
    };
    toast.querySelector(".oc-toast-dismiss")?.addEventListener("click", dismiss);

    const resolvedDuration = getToastDuration(
      tone,
      payload.duration ?? (arguments.length >= 3 ? duration : undefined)
    );
    if (resolvedDuration > 0) setTimeout(dismiss, resolvedDuration);
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
      if (header.dataset.collapsibleBound === "true") return;
      const targetId = header.dataset.target;
      const initialBody = targetId ? document.getElementById(targetId) : null;
      header.setAttribute("role", "button");
      header.tabIndex = 0;
      if (targetId) header.setAttribute("aria-controls", targetId);
      header.setAttribute("aria-expanded", initialBody?.classList.contains("collapsed") ? "false" : "true");

      const toggle = () => {
        const targetId = header.dataset.target;
        const body = document.getElementById(targetId);
        if (!body) return;
        const collapsed = body.classList.toggle("collapsed");
        header.classList.toggle("collapsed", collapsed);
        header.setAttribute("aria-expanded", collapsed ? "false" : "true");
      };

      header.addEventListener("click", toggle);
      header.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          toggle();
        }
      });
      header.dataset.collapsibleBound = "true";
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
    return escapeHtmlValue(str);
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
      const nextPath = entry.nativePath ?? entry.name ?? "";
      const input = document.getElementById(inputId);
      if (input) {
        input.value = nextPath;
        input.dispatchEvent(new Event("input", { bubbles: true }));
        input.dispatchEvent(new Event("change", { bubbles: true }));
      }
      if (PRIMARY_CLIP_INPUT_IDS.indexOf(inputId) !== -1) {
        setWorkspaceClip(nextPath, { originId: inputId });
      }
      return nextPath || null;
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
      if (inputId === "delivOutputDir") updateDeliverablesSummary();
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
let _workspaceClipPath = "";
let _syncingWorkspaceClip = false;

function getWorkspaceTabId(tabId) {
  if (tabId) return tabId;
  return document.querySelector(".oc-tab.active")?.dataset.tab ?? "cut";
}

function getPrimarySourceInputId(tabId) {
  const activeTab = getWorkspaceTabId(tabId);
  return WORKSPACE_META[activeTab]?.sourceIds?.[0] ?? PRIMARY_CLIP_INPUT_IDS[0];
}

function getWorkspaceSource(tabId) {
  if (_workspaceClipPath) return _workspaceClipPath;
  const activeTab = getWorkspaceTabId(tabId);
  const meta = WORKSPACE_META[activeTab] || {};
  const preferredIds = meta.sourceIds || [];
  const fallbackIds = PRIMARY_CLIP_INPUT_IDS;
  const orderedIds = preferredIds.concat(fallbackIds.filter(id => preferredIds.indexOf(id) === -1));

  for (const id of orderedIds) {
    const value = document.getElementById(id)?.value?.trim();
    if (value) return value;
  }
  return "";
}

function formatWorkspaceSource(pathValue) {
  if (!pathValue) return "No source selected";
  const normalized = pathValue.replace(/\\/g, "/");
  const parts = normalized.split("/");
  return parts[parts.length - 1] || pathValue;
}

function getSelectLabel(selectId, fallback = "") {
  const select = document.getElementById(selectId);
  if (!select || !select.options || select.selectedIndex < 0) return fallback;
  const option = select.options[select.selectedIndex];
  return (option?.textContent || option?.label || option?.value || fallback).trim();
}

function getSearchResultPath(item) {
  const candidate = [
    item?.path,
    item?.file,
    item?.filepath,
    item?.source_path,
    item?.clip_path,
    item?.asset_path,
  ].find((value) => typeof value === "string" && value.trim());
  return candidate ? candidate.trim() : "";
}

function getSearchResultKindLabel(item) {
  const raw = String(item?.kind ?? item?.type ?? item?.modality ?? item?.match_type ?? "").toLowerCase();
  if (/frame|image|visual/.test(raw)) return "Visual";
  if (/audio|speech|voice/.test(raw)) return "Audio";
  if (/text|transcript|caption|subtitle|segment/.test(raw)) return "Transcript";
  if (typeof item?.text === "string" || typeof item?.transcript === "string" || typeof item?.snippet === "string") {
    return "Transcript";
  }
  return "Library";
}

function getSearchResultTimeLabel(item) {
  const start = Number(
    item?.start
    ?? item?.start_time
    ?? item?.segment_start
    ?? item?.t0
    ?? item?.timestamp_start
  );
  const end = Number(
    item?.end
    ?? item?.end_time
    ?? item?.segment_end
    ?? item?.t1
    ?? item?.timestamp_end
  );
  if (Number.isFinite(start) && Number.isFinite(end) && end >= start) {
    return `${formatTimecode(start)} to ${formatTimecode(end)}`;
  }
  if (Number.isFinite(start)) return `From ${formatTimecode(start)}`;
  return "";
}

function getSearchResultPreview(item) {
  const preview = [
    item?.preview,
    item?.snippet,
    item?.segment_text,
    item?.text,
    item?.transcript,
    item?.description,
    item?.reason,
    item?.caption,
  ].find((value) => typeof value === "string" && value.trim());
  if (!preview) return "";
  const compact = preview.replace(/\s+/g, " ").trim();
  return compact.length > 156 ? `${compact.slice(0, 153)}…` : compact;
}

function getSearchResultScoreLabel(item, index) {
  const raw = Number(item?.score ?? item?.confidence ?? item?.similarity);
  if (Number.isFinite(raw)) {
    const pct = raw > 1 ? Math.round(raw) : Math.round(raw * 100);
    return `${pct}% match`;
  }
  return index === 0 ? "Top match" : `Match ${index + 1}`;
}

function buildSearchResultCard(item, index) {
  const path = getSearchResultPath(item);
  return {
    index,
    path,
    label: formatWorkspaceSource(path || `Result ${index + 1}`),
    kindLabel: getSearchResultKindLabel(item),
    timeLabel: getSearchResultTimeLabel(item),
    preview: getSearchResultPreview(item),
    scoreLabel: getSearchResultScoreLabel(item, index),
  };
}

function setCaptionsStatus(message, state = "idle", title) {
  const line = document.getElementById("captionsStatusLine");
  if (!line) return;
  line.textContent = message;
  line.dataset.state = state;
  line.title = title || message;
}

function setCaptionsSessionState(pillText, pillState, statusMessage, statusState = pillState, title) {
  setStatusPill("captionsSessionPill", pillText, pillState, title || statusMessage || pillText);
  setCaptionsStatus(statusMessage || pillText, statusState, title || statusMessage || pillText);
}

function syncCaptionsActionButtons() {
  const backendOnline = document.getElementById("connLabel")?.textContent?.trim() === "Online";
  const sourcePath = document.getElementById("clipPathCaptions")?.value?.trim() || getWorkspaceSource("captions");
  const hasSource = !!sourcePath;

  ["runTranscribeBtn", "runChaptersBtn", "runRepeatBtn"].forEach((id) => {
    const btn = document.getElementById(id);
    if (!btn || btn.classList.contains("loading")) return;
    btn.disabled = !backendOnline || !hasSource;
  });

  const copyBtn = document.getElementById("copySrtBtn");
  if (copyBtn) {
    copyBtn.disabled = !_lastCaptionsResult?.content;
    copyBtn.title = copyBtn.disabled
      ? "Copy becomes available after a transcript, chapter list, or repeat review has been generated."
      : (_lastCaptionsResult?.copyLabel || "Copy output");
  }

  const importBtn = document.getElementById("importSrtBtn");
  if (importBtn) {
    const canImport = !!(_lastCaptionsResult && _lastCaptionsResult.kind === "transcript" && _lastCaptionsResult.hasSrt);
    importBtn.disabled = !canImport;
    importBtn.title = canImport
      ? "Open the timeline import flow for the current SRT output."
      : "SRT Prep becomes available after a transcript pass produces subtitle output.";
  }
}

function updateCaptionsPlanSummary() {
  setTextAndTitle("captionsPlanModel", getSelectLabel("whisperModel", "turbo"), getSelectLabel("whisperModel", "turbo"));
  setTextAndTitle("captionsPlanLanguage", getSelectLabel("transcribeLang", "Auto-detect"), getSelectLabel("transcribeLang", "Auto-detect"));
  setTextAndTitle("captionsPlanStyle", getSelectLabel("captionStyle", "YouTube Bold"), getSelectLabel("captionStyle", "YouTube Bold"));

  const diarization = document.getElementById("enableDiarization")?.checked;
  const wordLevel = document.getElementById("enableWordLevel")?.checked ?? true;
  const note = `${wordLevel ? "Word timing is on" : "Word timing is off"}. ${diarization ? "Speaker splits are on." : "Speaker splits are off."}`;
  setTextAndTitle("captionsPlanNote", note, note);
}

function updateCaptionsWorkspaceSummary() {
  updateCaptionsPlanSummary();

  const backendOnline = document.getElementById("connLabel")?.textContent?.trim() === "Online";
  const sourcePath = document.getElementById("clipPathCaptions")?.value?.trim() || getWorkspaceSource("captions");
  const hasSource = !!sourcePath;

  setTextAndTitle(
    "captionsSourceValue",
    hasSource ? formatWorkspaceSource(sourcePath) : "Choose a clip to start",
    sourcePath || "Choose a clip to start a captions pass."
  );

  if (!backendOnline) {
    setCaptionsSessionState(
      "Offline",
      "error",
      "Reconnect the local OpenCut backend before running transcript, chapter, or repeat jobs.",
      "error"
    );
    if (!_lastCaptionsResult) {
      setTextAndTitle("captionsOutputValue", "Waiting on backend", "Reconnect the local OpenCut backend to generate transcript output.");
    }
  } else if (!hasSource) {
    setCaptionsSessionState(
      "Choose media",
      "empty",
      "Choose a clip, then transcribe to unlock chapters, repeat review, and subtitle export.",
      "idle"
    );
    if (!_lastCaptionsResult) {
      setTextAndTitle("captionsOutputValue", "No transcript yet", "Choose a clip and run transcription to generate reviewable output.");
    }
  } else if (_lastCaptionsResult) {
    setCaptionsSessionState(
      _lastCaptionsResult.sessionLabel || "Result ready",
      _lastCaptionsResult.sessionState || "success",
      _lastCaptionsResult.statusMessage || "Output ready for review.",
      _lastCaptionsResult.statusState || _lastCaptionsResult.sessionState || "success",
      _lastCaptionsResult.statusTitle || _lastCaptionsResult.outputTitle || _lastCaptionsResult.statusMessage
    );
    setTextAndTitle(
      "captionsOutputValue",
      _lastCaptionsResult.outputLabel || "Result ready",
      _lastCaptionsResult.outputTitle || _lastCaptionsResult.outputLabel || "Result ready"
    );
  } else {
    setCaptionsSessionState(
      "Ready",
      "success",
      "Clip ready. Start with transcription, then move into chapters or repeat review when the wording is stable.",
      "ready"
    );
    setTextAndTitle("captionsOutputValue", "No transcript yet", "Transcribe the selected clip to generate reviewable output.");
  }

  syncCaptionsActionButtons();
}

function renderCaptionsResultView(resultState) {
  const area = document.getElementById("captionsResultArea");
  const body = document.getElementById("captionsResultBody");
  const summary = document.getElementById("captionsResultSummary");
  const meta = document.getElementById("captionsResultMeta");
  const header = area?.querySelector(".oc-result-header");
  const copyBtn = document.getElementById("copySrtBtn");
  const importBtn = document.getElementById("importSrtBtn");
  const typeLabelMap = {
    transcript: "Transcript",
    chapters: "Chapter draft",
    repeat: "Repeat review",
  };

  if (!area || !body) return;

  area.classList.remove("hidden");
  area.dataset.kind = resultState.kind || "review";
  area.dataset.state = resultState.resultPillState || "success";
  body.value = resultState.content || "";
  body.title = resultState.contentTitle || resultState.outputTitle || resultState.resultMetaTitle || "Review output";
  body.setAttribute("aria-label", resultState.header || "Review output");
  if (summary) summary.textContent = resultState.summary || "Ready to review";
  if (meta) {
    meta.textContent = resultState.resultMeta || "Review output is ready.";
    meta.title = resultState.resultMetaTitle || resultState.resultMeta || "Review output is ready.";
  }
  if (header) header.textContent = resultState.header || "Review Output";
  setStatusPill(
    "captionsResultPill",
    resultState.resultPillText || "Ready",
    resultState.resultPillState || "success",
    resultState.resultPillTitle || resultState.summary || "Ready"
  );
  if (copyBtn) copyBtn.textContent = resultState.copyLabel || "Copy Output";
  if (importBtn) importBtn.textContent = resultState.importLabel || "Open SRT Prep";
  setTextAndTitle(
    "captionsResultTypeValue",
    resultState.insightType || typeLabelMap[resultState.kind] || "Review output",
    resultState.insightTypeTitle || resultState.resultMetaTitle || resultState.summary || "Review output"
  );
  setTextAndTitle(
    "captionsResultLengthValue",
    resultState.insightLength || resultState.outputLabel || resultState.summary || "Ready",
    resultState.insightLengthTitle || resultState.outputTitle || resultState.summary || "Ready"
  );
  setTextAndTitle(
    "captionsResultNextValue",
    resultState.insightNext || (resultState.hasSrt ? "Copy SRT or open SRT Prep" : "Copy notes or continue review"),
    resultState.insightNextTitle || resultState.statusMessage || resultState.summary || "Next action"
  );

  _lastCaptionsResult = resultState;
  updateCaptionsWorkspaceSummary();
  area.focus();
}

function showRepeatResult(result) {
  const repeats = Array.isArray(result?.repeats) ? result.repeats : [];
  const clipPath = document.getElementById("clipPathCaptions")?.value?.trim() || "";
  const threshold = Number(document.getElementById("repeatSimilarity")?.value ?? 0.85);
  const keepBest = document.getElementById("keepBestRepeat")?.checked ?? true;
  const content = repeats.length
    ? repeats.map((repeat, index) => {
        const start = Number(repeat.start ?? repeat.start_time ?? repeat.begin ?? repeat.t0);
        const end = Number(repeat.end ?? repeat.end_time ?? repeat.finish ?? repeat.t1);
        const hasRange = Number.isFinite(start) || Number.isFinite(end);
        const duration = Number.isFinite(start) && Number.isFinite(end) && end > start
          ? formatCompactDuration(end - start)
          : "";
        const similarityRaw = Number(repeat.similarity ?? repeat.score ?? repeat.confidence);
        const similarity = Number.isFinite(similarityRaw)
          ? `${Math.round(similarityRaw > 1 ? similarityRaw : similarityRaw * 100)}% similar`
          : "";
        const preview = String(
          repeat.text ?? repeat.preview ?? repeat.transcript ?? repeat.reference_text ?? repeat.candidate_text ?? ""
        ).trim();
        const headerParts = [`Repeat ${index + 1}`];
        if (hasRange) {
          headerParts.push(`${formatTimecode(Number.isFinite(start) ? start : 0)} → ${formatTimecode(Number.isFinite(end) ? end : 0)}`);
        }
        if (duration) headerParts.push(duration);
        if (similarity) headerParts.push(similarity);
        return preview ? `${headerParts.join(" • ")}\n${preview}` : headerParts.join(" • ");
      }).join("\n\n")
    : "No repeated lines were flagged with the current threshold.";

  renderCaptionsResultView({
    kind: "repeat",
    header: "Repeat Review",
    summary: repeats.length
      ? `${repeats.length} repeat range${repeats.length === 1 ? "" : "s"} flagged`
      : "No repeated takes flagged",
    content,
    resultPillText: repeats.length ? "Review ready" : "Clean pass",
    resultPillState: repeats.length ? "warning" : "success",
    resultMeta: `${formatWorkspaceSource(clipPath)} • ${Math.round(threshold * 100)}% threshold • ${keepBest ? "Keep best take on" : "Keep best take off"}`,
    resultMetaTitle: clipPath || "Repeat review",
    copyLabel: "Copy Notes",
    importLabel: "Open SRT Prep",
    canOpenSrtImport: false,
    hasSrt: false,
    sessionLabel: repeats.length ? "Review ready" : "Clean pass",
    sessionState: repeats.length ? "warning" : "success",
    statusMessage: repeats.length
      ? "Repeat review is ready. Tighten the threshold or move the flagged ranges into your next cleanup pass."
      : "No repeated takes were flagged. The current threshold looks clean for this clip.",
    statusState: repeats.length ? "warning" : "success",
    statusTitle: clipPath || "Repeat review",
    outputLabel: repeats.length
      ? `${repeats.length} repeat ${repeats.length === 1 ? "range" : "ranges"} flagged`
      : "No repeats flagged",
    outputTitle: clipPath || "Repeat review",
    insightType: "Repeat review",
    insightLength: repeats.length
      ? `${repeats.length} repeat ${repeats.length === 1 ? "range" : "ranges"}`
      : "Clean pass",
    insightNext: repeats.length ? "Tighten threshold or continue cleanup" : "Keep current threshold and continue",
  });
}

function rememberTimelineCuts(cuts, info = {}) {
  lastCuts = Array.isArray(cuts) ? cuts.slice() : [];
  const count = lastCuts.length;
  _lastCutsInfo = count ? {
    count,
    source: info.source || "Latest cut pass",
    clipPath: info.clipPath || document.getElementById("clipPathCut")?.value?.trim() || document.getElementById("clipPathVideo")?.value?.trim() || "",
    time: info.time || new Date(),
  } : null;
  updateTimelineReadiness();
}

function rememberTimelineMarkers(markers, info = {}) {
  lastMarkers = Array.isArray(markers) ? markers.slice() : [];
  const count = lastMarkers.length;
  _lastMarkersInfo = count ? {
    count,
    source: info.source || "Latest marker pass",
    clipPath: info.clipPath || document.getElementById("beatTrackPath")?.value?.trim() || document.getElementById("clipPathAudio")?.value?.trim() || "",
    time: info.time || new Date(),
  } : null;
  updateTimelineReadiness();
}

function noteTimelineAction(label, state = "success", statusMessage, title, detailLabel) {
  _lastTimelineAction = {
    label,
    state,
    statusMessage: statusMessage || label,
    title: title || statusMessage || label,
    detailLabel: detailLabel || label,
    time: new Date(),
  };
  updateTimelineReadiness();
}

function setTimelineStatus(message, state = "idle", title) {
  const line = document.getElementById("timelineStatusLine");
  if (!line) return;
  line.textContent = message;
  line.dataset.state = state;
  line.title = title || message;
}

function buildExportWindows() {
  if (Array.isArray(lastMarkers) && lastMarkers.length) {
    const markerWindows = lastMarkers.map((marker, index) => ({
      time: typeof marker === "number" ? marker : (marker.time ?? marker.t ?? 0),
      duration: Math.max(0, Number(marker.duration ?? marker.len ?? marker.window ?? 0)),
      name: marker.label ?? marker.name ?? `marker_${index + 1}`,
    })).filter(marker => marker.duration > 0);
    if (markerWindows.length) return markerWindows;
  }

  if (Array.isArray(lastCuts) && lastCuts.length) {
    return lastCuts.map((cut, index) => {
      const start = Number(cut.start ?? cut.time ?? 0);
      const end = Number(cut.end ?? start);
      return {
        time: start,
        duration: Math.max(0, end - start),
        name: cut.label ?? `cut_${index + 1}`,
      };
    }).filter(window => window.duration > 0);
  }

  return [];
}

function updateTimelineReadiness() {
  const backendOnline = document.getElementById("connLabel")?.textContent?.trim() === "Online";
  const bridgeReady = PProBridge.available();
  const clipPath = document.getElementById("clipPathVideo")?.value?.trim()
    || document.getElementById("clipPathCut")?.value?.trim()
    || getWorkspaceSource("timeline");
  const outputDir = document.getElementById("exportDir")?.value?.trim() || "";
  const srtPath = document.getElementById("srtFilePath")?.value?.trim() || "";
  const trackIndex = Math.max(1, parseInt(document.getElementById("srtTrackIndex")?.value ?? 1, 10) || 1);
  const renamePattern = document.getElementById("renamePattern")?.value?.trim() || "{name}_{index:03d}";
  const smartBinsStrategy = getSelectLabel("binStrategy", "File Type");
  const exportWindows = buildExportWindows();

  setStatusPill(
    "timelineBridgePill",
    bridgeReady ? "UXP ready" : "CEP fallback",
    bridgeReady ? "success" : "warning",
    bridgeReady
      ? "UXP sequence write-back is available in this Premiere session."
      : "Direct sequence write-back is not available here. Use the CEP panel for dependable timeline execution."
  );

  const cutsLabel = _lastCutsInfo
    ? `${_lastCutsInfo.count} cut${_lastCutsInfo.count === 1 ? "" : "s"} • ${_lastCutsInfo.source}`
    : "No cuts ready";
  const cutsTitle = _lastCutsInfo
    ? `${_lastCutsInfo.count} cut${_lastCutsInfo.count === 1 ? "" : "s"} from ${_lastCutsInfo.source}${_lastCutsInfo.clipPath ? ` • ${_lastCutsInfo.clipPath}` : ""}`
    : "Run silence, filler, or multicam cleanup to stage cuts for timeline write-back.";
  setTextAndTitle("timelineCutsValue", cutsLabel, cutsTitle);
  setTextAndTitle("timelineCutsSourceValue", _lastCutsInfo ? `${_lastCutsInfo.source} • ${_lastCutsInfo.count} cuts` : "Run a cut pass first", cutsTitle);

  const markersLabel = _lastMarkersInfo
    ? `${_lastMarkersInfo.count} marker${_lastMarkersInfo.count === 1 ? "" : "s"} • ${_lastMarkersInfo.source}`
    : "No markers ready";
  const markersTitle = _lastMarkersInfo
    ? `${_lastMarkersInfo.count} marker${_lastMarkersInfo.count === 1 ? "" : "s"} from ${_lastMarkersInfo.source}${_lastMarkersInfo.clipPath ? ` • ${_lastMarkersInfo.clipPath}` : ""}`
    : "Run beat detection to stage markers for sequence write-back or marker-based export.";
  setTextAndTitle("timelineMarkersValue", markersLabel, markersTitle);

  const lastActionLabel = _lastTimelineAction
    ? `${_lastTimelineAction.detailLabel} • ${formatLocaleTime(_lastTimelineAction.time)}`
    : "No write-back activity";
  setTextAndTitle(
    "timelineLastActionValue",
    lastActionLabel,
    _lastTimelineAction?.title || "No timeline write-back, export, or validation action has run in this session."
  );

  const sequenceLabel = bridgeReady
    ? (_lastSequenceInfo?.name || "UXP bridge ready")
    : "Use CEP panel for write-back";
  const sequenceTitle = bridgeReady
    ? (_lastSequenceInfo?.name || "The UXP bridge is ready. Direct sequence calls will use the active sequence when available.")
    : "Direct sequence write-back is not available in this UXP session.";
  setTextAndTitle("timelineSequenceValue", sequenceLabel, sequenceTitle);

  const exportSourceLabel = exportWindows.length
    ? `${exportWindows.length} ${exportWindows.length === 1 ? "window" : "windows"} ready`
    : "Awaiting cuts or markers";
  const exportSourceTitle = exportWindows.length
    ? `${exportWindows.length} export window${exportWindows.length === 1 ? "" : "s"} are staged from the latest ${Array.isArray(lastMarkers) && lastMarkers.length ? "marker" : "cut"} pass.`
    : "Run beat detection or create cuts first, then export those windows from the current clip.";
  setTextAndTitle("timelineExportSourceValue", exportSourceLabel, exportSourceTitle);
  setTextAndTitle("timelineExportOutputValue", outputDir ? formatWorkspaceSource(outputDir) : "Choose output folder", outputDir || "Choose an export destination for marker-based segments.");

  setStatusPill("timelineRenamePill", "CEP panel required", "warning", "Batch rename still executes through the CEP panel on this build.");
  setTextAndTitle("timelineRenamePatternValue", renamePattern, renamePattern);
  setStatusPill("timelineSmartBinsPill", "CEP panel required", "warning", "Smart bin execution still runs through the CEP panel on this build.");
  setTextAndTitle("timelineSmartBinsValue", smartBinsStrategy, smartBinsStrategy);

  setTextAndTitle("timelineSrtValue", srtPath ? formatWorkspaceSource(srtPath) : "Choose .srt file", srtPath || "Choose an .srt file to validate for CEP/native import.");
  setTextAndTitle("timelineSrtTrackValue", `Track ${trackIndex}`, `Track ${trackIndex} in the final CEP/native captions import flow.`);

  const hasCuts = Array.isArray(lastCuts) && lastCuts.length > 0;
  const hasMarkers = Array.isArray(lastMarkers) && lastMarkers.length > 0;

  const applyCutsBtn = document.getElementById("applyTimelineCutsBtn");
  if (applyCutsBtn && !applyCutsBtn.classList.contains("loading")) {
    applyCutsBtn.disabled = !bridgeReady || !hasCuts;
  }

  const addMarkersBtn = document.getElementById("addBeatMarkersBtn");
  if (addMarkersBtn && !addMarkersBtn.classList.contains("loading")) {
    addMarkersBtn.disabled = !bridgeReady || !hasMarkers;
  }

  const otioBtn = document.getElementById("exportOtioBtn");
  if (otioBtn && !otioBtn.classList.contains("loading")) {
    otioBtn.disabled = !backendOnline || !clipPath || (!hasCuts && !hasMarkers);
  }

  const batchExportBtn = document.getElementById("runBatchExportBtn");
  if (batchExportBtn && !batchExportBtn.classList.contains("loading")) {
    batchExportBtn.disabled = !backendOnline || !clipPath || !outputDir || exportWindows.length === 0;
  }

  const renameBtn = document.getElementById("runBatchRenameBtn");
  if (renameBtn && !renameBtn.classList.contains("loading")) {
    renameBtn.disabled = true;
  }

  const smartBinsBtn = document.getElementById("runSmartBinsBtn");
  if (smartBinsBtn && !smartBinsBtn.classList.contains("loading")) {
    smartBinsBtn.disabled = true;
  }

  const srtBtn = document.getElementById("runSrtImportBtn");
  if (srtBtn && !srtBtn.classList.contains("loading")) {
    srtBtn.disabled = !backendOnline || !srtPath;
  }

  if (!backendOnline) {
    setTimelineStatus(
      "Reconnect the local backend before exporting windows, validating captions, or packaging timeline handoff.",
      "error"
    );
  } else if (!bridgeReady && (hasCuts || hasMarkers)) {
    setTimelineStatus(
      "Timeline assets are ready, but direct sequence write-back still needs the CEP panel on this Premiere setup. OTIO export and SRT validation remain available here.",
      "warning"
    );
  } else if (!bridgeReady) {
    setTimelineStatus(
      "Generate cuts or beat markers first. Direct sequence write-back will fall back to the CEP panel on this setup.",
      "warning"
    );
  } else if (_lastTimelineAction) {
    setTimelineStatus(_lastTimelineAction.statusMessage, _lastTimelineAction.state, _lastTimelineAction.title);
  } else if (!hasCuts && !hasMarkers) {
    setTimelineStatus(
      "Generate cuts or beat markers first, then return here to write back, export OTIO, or validate captions for sequence import.",
      "idle"
    );
  } else {
    setTimelineStatus(
      "Timeline assets are ready. Apply cuts, add markers, export OTIO, or validate an SRT before the handoff pass.",
      "ready"
    );
  }
}

function focusControl(controlId) {
  const el = document.getElementById(controlId);
  if (!el) return;
  el.focus();
  if (typeof el.select === "function" && (el.tagName === "INPUT" || el.tagName === "TEXTAREA")) {
    el.select();
  }
}

function fillFieldFromSuggestion(button) {
  const targetId = button?.dataset.fillTarget;
  if (!targetId) return;
  const target = document.getElementById(targetId);
  if (!target) return;
  target.value = button.dataset.fillValue ?? "";
  target.dispatchEvent(new Event("input", { bubbles: true }));
  focusControl(targetId);
}

function handleWorkspaceAction(action) {
  const activeTab = getWorkspaceTabId();
  if (!action) return;

  if (action === "choose-clip") {
    browseFile(getPrimarySourceInputId(activeTab));
    return;
  }
  if (action === "open-search") {
    UIController.switchTab("search");
    focusControl("searchQuery");
    return;
  }
  if (action === "open-timeline") {
    UIController.switchTab("timeline");
    focusControl("applyTimelineCutsBtn");
    return;
  }
  if (action === "refresh-backend") {
    document.getElementById("refreshBtn")?.click();
    return;
  }
  if (action === "switch-cut") {
    UIController.switchTab("cut");
    focusControl("runSilenceBtn");
    return;
  }
  if (action.startsWith("focus-")) {
    focusControl(action.slice(6));
  }
}

function setWorkspaceGuide(guide) {
  const guideEl = document.getElementById("workspaceGuide");
  const kickerEl = document.getElementById("workspaceGuideKicker");
  const titleEl = document.getElementById("workspaceGuideTitle");
  const textEl = document.getElementById("workspaceGuideText");
  const actionEl = document.getElementById("workspaceGuideAction");
  const guideState = guide.state || "ready";

  if (guideEl) {
    guideEl.dataset.state = guideState;
    guideEl.title = guide.text || guide.title || "";
  }
  if (kickerEl) kickerEl.textContent = guide.kicker;
  if (titleEl) titleEl.textContent = guide.title;
  if (textEl) textEl.textContent = guide.text;
  if (actionEl) {
    actionEl.dataset.action = guide.action || "";
    actionEl.textContent = guide.actionLabel || "Open";
    actionEl.disabled = !guide.action;
    actionEl.hidden = !guide.actionLabel;
  }
}

function applyWorkspaceShellState(activeTab, backendOnline, sourcePath) {
  document.body.dataset.activeTab = activeTab || "cut";
  document.body.classList.toggle("oc-connected", !!backendOnline);
  document.body.classList.toggle("oc-disconnected", !backendOnline);
  document.body.classList.toggle("oc-has-source", !!sourcePath);
}

function setWorkspaceClip(pathValue, options = {}) {
  const nextValue = (pathValue || "").trim();
  _workspaceClipPath = nextValue;

  if (_syncingWorkspaceClip) {
    updateWorkspaceOverview();
    return;
  }

  _syncingWorkspaceClip = true;
  for (const id of PRIMARY_CLIP_INPUT_IDS) {
    if (options.originId && id === options.originId) continue;
    const input = document.getElementById(id);
    if (!input) continue;
    input.value = nextValue;
  }
  _syncingWorkspaceClip = false;
  updateWorkspaceOverview(options.tabId);
}

function updateWorkspaceOverview(tabId) {
  const activeTab = getWorkspaceTabId(tabId);
  const meta = WORKSPACE_META[activeTab] || WORKSPACE_META.cut;
  const sourcePath = getWorkspaceSource(activeTab);
  const backendLabel = document.getElementById("connLabel")?.textContent?.trim() || "Offline";
  const backendOnline = backendLabel === "Online";
  const overviewTitle = document.getElementById("workspaceOverviewTitle");
  const overviewSubtitle = document.getElementById("workspaceOverviewSubtitle");
  const sourceValue = document.getElementById("workspaceSourceValue");
  const backendValue = document.getElementById("workspaceBackendValue");
  const libraryValue = document.getElementById("workspaceLibraryValue");

  applyWorkspaceShellState(activeTab, backendOnline, sourcePath);

  if (overviewTitle) overviewTitle.textContent = meta.title;
  if (overviewSubtitle) overviewSubtitle.textContent = meta.subtitle;
  if (sourceValue) {
    sourceValue.textContent = formatWorkspaceSource(sourcePath);
    sourceValue.title = sourcePath || "Choose a clip or paste a path to start";
    sourceValue.closest(".oc-workspace-meta-item")?.setAttribute("data-state", sourcePath ? "ready" : "empty");
  }
  if (backendValue) {
    backendValue.textContent = backendLabel;
    backendValue.closest(".oc-workspace-meta-item")?.setAttribute("data-state", backendOnline ? "online" : "offline");
  }
  if (libraryValue) {
    const count = Array.isArray(_projectClips) ? _projectClips.length : 0;
    libraryValue.textContent = `${count} ${count === 1 ? "clip" : "clips"}`;
    libraryValue.closest(".oc-workspace-meta-item")?.setAttribute("data-state", count > 0 ? "ready" : "empty");
  }

  let guide = WORKSPACE_GUIDES[activeTab] || WORKSPACE_GUIDES.cut;
  if (!backendOnline) {
    guide = {
      kicker: "Backend offline",
      title: "Reconnect the local OpenCut backend before running jobs.",
      text: "The workspace is ready, but processing, engine refresh, and timeline handoff all depend on the backend service being online.",
      state: "error",
      action: "refresh-backend",
      actionLabel: "Refresh Backend",
    };
  } else if (TABS_REQUIRING_SOURCE.has(activeTab) && !sourcePath) {
    guide = {
      kicker: "Choose media",
      title: "Choose one active shot to unlock this workspace.",
      text: "OpenCut keeps the current clip in sync across Cut, Captions, Audio, and Video so you can move through the edit without repeated setup.",
      state: "warning",
      action: "choose-clip",
      actionLabel: "Choose Media",
    };
  } else if (activeTab === "timeline" && !lastCuts.length && !lastMarkers.length) {
    guide = {
      kicker: "Ready for write-back",
      title: "Generate cuts or markers first, then bring them back to the sequence.",
      text: "Run a cleanup or beat pass in Cut or Audio, then return here to apply the result, export OTIO, or batch markers.",
      state: "warning",
      action: "switch-cut",
      actionLabel: "Open Cut Workspace",
    };
  }
  setWorkspaceGuide(guide);
  updateCaptionsWorkspaceSummary();
  updateTimelineReadiness();
  updateDeliverablesSummary();
  syncSearchPanelState();
}

function getDeliverablesOutputSummary() {
  const outputDir = document.getElementById("delivOutputDir")?.value?.trim() || "";
  if (!outputDir) {
    return {
      label: "Session temp folder",
      title: "Deliverables will be saved to the session temp folder until you choose an output folder.",
    };
  }
  return {
    label: formatWorkspaceSource(outputDir),
    title: outputDir,
  };
}

function getSelectedReportTypes() {
  return [
    ["rptIncludeVfx", "vfx_sheet"],
    ["rptIncludeAdr", "adr_list"],
    ["rptIncludeMusic", "music_cue_sheet"],
    ["rptIncludeAssets", "asset_list"],
  ]
    .filter(([controlId]) => document.getElementById(controlId)?.checked)
    .map(([, type]) => type);
}

function getDeliverablesSelectionSummary(selectedTypes) {
  if (!selectedTypes.length) {
    return {
      label: "No documents selected",
      title: "Select at least one handoff document before generating a package.",
    };
  }
  if (selectedTypes.length === 4) {
    return {
      label: "All four handoff sheets",
      title: "VFX Sheet, ADR List, Music Cue Sheet, and Asset List will be generated in this package.",
    };
  }
  if (selectedTypes.length === 1) {
    const label = DELIVERABLE_LABELS[selectedTypes[0]] || humanizeDomain(selectedTypes[0]);
    return {
      label,
      title: `${label} is the only document selected for this package.`,
    };
  }
  const labels = selectedTypes.map((type) => DELIVERABLE_LABELS[type] || humanizeDomain(type));
  return {
    label: `${selectedTypes.length} documents selected`,
    title: labels.join(" • "),
  };
}

function getDeliverablesFormatSummary() {
  return {
    value: "csv",
    label: "CSV handoff sheets",
    title: "The UXP deliverables workflow currently exports CSV handoff sheets.",
  };
}

function updateDeliverablesPlanSummary(outputSummary = getDeliverablesOutputSummary()) {
  const selectedTypes = getSelectedReportTypes();
  const selectionSummary = getDeliverablesSelectionSummary(selectedTypes);
  const formatSummary = getDeliverablesFormatSummary();
  const backendOnline = document.getElementById("connLabel")?.textContent?.trim() === "Online";
  const hasSequence = !!_lastSequenceInfo;
  const canGenerate = hasSequence && backendOnline;

  setTextAndTitle("deliverablesPackageValue", selectionSummary.label, selectionSummary.title);
  setTextAndTitle("deliverablesReportFormatValue", formatSummary.label, formatSummary.title);
  setTextAndTitle("deliverablesReportDestinationValue", outputSummary.label, outputSummary.title);

  const formatSelect = document.getElementById("reportFormat");
  if (formatSelect) {
    formatSelect.value = formatSummary.value;
    formatSelect.title = formatSummary.title;
  }

  Object.entries(DELIVERABLE_BUTTON_IDS).forEach(([type, id]) => {
    const btn = document.getElementById(id);
    if (!btn || btn.classList.contains("loading")) return;
    const label = DELIVERABLE_LABELS[type] || humanizeDomain(type);
    btn.disabled = !canGenerate;
    btn.title = !backendOnline
      ? `Reconnect the backend before generating ${label}.`
      : (hasSequence
          ? `Generate ${label} as a CSV handoff sheet.`
          : `Load sequence info before generating ${label}.`);
  });

  const reportBtn = document.getElementById("runFullReportBtn");
  if (reportBtn && !reportBtn.classList.contains("loading")) {
    if (!selectedTypes.length) {
      reportBtn.textContent = "Select Documents First";
    } else if (selectedTypes.length === 1) {
      reportBtn.textContent = `Generate ${DELIVERABLE_LABELS[selectedTypes[0]] || "Selected Report"}`;
    } else if (selectedTypes.length === 4) {
      reportBtn.textContent = "Generate Full Report";
    } else {
      reportBtn.textContent = `Generate ${selectedTypes.length}-Doc Package`;
    }
    reportBtn.disabled = !canGenerate || selectedTypes.length === 0;
    reportBtn.title = !selectedTypes.length
      ? "Select at least one handoff document before generating a package."
      : (!backendOnline
          ? "Reconnect the backend before generating the selected handoff package."
          : (!hasSequence
          ? "Load sequence info before generating the selected handoff package."
          : `${selectionSummary.title} ${formatSummary.title}`));
  }
}

function setDeliverablesButtonsDisabled(disabled) {
  Object.values(DELIVERABLE_BUTTON_IDS).forEach((id) => {
    const btn = document.getElementById(id);
    if (btn && !btn.classList.contains("loading")) btn.disabled = disabled;
  });
}

function setDeliverablesStatus(message, state = "idle", title) {
  const line = document.getElementById("deliverablesStatus");
  if (!line) return;
  line.textContent = message;
  line.dataset.state = state;
  line.title = title || message;
}

function setSettingsStatus(id, message, state = "idle", title) {
  const line = document.getElementById(id);
  if (!line) return;
  line.textContent = message;
  line.dataset.state = state;
  line.title = title || message;
}

function updateDeliverablesSummary() {
  const info = _lastSequenceInfo;
  const output = getDeliverablesOutputSummary();
  const hasOutputDestination = !!document.getElementById("delivOutputDir")?.value?.trim();
  const selectedTypes = getSelectedReportTypes();
  const backendOnline = document.getElementById("connLabel")?.textContent?.trim() === "Online";

  if (info) {
    const resolution = (info.width && info.height) ? `${info.width} × ${info.height}` : "Unknown size";
    const duration = typeof info.duration === "number" ? formatTimecode(info.duration) : "Unknown duration";
    setStatusPill("seqInfoStatePill", "Loaded", "success", "Sequence info is ready for deliverables.");
    setTextAndTitle(
      "seqInfoSummary",
      `${info.name || "Active Sequence"} • ${resolution} • ${duration}`,
      `${info.name || "Active Sequence"} | ${resolution} | ${duration}`
    );
    setTextAndTitle(
      "deliverablesSequenceValue",
      info.name || "Active Sequence",
      info.name || "Active Sequence"
    );
    setDeliverablesButtonsDisabled(false);
  } else {
    setStatusPill("seqInfoStatePill", "Not Loaded", "empty", "Load the active Premiere sequence before generating deliverables.");
    setTextAndTitle(
      "seqInfoSummary",
      "Load the active Premiere sequence before generating handoff docs.",
      "Load the active Premiere sequence before generating handoff docs."
    );
    setTextAndTitle(
      "deliverablesSequenceValue",
      "Load sequence info",
      "Load sequence info before generating handoff docs."
    );
    setDeliverablesButtonsDisabled(true);
  }

  setTextAndTitle("deliverablesOutputValue", output.label, output.title);
  updateDeliverablesPlanSummary(output);

  if (_lastDeliverableActivity) {
    const activity = _lastDeliverableActivity;
    const label = activity.count
      ? `${activity.label} • ${activity.count} ${activity.count === 1 ? "doc" : "docs"}`
      : activity.label;
    const relative = formatRelativeTime(activity.time);
    setTextAndTitle(
      "deliverablesLastExportValue",
      `${label} (${relative})`,
      activity.output || `${label} at ${formatLocaleTime(activity.time)}`
    );
  } else {
    setTextAndTitle("deliverablesLastExportValue", "No exports yet", "No deliverables have been generated in this session.");
  }

  if (!backendOnline) {
    setDeliverablesStatus("Reconnect the local backend before generating handoff documents or report packages.", "error");
  } else if (!_lastSequenceInfo) {
    setDeliverablesStatus("Load sequence info, choose a destination if needed, then generate the handoff docs you need.", "idle");
  } else if (!selectedTypes.length) {
    setDeliverablesStatus("Sequence info is ready. Select at least one handoff document below, or generate a single sheet above for this pass.", "warning");
  } else if (!_lastDeliverableActivity && !hasOutputDestination) {
    setDeliverablesStatus("Sequence ready. Choose an output folder if you want handoff docs saved somewhere more durable than the session temp folder.", "warning", output.title);
  } else if (_lastDeliverableActivity) {
    const lastLabel = _lastDeliverableActivity.count
      ? `${_lastDeliverableActivity.label} finished`
      : `${_lastDeliverableActivity.label} ready`;
    setDeliverablesStatus(`${lastLabel}. Generate another document or refresh the sequence info before the next handoff pass.`, "success", _lastDeliverableActivity.output);
  } else {
    setDeliverablesStatus("Sequence info is ready. Generate a single document or run the full report when the handoff package is ready.", "ready");
  }
}

function setIndexStatus(message, state = "idle", title) {
  const line = document.getElementById("indexStatus");
  if (!line) return;
  line.textContent = message;
  line.dataset.state = state;
  line.title = title || message;
}

function resetClearIndexConfirmation({ resync = true } = {}) {
  _clearIndexConfirmUntil = 0;
  if (_clearIndexConfirmTimer) {
    clearTimeout(_clearIndexConfirmTimer);
    _clearIndexConfirmTimer = null;
  }
  const clearBtn = document.getElementById("clearIndexBtn");
  if (clearBtn && !clearBtn.classList.contains("loading")) {
    clearBtn.textContent = "Clear Index";
  }
  if (resync) syncSearchPanelState();
}

function requireClearIndexConfirmation() {
  const now = Date.now();
  if (_clearIndexConfirmUntil > now) {
    resetClearIndexConfirmation();
    return true;
  }

  _clearIndexConfirmUntil = now + INLINE_CONFIRM_MS;
  if (_clearIndexConfirmTimer) clearTimeout(_clearIndexConfirmTimer);
  _clearIndexConfirmTimer = setTimeout(() => resetClearIndexConfirmation(), INLINE_CONFIRM_MS);
  const clearBtn = document.getElementById("clearIndexBtn");
  if (clearBtn && !clearBtn.classList.contains("loading")) {
    clearBtn.textContent = "Confirm Clear";
    clearBtn.title = "Click again within 8 seconds to clear the current search index.";
  }
  setIndexStatus("Click Confirm Clear again to remove the current search index.", "warning");
  UIController.showToast("Click Confirm Clear again to remove the current search index.", "warning");
  return false;
}

function syncSearchPanelState() {
  const backendOnline = document.getElementById("connLabel")?.textContent?.trim() === "Online";
  const folder = document.getElementById("indexFolder")?.value?.trim() || "";
  const query = document.getElementById("searchQuery")?.value?.trim() || "";
  const nlpCommand = document.getElementById("nlpCommand")?.value?.trim() || "";
  const indexedFiles = Number(_lastIndexStats?.total_files || 0);
  const hasIndex = indexedFiles > 0;

  const runIndexBtn = document.getElementById("runIndexLibBtn");
  if (runIndexBtn && !runIndexBtn.classList.contains("loading")) {
    runIndexBtn.disabled = !backendOnline || !folder;
    runIndexBtn.title = !backendOnline
      ? "Reconnect the backend before indexing the library."
      : (folder
          ? "Build a searchable library from the selected media folder."
          : "Choose a media folder before indexing the library.");
  }

  const clearBtn = document.getElementById("clearIndexBtn");
  if (clearBtn && !clearBtn.classList.contains("loading")) {
    if (_clearIndexConfirmUntil && _clearIndexConfirmUntil <= Date.now()) {
      resetClearIndexConfirmation({ resync: false });
    }
    clearBtn.disabled = !backendOnline || !hasIndex;
    if (_clearIndexConfirmUntil > Date.now() && backendOnline && hasIndex) {
      clearBtn.textContent = "Confirm Clear";
      clearBtn.title = "Click again within 8 seconds to clear the current search index.";
    } else {
      clearBtn.textContent = "Clear Index";
      clearBtn.title = !backendOnline
        ? "Reconnect the backend before clearing the search index."
        : (hasIndex
            ? "Clear the current search index. You can rebuild it any time."
            : "Build an index before clearing it.");
    }
  }

  const searchBtn = document.getElementById("runFootageSearchBtn");
  if (searchBtn && !searchBtn.classList.contains("loading")) {
    searchBtn.disabled = !backendOnline || !hasIndex || !query;
    searchBtn.title = !backendOnline
      ? "Reconnect the backend before searching the library."
      : (!hasIndex
          ? "Index a folder before searching the library."
          : (query
              ? "Search the indexed library and load the best shot back into the workspace."
              : "Enter a descriptive query before searching the library."));
  }

  const nlpBtn = document.getElementById("runNlpBtn");
  if (nlpBtn && !nlpBtn.classList.contains("loading")) {
    nlpBtn.disabled = !backendOnline || !nlpCommand;
    nlpBtn.title = !backendOnline
      ? "Reconnect the backend before running natural-language commands."
      : (nlpCommand
          ? "Parse the current edit instruction and review the result before applying it."
          : "Enter a natural-language edit instruction before running it.");
  }
}

function resetSearchResults(kicker, message) {
  const list = document.getElementById("searchResultList");
  if (!list) return;
  list.innerHTML = `
    <div class="oc-empty-state oc-empty-state-inline">
      <div class="oc-empty-state-kicker">${UIController.escapeHtml(kicker)}</div>
      <p>${UIController.escapeHtml(message)}</p>
    </div>`;
}

async function refreshFootageIndexStats(options = {}) {
  const r = await BackendClient.get("/timeline/index-status");
  if (!r.ok) {
    if (!options.silent) {
      _lastIndexStats = { total_files: 0, total_segments: 0, index_size_bytes: 0 };
      setStatusPill("indexStatePill", "Unavailable", "warning", "The panel could not read the search index status.");
      setTextAndTitle("indexStatsValue", "Index status unavailable", "The panel could not read the search index status.");
      setIndexStatus("Could not read the current library index. Reconnect the backend, then refresh or re-index the folder.", "warning");
    }
    syncSearchPanelState();
    return null;
  }

  const stats = {
    total_files: Number(r.data?.total_files || 0),
    total_segments: Number(r.data?.total_segments || 0),
    index_size_bytes: Number(r.data?.index_size_bytes || 0),
  };
  _lastIndexStats = stats;

  const statsLabel = stats.total_files
    ? `${stats.total_files} ${stats.total_files === 1 ? "file" : "files"} indexed`
    : "0 files indexed";
  const statsTitle = stats.total_files
    ? `${stats.total_files} files indexed, ${stats.total_segments} transcript segments, ${formatBytes(stats.index_size_bytes)} on disk.`
    : "No footage index has been built yet.";

  setTextAndTitle("indexStatsValue", statsLabel, statsTitle);

  if (!options.preserveMessage) {
    if (stats.total_files > 0) {
      setStatusPill("indexStatePill", "Ready", "success", statsTitle);
      setIndexStatus(`Library ready. ${stats.total_files} indexed ${stats.total_files === 1 ? "file" : "files"} can be searched right away.`, "success", statsTitle);
    } else {
      setStatusPill("indexStatePill", "Empty", "empty", statsTitle);
      setIndexStatus("Index a folder to make descriptive search results available in this workspace.", "idle");
    }
  }

  syncSearchPanelState();
  return stats;
}

async function clearFootageIndex() {
  if (!(Number(_lastIndexStats?.total_files || 0) > 0)) {
    resetClearIndexConfirmation();
    setIndexStatus("Build an index before clearing it.", "warning");
    UIController.showToast("Build an index before clearing it.", "warning");
    syncSearchPanelState();
    return;
  }

  if (!requireClearIndexConfirmation()) {
    return;
  }

  UIController.setButtonLoading("clearIndexBtn", true);
  setStatusPill("indexStatePill", "Clearing", "working", "Clearing the current search index.");
  setIndexStatus("Clearing the current search index…", "working");

  const r = await BackendClient.del("/search/index");

  UIController.setButtonLoading("clearIndexBtn", false);

  if (!r.ok) {
    setStatusPill("indexStatePill", "Error", "error", r.error || "Failed to clear the search index.");
    setIndexStatus(r.error || "Failed to clear the search index.", "error");
    UIController.showToast(r.error || "Failed to clear the search index.", "error");
    return;
  }

  resetSearchResults("Search the library", "Index a folder again to bring searchable media back into this workspace.");
  setTextAndTitle("searchStatus", "The search index has been cleared. Re-index a folder to search footage again.", "The search index has been cleared. Re-index a folder to search footage again.");
  await refreshFootageIndexStats({ preserveMessage: true, silent: true });
  setStatusPill("indexStatePill", "Empty", "empty", "The search index is empty until you index a folder again.");
  setTextAndTitle("indexStatsValue", "0 files indexed", "The search index is empty until you index a folder again.");
  setIndexStatus("Search index cleared. Re-index a folder to make library results available again.", "success");
  UIController.showToast("Search index cleared.", "success");
  resetClearIndexConfirmation();
  syncSearchPanelState();
}

async function ensureSequenceInfo(options = {}) {
  if (_lastSequenceInfo && !options.force) return _lastSequenceInfo;

  let info = null;
  if (PProBridge.available()) {
    info = await PProBridge.getSequenceInfo();
  }

  _lastSequenceInfo = info || null;
  updateDeliverablesSummary();

  if (!_lastSequenceInfo && !options.silent) {
    UIController.showToast("Open an active Premiere sequence, then load sequence info before generating deliverables.", "warning");
    UIController.setStatus("Load sequence info before generating deliverables.", "error");
  }

  return _lastSequenceInfo;
}

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
  for (const id of PRIMARY_CLIP_INPUT_IDS) {
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
  if (_workspaceClipPath) {
    setWorkspaceClip(_workspaceClipPath);
    return;
  }
  updateWorkspaceOverview();
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
  UIController.showProcessing("Detecting silences…");
  UIController.setStatus("Running silence removal…");

  await JobPoller.start(
    "/silence",
    { filepath: clipPath, threshold: threshold, min_duration: minSilence, padding_before: padding / 1000, padding_after: padding / 1000, mode, method: detectMethod },
    (pct, msg) => {
      UIController.setProgress(pct);
      UIController.setProcessingMsg(msg || "Processing…");
    },
    (result) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runSilenceBtn", false);

      // Backend /silence returns ``segments_data`` (array of {start, end})
      // and ``segments`` (count). Older code read ``result.cuts`` which was
      // never populated, so the panel always showed "No silences found"
      // even on successful detection. ``cuts`` retained as fallback in case
      // some future engine variant emits it.
      const cuts = (result.cuts && result.cuts.length)
        ? result.cuts
        : (result.segments_data || []);
      if (cuts.length > 0) {
        rememberTimelineCuts(cuts, { source: "Silence Removal", clipPath });
        showCutResult({ ...result, cuts });
        UIController.showToast(`Removed ${cuts.length} silence region(s).`, "success");
        UIController.setStatus(`Done — ${cuts.length} cuts`);
      } else if (result.xml_path || result.output_path) {
        const out = result.xml_path || result.output_path;
        UIController.showToast(`Output: ${out}`, "success");
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
  const summary = document.getElementById("cutResultSummary");
  if (!area || !body) return;
  area.classList.remove("hidden");
  const cuts = result.cuts ?? [];
  const clipPath = document.getElementById("clipPathCut")?.value?.trim() || getWorkspaceSource("cut");
  const totalRemoved = cuts.reduce((sum, cut) => sum + Math.max(0, Number(cut.end) - Number(cut.start)), 0);
  const longestCut = cuts.reduce((max, cut) => Math.max(max, Math.max(0, Number(cut.end) - Number(cut.start))), 0);
  if (summary) {
    summary.textContent = cuts.length
      ? `${cuts.length} cut window${cuts.length === 1 ? "" : "s"} • ${formatCompactDuration(totalRemoved)} removed${longestCut ? ` • longest ${formatCompactDuration(longestCut)}` : ""}`
      : "No cuts detected";
  }
  setTextAndTitle(
    "cutResultSourceValue",
    clipPath ? formatWorkspaceSource(clipPath) : "Awaiting reviewed pass",
    clipPath || "Choose a clip and run a cleanup pass to stage cut windows."
  );
  setTextAndTitle(
    "cutResultRemovedValue",
    cuts.length ? `${formatCompactDuration(totalRemoved)} total` : "0 s",
    cuts.length
      ? `${formatCompactDuration(totalRemoved)} total removed across ${cuts.length} cut window${cuts.length === 1 ? "" : "s"}.`
      : "No cut windows are staged yet."
  );
  setTextAndTitle(
    "cutResultNextValue",
    cuts.length ? "Apply latest cuts to timeline" : "Adjust settings and rerun",
    cuts.length
      ? "Review the suggested windows, then apply the pass to the active sequence."
      : "Refine the thresholds or switch cleanup mode, then run another pass."
  );
  body.innerHTML = cuts.length
    ? cuts.map((cut, index) => {
        const start = Number(cut.start);
        const end = Number(cut.end);
        const duration = Math.max(0, end - start);
        return `
          <div class="oc-result-row">
            <span class="oc-result-chip">Cut ${index + 1}</span>
            <div class="oc-result-copy">
              <strong>${formatTimecode(start)} to ${formatTimecode(end)}</strong>
              <span>${formatCompactDuration(duration)} removed</span>
            </div>
          </div>`;
      }).join("")
    : `
      <div class="oc-empty-state oc-empty-state-inline">
        <div class="oc-empty-state-kicker">No changes yet</div>
        <p>Run silence detection or filler cleanup to generate timeline-ready cuts here.</p>
      </div>`;
  area.focus();
}

/** ── FILLER WORD DETECTION ── */
async function runFillerDetection() {
  const clipPath = document.getElementById("clipPathCut")?.value?.trim();
  if (!clipPath) { UIController.showToast("Please select a clip first.", "warning"); return; }

  const words   = document.getElementById("fillerWords")?.value ?? "um,uh,like";
  const padding = parseInt(document.getElementById("fillerPadding")?.value ?? 50);
  const fillerBackend = document.getElementById("fillerBackend")?.value ?? "whisper";

  UIController.setButtonLoading("runFillerBtn", true);
  UIController.showProcessing(fillerBackend === "crisper" ? "Detecting fillers with CrisperWhisper…" : "Detecting filler words…");

  await JobPoller.start(
    "/fillers",
    { filepath: clipPath, custom_words: words.split(",").map(w => w.trim()), filler_backend: fillerBackend },
    (pct, msg) => { UIController.setProgress(pct); UIController.setProcessingMsg(msg || "Transcribing…"); },
    (result) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runFillerBtn", false);
      // /fillers default (whisper backend) returns ``segments_data`` /
      // ``segments`` — only the crisper backend returns ``cuts``. Without
      // this fallback, default-backend filler detection silently reported
      // 0 and never staged cuts.
      const cuts = (result.cuts && result.cuts.length)
        ? result.cuts
        : (result.segments_data || []);
      if (cuts.length) rememberTimelineCuts(cuts, { source: "Filler Detection", clipPath });
      const count = result.count ?? cuts.length ?? 0;
      UIController.showToast(`Detected ${count} filler word(s).`, "success");
      UIController.setStatus(`Filler detection done — ${count} removed.`);
      if (cuts.length) showCutResult({ ...result, cuts });
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
  UIController.showProcessing("Transcribing — this may take a while…");
  setCaptionsSessionState(
    "Working",
    "working",
    "Transcribing the selected clip. OpenCut will keep the last review output visible until this pass finishes.",
    "working",
    clipPath
  );
  setTextAndTitle("captionsOutputValue", "Processing transcript…", clipPath);
  syncCaptionsActionButtons();

  // The "captionStyle" select offers visual styles (youtube_bold, neon_pop,
  // ...) that have nothing to do with the /captions transcription format.
  // The backend's ``format`` param accepts only srt/vtt/json/ass and silently
  // coerces unknowns to srt — so the user's style choice was being lost.
  // Send a real format here; the visual style will be applied later via
  // /styled-captions if/when the user runs that flow.
  await JobPoller.start(
    "/captions",
    { filepath: clipPath, model, language: lang === "auto" ? null : lang,
      format: "srt", diarize, word_timestamps: wordLevel },
    (pct, msg) => { UIController.setProgress(pct); UIController.setProcessingMsg(msg || "Transcribing…"); },
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
      if (!_lastCaptionsResult) {
        setTextAndTitle("captionsOutputValue", "No transcript yet", "Transcription failed. Retry when ready.");
      }
      setCaptionsSessionState("Retry needed", "warning", `Transcription failed. ${err}`, "error", clipPath);
      syncCaptionsActionButtons();
    }
  );
}

function showCaptionsResult(result) {
  const content = result.srt ?? result.text ?? JSON.stringify(result, null, 2);
  const nonEmptyLines = content.split(/\r?\n/).filter(Boolean).length;
  const clipPath = document.getElementById("clipPathCaptions")?.value?.trim() || "";
  const style = getSelectLabel("captionStyle", "Selected style");
  const language = getSelectLabel("transcribeLang", "Auto-detect");

  renderCaptionsResultView({
    kind: "transcript",
    header: result.srt ? "Transcript & Subtitle Output" : "Transcript Review",
    summary: result.srt ? `${nonEmptyLines} caption lines ready` : `${nonEmptyLines} transcript lines ready`,
    content,
    resultPillText: result.srt ? "SRT ready" : "Transcript ready",
    resultPillState: "success",
    resultMeta: `${formatWorkspaceSource(clipPath)} • ${language} • ${style}`,
    resultMetaTitle: clipPath || "Transcript ready",
    copyLabel: result.srt ? "Copy SRT" : "Copy Transcript",
    importLabel: "Open SRT Import",
    canOpenSrtImport: !!result.srt,
    hasSrt: !!result.srt,
    sessionLabel: result.srt ? "Transcript ready" : "Review ready",
    sessionState: "success",
    statusMessage: result.srt
      ? "Transcript ready. Copy the SRT or open Timeline > SRT Prep when you're ready to validate it for CEP or native caption import."
      : "Transcript ready. Copy the text, draft chapters, or run a repeat review from the same clip.",
    statusState: "success",
    statusTitle: clipPath || "Transcript ready",
    outputLabel: result.srt
      ? `${nonEmptyLines} caption ${nonEmptyLines === 1 ? "line" : "lines"}`
      : `${nonEmptyLines} transcript ${nonEmptyLines === 1 ? "line" : "lines"}`,
    outputTitle: clipPath || "Transcript ready",
    insightType: result.srt ? "Transcript + SRT" : "Transcript",
    insightLength: result.srt
      ? `${nonEmptyLines} caption ${nonEmptyLines === 1 ? "line" : "lines"}`
      : `${nonEmptyLines} transcript ${nonEmptyLines === 1 ? "line" : "lines"}`,
    insightNext: result.srt ? "Copy SRT or open SRT Prep" : "Copy transcript or draft chapters",
  });
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
  UIController.showProcessing("Generating chapters with AI…");
  setCaptionsSessionState(
    "Working",
    "working",
    "Drafting chapter markers from the selected clip. The last review output will stay available until the new pass is ready.",
    "working",
    clipPath
  );
  setTextAndTitle("captionsOutputValue", "Drafting chapters…", clipPath);
  syncCaptionsActionButtons();

  await JobPoller.start(
    "/captions/chapters",
    { filepath: clipPath, llm_provider: provider, llm_model: model },
    (pct, msg) => { UIController.setProgress(pct); UIController.setProcessingMsg(msg || "Generating…"); },
    (result) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runChaptersBtn", false);
      const count = result.chapters?.length ?? 0;
      UIController.showToast(`Generated ${count} chapter(s).`, "success");
      UIController.setStatus(`Chapter generation complete — ${count} chapters.`);
      const clipTitle = clipPath || "Chapter output";
      const providerLabel = getSelectLabel("llmProvider", "Selected provider");
      const chapterContent = count
        ? result.chapters.map((c, i) =>
            `${formatTimecode(c.seconds ?? c.start ?? 0)} — ${c.title ?? `Chapter ${i + 1}`}`
          ).join("\n")
        : "No chapters were suggested for the current clip with the selected model.";
      renderCaptionsResultView({
        kind: "chapters",
        header: "Chapter Draft",
        summary: count ? `${count} chapter${count === 1 ? "" : "s"} ready` : "No chapters drafted",
        content: chapterContent,
        resultPillText: count ? "Chapters" : "Needs review",
        resultPillState: count ? "success" : "warning",
        resultMeta: `${formatWorkspaceSource(clipPath)} • ${providerLabel} • ${model}`,
        resultMetaTitle: clipTitle,
        copyLabel: "Copy Chapters",
        importLabel: "Open SRT Import",
        canOpenSrtImport: false,
        hasSrt: false,
        sessionLabel: count ? "Chapters ready" : "Needs review",
        sessionState: count ? "success" : "warning",
        statusMessage: count
          ? "Chapter draft is ready. Copy the list into publishing notes or rerun with a different model for a tighter structure."
          : "No chapters were suggested. Try another model or confirm the transcript has enough structure to segment cleanly.",
        statusState: count ? "success" : "warning",
        statusTitle: clipTitle,
        outputLabel: count ? `${count} chapter${count === 1 ? "" : "s"}` : "No chapters drafted",
        outputTitle: clipTitle,
        insightType: "Chapter draft",
        insightLength: count ? `${count} chapter${count === 1 ? "" : "s"}` : "No chapters drafted",
        insightNext: count ? "Copy into publishing notes" : "Try another model or refine the transcript",
      });
    },
    (err) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runChaptersBtn", false);
      UIController.showToast(`Chapter generation error: ${err}`, "error");
      if (!_lastCaptionsResult) {
        setTextAndTitle("captionsOutputValue", "No chapter draft", "Chapter generation failed. Retry when ready.");
      }
      setCaptionsSessionState("Retry needed", "warning", `Chapter generation failed. ${err}`, "error", clipPath);
      syncCaptionsActionButtons();
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
  UIController.showProcessing("Detecting repeated segments…");
  setCaptionsSessionState(
    "Working",
    "working",
    "Checking the current clip for duplicated spoken lines and alternate takes.",
    "working",
    clipPath
  );
  setTextAndTitle("captionsOutputValue", "Scanning for repeats…", clipPath);
  syncCaptionsActionButtons();

  await JobPoller.start(
    "/captions/repeat-detect",
    { filepath: clipPath, threshold, keep_best: keepBest },
    (pct, msg) => { UIController.setProgress(pct); UIController.setProcessingMsg(msg || "Analysing…"); },
    (result) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runRepeatBtn", false);
      const count = result.repeats?.length ?? result.removed_count ?? 0;
      UIController.showToast(`Detected ${count} repeat(s).`, "success");
      UIController.setStatus(`Repeat detection done — ${count} found.`);
      showRepeatResult(result);
    },
    (err) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runRepeatBtn", false);
      UIController.showToast(`Error: ${err}`, "error");
      if (!_lastCaptionsResult) {
        setTextAndTitle("captionsOutputValue", "No repeat review", "Repeat detection failed. Retry when ready.");
      }
      setCaptionsSessionState("Retry needed", "warning", `Repeat detection failed. ${err}`, "error", clipPath);
      syncCaptionsActionButtons();
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
  UIController.showProcessing("Applying noise reduction…");

  await JobPoller.start(
    "/audio/denoise",
    { filepath: clipPath, method, strength },
    (pct, msg) => { UIController.setProgress(pct); UIController.setProcessingMsg(msg || "Processing…"); },
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
  UIController.showProcessing("Normalizing audio…");

  await JobPoller.start(
    "/audio/normalize",
    { filepath: clipPath, target_lufs: targetLufs, true_peak: truePeak ? -1.0 : null },
    (pct, msg) => { UIController.setProgress(pct); UIController.setProcessingMsg(msg || "Normalizing…"); },
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
  UIController.showProcessing("Matching loudness to reference…");

  await JobPoller.start(
    "/audio/loudness-match",
    { files: [clipPath, refPath], target_lufs: -14.0 },
    (pct, msg) => { UIController.setProgress(pct); UIController.setProcessingMsg(msg || "Matching…"); },
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
  UIController.showProcessing("Detecting beats…");

  await JobPoller.start(
    "/audio/beat-markers",
    { filepath: trackPath, subdivisions: Math.max(1, Math.round(sensitivity * 4)) },
    (pct, msg) => { UIController.setProgress(pct); UIController.setProcessingMsg(msg || "Analysing tempo…"); },
    async (result) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runBeatMarkersBtn", false);

      const beats = result.beats ?? result.markers ?? [];
      rememberTimelineMarkers(beats, { source: "Beat Detection", clipPath: trackPath });
      UIController.showToast(`Detected ${beats.length} beats. Adding markers to timeline…`, "success");
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
  UIController.showProcessing("Matching color grading…");

  await JobPoller.start(
    "/video/color-match",
    { source: clipPath, reference: refPath, strength },
    (pct, msg) => { UIController.setProgress(pct); UIController.setProcessingMsg(msg || "Grading…"); },
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
  UIController.showProcessing("Applying auto zoom / reframe…");

  await JobPoller.start(
    "/video/auto-zoom",
    { filepath: clipPath, zoom_amount: maxZoom, easing: "ease_in_out" },
    (pct, msg) => { UIController.setProgress(pct); UIController.setProcessingMsg(msg || "Reframing…"); },
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
  UIController.showProcessing("Generating multicam cuts…");

  // Backend /video/multicam-cuts wants either ``segments`` (inline list),
  // ``diarization_file`` (a JSON file with diarization data), or a single
  // ``filepath`` it can transcribe to derive speakers from. Sending the
  // second camera path as ``diarization_file`` always 400s with
  // "Could not read diarization_file" because it's a video, not JSON.
  // The right flow: transcribe cam1 to derive speaker turns, then map
  // speaker → camera afterwards. cam2 is needed only for the eventual
  // timeline-side multicam application, not for the cut generation pass.
  await JobPoller.start(
    "/video/multicam-cuts",
    { filepath: cam1Path, min_cut_duration: 1.0 },
    (pct, msg) => { UIController.setProgress(pct); UIController.setProcessingMsg(msg || "Analysing cameras…"); },
    async (result) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runMulticamBtn", false);
      const cuts = result.cuts ?? [];
      rememberTimelineCuts(cuts, { source: "Multicam Cut Pass", clipPath: cam1Path });
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
    noteTimelineAction(
      "Cuts unavailable",
      "warning",
      "No cuts are staged for sequence write-back yet. Run silence, filler, or multicam cleanup first.",
      "No cuts are staged for sequence write-back yet."
    );
    return;
  }

  if (PProBridge.available()) {
  UIController.setStatus("Applying cuts to timeline via UXP…");
    const result = await PProBridge.applyCuts(cutsToApply);
    if (result.ok) {
      UIController.showToast(`Applied ${result.applied} cut(s) to active sequence.`, "success");
      UIController.setStatus(`Applied ${result.applied} cut(s).`);
      noteTimelineAction(
        "Cuts applied",
        "success",
        `Applied ${result.applied} cut${result.applied === 1 ? "" : "s"} to the active sequence.`,
        `Applied ${result.applied} cut${result.applied === 1 ? "" : "s"} to the active sequence.`,
        `${result.applied} cut${result.applied === 1 ? "" : "s"} applied`
      );
    } else {
      UIController.showToast(
        `UXP timeline write failed: ${result.reason}. Use CEP panel for Premiere < 25.6.`,
        "warning"
      );
      UIController.setStatus("Timeline write failed — see CEP panel.");
      noteTimelineAction(
        "CEP fallback needed",
        "warning",
        `Direct UXP write-back failed. ${result.reason}. Use the CEP panel for this sequence pass.`,
        result.reason || "Direct UXP write-back failed."
      );
    }
  } else {
    UIController.showToast(
      "Connect via CEP panel for timeline operations (UXP timeline API in preview).",
      "info"
    );
    UIController.setStatus("UXP timeline API unavailable — use CEP panel.");
    noteTimelineAction(
      "CEP fallback needed",
      "warning",
      "Direct sequence write-back is not available in this UXP session. Use the CEP panel for timeline operations.",
      "Direct sequence write-back is not available in this UXP session."
    );
  }
}

/** ── ADD SEQUENCE MARKERS (UXP + fallback) ── */
async function addSequenceMarkers(markers, color) {
  const markersToAdd = markers ?? lastMarkers;
  if (!markersToAdd || markersToAdd.length === 0) {
    UIController.showToast("No markers to add. Run beat detection first.", "warning");
    noteTimelineAction(
      "Markers unavailable",
      "warning",
      "No markers are staged for sequence write-back yet. Run beat detection first.",
      "No markers are staged for sequence write-back yet."
    );
    return;
  }

  const markerColor = color ?? document.getElementById("beatMarkerColor")?.value ?? "green";
  const formatted = markersToAdd.map(m => ({
    time:  typeof m === "number" ? m : (m.time ?? m.t ?? 0),
    label: m.label ?? "Beat",
    color: markerColor,
  }));

  if (PProBridge.available()) {
  UIController.setStatus("Adding markers to sequence via UXP…");
    const result = await PProBridge.addMarkers(formatted);
    if (result.ok) {
      UIController.showToast(`Added ${result.count} marker(s) to active sequence.`, "success");
      UIController.setStatus(`Added ${result.count} marker(s).`);
      noteTimelineAction(
        "Markers added",
        "success",
        `Added ${result.count} marker${result.count === 1 ? "" : "s"} to the active sequence.`,
        `Added ${result.count} marker${result.count === 1 ? "" : "s"} to the active sequence.`,
        `${result.count} marker${result.count === 1 ? "" : "s"} added`
      );
    } else {
      UIController.showToast(
        `UXP marker insertion failed: ${result.reason}. Use CEP panel as fallback.`,
        "warning"
      );
      noteTimelineAction(
        "CEP fallback needed",
        "warning",
        `Marker insertion failed in UXP. ${result.reason}. Use the CEP panel as fallback.`,
        result.reason || "Marker insertion failed in UXP."
      );
    }
  } else {
    UIController.showToast(
      "Connect via CEP panel for timeline operations (UXP timeline API in preview).",
      "info"
    );
    noteTimelineAction(
      "CEP fallback needed",
      "warning",
      "Marker insertion is not available in this UXP session. Use the CEP panel for timeline operations.",
      "Marker insertion is not available in this UXP session."
    );
  }
}

/** ── BATCH EXPORT ── */
async function runBatchExport() {
  const preset    = document.getElementById("exportPreset")?.value ?? "youtube";
  const outputDir = document.getElementById("exportDir")?.value?.trim();
  if (!outputDir) { UIController.showToast("Please select an output folder.", "warning"); return; }

  const clipPath = document.getElementById("clipPathVideo")?.value?.trim() ?? "";
  const markersToExport = buildExportWindows();
  if (!clipPath) { UIController.showToast("Please select a clip first.", "warning"); return; }
  if (markersToExport.length === 0) { UIController.showToast("No markers or cuts to export. Run beat detection or silence removal first.", "warning"); return; }

  UIController.setButtonLoading("runBatchExportBtn", true);
  UIController.showProcessing("Starting batch export from markers…");

  await JobPoller.start(
    "/timeline/export-from-markers",
    { input_file: clipPath, markers: markersToExport, output_dir: outputDir, format: preset },
    (pct, msg) => { UIController.setProgress(pct); UIController.setProcessingMsg(msg || "Exporting…"); },
    (result) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runBatchExportBtn", false);
      const count = result.exported ?? result.count ?? 0;
      UIController.showToast(`Exported ${count} segment(s).`, "success");
      UIController.setStatus(`Batch export done — ${count} files.`);
      noteTimelineAction(
        "Batch export complete",
        "success",
        `Marker-based export finished with ${count} segment${count === 1 ? "" : "s"} in ${outputDir}.`,
        outputDir,
        `${count} export${count === 1 ? "" : "s"} ready`
      );
    },
    (err) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runBatchExportBtn", false);
      UIController.showToast(`Export error: ${err}`, "error");
      noteTimelineAction("Batch export error", "error", `Marker-based export failed. ${err}`, err);
    }
  );
}

/** ── BATCH RENAME ── */
async function runBatchRename() {
  const pattern = document.getElementById("renamePattern")?.value?.trim() ?? "{name}_{index:03d}";
  UIController.showToast("Batch rename still runs through the CEP panel in this build.", "info");
  noteTimelineAction(
    "Rename via CEP",
    "warning",
    "Batch rename is planned from this workspace, but execution still lives in the CEP panel today.",
    pattern,
    "Rename handoff"
  );
}

/** ── SMART BINS ── */
async function runSmartBins() {
  const strategy = getSelectLabel("binStrategy", "File Type");
  UIController.showToast("Smart bins still execute through the CEP panel in this build.", "info");
  noteTimelineAction(
    "Smart bins via CEP",
    "warning",
    "Smart bin rules can be planned here, but execution still lives in the CEP panel today.",
    strategy,
    "Smart bin handoff"
  );
}

/** ── SRT IMPORT ── */
async function runSrtImport() {
  const srtPath    = document.getElementById("srtFilePath")?.value?.trim();
  const trackIndex = parseInt(document.getElementById("srtTrackIndex")?.value ?? 1);
  if (!srtPath) { UIController.showToast("Please select an SRT file.", "warning"); return; }

  UIController.setButtonLoading("runSrtImportBtn", true);
  UIController.showProcessing("Validating SRT for timeline import…");

  await JobPoller.start(
    "/timeline/srt-to-captions",
    { srt_path: srtPath, track_index: trackIndex },
    (pct, msg) => { UIController.setProgress(pct); UIController.setProcessingMsg(msg || "Validating…"); },
    (result) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runSrtImportBtn", false);
      const count = result.segments?.length ?? result.captions_imported ?? result.count ?? 0;
      UIController.showToast(`Validated ${count} caption segment(s).`, "success");
      UIController.setStatus(`SRT ready — ${count} caption segments parsed.`);
      noteTimelineAction(
        "SRT validated",
        "success",
        `SRT validation is ready. Use the CEP or native captions flow to place ${count} caption segment${count === 1 ? "" : "s"} on track ${trackIndex}.`,
        srtPath,
        `${count} caption segment${count === 1 ? "" : "s"} parsed`
      );
    },
    (err) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runSrtImportBtn", false);
      UIController.showToast(`SRT validation error: ${err}`, "error");
      noteTimelineAction("SRT validation error", "error", `SRT validation failed. ${err}`, err);
    }
  );
}

/** ── INDEX LIBRARY ── */
async function runIndexLibrary() {
  const folder       = document.getElementById("indexFolder")?.value?.trim();
  if (!folder) {
    setStatusPill("indexStatePill", "Needs Folder", "warning", "Choose a media folder before building the search index.");
    setIndexStatus("Choose a media folder before building the search index.", "warning");
    UIController.showToast("Please select a media folder to index.", "warning");
    syncSearchPanelState();
    return;
  }

  const statusLine = document.getElementById("indexStatus");
  UIController.setButtonLoading("runIndexLibBtn", true);
  UIController.showProcessing("Indexing media library…");
  setStatusPill("indexStatePill", "Indexing", "working", folder);
  setIndexStatus("Indexing the media library…", "working", folder);
  if (statusLine) statusLine.textContent = "Indexing the media library…";

  await JobPoller.start(
    "/search/index",
    { folder, model: "base" },
    (pct, msg) => {
      UIController.setProgress(pct);
      UIController.setProcessingMsg(msg || "Scanning…");
      if (statusLine) statusLine.textContent = msg || "Scanning…";
    },
    async (result) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runIndexLibBtn", false);
      const count = result.indexed ?? result.files ?? 0;
      const errorCount = Array.isArray(result.errors) ? result.errors.length : 0;
      const pillState = errorCount ? "warning" : (count > 0 ? "success" : "empty");
      const pillLabel = errorCount ? "Needs Review" : (count > 0 ? "Ready" : "Empty");
      await refreshFootageIndexStats({ preserveMessage: true, silent: true });
      setStatusPill(
        "indexStatePill",
        pillLabel,
        pillState,
        `${count} ${count === 1 ? "file" : "files"} indexed.`
      );
      setIndexStatus(
        errorCount
          ? `Library indexed with a few skips. ${count} ${count === 1 ? "file is" : "files are"} ready to search, and ${errorCount} ${errorCount === 1 ? "item needs" : "items need"} attention.`
          : `Library indexed. ${count} ${count === 1 ? "file is" : "files are"} ready to search.`,
        errorCount ? "warning" : "success"
      );
      UIController.showToast(`Library indexed — ${count} files.`, "success");
      UIController.setStatus(`Library indexed — ${count} files.`, "success");
      setTextAndTitle(
        "searchStatus",
        count > 0
          ? "The library is ready. Search with descriptive phrases, then load the best match into the workspace."
          : "Index another folder or broaden the source media to make search more useful.",
        folder
      );
    },
    (err) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runIndexLibBtn", false);
      setStatusPill("indexStatePill", "Error", "error", err);
      setIndexStatus(`Could not index the library. ${err}`, "error");
      if (statusLine) statusLine.textContent = `Could not index the library. ${err}`;
      UIController.showToast(`Index error: ${err}`, "error");
      syncSearchPanelState();
    }
  );
}

/** ── FOOTAGE SEARCH ── */
async function runFootageSearch() {
  const query = document.getElementById("searchQuery")?.value?.trim();
  const limit = parseInt(document.getElementById("searchResultCount")?.value ?? 10);
  const backendOnline = document.getElementById("connLabel")?.textContent?.trim() === "Online";
  const indexedFiles = Number(_lastIndexStats?.total_files || 0);
  if (!backendOnline) {
    setTextAndTitle("searchStatus", "Reconnect the backend before searching the library.", "Reconnect the backend before searching the library.");
    UIController.showToast("Reconnect the backend before searching the library.", "warning");
    syncSearchPanelState();
    return;
  }
  if (indexedFiles <= 0) {
    setTextAndTitle("searchStatus", "Index a folder before searching the library.", "Index a folder before searching the library.");
    UIController.showToast("Index a folder before searching the library.", "warning");
    syncSearchPanelState();
    return;
  }
  if (!query) {
    setTextAndTitle("searchStatus", "Enter a descriptive query to search the indexed library.", "Enter a descriptive query to search the indexed library.");
    UIController.showToast("Please enter a search query.", "warning");
    syncSearchPanelState();
    return;
  }

  UIController.setButtonLoading("runFootageSearchBtn", true);
  UIController.setStatus("Searching footage…", "working");
  setTextAndTitle("searchStatus", `Searching for "${query}"…`, `Searching for "${query}"…`);

  const r = await BackendClient.post("/search/footage", { query, top_k: limit });

  UIController.setButtonLoading("runFootageSearchBtn", false);

  if (!r.ok) {
    setTextAndTitle("searchStatus", `Could not search the library. ${r.error}`, r.error);
    UIController.showToast(`Search error: ${r.error}`, "error");
    UIController.setStatus("Search failed. Review the query or reconnect the backend.", "error");
    syncSearchPanelState();
    return;
  }

  const results = r.data?.results ?? r.data ?? [];
  const list    = document.getElementById("searchResultList");
  const cards = Array.isArray(results) ? results.map((item, index) => buildSearchResultCard(item, index)) : [];
  if (list) {
    list.innerHTML = "";
    if (cards.length === 0) {
      list.innerHTML = `
        <div class="oc-empty-state oc-empty-state-inline">
          <div class="oc-empty-state-kicker">No matches yet</div>
          <p>Try more descriptive language, quote a spoken phrase, or index a broader folder to widen the search space.</p>
        </div>`;
    } else {
      cards.forEach((card, index) => {
        const el = document.createElement("button");
        el.type = "button";
        el.className = `oc-result-list-item${index === 0 ? " is-featured" : ""}`;
        el.innerHTML = `
          <span class="oc-result-list-item-copy">
            <span class="oc-result-list-item-topline">
              <span class="oc-result-list-item-title">${UIController.escapeHtml(card.label)}</span>
              <span class="oc-result-list-item-badge">${UIController.escapeHtml(card.kindLabel)}</span>
              ${card.timeLabel ? `<span class="oc-result-list-item-badge">${UIController.escapeHtml(card.timeLabel)}</span>` : ""}
            </span>
            <span class="oc-result-list-item-path">${UIController.escapeHtml(card.path || "Path unavailable for this result")}</span>
            ${card.preview ? `<span class="oc-result-list-item-excerpt">${UIController.escapeHtml(card.preview)}</span>` : ""}
          </span>
          <span class="oc-result-list-item-meta">${UIController.escapeHtml(card.scoreLabel)}</span>`;
        el.title = card.path || card.preview || card.label;
        el.addEventListener("click", () => {
          if (!card.path) {
            setTextAndTitle(
              "searchStatus",
              "This match does not include a loadable file path yet. Refine the query or index a folder with source media.",
              card.label
            );
            UIController.showToast("This result does not include a loadable file path yet.", "warning");
            return;
          }
          list.querySelectorAll(".oc-result-list-item").forEach((node) => {
            node.classList.remove("is-selected");
            node.setAttribute("aria-pressed", "false");
          });
          el.classList.add("is-selected");
          el.setAttribute("aria-pressed", "true");
          setWorkspaceClip(card.path, { tabId: "search" });
          setTextAndTitle(
            "searchStatus",
            `Loaded ${card.label} into the workspace. Search results stay visible while you compare alternate shots.`,
            card.path
          );
          UIController.showToast(`Loaded ${card.label} into the workspace.`, "success");
          UIController.setStatus(`Search match loaded — ${card.label}.`, "success");
        });
        el.setAttribute("aria-pressed", "false");
        list.appendChild(el);
      });
    }
  }

  setTextAndTitle(
    "searchStatus",
    cards.length
      ? `${cards.length} ${cards.length === 1 ? "match is" : "matches are"} ready. Start with ${cards[0].label}, or compare results before loading a shot back into the workspace.`
      : "No matches yet. Try more descriptive language, or index a broader folder.",
    query
  );
  UIController.setStatus(
    cards.length
      ? `Search ready — ${cards.length} result${cards.length === 1 ? "" : "s"}.`
      : "Search returned no matches."
  );
  syncSearchPanelState();
}

/** ── NLP COMMAND ── */
async function runNlpCommand() {
  const llmProvider = window._llmSettings?.provider || "ollama";
  const command  = document.getElementById("nlpCommand")?.value?.trim();
  const provider = document.getElementById("nlpLlmProvider")?.value ?? llmProvider;
  const backendOnline = document.getElementById("connLabel")?.textContent?.trim() === "Online";
  if (!backendOnline) {
    UIController.showToast("Reconnect the backend before running natural-language commands.", "warning");
    syncSearchPanelState();
    return;
  }
  if (!command) { UIController.showToast("Please enter a natural language command.", "warning"); return; }

  UIController.setButtonLoading("runNlpBtn", true);
  UIController.showProcessing("Parsing command with AI…");

  await JobPoller.start(
    "/nlp/command",
    { command, llm_provider: provider },
    (pct, msg) => { UIController.setProgress(pct); UIController.setProcessingMsg(msg || "Thinking…"); },
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
  const summary = document.getElementById("nlpResultSummary");
  if (!area || !body) return;
  const action = result.action ?? result;
  area.classList.remove("oc-hidden");
  body.textContent = JSON.stringify(action, null, 2);
  if (summary) {
    if (Array.isArray(action?.cuts) && action.cuts.length) {
      summary.textContent = `${action.cuts.length} cut${action.cuts.length === 1 ? "" : "s"} ready`;
    } else if (Array.isArray(action?.markers) && action.markers.length) {
      summary.textContent = `${action.markers.length} marker${action.markers.length === 1 ? "" : "s"} ready`;
    } else {
      summary.textContent = "Review before applying";
    }
  }
  area.focus();
}

/** ── LOAD SEQUENCE INFO ── */
async function loadSequenceInfo() {
  UIController.setButtonLoading("loadSeqInfoBtn", true);
  UIController.setStatus("Loading sequence info…", "working");

  const info = await ensureSequenceInfo({ force: true, silent: true });

  UIController.setButtonLoading("loadSeqInfoBtn", false);

  const grid = document.getElementById("seqInfoGrid");
  if (!grid) return;

  if (!info) {
    grid.innerHTML = `
      <div class="oc-empty-state oc-empty-state-inline">
        <div class="oc-empty-state-kicker">No active sequence</div>
        <p>Open a sequence in Premiere, then reload sequence info here before generating deliverables.</p>
      </div>`;
    setDeliverablesStatus("No active sequence loaded. Open a Premiere sequence, then refresh this card before generating deliverables.", "warning");
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
    `<div class="oc-info-pair">` +
      `<span class="oc-info-key">${UIController.escapeHtml(k)}</span>` +
      `<span class="oc-info-val">${UIController.escapeHtml(String(v ?? "—"))}</span>` +
    `</div>`
  ).join("");

  updateDeliverablesSummary();
  UIController.setStatus(`Sequence ready — ${info.name ?? "Active sequence"}`, "success");
}

/** ── DELIVERABLES ── */
async function runDeliverables(type) {
  const deliverableLabel = DELIVERABLE_LABELS[type] || humanizeDomain(type);
  const backendOnline = document.getElementById("connLabel")?.textContent?.trim() === "Online";
  if (!backendOnline) {
    setDeliverablesStatus(`Reconnect the backend before generating ${deliverableLabel}.`, "error");
    UIController.showToast("Reconnect the backend before generating deliverables.", "warning");
    UIController.setStatus("Reconnect the backend before generating deliverables.", "error");
    return;
  }
  const seqData = await ensureSequenceInfo({ silent: true });
  const outputDir = document.getElementById("delivOutputDir")?.value?.trim();
  if (!seqData) {
    setDeliverablesStatus(`Load the active Premiere sequence before generating ${deliverableLabel}.`, "warning");
    UIController.showToast("Load sequence info before generating deliverables.", "warning");
    UIController.setStatus("Load sequence info before generating deliverables.", "error");
    return;
  }

  const btnId = DELIVERABLE_BUTTON_IDS[type];
  if (btnId) UIController.setButtonLoading(btnId, true);
  setDeliverablesStatus(`Generating ${deliverableLabel}…`, "working");
  UIController.showProcessing(`Generating ${deliverableLabel}…`);

  await JobPoller.start(
    `/deliverables/${type.replace(/_/g, "-")}`,
    { sequence_data: seqData, output_dir: outputDir || null },
    (pct, msg) => { UIController.setProgress(pct); UIController.setProcessingMsg(msg || "Generating…"); },
    (result) => {
      UIController.hideProcessing();
      if (btnId) UIController.setButtonLoading(btnId, false);
      const outputPath = result.output ?? result.output_path ?? "";
      const outputLabel = outputPath ? formatWorkspaceSource(outputPath) : "saved";
      const documentCount = Math.max(
        1,
        Number(
          result.count
          ?? result.generated
          ?? result.documents?.length
          ?? result.outputs?.length
          ?? result.files?.length
          ?? 1
        ) || 1
      );
      _lastDeliverableActivity = {
        label: deliverableLabel,
        output: outputPath,
        time: Date.now(),
        count: documentCount,
      };
      updateDeliverablesSummary();
      setDeliverablesStatus(`${deliverableLabel} is ready. Review the file and continue building the handoff package.`, "success", outputPath || deliverableLabel);
      UIController.showToast(
        `${deliverableLabel} ready: ${outputLabel}`,
        "success"
      );
      UIController.setStatus(`${deliverableLabel} generated.`, "success");
    },
    (err) => {
      UIController.hideProcessing();
      if (btnId) UIController.setButtonLoading(btnId, false);
      setDeliverablesStatus(`Could not generate ${deliverableLabel}. ${err}`, "error");
      UIController.showToast(`Deliverable error: ${err}`, "error");
    }
  );
}

/** ── FULL PROJECT REPORT (generates all 4 deliverables) ── */
async function runFullReport() {
  const backendOnline = document.getElementById("connLabel")?.textContent?.trim() === "Online";
  if (!backendOnline) {
    setDeliverablesStatus("Reconnect the backend before generating the report package.", "error");
    UIController.showToast("Reconnect the backend before generating the report package.", "warning");
    UIController.setStatus("Reconnect the backend before generating the report package.", "error");
    return;
  }
  const seqData = await ensureSequenceInfo({ silent: true });
  const outputDir = document.getElementById("delivOutputDir")?.value?.trim();
  const selectedTypes = getSelectedReportTypes();
  const packageSummary = getDeliverablesSelectionSummary(selectedTypes);
  const formatSummary = getDeliverablesFormatSummary();
  if (!seqData) {
    setDeliverablesStatus("Load the active Premiere sequence before generating the full report.", "warning");
    UIController.showToast("Load sequence info before generating the full report.", "warning");
    UIController.setStatus("Load sequence info before generating the full report.", "error");
    return;
  }
  if (!selectedTypes.length) {
    setDeliverablesStatus("Select at least one handoff document before generating the report package.", "warning");
    UIController.showToast("Select at least one handoff document before generating the report package.", "warning");
    UIController.setStatus("Select documents before generating the report package.", "warning");
    updateDeliverablesSummary();
    return;
  }

  const types = selectedTypes.map((type) => type.replace(/_/g, "-"));
  const packageLabel = selectedTypes.length === 1
    ? (DELIVERABLE_LABELS[selectedTypes[0]] || "Selected Report")
    : (selectedTypes.length === 4 ? "Full Report" : `${selectedTypes.length}-Doc Package`);

  UIController.setButtonLoading("runFullReportBtn", true);
  setDeliverablesStatus(`Generating ${packageLabel}…`, "working", packageSummary.title);
  UIController.showProcessing(`Generating ${packageLabel}…`);
  UIController.setProgress(0);

  let generated = 0;
  let errors = 0;
  const outputPaths = [];
  for (let index = 0; index < types.length; index += 1) {
    const type = types[index];
    const label = DELIVERABLE_LABELS[type.replace(/-/g, "_")] || humanizeDomain(type);
    UIController.setProgress(Math.round((index / types.length) * 100));
    UIController.setProcessingMsg(`Generating ${label} (${index + 1}/${types.length})…`);
    const r = await BackendClient.post(`/deliverables/${type}`, {
      sequence_data: seqData,
      output_dir: outputDir || null,
      format: formatSummary.value,
    });
    if (r.ok) {
      generated++;
      if (r.data?.output) outputPaths.push(r.data.output);
    } else {
      errors++;
    }
  }

  UIController.hideProcessing();
  UIController.setButtonLoading("runFullReportBtn", false);
  _lastDeliverableActivity = {
    label: packageLabel,
    output: outputPaths[0] || outputDir || "",
    time: Date.now(),
    count: generated,
  };
  updateDeliverablesSummary();
  if (errors === 0) {
    setDeliverablesStatus(`${packageLabel} ready. ${generated} ${generated === 1 ? "document is" : "documents are"} available for review.`, "success", outputPaths[0] || outputDir || packageSummary.title);
    UIController.showToast(`Generated ${generated} CSV handoff ${generated === 1 ? "document" : "documents"}.`, "success");
  } else {
    setDeliverablesStatus(`${packageLabel} generated with a few gaps. ${generated} documents completed and ${errors} ${errors === 1 ? "step needs" : "steps need"} attention.`, "warning", outputPaths[0] || outputDir || packageSummary.title);
    UIController.showToast(`Generated ${generated} CSV documents; ${errors} ${errors === 1 ? "step needs" : "steps need"} attention.`, "warning");
  }
  UIController.setStatus(`${packageLabel} ready — ${generated} CSV ${generated === 1 ? "document" : "documents"}.`);
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
  UIController.showProcessing("Generating AI B-roll…");
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
  UIController.showProcessing("Running multimodal diarization…");
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
  UIController.showProcessing(`Uploading to ${platform}…`);
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

function formatCompactDuration(seconds) {
  if (typeof seconds !== "number" || isNaN(seconds) || seconds <= 0) return "0 s";
  if (seconds < 1) return `${Math.round(seconds * 1000)} ms`;
  if (seconds < 60) return `${seconds.toFixed(seconds >= 10 ? 1 : 2)} s`;
  const minutes = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60);
  return `${minutes}m ${secs}s`;
}

function humanizeDomain(domain) {
  return String(domain || "")
    .split("_")
    .filter(Boolean)
    .map(part => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function formatLocaleTime(value) {
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) return "just now";
  return new Intl.DateTimeFormat(undefined, {
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
}

function formatRelativeTime(value) {
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) return "just now";
  const diffMs = date.getTime() - Date.now();
  const absMinutes = Math.abs(diffMs) / 60000;
  const formatter = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });
  if (absMinutes < 1) return "just now";
  if (absMinutes < 60) return formatter.format(Math.round(diffMs / 60000), "minute");
  const absHours = Math.abs(diffMs) / 3600000;
  if (absHours < 24) return formatter.format(Math.round(diffMs / 3600000), "hour");
  return formatter.format(Math.round(diffMs / 86400000), "day");
}

function formatBytes(bytes) {
  if (typeof bytes !== "number" || !isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(value >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function setTextAndTitle(id, text, title) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = text;
  el.title = title || text;
}

function setStatusPill(id, text, state, title) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = text;
  el.dataset.state = state;
  el.title = title || text;
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

  if (alive) {
    UIController.setStatus("OpenCut backend connected.", "success");
    UIController.setStatusRight(`v${VERSION}`);
  } else {
    UIController.setStatus("OpenCut backend offline. Start the local service to run jobs.", "error");
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

  updateWorkspaceOverview();
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
  const tabs = Array.from(document.querySelectorAll(".oc-tab"));
  tabs.forEach((btn, index) => {
    btn.addEventListener("click", () => UIController.switchTab(btn.dataset.tab));
    btn.addEventListener("keydown", (event) => {
      let target = null;
      if (event.key === "ArrowRight" || event.key === "ArrowDown") {
        target = tabs[(index + 1) % tabs.length];
      } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
        target = tabs[(index - 1 + tabs.length) % tabs.length];
      } else if (event.key === "Home") {
        target = tabs[0];
      } else if (event.key === "End") {
        target = tabs[tabs.length - 1];
      }

      if (target) {
        event.preventDefault();
        target.focus();
        UIController.switchTab(target.dataset.tab);
      }
    });
  });

  PRIMARY_CLIP_INPUT_IDS.forEach((id) => {
    const input = document.getElementById(id);
    if (!input) return;
    input.addEventListener("input", () => {
      if (_syncingWorkspaceClip) return;
      _workspaceClipPath = input.value.trim();
      updateWorkspaceOverview();
    });
    input.addEventListener("change", () => {
      if (_syncingWorkspaceClip) return;
      setWorkspaceClip(input.value, { originId: id });
    });
  });

  ["whisperModel", "transcribeLang", "captionStyle", "enableDiarization", "enableWordLevel"].forEach((id) => {
    const control = document.getElementById(id);
    if (!control) return;
    control.addEventListener("change", () => updateCaptionsWorkspaceSummary());
  });

  ["exportDir", "srtFilePath", "renamePattern"].forEach((id) => {
    const control = document.getElementById(id);
    if (!control) return;
    control.addEventListener("input", () => updateTimelineReadiness());
    control.addEventListener("change", () => updateTimelineReadiness());
  });

  ["renameScope", "binStrategy", "srtTrackIndex", "exportPreset", "beatMarkerColor", "otioExportMode"].forEach((id) => {
    const control = document.getElementById(id);
    if (!control) return;
    control.addEventListener("change", () => updateTimelineReadiness());
  });

  ["indexFolder", "searchQuery", "nlpCommand"].forEach((id) => {
    const control = document.getElementById(id);
    if (!control) return;
    control.addEventListener("input", () => syncSearchPanelState());
    control.addEventListener("change", () => syncSearchPanelState());
  });

  ["rptIncludeVfx", "rptIncludeAdr", "rptIncludeMusic", "rptIncludeAssets", "reportFormat"].forEach((id) => {
    const control = document.getElementById(id);
    if (!control) return;
    control.addEventListener("change", () => updateDeliverablesSummary());
  });

  // ── Refresh button ──
  document.getElementById("refreshBtn")?.addEventListener("click", () => checkConnection());
  document.getElementById("workspaceChooseClipBtn")?.addEventListener("click", () => handleWorkspaceAction("choose-clip"));
  document.getElementById("workspaceSearchBtn")?.addEventListener("click", () => handleWorkspaceAction("open-search"));
  document.getElementById("workspaceTimelineBtn")?.addEventListener("click", () => handleWorkspaceAction("open-timeline"));
  document.getElementById("workspaceGuideAction")?.addEventListener("click", (event) => {
    handleWorkspaceAction(event.currentTarget?.dataset?.action || "");
  });

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
  document.getElementById("copySrtBtn")?.addEventListener("click", async () => {
    const body = document.getElementById("captionsResultBody");
    if (body?.value) {
      const copiedLabel = (_lastCaptionsResult?.copyLabel || "Copy Output").replace(/^Copy\s+/, "");
      await copyTextToClipboard(body.value, { successLabel: copiedLabel });
    }
  });
  document.getElementById("importSrtBtn")?.addEventListener("click", async () => {
    if (!(_lastCaptionsResult && _lastCaptionsResult.kind === "transcript" && _lastCaptionsResult.hasSrt)) {
      UIController.showToast("Timeline import is only available when an SRT transcript is ready.", "info");
      return;
    }
    UIController.switchTab("timeline");
    focusControl("srtFilePath");
    UIController.setStatus("Timeline SRT prep ready.", "working");
    UIController.showToast("Choose the saved .srt file, then validate it for CEP or native captions import.", "info");
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
      noteTimelineAction(
        "OTIO exported",
        "success",
        "OTIO export is ready for Resolve, Final Cut, Avid, or any OTIO-compatible tool.",
        r.data?.output_path || "OTIO export"
      );
    } else {
      UIController.showToast(`OTIO export failed: ${r.error}`, "error");
      noteTimelineAction("OTIO export error", "error", `OTIO export failed. ${r.error}`, r.error);
    }
  });

  // ── Search ──
  document.getElementById("browseIndexFolder")?.addEventListener("click", () => browseFolder("indexFolder"));
  document.getElementById("runIndexLibBtn")?.addEventListener("click",    runIndexLibrary);
  document.getElementById("clearIndexBtn")?.addEventListener("click",     clearFootageIndex);
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
  document.getElementById("nlpCommand")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") runNlpCommand();
  });
  document.querySelectorAll("[data-fill-target]").forEach((button) => {
    button.addEventListener("click", () => fillFieldFromSuggestion(button));
  });

  // ── Deliverables ──
  document.getElementById("browseDelivDir")?.addEventListener("click", () => browseFolder("delivOutputDir"));
  document.getElementById("delivOutputDir")?.addEventListener("input", updateDeliverablesSummary);
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
  if (r.ok) UIController.showToast("Installing Depth Anything V2…", "info");
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
  document.getElementById("uxpRefreshMigrationRiskBtn")?.addEventListener("click", uxpLoadMigrationRisk);
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
  UIController.showProcessing("Running depth effect…");
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
  UIController.showProcessing("Analyzing emotions…");
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
  UIController.showProcessing("Analyzing B-roll points…");
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
UIController.showToast(`Executing ${actions.length} action(s)…`, "info");
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
    UIController.showToast("Live updates are already connected.", "info");
    return;
  }
  try {
    _uxpWs = new WebSocket("ws://127.0.0.1:5680");
  } catch (e) {
    UIController.showToast("Could not open the live-updates bridge.", "warning");
    return;
  }

  _uxpWs.onopen = () => {
    _uxpWsConnected = true;
    _uxpWs.send(JSON.stringify({ type: "identify", client_type: "uxp", id: "uxp-1" }));
    _uxpWs.send(JSON.stringify({ type: "command", action: "subscribe", params: { events: ["progress", "job_complete", "job_error"] }, id: "sub-1" }));
    uxpUpdateWsStatus();
    UIController.setStatus("Live updates connected.", "success");
    UIController.showToast("Live updates connected.", "success");
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
  const panelStateEl = document.getElementById("uxpWsPanelState");
  const connectBtn = document.getElementById("uxpWsConnectBtn");
  const startBtn = document.getElementById("uxpWsStartBtn");
  const stopBtn = document.getElementById("uxpWsStopBtn");
  let statusText = _uxpWsConnected ? "Live updates connected" : "Bridge unavailable";
  let statusState = _uxpWsConnected ? "connected" : "unknown";
  let panelStateText = _uxpWsConnected ? "Connected" : "Waiting to connect";
  let bridgeRunning = false;
  let clients = 0;
  let statusDetail = "Start the bridge, then connect this panel to receive live job updates.";

  const r = await BackendClient.get("/ws/status");
  if (r.ok && r.data) {
    bridgeRunning = !!r.data.running;
    clients = Number(r.data.clients || 0);
    if (_uxpWsConnected) {
      statusText = clients > 0 ? "Live updates connected" : "Panel connected";
      statusState = "connected";
      panelStateText = "Connected";
      statusDetail = "Live updates are flowing into this panel. Long-running jobs can report progress, completion, and cancel feedback here.";
    } else if (bridgeRunning) {
      statusText = clients > 0 ? "Bridge ready" : "Bridge idle";
      statusState = "ready";
      panelStateText = "Ready to connect";
      statusDetail = "The bridge is running, but this panel is not attached yet. Connect live updates to bring job progress back into the workspace.";
    } else {
      statusText = "Bridge stopped";
      statusState = "stopped";
      panelStateText = "Not linked";
      statusDetail = "Start the live-updates bridge to bring long-running job feedback back into the panel.";
    }
  } else if (!_uxpWsConnected) {
    statusText = "Bridge unavailable";
    statusState = "error";
    panelStateText = "Unavailable";
    statusDetail = "The panel could not read bridge status right now. Refresh the backend, then try again.";
  }

  if (statusEl) {
    statusEl.textContent = statusText;
    statusEl.dataset.state = statusState;
    statusEl.title = statusDetail;
  }
  if (countEl) {
    countEl.textContent = `${clients} ${clients === 1 ? "listener" : "listeners"}`;
    countEl.dataset.state = clients > 0 ? "active" : "idle";
    countEl.title = clients
      ? `${clients} ${clients === 1 ? "listener is" : "listeners are"} currently attached to the live-updates bridge.`
      : "No listeners are attached to the live-updates bridge right now.";
  }
  if (panelStateEl) {
    panelStateEl.textContent = panelStateText;
    panelStateEl.title = statusDetail;
  }
  setSettingsStatus("settingsBridgeStatus", statusDetail, statusState === "connected" ? "success" : (statusState === "error" ? "error" : (statusState === "stopped" ? "warning" : "ready")), statusDetail);
  if (startBtn) {
    startBtn.disabled = bridgeRunning;
    startBtn.title = bridgeRunning
      ? "The live-updates bridge is already running."
      : "Start the live-updates bridge for panel progress and completion updates.";
  }
  if (stopBtn) {
    stopBtn.disabled = !bridgeRunning;
    stopBtn.title = bridgeRunning
      ? "Stop the live-updates bridge."
      : "The live-updates bridge is not running.";
  }
  if (connectBtn) {
    connectBtn.textContent = _uxpWsConnected ? "Live Updates Connected" : "Connect Live Updates";
    connectBtn.disabled = !bridgeRunning || _uxpWsConnected;
    connectBtn.title = _uxpWsConnected
      ? "This panel is already receiving live updates."
      : (!bridgeRunning
          ? "Start the bridge before connecting this panel."
          : "Connect this panel to receive live job updates.");
  }
}

async function uxpWsStartBridge() {
  const r = await BackendClient.post("/ws/start", {});
  if (r.ok && r.data?.success) {
    UIController.setStatus("Live-updates bridge started.", "success");
    UIController.showToast("Live-updates bridge started.", "success");
    setTimeout(() => uxpWsConnect(), 500);
  } else {
    UIController.showToast(r.error || "Failed to start bridge.", "error");
  }
}

async function uxpWsStopBridge() {
  uxpWsDisconnect();
  const r = await BackendClient.post("/ws/stop", {});
  if (r.ok) {
    UIController.setStatus("Live-updates bridge stopped.", "neutral");
    UIController.showToast("Live-updates bridge stopped.", "success");
    uxpUpdateWsStatus();
  }
}

// ─────────────────────────────────────────────────────────────
// Engine Registry UI
// ─────────────────────────────────────────────────────────────
async function uxpLoadEngines() {
  const grid = document.getElementById("uxpEngineGrid");
  if (!grid) return;
  setTextAndTitle("settingsEngineDefaultsValue", "Loading…", "Loading automatic engine routing coverage.");
  setTextAndTitle("settingsEnginePinnedValue", "Loading…", "Loading pinned engine preferences.");
  setTextAndTitle("settingsEngineCoverageValue", "Loading…", "Loading engine availability.");
  setSettingsStatus("settingsEngineStatus", "Loading engine availability…", "working");
  grid.innerHTML = `
    <div class="oc-empty-state oc-empty-state-inline">
      <div class="oc-empty-state-kicker">Engine routing</div>
      <p>Loading available engines and saved preferences…</p>
    </div>`;

  const r = await BackendClient.get("/engines");
  if (!r.ok || !r.data?.engines) {
    setTextAndTitle("settingsEngineDefaultsValue", "Unavailable", "OpenCut could not load automatic engine routing coverage.");
    setTextAndTitle("settingsEnginePinnedValue", "Unavailable", "OpenCut could not load pinned engine preferences.");
    setTextAndTitle("settingsEngineCoverageValue", "Refresh needed", "Refresh availability after the backend reconnects.");
    setSettingsStatus("settingsEngineStatus", "Engine availability could not be loaded. Refresh the backend, then try again.", "error");
    grid.innerHTML = `
      <div class="oc-empty-state oc-empty-state-inline">
        <div class="oc-empty-state-kicker">Engine data unavailable</div>
        <p>OpenCut could not load engine availability right now. Refresh the backend and try again.</p>
      </div>`;
    return;
  }

  const engines = r.data.engines;
  const domains = Object.keys(engines).sort();
  let pinnedCount = 0;
  let autoCount = 0;
  let issueCount = 0;
  let availableEngineCount = 0;
  let html = "";
  const usedEngineIds = new Set();

  for (const domain of domains) {
    const info = engines[domain];
    const entries = Array.isArray(info.engines) ? info.engines : [];
    const active = info.active || "";
    const preferred = info.preferred || "";
    const domainLabel = humanizeDomain(domain);
    const activeInfo = entries.find(eng => eng.name === active) || null;
    const preferredInfo = entries.find(eng => eng.name === preferred) || null;
    const availableCount = entries.filter(eng => eng.available).length;
    const domainIdBase = `engine-${safeDomIdSegment(domain)}`;
    let domainId = domainIdBase;
    let idSuffix = 2;
    while (usedEngineIds.has(domainId)) {
      domainId = `${domainIdBase}-${idSuffix}`;
      idSuffix += 1;
    }
    usedEngineIds.add(domainId);
    availableEngineCount += availableCount;
    const modeLabel = preferredInfo ? "Pinned" : availableCount ? "Auto" : "Needs attention";
    const modeClass = preferredInfo ? "manual" : availableCount ? "auto" : "warning";
    if (preferredInfo) pinnedCount += 1;
    else if (availableCount) autoCount += 1;
    else issueCount += 1;
    let summary = "";

    if (preferredInfo) {
      summary = `${preferredInfo.display_name} is preferred for ${domainLabel.toLowerCase()}.`;
      if (activeInfo && activeInfo.name === preferredInfo.name) {
        summary += " It is also active right now.";
      } else if (activeInfo) {
        summary += ` Current active engine: ${activeInfo.display_name}.`;
      }
    } else if (activeInfo) {
      summary = `${activeInfo.display_name} is active right now. Auto mode keeps the best available engine selected for this system.`;
    } else if (availableCount) {
      summary = `${availableCount} ${availableCount === 1 ? "engine is" : "engines are"} available. Auto mode will pick the best fit at run time.`;
    } else {
      summary = "No available engines detected yet. Refresh availability after installs finish.";
    }

    html += `<div class="oc-engine-row">`;
    html += `<div class="oc-engine-copy">`;
    html += `<div class="oc-engine-title-row">`;
    html += `<label class="oc-engine-domain" for="${UIController.escapeHtml(domainId)}">${UIController.escapeHtml(domainLabel)}</label>`;
    html += `<span class="oc-engine-state is-${UIController.escapeHtml(modeClass)}">${UIController.escapeHtml(modeLabel)}</span>`;
    html += `</div>`;
    html += `<p class="oc-engine-meta">${UIController.escapeHtml(summary)}</p>`;
    html += `</div>`;
    html += `<select class="oc-select oc-engine-sel" id="${UIController.escapeHtml(domainId)}" data-domain="${UIController.escapeHtml(domain)}" aria-label="${UIController.escapeHtml(domainLabel)} engine preference">`;
    html += `<option value="">Auto (best available)</option>`;
    for (const eng of entries) {
      const sel = (preferred === eng.name) ? " selected" : "";
      const avail = eng.available ? "" : " - unavailable";
      const label = `${eng.display_name} - ${eng.quality}/${eng.speed}${avail}${eng.name === active ? " - active" : ""}`;
      html += `<option value="${UIController.escapeHtml(eng.name)}"${sel}>${UIController.escapeHtml(label)}</option>`;
    }
    html += `</select></div>`;
  }

  grid.innerHTML = html;
  setTextAndTitle(
    "settingsEngineDefaultsValue",
    `${autoCount} auto`,
    `${autoCount} ${autoCount === 1 ? "domain is" : "domains are"} currently using automatic engine routing.`
  );
  setTextAndTitle(
    "settingsEnginePinnedValue",
    `${pinnedCount} pinned`,
    `${pinnedCount} ${pinnedCount === 1 ? "domain has" : "domains have"} a pinned engine preference.`
  );
  setTextAndTitle(
    "settingsEngineCoverageValue",
    issueCount
      ? `${issueCount} ${issueCount === 1 ? "issue" : "issues"}`
      : `${availableEngineCount} ${availableEngineCount === 1 ? "engine" : "engines"} ready`,
    `${availableEngineCount} available ${availableEngineCount === 1 ? "engine" : "engines"} detected across ${domains.length} ${domains.length === 1 ? "domain" : "domains"}.`
  );
  if (issueCount) {
    setSettingsStatus(
      "settingsEngineStatus",
      `${issueCount} ${issueCount === 1 ? "domain still needs attention" : "domains still need attention"}. Refresh availability after installs finish or return those domains to Auto when engines are ready.`,
      "warning"
    );
  } else if (pinnedCount) {
    setSettingsStatus(
      "settingsEngineStatus",
      "Engine routing is healthy. Auto covers the remaining domains, and pinned preferences will stay in place until you change them.",
      "success"
    );
  } else {
    setSettingsStatus(
      "settingsEngineStatus",
      "Engine routing is healthy. Auto mode will keep the best available engine selected for each domain.",
      "ready"
    );
  }

  grid.querySelectorAll(".oc-engine-sel").forEach(sel => {
    sel.addEventListener("change", async () => {
      const dom = sel.dataset.domain;
      const eng = sel.value;
      const domainLabel = humanizeDomain(dom);
      const selectedLabel = sel.options[sel.selectedIndex]?.textContent || "Auto";
      const pr = await BackendClient.post("/engines/preference", { domain: dom, engine: eng });
      if (pr.ok && pr.data?.success) {
        UIController.setStatus(
          eng
            ? `${domainLabel} engine routing updated.`
            : `${domainLabel} returned to automatic engine routing.`,
          "success"
        );
        UIController.showToast(
          eng
            ? `${domainLabel} now prefers ${selectedLabel}.`
            : `${domainLabel} is back on Auto routing.`,
          "success"
        );
        await uxpLoadEngines();
      } else {
        UIController.showToast(pr.error || "Failed to save preference.", "error");
        await uxpLoadEngines();
      }
    });
  });
}

// ─────────────────────────────────────────────────────────────
// UXP Migration Risk Dashboard
// ─────────────────────────────────────────────────────────────
function migrationStatusLabel(status) {
  return ({
    direct_uxp: "Direct UXP",
    partial_uxp: "Partial UXP",
    cep_only: "CEP fallback",
    different_mechanism: "Different mechanism",
  })[status] || status;
}

function migrationStateClass(row) {
  if (row.risk === "high" || row.status === "cep_only") return "warning";
  if (row.status === "partial_uxp" || row.risk === "medium") return "manual";
  return "auto";
}

async function uxpLoadMigrationRisk() {
  const grid = document.getElementById("uxpMigrationRiskGrid");
  if (!grid) return;
  setTextAndTitle("settingsMigrationDirectValue", "Loading…", "Loading direct UXP host-action coverage.");
  setTextAndTitle("settingsMigrationFallbackValue", "Loading…", "Loading CEP fallback count.");
  setTextAndTitle("settingsMigrationRiskValue", "Loading…", "Loading high-risk migration count.");
  setSettingsStatus("settingsMigrationStatus", "Loading UXP migration risk data…", "working");
  grid.innerHTML = `
    <div class="oc-empty-state oc-empty-state-inline">
      <div class="oc-empty-state-kicker">Migration risk</div>
      <p>Loading CEP and UXP host-action coverage…</p>
    </div>`;

  try {
    const response = await fetch(`uxp-migration-dashboard.json?ts=${Date.now()}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const manifest = await response.json();
    const summary = manifest.summary || {};
    const rows = Array.isArray(manifest.rows) ? manifest.rows : [];
    const direct = Number(summary.direct_uxp ?? 0);
    const partial = Number(summary.partial_uxp ?? 0);
    const fallback = Number(summary.cep_only ?? 0);
    const highRisk = Number(summary.high_risk ?? 0);

    setTextAndTitle("settingsMigrationDirectValue", `${direct} direct`, `${direct} host actions are covered by direct UXP APIs.`);
    setTextAndTitle("settingsMigrationFallbackValue", `${fallback} fallback`, `${fallback} host actions still require CEP or hybrid handling.`);
    setTextAndTitle("settingsMigrationRiskValue", `${highRisk} high`, `${highRisk} host actions are high-risk migration items.`);

    if (!rows.length) {
      grid.innerHTML = `
        <div class="oc-empty-state oc-empty-state-inline">
          <div class="oc-empty-state-kicker">Migration data empty</div>
          <p>The generated migration dashboard did not contain host-action rows.</p>
        </div>`;
      setSettingsStatus("settingsMigrationStatus", "Migration dashboard is empty. Regenerate the F260 dashboard artifact.", "warning");
      return;
    }

    grid.innerHTML = rows.map((row) => {
      const statusLabel = migrationStatusLabel(row.status);
      const stateClass = migrationStateClass(row);
      const tags = [
        row.risk ? `Risk: ${row.risk}` : "",
        row.needs_hybrid ? "Hybrid candidate" : "",
        Array.isArray(row.f_numbers) && row.f_numbers.length ? row.f_numbers.join(", ") : "",
      ].filter(Boolean).join(" | ");
      const summaryText = `${row.role || "Host action"} ${row.replacement_plan || row.uxp_path || ""}`.trim();
      return `<div class="oc-engine-row">
        <div class="oc-engine-copy">
          <div class="oc-engine-title-row">
            <span class="oc-engine-domain">${UIController.escapeHtml(row.name || "Unknown action")}</span>
            <span class="oc-engine-state is-${UIController.escapeHtml(stateClass)}">${UIController.escapeHtml(statusLabel)}</span>
          </div>
          <p class="oc-engine-meta">${UIController.escapeHtml(summaryText)}</p>
          <p class="oc-engine-meta">${UIController.escapeHtml(tags)}</p>
        </div>
      </div>`;
    }).join("");

    const status = fallback || partial || highRisk
      ? `${fallback} CEP fallback and ${partial} partial UXP host actions remain.`
      : "All catalogued host actions are direct UXP or resolved through non-CEP mechanisms.";
    setSettingsStatus("settingsMigrationStatus", status, fallback || highRisk ? "warning" : "success");
  } catch (e) {
    setTextAndTitle("settingsMigrationDirectValue", "Unavailable", "OpenCut could not load the generated UXP migration dashboard.");
    setTextAndTitle("settingsMigrationFallbackValue", "Unavailable", "OpenCut could not load CEP fallback count.");
    setTextAndTitle("settingsMigrationRiskValue", "Refresh needed", "Refresh after regenerating or packaging the dashboard artifact.");
    setSettingsStatus("settingsMigrationStatus", "Migration dashboard could not be loaded. Regenerate the F260 dashboard artifact, then refresh.", "error");
    grid.innerHTML = `
      <div class="oc-empty-state oc-empty-state-inline">
        <div class="oc-empty-state-kicker">Migration data unavailable</div>
        <p>OpenCut could not load the bundled UXP migration dashboard right now.</p>
      </div>`;
  }
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
  const headerVersion = document.getElementById("uxpHeaderVersion");
  const aboutVersion = document.getElementById("uxpVersionDisplay");
  if (headerVersion) headerVersion.textContent = `v${VERSION}`;
  if (aboutVersion) aboutVersion.textContent = `${VERSION} (UXP)`;

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
    updateTimelineReadiness();
  });

  // Wire up all UI interactions
  UIController.initCollapsibles();
  bindSliders();
  bindEvents();
  initKeyboardShortcuts();
  UIController.switchTab(document.querySelector(".oc-tab.active")?.dataset.tab ?? "cut");
  updateWorkspaceOverview();
  updateDeliverablesSummary();
  updateTimelineReadiness();
  await uxpLoadMigrationRisk();

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
    await refreshFootageIndexStats({ silent: true });

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
  } else {
    await refreshFootageIndexStats({ silent: true });
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

  // -------------------------------------------------------------
  // Agent tab — F143 conductor + Q3/Q7/Q8 + F146 MCP bridge
  // Surfaces the four 2026-05-25 backends in a single UXP-only tab.
  // CEP equivalent: command palette + NLP tab (see PANEL_PARITY.json).
  // -------------------------------------------------------------
  initAgentTab();

  // -------------------------------------------------------------
  // F236 FCC caption display-settings card (effective 2026-08-17).
  // Lazy-init: fetches the token catalogue on first paint, builds
  // the dropdown options, wires Preview + Reset.
  // -------------------------------------------------------------
  initCaptionDisplaySettingsCard();

  console.log("[OpenCut UXP] Ready.");
}

// =============================================================
// Agent tab module (F143 / Q3 / Q7 / Q8 / F146 surfacing)
// =============================================================
function initAgentTab() {
  const $ = (id) => document.getElementById(id);
  let activeSessionId = "";

  const setStatus = (id, text) => {
    const el = $(id);
    if (el) el.textContent = text;
  };

  const showBox = (id, show) => {
    const el = $(id);
    if (el) el.hidden = !show;
  };

  function renderPlan(plan) {
    const list = $("agentChatPlanList");
    if (!list) return;
    list.innerHTML = "";
    if (!Array.isArray(plan) || plan.length === 0) {
      const li = document.createElement("li");
      li.textContent = "No steps matched — try a more specific intent.";
      list.appendChild(li);
      return;
    }
    plan.forEach((step) => {
      const li = document.createElement("li");
      li.dataset.stepId = step.step_id || "";
      const status = String(step.status || "planned");
      const tag = status === "ok" ? "✓"
                : status === "failed" ? "✗"
                : status === "rejected" ? "✗"
                : "·";
      li.textContent = `${tag} ${step.label || step.endpoint} (${step.endpoint})`;
      if (step.status === "failed" || step.status === "rejected") {
        li.classList.add("oc-step-error");
      }
      list.appendChild(li);
    });
  }

  function renderReview(review) {
    const box = $("agentChatReviewBox");
    const summary = $("agentChatReviewSummary");
    const notes = $("agentChatReviewNotes");
    if (!box || !summary || !notes) return;
    box.hidden = false;
    const score = typeof review.drift_score === "number" ? review.drift_score : 100;
    const matched = review.matched ? "Matched" : "Drift detected";
    summary.textContent = `${matched} (drift score ${score}/100, source: ${review.source || "heuristic"})`;
    notes.innerHTML = "";
    (review.drift_notes || []).forEach((n) => {
      const li = document.createElement("li");
      li.textContent = String(n);
      notes.appendChild(li);
    });
    if (review.suggested_retry && review.suggested_retry.endpoint) {
      const li = document.createElement("li");
      li.textContent = `↻ Suggested retry: ${review.suggested_retry.label} (${review.suggested_retry.endpoint})`;
      notes.appendChild(li);
    }
  }

  // --- Chat Conductor wiring -------------------------------------
  const planBtn = $("agentChatPlanBtn");
  if (planBtn) {
    planBtn.addEventListener("click", async () => {
      const intent = ($("agentChatIntent")?.value || "").trim();
      if (!intent) {
        setStatus("agentChatStatus", "Enter an intent first.");
        return;
      }
      planBtn.disabled = true;
      setStatus("agentChatStatus", "Building plan…");
      try {
        const resp = await BackendClient.post("/agent/chat/plan", { intent });
        if (!resp || resp.error) {
          setStatus("agentChatStatus", "Plan failed: " + (resp?.error || "unknown"));
          return;
        }
        activeSessionId = resp.session_id || "";
        renderPlan(resp.plan);
        showBox("agentChatPlanBox", true);
        showBox("agentChatReviewBox", false);
        setStatus("agentChatStatus",
          `Plan: ${resp.plan?.length || 0} step(s) via ${resp.source}. Session ${activeSessionId.slice(0, 8)}.`);
        const reviewBtn = $("agentChatReviewBtn");
        if (reviewBtn) reviewBtn.disabled = false;
      } catch (err) {
        setStatus("agentChatStatus", "Plan error: " + (err?.message || err));
      } finally {
        planBtn.disabled = false;
      }
    });
  }

  const reviewBtn = $("agentChatReviewBtn");
  if (reviewBtn) {
    reviewBtn.addEventListener("click", async () => {
      if (!activeSessionId) {
        setStatus("agentChatStatus", "Run Plan first to start a session.");
        return;
      }
      reviewBtn.disabled = true;
      setStatus("agentChatStatus", "Running self-review…");
      try {
        const resp = await BackendClient.post("/agent/chat/review", {
          session_id: activeSessionId,
        });
        if (!resp || resp.error) {
          setStatus("agentChatStatus", "Review failed: " + (resp?.error || "unknown"));
          return;
        }
        renderReview(resp);
        setStatus("agentChatStatus",
          `Reviewed (${resp.source}). Drift score ${resp.drift_score}/100.`);
      } catch (err) {
        setStatus("agentChatStatus", "Review error: " + (err?.message || err));
      } finally {
        reviewBtn.disabled = false;
      }
    });
  }

  const clearBtn = $("agentChatClearBtn");
  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      activeSessionId = "";
      const ta = $("agentChatIntent");
      if (ta) ta.value = "";
      showBox("agentChatPlanBox", false);
      showBox("agentChatReviewBox", false);
      const rb = $("agentChatReviewBtn");
      if (rb) rb.disabled = true;
      setStatus("agentChatStatus", "Cleared.");
    });
  }

  // --- Enhance wiring -------------------------------------------
  async function runEnhance(dryRun) {
    const filepath = ($("enhanceClipPath")?.value || "").trim();
    const style = $("enhanceStyle")?.value || "social";
    if (!filepath) {
      setStatus("enhanceStatus", "Enter a clip path first.");
      return;
    }
    const btnId = dryRun ? "enhanceDryRunBtn" : "enhanceRunBtn";
    const btn = $(btnId);
    if (btn) btn.disabled = true;
    setStatus("enhanceStatus", dryRun ? "Building plan…" : "Running Enhance…");
    try {
      const endpoint = dryRun ? "/enhance/auto/dry-run" : "/enhance/auto";
      const resp = await BackendClient.post(endpoint, { filepath, style });
      if (!resp || resp.error) {
        setStatus("enhanceStatus", "Failed: " + (resp?.error || "unknown"));
        return;
      }
      if (dryRun) {
        const steps = resp.steps || [];
        const skipped = steps.filter((s) => s.status === "skipped").length;
        setStatus("enhanceStatus",
          `Plan: ${steps.length - skipped} step(s) will run, ${skipped} skipped.`);
      } else {
        setStatus("enhanceStatus", "Job queued — watch the progress bar above.");
      }
    } catch (err) {
      setStatus("enhanceStatus", "Error: " + (err?.message || err));
    } finally {
      if (btn) btn.disabled = false;
    }
  }
  $("enhanceDryRunBtn")?.addEventListener("click", () => runEnhance(true));
  $("enhanceRunBtn")?.addEventListener("click", () => runEnhance(false));

  // --- Variants wiring -------------------------------------------
  async function runVariants(dryRun) {
    const filepath = ($("variantsClipPath")?.value || "").trim();
    const start = Number($("variantsStart")?.value || 0);
    const end = Number($("variantsEnd")?.value || 0);
    const n_variants = Math.max(2, Math.min(6, Number($("variantsN")?.value || 3)));
    if (!filepath) {
      setStatus("variantsStatus", "Enter a clip path first.");
      return;
    }
    if (end <= start) {
      setStatus("variantsStatus", "End must be greater than start.");
      return;
    }
    const btnId = dryRun ? "variantsDryRunBtn" : "variantsRunBtn";
    const btn = $(btnId);
    if (btn) btn.disabled = true;
    setStatus("variantsStatus", dryRun ? "Planning variants…" : "Rendering variants…");
    try {
      const endpoint = dryRun ? "/shorts/variants/dry-run" : "/shorts/variants";
      const resp = await BackendClient.post(endpoint, {
        filepath, start, end, n_variants,
      });
      if (!resp || resp.error) {
        setStatus("variantsStatus", "Failed: " + (resp?.error || "unknown"));
        return;
      }
      const variants = resp.variants || [];
      setStatus("variantsStatus",
        dryRun
          ? `Plan: ${variants.length} variant(s) will be generated.`
          : `Generated ${variants.length} variant(s).`);
    } catch (err) {
      setStatus("variantsStatus", "Error: " + (err?.message || err));
    } finally {
      if (btn) btn.disabled = false;
    }
  }
  $("variantsDryRunBtn")?.addEventListener("click", () => runVariants(true));
  $("variantsRunBtn")?.addEventListener("click", () => runVariants(false));

  // --- Sequence Index wiring -------------------------------------
  $("sequenceIndexBuildBtn")?.addEventListener("click", async () => {
    const btn = $("sequenceIndexBuildBtn");
    if (btn) btn.disabled = true;
    setStatus("sequenceIndexStatus", "Reading active sequence…");
    try {
      // PProBridge.getSequenceInfo() returns the same JSON shape that
      // host/index.jsx::ocGetSequenceInfo() produces, which is exactly
      // what /timeline/sequence-index expects.
      const sequence = await PProBridge.getSequenceInfo();
      if (!sequence || sequence.error) {
        setStatus("sequenceIndexStatus", "No active sequence: " + (sequence?.error || "unknown"));
        return;
      }
      const resp = await BackendClient.post("/timeline/sequence-index", { sequence });
      if (!resp || resp.error) {
        setStatus("sequenceIndexStatus", "Failed: " + (resp?.error || "unknown"));
        return;
      }
      const summary = $("sequenceIndexSummary");
      const box = $("sequenceIndexBox");
      if (summary && box) {
        box.hidden = false;
        summary.textContent =
          `${resp.sequence_name || "Sequence"} — ${resp.total_rows} clips ` +
          `(${resp.duration_s?.toFixed?.(1) || 0}s @ ${resp.fps}fps), ` +
          `${resp.marker_count} markers.`;
      }
      setStatus("sequenceIndexStatus",
        `Index built: ${resp.total_rows} rows across ${resp.width}×${resp.height} sequence.`);
    } catch (err) {
      setStatus("sequenceIndexStatus", "Error: " + (err?.message || err));
    } finally {
      if (btn) btn.disabled = false;
    }
  });

  $("sequenceIndexInfoBtn")?.addEventListener("click", async () => {
    try {
      const info = await BackendClient.get("/timeline/sequence-index/info");
      setStatus("sequenceIndexStatus",
        info?.available
          ? `Capability OK — sort keys: ${(info.sort_keys || []).join(", ")}`
          : "Sequence Index unavailable");
    } catch (err) {
      setStatus("sequenceIndexStatus", "Capability check failed: " + (err?.message || err));
    }
  });

  // --- MCP Bridge wiring -----------------------------------------
  $("mcpBridgeInfoBtn")?.addEventListener("click", async () => {
    try {
      const info = await BackendClient.get("/mcp/info");
      setStatus("mcpBridgeStatus",
        `v${info.version}: ${info.curated_count} curated + ${info.extended_count} extended ` +
        `(extended ${info.extended_enabled_by_default ? "on" : "off"} by default).`);
    } catch (err) {
      setStatus("mcpBridgeStatus", "Capability check failed: " + (err?.message || err));
    }
  });
  $("mcpBridgeListBtn")?.addEventListener("click", async () => {
    try {
      const resp = await BackendClient.get("/mcp/tools?include_extended=false");
      const names = (resp.tools || []).map((t) => t.name).slice(0, 5);
      setStatus("mcpBridgeStatus",
        `${resp.count} tool(s). First 5: ${names.join(", ")}`);
    } catch (err) {
      setStatus("mcpBridgeStatus", "List failed: " + (err?.message || err));
    }
  });
}

// =============================================================
// F236 FCC caption display-settings card
// Effective 2026-08-17 (47 CFR § 79.103); users must be able to
// readily access these controls. Backend contract:
//   GET  /captions/display-settings/tokens
//   POST /captions/display-settings/preview {settings, sample_text}
// =============================================================
function initCaptionDisplaySettingsCard() {
  const $ = (id) => document.getElementById(id);
  const card = $("captionDisplaySettingsCard");
  if (!card) return;

  // Selector → token-category mapping (mirrors token_schema()).
  const SELECT_MAP = [
    { id: "capDispFont", category: "font", labelFn: (o) => `${o.id} (${o.font_family})` },
    { id: "capDispSize", category: "size", labelFn: (o) => `${o.id} (${o.font_size})` },
    { id: "capDispTextColor", category: "color", labelFn: (o) => `${o.id} (${o.hex})`, key: "text_color" },
    { id: "capDispTextOpacity", category: "opacity", labelFn: (o) => `${o.id} (${o.alpha})`, key: "text_opacity" },
    { id: "capDispBgColor", category: "color", labelFn: (o) => `${o.id} (${o.hex})`, key: "background_color" },
    { id: "capDispBgOpacity", category: "opacity", labelFn: (o) => `${o.id} (${o.alpha})`, key: "background_opacity" },
    { id: "capDispEdge", category: "edge_style", labelFn: (o) => `${o.id}` },
  ];

  let cachedSchema = null;

  const setStatus = (text) => {
    const el = $("capDispStatus");
    if (el) el.textContent = text;
  };

  function populateSelects(schema) {
    cachedSchema = schema;
    const tokens = schema.tokens || {};
    const defaults = schema.defaults || {};
    for (const spec of SELECT_MAP) {
      const sel = $(spec.id);
      if (!sel) continue;
      const opts = tokens[spec.category] || [];
      sel.innerHTML = "";
      for (const opt of opts) {
        const option = document.createElement("option");
        option.value = String(opt.id);
        option.textContent = spec.labelFn(opt);
        sel.appendChild(option);
      }
      const settingKey = spec.key || spec.category;
      const def = defaults[settingKey] || (opts[0] && opts[0].id);
      if (def) sel.value = def;
    }
  }

  function readSettings() {
    const out = {};
    for (const spec of SELECT_MAP) {
      const sel = $(spec.id);
      if (!sel) continue;
      out[spec.key || spec.category] = sel.value;
    }
    return out;
  }

  function applyPreviewStyles(payload) {
    const area = $("capDispPreviewArea");
    const sample = $("capDispPreviewSample");
    const box = $("capDispPreviewBox");
    if (!area || !sample || !box) return;
    box.hidden = false;
    sample.textContent = payload.sample_text || "Caption preview";
    const styles = payload.preview_css || {};
    // The /captions/display-settings/preview payload returns ready-to-use
    // CSS strings keyed by attribute name. Apply directly.
    for (const [key, value] of Object.entries(styles)) {
      try {
        sample.style[key] = value;
      } catch (_e) { /* hostile UXP CSS prop name — skip */ }
    }
  }

  async function refreshPreview() {
    const btn = $("capDispPreviewBtn");
    if (btn) btn.disabled = true;
    setStatus("Rendering preview…");
    try {
      const resp = await BackendClient.post("/captions/display-settings/preview", {
        settings: readSettings(),
        sample_text: "The quick brown fox jumps over the lazy dog.",
      });
      if (!resp || resp.error) {
        setStatus("Preview failed: " + (resp?.error || "unknown"));
        return;
      }
      applyPreviewStyles(resp);
      setStatus("Preview updated. These are the FCC display tokens applied to burn-in.");
    } catch (err) {
      setStatus("Preview error: " + (err?.message || err));
    } finally {
      if (btn) btn.disabled = false;
    }
  }

  function resetDefaults() {
    if (!cachedSchema) return;
    const defaults = cachedSchema.defaults || {};
    for (const spec of SELECT_MAP) {
      const sel = $(spec.id);
      if (!sel) continue;
      const settingKey = spec.key || spec.category;
      if (defaults[settingKey]) sel.value = defaults[settingKey];
    }
    setStatus("Reset to FCC defaults. Click Preview to re-render.");
  }

  // Lazy-load the token catalogue once the card is first painted.
  // (The Captions tab may not be the user's first stop, so don't block
  // initApp() on this network call.)
  async function lazyLoadTokens() {
    try {
      const schema = await BackendClient.get("/captions/display-settings/tokens");
      if (!schema || schema.error) {
        setStatus("Could not load FCC token schema. The card will stay empty.");
        return;
      }
      populateSelects(schema);
      // Surface the compliance-date string in the hint if the backend supplies one.
      const notice = document.getElementById("fccComplianceNotice");
      if (notice && schema.compliance_date) {
        notice.textContent = notice.textContent.replace("2026-08-17", schema.compliance_date);
      }
      setStatus("Defaults loaded. Adjust tokens then Preview.");
    } catch (err) {
      setStatus("Token-schema fetch failed: " + (err?.message || err));
    }
  }

  // Wire buttons.
  $("capDispPreviewBtn")?.addEventListener("click", refreshPreview);
  $("capDispResetBtn")?.addEventListener("click", resetDefaults);

  // Kick off the lazy load; intentionally fire-and-forget so the rest
  // of the panel finishes painting first.
  setTimeout(lazyLoadTokens, 600);
}

// Bootstrap
initApp().catch(err => {
  console.error("[OpenCut UXP] Init error:", err);
});
