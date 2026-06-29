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
const WS_BRIDGE_DEFAULT_PORT = 5680;
const WS_RECONNECT_BASE_MS = 5000;
const WS_RECONNECT_MAX_MS = 30000;
const MEDIA_SCAN_MS    = 30000;
const INLINE_CONFIRM_MS = 8000;
const SSE_AVAILABLE    = typeof EventSource !== "undefined";
const VERSION          = "1.33.1";
const UXP_DEFAULT_LOCALE = "en";
const UXP_LOCALE_DIR   = "locales";
const UXP_LOCALE_PATH  = `${UXP_LOCALE_DIR}/${UXP_DEFAULT_LOCALE}.json`;
const PRIMARY_CLIP_INPUT_IDS = ["clipPathCut", "clipPathCaptions", "clipPathAudio", "clipPathVideo"];
const TABS_REQUIRING_SOURCE = new Set(["cut", "captions", "audio", "video"]);
const CONNECTION_LABEL_KEYS = {
  connected: "conn.online",
  connecting: "conn.connecting",
  disconnected: "conn.offline",
};
const DELIVERABLE_LABELS = {
  vfx_sheet: "VFX Sheet",
  adr_list: "ADR List",
  music_cue_sheet: "Music Cue Sheet",
  asset_list: "Asset List",
};
const DELIVERABLE_LABEL_KEYS = {
  vfx_sheet: "uxp.deliverables.vfx_sheet",
  adr_list: "uxp.deliverables.adr_list",
  music_cue_sheet: "uxp.deliverables.music_cue_sheet",
  asset_list: "uxp.deliverables.asset_list",
};
const DELIVERABLE_BUTTON_IDS = {
  vfx_sheet: "delivVfxSheetBtn",
  adr_list: "delivAdrListBtn",
  music_cue_sheet: "delivMusicCueBtn",
  asset_list: "delivAssetListBtn",
};
const CAPTION_STYLES_ENDPOINT = "/captions/styles";
const DEFAULT_CAPTION_STYLE_ID = "minimal_clean";
const UXP_CAPTION_STYLE_FALLBACK = [
  {
    id: DEFAULT_CAPTION_STYLE_ID,
    name: "Minimal Clean",
    category: "minimal",
    preview_description: "Simple white text, no frills",
    animation_type: "fade",
  },
];
const WORKSPACE_META   = {
  cut: {
    titleKey: "uxp.workspace.cut_title",
    title: "Cut & Clean",
    subtitleKey: "uxp.workspace.cut_subtitle",
    subtitle: "Trim dead space, fillers, and rough pacing with a tighter review flow.",
    sourceIds: ["clipPathCut"],
  },
  captions: {
    titleKey: "uxp.tabs.captions",
    title: "Captions",
    subtitleKey: "uxp.workspace.captions_subtitle",
    subtitle: "Transcribe, structure, and style subtitles without leaving the panel.",
    sourceIds: ["clipPathCaptions"],
  },
  audio: {
    titleKey: "uxp.tabs.audio",
    title: "Audio",
    subtitleKey: "uxp.workspace.audio_subtitle",
    subtitle: "Denoise, normalize, loudness-match, and cut to rhythm from one focused surface.",
    sourceIds: ["clipPathAudio"],
  },
  video: {
    titleKey: "uxp.tabs.video",
    title: "Video",
    subtitleKey: "uxp.workspace.video_subtitle",
    subtitle: "Shape the image, plan coverage, and build short-form versions with a cleaner finishing toolkit.",
    sourceIds: ["clipPathVideo"],
  },
  timeline: {
    titleKey: "uxp.tabs.timeline",
    title: "Timeline",
    subtitleKey: "uxp.workspace.timeline_subtitle",
    subtitle: "Write changes back into Premiere, export interchange, and run batch production tasks with confidence.",
    sourceIds: ["clipPathCut", "clipPathVideo", "clipPathAudio"],
  },
  search: {
    titleKey: "uxp.tabs.search",
    title: "Search",
    subtitleKey: "uxp.workspace.search_subtitle",
    subtitle: "Index the library, search footage, and trigger edit actions from natural-language commands.",
    sourceIds: ["clipPathVideo", "clipPathCaptions", "clipPathCut", "clipPathAudio"],
  },
  deliverables: {
    titleKey: "uxp.tabs.deliverables",
    title: "Deliverables",
    subtitleKey: "uxp.workspace.deliverables_subtitle",
    subtitle: "Review sequence context and export reports, documents, and final handoff assets.",
    sourceIds: ["clipPathCut", "clipPathVideo", "clipPathAudio"],
  },
  agent: {
    titleKey: "uxp.tabs.agent",
    title: "Agent",
    subtitleKey: "uxp.workspace.agent_subtitle",
    subtitle: "Plan, enhance, index, and bridge assisted edit workflows from one workspace.",
    sourceIds: ["enhanceClipPath", "variantsClipPath"],
  },
  settings: {
    titleKey: "uxp.tabs.settings",
    title: "Settings",
    subtitleKey: "uxp.workspace.settings_subtitle",
    subtitle: "Tune engine routing, realtime connections, and shared defaults across the studio.",
    sourceIds: [],
  },
};
const WORKSPACE_GUIDES = {
  cut: {
    kickerKey: "uxp.guide.cut_kicker",
    kicker: "Cut pass",
    titleKey: "uxp.guide.cut_title",
    title: "Build a cleaner first pass of the edit.",
    textKey: "uxp.guide.cut_text",
    text: "Start with silence detection or filler cleanup, then review the suggested cut ranges before writing them back to the timeline.",
    action: "focus-runSilenceBtn",
    actionLabelKey: "uxp.guide.run_silence",
    actionLabel: "Run Silence Detection",
  },
  captions: {
    kickerKey: "uxp.guide.captions_kicker",
    kicker: "Transcript",
    titleKey: "uxp.guide.captions_title",
    title: "Turn the active shot into reviewable text.",
    textKey: "uxp.guide.captions_text",
    text: "Transcribe first, then move into chapters, repeat detection, or timeline import once the wording looks right.",
    action: "focus-runTranscribeBtn",
    actionLabelKey: "uxp.guide.transcribe_clip",
    actionLabel: "Transcribe Clip",
  },
  audio: {
    kickerKey: "uxp.guide.audio_kicker",
    kicker: "Audio pass",
    titleKey: "uxp.guide.audio_title",
    title: "Clean the voice bed before the rest of the finish.",
    textKey: "uxp.guide.audio_text",
    text: "Start with denoise or normalization, then add rhythm markers if the cut needs to lock to music.",
    action: "focus-runDenoiseBtn",
    actionLabelKey: "uxp.guide.run_denoise",
    actionLabel: "Run Denoise",
  },
  video: {
    kickerKey: "uxp.guide.video_kicker",
    kicker: "Finishing",
    titleKey: "uxp.guide.video_title",
    title: "Shape the frame and build derivative edits from one source.",
    textKey: "uxp.guide.video_text",
    text: "Use color, reframe, multicam, and short-form tools without repatching the same clip every time.",
    action: "focus-runColorMatchBtn",
    actionLabelKey: "uxp.guide.match_color",
    actionLabel: "Match Color",
  },
  timeline: {
    kickerKey: "uxp.guide.timeline_kicker",
    kicker: "Write-back",
    titleKey: "uxp.guide.timeline_title",
    title: "Send approved changes back into Premiere with less friction.",
    textKey: "uxp.guide.timeline_text",
    text: "Apply the latest cuts or markers, then export OTIO, markers, or captions from the same review session.",
    action: "focus-applyTimelineCutsBtn",
    actionLabelKey: "uxp.guide.apply_latest_cuts",
    actionLabel: "Apply Latest Cuts",
  },
  search: {
    kickerKey: "uxp.guide.search_kicker",
    kicker: "Discovery",
    titleKey: "uxp.guide.search_title",
    title: "Search the library, then reuse the result instantly.",
    textKey: "uxp.guide.search_text",
    text: "Index a folder once, search with natural language, and pull the matching shot back into the rest of the workspace.",
    action: "focus-searchQuery",
    actionLabelKey: "uxp.guide.start_search",
    actionLabel: "Start Search",
  },
  deliverables: {
    kickerKey: "uxp.guide.deliverables_kicker",
    kicker: "Handoff",
    titleKey: "uxp.guide.deliverables_title",
    title: "Pull sequence context before generating delivery docs.",
    textKey: "uxp.guide.deliverables_text",
    text: "Load the active Premiere sequence, choose an output folder, and generate reports with cleaner defaults.",
    action: "focus-loadSeqInfoBtn",
    actionLabelKey: "uxp.guide.load_sequence_info",
    actionLabel: "Load Sequence Info",
  },
  agent: {
    kickerKey: "uxp.guide.agent_kicker",
    kicker: "Agent tools",
    titleKey: "uxp.guide.agent_title",
    title: "Plan and run assisted edit workflows.",
    textKey: "uxp.guide.agent_text",
    text: "Use the conductor, one-click enhance, variants, sequence index, and MCP bridge from one tab.",
    action: "focus-agentChatIntent",
    actionLabelKey: "uxp.guide.describe_intent",
    actionLabel: "Describe Intent",
  },
  settings: {
    kickerKey: "uxp.guide.settings_kicker",
    kicker: "Studio setup",
    titleKey: "uxp.guide.settings_title",
    title: "Keep routing, engines, and live services healthy.",
    textKey: "uxp.guide.settings_text",
    text: "Refresh engine availability, verify the bridge, and make sure the panel is connected to the right backend.",
    action: "focus-uxpRefreshEnginesBtn",
    actionLabelKey: "uxp.guide.refresh_engines",
    actionLabel: "Refresh Engines",
  },
};

// ─────────────────────────────────────────────────────────────
// i18n / Localization
// ─────────────────────────────────────────────────────────────
let _currentLang = UXP_DEFAULT_LOCALE;
let _i18n = {};

function t(key, fallback) {
  return (_i18n && _i18n[key]) || fallback || key;
}

function formatI18n(key, fallback, values = {}) {
  let text = t(key, fallback);
  Object.keys(values).forEach((name) => {
    const val = String(values[name]);
    text = text.replace(new RegExp(`\\{${name}\\}`, "g"), () => val);
  });
  return text;
}

function formatCountI18n(count, oneKey, oneFallback, manyKey, manyFallback, values = {}) {
  return formatI18n(
    count === 1 ? oneKey : manyKey,
    count === 1 ? oneFallback : manyFallback,
    { count, ...values }
  );
}

function getDeliverableLabel(type) {
  return t(DELIVERABLE_LABEL_KEYS[type], DELIVERABLE_LABELS[type] || humanizeDomain(type));
}

function applyI18nToDOM(root = document) {
  const scope = root || document;
  scope.querySelectorAll("[data-i18n]").forEach((node) => {
    const key = node.getAttribute("data-i18n");
    if (!key) return;
    const labelTarget = node.querySelector(".btn-label, .i18n-text");
    if (!node.hasAttribute("data-i18n-fallback")) {
      node.setAttribute("data-i18n-fallback", labelTarget ? labelTarget.textContent : node.textContent);
    }
    const fallback = node.getAttribute("data-i18n-fallback") || "";
    const translated = t(key, fallback);
    if (labelTarget) labelTarget.textContent = translated;
    else node.textContent = translated;
  });

  const attrMappings = [
    ["data-i18n-title", "title"],
    ["data-i18n-label", "label"],
    ["data-i18n-alt", "alt"],
    ["data-i18n-placeholder", "placeholder"],
    ["data-i18n-aria-label", "aria-label"],
  ];
  attrMappings.forEach(([dataAttr, targetAttr]) => {
    scope.querySelectorAll(`[${dataAttr}]`).forEach((node) => {
      const key = node.getAttribute(dataAttr);
      if (!key) return;
      const fallbackAttr = `${dataAttr}-fallback`;
      if (!node.hasAttribute(fallbackAttr)) {
        node.setAttribute(fallbackAttr, node.getAttribute(targetAttr) || "");
      }
      node.setAttribute(targetAttr, t(key, node.getAttribute(fallbackAttr) || ""));
    });
  });
}

function normalizeLocaleTag(lang = UXP_DEFAULT_LOCALE) {
  return String(lang || UXP_DEFAULT_LOCALE).trim().toLowerCase().replace(/_/g, "-") || UXP_DEFAULT_LOCALE;
}

function getLocaleOverride() {
  try {
    if (typeof window !== "undefined" && window.location) {
      const override = new URLSearchParams(window.location.search).get("lang");
      return override ? normalizeLocaleTag(override) : "";
    }
  } catch (err) {
    console.debug("[OpenCut UXP] Locale override unavailable.", err);
  }
  return "";
}

function getPreferredLocale() {
  const override = getLocaleOverride();
  if (override) return override;
  if (typeof navigator !== "undefined") {
    const languages = Array.isArray(navigator.languages) && navigator.languages.length
      ? navigator.languages
      : [navigator.language];
    const first = languages.find(Boolean);
    if (first) return normalizeLocaleTag(first);
  }
  return UXP_DEFAULT_LOCALE;
}

function getLocaleCandidates(lang) {
  const normalized = normalizeLocaleTag(lang);
  const base = normalized.split("-")[0];
  const candidates = [normalized];
  if (base && base !== normalized) candidates.push(base);
  if (UXP_DEFAULT_LOCALE !== normalized && UXP_DEFAULT_LOCALE !== base) candidates.push(UXP_DEFAULT_LOCALE);
  return candidates;
}

async function fetchLocaleJson(path) {
  try {
    const response = await fetch(path);
    if (response && (response.ok || response.status === 0)) {
      return await response.json();
    }
  } catch (err) {
    console.warn(`[OpenCut UXP] Locale file unavailable: ${path}`, err);
  }
  return null;
}

async function loadLocale(lang = getPreferredLocale()) {
  const requestedLang = normalizeLocaleTag(lang);
  const baseLocale = await fetchLocaleJson(UXP_LOCALE_PATH) || {};
  let activeLang = UXP_DEFAULT_LOCALE;
  let activeLocale = {};

  for (const candidate of getLocaleCandidates(requestedLang)) {
    if (candidate === UXP_DEFAULT_LOCALE) {
      activeLocale = baseLocale;
      activeLang = UXP_DEFAULT_LOCALE;
      break;
    }
    const locale = await fetchLocaleJson(`${UXP_LOCALE_DIR}/${candidate}.json`);
    if (locale) {
      activeLocale = locale;
      activeLang = candidate;
      break;
    }
  }
  _i18n = { ...baseLocale, ...activeLocale };
  _currentLang = Object.keys(_i18n).length ? activeLang : UXP_DEFAULT_LOCALE;
  if (!Object.keys(_i18n).length) {
    console.warn("[OpenCut UXP] Locale files unavailable; using inline English fallbacks.");
  }
  document.documentElement.lang = _currentLang;
  applyI18nToDOM();
  return _currentLang;
}

function localizeWorkspaceMeta(meta, field) {
  return t(meta?.[`${field}Key`], meta?.[field] || "");
}

function getWorkspaceTitle(tabId) {
  const fallbackTitle = tabId ? tabId.charAt(0).toUpperCase() + tabId.slice(1) : "Cut & Clean";
  return localizeWorkspaceMeta(WORKSPACE_META[tabId] || WORKSPACE_META.cut, "title") || fallbackTitle;
}

function isBackendConnected() {
  return document.getElementById("connectionStatus")?.dataset.state === "connected";
}

function isTimeoutError(err) {
  const message = String(err?.message || err || "");
  return err?.name === "AbortError" || /timed out|abort/i.test(message);
}

function showSelectClipWarning() {
  UIController.showToast(t("uxp.runtime.select_clip_first", "Select a clip first."), "warning");
}

async function copyTextToClipboard(text, { successLabel = t("uxp.runtime.output", "Output") } = {}) {
  const value = String(text || "");
  if (!value) return false;

  if (
    typeof navigator === "undefined" ||
    !navigator.clipboard ||
    typeof navigator.clipboard.writeText !== "function"
  ) {
    UIController.showToast(t("uxp.runtime.clipboard_unavailable", "Clipboard is unavailable in this UXP environment. Copy the output manually."), "warning");
    return false;
  }

  try {
    await navigator.clipboard.writeText(value);
    UIController.showToast(formatI18n("uxp.runtime.copied_to_clipboard", "{label} copied to clipboard.", { label: successLabel }), "success");
    return true;
  } catch (err) {
    UIController.showToast(t("uxp.runtime.clipboard_permission_denied", "Clipboard permission is unavailable or denied. Copy the output manually."), "warning");
    return false;
  }
}

function normalizeHttpsExternalUrl(value) {
  const raw = String(value || "").trim();
  if (!raw) return null;

  try {
    const parsed = new URL(raw);
    return parsed.protocol === "https:" ? parsed.href : null;
  } catch (_) {
    return null;
  }
}

async function openHttpsExternalUrl(value, developerText) {
  const url = normalizeHttpsExternalUrl(value);
  if (!url) {
    UIController.showToast(t("uxp.runtime.invalid_https_authorization_url", "Invalid HTTPS authorization URL received from server."), "warning");
    return false;
  }

  try {
    const shell = require("uxp").shell;
    const result = await shell.openExternal(
      url,
      developerText || t("uxp.runtime.opening_secure_authorization", "Opening a secure authorization page in your browser"),
    );
    if (typeof result === "string" && result.trim()) {
      UIController.showToast(formatI18n("uxp.runtime.open_url_in_browser", "Open this URL in your browser: {url}", { url }), "info", 10000);
      return false;
    }
    return true;
  } catch (_) {
    UIController.showToast(formatI18n("uxp.runtime.open_url_in_browser", "Open this URL in your browser: {url}", { url }), "info", 10000);
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

async function refreshBackendBaseUrl() {
  const detected = await detectBackend();
  if (detected !== BACKEND) {
    console.log(`[OpenCut UXP] Backend switched from ${BACKEND} to ${detected}`);
  }
  BACKEND = detected;
  return BACKEND;
}

// ─────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────
let csrfToken     = null;
let activeJobId   = null;
let _jobStartInFlight = false;
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
const _jobLockedButtonStates = new Map();
const JOB_ACTION_BUTTON_SELECTOR = [
  'button[id^="run"]',
  "#socialUploadBtnUxp",
  "#enhanceRunBtn",
  "#variantsRunBtn",
  "#sequenceIndexBuildBtn",
].join(", ");

function hasActiveJob() {
  return Boolean(activeJobId || _jobStartInFlight);
}

function setJobActionsLocked(locked) {
  if (locked) {
    document.querySelectorAll(JOB_ACTION_BUTTON_SELECTOR).forEach((btn) => {
      if (!_jobLockedButtonStates.has(btn)) {
        _jobLockedButtonStates.set(btn, btn.disabled);
      }
      btn.disabled = true;
      btn.setAttribute("aria-disabled", "true");
      btn.dataset.jobLocked = "true";
      btn.title = t("uxp.status.job_running_action_title", "Another OpenCut job is running. Wait for it to finish or cancel it first.");
    });
    syncQuickActionButtons();
    return;
  }

  _jobLockedButtonStates.forEach((wasDisabled, btn) => {
    if (!btn || !document.contains(btn)) return;
    btn.removeAttribute("aria-disabled");
    delete btn.dataset.jobLocked;
    if (btn.dataset.backendLocked !== "true") btn.removeAttribute("title");
    if (!btn.classList.contains("loading")) {
      btn.disabled = wasDisabled;
    }
  });
  _jobLockedButtonStates.clear();
  syncQuickActionButtons();
}

function clearTrackedJob(jobId = null) {
  if (!jobId || activeJobId === jobId) {
    activeJobId = null;
  }
  _jobStartInFlight = false;
  setJobActionsLocked(false);
  syncQuickActionButtons();
}

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
    ocGetCaptionTrackSnapshot: "getCaptionTrackSnapshot",
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
          const created = await markerList.getFirstMarkerAtTime(m.time);
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
              const duration = (await child.getOutPoint?.())?.seconds ?? 0;
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

  async function _activeSequenceContext() {
    if (!available || !ppro?.app?.getProjectList) {
      return {
        ok: false,
        reason: "Premiere UXP project APIs are unavailable.",
        reason_code: "uxp_unavailable",
      };
    }
    try {
      const projects = await ppro.app.getProjectList();
      if (!projects || projects.length === 0) {
        return { ok: false, reason: "No open project.", reason_code: "no_open_project" };
      }
      const project = projects[0];
      const sequence = await project.getActiveSequence?.();
      if (!sequence) {
        return { ok: false, reason: "No active sequence.", reason_code: "no_active_sequence", project };
      }
      return { ok: true, project, sequence };
    } catch (e) {
      return { ok: false, reason: e.message, reason_code: "project_api_error" };
    }
  }

  function _timeValueToSeconds(value) {
    if (value == null) return null;
    if (typeof value === "number") return Number.isFinite(value) ? value : null;
    if (typeof value === "string") {
      const parsed = Number(value);
      return Number.isFinite(parsed) ? parsed : null;
    }
    if (typeof value === "object") {
      if (value.seconds != null) {
        const seconds = Number(value.seconds);
        return Number.isFinite(seconds) ? seconds : null;
      }
      if (value.ticks != null) {
        const seconds = Number(value.ticks) / 254016000000;
        return Number.isFinite(seconds) ? seconds : null;
      }
      if (value.value != null) return _timeValueToSeconds(value.value);
    }
    return null;
  }

  async function _captionItemField(item, methodNames, propertyNames) {
    for (const methodName of methodNames) {
      if (typeof item?.[methodName] !== "function") continue;
      try {
        const value = await item[methodName]();
        if (value != null) return value;
      } catch (_) { /* try the next documented or observed shape */ }
    }
    for (const propertyName of propertyNames) {
      if (item?.[propertyName] != null) return item[propertyName];
    }
    return null;
  }

  function _captionText(value) {
    if (value == null) return "";
    if (typeof value === "object") {
      return String(value.text ?? value.value ?? value.content ?? value.name ?? "").trim();
    }
    return String(value).trim();
  }

  function _captionTrackItemTypeCandidates(payload) {
    const candidates = [];
    const add = (value) => {
      const numeric = Number(value);
      if (Number.isFinite(numeric) && !candidates.includes(numeric)) candidates.push(numeric);
    };
    add(payload.trackItemType);
    add(payload.captionTrackItemType);
    add(ppro?.Constants?.TrackItemType?.CAPTION);
    add(ppro?.TrackItemType?.CAPTION);
    add(0);
    add(1);
    return candidates;
  }

  async function _captionTrackItems(track, payload) {
    if (typeof track?.getTrackItems !== "function") {
      return {
        ok: false,
        reason: "CaptionTrack.getTrackItems is unavailable.",
        reason_code: "caption_api_missing",
      };
    }
    const includeEmpty = payload.includeEmptyTrackItems === true;
    const attempts = [];
    for (const trackItemType of _captionTrackItemTypeCandidates(payload)) {
      attempts.push([trackItemType, includeEmpty]);
      attempts.push([trackItemType]);
    }
    attempts.push([]);
    const warnings = [];
    for (const args of attempts) {
      try {
        const items = await track.getTrackItems(...args);
        return {
          ok: true,
          items: Array.from(items || []),
          warnings,
        };
      } catch (e) {
        warnings.push(`getTrackItems(${args.length}) failed: ${e.message}`);
      }
    }
    return {
      ok: false,
      reason: "CaptionTrack.getTrackItems did not return readable items.",
      reason_code: "caption_track_items_unavailable",
      warnings,
    };
  }

  async function _captionSegmentFromItem(item, trackIndex, itemIndex, sequenceName) {
    const textValue = await _captionItemField(
      item,
      ["getText", "getName", "getTitle"],
      ["text", "name", "title", "content"]
    );
    const startValue = await _captionItemField(
      item,
      ["getStartTime", "getStart", "getInPoint"],
      ["start", "startTime", "inPoint"]
    );
    const endValue = await _captionItemField(
      item,
      ["getEndTime", "getEnd", "getOutPoint"],
      ["end", "endTime", "outPoint"]
    );
    const nodeId = await _captionItemField(
      item,
      ["getNodeId", "getId"],
      ["nodeId", "id", "guid"]
    );
    const start = _timeValueToSeconds(startValue) ?? 0;
    const end = _timeValueToSeconds(endValue) ?? start;
    const sourceId = nodeId ? String(nodeId) : `uxp_track_${trackIndex}_item_${itemIndex}`;
    const captionId = `uxp_${sourceId}`;
    return {
      caption_id: captionId,
      source_caption_id: sourceId,
      source_caption_ids: [sourceId],
      source_segment_id: sourceId,
      start,
      end,
      text: _captionText(textValue),
      words: [],
      display_setting_token_ids: [],
      display_settings: {},
      export_format: "uxp_caption_track_snapshot",
      track_index: trackIndex,
      item_index: itemIndex,
      host_locators: {
        sequence_name: sequenceName,
        caption_track_index: trackIndex,
        caption_item_index: itemIndex,
        caption_item_id: sourceId,
      },
    };
  }

  async function getCaptionTrackSnapshot(payload = {}) {
    const parsed = _parseHostPayload(payload, {});
    const context = await _activeSequenceContext();
    if (!context.ok) {
      return {
        ok: false,
        reason: context.reason,
        reason_code: context.reason_code,
        api_names: ["Sequence.getCaptionTrackCount", "Sequence.getCaptionTrack", "CaptionTrack.getTrackItems"],
        tracks: [],
        segments: [],
        count: 0,
      };
    }
    const seq = context.sequence;
    if (typeof seq.getCaptionTrackCount !== "function" || typeof seq.getCaptionTrack !== "function") {
      return {
        ok: false,
        reason: "Active sequence does not expose caption-track read APIs.",
        reason_code: "caption_api_missing",
        api_names: ["Sequence.getCaptionTrackCount", "Sequence.getCaptionTrack", "CaptionTrack.getTrackItems"],
        tracks: [],
        segments: [],
        count: 0,
      };
    }
    try {
      const sequenceName = await seq.getName?.() ?? "";
      const trackCount = Math.max(0, Number(await seq.getCaptionTrackCount()) || 0);
      if (trackCount === 0) {
        return {
          ok: true,
          reason: "Active sequence has no caption tracks.",
          reason_code: "no_caption_tracks",
          read_only: true,
          schema: "opencut.caption_track_snapshot",
          schema_version: 1,
          sequence: { name: sequenceName, caption_track_count: 0 },
          tracks: [],
          segments: [],
          count: 0,
          metadata_preserved: false,
          warnings: [],
        };
      }
      const tracks = [];
      const segments = [];
      const warnings = [];
      for (let trackIndex = 0; trackIndex < trackCount; trackIndex += 1) {
        let track = await seq.getCaptionTrack(trackIndex);
        if (!track && trackIndex === 0) track = await seq.getCaptionTrack();
        if (!track) {
          warnings.push(`caption_track_${trackIndex}_missing`);
          continue;
        }
        const itemResult = await _captionTrackItems(track, parsed);
        if (!itemResult.ok) {
          return {
            ok: false,
            reason: itemResult.reason,
            reason_code: itemResult.reason_code,
            api_names: ["CaptionTrack.getTrackItems"],
            tracks,
            segments,
            count: segments.length,
            warnings: itemResult.warnings || warnings,
          };
        }
        warnings.push(...(itemResult.warnings || []));
        const trackSegments = [];
        for (let itemIndex = 0; itemIndex < itemResult.items.length; itemIndex += 1) {
          const segment = await _captionSegmentFromItem(itemResult.items[itemIndex], trackIndex, itemIndex, sequenceName);
          trackSegments.push(segment);
          segments.push(segment);
        }
        tracks.push({
          track_index: trackIndex,
          count: trackSegments.length,
          segments: trackSegments,
        });
      }
      return {
        ok: true,
        reason_code: segments.length ? "caption_tracks_read" : "caption_tracks_empty",
        read_only: true,
        schema: "opencut.caption_track_snapshot",
        schema_version: 1,
        sequence: { name: sequenceName, caption_track_count: trackCount },
        track_count: trackCount,
        tracks,
        segments,
        count: segments.length,
        metadata_preserved: segments.length > 0,
        warnings,
      };
    } catch (e) {
      console.warn("[PProBridge] getCaptionTrackSnapshot failed:", e.message);
      return { ok: false, reason: e.message, reason_code: "caption_snapshot_failed", tracks: [], segments: [], count: 0 };
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
      case "ocGetCaptionTrackSnapshot": return getCaptionTrackSnapshot(payload);
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
    getCaptionTrackSnapshot,
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
    getCaptionTrackSnapshot: (payload) => PProBridge.getCaptionTrackSnapshot(payload),
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
    if (hasActiveJob()) {
      onError(t("uxp.runtime.job_already_running", "Another OpenCut job is already running."));
      return;
    }

    _jobStartInFlight = true;
    setJobActionsLocked(true);
    let r;
    try {
      r = await BackendClient.post(endpoint, body);
    } catch (err) {
      clearTrackedJob();
      onError(err?.message ?? "Failed to start job");
      return;
    }
    if (!r.ok) {
      clearTrackedJob();
      onError(r.error ?? "Failed to start job");
      return;
    }

    const jobId = r.data?.job_id ?? r.data?.id ?? null;
    if (!jobId) {
      // Synchronous response — job completed inline
      clearTrackedJob();
      onProgress(100, "Done");
      onComplete(r.data);
      return;
    }

    activeJobId = jobId;
    _jobStartInFlight = false;

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
          clearTrackedJob(jobId);
          onComplete(job.result ?? job);
          _fireCompletionHooks();
        } else if (status === "error" || status === "failed" || status === "cancelled") {
          es.close();
          _activeSSE = null;
          clearTrackedJob(jobId);
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
  const MAX_STATUS_POLL_FAILURES = 3;

  function schedulePollJob(jobId, onProgress, onComplete, onError, attempt, statusFailures = 0) {
    setTimeout(() => {
      if (activeJobId === jobId) {
        pollJob(jobId, onProgress, onComplete, onError, attempt, statusFailures);
      }
    }, POLL_INTERVAL_MS);
  }

  async function pollJob(jobId, onProgress, onComplete, onError, attempt = 0, statusFailures = 0) {
    let r;
    try {
      r = await BackendClient.get(`/status/${jobId}`);
    } catch (err) {
      r = { ok: false, error: err?.message ?? "Polling error" };
    }
    if (!r.ok) {
      const nextStatusFailures = statusFailures + 1;
      if (nextStatusFailures < MAX_STATUS_POLL_FAILURES) {
        schedulePollJob(jobId, onProgress, onComplete, onError, attempt, nextStatusFailures);
        return;
      }
      onError(r.error ?? "Polling error");
      clearTrackedJob(jobId);
      _fireCompletionHooks();
      return;
    }

    const job = r.data;
    const status  = job.status ?? "running";
    const pct     = typeof job.progress === "number" ? job.progress : 0;
    const msg     = job.message ?? job.msg ?? "Processing…";

    onProgress(pct, msg);

    if (status === "done" || status === "complete" || status === "success") {
      clearTrackedJob(jobId);
      onComplete(job.result ?? job);
      _fireCompletionHooks();
      return;
    }

    // 'interrupted' is the terminal state set on server startup for jobs
    // that were running when the server died; treat it like an error so
    // the panel doesn't poll forever for progress that will never arrive.
    if (status === "error" || status === "failed" || status === "cancelled" || status === "interrupted") {
      clearTrackedJob(jobId);
      onError(job.error ?? job.message ?? "Job failed");
      _fireCompletionHooks();
      return;
    }

    if (attempt >= MAX_POLL_ATTEMPTS) {
      clearTrackedJob(jobId);
      onError("Polling timed out — the job is still running on the server.");
      _fireCompletionHooks();
      return;
    }

    // Still running — schedule next poll
    schedulePollJob(jobId, onProgress, onComplete, onError, attempt + 1, 0);
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
    if (!activeJobId) return false;
    const jobId = activeJobId;
    // Close SSE stream first to prevent stale events after cancel
    if (_activeSSE) { _activeSSE.close(); _activeSSE = null; }
    try {
      await BackendClient.post(`/cancel/${jobId}`, {});
    } finally {
      clearTrackedJob(jobId);
      _fireCompletionHooks();
    }
    return true;
  }

  /**
   * Poll an already-started job by ID until completion.
   * Returns the final result object or throws.
   */
  function poll(jobId) {
    return new Promise((resolve, reject) => {
      if (hasActiveJob()) {
        reject(new Error(t("uxp.runtime.job_already_running", "Another OpenCut job is already running.")));
        return;
      }
      activeJobId = jobId;
      setJobActionsLocked(true);
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
    setStatus(
      t("uxp.status.workspace", "{workspace} workspace").replace("{workspace}", getWorkspaceTitle(tabId))
    );
  }

  // ── Processing banner ──
  function showProcessing(msg = "Processing…") {
    const banner = document.getElementById("processingBanner");
    if (banner) banner.classList.remove("hidden");
    document.getElementById("mainContent")?.setAttribute("aria-busy", "true");
    setProcessingMsg(msg);
    setProgress(0);
    startElapsedTimer();
    syncQuickActionButtons();
  }

  function hideProcessing() {
    const banner = document.getElementById("processingBanner");
    if (banner) banner.classList.add("hidden");
    document.getElementById("mainContent")?.setAttribute("aria-busy", "false");
    stopElapsedTimer();
    syncQuickActionButtons();
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
    const labels = {
      connected: t("conn.online", "Online"),
      connecting: t("conn.connecting", "Connecting..."),
      disconnected: t("conn.offline", "Offline"),
    };
    label.textContent = labels[state] ?? state;
    updateWorkspaceOverview();
  }

  // ── Toast notifications ──
  function getToastHeading(type) {
    switch (type) {
      case "success": return "Ready";
      case "warning": return "Needs attention";
      case "error": return "Action failed";
      default: return "Status update";
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
    const maxVisibleToasts = 4;
    while (area.children.length >= maxVisibleToasts) {
      area.firstElementChild?.remove();
    }

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
    const locked = btn.dataset.backendLocked === "true" || btn.dataset.jobLocked === "true";
    btn.disabled = loading || locked;
    btn.setAttribute("aria-disabled", btn.disabled ? "true" : "false");
  }

  function clearButtonLoadingStates() {
    document.querySelectorAll("button.loading").forEach((btn) => {
      btn.classList.remove("loading");
      const locked = btn.dataset.backendLocked === "true" || btn.dataset.jobLocked === "true";
      btn.disabled = locked;
      btn.setAttribute("aria-disabled", btn.disabled ? "true" : "false");
    });
  }

  function escapeHtml(str) {
    return escapeHtmlValue(str);
  }

  return {
    switchTab, showProcessing, hideProcessing, setProcessingMsg, setProgress,
    setStatus, setStatusRight, setConnection, showToast,
    bindSlider, initCollapsibles, setButtonLoading, clearButtonLoadingStates,
    escapeHtml,
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
    UIController.showToast(t("uxp.runtime.file_browser_unavailable", "File browser not available in this environment."), "warning");
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
    UIController.showToast(t("uxp.runtime.folder_browser_unavailable", "Folder browser not available in this environment."), "warning");
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
  if (!pathValue) return t("uxp.workspace.no_source_selected", "No source selected");
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
  if (/frame|image|visual/.test(raw)) return t("uxp.search.runtime.kind_visual", "Visual");
  if (/audio|speech|voice/.test(raw)) return t("uxp.search.runtime.kind_audio", "Audio");
  if (/text|transcript|caption|subtitle|segment/.test(raw)) return t("uxp.search.runtime.kind_transcript", "Transcript");
  if (typeof item?.text === "string" || typeof item?.transcript === "string" || typeof item?.snippet === "string") {
    return t("uxp.search.runtime.kind_transcript", "Transcript");
  }
  return t("uxp.search.runtime.kind_library", "Library");
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
    return formatI18n("uxp.search.runtime.time_range", "{start} to {end}", {
      start: formatTimecode(start),
      end: formatTimecode(end),
    });
  }
  if (Number.isFinite(start)) {
    return formatI18n("uxp.search.runtime.time_from", "From {start}", { start: formatTimecode(start) });
  }
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
    return formatI18n("uxp.search.runtime.match_percent", "{percent}% match", { percent: pct });
  }
  return index === 0
    ? t("uxp.search.runtime.top_match", "Top match")
    : formatI18n("uxp.search.runtime.match_number", "Match {index}", { index: index + 1 });
}

function buildSearchResultCard(item, index) {
  const path = getSearchResultPath(item);
  return {
    index,
    path,
    label: path
      ? formatWorkspaceSource(path)
      : formatI18n("uxp.search.runtime.result_number", "Result {index}", { index: index + 1 }),
    kindLabel: getSearchResultKindLabel(item),
    timeLabel: getSearchResultTimeLabel(item),
    preview: getSearchResultPreview(item),
    scoreLabel: getSearchResultScoreLabel(item, index),
  };
}

function normalizeCaptionStyleCatalog(styles) {
  if (!Array.isArray(styles)) return [];
  const seen = new Set();
  const out = [];
  styles.forEach((style) => {
    const id = String(style?.id || "").trim();
    if (!id || seen.has(id)) return;
    seen.add(id);
    out.push({
      id,
      name: String(style?.name || id).trim() || id,
      category: String(style?.category || "uncategorized").trim() || "uncategorized",
      description: String(style?.preview_description || "").trim(),
      animation: String(style?.animation_type || "").trim(),
    });
  });
  return out;
}

function captionStyleGroupLabel(category) {
  const normalized = String(category || "uncategorized").trim();
  if (!normalized) return t("uxp.captions.styles_group_uncategorized", "Uncategorized");
  return normalized
    .split(/[-_\s]+/)
    .filter(Boolean)
    .map((part) => part.slice(0, 1).toUpperCase() + part.slice(1))
    .join(" ");
}

function populateCaptionStyleSelect(styles, source = "backend") {
  const select = document.getElementById("captionStyle");
  if (!select) return false;
  const normalized = normalizeCaptionStyleCatalog(styles);
  if (!normalized.length) return false;

  const previous = select.value || DEFAULT_CAPTION_STYLE_ID;
  select.textContent = "";

  const categories = new Map();
  normalized.forEach((style) => {
    if (!categories.has(style.category)) categories.set(style.category, []);
    categories.get(style.category).push(style);
  });

  categories.forEach((items, category) => {
    const group = document.createElement("optgroup");
    group.label = captionStyleGroupLabel(category);
    items.forEach((style) => {
      const option = document.createElement("option");
      option.value = style.id;
      option.textContent = style.name;
      option.dataset.category = style.category;
      if (style.description) option.title = style.description;
      if (style.animation) option.dataset.animation = style.animation;
      if (source !== "backend") option.dataset.catalogFallback = "true";
      group.appendChild(option);
    });
    select.appendChild(group);
  });

  const values = new Set(normalized.map((style) => style.id));
  select.value = values.has(previous) ? previous : normalized[0].id;
  select.dataset.catalogSource = source;
  select.dataset.catalogCount = String(normalized.length);
  updateCaptionsWorkspaceSummary();
  return true;
}

async function loadCaptionStyleCatalog({ silent = false } = {}) {
  const select = document.getElementById("captionStyle");
  if (!select) return false;
  try {
    const response = await BackendClient.get(CAPTION_STYLES_ENDPOINT);
    const data = response?.data ?? response ?? {};
    const styles = Array.isArray(data.styles) ? data.styles : [];
    if (!response?.ok || !populateCaptionStyleSelect(styles, "backend")) {
      throw new Error(response?.error || t("uxp.captions.runtime.styles_catalog_empty", "Caption style catalog is empty."));
    }
    return true;
  } catch (err) {
    populateCaptionStyleSelect(UXP_CAPTION_STYLE_FALLBACK, "fallback");
    if (!silent) {
      UIController.showToast(
        formatI18n("uxp.captions.runtime.styles_catalog_failed", "Caption styles could not be loaded: {error}", {
          error: err?.message || err,
        }),
        "warning"
      );
    }
    return false;
  }
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
  const backendOnline = isBackendConnected();
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
      ? t(
          "uxp.captions.runtime.copy_unavailable_title",
          "Copy becomes available after a transcript, chapter list, or repeat review has been generated."
        )
      : (_lastCaptionsResult?.copyLabel || t("uxp.captions.copy_output", "Copy Output"));
  }

  const importBtn = document.getElementById("importSrtBtn");
  if (importBtn) {
    const canImport = !!(_lastCaptionsResult && _lastCaptionsResult.kind === "transcript" && _lastCaptionsResult.hasSrt);
    importBtn.disabled = !canImport;
    importBtn.title = canImport
      ? t("uxp.captions.runtime.open_srt_import_title", "Open the timeline import flow for the current SRT output.")
      : t(
          "uxp.captions.runtime.srt_import_unavailable_title",
          "SRT Prep becomes available after a transcript pass produces subtitle output."
        );
  }
}

function updateCaptionsPlanSummary() {
  setTextAndTitle("captionsPlanModel", getSelectLabel("whisperModel", "turbo"), getSelectLabel("whisperModel", "turbo"));
  setTextAndTitle(
    "captionsPlanLanguage",
    getSelectLabel("transcribeLang", t("uxp.captions.language_auto", "Auto-detect")),
    getSelectLabel("transcribeLang", t("uxp.captions.language_auto", "Auto-detect"))
  );
  setTextAndTitle(
    "captionsPlanStyle",
    getSelectLabel("captionStyle", t("uxp.captions.style_minimal_clean", "Minimal Clean")),
    getSelectLabel("captionStyle", t("uxp.captions.style_minimal_clean", "Minimal Clean"))
  );

  const diarization = document.getElementById("enableDiarization")?.checked;
  const wordLevel = document.getElementById("enableWordLevel")?.checked ?? true;
  const wordTiming = wordLevel
    ? t("uxp.captions.runtime.word_timing_on", "Word timing is on")
    : t("uxp.captions.runtime.word_timing_off", "Word timing is off");
  const speakerSplits = diarization
    ? t("uxp.captions.runtime.speaker_splits_on", "Speaker splits are on.")
    : t("uxp.captions.runtime.speaker_splits_off", "Speaker splits are off.");
  const note = `${wordTiming}. ${speakerSplits}`;
  setTextAndTitle("captionsPlanNote", note, note);
}

function updateCaptionsWorkspaceSummary() {
  updateCaptionsPlanSummary();

  const backendOnline = isBackendConnected();
  const sourcePath = document.getElementById("clipPathCaptions")?.value?.trim() || getWorkspaceSource("captions");
  const hasSource = !!sourcePath;

  setTextAndTitle(
    "captionsSourceValue",
    hasSource ? formatWorkspaceSource(sourcePath) : t("uxp.captions.choose_clip_to_start", "Choose a clip to start"),
    sourcePath || t("uxp.captions.runtime.choose_clip_title", "Choose a clip to start a captions pass.")
  );

  if (!backendOnline) {
    setCaptionsSessionState(
      t("conn.offline", "Offline"),
      "error",
      t(
        "uxp.captions.runtime.reconnect_backend",
        "Reconnect the local OpenCut backend before running transcript, chapter, or repeat jobs."
      ),
      "error"
    );
    if (!_lastCaptionsResult) {
      setTextAndTitle(
        "captionsOutputValue",
        t("uxp.captions.runtime.waiting_on_backend", "Waiting on backend"),
        t("uxp.captions.runtime.waiting_on_backend_title", "Reconnect the local OpenCut backend to generate transcript output.")
      );
    }
  } else if (!hasSource) {
    setCaptionsSessionState(
      t("uxp.captions.runtime.choose_media", "Choose media"),
      "empty",
      t("uxp.captions.choose_clip_status", "Choose a clip, then transcribe to unlock chapters, repeat review, and subtitle export."),
      "idle"
    );
    if (!_lastCaptionsResult) {
      setTextAndTitle(
        "captionsOutputValue",
        t("uxp.captions.no_transcript_yet", "No transcript yet"),
        t("uxp.captions.runtime.no_transcript_choose_clip_title", "Choose a clip and run transcription to generate reviewable output.")
      );
    }
  } else if (_lastCaptionsResult) {
    setCaptionsSessionState(
      _lastCaptionsResult.sessionLabel || t("uxp.captions.runtime.result_ready", "Result ready"),
      _lastCaptionsResult.sessionState || "success",
      _lastCaptionsResult.statusMessage || t("uxp.captions.runtime.output_ready_review", "Output ready for review."),
      _lastCaptionsResult.statusState || _lastCaptionsResult.sessionState || "success",
      _lastCaptionsResult.statusTitle || _lastCaptionsResult.outputTitle || _lastCaptionsResult.statusMessage
    );
    setTextAndTitle(
      "captionsOutputValue",
      _lastCaptionsResult.outputLabel || t("uxp.captions.runtime.result_ready", "Result ready"),
      _lastCaptionsResult.outputTitle
        || _lastCaptionsResult.outputLabel
        || t("uxp.captions.runtime.result_ready", "Result ready")
    );
  } else {
    setCaptionsSessionState(
      t("uxp.captions.runtime.ready", "Ready"),
      "success",
      t(
        "uxp.captions.runtime.clip_ready",
        "Clip ready. Start with transcription, then move into chapters or repeat review when the wording is stable."
      ),
      "ready"
    );
    setTextAndTitle(
      "captionsOutputValue",
      t("uxp.captions.no_transcript_yet", "No transcript yet"),
      t("uxp.captions.runtime.no_transcript_selected_title", "Transcribe the selected clip to generate reviewable output.")
    );
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
    transcript: t("uxp.captions.transcript", "Transcript"),
    chapters: t("uxp.captions.runtime.chapter_draft", "Chapter draft"),
    repeat: t("uxp.captions.runtime.repeat_review", "Repeat review"),
  };
  const reviewOutput = t("uxp.captions.runtime.review_output", "Review output");
  const ready = t("uxp.captions.runtime.ready", "Ready");

  if (!area || !body) return;

  area.classList.remove("hidden");
  area.dataset.kind = resultState.kind || "review";
  area.dataset.state = resultState.resultPillState || "success";
  body.value = resultState.content || "";
  body.title = resultState.contentTitle || resultState.outputTitle || resultState.resultMetaTitle || reviewOutput;
  body.setAttribute("aria-label", resultState.header || reviewOutput);
  if (summary) summary.textContent = resultState.summary || t("uxp.captions.ready_to_review", "Ready to review");
  if (meta) {
    meta.textContent = resultState.resultMeta || t("uxp.captions.runtime.review_output_ready", "Review output is ready.");
    meta.title = resultState.resultMetaTitle || resultState.resultMeta || t("uxp.captions.runtime.review_output_ready", "Review output is ready.");
  }
  if (header) header.textContent = resultState.header || t("uxp.captions.runtime.review_output_header", "Review Output");
  setStatusPill(
    "captionsResultPill",
    resultState.resultPillText || ready,
    resultState.resultPillState || "success",
    resultState.resultPillTitle || resultState.summary || ready
  );
  if (copyBtn) copyBtn.textContent = resultState.copyLabel || t("uxp.captions.copy_output", "Copy Output");
  if (importBtn) importBtn.textContent = resultState.importLabel || t("uxp.captions.open_srt_prep", "Open SRT Prep");
  setTextAndTitle(
    "captionsResultTypeValue",
    resultState.insightType || typeLabelMap[resultState.kind] || reviewOutput,
    resultState.insightTypeTitle || resultState.resultMetaTitle || resultState.summary || reviewOutput
  );
  setTextAndTitle(
    "captionsResultLengthValue",
    resultState.insightLength || resultState.outputLabel || resultState.summary || ready,
    resultState.insightLengthTitle || resultState.outputTitle || resultState.summary || ready
  );
  setTextAndTitle(
    "captionsResultNextValue",
    resultState.insightNext || (
      resultState.hasSrt
        ? t("uxp.captions.runtime.next_copy_srt", "Copy SRT or open SRT Prep")
        : t("uxp.captions.runtime.next_copy_notes", "Copy notes or continue review")
    ),
    resultState.insightNextTitle || resultState.statusMessage || resultState.summary || t("uxp.captions.runtime.next_action", "Next action")
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
          ? formatI18n("uxp.captions.runtime.percent_similar", "{percent}% similar", {
              percent: Math.round(similarityRaw > 1 ? similarityRaw : similarityRaw * 100),
            })
          : "";
        const preview = String(
          repeat.text ?? repeat.preview ?? repeat.transcript ?? repeat.reference_text ?? repeat.candidate_text ?? ""
        ).trim();
        const headerParts = [formatI18n("uxp.captions.runtime.repeat_index", "Repeat {index}", { index: index + 1 })];
        if (hasRange) {
          headerParts.push(
            formatI18n("uxp.captions.runtime.time_range", "{start} to {end}", {
              start: formatTimecode(Number.isFinite(start) ? start : 0),
              end: formatTimecode(Number.isFinite(end) ? end : 0),
            })
          );
        }
        if (duration) headerParts.push(duration);
        if (similarity) headerParts.push(similarity);
        return preview ? `${headerParts.join(" - ")}\n${preview}` : headerParts.join(" - ");
      }).join("\n\n")
    : t("uxp.captions.runtime.no_repeated_lines", "No repeated lines were flagged with the current threshold.");

  renderCaptionsResultView({
    kind: "repeat",
    header: t("uxp.captions.runtime.repeat_review_header", "Repeat Review"),
    summary: repeats.length
      ? formatI18n("uxp.captions.runtime.repeat_ranges_flagged", "{count} repeat range{plural} flagged", {
          count: repeats.length,
          plural: repeats.length === 1 ? "" : "s",
        })
      : t("uxp.captions.runtime.no_repeated_takes_flagged", "No repeated takes flagged"),
    content,
    resultPillText: repeats.length
      ? t("uxp.captions.runtime.review_ready", "Review ready")
      : t("uxp.captions.runtime.clean_pass", "Clean pass"),
    resultPillState: repeats.length ? "warning" : "success",
    resultMeta: formatI18n("uxp.captions.runtime.repeat_result_meta", "{source} - {threshold}% threshold - {keepBest}", {
      source: formatWorkspaceSource(clipPath),
      threshold: Math.round(threshold * 100),
      keepBest: keepBest
        ? t("uxp.captions.runtime.keep_best_take_on", "Keep best take on")
        : t("uxp.captions.runtime.keep_best_take_off", "Keep best take off"),
    }),
    resultMetaTitle: clipPath || t("uxp.captions.runtime.repeat_review", "Repeat review"),
    copyLabel: t("uxp.captions.runtime.copy_notes", "Copy Notes"),
    copySuccessLabel: t("uxp.captions.runtime.notes_label", "Notes"),
    importLabel: t("uxp.captions.open_srt_prep", "Open SRT Prep"),
    canOpenSrtImport: false,
    hasSrt: false,
    sessionLabel: repeats.length
      ? t("uxp.captions.runtime.review_ready", "Review ready")
      : t("uxp.captions.runtime.clean_pass", "Clean pass"),
    sessionState: repeats.length ? "warning" : "success",
    statusMessage: repeats.length
      ? t(
          "uxp.captions.runtime.repeat_review_ready_status",
          "Repeat review is ready. Tighten the threshold or move the flagged ranges into your next cleanup pass."
        )
      : t(
          "uxp.captions.runtime.repeat_review_clean_status",
          "No repeated takes were flagged. The current threshold looks clean for this clip."
        ),
    statusState: repeats.length ? "warning" : "success",
    statusTitle: clipPath || t("uxp.captions.runtime.repeat_review", "Repeat review"),
    outputLabel: repeats.length
      ? formatI18n("uxp.captions.runtime.repeat_ranges_flagged", "{count} repeat range{plural} flagged", {
          count: repeats.length,
          plural: repeats.length === 1 ? "" : "s",
        })
      : t("uxp.captions.runtime.no_repeats_flagged", "No repeats flagged"),
    outputTitle: clipPath || t("uxp.captions.runtime.repeat_review", "Repeat review"),
    insightType: t("uxp.captions.runtime.repeat_review", "Repeat review"),
    insightLength: repeats.length
      ? formatI18n("uxp.captions.runtime.repeat_ranges", "{count} repeat {unit}", {
          count: repeats.length,
          unit: repeats.length === 1
            ? t("uxp.captions.runtime.range_singular", "range")
            : t("uxp.captions.runtime.range_plural", "ranges"),
        })
      : t("uxp.captions.runtime.clean_pass", "Clean pass"),
    insightNext: repeats.length
      ? t("uxp.captions.runtime.next_tighten_threshold", "Tighten threshold or continue cleanup")
      : t("uxp.captions.runtime.next_keep_threshold", "Keep current threshold and continue"),
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
  const backendOnline = isBackendConnected();
  const bridgeReady = PProBridge.available();
  const clipPath = document.getElementById("clipPathVideo")?.value?.trim()
    || document.getElementById("clipPathCut")?.value?.trim()
    || getWorkspaceSource("timeline");
  const outputDir = document.getElementById("exportDir")?.value?.trim() || "";
  const srtPath = document.getElementById("srtFilePath")?.value?.trim() || "";
  const trackIndex = Math.max(1, parseInt(document.getElementById("srtTrackIndex")?.value ?? 1, 10) || 1);
  const renamePattern = document.getElementById("renamePattern")?.value?.trim() || "{name}_{index:03d}";
  const smartBinsStrategy = getSelectLabel("binStrategy", t("uxp.timeline.organize_file_type", "File Type"));
  const exportWindows = buildExportWindows();

  setStatusPill(
    "timelineBridgePill",
    bridgeReady ? t("uxp.timeline.uxp_ready", "UXP ready") : t("uxp.timeline.cep_fallback", "CEP fallback"),
    bridgeReady ? "success" : "warning",
    bridgeReady
      ? t("uxp.timeline.runtime.uxp_writeback_available_title", "UXP sequence write-back is available in this Premiere session.")
      : t("uxp.timeline.runtime.direct_write_unavailable_cep_title", "Direct sequence write-back is not available here. Use the CEP panel for dependable timeline execution.")
  );

  const cutsLabel = _lastCutsInfo
    ? formatCountI18n(
        _lastCutsInfo.count,
        "uxp.timeline.runtime.cuts_ready_source_one",
        "{count} cut - {source}",
        "uxp.timeline.runtime.cuts_ready_source_many",
        "{count} cuts - {source}",
        { source: _lastCutsInfo.source }
      )
    : t("uxp.timeline.no_cuts_ready", "No cuts ready");
  const cutsPath = _lastCutsInfo?.clipPath
    ? formatI18n("uxp.timeline.runtime.source_path_suffix", " - {path}", { path: _lastCutsInfo.clipPath })
    : "";
  const cutsTitle = _lastCutsInfo
    ? formatCountI18n(
        _lastCutsInfo.count,
        "uxp.timeline.runtime.cuts_from_source_one",
        "{count} cut from {source}{path}",
        "uxp.timeline.runtime.cuts_from_source_many",
        "{count} cuts from {source}{path}",
        { source: _lastCutsInfo.source, path: cutsPath }
      )
    : t("uxp.timeline.runtime.stage_cuts_title", "Run silence, filler, or multicam cleanup to stage cuts for timeline write-back.");
  setTextAndTitle("timelineCutsValue", cutsLabel, cutsTitle);
  setTextAndTitle("timelineCutsSourceValue", _lastCutsInfo ? cutsLabel : t("uxp.timeline.run_cut_pass_first", "Run a cut pass first"), cutsTitle);

  const markersLabel = _lastMarkersInfo
    ? formatCountI18n(
        _lastMarkersInfo.count,
        "uxp.timeline.runtime.markers_ready_source_one",
        "{count} marker - {source}",
        "uxp.timeline.runtime.markers_ready_source_many",
        "{count} markers - {source}",
        { source: _lastMarkersInfo.source }
      )
    : t("uxp.timeline.no_markers_ready", "No markers ready");
  const markersPath = _lastMarkersInfo?.clipPath
    ? formatI18n("uxp.timeline.runtime.source_path_suffix", " - {path}", { path: _lastMarkersInfo.clipPath })
    : "";
  const markersTitle = _lastMarkersInfo
    ? formatCountI18n(
        _lastMarkersInfo.count,
        "uxp.timeline.runtime.markers_from_source_one",
        "{count} marker from {source}{path}",
        "uxp.timeline.runtime.markers_from_source_many",
        "{count} markers from {source}{path}",
        { source: _lastMarkersInfo.source, path: markersPath }
      )
    : t("uxp.timeline.runtime.stage_markers_title", "Run beat detection to stage markers for sequence write-back or marker-based export.");
  setTextAndTitle("timelineMarkersValue", markersLabel, markersTitle);

  const lastActionLabel = _lastTimelineAction
    ? `${_lastTimelineAction.detailLabel} • ${formatLocaleTime(_lastTimelineAction.time)}`
    : t("uxp.timeline.no_write_back_activity", "No write-back activity");
  setTextAndTitle(
    "timelineLastActionValue",
    lastActionLabel,
    _lastTimelineAction?.title || t("uxp.timeline.runtime.no_timeline_activity_title", "No timeline write-back, export, or validation action has run in this session.")
  );

  const sequenceLabel = bridgeReady
    ? (_lastSequenceInfo?.name || t("uxp.timeline.uxp_bridge_ready", "UXP bridge ready"))
    : t("uxp.timeline.use_cep_panel_for_writeback", "Use CEP panel for write-back");
  const sequenceTitle = bridgeReady
    ? (_lastSequenceInfo?.name || t("uxp.timeline.runtime.uxp_bridge_ready_title", "The UXP bridge is ready. Direct sequence calls will use the active sequence when available."))
    : t("uxp.timeline.runtime.direct_write_unavailable_short", "Direct sequence write-back is not available in this UXP session.");
  setTextAndTitle("timelineSequenceValue", sequenceLabel, sequenceTitle);

  const exportSourceLabel = exportWindows.length
    ? formatCountI18n(
        exportWindows.length,
        "uxp.timeline.runtime.export_window_ready_one",
        "{count} window ready",
        "uxp.timeline.runtime.export_window_ready_many",
        "{count} windows ready"
      )
    : t("uxp.timeline.awaiting_cuts_or_markers", "Awaiting cuts or markers");
  const exportSource = Array.isArray(lastMarkers) && lastMarkers.length
    ? t("uxp.timeline.runtime.marker_lower", "marker")
    : t("uxp.timeline.runtime.cut_lower", "cut");
  const exportSourceTitle = exportWindows.length
    ? formatCountI18n(
        exportWindows.length,
        "uxp.timeline.runtime.export_windows_staged_one",
        "{count} export window is staged from the latest {source} pass.",
        "uxp.timeline.runtime.export_windows_staged_many",
        "{count} export windows are staged from the latest {source} pass.",
        { source: exportSource }
      )
    : t("uxp.timeline.runtime.stage_export_windows_title", "Run beat detection or create cuts first, then export those windows from the current clip.");
  setTextAndTitle("timelineExportSourceValue", exportSourceLabel, exportSourceTitle);
  setTextAndTitle("timelineExportOutputValue", outputDir ? formatWorkspaceSource(outputDir) : t("uxp.timeline.choose_output_folder", "Choose output folder"), outputDir || t("uxp.timeline.runtime.choose_export_destination_title", "Choose an export destination for marker-based segments."));

  setStatusPill("timelineRenamePill", t("uxp.timeline.cep_panel_required", "CEP panel required"), "warning", t("uxp.timeline.runtime.batch_rename_cep_title", "Batch rename still executes through the CEP panel on this build."));
  setTextAndTitle("timelineRenamePatternValue", renamePattern, renamePattern);
  setStatusPill("timelineSmartBinsPill", t("uxp.timeline.cep_panel_required", "CEP panel required"), "warning", t("uxp.timeline.runtime.smart_bins_cep_title", "Smart bin execution still runs through the CEP panel on this build."));
  setTextAndTitle("timelineSmartBinsValue", smartBinsStrategy, smartBinsStrategy);

  setTextAndTitle("timelineSrtValue", srtPath ? formatWorkspaceSource(srtPath) : t("uxp.timeline.choose_srt_file", "Choose .srt file"), srtPath || t("uxp.timeline.runtime.choose_srt_file_title", "Choose an .srt file to validate before the CEP ocAddNativeCaptionTrack bridge places it."));
  const trackLabel = formatI18n("uxp.timeline.runtime.track_index", "Track {index}", { index: trackIndex });
  setTextAndTitle("timelineSrtTrackValue", trackLabel, formatI18n("uxp.timeline.runtime.target_track_title", "Track {index} target for the CEP ocAddNativeCaptionTrack handoff.", { index: trackIndex }));

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
      t("uxp.timeline.runtime.backend_reconnect_timeline_status", "Reconnect the local backend before exporting windows, validating captions, or packaging timeline handoff."),
      "error"
    );
  } else if (!bridgeReady && (hasCuts || hasMarkers)) {
    setTimelineStatus(
      t("uxp.timeline.runtime.assets_ready_cep_status", "Timeline assets are ready, but direct sequence write-back still needs the CEP panel on this Premiere setup. OTIO export and SRT validation remain available here."),
      "warning"
    );
  } else if (!bridgeReady) {
    setTimelineStatus(
      t("uxp.timeline.runtime.generate_assets_cep_status", "Generate cuts or beat markers first. Direct sequence write-back will fall back to the CEP panel on this setup."),
      "warning"
    );
  } else if (_lastTimelineAction) {
    setTimelineStatus(_lastTimelineAction.statusMessage, _lastTimelineAction.state, _lastTimelineAction.title);
  } else if (!hasCuts && !hasMarkers) {
    setTimelineStatus(
      t("uxp.timeline.status_line", "Generate cuts or beat markers first, then return here to write back, export OTIO, or validate captions for sequence import."),
      "idle"
    );
  } else {
    setTimelineStatus(
      t("uxp.timeline.runtime.assets_ready_actions_status", "Timeline assets are ready. Apply cuts, add markers, export OTIO, or validate an SRT before the handoff pass."),
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
  const kicker = t(guide.kickerKey, guide.kicker);
  const title = t(guide.titleKey, guide.title);
  const text = t(guide.textKey, guide.text);
  const actionLabel = t(guide.actionLabelKey, guide.actionLabel || t("common.open", "Open"));

  if (guideEl) {
    guideEl.dataset.state = guideState;
    guideEl.title = text || title || "";
  }
  if (kickerEl) kickerEl.textContent = kicker;
  if (titleEl) titleEl.textContent = title;
  if (textEl) textEl.textContent = text;
  if (actionEl) {
    actionEl.dataset.action = guide.action || "";
    actionEl.textContent = actionLabel;
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
  const connectionState = document.getElementById("connectionStatus")?.dataset.state || "disconnected";
  const backendLabel = t(CONNECTION_LABEL_KEYS[connectionState] || "conn.offline", "Offline");
  const backendOnline = isBackendConnected();
  const overviewTitle = document.getElementById("workspaceOverviewTitle");
  const overviewSubtitle = document.getElementById("workspaceOverviewSubtitle");
  const sourceValue = document.getElementById("workspaceSourceValue");
  const backendValue = document.getElementById("workspaceBackendValue");
  const libraryValue = document.getElementById("workspaceLibraryValue");

  applyWorkspaceShellState(activeTab, backendOnline, sourcePath);

  if (overviewTitle) overviewTitle.textContent = localizeWorkspaceMeta(meta, "title");
  if (overviewSubtitle) overviewSubtitle.textContent = localizeWorkspaceMeta(meta, "subtitle");
  if (sourceValue) {
    sourceValue.textContent = formatWorkspaceSource(sourcePath);
    sourceValue.title = sourcePath || t("uxp.workspace.choose_source_title", "Choose a clip or paste a path to start");
    sourceValue.closest(".oc-workspace-meta-item")?.setAttribute("data-state", sourcePath ? "ready" : "empty");
  }
  if (backendValue) {
    backendValue.textContent = backendLabel;
    backendValue.closest(".oc-workspace-meta-item")?.setAttribute("data-state", backendOnline ? "online" : "offline");
  }
  if (libraryValue) {
    const count = Array.isArray(_projectClips) ? _projectClips.length : 0;
    libraryValue.textContent = (count === 1
      ? t("uxp.workspace.library_clip_count_one", "{count} clip")
      : t("uxp.workspace.library_clip_count_many", "{count} clips")
    ).replace("{count}", String(count));
    libraryValue.closest(".oc-workspace-meta-item")?.setAttribute("data-state", count > 0 ? "ready" : "empty");
  }

  let guide = WORKSPACE_GUIDES[activeTab] || WORKSPACE_GUIDES.cut;
  if (!backendOnline) {
    guide = {
      kickerKey: "uxp.guide.backend_offline_kicker",
      kicker: "Backend offline",
      titleKey: "uxp.guide.backend_offline_title",
      title: "Start the OpenCut backend to begin processing.",
      textKey: "uxp.guide.backend_offline_text",
      text: "Run Start-OpenCut.bat (from the installer) or open a terminal and run: python -m opencut.server. Keep the terminal open while using the panel. The Live Updates Bridge in Settings is optional and not required for the panel to work.",
      state: "error",
      action: "refresh-backend",
      actionLabelKey: "uxp.guide.refresh_backend",
      actionLabel: "Refresh Backend",
    };
  } else if (TABS_REQUIRING_SOURCE.has(activeTab) && !sourcePath) {
    guide = {
      kickerKey: "uxp.guide.choose_media_kicker",
      kicker: "Choose media",
      titleKey: "uxp.guide.choose_media_title",
      title: "Choose one active shot to unlock this workspace.",
      textKey: "uxp.guide.choose_media_text",
      text: "OpenCut keeps the current clip in sync across Cut, Captions, Audio, and Video so you can move through the edit without repeated setup.",
      state: "warning",
      action: "choose-clip",
      actionLabelKey: "uxp.workspace.choose_media",
      actionLabel: "Choose Media",
    };
  } else if (activeTab === "timeline" && !lastCuts.length && !lastMarkers.length) {
    guide = {
      kickerKey: "uxp.guide.writeback_ready_kicker",
      kicker: "Ready for write-back",
      titleKey: "uxp.guide.writeback_ready_title",
      title: "Generate cuts or markers first, then bring them back to the sequence.",
      textKey: "uxp.guide.writeback_ready_text",
      text: "Run a cleanup or beat pass in Cut or Audio, then return here to apply the result, export OTIO, or batch markers.",
      state: "warning",
      action: "switch-cut",
      actionLabelKey: "uxp.guide.open_cut_workspace",
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
      label: t("uxp.deliverables.session_temp_folder", "Session temp folder"),
      title: t("uxp.deliverables.session_temp_folder_title", "Deliverables will be saved to the session temp folder until you choose an output folder."),
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
      label: t("uxp.deliverables.runtime.no_documents_selected", "No documents selected"),
      title: t("uxp.deliverables.runtime.select_document_before_package", "Select at least one handoff document before generating a package."),
    };
  }
  if (selectedTypes.length === 4) {
    return {
      label: t("uxp.deliverables.all_four_handoff_sheets", "All four handoff sheets"),
      title: t("uxp.deliverables.runtime.all_documents_package_title", "VFX Sheet, ADR List, Music Cue Sheet, and Asset List will be generated in this package."),
    };
  }
  if (selectedTypes.length === 1) {
    const label = getDeliverableLabel(selectedTypes[0]);
    return {
      label,
      title: formatI18n("uxp.deliverables.runtime.single_document_selected_title", "{label} is the only document selected for this package.", { label }),
    };
  }
  const labels = selectedTypes.map((type) => getDeliverableLabel(type));
  return {
    label: formatI18n("uxp.deliverables.runtime.documents_selected", "{count} documents selected", { count: selectedTypes.length }),
    title: labels.join(" | "),
  };
}

function getDeliverablesFormatSummary() {
  return {
    value: "csv",
    label: t("uxp.deliverables.csv_handoff_sheets", "CSV handoff sheets"),
    title: t("uxp.deliverables.runtime.csv_handoff_sheets_title", "The UXP deliverables workflow currently exports CSV handoff sheets."),
  };
}

function updateDeliverablesPlanSummary(outputSummary = getDeliverablesOutputSummary()) {
  const selectedTypes = getSelectedReportTypes();
  const selectionSummary = getDeliverablesSelectionSummary(selectedTypes);
  const formatSummary = getDeliverablesFormatSummary();
  const backendOnline = isBackendConnected();
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
    const label = getDeliverableLabel(type);
    btn.disabled = !canGenerate;
    btn.title = !backendOnline
      ? formatI18n("uxp.deliverables.runtime.reconnect_before_generating_label", "Reconnect the backend before generating {label}.", { label })
      : (hasSequence
          ? formatI18n("uxp.deliverables.runtime.generate_label_csv_title", "Generate {label} as a CSV handoff sheet.", { label })
          : formatI18n("uxp.deliverables.runtime.load_sequence_before_generating_label", "Load sequence info before generating {label}.", { label }));
  });

  const reportBtn = document.getElementById("runFullReportBtn");
  if (reportBtn && !reportBtn.classList.contains("loading")) {
    if (!selectedTypes.length) {
      reportBtn.textContent = t("uxp.deliverables.runtime.select_documents_first", "Select Documents First");
    } else if (selectedTypes.length === 1) {
      reportBtn.textContent = formatI18n("uxp.deliverables.runtime.generate_label", "Generate {label}", { label: getDeliverableLabel(selectedTypes[0]) });
    } else if (selectedTypes.length === 4) {
      reportBtn.textContent = t("uxp.deliverables.generate_full_report", "Generate Full Report");
    } else {
      reportBtn.textContent = formatI18n("uxp.deliverables.runtime.generate_doc_package", "Generate {count}-Doc Package", { count: selectedTypes.length });
    }
    reportBtn.disabled = !canGenerate || selectedTypes.length === 0;
    reportBtn.title = !selectedTypes.length
      ? t("uxp.deliverables.runtime.select_document_before_package", "Select at least one handoff document before generating a package.")
      : (!backendOnline
          ? t("uxp.deliverables.runtime.reconnect_before_selected_package", "Reconnect the backend before generating the selected handoff package.")
          : (!hasSequence
          ? t("uxp.deliverables.runtime.load_sequence_before_selected_package", "Load sequence info before generating the selected handoff package.")
          : formatI18n("uxp.deliverables.runtime.package_title_with_format", "{selectionTitle} {formatTitle}", {
              selectionTitle: selectionSummary.title,
              formatTitle: formatSummary.title,
            })));
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
  const backendOnline = isBackendConnected();

  if (info) {
    const resolution = (info.width && info.height) ? `${info.width} × ${info.height}` : t("uxp.deliverables.runtime.unknown_size", "Unknown size");
    const duration = typeof info.duration === "number" ? formatTimecode(info.duration) : t("uxp.deliverables.runtime.unknown_duration", "Unknown duration");
    const sequenceName = info.name || t("uxp.deliverables.runtime.active_sequence", "Active Sequence");
    setStatusPill(
      "seqInfoStatePill",
      t("uxp.deliverables.runtime.loaded", "Loaded"),
      "success",
      t("uxp.deliverables.runtime.sequence_info_ready", "Sequence info is ready for deliverables.")
    );
    setTextAndTitle(
      "seqInfoSummary",
      formatI18n("uxp.deliverables.runtime.sequence_summary", "{name} | {resolution} | {duration}", { name: sequenceName, resolution, duration }),
      formatI18n("uxp.deliverables.runtime.sequence_summary", "{name} | {resolution} | {duration}", { name: sequenceName, resolution, duration })
    );
    setTextAndTitle(
      "deliverablesSequenceValue",
      sequenceName,
      sequenceName
    );
    setDeliverablesButtonsDisabled(false);
  } else {
    setStatusPill(
      "seqInfoStatePill",
      t("uxp.deliverables.not_loaded", "Not Loaded"),
      "empty",
      t("uxp.deliverables.sequence_summary_placeholder", "Load the active Premiere sequence before generating handoff docs.")
    );
    setTextAndTitle(
      "seqInfoSummary",
      t("uxp.deliverables.sequence_summary_placeholder", "Load the active Premiere sequence before generating handoff docs."),
      t("uxp.deliverables.sequence_summary_placeholder", "Load the active Premiere sequence before generating handoff docs.")
    );
    setTextAndTitle(
      "deliverablesSequenceValue",
      t("uxp.deliverables.load_sequence_info_value", "Load sequence info"),
      t("uxp.deliverables.load_sequence_info_before_docs", "Load sequence info before generating handoff docs.")
    );
    setDeliverablesButtonsDisabled(true);
  }

  setTextAndTitle("deliverablesOutputValue", output.label, output.title);
  updateDeliverablesPlanSummary(output);

  if (_lastDeliverableActivity) {
    const activity = _lastDeliverableActivity;
    const label = activity.count
      ? formatCountI18n(
          activity.count,
          "uxp.deliverables.runtime.activity_docs_one",
          "{label} | {count} doc",
          "uxp.deliverables.runtime.activity_docs_many",
          "{label} | {count} docs",
          { label: activity.label }
        )
      : activity.label;
    const relative = formatRelativeTime(activity.time);
    setTextAndTitle(
      "deliverablesLastExportValue",
      formatI18n("uxp.deliverables.runtime.activity_with_time", "{label} ({relative})", { label, relative }),
      activity.output || formatI18n("uxp.deliverables.runtime.activity_at_time", "{label} at {time}", { label, time: formatLocaleTime(activity.time) })
    );
  } else {
    setTextAndTitle(
      "deliverablesLastExportValue",
      t("uxp.deliverables.no_exports_yet", "No exports yet"),
      t("uxp.deliverables.no_deliverables_generated", "No deliverables have been generated in this session.")
    );
  }

  if (!backendOnline) {
    setDeliverablesStatus(t("uxp.deliverables.status_backend_offline", "Reconnect the local backend before generating handoff documents or report packages."), "error");
  } else if (!_lastSequenceInfo) {
    setDeliverablesStatus(t("uxp.deliverables.status_line", "Load sequence info, choose a destination if needed, then generate the handoff docs you need."), "idle");
  } else if (!selectedTypes.length) {
    setDeliverablesStatus(t("uxp.deliverables.status_select_handoff_document", "Sequence info is ready. Select at least one handoff document below, or generate a single sheet above for this pass."), "warning");
  } else if (!_lastDeliverableActivity && !hasOutputDestination) {
    setDeliverablesStatus(t("uxp.deliverables.status_choose_output_folder", "Sequence ready. Choose an output folder if you want handoff docs saved somewhere more durable than the session temp folder."), "warning", output.title);
  } else if (_lastDeliverableActivity) {
    const lastLabel = _lastDeliverableActivity.count
      ? formatI18n("uxp.deliverables.runtime.last_activity_finished", "{label} finished", { label: _lastDeliverableActivity.label })
      : formatI18n("uxp.deliverables.runtime.last_activity_ready", "{label} ready", { label: _lastDeliverableActivity.label });
    setDeliverablesStatus(
      formatI18n("uxp.deliverables.runtime.last_activity_next_handoff", "{label}. Generate another document or refresh the sequence info before the next handoff pass.", { label: lastLabel }),
      "success",
      _lastDeliverableActivity.output
    );
  } else {
    setDeliverablesStatus(t("uxp.deliverables.status_ready_run_report", "Sequence info is ready. Generate a single document or run the full report when the handoff package is ready."), "ready");
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
    clearBtn.textContent = t("uxp.search.clear_index", "Clear Index");
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
    clearBtn.textContent = t("uxp.search.runtime.confirm_clear", "Confirm Clear");
    clearBtn.title = t("uxp.search.runtime.confirm_clear_title", "Click again within 8 seconds to clear the current search index.");
  }
  setIndexStatus(t("uxp.search.runtime.confirm_clear_status", "Click Confirm Clear again to remove the current search index."), "warning");
  UIController.showToast(t("uxp.search.runtime.confirm_clear_status", "Click Confirm Clear again to remove the current search index."), "warning");
  return false;
}

function syncSearchPanelState() {
  const backendOnline = isBackendConnected();
  const folder = document.getElementById("indexFolder")?.value?.trim() || "";
  const query = document.getElementById("searchQuery")?.value?.trim() || "";
  const nlpCommand = document.getElementById("nlpCommand")?.value?.trim() || "";
  const indexedFiles = Number(_lastIndexStats?.total_files || 0);
  const hasIndex = indexedFiles > 0;

  const runIndexBtn = document.getElementById("runIndexLibBtn");
  if (runIndexBtn && !runIndexBtn.classList.contains("loading")) {
    runIndexBtn.disabled = !backendOnline || !folder;
    runIndexBtn.title = !backendOnline
      ? t("uxp.search.runtime.reconnect_before_indexing", "Reconnect the backend before indexing the library.")
      : (folder
          ? t("uxp.search.runtime.build_searchable_library_title", "Build a searchable library from the selected media folder.")
          : t("uxp.search.runtime.choose_folder_before_indexing", "Choose a media folder before indexing the library."));
  }

  const clearBtn = document.getElementById("clearIndexBtn");
  if (clearBtn && !clearBtn.classList.contains("loading")) {
    if (_clearIndexConfirmUntil && _clearIndexConfirmUntil <= Date.now()) {
      resetClearIndexConfirmation({ resync: false });
    }
    clearBtn.disabled = !backendOnline || !hasIndex;
    if (_clearIndexConfirmUntil > Date.now() && backendOnline && hasIndex) {
      clearBtn.textContent = t("uxp.search.runtime.confirm_clear", "Confirm Clear");
      clearBtn.title = t("uxp.search.runtime.confirm_clear_title", "Click again within 8 seconds to clear the current search index.");
    } else {
      clearBtn.textContent = t("uxp.search.clear_index", "Clear Index");
      clearBtn.title = !backendOnline
        ? t("uxp.search.runtime.reconnect_before_clearing_index", "Reconnect the backend before clearing the search index.")
        : (hasIndex
            ? t("uxp.search.runtime.clear_current_index_title", "Clear the current search index. You can rebuild it any time.")
            : t("uxp.search.runtime.build_index_before_clearing", "Build an index before clearing it."));
    }
  }

  const searchBtn = document.getElementById("runFootageSearchBtn");
  if (searchBtn && !searchBtn.classList.contains("loading")) {
    searchBtn.disabled = !backendOnline || !hasIndex || !query;
    searchBtn.title = !backendOnline
      ? t("uxp.search.runtime.reconnect_before_searching", "Reconnect the backend before searching the library.")
      : (!hasIndex
          ? t("uxp.search.runtime.index_folder_before_searching", "Index a folder before searching the library.")
          : (query
              ? t("uxp.search.runtime.search_indexed_library_title", "Search the indexed library and load the best shot back into the workspace.")
              : t("uxp.search.runtime.enter_query_before_searching", "Enter a descriptive query before searching the library.")));
  }

  const nlpBtn = document.getElementById("runNlpBtn");
  if (nlpBtn && !nlpBtn.classList.contains("loading")) {
    nlpBtn.disabled = !backendOnline || !nlpCommand;
    nlpBtn.title = !backendOnline
      ? t("uxp.search.runtime.reconnect_before_nlp", "Reconnect the backend before running natural-language commands.")
      : (nlpCommand
          ? t("uxp.search.runtime.parse_instruction_title", "Parse the current edit instruction and review the result before applying it.")
          : t("uxp.search.runtime.enter_instruction_before_nlp", "Enter a natural-language edit instruction before running it."));
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
      setStatusPill(
        "indexStatePill",
        t("uxp.search.runtime.unavailable", "Unavailable"),
        "warning",
        t("uxp.search.runtime.index_status_unavailable_title", "The panel could not read the search index status.")
      );
      setTextAndTitle(
        "indexStatsValue",
        t("uxp.search.runtime.index_status_unavailable", "Index status unavailable"),
        t("uxp.search.runtime.index_status_unavailable_title", "The panel could not read the search index status.")
      );
      setIndexStatus(t("uxp.search.runtime.index_status_read_failed", "Could not read the current library index. Reconnect the backend, then refresh or re-index the folder."), "warning");
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
    ? formatCountI18n(
        stats.total_files,
        "uxp.search.runtime.files_indexed_one",
        "{count} file indexed",
        "uxp.search.runtime.files_indexed_many",
        "{count} files indexed"
      )
    : t("uxp.search.zero_files_indexed", "0 files indexed");
  const statsTitle = stats.total_files
    ? formatI18n("uxp.search.runtime.index_stats_title", "{files} files indexed, {segments} transcript segments, {size} on disk.", {
        files: stats.total_files,
        segments: stats.total_segments,
        size: formatBytes(stats.index_size_bytes),
      })
    : t("uxp.search.runtime.no_footage_index_yet", "No footage index has been built yet.");

  setTextAndTitle("indexStatsValue", statsLabel, statsTitle);

  if (!options.preserveMessage) {
    if (stats.total_files > 0) {
      setStatusPill("indexStatePill", t("uxp.search.ready", "Ready"), "success", statsTitle);
      setIndexStatus(
        formatCountI18n(
          stats.total_files,
          "uxp.search.runtime.library_ready_one",
          "Library ready. {count} indexed file can be searched right away.",
          "uxp.search.runtime.library_ready_many",
          "Library ready. {count} indexed files can be searched right away."
        ),
        "success",
        statsTitle
      );
    } else {
      setStatusPill("indexStatePill", t("uxp.search.runtime.empty", "Empty"), "empty", statsTitle);
      setIndexStatus(t("uxp.search.runtime.index_folder_for_results", "Index a folder to make descriptive search results available in this workspace."), "idle");
    }
  }

  syncSearchPanelState();
  return stats;
}

async function clearFootageIndex() {
  if (!(Number(_lastIndexStats?.total_files || 0) > 0)) {
    resetClearIndexConfirmation();
    setIndexStatus(t("uxp.search.runtime.build_index_before_clearing", "Build an index before clearing it."), "warning");
    UIController.showToast(t("uxp.search.runtime.build_index_before_clearing", "Build an index before clearing it."), "warning");
    syncSearchPanelState();
    return;
  }

  if (!requireClearIndexConfirmation()) {
    return;
  }

  UIController.setButtonLoading("clearIndexBtn", true);
  setStatusPill(
    "indexStatePill",
    t("uxp.search.runtime.clearing", "Clearing"),
    "working",
    t("uxp.search.runtime.clearing_index_title", "Clearing the current search index.")
  );
  setIndexStatus(t("uxp.search.runtime.clearing_index", "Clearing the current search index..."), "working");

  const r = await BackendClient.del("/search/index");

  UIController.setButtonLoading("clearIndexBtn", false);

  if (!r.ok) {
    const error = r.error || t("uxp.search.runtime.clear_index_failed", "Failed to clear the search index.");
    setStatusPill("indexStatePill", t("uxp.search.runtime.error", "Error"), "error", error);
    setIndexStatus(error, "error");
    UIController.showToast(error, "error");
    return;
  }

  resetSearchResults(
    t("uxp.search.search_the_library", "Search the library"),
    t("uxp.search.runtime.reindex_after_clear", "Index a folder again to bring searchable media back into this workspace.")
  );
  setTextAndTitle(
    "searchStatus",
    t("uxp.search.runtime.index_cleared_status", "The search index has been cleared. Re-index a folder to search footage again."),
    t("uxp.search.runtime.index_cleared_status", "The search index has been cleared. Re-index a folder to search footage again.")
  );
  await refreshFootageIndexStats({ preserveMessage: true, silent: true });
  setStatusPill("indexStatePill", t("uxp.search.runtime.empty", "Empty"), "empty", t("uxp.search.runtime.index_empty_until_reindex", "The search index is empty until you index a folder again."));
  setTextAndTitle("indexStatsValue", t("uxp.search.zero_files_indexed", "0 files indexed"), t("uxp.search.runtime.index_empty_until_reindex", "The search index is empty until you index a folder again."));
  setIndexStatus(t("uxp.search.runtime.search_index_cleared", "Search index cleared. Re-index a folder to make library results available again."), "success");
  UIController.showToast(t("uxp.search.runtime.search_index_cleared_toast", "Search index cleared."), "success");
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
    UIController.showToast(t("uxp.deliverables.runtime.open_sequence_before_docs", "Open an active Premiere sequence, then load sequence info before generating deliverables."), "warning");
    UIController.setStatus(t("uxp.deliverables.runtime.load_sequence_before_docs_status", "Load sequence info before generating deliverables."), "error");
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
    clipSelect.innerHTML = `<option value="">${UIController.escapeHtml(t("uxp.common.select_clip", "-- Select a clip --"))}</option>` +
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
  if (!clipPath) { showSelectClipWarning(); return; }

  const threshold = parseFloat(document.getElementById("silenceThreshold")?.value ?? -35);
  const minSilence = parseFloat(document.getElementById("minSilence")?.value ?? 0.5);
  const padding = parseInt(document.getElementById("silencePadding")?.value ?? 80);
  const mode = document.getElementById("silenceMode")?.value ?? "remove";
  const detectMethod = document.getElementById("silenceDetectMethod")?.value ?? "auto";

  UIController.setButtonLoading("runSilenceBtn", true);
  UIController.showProcessing(t("uxp.cut.runtime.detecting_silences", "Detecting silences..."));
  UIController.setStatus(t("uxp.cut.runtime.running_silence_removal", "Running silence removal..."));

  await JobPoller.start(
    "/silence",
    { filepath: clipPath, threshold: threshold, min_duration: minSilence, padding_before: padding / 1000, padding_after: padding / 1000, mode, method: detectMethod },
    (pct, msg) => {
      UIController.setProgress(pct);
      UIController.setProcessingMsg(msg || t("processing.processing", "Processing..."));
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
        UIController.showToast(formatI18n("uxp.cut.runtime.silences_removed", "Removed {count} silence region(s).", { count: cuts.length }), "success");
        UIController.setStatus(formatI18n("uxp.cut.runtime.silence_done_status", "Done - {count} cuts", { count: cuts.length }));
      } else if (result.xml_path || result.output_path) {
        const out = result.xml_path || result.output_path;
        UIController.showToast(formatI18n("uxp.runtime.output_path", "Output: {path}", { path: out }), "success");
        UIController.setStatus(t("uxp.cut.runtime.silence_complete", "Silence removal complete."));
      } else {
        UIController.showToast(t("uxp.cut.runtime.no_silences_found", "No silences found with current settings."), "info");
        UIController.setStatus(t("uxp.cut.runtime.no_silences_status", "No silences detected."));
      }
    },
    (err) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runSilenceBtn", false);
      UIController.showToast(formatI18n("uxp.runtime.error_prefix", "Error: {error}", { error: err }), "error");
      UIController.setStatus(t("uxp.cut.runtime.silence_error_status", "Error during silence removal."));
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
    const longestSuffix = longestCut
      ? formatI18n("uxp.cut.runtime.longest_cut_suffix", " - longest {duration}", { duration: formatCompactDuration(longestCut) })
      : "";
    summary.textContent = cuts.length
      ? formatI18n("uxp.cut.runtime.cut_windows_summary", "{count} cut window{plural} - {removed} removed{longest}", {
          count: cuts.length,
          plural: cuts.length === 1 ? "" : "s",
          removed: formatCompactDuration(totalRemoved),
          longest: longestSuffix,
        })
      : t("uxp.cut.runtime.no_cuts_detected", "No cuts detected");
  }
  setTextAndTitle(
    "cutResultSourceValue",
    clipPath ? formatWorkspaceSource(clipPath) : t("uxp.cut.runtime.awaiting_reviewed_pass", "Awaiting reviewed pass"),
    clipPath || t("uxp.cut.runtime.choose_clip_cleanup_title", "Choose a clip and run a cleanup pass to stage cut windows.")
  );
  setTextAndTitle(
    "cutResultRemovedValue",
    cuts.length ? formatI18n("uxp.cut.runtime.removed_total", "{duration} total", { duration: formatCompactDuration(totalRemoved) }) : "0 s",
    cuts.length
      ? formatI18n("uxp.cut.runtime.removed_total_title", "{duration} total removed across {count} cut window{plural}.", {
          duration: formatCompactDuration(totalRemoved),
          count: cuts.length,
          plural: cuts.length === 1 ? "" : "s",
        })
      : t("uxp.cut.runtime.no_cut_windows_staged", "No cut windows are staged yet.")
  );
  setTextAndTitle(
    "cutResultNextValue",
    cuts.length ? t("uxp.cut.runtime.apply_latest_cuts", "Apply latest cuts to timeline") : t("uxp.cut.runtime.adjust_settings_rerun", "Adjust settings and rerun"),
    cuts.length
      ? t("uxp.cut.runtime.apply_latest_cuts_title", "Review the suggested windows, then apply the pass to the active sequence.")
      : t("uxp.cut.runtime.adjust_settings_rerun_title", "Refine the thresholds or switch cleanup mode, then run another pass.")
  );
  body.innerHTML = cuts.length
    ? cuts.map((cut, index) => {
        const start = Number(cut.start);
        const end = Number(cut.end);
        const duration = Math.max(0, end - start);
        return `
          <div class="oc-result-row">
            <span class="oc-result-chip">${UIController.escapeHtml(formatI18n("uxp.cut.runtime.cut_index", "Cut {index}", { index: index + 1 }))}</span>
            <div class="oc-result-copy">
              <strong>${formatTimecode(start)} to ${formatTimecode(end)}</strong>
              <span>${UIController.escapeHtml(formatI18n("uxp.cut.runtime.duration_removed", "{duration} removed", { duration: formatCompactDuration(duration) }))}</span>
            </div>
          </div>`;
      }).join("")
    : `
      <div class="oc-empty-state oc-empty-state-inline">
        <div class="oc-empty-state-kicker">${UIController.escapeHtml(t("uxp.cut.runtime.no_changes_yet", "No changes yet"))}</div>
        <p>${UIController.escapeHtml(t("uxp.cut.runtime.no_changes_hint", "Run silence detection or filler cleanup to generate timeline-ready cuts here."))}</p>
      </div>`;
  area.focus();
}

/** ── FILLER WORD DETECTION ── */
async function runFillerDetection() {
  const clipPath = document.getElementById("clipPathCut")?.value?.trim();
  if (!clipPath) { showSelectClipWarning(); return; }

  const words   = document.getElementById("fillerWords")?.value ?? "um,uh,like";
  const padding = parseInt(document.getElementById("fillerPadding")?.value ?? 50);
  const fillerBackend = document.getElementById("fillerBackend")?.value ?? "whisper";

  UIController.setButtonLoading("runFillerBtn", true);
  UIController.showProcessing(
    fillerBackend === "crisper"
      ? t("uxp.cut.runtime.detecting_fillers_crisper", "Detecting fillers with CrisperWhisper...")
      : t("uxp.cut.runtime.detecting_filler_words", "Detecting filler words...")
  );

  await JobPoller.start(
    "/fillers",
    { filepath: clipPath, custom_words: words.split(",").map(w => w.trim()), filler_backend: fillerBackend },
    (pct, msg) => { UIController.setProgress(pct); UIController.setProcessingMsg(msg || t("uxp.cut.runtime.transcribing", "Transcribing...")); },
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
      UIController.showToast(formatI18n("uxp.cut.runtime.filler_detected", "Detected {count} filler word(s).", { count }), "success");
      UIController.setStatus(formatI18n("uxp.cut.runtime.filler_done_status", "Filler detection done - {count} removed.", { count }));
      if (cuts.length) showCutResult({ ...result, cuts });
    },
    (err) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runFillerBtn", false);
      UIController.showToast(formatI18n("uxp.runtime.error_prefix", "Error: {error}", { error: err }), "error");
    }
  );
}

/** ── TRANSCRIBE ── */
async function runTranscribe() {
  const clipPath = document.getElementById("clipPathCaptions")?.value?.trim();
  if (!clipPath) { showSelectClipWarning(); return; }

  const model    = document.getElementById("whisperModel")?.value ?? "medium";
  const lang     = document.getElementById("transcribeLang")?.value ?? "auto";
  const style    = document.getElementById("captionStyle")?.value ?? DEFAULT_CAPTION_STYLE_ID;
  const diarize  = document.getElementById("enableDiarization")?.checked ?? false;
  const wordLevel = document.getElementById("enableWordLevel")?.checked ?? true;

  UIController.setButtonLoading("runTranscribeBtn", true);
  UIController.showProcessing(t("uxp.captions.runtime.transcribing_long", "Transcribing - this may take a while..."));
  setCaptionsSessionState(
    t("uxp.captions.runtime.working", "Working"),
    "working",
    t(
      "uxp.captions.runtime.transcribing_status",
      "Transcribing the selected clip. OpenCut will keep the last review output visible until this pass finishes."
    ),
    "working",
    clipPath
  );
  setTextAndTitle("captionsOutputValue", t("uxp.captions.runtime.processing_transcript", "Processing transcript..."), clipPath);
  syncCaptionsActionButtons();

  // The "captionStyle" select offers visual styles from /captions/styles.
  // Those style IDs have nothing to do with the /captions transcription format.
  // The backend's ``format`` param accepts only srt/vtt/json/ass and silently
  // coerces unknowns to srt — so the user's style choice was being lost.
  // Send a real format here; the visual style will be applied later via
  // /styled-captions if/when the user runs that flow.
  await JobPoller.start(
    "/captions",
    { filepath: clipPath, model, language: lang === "auto" ? null : lang,
      format: "srt", caption_style: style, diarize, word_timestamps: wordLevel },
    (pct, msg) => { UIController.setProgress(pct); UIController.setProcessingMsg(msg || t("uxp.cut.runtime.transcribing", "Transcribing...")); },
    (result) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runTranscribeBtn", false);
      showCaptionsResult(result);
      UIController.showToast(t("uxp.captions.runtime.transcription_complete", "Transcription complete."), "success");
      UIController.setStatus(t("uxp.captions.runtime.transcription_done_status", "Transcription done."));
    },
    (err) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runTranscribeBtn", false);
      UIController.showToast(formatI18n("uxp.captions.runtime.transcription_error", "Transcription error: {error}", { error: err }), "error");
      if (!_lastCaptionsResult) {
        setTextAndTitle(
          "captionsOutputValue",
          t("uxp.captions.no_transcript_yet", "No transcript yet"),
          t("uxp.captions.runtime.transcription_failed_retry_title", "Transcription failed. Retry when ready.")
        );
      }
      setCaptionsSessionState(
        t("uxp.captions.runtime.retry_needed", "Retry needed"),
        "warning",
        formatI18n("uxp.captions.runtime.transcription_failed_status", "Transcription failed. {error}", { error: err }),
        "error",
        clipPath
      );
      syncCaptionsActionButtons();
    }
  );
}

function showCaptionsResult(result) {
  const content = result.srt ?? result.text ?? JSON.stringify(result, null, 2);
  const nonEmptyLines = content.split(/\r?\n/).filter(Boolean).length;
  const clipPath = document.getElementById("clipPathCaptions")?.value?.trim() || "";
  const style = getSelectLabel("captionStyle", t("uxp.captions.runtime.selected_style", "Selected style"));
  const language = getSelectLabel("transcribeLang", t("uxp.captions.language_auto", "Auto-detect"));
  const lineUnit = nonEmptyLines === 1
    ? t("uxp.captions.runtime.line_singular", "line")
    : t("uxp.captions.runtime.line_plural", "lines");
  const captionLineSummary = formatI18n("uxp.captions.runtime.caption_lines_ready", "{count} caption {unit} ready", {
    count: nonEmptyLines,
    unit: lineUnit,
  });
  const transcriptLineSummary = formatI18n("uxp.captions.runtime.transcript_lines_ready", "{count} transcript {unit} ready", {
    count: nonEmptyLines,
    unit: lineUnit,
  });
  const captionLineLabel = formatI18n("uxp.captions.runtime.caption_lines", "{count} caption {unit}", {
    count: nonEmptyLines,
    unit: lineUnit,
  });
  const transcriptLineLabel = formatI18n("uxp.captions.runtime.transcript_lines", "{count} transcript {unit}", {
    count: nonEmptyLines,
    unit: lineUnit,
  });

  renderCaptionsResultView({
    kind: "transcript",
    header: result.srt
      ? t("uxp.captions.runtime.transcript_subtitle_output", "Transcript & Subtitle Output")
      : t("uxp.captions.runtime.transcript_review", "Transcript Review"),
    summary: result.srt ? captionLineSummary : transcriptLineSummary,
    content,
    resultPillText: result.srt
      ? t("uxp.captions.runtime.srt_ready", "SRT ready")
      : t("uxp.captions.runtime.transcript_ready", "Transcript ready"),
    resultPillState: "success",
    resultMeta: formatI18n("uxp.captions.runtime.transcript_result_meta", "{source} - {language} - {style}", {
      source: formatWorkspaceSource(clipPath),
      language,
      style,
    }),
    resultMetaTitle: clipPath || t("uxp.captions.runtime.transcript_ready", "Transcript ready"),
    copyLabel: result.srt
      ? t("uxp.captions.runtime.copy_srt", "Copy SRT")
      : t("uxp.captions.runtime.copy_transcript", "Copy Transcript"),
    copySuccessLabel: result.srt
      ? t("uxp.captions.runtime.srt_label", "SRT")
      : t("uxp.captions.transcript", "Transcript"),
    importLabel: t("uxp.captions.runtime.open_srt_import", "Open SRT Import"),
    canOpenSrtImport: !!result.srt,
    hasSrt: !!result.srt,
    sessionLabel: result.srt
      ? t("uxp.captions.runtime.transcript_ready", "Transcript ready")
      : t("uxp.captions.runtime.review_ready", "Review ready"),
    sessionState: "success",
    statusMessage: result.srt
      ? t(
          "uxp.captions.runtime.transcript_srt_ready_status",
          "Transcript ready. Copy the SRT or open Timeline > SRT Prep when you're ready to validate it for the CEP ocAddNativeCaptionTrack handoff."
        )
      : t(
          "uxp.captions.runtime.transcript_text_ready_status",
          "Transcript ready. Copy the text, draft chapters, or run a repeat review from the same clip."
        ),
    statusState: "success",
    statusTitle: clipPath || t("uxp.captions.runtime.transcript_ready", "Transcript ready"),
    outputLabel: result.srt ? captionLineLabel : transcriptLineLabel,
    outputTitle: clipPath || t("uxp.captions.runtime.transcript_ready", "Transcript ready"),
    insightType: result.srt
      ? t("uxp.captions.runtime.transcript_plus_srt", "Transcript + SRT")
      : t("uxp.captions.transcript", "Transcript"),
    insightLength: result.srt ? captionLineLabel : transcriptLineLabel,
    insightNext: result.srt
      ? t("uxp.captions.runtime.next_copy_srt", "Copy SRT or open SRT Prep")
      : t("uxp.captions.runtime.next_copy_transcript", "Copy transcript or draft chapters"),
  });
}

/** ── CHAPTER GENERATION ── */
async function runChapterGeneration() {
  const llmProvider = window._llmSettings?.provider || "ollama";
  const llmModel = window._llmSettings?.model || "llama3";
  const provider = document.getElementById("llmProvider")?.value ?? llmProvider;
  const model    = document.getElementById("llmModel")?.value ?? llmModel;
  const clipPath = document.getElementById("clipPathCaptions")?.value?.trim();
  if (!clipPath) {
    UIController.showToast(
      t("uxp.captions.runtime.transcribe_or_select_clip", "Please transcribe a clip first, or select one."),
      "warning"
    );
    return;
  }

  UIController.setButtonLoading("runChaptersBtn", true);
  UIController.showProcessing(t("uxp.captions.runtime.generating_chapters", "Generating chapters with AI..."));
  setCaptionsSessionState(
    t("uxp.captions.runtime.working", "Working"),
    "working",
    t(
      "uxp.captions.runtime.drafting_chapters_status",
      "Drafting chapter markers from the selected clip. The last review output will stay available until the new pass is ready."
    ),
    "working",
    clipPath
  );
  setTextAndTitle("captionsOutputValue", t("uxp.captions.runtime.drafting_chapters", "Drafting chapters..."), clipPath);
  syncCaptionsActionButtons();

  await JobPoller.start(
    "/captions/chapters",
    { filepath: clipPath, llm_provider: provider, llm_model: model },
    (pct, msg) => { UIController.setProgress(pct); UIController.setProcessingMsg(msg || t("uxp.captions.runtime.generating", "Generating...")); },
    (result) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runChaptersBtn", false);
      const count = result.chapters?.length ?? 0;
      UIController.showToast(
        formatI18n("uxp.captions.runtime.generated_chapters", "Generated {count} chapter(s).", { count }),
        "success"
      );
      UIController.setStatus(
        formatI18n("uxp.captions.runtime.chapter_generation_done_status", "Chapter generation complete - {count} chapters.", {
          count,
        })
      );
      const clipTitle = clipPath || t("uxp.captions.runtime.chapter_output", "Chapter output");
      const providerLabel = getSelectLabel("llmProvider", t("uxp.captions.runtime.selected_provider", "Selected provider"));
      const chapterContent = count
        ? result.chapters.map((c, i) =>
            formatI18n("uxp.captions.runtime.chapter_line", "{time} - {title}", {
              time: formatTimecode(c.seconds ?? c.start ?? 0),
              title: c.title ?? formatI18n("uxp.captions.runtime.chapter_index", "Chapter {index}", { index: i + 1 }),
            })
          ).join("\n")
        : t(
            "uxp.captions.runtime.no_chapters_suggested",
            "No chapters were suggested for the current clip with the selected model."
          );
      renderCaptionsResultView({
        kind: "chapters",
        header: t("uxp.captions.runtime.chapter_draft_header", "Chapter Draft"),
        summary: count
          ? formatI18n("uxp.captions.runtime.chapters_ready", "{count} chapter{plural} ready", {
              count,
              plural: count === 1 ? "" : "s",
            })
          : t("uxp.captions.runtime.no_chapters_drafted", "No chapters drafted"),
        content: chapterContent,
        resultPillText: count
          ? t("uxp.captions.runtime.chapters", "Chapters")
          : t("uxp.captions.runtime.needs_review", "Needs review"),
        resultPillState: count ? "success" : "warning",
        resultMeta: formatI18n("uxp.captions.runtime.chapter_result_meta", "{source} - {provider} - {model}", {
          source: formatWorkspaceSource(clipPath),
          provider: providerLabel,
          model,
        }),
        resultMetaTitle: clipTitle,
        copyLabel: t("uxp.captions.runtime.copy_chapters", "Copy Chapters"),
        copySuccessLabel: t("uxp.captions.runtime.chapters_label", "Chapters"),
        importLabel: t("uxp.captions.runtime.open_srt_import", "Open SRT Import"),
        canOpenSrtImport: false,
        hasSrt: false,
        sessionLabel: count
          ? t("uxp.captions.runtime.chapters_ready_label", "Chapters ready")
          : t("uxp.captions.runtime.needs_review", "Needs review"),
        sessionState: count ? "success" : "warning",
        statusMessage: count
          ? t(
              "uxp.captions.runtime.chapter_draft_ready_status",
              "Chapter draft is ready. Copy the list into publishing notes or rerun with a different model for a tighter structure."
            )
          : t(
              "uxp.captions.runtime.no_chapters_status",
              "No chapters were suggested. Try another model or confirm the transcript has enough structure to segment cleanly."
            ),
        statusState: count ? "success" : "warning",
        statusTitle: clipTitle,
        outputLabel: count
          ? formatI18n("uxp.captions.runtime.chapter_count", "{count} chapter{plural}", {
              count,
              plural: count === 1 ? "" : "s",
            })
          : t("uxp.captions.runtime.no_chapters_drafted", "No chapters drafted"),
        outputTitle: clipTitle,
        insightType: t("uxp.captions.runtime.chapter_draft", "Chapter draft"),
        insightLength: count
          ? formatI18n("uxp.captions.runtime.chapter_count", "{count} chapter{plural}", {
              count,
              plural: count === 1 ? "" : "s",
            })
          : t("uxp.captions.runtime.no_chapters_drafted", "No chapters drafted"),
        insightNext: count
          ? t("uxp.captions.runtime.next_copy_chapters", "Copy into publishing notes")
          : t("uxp.captions.runtime.next_try_model", "Try another model or refine the transcript"),
      });
    },
    (err) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runChaptersBtn", false);
      UIController.showToast(formatI18n("uxp.captions.runtime.chapter_generation_error", "Chapter generation error: {error}", { error: err }), "error");
      if (!_lastCaptionsResult) {
        setTextAndTitle(
          "captionsOutputValue",
          t("uxp.captions.runtime.no_chapter_draft", "No chapter draft"),
          t("uxp.captions.runtime.chapter_generation_failed_retry_title", "Chapter generation failed. Retry when ready.")
        );
      }
      setCaptionsSessionState(
        t("uxp.captions.runtime.retry_needed", "Retry needed"),
        "warning",
        formatI18n("uxp.captions.runtime.chapter_generation_failed_status", "Chapter generation failed. {error}", { error: err }),
        "error",
        clipPath
      );
      syncCaptionsActionButtons();
    }
  );
}

/** ── REPEAT DETECTION ── */
async function runRepeatDetection() {
  const clipPath = document.getElementById("clipPathCaptions")?.value?.trim();
  if (!clipPath) { showSelectClipWarning(); return; }

  const threshold = parseFloat(document.getElementById("repeatSimilarity")?.value ?? 0.85);
  const keepBest  = document.getElementById("keepBestRepeat")?.checked ?? true;

  UIController.setButtonLoading("runRepeatBtn", true);
  UIController.showProcessing(t("uxp.captions.runtime.detecting_repeated_segments", "Detecting repeated segments..."));
  setCaptionsSessionState(
    t("uxp.captions.runtime.working", "Working"),
    "working",
    t("uxp.captions.runtime.checking_repeats_status", "Checking the current clip for duplicated spoken lines and alternate takes."),
    "working",
    clipPath
  );
  setTextAndTitle("captionsOutputValue", t("uxp.captions.runtime.scanning_repeats", "Scanning for repeats..."), clipPath);
  syncCaptionsActionButtons();

    await JobPoller.start(
    "/captions/repeat-detect",
    { filepath: clipPath, threshold, keep_best: keepBest },
    (pct, msg) => { UIController.setProgress(pct); UIController.setProcessingMsg(msg || t("uxp.captions.runtime.analyzing", "Analyzing...")); },
    (result) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runRepeatBtn", false);
      const count = result.repeats?.length ?? result.removed_count ?? 0;
      UIController.showToast(
        formatI18n("uxp.captions.runtime.detected_repeats", "Detected {count} repeat(s).", { count }),
        "success"
      );
      UIController.setStatus(
        formatI18n("uxp.captions.runtime.repeat_detection_done_status", "Repeat detection done - {count} found.", { count })
      );
      showRepeatResult(result);
    },
    (err) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runRepeatBtn", false);
      UIController.showToast(formatI18n("uxp.runtime.error_prefix", "Error: {error}", { error: err }), "error");
      if (!_lastCaptionsResult) {
        setTextAndTitle(
          "captionsOutputValue",
          t("uxp.captions.runtime.no_repeat_review", "No repeat review"),
          t("uxp.captions.runtime.repeat_detection_failed_retry_title", "Repeat detection failed. Retry when ready.")
        );
      }
      setCaptionsSessionState(
        t("uxp.captions.runtime.retry_needed", "Retry needed"),
        "warning",
        formatI18n("uxp.captions.runtime.repeat_detection_failed_status", "Repeat detection failed. {error}", { error: err }),
        "error",
        clipPath
      );
      syncCaptionsActionButtons();
    }
  );
}

/** ── DENOISE ── */
async function runDenoise() {
  const clipPath = document.getElementById("clipPathAudio")?.value?.trim();
  if (!clipPath) { showSelectClipWarning(); return; }

  const method   = document.getElementById("denoiseMethod")?.value ?? "noisereduce";
  const strength = parseInt(document.getElementById("denoiseStrength")?.value ?? 75) / 100;

  UIController.setButtonLoading("runDenoiseBtn", true);
  UIController.showProcessing(t("uxp.audio.runtime.applying_noise_reduction", "Applying noise reduction..."));

  await JobPoller.start(
    "/audio/denoise",
    { filepath: clipPath, method, strength },
    (pct, msg) => { UIController.setProgress(pct); UIController.setProcessingMsg(msg || t("processing.processing", "Processing...")); },
    (result) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runDenoiseBtn", false);
      const output = result.output ?? result.output_path ?? t("uxp.runtime.saved", "saved");
      UIController.showToast(formatI18n("uxp.audio.runtime.denoise_complete_output", "Denoise complete. Output: {output}", { output }), "success");
      UIController.setStatus(t("uxp.audio.runtime.denoise_complete_status", "Denoise complete."));
    },
    (err) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runDenoiseBtn", false);
      UIController.showToast(formatI18n("uxp.audio.runtime.denoise_error", "Denoise error: {error}", { error: err }), "error");
    }
  );
}

/** ── NORMALIZE ── */
async function runNormalize() {
  const clipPath = document.getElementById("clipPathAudio")?.value?.trim();
  if (!clipPath) { showSelectClipWarning(); return; }

  const targetLufs = parseFloat(document.getElementById("targetLufs")?.value ?? -14);
  const truePeak   = document.getElementById("normalizeTruePeak")?.checked ?? true;

  UIController.setButtonLoading("runNormalizeBtn", true);
  UIController.showProcessing(t("uxp.audio.runtime.normalizing_audio", "Normalizing audio..."));

  await JobPoller.start(
    "/audio/normalize",
    { filepath: clipPath, target_lufs: targetLufs, true_peak: truePeak ? -1.0 : null },
    (pct, msg) => { UIController.setProgress(pct); UIController.setProcessingMsg(msg || t("uxp.audio.runtime.normalizing", "Normalizing...")); },
    (result) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runNormalizeBtn", false);
      const output = result.output ?? result.output_path ?? t("uxp.runtime.saved", "saved");
      UIController.showToast(formatI18n("uxp.audio.runtime.normalization_complete_output", "Normalization complete. Output: {output}", { output }), "success");
      UIController.setStatus(t("uxp.audio.runtime.normalization_complete_status", "Normalization complete."));
    },
    (err) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runNormalizeBtn", false);
      UIController.showToast(formatI18n("uxp.audio.runtime.normalization_error", "Normalization error: {error}", { error: err }), "error");
    }
  );
}

/** ── LOUDNESS MATCH ── */
async function runLoudnessMatch() {
  const clipPath = document.getElementById("clipPathAudio")?.value?.trim();
  const refPath  = document.getElementById("refClipLoudness")?.value?.trim();
  if (!clipPath || !refPath) {
    UIController.showToast(t("uxp.audio.runtime.select_input_reference", "Please select both input and reference clips."), "warning");
    return;
  }

  UIController.setButtonLoading("runLoudnessBtn", true);
  UIController.showProcessing(t("uxp.audio.runtime.matching_loudness_reference", "Matching loudness to reference..."));

  await JobPoller.start(
    "/audio/loudness-match",
    { files: [clipPath, refPath], target_lufs: -14.0 },
    (pct, msg) => { UIController.setProgress(pct); UIController.setProcessingMsg(msg || t("uxp.audio.runtime.matching", "Matching...")); },
    (result) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runLoudnessBtn", false);
      UIController.showToast(t("uxp.audio.runtime.loudness_match_complete", "Loudness match complete."), "success");
      UIController.setStatus(t("uxp.audio.runtime.loudness_match_done", "Loudness match done."));
    },
    (err) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runLoudnessBtn", false);
      UIController.showToast(formatI18n("uxp.runtime.error_prefix", "Error: {error}", { error: err }), "error");
    }
  );
}

/** ── BEAT MARKERS ── */
async function runBeatMarkers() {
  const trackPath = document.getElementById("beatTrackPath")?.value?.trim();
  if (!trackPath) { UIController.showToast(t("uxp.audio.runtime.select_audio_file", "Please select an audio/music file."), "warning"); return; }

  const sensitivity = parseInt(document.getElementById("beatSensitivity")?.value ?? 60) / 100;

  UIController.setButtonLoading("runBeatMarkersBtn", true);
  UIController.showProcessing(t("uxp.audio.runtime.detecting_beats", "Detecting beats..."));

  await JobPoller.start(
    "/audio/beat-markers",
    { filepath: trackPath, subdivisions: Math.max(1, Math.round(sensitivity * 4)) },
    (pct, msg) => { UIController.setProgress(pct); UIController.setProcessingMsg(msg || t("uxp.audio.runtime.analyzing_tempo", "Analyzing tempo...")); },
    async (result) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runBeatMarkersBtn", false);

      const beats = result.beats ?? result.markers ?? [];
      rememberTimelineMarkers(beats, { source: t("uxp.audio.beat_detection", "Beat Detection"), clipPath: trackPath });
      UIController.showToast(formatI18n("uxp.audio.runtime.beats_detected_add_markers", "Detected {count} beats. Adding markers to timeline...", { count: beats.length }), "success");
      UIController.setStatus(formatI18n("uxp.audio.runtime.beat_detection_done", "Beat detection done - {count} beats.", { count: beats.length }));

      // Attempt direct UXP marker insertion
      await addSequenceMarkers(beats, "green");
    },
    (err) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runBeatMarkersBtn", false);
      UIController.showToast(formatI18n("uxp.audio.runtime.beat_detection_error", "Beat detection error: {error}", { error: err }), "error");
    }
  );
}

/** ── COLOR MATCH ── */
async function runColorMatch() {
  const clipPath = document.getElementById("clipPathVideo")?.value?.trim();
  const refPath  = document.getElementById("colorRefPath")?.value?.trim();
  if (!clipPath || !refPath) {
    UIController.showToast(
      t("uxp.video.runtime.select_input_reference", "Please select both input and reference clips."),
      "warning"
    );
    return;
  }

  const strength = parseInt(document.getElementById("colorMatchStrength")?.value ?? 80) / 100;

  UIController.setButtonLoading("runColorMatchBtn", true);
  UIController.showProcessing(t("uxp.video.runtime.matching_color_grading", "Matching color grading..."));

  await JobPoller.start(
    "/video/color-match",
    { source: clipPath, reference: refPath, strength },
    (pct, msg) => { UIController.setProgress(pct); UIController.setProcessingMsg(msg || t("uxp.video.runtime.grading", "Grading...")); },
    (result) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runColorMatchBtn", false);
      const output = result.output ?? result.output_path ?? t("uxp.runtime.saved", "saved");
      UIController.showToast(formatI18n("uxp.video.runtime.color_match_complete_output", "Color match complete. Output: {output}", { output }), "success");
      UIController.setStatus(t("uxp.video.runtime.color_match_complete_status", "Color match complete."));
    },
    (err) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runColorMatchBtn", false);
      UIController.showToast(formatI18n("uxp.video.runtime.color_match_error", "Color match error: {error}", { error: err }), "error");
    }
  );
}

/** ── AUTO ZOOM ── */
async function runAutoZoom() {
  const clipPath = document.getElementById("clipPathVideo")?.value?.trim();
  if (!clipPath) { showSelectClipWarning(); return; }

  const aspect    = document.getElementById("zoomAspect")?.value ?? "9:16";
  const maxZoom   = parseFloat(document.getElementById("zoomFactor")?.value ?? 1.4);

  UIController.setButtonLoading("runAutoZoomBtn", true);
  UIController.showProcessing(t("uxp.video.runtime.applying_auto_zoom_reframe", "Applying auto zoom / reframe..."));

  await JobPoller.start(
    "/video/auto-zoom",
    { filepath: clipPath, zoom_amount: maxZoom, easing: "ease_in_out" },
    (pct, msg) => { UIController.setProgress(pct); UIController.setProcessingMsg(msg || t("uxp.video.runtime.reframing", "Reframing...")); },
    (result) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runAutoZoomBtn", false);
      const output = result.output ?? result.output_path ?? t("uxp.runtime.saved", "saved");
      UIController.showToast(formatI18n("uxp.video.runtime.auto_zoom_complete_output", "Auto zoom complete. Output: {output}", { output }), "success");
      UIController.setStatus(t("uxp.video.runtime.auto_zoom_complete_status", "Auto zoom complete."));
    },
    (err) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runAutoZoomBtn", false);
      UIController.showToast(formatI18n("uxp.video.runtime.auto_zoom_error", "Auto zoom error: {error}", { error: err }), "error");
    }
  );
}

/** ── MULTICAM CUTS ── */
async function runMulticamCuts() {
  const cam1Path = document.getElementById("clipPathVideo")?.value?.trim();
  const cam2Path = document.getElementById("cam2Path")?.value?.trim();
  if (!cam1Path || !cam2Path) {
    UIController.showToast(t("uxp.video.runtime.select_both_camera_files", "Please select both camera files."), "warning");
    return;
  }

  const strategy = document.getElementById("multicamStrategy")?.value ?? "activity";

  UIController.setButtonLoading("runMulticamBtn", true);
  UIController.showProcessing(t("uxp.video.runtime.generating_multicam_cuts", "Generating multicam cuts..."));

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
    (pct, msg) => { UIController.setProgress(pct); UIController.setProcessingMsg(msg || t("uxp.video.runtime.analyzing_cameras", "Analyzing cameras...")); },
    async (result) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runMulticamBtn", false);
      const cuts = result.cuts ?? [];
      rememberTimelineCuts(cuts, { source: t("uxp.video.runtime.multicam_cut_pass", "Multicam Cut Pass"), clipPath: cam1Path });
      UIController.showToast(
        formatI18n("uxp.video.runtime.multicam_cuts_generated", "Generated {count} multicam cut point(s).", {
          count: cuts.length,
        }),
        "success"
      );
      UIController.setStatus(
        formatI18n("uxp.video.runtime.multicam_cuts_ready_status", "Multicam cuts ready - {count} cuts.", {
          count: cuts.length,
        })
      );
      // Attempt to apply directly to timeline
      await applyTimelineCuts(cuts);
    },
    (err) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runMulticamBtn", false);
      UIController.showToast(formatI18n("uxp.video.runtime.multicam_error", "Multicam error: {error}", { error: err }), "error");
    }
  );
}

/** ── APPLY TIMELINE CUTS (UXP + fallback) ── */
async function applyTimelineCuts(cuts) {
  const cutsToApply = cuts ?? lastCuts;
  if (!cutsToApply || cutsToApply.length === 0) {
    UIController.showToast(t("uxp.timeline.runtime.no_cuts_to_apply", "No cuts to apply. Run silence removal or filler detection first."), "warning");
    noteTimelineAction(
      t("uxp.timeline.runtime.cuts_unavailable_title", "Cuts unavailable"),
      "warning",
      t("uxp.timeline.runtime.no_cuts_staged_detail", "No cuts are staged for sequence write-back yet. Run silence, filler, or multicam cleanup first."),
      t("uxp.timeline.runtime.no_cuts_staged_short", "No cuts are staged for sequence write-back yet.")
    );
    return;
  }

  if (PProBridge.available()) {
  UIController.setStatus(t("uxp.timeline.runtime.applying_cuts_uxp", "Applying cuts to timeline via UXP..."));
    const result = await PProBridge.applyCuts(cutsToApply);
    if (result.ok) {
      UIController.showToast(formatI18n("uxp.timeline.runtime.applied_cuts_sequence", "Applied {count} cut(s) to active sequence.", { count: result.applied }), "success");
      UIController.setStatus(formatI18n("uxp.timeline.runtime.applied_cuts_status", "Applied {count} cut(s).", { count: result.applied }));
      noteTimelineAction(
        t("uxp.timeline.runtime.cuts_applied_title", "Cuts applied"),
        "success",
        formatI18n("uxp.timeline.runtime.applied_cuts_detail", "Applied {count} cut(s) to the active sequence.", { count: result.applied }),
        formatI18n("uxp.timeline.runtime.applied_cuts_detail", "Applied {count} cut(s) to the active sequence.", { count: result.applied }),
        formatI18n("uxp.timeline.runtime.cuts_applied_short", "{count} cut(s) applied", { count: result.applied })
      );
    } else {
      UIController.showToast(
        formatI18n("uxp.timeline.runtime.timeline_write_failed", "UXP timeline write failed: {reason}. Use CEP panel for Premiere < 25.6.", { reason: result.reason || t("common.unknown", "unknown") }),
        "warning"
      );
      UIController.setStatus(t("uxp.timeline.runtime.timeline_write_failed_status", "Timeline write failed - see CEP panel."));
      noteTimelineAction(
        t("uxp.timeline.runtime.cep_fallback_needed_title", "CEP fallback needed"),
        "warning",
        formatI18n("uxp.timeline.runtime.direct_write_failed_detail", "Direct UXP write-back failed. {reason}. Use the CEP panel for this sequence pass.", { reason: result.reason || t("common.unknown", "unknown") }),
        result.reason || t("uxp.timeline.runtime.direct_write_failed_short", "Direct UXP write-back failed.")
      );
    }
  } else {
    UIController.showToast(
      t("uxp.timeline.runtime.timeline_api_preview_toast", "Connect via CEP panel for timeline operations (UXP timeline API in preview)."),
      "info"
    );
    UIController.setStatus(t("uxp.timeline.runtime.timeline_api_unavailable_status", "UXP timeline API unavailable - use CEP panel."));
    noteTimelineAction(
      t("uxp.timeline.runtime.cep_fallback_needed_title", "CEP fallback needed"),
      "warning",
      t("uxp.timeline.runtime.direct_write_unavailable_detail", "Direct sequence write-back is not available in this UXP session. Use the CEP panel for timeline operations."),
      t("uxp.timeline.runtime.direct_write_unavailable_short", "Direct sequence write-back is not available in this UXP session.")
    );
  }
}

/** ── ADD SEQUENCE MARKERS (UXP + fallback) ── */
async function addSequenceMarkers(markers, color) {
  const markersToAdd = markers ?? lastMarkers;
  if (!markersToAdd || markersToAdd.length === 0) {
    UIController.showToast(t("uxp.timeline.runtime.no_markers_to_add", "No markers to add. Run beat detection first."), "warning");
    noteTimelineAction(
      t("uxp.timeline.runtime.markers_unavailable_title", "Markers unavailable"),
      "warning",
      t("uxp.timeline.runtime.no_markers_staged_detail", "No markers are staged for sequence write-back yet. Run beat detection first."),
      t("uxp.timeline.runtime.no_markers_staged_short", "No markers are staged for sequence write-back yet.")
    );
    return;
  }

  const markerColor = color ?? document.getElementById("beatMarkerColor")?.value ?? "green";
  const formatted = markersToAdd.map(m => ({
    time:  typeof m === "number" ? m : (m.time ?? m.t ?? 0),
    label: m.label ?? t("uxp.timeline.runtime.beat_marker_label", "Beat"),
    color: markerColor,
  }));

  if (PProBridge.available()) {
  UIController.setStatus(t("uxp.timeline.runtime.adding_markers_uxp", "Adding markers to sequence via UXP..."));
    const result = await PProBridge.addMarkers(formatted);
    if (result.ok) {
      UIController.showToast(formatI18n("uxp.timeline.runtime.added_markers_sequence", "Added {count} marker(s) to active sequence.", { count: result.count }), "success");
      UIController.setStatus(formatI18n("uxp.timeline.runtime.added_markers_status", "Added {count} marker(s).", { count: result.count }));
      noteTimelineAction(
        t("uxp.timeline.runtime.markers_added_title", "Markers added"),
        "success",
        formatI18n("uxp.timeline.runtime.added_markers_detail", "Added {count} marker(s) to the active sequence.", { count: result.count }),
        formatI18n("uxp.timeline.runtime.added_markers_detail", "Added {count} marker(s) to the active sequence.", { count: result.count }),
        formatI18n("uxp.timeline.runtime.markers_added_short", "{count} marker(s) added", { count: result.count })
      );
    } else {
      UIController.showToast(
        formatI18n("uxp.timeline.runtime.marker_insertion_failed", "UXP marker insertion failed: {reason}. Use CEP panel as fallback.", { reason: result.reason || t("common.unknown", "unknown") }),
        "warning"
      );
      noteTimelineAction(
        t("uxp.timeline.runtime.cep_fallback_needed_title", "CEP fallback needed"),
        "warning",
        formatI18n("uxp.timeline.runtime.marker_insertion_failed_detail", "Marker insertion failed in UXP. {reason}. Use the CEP panel as fallback.", { reason: result.reason || t("common.unknown", "unknown") }),
        result.reason || t("uxp.timeline.runtime.marker_insertion_failed_short", "Marker insertion failed in UXP.")
      );
    }
  } else {
    UIController.showToast(
      t("uxp.timeline.runtime.timeline_api_preview_toast", "Connect via CEP panel for timeline operations (UXP timeline API in preview)."),
      "info"
    );
    noteTimelineAction(
      t("uxp.timeline.runtime.cep_fallback_needed_title", "CEP fallback needed"),
      "warning",
      t("uxp.timeline.runtime.marker_write_unavailable_detail", "Marker insertion is not available in this UXP session. Use the CEP panel for timeline operations."),
      t("uxp.timeline.runtime.marker_write_unavailable_short", "Marker insertion is not available in this UXP session.")
    );
  }
}

/** ── BATCH EXPORT ── */
async function runBatchExport() {
  const preset    = document.getElementById("exportPreset")?.value ?? "youtube";
  const outputDir = document.getElementById("exportDir")?.value?.trim();
  if (!outputDir) { UIController.showToast(t("uxp.timeline.runtime.select_output_folder", "Please select an output folder."), "warning"); return; }

  const clipPath = document.getElementById("clipPathVideo")?.value?.trim() ?? "";
  const markersToExport = buildExportWindows();
  if (!clipPath) { showSelectClipWarning(); return; }
  if (markersToExport.length === 0) {
    UIController.showToast(t("uxp.timeline.runtime.no_markers_or_cuts_to_export", "No markers or cuts to export. Run beat detection or silence removal first."), "warning");
    return;
  }

  UIController.setButtonLoading("runBatchExportBtn", true);
  UIController.showProcessing(t("uxp.timeline.runtime.starting_batch_export", "Starting batch export from markers..."));

  await JobPoller.start(
    "/timeline/export-from-markers",
    { input_file: clipPath, markers: markersToExport, output_dir: outputDir, format: preset },
    (pct, msg) => { UIController.setProgress(pct); UIController.setProcessingMsg(msg || t("uxp.timeline.runtime.exporting", "Exporting...")); },
    (result) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runBatchExportBtn", false);
      const count = result.exported ?? result.count ?? 0;
      UIController.showToast(formatI18n("uxp.timeline.runtime.exported_segments", "Exported {count} segment(s).", { count }), "success");
      UIController.setStatus(formatI18n("uxp.timeline.runtime.batch_export_done_status", "Batch export done - {count} files.", { count }));
      noteTimelineAction(
        t("uxp.timeline.runtime.batch_export_complete_title", "Batch export complete"),
        "success",
        formatI18n("uxp.timeline.runtime.marker_export_complete_detail", "Marker-based export finished with {count} segment(s) in {outputDir}.", { count, outputDir }),
        outputDir,
        formatI18n("uxp.timeline.runtime.exports_ready_short", "{count} export(s) ready", { count })
      );
    },
    (err) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runBatchExportBtn", false);
      UIController.showToast(formatI18n("uxp.timeline.runtime.export_error", "Export error: {error}", { error: err }), "error");
      noteTimelineAction(
        t("uxp.timeline.runtime.batch_export_error_title", "Batch export error"),
        "error",
        formatI18n("uxp.timeline.runtime.marker_export_failed_detail", "Marker-based export failed. {error}", { error: err }),
        err
      );
    }
  );
}

/** ── BATCH RENAME ── */
async function runBatchRename() {
  const pattern = document.getElementById("renamePattern")?.value?.trim() ?? "{name}_{index:03d}";
  UIController.showToast(t("uxp.timeline.runtime.batch_rename_cep", "Batch rename still runs through the CEP panel in this build."), "info");
  noteTimelineAction(
    t("uxp.timeline.runtime.rename_via_cep_title", "Rename via CEP"),
    "warning",
    t("uxp.timeline.runtime.rename_via_cep_detail", "Batch rename is planned from this workspace, but execution still lives in the CEP panel today."),
    pattern,
    t("uxp.timeline.runtime.rename_handoff", "Rename handoff")
  );
}

/** ── SMART BINS ── */
async function runSmartBins() {
  const strategy = getSelectLabel("binStrategy", "File Type");
  UIController.showToast(t("uxp.timeline.runtime.smart_bins_cep", "Smart bins still execute through the CEP panel in this build."), "info");
  noteTimelineAction(
    t("uxp.timeline.runtime.smart_bins_via_cep_title", "Smart bins via CEP"),
    "warning",
    t("uxp.timeline.runtime.smart_bins_via_cep_detail", "Smart bin rules can be planned here, but execution still lives in the CEP panel today."),
    strategy,
    t("uxp.timeline.runtime.smart_bin_handoff", "Smart bin handoff")
  );
}

/** ── SRT IMPORT ── */
async function runSrtImport() {
  const srtPath    = document.getElementById("srtFilePath")?.value?.trim();
  const trackIndex = parseInt(document.getElementById("srtTrackIndex")?.value ?? 1);
  if (!srtPath) { UIController.showToast(t("uxp.timeline.runtime.select_srt_file", "Please select an SRT file."), "warning"); return; }

  UIController.setButtonLoading("runSrtImportBtn", true);
  UIController.showProcessing(t("uxp.timeline.runtime.validating_srt_import", "Validating SRT for timeline import..."));

  await JobPoller.start(
    "/timeline/srt-to-captions",
    { srt_path: srtPath, track_index: trackIndex },
    (pct, msg) => { UIController.setProgress(pct); UIController.setProcessingMsg(msg || t("uxp.timeline.runtime.validating", "Validating...")); },
    (result) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runSrtImportBtn", false);
      const count = result.segments?.length ?? result.captions_imported ?? result.count ?? 0;
      UIController.showToast(formatI18n("uxp.timeline.runtime.srt_validated_segments", "Validated {count} caption segment(s).", { count }), "success");
      UIController.setStatus(formatI18n("uxp.timeline.runtime.srt_ready_status", "SRT ready - {count} caption segments parsed.", { count }));
      noteTimelineAction(
        t("uxp.timeline.runtime.srt_validated_title", "SRT validated"),
        "success",
        formatI18n("uxp.timeline.runtime.srt_validation_ready_detail", "SRT validation is ready. Use the CEP ocAddNativeCaptionTrack bridge action to place {count} caption segment(s) on track {trackIndex}.", { count, trackIndex }),
        srtPath,
        formatI18n("uxp.timeline.runtime.caption_segments_parsed", "{count} caption segment(s) parsed", { count })
      );
    },
    (err) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runSrtImportBtn", false);
      UIController.showToast(formatI18n("uxp.timeline.runtime.srt_validation_error", "SRT validation error: {error}", { error: err }), "error");
      noteTimelineAction(
        t("uxp.timeline.runtime.srt_validation_error_title", "SRT validation error"),
        "error",
        formatI18n("uxp.timeline.runtime.srt_validation_failed_detail", "SRT validation failed. {error}", { error: err }),
        err
      );
    }
  );
}

/** ── INDEX LIBRARY ── */
async function runIndexLibrary() {
  const folder       = document.getElementById("indexFolder")?.value?.trim();
  if (!folder) {
    setStatusPill(
      "indexStatePill",
      t("uxp.search.runtime.needs_folder", "Needs Folder"),
      "warning",
      t("uxp.search.runtime.choose_folder_before_building", "Choose a media folder before building the search index.")
    );
    setIndexStatus(t("uxp.search.runtime.choose_folder_before_building", "Choose a media folder before building the search index."), "warning");
    UIController.showToast(t("uxp.search.runtime.select_folder_to_index", "Please select a media folder to index."), "warning");
    syncSearchPanelState();
    return;
  }

  const statusLine = document.getElementById("indexStatus");
  UIController.setButtonLoading("runIndexLibBtn", true);
  UIController.showProcessing(t("uxp.search.runtime.indexing_media_library", "Indexing media library..."));
  setStatusPill("indexStatePill", t("uxp.search.runtime.indexing", "Indexing"), "working", folder);
  setIndexStatus(t("uxp.search.runtime.indexing_media_library_status", "Indexing the media library..."), "working", folder);
  if (statusLine) statusLine.textContent = t("uxp.search.runtime.indexing_media_library_status", "Indexing the media library...");

  await JobPoller.start(
    "/search/index",
    { folder, model: "base" },
    (pct, msg) => {
      UIController.setProgress(pct);
      const progressMsg = msg || t("uxp.search.runtime.scanning", "Scanning...");
      UIController.setProcessingMsg(progressMsg);
      if (statusLine) statusLine.textContent = progressMsg;
    },
    async (result) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runIndexLibBtn", false);
      const count = result.indexed ?? result.files ?? 0;
      const errorCount = Array.isArray(result.errors) ? result.errors.length : 0;
      const pillState = errorCount ? "warning" : (count > 0 ? "success" : "empty");
      const pillLabel = errorCount
        ? t("uxp.search.runtime.needs_review", "Needs Review")
        : (count > 0 ? t("uxp.search.ready", "Ready") : t("uxp.search.runtime.empty", "Empty"));
      await refreshFootageIndexStats({ preserveMessage: true, silent: true });
      setStatusPill(
        "indexStatePill",
        pillLabel,
        pillState,
        formatCountI18n(
          count,
          "uxp.search.runtime.files_indexed_sentence_one",
          "{count} file indexed.",
          "uxp.search.runtime.files_indexed_sentence_many",
          "{count} files indexed."
        )
      );
      setIndexStatus(
        errorCount
          ? formatI18n("uxp.search.runtime.library_indexed_with_skips", "Library indexed with a few skips. {count} file(s) are ready to search, and {errorCount} item(s) need attention.", { count, errorCount })
          : formatCountI18n(
              count,
              "uxp.search.runtime.library_indexed_one",
              "Library indexed. {count} file is ready to search.",
              "uxp.search.runtime.library_indexed_many",
              "Library indexed. {count} files are ready to search."
            ),
        errorCount ? "warning" : "success"
      );
      const indexedToast = formatCountI18n(
        count,
        "uxp.search.runtime.library_indexed_toast_one",
        "Library indexed - {count} file.",
        "uxp.search.runtime.library_indexed_toast_many",
        "Library indexed - {count} files."
      );
      UIController.showToast(indexedToast, "success");
      UIController.setStatus(indexedToast, "success");
      setTextAndTitle(
        "searchStatus",
        count > 0
          ? t("uxp.search.runtime.library_ready_search_prompt", "The library is ready. Search with descriptive phrases, then load the best match into the workspace.")
          : t("uxp.search.runtime.index_another_folder_prompt", "Index another folder or broaden the source media to make search more useful."),
        folder
      );
    },
    (err) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runIndexLibBtn", false);
      setStatusPill("indexStatePill", t("uxp.search.runtime.error", "Error"), "error", err);
      const errorText = formatI18n("uxp.search.runtime.index_library_failed", "Could not index the library. {error}", { error: err });
      setIndexStatus(errorText, "error");
      if (statusLine) statusLine.textContent = errorText;
      UIController.showToast(formatI18n("uxp.search.runtime.index_error", "Index error: {error}", { error: err }), "error");
      syncSearchPanelState();
    }
  );
}

/** ── FOOTAGE SEARCH ── */
async function runFootageSearch() {
  const query = document.getElementById("searchQuery")?.value?.trim();
  const limit = parseInt(document.getElementById("searchResultCount")?.value ?? 10);
  const backendOnline = isBackendConnected();
  const indexedFiles = Number(_lastIndexStats?.total_files || 0);
  if (!backendOnline) {
    setTextAndTitle(
      "searchStatus",
      t("uxp.search.runtime.reconnect_before_searching", "Reconnect the backend before searching the library."),
      t("uxp.search.runtime.reconnect_before_searching", "Reconnect the backend before searching the library.")
    );
    UIController.showToast(t("uxp.search.runtime.reconnect_before_searching", "Reconnect the backend before searching the library."), "warning");
    syncSearchPanelState();
    return;
  }
  if (indexedFiles <= 0) {
    setTextAndTitle(
      "searchStatus",
      t("uxp.search.runtime.index_folder_before_searching", "Index a folder before searching the library."),
      t("uxp.search.runtime.index_folder_before_searching", "Index a folder before searching the library.")
    );
    UIController.showToast(t("uxp.search.runtime.index_folder_before_searching", "Index a folder before searching the library."), "warning");
    syncSearchPanelState();
    return;
  }
  if (!query) {
    setTextAndTitle(
      "searchStatus",
      t("uxp.search.runtime.enter_descriptive_query", "Enter a descriptive query to search the indexed library."),
      t("uxp.search.runtime.enter_descriptive_query", "Enter a descriptive query to search the indexed library.")
    );
    UIController.showToast(t("uxp.search.runtime.enter_search_query", "Please enter a search query."), "warning");
    syncSearchPanelState();
    return;
  }

  UIController.setButtonLoading("runFootageSearchBtn", true);
  UIController.setStatus(t("uxp.search.runtime.searching_footage", "Searching footage..."), "working");
  setTextAndTitle(
    "searchStatus",
    formatI18n("uxp.search.runtime.searching_for_query", "Searching for \"{query}\"...", { query }),
    formatI18n("uxp.search.runtime.searching_for_query", "Searching for \"{query}\"...", { query })
  );

  const r = await BackendClient.post("/search/footage", { query, top_k: limit });

  UIController.setButtonLoading("runFootageSearchBtn", false);

  if (!r.ok) {
    setTextAndTitle(
      "searchStatus",
      formatI18n("uxp.search.runtime.search_failed_detail", "Could not search the library. {error}", { error: r.error }),
      r.error
    );
    UIController.showToast(formatI18n("uxp.search.runtime.search_error", "Search error: {error}", { error: r.error }), "error");
    UIController.setStatus(t("uxp.search.runtime.search_failed_status", "Search failed. Review the query or reconnect the backend."), "error");
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
          <div class="oc-empty-state-kicker">${UIController.escapeHtml(t("uxp.search.runtime.no_matches_yet", "No matches yet"))}</div>
          <p>${UIController.escapeHtml(t("uxp.search.runtime.no_matches_detail", "Try more descriptive language, quote a spoken phrase, or index a broader folder to widen the search space."))}</p>
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
            <span class="oc-result-list-item-path">${UIController.escapeHtml(card.path || t("uxp.search.runtime.path_unavailable", "Path unavailable for this result"))}</span>
            ${card.preview ? `<span class="oc-result-list-item-excerpt">${UIController.escapeHtml(card.preview)}</span>` : ""}
          </span>
          <span class="oc-result-list-item-meta">${UIController.escapeHtml(card.scoreLabel)}</span>`;
        el.title = card.path || card.preview || card.label;
        el.addEventListener("click", () => {
          if (!card.path) {
            setTextAndTitle(
              "searchStatus",
              t("uxp.search.runtime.match_missing_path_detail", "This match does not include a loadable file path yet. Refine the query or index a folder with source media."),
              card.label
            );
            UIController.showToast(t("uxp.search.runtime.match_missing_path_toast", "This result does not include a loadable file path yet."), "warning");
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
            formatI18n("uxp.search.runtime.loaded_match_detail", "Loaded {label} into the workspace. Search results stay visible while you compare alternate shots.", { label: card.label }),
            card.path
          );
          UIController.showToast(formatI18n("uxp.search.runtime.loaded_match_toast", "Loaded {label} into the workspace.", { label: card.label }), "success");
          UIController.setStatus(formatI18n("uxp.search.runtime.loaded_match_status", "Search match loaded - {label}.", { label: card.label }), "success");
        });
        el.setAttribute("aria-pressed", "false");
        list.appendChild(el);
      });
    }
  }

  setTextAndTitle(
    "searchStatus",
    cards.length
      ? formatCountI18n(
          cards.length,
          "uxp.search.runtime.matches_ready_one",
          "{count} match is ready. Start with {label}, or compare results before loading a shot back into the workspace.",
          "uxp.search.runtime.matches_ready_many",
          "{count} matches are ready. Start with {label}, or compare results before loading a shot back into the workspace.",
          { label: cards[0].label }
        )
      : t("uxp.search.runtime.no_matches_retry", "No matches yet. Try more descriptive language, or index a broader folder."),
    query
  );
  UIController.setStatus(
    cards.length
      ? formatCountI18n(
          cards.length,
          "uxp.search.runtime.search_ready_one",
          "Search ready - {count} result.",
          "uxp.search.runtime.search_ready_many",
          "Search ready - {count} results."
        )
      : t("uxp.search.runtime.search_no_matches_status", "Search returned no matches.")
  );
  syncSearchPanelState();
}

/** ── NLP COMMAND ── */
async function runNlpCommand() {
  const llmProvider = window._llmSettings?.provider || "ollama";
  const command  = document.getElementById("nlpCommand")?.value?.trim();
  const provider = document.getElementById("nlpLlmProvider")?.value ?? llmProvider;
  const backendOnline = isBackendConnected();
  if (!backendOnline) {
    UIController.showToast(t("uxp.search.runtime.reconnect_before_nlp", "Reconnect the backend before running natural-language commands."), "warning");
    syncSearchPanelState();
    return;
  }
  if (!command) { UIController.showToast(t("uxp.search.runtime.enter_natural_language_command", "Please enter a natural language command."), "warning"); return; }

  UIController.setButtonLoading("runNlpBtn", true);
  UIController.showProcessing(t("uxp.search.runtime.parsing_command", "Parsing command..."));

  await JobPoller.start(
    "/nlp/command",
    { command, llm_provider: provider },
    (pct, msg) => { UIController.setProgress(pct); UIController.setProcessingMsg(msg || t("uxp.search.runtime.thinking", "Thinking...")); },
    (result) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runNlpBtn", false);
      showNlpResult(result);
      UIController.showToast(t("uxp.search.runtime.nlp_command_parsed", "NLP command parsed."), "success");
      UIController.setStatus(t("uxp.search.runtime.nlp_command_parsed_status", "NLP command parsed - review and apply."));
    },
    (err) => {
      UIController.hideProcessing();
      UIController.setButtonLoading("runNlpBtn", false);
      UIController.showToast(formatI18n("uxp.search.runtime.nlp_error", "NLP error: {error}", { error: err }), "error");
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
      summary.textContent = formatCountI18n(
        action.cuts.length,
        "uxp.search.runtime.cuts_ready_one",
        "{count} cut ready",
        "uxp.search.runtime.cuts_ready_many",
        "{count} cuts ready"
      );
    } else if (Array.isArray(action?.markers) && action.markers.length) {
      summary.textContent = formatCountI18n(
        action.markers.length,
        "uxp.search.runtime.markers_ready_one",
        "{count} marker ready",
        "uxp.search.runtime.markers_ready_many",
        "{count} markers ready"
      );
    } else {
      summary.textContent = t("uxp.search.review_before_applying", "Review before applying");
    }
  }
  area.focus();
}

/** ── LOAD SEQUENCE INFO ── */
async function loadSequenceInfo() {
  UIController.setButtonLoading("loadSeqInfoBtn", true);
  UIController.setStatus(t("uxp.deliverables.runtime.loading_sequence_info", "Loading sequence info..."), "working");

  const info = await ensureSequenceInfo({ force: true, silent: true });

  UIController.setButtonLoading("loadSeqInfoBtn", false);

  const grid = document.getElementById("seqInfoGrid");
  if (!grid) return;

  if (!info) {
    grid.innerHTML = `
      <div class="oc-empty-state oc-empty-state-inline">
        <div class="oc-empty-state-kicker">${UIController.escapeHtml(t("uxp.deliverables.runtime.no_active_sequence", "No active sequence"))}</div>
        <p>${UIController.escapeHtml(t("uxp.deliverables.runtime.open_sequence_reload_info", "Open a sequence in Premiere, then reload sequence info here before generating deliverables."))}</p>
      </div>`;
    setDeliverablesStatus(t("uxp.deliverables.runtime.no_active_sequence_loaded", "No active sequence loaded. Open a Premiere sequence, then refresh this card before generating deliverables."), "warning");
    UIController.setStatus(t("uxp.deliverables.runtime.no_active_sequence_status", "No active sequence."));
    return;
  }

  const rows = [
    [t("uxp.deliverables.runtime.sequence_name", "Name"), info.name],
    [t("uxp.deliverables.runtime.duration", "Duration"), typeof info.duration === "number" ? formatTimecode(info.duration) : (info.duration ?? t("uxp.deliverables.runtime.not_available", "-"))],
    [t("uxp.deliverables.runtime.frame_rate", "Frame Rate"), info.framerate],
    [t("uxp.deliverables.runtime.resolution", "Resolution"), `${info.width} × ${info.height}`],
    [t("uxp.deliverables.runtime.video_tracks", "Video Tracks"), info.videoTracks],
    [t("uxp.deliverables.runtime.audio_tracks", "Audio Tracks"), info.audioTracks],
  ];

  grid.innerHTML = rows.map(([k, v]) =>
    `<div class="oc-info-pair">` +
      `<span class="oc-info-key">${UIController.escapeHtml(k)}</span>` +
      `<span class="oc-info-val">${UIController.escapeHtml(String(v ?? t("uxp.deliverables.runtime.not_available", "-")))}</span>` +
    `</div>`
  ).join("");

  updateDeliverablesSummary();
  UIController.setStatus(
    formatI18n("uxp.deliverables.runtime.sequence_ready_status", "Sequence ready - {name}", { name: info.name ?? t("uxp.deliverables.runtime.active_sequence_lower", "Active sequence") }),
    "success"
  );
}

/** ── DELIVERABLES ── */
async function runDeliverables(type) {
  const deliverableLabel = getDeliverableLabel(type);
  const backendOnline = isBackendConnected();
  if (!backendOnline) {
    setDeliverablesStatus(formatI18n("uxp.deliverables.runtime.reconnect_before_generating_label", "Reconnect the backend before generating {label}.", { label: deliverableLabel }), "error");
    UIController.showToast(t("uxp.deliverables.runtime.reconnect_before_generating_docs", "Reconnect the backend before generating deliverables."), "warning");
    UIController.setStatus(t("uxp.deliverables.runtime.reconnect_before_generating_docs", "Reconnect the backend before generating deliverables."), "error");
    return;
  }
  const seqData = await ensureSequenceInfo({ silent: true });
  const outputDir = document.getElementById("delivOutputDir")?.value?.trim();
  if (!seqData) {
    setDeliverablesStatus(formatI18n("uxp.deliverables.runtime.load_active_sequence_before_label", "Load the active Premiere sequence before generating {label}.", { label: deliverableLabel }), "warning");
    UIController.showToast(t("uxp.deliverables.runtime.load_sequence_before_docs_status", "Load sequence info before generating deliverables."), "warning");
    UIController.setStatus(t("uxp.deliverables.runtime.load_sequence_before_docs_status", "Load sequence info before generating deliverables."), "error");
    return;
  }

  const btnId = DELIVERABLE_BUTTON_IDS[type];
  if (btnId) UIController.setButtonLoading(btnId, true);
  setDeliverablesStatus(formatI18n("uxp.deliverables.runtime.generating_label", "Generating {label}...", { label: deliverableLabel }), "working");
  UIController.showProcessing(formatI18n("uxp.deliverables.runtime.generating_label", "Generating {label}...", { label: deliverableLabel }));

  await JobPoller.start(
    `/deliverables/${type.replace(/_/g, "-")}`,
    { sequence_data: seqData, output_dir: outputDir || null },
    (pct, msg) => { UIController.setProgress(pct); UIController.setProcessingMsg(msg || t("uxp.deliverables.runtime.generating", "Generating...")); },
    (result) => {
      UIController.hideProcessing();
      if (btnId) UIController.setButtonLoading(btnId, false);
      const outputPath = result.output ?? result.output_path ?? "";
      const outputLabel = outputPath ? formatWorkspaceSource(outputPath) : t("uxp.deliverables.runtime.saved", "saved");
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
      setDeliverablesStatus(
        formatI18n("uxp.deliverables.runtime.deliverable_ready_detail", "{label} is ready. Review the file and continue building the handoff package.", { label: deliverableLabel }),
        "success",
        outputPath || deliverableLabel
      );
      UIController.showToast(
        formatI18n("uxp.deliverables.runtime.deliverable_ready_output", "{label} ready: {output}", { label: deliverableLabel, output: outputLabel }),
        "success"
      );
      UIController.setStatus(formatI18n("uxp.deliverables.runtime.deliverable_generated_status", "{label} generated.", { label: deliverableLabel }), "success");
    },
    (err) => {
      UIController.hideProcessing();
      if (btnId) UIController.setButtonLoading(btnId, false);
      setDeliverablesStatus(formatI18n("uxp.deliverables.runtime.deliverable_failed_detail", "Could not generate {label}. {error}", { label: deliverableLabel, error: err }), "error");
      UIController.showToast(formatI18n("uxp.deliverables.runtime.deliverable_error", "Deliverable error: {error}", { error: err }), "error");
    }
  );
}

/** ── FULL PROJECT REPORT (generates all 4 deliverables) ── */
async function runFullReport() {
  const backendOnline = isBackendConnected();
  if (!backendOnline) {
    setDeliverablesStatus(t("uxp.deliverables.runtime.reconnect_before_report_package", "Reconnect the backend before generating the report package."), "error");
    UIController.showToast(t("uxp.deliverables.runtime.reconnect_before_report_package", "Reconnect the backend before generating the report package."), "warning");
    UIController.setStatus(t("uxp.deliverables.runtime.reconnect_before_report_package", "Reconnect the backend before generating the report package."), "error");
    return;
  }
  const seqData = await ensureSequenceInfo({ silent: true });
  const outputDir = document.getElementById("delivOutputDir")?.value?.trim();
  const selectedTypes = getSelectedReportTypes();
  const packageSummary = getDeliverablesSelectionSummary(selectedTypes);
  const formatSummary = getDeliverablesFormatSummary();
  if (!seqData) {
    setDeliverablesStatus(t("uxp.deliverables.runtime.load_sequence_before_full_report", "Load the active Premiere sequence before generating the full report."), "warning");
    UIController.showToast(t("uxp.deliverables.runtime.load_sequence_before_full_report_toast", "Load sequence info before generating the full report."), "warning");
    UIController.setStatus(t("uxp.deliverables.runtime.load_sequence_before_full_report_toast", "Load sequence info before generating the full report."), "error");
    return;
  }
  if (!selectedTypes.length) {
    setDeliverablesStatus(t("uxp.deliverables.runtime.select_document_before_report_package", "Select at least one handoff document before generating the report package."), "warning");
    UIController.showToast(t("uxp.deliverables.runtime.select_document_before_report_package", "Select at least one handoff document before generating the report package."), "warning");
    UIController.setStatus(t("uxp.deliverables.runtime.select_documents_before_report_status", "Select documents before generating the report package."), "warning");
    updateDeliverablesSummary();
    return;
  }

  const types = selectedTypes.map((type) => type.replace(/_/g, "-"));
  const packageLabel = selectedTypes.length === 1
    ? getDeliverableLabel(selectedTypes[0])
    : (selectedTypes.length === 4
        ? t("uxp.deliverables.runtime.full_report", "Full Report")
        : formatI18n("uxp.deliverables.runtime.doc_package_label", "{count}-Doc Package", { count: selectedTypes.length }));

  UIController.setButtonLoading("runFullReportBtn", true);
  setDeliverablesStatus(formatI18n("uxp.deliverables.runtime.generating_label", "Generating {label}...", { label: packageLabel }), "working", packageSummary.title);
  UIController.showProcessing(formatI18n("uxp.deliverables.runtime.generating_label", "Generating {label}...", { label: packageLabel }));
  UIController.setProgress(0);

  let generated = 0;
  let errors = 0;
  const outputPaths = [];
  for (let index = 0; index < types.length; index += 1) {
    const type = types[index];
    const label = getDeliverableLabel(type.replace(/-/g, "_"));
    UIController.setProgress(Math.round((index / types.length) * 100));
    UIController.setProcessingMsg(formatI18n("uxp.deliverables.runtime.generating_step", "Generating {label} ({step}/{total})...", {
      label,
      step: index + 1,
      total: types.length,
    }));
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
    setDeliverablesStatus(
      formatCountI18n(
        generated,
        "uxp.deliverables.runtime.package_ready_one",
        "{label} ready. {count} document is available for review.",
        "uxp.deliverables.runtime.package_ready_many",
        "{label} ready. {count} documents are available for review.",
        { label: packageLabel }
      ),
      "success",
      outputPaths[0] || outputDir || packageSummary.title
    );
    UIController.showToast(
      formatCountI18n(
        generated,
        "uxp.deliverables.runtime.generated_csv_handoff_one",
        "Generated {count} CSV handoff document.",
        "uxp.deliverables.runtime.generated_csv_handoff_many",
        "Generated {count} CSV handoff documents."
      ),
      "success"
    );
  } else {
    setDeliverablesStatus(
      formatI18n("uxp.deliverables.runtime.package_generated_with_gaps", "{label} generated with a few gaps. {generated} documents completed and {errors} step(s) need attention.", { label: packageLabel, generated, errors }),
      "warning",
      outputPaths[0] || outputDir || packageSummary.title
    );
    UIController.showToast(
      formatI18n("uxp.deliverables.runtime.generated_csv_with_gaps", "Generated {generated} CSV documents; {errors} step(s) need attention.", { generated, errors }),
      "warning"
    );
  }
  UIController.setStatus(
    formatCountI18n(
      generated,
      "uxp.deliverables.runtime.package_ready_status_one",
      "{label} ready - {count} CSV document.",
      "uxp.deliverables.runtime.package_ready_status_many",
      "{label} ready - {count} CSV documents.",
      { label: packageLabel }
    )
  );
}

// ─────────────────────────────────────────────────────────────
// AI B-Roll Generation
// ─────────────────────────────────────────────────────────────
async function runBrollGenerate() {
  const prompt = document.getElementById("brollGenPromptUxp")?.value?.trim();
  if (!prompt) {
    UIController.showToast(t("uxp.video.runtime.enter_broll_description", "Enter a B-roll description."), "warning");
    return;
  }
  const backend = document.getElementById("brollGenBackendUxp")?.value ?? "auto";
  const seedEl = document.getElementById("brollGenSeedUxp");
  const payload = { prompt, backend };
  if (seedEl?.value) payload.seed = parseInt(seedEl.value);
  UIController.setButtonLoading("runBrollGenBtnUxp", true);
  UIController.showProcessing(t("uxp.video.runtime.generating_ai_broll", "Generating AI B-roll..."));
  try {
    const r = await BackendClient.post("/video/broll-generate", payload);
    if (r.ok && r.data?.job_id) {
      const result = await JobPoller.poll(r.data.job_id);
      const output = result?.output_path?.split(/[/\\]/).pop() || t("uxp.video.runtime.done", "done");
      UIController.showToast(
        formatI18n("uxp.video.runtime.broll_generated_output", "B-roll generated: {output}", { output }),
        "success",
      );
      UIController.setStatus(formatI18n("uxp.video.runtime.broll_generated_status", "B-roll generation complete: {output}", { output }));
    } else if (!r.ok) {
      const error = r.error || r.data?.error || t("common.unknown", "unknown");
      UIController.showToast(formatI18n("uxp.video.runtime.broll_generation_failed", "B-roll generation failed: {error}", { error }), "error");
    }
  } catch (e) {
    const error = e.message || t("common.unknown", "unknown");
    UIController.showToast(formatI18n("uxp.video.runtime.broll_generation_failed", "B-roll generation failed: {error}", { error }), "error");
  }
  UIController.hideProcessing();
  UIController.setButtonLoading("runBrollGenBtnUxp", false);
}

// ─────────────────────────────────────────────────────────────
// Multimodal Diarization
// ─────────────────────────────────────────────────────────────
async function runMultimodalDiarize() {
  const clipPath = document.getElementById("clipPathVideo")?.value?.trim();
  if (!clipPath) { showSelectClipWarning(); return; }
  const numSpeakers = document.getElementById("mmDiarizeNumSpeakersUxp")?.value || "";
  const payload = { filepath: clipPath, sample_fps: 2.0, min_face_confidence: 0.5 };
  if (numSpeakers) payload.num_speakers = parseInt(numSpeakers);
  UIController.setButtonLoading("runMmDiarizeBtnUxp", true);
  UIController.showProcessing(t("uxp.video.runtime.running_multimodal_diarization", "Running multimodal diarization..."));
  try {
    const r = await BackendClient.post("/video/multimodal-diarize", payload);
    if (r.ok && r.data?.job_id) {
      const result = await JobPoller.poll(r.data.job_id);
      if (result) {
        const summary = formatI18n(
          "uxp.video.runtime.diarization_summary",
          "{speakers} speakers, {faces} faces, {mapped} mapped",
          {
            speakers: result.num_speakers ?? 0,
            faces: result.num_faces ?? 0,
            mapped: (result.mappings ?? []).length,
          },
        );
        UIController.showToast(formatI18n("uxp.video.runtime.diarization_complete_summary", "Diarization complete: {summary}", { summary }), "success");
        UIController.setStatus(summary);
      }
    } else if (!r.ok) {
      const error = r.error || r.data?.error || t("common.unknown", "unknown");
      UIController.showToast(formatI18n("uxp.video.runtime.diarization_failed", "Diarization failed: {error}", { error }), "error");
    }
  } catch (e) {
    const error = e.message || t("common.unknown", "unknown");
    UIController.showToast(formatI18n("uxp.video.runtime.diarization_failed", "Diarization failed: {error}", { error }), "error");
  }
  UIController.hideProcessing();
  UIController.setButtonLoading("runMmDiarizeBtnUxp", false);
}

// ─────────────────────────────────────────────────────────────
// Social Media Upload
// ─────────────────────────────────────────────────────────────
async function runSocialUpload() {
  const clipPath = document.getElementById("clipPathVideo")?.value?.trim();
  if (!clipPath) { UIController.showToast(t("uxp.video.runtime.select_video_to_upload", "Select a video to upload."), "warning"); return; }
  const platform = document.getElementById("socialPlatformUxp")?.value ?? "youtube";
  const title = document.getElementById("socialTitleUxp")?.value ?? "";
  const description = document.getElementById("socialDescriptionUxp")?.value ?? "";
  const privacy = document.getElementById("socialPrivacyUxp")?.value ?? "private";
  UIController.setButtonLoading("socialUploadBtnUxp", true);
  UIController.showProcessing(formatI18n("uxp.video.runtime.uploading_to_platform", "Uploading to {platform}...", { platform }));
  const r = await BackendClient.post("/social/upload", {
    filepath: clipPath, platform, title, description, privacy,
  });
  if (r.ok && r.data?.job_id) {
    const result = await JobPoller.poll(r.data.job_id);
    if (result?.url) {
      UIController.showToast(formatI18n("uxp.video.runtime.uploaded_view_url", "Uploaded. View at: {url}", { url: result.url }), "success");
    } else if (result) {
      UIController.showToast(formatI18n("uxp.video.runtime.uploaded_to_platform", "Uploaded to {platform}.", { platform }), "success");
    }
  } else if (!r.ok) {
    const error = r.error || r.data?.error || t("common.unknown", "unknown");
    UIController.showToast(formatI18n("uxp.video.runtime.upload_failed", "Upload failed: {error}", { error }), "error");
  }
  UIController.hideProcessing();
  UIController.setButtonLoading("socialUploadBtnUxp", false);
}

async function socialConnectUxp() {
  const platform = document.getElementById("socialPlatformUxp")?.value ?? "youtube";
  const r = await BackendClient.post("/social/auth-url", { platform });
  if (r.ok && r.data?.auth_url) {
    const opened = await openHttpsExternalUrl(
      r.data.auth_url,
      formatI18n("uxp.video.runtime.opening_platform_authorization", "Opening {platform} authorization page in your browser", { platform }),
    );
    if (opened) {
      UIController.showToast(formatI18n("uxp.video.runtime.opening_platform_authorization_done", "Opening {platform} authorization page in your browser.", { platform }), "info");
    }
  } else {
    UIController.showToast(formatI18n("uxp.video.runtime.oauth_not_configured_platform", "OAuth not configured for {platform}. Set API credentials.", { platform }), "warning");
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
  if (Number.isNaN(date.getTime())) return t("uxp.runtime.just_now", "just now");
  return new Intl.DateTimeFormat(undefined, {
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
}

function formatRelativeTime(value) {
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) return t("uxp.runtime.just_now", "just now");
  const diffMs = date.getTime() - Date.now();
  const absMinutes = Math.abs(diffMs) / 60000;
  const formatter = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });
  if (absMinutes < 1) return t("uxp.runtime.just_now", "just now");
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

async function checkConnection({ rescan = false, background = false } = {}) {
  if (rescan) {
    UIController.setStatus(t("uxp.status.scanning_backend_ports", "Scanning OpenCut backend ports..."), "working");
    await refreshBackendBaseUrl();
  }

  if (!background) {
    UIController.setConnection("connecting");
  }
  let r = await BackendClient.get("/health");
  if (!r.ok && !rescan) {
    const previousBackend = BACKEND;
    const detected = await detectBackend();
    if (detected !== previousBackend) {
      BACKEND = detected;
      console.log(`[OpenCut UXP] Backend moved from ${previousBackend} to ${BACKEND}`);
      r = await BackendClient.get("/health");
    }
  }
  const alive = r.ok;
  if (alive && r.data?.csrf_token) csrfToken = r.data.csrf_token;
  UIController.setConnection(alive ? "connected" : "disconnected");

  const wasAlive = _lastConnectionState;
  if (alive) {
    UIController.setStatus(t("uxp.status.backend_connected", "OpenCut backend connected."), "success");
    UIController.setStatusRight(`v${VERSION}`);
  } else {
    UIController.setStatus(
      t("uxp.status.backend_offline", "OpenCut backend offline. Start the local service to run jobs."),
      "error"
    );
    UIController.setStatusRight("");
  }

  // Toggle all action buttons based on connection state
  if (wasAlive !== alive) {
    _lastConnectionState = alive;
    document.querySelectorAll(".oc-btn-primary").forEach(btn => {
      // Don't override buttons already disabled for other reasons (loading state)
      if (!btn.classList.contains("loading")) {
        btn.disabled = !alive;
        if (!alive) {
          btn.dataset.backendLocked = "true";
          btn.setAttribute("aria-disabled", "true");
          btn.title = t("uxp.status.backend_offline_action_title", "Start or reconnect the OpenCut backend before running this action.");
        } else if (btn.dataset.backendLocked === "true") {
          delete btn.dataset.backendLocked;
          btn.removeAttribute("aria-disabled");
          btn.removeAttribute("title");
        }
      }
    });
    syncQuickActionButtons();
    // Show reconnection toast when server comes back
    if (alive && wasAlive === false) {
      UIController.showToast(t("uxp.status.server_reconnected", "Server reconnected."), "success");
    }
  }

  updateWorkspaceOverview();
  return alive;
}

// ─────────────────────────────────────────────────────────────
// Settings quick actions
// ─────────────────────────────────────────────────────────────
const QUICK_ACTION_DEFS = Object.freeze({
  "silence-detect": { tab: "cut", targetId: "runSilenceBtn" },
  "caption-generate": { tab: "captions", targetId: "runTranscribeBtn" },
  "audio-normalize": { tab: "audio", targetId: "runNormalizeBtn" },
});

function syncQuickActionButtons() {
  document.querySelectorAll('[data-quick-action="cancel-job"]').forEach((btn) => {
    btn.disabled = !activeJobId;
    btn.setAttribute("aria-disabled", activeJobId ? "false" : "true");
    btn.title = activeJobId
      ? t("uxp.settings.cancel_active_job_text", "Stop a long-running process from the visible action button.")
      : t("uxp.settings.no_active_job", "No active job to cancel.");
  });
}

async function cancelActiveJobFromSettings() {
  if (!activeJobId) {
    UIController.showToast(t("uxp.settings.no_active_job", "No active job to cancel."), "info");
    return;
  }
  const cancelled = await JobPoller.cancel();
  if (!cancelled) return;
  UIController.clearButtonLoadingStates();
  UIController.hideProcessing();
  UIController.showToast(t("uxp.runtime.job_cancelled", "Job cancelled."), "warning");
  syncQuickActionButtons();
}

function runSettingsQuickAction(actionId) {
  if (actionId === "cancel-job") {
    cancelActiveJobFromSettings();
    return;
  }

  const def = QUICK_ACTION_DEFS[actionId];
  if (!def) {
    UIController.showToast(t("uxp.settings.quick_action_missing", "That action is not available in this panel build."), "warning");
    return;
  }

  UIController.switchTab(def.tab);
  requestAnimationFrame(() => {
    const target = document.getElementById(def.targetId);
    if (!target) {
      UIController.showToast(t("uxp.settings.quick_action_missing", "That action is not available in this panel build."), "warning");
      return;
    }
    target.focus({ preventScroll: false });
    if (target.disabled || target.getAttribute("aria-disabled") === "true") {
      UIController.showToast(
        target.title || t("uxp.settings.quick_action_unavailable", "This action is not ready yet. Check the active workspace status."),
        "warning"
      );
      return;
    }
    target.click();
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
  document.getElementById("refreshBtn")?.addEventListener("click", () => checkConnection({ rescan: true }));
  document.querySelectorAll("[data-quick-action]").forEach((btn) => {
    btn.addEventListener("click", (event) => {
      runSettingsQuickAction(event.currentTarget?.dataset?.quickAction || "");
    });
  });
  document.getElementById("workspaceChooseClipBtn")?.addEventListener("click", () => handleWorkspaceAction("choose-clip"));
  document.getElementById("workspaceSearchBtn")?.addEventListener("click", () => handleWorkspaceAction("open-search"));
  document.getElementById("workspaceTimelineBtn")?.addEventListener("click", () => handleWorkspaceAction("open-timeline"));
  document.getElementById("workspaceGuideAction")?.addEventListener("click", (event) => {
    handleWorkspaceAction(event.currentTarget?.dataset?.action || "");
  });

  // ── Cancel button ──
  document.getElementById("cancelBtn")?.addEventListener("click", async () => {
    const cancelled = await JobPoller.cancel();
    if (!cancelled) return;
    UIController.clearButtonLoadingStates();
    UIController.hideProcessing();
    UIController.showToast(t("uxp.runtime.job_cancelled", "Job cancelled."), "warning");
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
      const copiedLabel = _lastCaptionsResult?.copySuccessLabel || t("uxp.runtime.output", "Output");
      await copyTextToClipboard(body.value, { successLabel: copiedLabel });
    }
  });
  document.getElementById("importSrtBtn")?.addEventListener("click", async () => {
    if (!(_lastCaptionsResult && _lastCaptionsResult.kind === "transcript" && _lastCaptionsResult.hasSrt)) {
      UIController.showToast(
        t("uxp.captions.runtime.timeline_import_needs_srt", "Timeline import is only available when an SRT transcript is ready."),
        "info"
      );
      return;
    }
    UIController.switchTab("timeline");
    focusControl("srtFilePath");
    UIController.setStatus(t("uxp.captions.runtime.timeline_srt_prep_ready", "Timeline SRT prep ready."), "working");
    UIController.showToast(
      t(
        "uxp.captions.runtime.choose_saved_srt",
        "Choose the saved .srt file, then validate it for the CEP ocAddNativeCaptionTrack bridge."
      ),
      "info"
    );
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
    if (!clipPath) { showSelectClipWarning(); return; }
    const mode = document.getElementById("otioExportMode")?.value ?? "cuts";
    const payload = { filepath: clipPath, mode };
    if (mode === "cuts") {
      if (!lastCuts || lastCuts.length === 0) {
        UIController.showToast(t("uxp.timeline.runtime.no_cuts_available", "No cuts available. Run silence removal first."), "warning");
        return;
      }
      payload.cuts = lastCuts;
    } else if (mode === "markers") {
      if (!lastMarkers || lastMarkers.length === 0) {
        UIController.showToast(t("uxp.timeline.runtime.no_markers_available", "No markers available. Run beat detection first."), "warning");
        return;
      }
      payload.markers = lastMarkers.map(m => ({
        time: typeof m === "number" ? m : (m.time ?? m.t ?? 0),
        name: m.label ?? t("uxp.timeline.runtime.marker_label", "Marker"),
      }));
    }
    UIController.setButtonLoading("exportOtioBtn", true);
    const r = await BackendClient.post("/timeline/export-otio", payload);
    UIController.setButtonLoading("exportOtioBtn", false);
    if (r.ok) {
      const output = r.data?.output_path?.split(/[/\\]/).pop() || t("uxp.video.runtime.done", "done");
      UIController.showToast(formatI18n("uxp.timeline.runtime.otio_exported_output", "OTIO exported: {output}", { output }), "success");
      noteTimelineAction(
        t("uxp.timeline.runtime.otio_exported_title", "OTIO exported"),
        "success",
        t("uxp.timeline.runtime.otio_export_ready_detail", "OTIO export is ready for Resolve, Final Cut, Avid, or any OTIO-compatible tool."),
        r.data?.output_path || t("uxp.timeline.runtime.otio_export_default", "OTIO export")
      );
    } else {
      const error = r.error || r.data?.error || t("common.unknown", "unknown");
      UIController.showToast(formatI18n("uxp.timeline.runtime.otio_export_failed", "OTIO export failed: {error}", { error }), "error");
      noteTimelineAction(
        t("uxp.timeline.runtime.otio_export_error_title", "OTIO export error"),
        "error",
        formatI18n("uxp.timeline.runtime.otio_export_failed_detail", "OTIO export failed. {error}", { error }),
        error
      );
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
    if (!bodyEl?.textContent) {
      UIController.showToast(t("uxp.search.runtime.no_nlp_result_to_apply", "No NLP result to apply."), "warning");
      return;
    }
    try {
      const action = JSON.parse(bodyEl.textContent);
      if (action.cuts)    await applyTimelineCuts(action.cuts);
      else if (action.markers) await addSequenceMarkers(action.markers, null);
      else UIController.showToast(t("uxp.search.runtime.unknown_nlp_action", "Unknown NLP action type. Check result JSON."), "info");
    } catch (_) {
      UIController.showToast(t("uxp.search.runtime.nlp_json_parse_failed", "Could not parse NLP result as JSON."), "error");
    }
  });
  document.getElementById("searchQuery")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !hasActiveJob()) runFootageSearch();
  });
  document.getElementById("nlpCommand")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !hasActiveJob()) runNlpCommand();
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
    if (r.ok) {
      UIController.showToast(t("uxp.video.runtime.installing_depth_anything", "Installing Depth Anything V2..."), "info");
    } else {
      const error = r.error || t("common.unknown", "unknown");
      UIController.showToast(formatI18n("uxp.video.runtime.depth_install_failed", "Install failed: {error}", { error }), "error");
    }
  });

  // ── Emotion Highlights ──
  document.getElementById("runEmotionBtnUxp")?.addEventListener("click", runEmotionHighlights);

  // ── B-Roll Analysis ──
  document.getElementById("runBrollPlanBtnUxp")?.addEventListener("click", runBrollAnalysis);

  // ── Chat Editor ──
  document.getElementById("chatSendBtnUxp")?.addEventListener("click", sendChatMessage);
  document.getElementById("chatInputUxp")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !hasActiveJob()) sendChatMessage();
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
  document.getElementById("previewShortsPlanBtnUxp")?.addEventListener("click", previewShortsPlanUxp);
  document.getElementById("runShortsPipelineBtnUxp")?.addEventListener("click", runShortsPipelineUxp);
  document.getElementById("renderShortsApprovedBtnUxp")?.addEventListener("click", renderApprovedShortsUxp);

  // ── Social Media ──
  document.getElementById("socialConnectBtnUxp")?.addEventListener("click", socialConnectUxp);
  document.getElementById("socialUploadBtnUxp")?.addEventListener("click", runSocialUpload);

  // ── Settings: WebSocket ──
  document.getElementById("uxpWsStartBtn")?.addEventListener("click", uxpWsStartBridge);
  document.getElementById("uxpWsStopBtn")?.addEventListener("click", uxpWsStopBridge);
  document.getElementById("uxpWsConnectBtn")?.addEventListener("click", () => { void uxpWsConnect(); });

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
  if (!clipPath) { showSelectClipWarning(); return; }
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
    UIController.showProcessing(t("uxp.video.runtime.running_depth_effect", "Running depth effect..."));
    try {
      const result = await JobPoller.poll(r.data.job_id);
      const output = result?.output_path?.split(/[/\\]/).pop() || t("uxp.video.runtime.done", "done");
      UIController.showToast(formatI18n("uxp.video.runtime.depth_effect_complete", "Depth effect complete: {output}", { output }), "success");
    } catch (e) {
      const error = e.message || t("common.unknown", "unknown");
      UIController.showToast(formatI18n("uxp.video.runtime.depth_effect_failed", "Depth effect failed: {error}", { error }), "error");
    }
    UIController.hideProcessing();
    UIController.setButtonLoading("runDepthBtnUxp", false);
  } else if (!r.ok) {
    UIController.setButtonLoading("runDepthBtnUxp", false);
    const error = r.error || r.data?.error || t("uxp.video.runtime.failed_to_start_depth_effect", "Failed to start depth effect");
    UIController.showToast(formatI18n("uxp.video.runtime.depth_effect_start_error", "Error: {error}", { error }), "error");
  }
}

// ─────────────────────────────────────────────────────────────
// Emotion Highlights
// ─────────────────────────────────────────────────────────────
async function runEmotionHighlights() {
  const clipPath = document.getElementById("clipPathVideo")?.value?.trim() ?? "";
  if (!clipPath) { showSelectClipWarning(); return; }

  UIController.setButtonLoading("runEmotionBtnUxp", true);
  const r = await BackendClient.post("/video/emotion-highlights", { filepath: clipPath });
  if (r.ok && r.data?.job_id) {
    UIController.showProcessing(t("uxp.video.runtime.analyzing_emotions", "Analyzing emotions..."));
    try {
      const result = await JobPoller.poll(r.data.job_id);
      const peaks = result?.peaks?.length ?? 0;
      UIController.showToast(
        formatI18n("uxp.video.runtime.emotion_analysis_complete", "Emotion analysis complete: {count} emotional peak(s) found.", { count: peaks }),
        "success",
      );
    } catch (e) {
      const error = e.message || t("common.unknown", "unknown");
      UIController.showToast(formatI18n("uxp.video.runtime.emotion_analysis_failed", "Emotion analysis failed: {error}", { error }), "error");
    }
    UIController.hideProcessing();
    UIController.setButtonLoading("runEmotionBtnUxp", false);
  } else if (!r.ok) {
    UIController.setButtonLoading("runEmotionBtnUxp", false);
    const error = r.error || r.data?.error || t("uxp.video.runtime.failed_to_start_emotion_analysis", "Failed to start emotion analysis");
    UIController.showToast(formatI18n("uxp.video.runtime.emotion_analysis_start_error", "Error: {error}", { error }), "error");
  }
}

// ─────────────────────────────────────────────────────────────
// B-Roll Analysis
// ─────────────────────────────────────────────────────────────
async function runBrollAnalysis() {
  const clipPath = document.getElementById("clipPathVideo")?.value?.trim() ?? "";
  if (!clipPath) { showSelectClipWarning(); return; }

  UIController.setButtonLoading("runBrollPlanBtnUxp", true);
  const r = await BackendClient.post("/video/broll-plan", { filepath: clipPath });
  if (r.ok && r.data?.job_id) {
    UIController.showProcessing(t("uxp.video.runtime.analyzing_broll_points", "Analyzing B-roll points..."));
    try {
      const result = await JobPoller.poll(r.data.job_id);
      const windows = result?.windows?.length ?? 0;
      UIController.showToast(
        formatI18n("uxp.video.runtime.broll_analysis_complete", "B-roll analysis complete: {count} insertion point(s) found.", { count: windows }),
        "success",
      );
    } catch (e) {
      const error = e.message || t("common.unknown", "unknown");
      UIController.showToast(formatI18n("uxp.video.runtime.broll_analysis_failed", "B-roll analysis failed: {error}", { error }), "error");
    }
    UIController.hideProcessing();
    UIController.setButtonLoading("runBrollPlanBtnUxp", false);
  } else if (!r.ok) {
    UIController.setButtonLoading("runBrollPlanBtnUxp", false);
    const error = r.error || r.data?.error || t("uxp.video.runtime.failed_to_start_broll_analysis", "Failed to start B-roll analysis");
    UIController.showToast(formatI18n("uxp.video.runtime.broll_analysis_start_error", "Error: {error}", { error }), "error");
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
    userDiv.textContent = formatI18n("uxp.runtime.chat_user_prefix", "You: {message}", { message });
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
    const reply = r.data.response || t("uxp.runtime.no_response", "No response.");
    if (history) {
      const replyDiv = document.createElement("div");
      replyDiv.className = "oc-chat-assistant";
      replyDiv.textContent = formatI18n("uxp.runtime.chat_opencut_prefix", "OpenCut: {reply}", { reply });
      history.appendChild(replyDiv);
      history.scrollTop = history.scrollHeight;
    }
    // Auto-execute actions if present
    const actions = r.data.actions || [];
    if (actions.length > 0) {
      UIController.showToast(formatI18n("uxp.runtime.executing_actions", "Executing {count} action(s)...", { count: actions.length }), "info");
    }
  } else {
    if (history) {
      const errDiv = document.createElement("div");
      errDiv.className = "oc-chat-error";
      errDiv.textContent = formatI18n("uxp.runtime.error_prefix", "Error: {error}", { error: r.error || t("uxp.runtime.failed", "Failed") });
      history.appendChild(errDiv);
    }
  }
}

// ─────────────────────────────────────────────────────────────
// WebSocket Client
// ─────────────────────────────────────────────────────────────
let _uxpWs = null;
let _uxpWsReconnectTimer = null;
let _uxpWsReconnectDelayMs = WS_RECONNECT_BASE_MS;
let _uxpWsConnected = false;
let _uxpWsManualDisconnect = false;
let _uxpWsEndpoint = "";

function uxpWsUrlFromBackend(port = WS_BRIDGE_DEFAULT_PORT) {
  try {
    const backendUrl = new URL(BACKEND);
    const protocol = backendUrl.protocol === "https:" ? "wss:" : "ws:";
    const host = backendUrl.hostname || "127.0.0.1";
    return `${protocol}//${host}:${port}`;
  } catch (e) {
    return `ws://127.0.0.1:${port}`;
  }
}

function uxpSetWsEndpoint(data) {
  if (!data) return;
  const endpoint = typeof data.url === "string" ? data.url.trim() : "";
  if (/^wss?:\/\//i.test(endpoint)) {
    _uxpWsEndpoint = endpoint;
    return;
  }
  const port = Number(data.port || data.websocket_port || WS_BRIDGE_DEFAULT_PORT);
  if (Number.isInteger(port) && port >= 1024 && port <= 65535) {
    _uxpWsEndpoint = uxpWsUrlFromBackend(port);
  }
}

async function uxpWsResolveUrl() {
  if (_uxpWsEndpoint) return _uxpWsEndpoint;
  try {
    const r = await BackendClient.get("/ws/status");
    if (r.ok && r.data) uxpSetWsEndpoint(r.data);
  } catch (e) {
    // Fall back to the conventional bridge port if status is temporarily unavailable.
  }
  return _uxpWsEndpoint || uxpWsUrlFromBackend();
}

function uxpWsClearReconnectTimer() {
  if (_uxpWsReconnectTimer) {
    clearTimeout(_uxpWsReconnectTimer);
    _uxpWsReconnectTimer = null;
  }
}

function uxpWsScheduleReconnect() {
  if (_uxpWsManualDisconnect || _uxpWsReconnectTimer) return;
  const delay = _uxpWsReconnectDelayMs;
  _uxpWsReconnectTimer = setTimeout(() => {
    _uxpWsReconnectTimer = null;
    void uxpWsConnect({ reconnect: true });
  }, delay);
  _uxpWsReconnectDelayMs = Math.min(delay * 2, WS_RECONNECT_MAX_MS);
}

async function uxpWsConnect({ reconnect = false } = {}) {
  if (_uxpWs && (_uxpWs.readyState === WebSocket.OPEN || _uxpWs.readyState === WebSocket.CONNECTING)) {
    if (!reconnect) {
      UIController.showToast(t("uxp.settings.live_updates_already_connected", "Live updates are already connected."), "info");
    }
    return;
  }
  _uxpWsManualDisconnect = false;
  uxpWsClearReconnectTimer();
  const wsUrl = await uxpWsResolveUrl();
  let socket = null;
  try {
    socket = new WebSocket(wsUrl);
  } catch (e) {
    UIController.showToast(t("uxp.settings.bridge_open_failed", "Could not open the live-updates bridge."), "warning");
    uxpWsScheduleReconnect();
    return;
  }
  _uxpWs = socket;

  socket.onopen = () => {
    _uxpWsConnected = true;
    _uxpWsReconnectDelayMs = WS_RECONNECT_BASE_MS;
    socket.send(JSON.stringify({ type: "identify", client_type: "uxp", id: "uxp-1" }));
    socket.send(JSON.stringify({ type: "command", action: "subscribe", params: { events: ["progress", "job_complete", "job_error"] }, id: "sub-1" }));
    uxpUpdateWsStatus();
    UIController.setStatus(t("uxp.settings.live_updates_connected_status", "Live updates connected."), "success");
    if (!reconnect) {
      UIController.showToast(t("uxp.settings.live_updates_connected_status", "Live updates connected."), "success");
    }
  };

  socket.onmessage = (evt) => {
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

  socket.onclose = () => {
    _uxpWsConnected = false;
    if (_uxpWs === socket) _uxpWs = null;
    uxpUpdateWsStatus();
    uxpWsScheduleReconnect();
  };

  socket.onerror = () => { _uxpWsConnected = false; };
}

function uxpWsDisconnect() {
  _uxpWsManualDisconnect = true;
  _uxpWsReconnectDelayMs = WS_RECONNECT_BASE_MS;
  uxpWsClearReconnectTimer();
  const socket = _uxpWs;
  _uxpWs = null;
  if (socket) {
    try {
      socket.close();
    } catch (e) { /* ignore */ }
  }
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
  let statusText = _uxpWsConnected
    ? t("uxp.settings.live_updates_connected", "Live updates connected")
    : t("uxp.settings.bridge_unavailable", "Bridge unavailable");
  let statusState = _uxpWsConnected ? "connected" : "unknown";
  let panelStateText = _uxpWsConnected
    ? t("uxp.settings.connected", "Connected")
    : t("uxp.settings.waiting_to_connect", "Waiting to connect");
  let bridgeRunning = false;
  let clients = 0;
  let statusDetail = t("uxp.settings.start_bridge_then_connect", "Start the bridge, then connect this panel to receive live job updates.");

  const r = await BackendClient.get("/ws/status");
  if (r.ok && r.data) {
    uxpSetWsEndpoint(r.data);
    bridgeRunning = !!r.data.running;
    clients = Number(r.data.clients || 0);
    if (_uxpWsConnected) {
      statusText = clients > 0
        ? t("uxp.settings.live_updates_connected", "Live updates connected")
        : t("uxp.settings.panel_connected", "Panel connected");
      statusState = "connected";
      panelStateText = t("uxp.settings.connected", "Connected");
      statusDetail = t("uxp.settings.live_updates_flowing", "Live updates are flowing into this panel. Long-running jobs can report progress, completion, and cancel feedback here.");
    } else if (bridgeRunning) {
      statusText = clients > 0
        ? t("uxp.settings.bridge_ready", "Bridge ready")
        : t("uxp.settings.bridge_idle", "Bridge idle");
      statusState = "ready";
      panelStateText = t("uxp.settings.ready_to_connect", "Ready to connect");
      statusDetail = t("uxp.settings.bridge_running_not_attached", "The bridge is running, but this panel is not attached yet. Connect live updates to bring job progress back into the workspace.");
    } else {
      statusText = t("uxp.settings.bridge_stopped", "Bridge stopped");
      statusState = "stopped";
      panelStateText = t("uxp.settings.not_linked", "Not linked");
      statusDetail = t("uxp.settings.start_live_updates_bridge", "Start the live-updates bridge to bring long-running job feedback back into the panel.");
    }
  } else if (!_uxpWsConnected) {
    statusText = t("uxp.settings.bridge_unavailable", "Bridge unavailable");
    statusState = "error";
    panelStateText = t("uxp.settings.unavailable", "Unavailable");
    statusDetail = t("uxp.settings.bridge_status_unreadable", "The panel could not read bridge status right now. Refresh the backend, then try again.");
  }

  if (statusEl) {
    statusEl.textContent = statusText;
    statusEl.dataset.state = statusState;
    statusEl.title = statusDetail;
  }
  if (countEl) {
    const listenerLabel = clients === 1
      ? t("uxp.settings.listener", "listener")
      : t("uxp.settings.listeners_count_label", "listeners");
    countEl.textContent = formatCountI18n(
      clients,
      "uxp.settings.listener_count_one",
      "{count} listener",
      "uxp.settings.listener_count_many",
      "{count} listeners"
    );
    countEl.dataset.state = clients > 0 ? "active" : "idle";
    countEl.title = clients
      ? formatI18n(
          clients === 1 ? "uxp.settings.listener_attached_title_one" : "uxp.settings.listener_attached_title_many",
          clients === 1
            ? "{count} listener is currently attached to the live-updates bridge."
            : "{count} listeners are currently attached to the live-updates bridge.",
          { count: clients }
        )
      : t("uxp.settings.no_listeners_attached_title", "No listeners are attached to the live-updates bridge right now.");
  }
  if (panelStateEl) {
    panelStateEl.textContent = panelStateText;
    panelStateEl.title = statusDetail;
  }
  setSettingsStatus("settingsBridgeStatus", statusDetail, statusState === "connected" ? "success" : (statusState === "error" ? "error" : (statusState === "stopped" ? "warning" : "ready")), statusDetail);
  if (startBtn) {
    startBtn.disabled = bridgeRunning;
    startBtn.title = bridgeRunning
      ? t("uxp.settings.live_updates_bridge_already_running_title", "The live-updates bridge is already running.")
      : t("uxp.settings.start_live_updates_bridge_title", "Start the live-updates bridge for panel progress and completion updates.");
  }
  if (stopBtn) {
    stopBtn.disabled = !bridgeRunning;
    stopBtn.title = bridgeRunning
      ? t("uxp.settings.stop_live_updates_bridge_title", "Stop the live-updates bridge.")
      : t("uxp.settings.bridge_not_running_title", "The live-updates bridge is not running.");
  }
  if (connectBtn) {
    connectBtn.textContent = _uxpWsConnected
      ? t("uxp.settings.live_updates_connected_button", "Live Updates Connected")
      : t("uxp.settings.connect_live_updates", "Connect Live Updates");
    connectBtn.disabled = !bridgeRunning || _uxpWsConnected;
    connectBtn.title = _uxpWsConnected
      ? t("uxp.settings.panel_receiving_updates_title", "This panel is already receiving live updates.")
      : (!bridgeRunning
          ? t("uxp.settings.start_bridge_before_connecting_title", "Start the bridge before connecting this panel.")
          : t("uxp.settings.connect_panel_updates_title", "Connect this panel to receive live job updates."));
  }
}

async function uxpWsStartBridge() {
  const r = await BackendClient.post("/ws/start", {});
  if (r.ok && r.data?.success) {
    uxpSetWsEndpoint(r.data);
    UIController.setStatus(t("uxp.settings.live_updates_bridge_started", "Live-updates bridge started."), "success");
    UIController.showToast(t("uxp.settings.live_updates_bridge_started", "Live-updates bridge started."), "success");
    setTimeout(() => { void uxpWsConnect(); }, 500);
  } else {
    UIController.showToast(r.error || t("uxp.settings.failed_start_bridge", "Failed to start bridge."), "error");
  }
}

async function uxpWsStopBridge() {
  uxpWsDisconnect();
  const r = await BackendClient.post("/ws/stop", {});
  if (r.ok) {
    UIController.setStatus(t("uxp.settings.live_updates_bridge_stopped", "Live-updates bridge stopped."), "neutral");
    UIController.showToast(t("uxp.settings.live_updates_bridge_stopped", "Live-updates bridge stopped."), "success");
    uxpUpdateWsStatus();
  }
}

// ─────────────────────────────────────────────────────────────
// Engine Registry UI
// ─────────────────────────────────────────────────────────────
async function uxpLoadEngines() {
  const grid = document.getElementById("uxpEngineGrid");
  if (!grid) return;
  setTextAndTitle("settingsEngineDefaultsValue", t("uxp.settings.loading", "Loading..."), t("uxp.settings.loading_automatic_engine_routing_coverage", "Loading automatic engine routing coverage."));
  setTextAndTitle("settingsEnginePinnedValue", t("uxp.settings.loading", "Loading..."), t("uxp.settings.loading_pinned_engine_preferences", "Loading pinned engine preferences."));
  setTextAndTitle("settingsEngineCoverageValue", t("uxp.settings.loading", "Loading..."), t("uxp.settings.loading_engine_availability_title", "Loading engine availability."));
  setSettingsStatus("settingsEngineStatus", t("uxp.settings.loading_engine_availability", "Loading engine availability..."), "working");
  grid.innerHTML = `
    <div class="oc-empty-state oc-empty-state-inline">
      <div class="oc-empty-state-kicker">${UIController.escapeHtml(t("uxp.settings.engine_routing", "Engine routing"))}</div>
      <p>${UIController.escapeHtml(t("uxp.settings.loading_engines_preferences", "Loading available engines and saved preferences..."))}</p>
    </div>`;

  const r = await BackendClient.get("/engines");
  if (!r.ok || !r.data?.engines) {
    setTextAndTitle("settingsEngineDefaultsValue", t("uxp.settings.unavailable", "Unavailable"), t("uxp.settings.engine_defaults_unavailable_title", "OpenCut could not load automatic engine routing coverage."));
    setTextAndTitle("settingsEnginePinnedValue", t("uxp.settings.unavailable", "Unavailable"), t("uxp.settings.engine_pinned_unavailable_title", "OpenCut could not load pinned engine preferences."));
    setTextAndTitle("settingsEngineCoverageValue", t("uxp.settings.refresh_needed", "Refresh needed"), t("uxp.settings.engine_refresh_needed_title", "Refresh availability after the backend reconnects."));
    setSettingsStatus("settingsEngineStatus", t("uxp.settings.engine_availability_failed", "Engine availability could not be loaded. Refresh the backend, then try again."), "error");
    grid.innerHTML = `
      <div class="oc-empty-state oc-empty-state-inline">
        <div class="oc-empty-state-kicker">${UIController.escapeHtml(t("uxp.settings.engine_data_unavailable", "Engine data unavailable"))}</div>
        <p>${UIController.escapeHtml(t("uxp.settings.engine_data_unavailable_hint", "OpenCut could not load engine availability right now. Refresh the backend and try again."))}</p>
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
    const modeLabel = preferredInfo
      ? t("uxp.settings.pinned", "Pinned")
      : availableCount
        ? t("uxp.settings.auto", "Auto")
        : t("uxp.settings.needs_attention", "Needs attention");
    const modeClass = preferredInfo ? "manual" : availableCount ? "auto" : "warning";
    if (preferredInfo) pinnedCount += 1;
    else if (availableCount) autoCount += 1;
    else issueCount += 1;
    let summary = "";

    if (preferredInfo) {
      summary = formatI18n("uxp.settings.engine_preferred_summary", "{engine} is preferred for {domain}.", {
        engine: preferredInfo.display_name,
        domain: domainLabel.toLowerCase(),
      });
      if (activeInfo && activeInfo.name === preferredInfo.name) {
        summary += " " + t("uxp.settings.engine_also_active_summary", "It is also active right now.");
      } else if (activeInfo) {
        summary += " " + formatI18n("uxp.settings.engine_current_active_summary", "Current active engine: {engine}.", {
          engine: activeInfo.display_name,
        });
      }
    } else if (activeInfo) {
      summary = formatI18n("uxp.settings.engine_auto_active_summary", "{engine} is active right now. Auto mode keeps the best available engine selected for this system.", {
        engine: activeInfo.display_name,
      });
    } else if (availableCount) {
      summary = formatI18n(
        availableCount === 1 ? "uxp.settings.engine_available_one_summary" : "uxp.settings.engine_available_many_summary",
        availableCount === 1
          ? "{count} engine is available. Auto mode will pick the best fit at run time."
          : "{count} engines are available. Auto mode will pick the best fit at run time.",
        { count: availableCount }
      );
    } else {
      summary = t("uxp.settings.no_engines_available_summary", "No available engines detected yet. Refresh availability after installs finish.");
    }

    html += `<div class="oc-engine-row">`;
    html += `<div class="oc-engine-copy">`;
    html += `<div class="oc-engine-title-row">`;
    html += `<label class="oc-engine-domain" for="${UIController.escapeHtml(domainId)}">${UIController.escapeHtml(domainLabel)}</label>`;
    html += `<span class="oc-engine-state is-${UIController.escapeHtml(modeClass)}">${UIController.escapeHtml(modeLabel)}</span>`;
    html += `</div>`;
    html += `<p class="oc-engine-meta">${UIController.escapeHtml(summary)}</p>`;
    html += `</div>`;
    html += `<select class="oc-select oc-engine-sel" id="${UIController.escapeHtml(domainId)}" data-domain="${UIController.escapeHtml(domain)}" aria-label="${UIController.escapeHtml(formatI18n("uxp.settings.engine_preference_label", "{domain} engine preference", { domain: domainLabel }))}">`;
    html += `<option value="">${UIController.escapeHtml(t("uxp.settings.auto_best_available", "Auto (best available)"))}</option>`;
    for (const eng of entries) {
      const sel = (preferred === eng.name) ? " selected" : "";
      const avail = eng.available ? "" : t("uxp.settings.option_unavailable_suffix", " - unavailable");
      const activeSuffix = eng.name === active ? t("uxp.settings.option_active_suffix", " - active") : "";
      const label = formatI18n(
        "uxp.settings.engine_option_label",
        "{name} - {quality}/{speed}{unavailable}{active}",
        {
          name: eng.display_name,
          quality: eng.quality,
          speed: eng.speed,
          unavailable: avail,
          active: activeSuffix,
        }
      );
      html += `<option value="${UIController.escapeHtml(eng.name)}"${sel}>${UIController.escapeHtml(label)}</option>`;
    }
    html += `</select></div>`;
  }

  grid.innerHTML = html;
  setTextAndTitle(
    "settingsEngineDefaultsValue",
    formatI18n("uxp.settings.auto_count", "{count} auto", { count: autoCount }),
    formatI18n(
      autoCount === 1 ? "uxp.settings.auto_title_one" : "uxp.settings.auto_title_many",
      autoCount === 1
        ? "{count} domain is currently using automatic engine routing."
        : "{count} domains are currently using automatic engine routing.",
      { count: autoCount }
    )
  );
  setTextAndTitle(
    "settingsEnginePinnedValue",
    formatI18n("uxp.settings.pinned_count", "{count} pinned", { count: pinnedCount }),
    formatI18n(
      pinnedCount === 1 ? "uxp.settings.pinned_title_one" : "uxp.settings.pinned_title_many",
      pinnedCount === 1
        ? "{count} domain has a pinned engine preference."
        : "{count} domains have pinned engine preferences.",
      { count: pinnedCount }
    )
  );
  setTextAndTitle(
    "settingsEngineCoverageValue",
    issueCount
      ? formatI18n(issueCount === 1 ? "uxp.settings.issue_count_one" : "uxp.settings.issue_count_many", issueCount === 1 ? "{count} issue" : "{count} issues", { count: issueCount })
      : formatI18n(availableEngineCount === 1 ? "uxp.settings.engine_ready_count_one" : "uxp.settings.engine_ready_count_many", availableEngineCount === 1 ? "{count} engine ready" : "{count} engines ready", { count: availableEngineCount }),
    formatI18n(
      "uxp.settings.engine_coverage_title",
      "{engineCount} available engines detected across {domainCount} domains.",
      { engineCount: availableEngineCount, domainCount: domains.length }
    )
  );
  if (issueCount) {
    setSettingsStatus(
      "settingsEngineStatus",
      formatI18n(
        issueCount === 1 ? "uxp.settings.engine_issue_status_one" : "uxp.settings.engine_issue_status_many",
        issueCount === 1
          ? "{count} domain still needs attention. Refresh availability after installs finish or return that domain to Auto when engines are ready."
          : "{count} domains still need attention. Refresh availability after installs finish or return those domains to Auto when engines are ready.",
        { count: issueCount }
      ),
      "warning"
    );
  } else if (pinnedCount) {
    setSettingsStatus(
      "settingsEngineStatus",
      t("uxp.settings.engine_routing_pinned_healthy", "Engine routing is healthy. Auto covers the remaining domains, and pinned preferences will stay in place until you change them."),
      "success"
    );
  } else {
    setSettingsStatus(
      "settingsEngineStatus",
      t("uxp.settings.engine_routing_auto_healthy", "Engine routing is healthy. Auto mode will keep the best available engine selected for each domain."),
      "ready"
    );
  }

  grid.querySelectorAll(".oc-engine-sel").forEach(sel => {
    sel.addEventListener("change", async () => {
      const dom = sel.dataset.domain;
      const eng = sel.value;
      const domainLabel = humanizeDomain(dom);
      const selectedLabel = sel.options[sel.selectedIndex]?.textContent || t("uxp.settings.auto", "Auto");
      const pr = await BackendClient.post("/engines/preference", { domain: dom, engine: eng });
      if (pr.ok && pr.data?.success) {
        UIController.setStatus(
          eng
            ? formatI18n("uxp.settings.engine_routing_updated_status", "{domain} engine routing updated.", { domain: domainLabel })
            : formatI18n("uxp.settings.engine_routing_auto_status", "{domain} returned to automatic engine routing.", { domain: domainLabel }),
          "success"
        );
        UIController.showToast(
          eng
            ? formatI18n("uxp.settings.engine_preference_saved_toast", "{domain} now prefers {engine}.", { domain: domainLabel, engine: selectedLabel })
            : formatI18n("uxp.settings.engine_auto_saved_toast", "{domain} is back on Auto routing.", { domain: domainLabel }),
          "success"
        );
        await uxpLoadEngines();
      } else {
        UIController.showToast(pr.error || t("uxp.settings.failed_save_preference", "Failed to save preference."), "error");
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
    direct_uxp: t("uxp.settings.status_direct_uxp", "Direct UXP"),
    partial_uxp: t("uxp.settings.status_partial_uxp", "Partial UXP"),
    cep_only: t("uxp.settings.status_cep_fallback", "CEP fallback"),
    different_mechanism: t("uxp.settings.status_different_mechanism", "Different mechanism"),
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
  setTextAndTitle("settingsMigrationDirectValue", t("uxp.settings.loading", "Loading..."), t("uxp.settings.loading_direct_uxp_coverage", "Loading direct UXP host-action coverage."));
  setTextAndTitle("settingsMigrationFallbackValue", t("uxp.settings.loading", "Loading..."), t("uxp.settings.loading_cep_fallback_count", "Loading CEP fallback count."));
  setTextAndTitle("settingsMigrationRiskValue", t("uxp.settings.loading", "Loading..."), t("uxp.settings.loading_high_risk_migration_count", "Loading high-risk migration count."));
  setSettingsStatus("settingsMigrationStatus", t("uxp.settings.loading_migration_risk_data", "Loading UXP migration risk data..."), "working");
  grid.innerHTML = `
    <div class="oc-empty-state oc-empty-state-inline">
      <div class="oc-empty-state-kicker">${UIController.escapeHtml(t("uxp.settings.migration_risk", "Migration risk"))}</div>
      <p>${UIController.escapeHtml(t("uxp.settings.loading_host_action_coverage", "Loading CEP and UXP host-action coverage..."))}</p>
    </div>`;

  try {
    const response = await fetch(`uxp-migration-dashboard.json?ts=${Date.now()}`, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(formatI18n("uxp.settings.migration_fetch_failed", "HTTP {status}", { status: response.status }));
    }
    const manifest = await response.json();
    const summary = manifest.summary || {};
    const rows = Array.isArray(manifest.rows) ? manifest.rows : [];
    const direct = Number(summary.direct_uxp ?? 0);
    const partial = Number(summary.partial_uxp ?? 0);
    const fallback = Number(summary.cep_only ?? 0);
    const highRisk = Number(summary.high_risk ?? 0);

    setTextAndTitle(
      "settingsMigrationDirectValue",
      formatI18n("uxp.settings.direct_count", "{count} direct", { count: direct }),
      formatI18n("uxp.settings.direct_count_title", "{count} host actions are covered by direct UXP APIs.", { count: direct })
    );
    setTextAndTitle(
      "settingsMigrationFallbackValue",
      formatI18n("uxp.settings.fallback_count", "{count} fallback", { count: fallback }),
      formatI18n("uxp.settings.fallback_count_title", "{count} host actions still require CEP or hybrid handling.", { count: fallback })
    );
    setTextAndTitle(
      "settingsMigrationRiskValue",
      formatI18n("uxp.settings.high_risk_count", "{count} high", { count: highRisk }),
      formatI18n("uxp.settings.high_risk_count_title", "{count} host actions are high-risk migration items.", { count: highRisk })
    );

    if (!rows.length) {
      grid.innerHTML = `
        <div class="oc-empty-state oc-empty-state-inline">
          <div class="oc-empty-state-kicker">${UIController.escapeHtml(t("uxp.settings.migration_dashboard_empty", "Migration data empty"))}</div>
          <p>${UIController.escapeHtml(t("uxp.settings.migration_dashboard_empty_hint", "The generated migration dashboard did not contain host-action rows."))}</p>
        </div>`;
      setSettingsStatus("settingsMigrationStatus", t("uxp.settings.migration_dashboard_empty_status", "Migration dashboard is empty. Regenerate the F260 dashboard artifact."), "warning");
      return;
    }

    grid.innerHTML = rows.map((row) => {
      const statusLabel = migrationStatusLabel(row.status);
      const stateClass = migrationStateClass(row);
      const tags = [
        row.risk ? formatI18n("uxp.settings.risk_tag", "Risk: {risk}", { risk: row.risk }) : "",
        row.needs_hybrid ? t("uxp.settings.hybrid_candidate", "Hybrid candidate") : "",
        Array.isArray(row.f_numbers) && row.f_numbers.length ? row.f_numbers.join(", ") : "",
      ].filter(Boolean).join(t("uxp.settings.tag_separator", " | "));
      const summaryText = formatI18n("uxp.settings.migration_row_summary", "{role} {plan}", {
        role: row.role || t("uxp.settings.host_action", "Host action"),
        plan: row.replacement_plan || row.uxp_path || "",
      }).trim();
      return `<div class="oc-engine-row">
        <div class="oc-engine-copy">
          <div class="oc-engine-title-row">
            <span class="oc-engine-domain">${UIController.escapeHtml(row.name || t("uxp.settings.unknown_action", "Unknown action"))}</span>
            <span class="oc-engine-state is-${UIController.escapeHtml(stateClass)}">${UIController.escapeHtml(statusLabel)}</span>
          </div>
          <p class="oc-engine-meta">${UIController.escapeHtml(summaryText)}</p>
          <p class="oc-engine-meta">${UIController.escapeHtml(tags)}</p>
        </div>
      </div>`;
    }).join("");

    const status = fallback || partial || highRisk
      ? formatI18n("uxp.settings.migration_remaining_status", "{fallback} CEP fallback and {partial} partial UXP host actions remain.", { fallback, partial })
      : t("uxp.settings.migration_resolved_status", "All catalogued host actions are direct UXP or resolved through non-CEP mechanisms.");
    setSettingsStatus("settingsMigrationStatus", status, fallback || highRisk ? "warning" : "success");
  } catch (e) {
    setTextAndTitle("settingsMigrationDirectValue", t("uxp.settings.unavailable", "Unavailable"), t("uxp.settings.migration_direct_unavailable_title", "OpenCut could not load the generated UXP migration dashboard."));
    setTextAndTitle("settingsMigrationFallbackValue", t("uxp.settings.unavailable", "Unavailable"), t("uxp.settings.migration_fallback_unavailable_title", "OpenCut could not load CEP fallback count."));
    setTextAndTitle("settingsMigrationRiskValue", t("uxp.settings.refresh_needed", "Refresh needed"), t("uxp.settings.migration_refresh_needed_title", "Refresh after regenerating or packaging the dashboard artifact."));
    setSettingsStatus("settingsMigrationStatus", t("uxp.settings.migration_dashboard_unavailable_status", "Migration dashboard could not be loaded. Regenerate the F260 dashboard artifact, then refresh."), "error");
    grid.innerHTML = `
      <div class="oc-empty-state oc-empty-state-inline">
        <div class="oc-empty-state-kicker">${UIController.escapeHtml(t("uxp.settings.migration_data_unavailable", "Migration data unavailable"))}</div>
        <p>${UIController.escapeHtml(t("uxp.settings.migration_data_unavailable_hint", "OpenCut could not load the bundled UXP migration dashboard right now."))}</p>
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
  if (!clipPath) { showSelectClipWarning(); return; }
  const scale = parseInt(document.getElementById("upscaleScaleUxp")?.value ?? "2", 10);
  const model = document.getElementById("upscaleModelUxp")?.value ?? "realesrgan-x4plus";

  UIController.setButtonLoading("runUpscaleBtnUxp", true);
  UIController.showProcessing(t("uxp.video.runtime.upscaling_video", "Upscaling video..."));
  try {
    const r = await BackendClient.post("/video/ai/upscale", { filepath: clipPath, scale, model });
    if (r.ok && r.data?.job_id) {
      const result = await JobPoller.poll(r.data.job_id);
      const output = result?.output_path || t("uxp.video.runtime.done", "done");
      UIController.showToast(formatI18n("uxp.video.runtime.upscaled_output", "Upscaled: {output}", { output }), "success");
    } else if (!r.ok) {
      const error = r.data?.error || r.error || t("uxp.video.runtime.upscale_failed_default", "Upscale failed.");
      UIController.showToast(formatI18n("uxp.video.runtime.upscale_failed", "Upscale failed: {error}", { error }), "error");
    }
  } catch (e) {
    const error = e.message || t("common.unknown", "unknown");
    UIController.showToast(formatI18n("uxp.video.runtime.upscale_failed", "Upscale failed: {error}", { error }), "error");
  }
  UIController.hideProcessing();
  UIController.setButtonLoading("runUpscaleBtnUxp", false);
}

// ─────────────────────────────────────────────────────────────
// Scene Detection
// ─────────────────────────────────────────────────────────────
async function runSceneDetectUxp() {
  const clipPath = document.getElementById("clipPathVideo")?.value?.trim() ?? "";
  if (!clipPath) { showSelectClipWarning(); return; }
  const method = document.getElementById("sceneMethodUxp")?.value ?? "ffmpeg";
  const threshold = parseFloat(document.getElementById("sceneThresholdUxp")?.value ?? "0.3");

  UIController.setButtonLoading("runSceneDetectBtnUxp", true);
  UIController.showProcessing(t("uxp.video.runtime.detecting_scenes", "Detecting scenes..."));
  try {
    const r = await BackendClient.post("/video/scenes", { filepath: clipPath, method, threshold });
    if (r.ok && r.data?.job_id) {
      const result = await JobPoller.poll(r.data.job_id);
      const count = result?.scenes?.length || result?.total_scenes || 0;
      UIController.showToast(formatI18n("uxp.video.runtime.scene_boundaries_found", "Found {count} scene boundaries.", { count }), "success");
    } else if (!r.ok) {
      const error = r.data?.error || r.error || t("uxp.video.runtime.scene_detection_failed_default", "Scene detection failed.");
      UIController.showToast(formatI18n("uxp.video.runtime.scene_detection_failed", "Scene detection failed: {error}", { error }), "error");
    }
  } catch (e) {
    const error = e.message || t("common.unknown", "unknown");
    UIController.showToast(formatI18n("uxp.video.runtime.scene_detection_failed", "Scene detection failed: {error}", { error }), "error");
  }
  UIController.hideProcessing();
  UIController.setButtonLoading("runSceneDetectBtnUxp", false);
}

// ─────────────────────────────────────────────────────────────
// Style Transfer
// ─────────────────────────────────────────────────────────────
async function runStyleTransferUxp() {
  const clipPath = document.getElementById("clipPathVideo")?.value?.trim() ?? "";
  if (!clipPath) { showSelectClipWarning(); return; }
  const style = document.getElementById("styleNameUxp")?.value ?? "candy";
  const intensity = (parseInt(document.getElementById("styleIntensityUxp")?.value ?? "100", 10)) / 100;

  UIController.setButtonLoading("runStyleTransferBtnUxp", true);
  UIController.showProcessing(t("uxp.video.runtime.applying_style_transfer", "Applying style transfer..."));
  try {
    const r = await BackendClient.post("/video/style/apply", { filepath: clipPath, style, intensity });
    if (r.ok && r.data?.job_id) {
      const result = await JobPoller.poll(r.data.job_id);
      const output = result?.output_path || t("uxp.video.runtime.done", "done");
      UIController.showToast(formatI18n("uxp.video.runtime.style_applied_output", "Style applied: {output}", { output }), "success");
    } else if (!r.ok) {
      const error = r.data?.error || r.error || t("uxp.video.runtime.style_transfer_failed_default", "Style transfer failed.");
      UIController.showToast(formatI18n("uxp.video.runtime.style_transfer_failed", "Style transfer failed: {error}", { error }), "error");
    }
  } catch (e) {
    const error = e.message || t("common.unknown", "unknown");
    UIController.showToast(formatI18n("uxp.video.runtime.style_transfer_failed", "Style transfer failed: {error}", { error }), "error");
  }
  UIController.hideProcessing();
  UIController.setButtonLoading("runStyleTransferBtnUxp", false);
}

// ─────────────────────────────────────────────────────────────
// Shorts Pipeline
// ─────────────────────────────────────────────────────────────
let shortsReviewPlanUxp = null;

function getShortsPlatformPresetsUxp() {
  const preset = document.getElementById("shortsPlatformUxp")?.value?.trim() || "tiktok";
  return [preset];
}

function getShortsLlmConfigUxp() {
  return {
    provider: document.getElementById("llmProvider")?.value || window._llmSettings?.provider || "ollama",
    model: document.getElementById("llmModel")?.value || window._llmSettings?.model || "llama3",
    apiKey: window._llmSettings?.api_key || "",
    baseUrl: window._llmSettings?.base_url || "",
  };
}

function getShortsPayloadUxp() {
  const clipPath = document.getElementById("clipPathVideo")?.value?.trim() ?? "";
  const maxShorts = parseInt(document.getElementById("shortsMaxUxp")?.value ?? "5", 10);
  const minDuration = parseFloat(document.getElementById("shortsMinUxp")?.value ?? "15");
  const maxDuration = parseFloat(document.getElementById("shortsMaxDurUxp")?.value ?? "60");
  const llm = getShortsLlmConfigUxp();
  return {
    filepath: clipPath,
    platform_presets: getShortsPlatformPresetsUxp(),
    caption_style: document.getElementById("shortsCaptionStyleUxp")?.value || "default",
    max_candidates: maxShorts,
    max_shorts: maxShorts,
    min_duration: minDuration,
    max_duration: maxDuration,
    face_track: document.getElementById("shortsFaceTrackUxp")?.checked ?? true,
    burn_captions: document.getElementById("shortsCaptionsUxp")?.checked ?? true,
    llm_provider: llm.provider,
    llm_model: llm.model,
    llm_api_key: llm.apiKey,
    llm_base_url: llm.baseUrl,
  };
}

function selectedShortsCandidateIdsUxp() {
  const list = document.getElementById("shortsReviewListUxp");
  if (!list) return [];
  return Array.from(list.querySelectorAll("input[data-candidate-id]"))
    .filter(input => input.checked)
    .map(input => input.getAttribute("data-candidate-id"))
    .filter(Boolean);
}

function updateShortsReviewActionsUxp() {
  const ids = selectedShortsCandidateIdsUxp();
  const renderBtn = document.getElementById("renderShortsApprovedBtnUxp");
  if (renderBtn) renderBtn.disabled = !shortsReviewPlanUxp || ids.length === 0;
  const board = document.getElementById("shortsReviewBoardUxp");
  if (board) board.dataset.state = ids.length ? "render" : "plan";
  const summary = document.getElementById("shortsReviewSummaryUxp");
  if (summary && shortsReviewPlanUxp && !shortsReviewPlanUxp.requires_analysis) {
    summary.textContent = ids.length
      ? formatI18n(
          "uxp.video.runtime.shorts_render_state_candidates",
          "Render state: {count} approved candidate(s) will be sent to /video/shorts-pipeline.",
          { count: ids.length },
        )
      : t("uxp.video.runtime.shorts_plan_state_approve_one", "Plan state: approve at least one candidate before rendering.");
  }
}

function appendShortsReviewTextUxp(parent, className, text) {
  if (!text) return;
  const node = document.createElement("span");
  node.className = className;
  node.textContent = text;
  parent.appendChild(node);
}

function renderShortsReviewBoardUxp(plan) {
  shortsReviewPlanUxp = plan || null;
  const board = document.getElementById("shortsReviewBoardUxp");
  const summary = document.getElementById("shortsReviewSummaryUxp");
  const list = document.getElementById("shortsReviewListUxp");
  if (!board || !summary || !list) return;
  board.classList.remove("hidden");
  list.textContent = "";
  const candidates = Array.isArray(plan?.candidates) ? plan.candidates : [];
  board.dataset.state = plan?.requires_analysis ? "analyze" : "plan";
  summary.textContent = plan?.requires_analysis
    ? t("uxp.video.runtime.shorts_analyze_state_requires_data", "Analyze state: cached transcript or highlight data is required before review.")
    : formatI18n(
        "uxp.video.runtime.shorts_plan_state_candidates",
        "Plan state: {count} candidate(s) ready for approval.",
        { count: candidates.length },
      );
  if (!candidates.length) {
    const empty = document.createElement("div");
    empty.className = "shorts-review-card";
    const steps = (plan?.steps || []).map(step => `${step.step_type}: ${step.status}`).join(", ");
    empty.textContent = steps || t("uxp.video.runtime.shorts_no_candidate_windows", "No candidate windows are available yet.");
    list.appendChild(empty);
    updateShortsReviewActionsUxp();
    return;
  }
  candidates.forEach((candidate, index) => {
    const card = document.createElement("div");
    card.className = "shorts-review-card";
    const label = document.createElement("label");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = true;
    checkbox.setAttribute("data-candidate-id", candidate.candidate_id || "");
    checkbox.addEventListener("change", () => {
      card.classList.toggle("is-rejected", !checkbox.checked);
      updateShortsReviewActionsUxp();
    });
    label.appendChild(checkbox);

    const body = document.createElement("span");
    appendShortsReviewTextUxp(body, "shorts-review-title", `${index + 1}. ${candidate.title || t("uxp.video.runtime.shorts_candidate_default", "Candidate")}`);
    appendShortsReviewTextUxp(
      body,
      "shorts-review-meta",
      formatI18n("uxp.video.runtime.shorts_score_range", "Score {score} | {start}s-{end}s", {
        score: candidate.score || 0,
        start: candidate.start || 0,
        end: candidate.end || 0,
      }),
    );
    appendShortsReviewTextUxp(body, "shorts-review-reason", candidate.selection_reason || (candidate.reasons || []).join("; "));
    appendShortsReviewTextUxp(body, "shorts-review-excerpt", candidate.transcript_excerpt || "");
    appendShortsReviewTextUxp(
      body,
      "shorts-review-platforms",
      formatI18n("uxp.video.runtime.shorts_targets", "Targets: {targets}", {
        targets: (candidate.platform_presets || []).map(p => p.preset_id || p).join(", ") || getShortsPlatformPresetsUxp().join(", "),
      }),
    );
    appendShortsReviewTextUxp(
      body,
      "shorts-review-meta",
      formatI18n("uxp.video.runtime.shorts_caption_style", "Caption style: {style}", {
        style: candidate.caption_style || document.getElementById("shortsCaptionStyleUxp")?.value || t("uxp.video.runtime.default_value", "default"),
      }),
    );
    appendShortsReviewTextUxp(
      body,
      "shorts-review-meta",
      formatI18n("uxp.video.runtime.shorts_thumbnail_first_frame", "Thumbnail: first frame at {start}s", { start: candidate.start || 0 }),
    );
    label.appendChild(body);
    card.appendChild(label);
    list.appendChild(card);
  });
  updateShortsReviewActionsUxp();
}

async function previewShortsPlanUxp() {
  const payload = getShortsPayloadUxp();
  if (!payload.filepath) { showSelectClipWarning(); return; }
  UIController.setButtonLoading("previewShortsPlanBtnUxp", true);
  const board = document.getElementById("shortsReviewBoardUxp");
  const summary = document.getElementById("shortsReviewSummaryUxp");
  if (board && summary) {
    board.classList.remove("hidden");
    board.dataset.state = "plan";
    summary.textContent = t("uxp.video.runtime.shorts_plan_building_board", "Plan state: building candidate review board.");
  }
  try {
    const r = await BackendClient.post("/video/magic-clips/plan", payload);
    if (!r.ok) {
      const error = r.error || r.data?.error || t("uxp.video.runtime.magic_clips_plan_failed_default", "Magic Clips plan failed.");
      UIController.showToast(formatI18n("uxp.video.runtime.magic_clips_plan_failed", "Magic Clips plan failed: {error}", { error }), "error");
      return;
    }
    renderShortsReviewBoardUxp(r.data);
  } catch (e) {
    const error = e.message || t("common.unknown", "unknown");
    UIController.showToast(formatI18n("uxp.video.runtime.magic_clips_plan_failed", "Magic Clips plan failed: {error}", { error }), "error");
  } finally {
    UIController.setButtonLoading("previewShortsPlanBtnUxp", false);
  }
}

function shortsBundleFileNameUxp(path) {
  return String(path || "").split(/[\\/]/).pop() || "output";
}

function renderShortsBundleSummaryUxp(result) {
  const bundle = result?.magic_clips_bundle;
  const candidates = Array.isArray(bundle?.candidates) ? bundle.candidates : [];
  const board = document.getElementById("shortsReviewBoardUxp");
  const list = document.getElementById("shortsReviewListUxp");
  const summary = document.getElementById("shortsReviewSummaryUxp");
  if (!candidates.length || !board || !list || !summary) return;
  board.classList.remove("hidden");
  board.dataset.state = "complete";
  const outputCount = bundle.output_count || 0;
  summary.textContent = formatI18n(
    "uxp.video.runtime.shorts_bundle_state_outputs",
    "Bundle state: {count} output(s) saved with {manifest}.",
    { count: outputCount, manifest: shortsBundleFileNameUxp(result.magic_clips_bundle_manifest) },
  );
  list.innerHTML = "";
  candidates.forEach((candidate, index) => {
    const card = document.createElement("div");
    card.className = "shorts-review-card";
    appendShortsReviewTextUxp(card, "shorts-review-title", `${index + 1}. ${candidate.title || t("uxp.video.runtime.shorts_candidate_default", "Candidate")}`);
    appendShortsReviewTextUxp(
      card,
      "shorts-review-meta",
      formatI18n("uxp.video.runtime.shorts_score_range", "Score {score} | {start}s-{end}s", {
        score: candidate.score || 0,
        start: candidate.start || 0,
        end: candidate.end || 0,
      }),
    );
    appendShortsReviewTextUxp(card, "shorts-review-excerpt", candidate.transcript_excerpt || "");
    (candidate.outputs || []).forEach(output => {
      appendShortsReviewTextUxp(
        card,
        "shorts-review-platforms",
        formatI18n("uxp.video.runtime.shorts_platform_output", "{platform}: {output}", {
          platform: output.platform_preset || t("uxp.video.runtime.default_value", "default"),
          output: shortsBundleFileNameUxp(output.export_path),
        }),
      );
    });
    list.appendChild(card);
  });
}

async function renderApprovedShortsUxp() {
  const ids = selectedShortsCandidateIdsUxp();
  if (!shortsReviewPlanUxp || !ids.length) {
    UIController.showToast(t("uxp.video.runtime.shorts_approve_candidate_first", "Approve at least one Magic Clips candidate first."), "warning");
    return;
  }
  const payload = getShortsPayloadUxp();
  if (!payload.filepath) { showSelectClipWarning(); return; }
  payload.magic_clips_plan = shortsReviewPlanUxp;
  payload.plan_id = shortsReviewPlanUxp.plan_id || "";
  payload.candidate_ids = ids;
  payload.approved_candidate_ids = ids;

  UIController.setButtonLoading("renderShortsApprovedBtnUxp", true);
  UIController.showProcessing(t("uxp.video.runtime.rendering_approved_shorts", "Rendering approved short-form clips..."));
  try {
    const r = await BackendClient.post("/video/shorts-pipeline", payload);
    if (r.ok && r.data?.job_id) {
      const result = await JobPoller.poll(r.data.job_id);
      const count = result?.total_clips || result?.clips?.length || 0;
      renderShortsBundleSummaryUxp(result);
      UIController.showToast(formatI18n("uxp.video.runtime.rendered_approved_shorts", "Rendered {count} approved short-form clip(s).", { count }), "success");
    } else if (!r.ok) {
      const error = r.data?.error || r.error || t("uxp.video.runtime.approved_render_failed_default", "Approved render failed.");
      UIController.showToast(formatI18n("uxp.video.runtime.approved_render_failed", "Approved render failed: {error}", { error }), "error");
    }
  } catch (e) {
    const error = e.message || t("common.unknown", "unknown");
    UIController.showToast(formatI18n("uxp.video.runtime.approved_render_failed", "Approved render failed: {error}", { error }), "error");
  } finally {
    UIController.hideProcessing();
    UIController.setButtonLoading("renderShortsApprovedBtnUxp", false);
  }
}

async function runShortsPipelineUxp() {
  const payload = getShortsPayloadUxp();
  if (!payload.filepath) { showSelectClipWarning(); return; }

  UIController.setButtonLoading("runShortsPipelineBtnUxp", true);
  UIController.showProcessing(t("uxp.video.runtime.generating_short_form_clips", "Generating short-form clips..."));
  try {
    const r = await BackendClient.post("/video/shorts-pipeline", payload);
    if (r.ok && r.data?.job_id) {
      const result = await JobPoller.poll(r.data.job_id);
      const count = result?.total_clips || result?.clips?.length || 0;
      renderShortsBundleSummaryUxp(result);
      UIController.showToast(formatI18n("uxp.video.runtime.generated_short_form_clips", "Generated {count} short-form clip(s).", { count }), "success");
    } else if (!r.ok) {
      const error = r.data?.error || r.error || t("uxp.video.runtime.shorts_pipeline_failed_default", "Shorts pipeline failed.");
      UIController.showToast(formatI18n("uxp.video.runtime.shorts_pipeline_failed", "Shorts pipeline failed: {error}", { error }), "error");
    }
  } catch (e) {
    const error = e.message || t("common.unknown", "unknown");
    UIController.showToast(formatI18n("uxp.video.runtime.shorts_pipeline_failed", "Shorts pipeline failed: {error}", { error }), "error");
  } finally {
    UIController.hideProcessing();
    UIController.setButtonLoading("runShortsPipelineBtnUxp", false);
  }
}

async function initApp() {
  console.log(`[OpenCut UXP] v${VERSION} initialising...`);
  const headerVersion = document.getElementById("uxpHeaderVersion");
  const aboutVersion = document.getElementById("uxpVersionDisplay");
  if (headerVersion) headerVersion.textContent = `v${VERSION}`;
  if (aboutVersion) aboutVersion.textContent = `${VERSION} (UXP)`;
  await loadLocale();

  // Keep the panel navigable while backend discovery, UXP bridge checks, and
  // migration metadata load in the background.
  UIController.initCollapsibles();
  bindSliders();
  bindEvents();
  syncQuickActionButtons();
  populateCaptionStyleSelect(UXP_CAPTION_STYLE_FALLBACK, "fallback");
  UIController.switchTab(document.querySelector(".oc-tab.active")?.dataset.tab ?? "cut");
  updateWorkspaceOverview();
  updateDeliverablesSummary();
  updateTimelineReadiness();

  // Detect which port the backend is running on (5679-5689)
  await refreshBackendBaseUrl();
  console.log(`[OpenCut UXP] Backend detected at: ${BACKEND}`);

  // Init UXP Premiere Pro bridge (non-blocking)
  PProBridge.init().then(() => {
    if (PProBridge.available()) {
      UIController.showToast(t("uxp.runtime.premiere_api_available", "UXP Premiere Pro API available."), "success");
      const notice = document.getElementById("uxpTimelineNotice");
      if (notice) notice.style.display = "none";
    }
    updateTimelineReadiness();
  });
  uxpLoadEngines();
  uxpUpdateWsStatus();
  await uxpLoadMigrationRisk();

  // Initial connection check
  const alive = await checkConnection();
  if (alive) {
    _healthBackoff = HEALTH_CHECK_MS; // reset backoff on initial success
    await BackendClient.fetchCsrf();
    await loadCaptionStyleCatalog({ silent: true });
    await loadLlmSettings();
    UIController.showToast(t("uxp.status.backend_connected", "OpenCut backend connected."), "success");

    // Auto-connect WebSocket for real-time progress
    void uxpWsConnect();

    // Scan project media so clip path inputs have autocomplete
    await scanProjectClips();
    await refreshFootageIndexStats({ silent: true });

    // Start periodic backend media scan
    startMediaScanInterval();

    // One-time update check
    const ur = await BackendClient.get("/system/update-check");
    if (ur.ok && ur.data && ur.data.update_available) {
      UIController.showToast(
        formatI18n(
          "uxp.status.update_available",
          "OpenCut v{version} available - visit GitHub to update",
          { version: ur.data.latest_version }
        ),
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
      const ok = await checkConnection({ background: true });
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

  const responseData = (response) => response?.data ?? response ?? {};
  const responseError = (response) => response?.error || response?.data?.error || t("common.unknown", "unknown");

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
      li.textContent = t("uxp.agent.runtime.no_steps_matched", "No steps matched - try a more specific intent.");
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
    const matched = review.matched
      ? t("uxp.agent.runtime.review_matched", "Matched")
      : t("uxp.agent.runtime.review_drift_detected", "Drift detected");
    summary.textContent = formatI18n(
      "uxp.agent.runtime.review_summary",
      "{matched} (drift score {score}/100, source: {source})",
      { matched, score, source: review.source || t("uxp.agent.runtime.heuristic", "heuristic") }
    );
    notes.innerHTML = "";
    (review.drift_notes || []).forEach((n) => {
      const li = document.createElement("li");
      li.textContent = String(n);
      notes.appendChild(li);
    });
    if (review.suggested_retry && review.suggested_retry.endpoint) {
      const li = document.createElement("li");
      li.textContent = formatI18n(
        "uxp.agent.runtime.suggested_retry",
        "Suggested retry: {label} ({endpoint})",
        {
          label: review.suggested_retry.label,
          endpoint: review.suggested_retry.endpoint,
        }
      );
      notes.appendChild(li);
    }
  }

  // --- Chat Conductor wiring -------------------------------------
  const planBtn = $("agentChatPlanBtn");
  if (planBtn) {
    planBtn.addEventListener("click", async () => {
      const intent = ($("agentChatIntent")?.value || "").trim();
      if (!intent) {
        setStatus("agentChatStatus", t("uxp.agent.runtime.enter_intent_first", "Enter an intent first."));
        return;
      }
      planBtn.disabled = true;
      setStatus("agentChatStatus", t("uxp.agent.runtime.building_plan", "Building plan..."));
      try {
        const resp = await BackendClient.post("/agent/chat/plan", { intent });
        if (!resp || resp.error || resp.ok === false) {
          setStatus("agentChatStatus", formatI18n("uxp.agent.runtime.plan_failed", "Plan failed: {error}", { error: responseError(resp) }));
          return;
        }
        const data = responseData(resp);
        activeSessionId = data.session_id || "";
        renderPlan(data.plan);
        showBox("agentChatPlanBox", true);
        showBox("agentChatReviewBox", false);
        setStatus(
          "agentChatStatus",
          formatI18n(
            "uxp.agent.runtime.plan_ready",
            "Plan: {count} step(s) via {source}. Session {session}.",
            {
              count: data.plan?.length || 0,
              source: data.source || t("common.unknown", "unknown"),
              session: activeSessionId.slice(0, 8) || t("common.unknown", "unknown"),
            }
          )
        );
        const reviewBtn = $("agentChatReviewBtn");
        if (reviewBtn) reviewBtn.disabled = false;
      } catch (err) {
        setStatus("agentChatStatus", formatI18n("uxp.agent.runtime.plan_error", "Plan error: {error}", { error: err?.message || err }));
      } finally {
        planBtn.disabled = false;
      }
    });
  }

  const reviewBtn = $("agentChatReviewBtn");
  if (reviewBtn) {
    reviewBtn.addEventListener("click", async () => {
      if (!activeSessionId) {
        setStatus("agentChatStatus", t("uxp.agent.runtime.run_plan_first", "Run Plan first to start a session."));
        return;
      }
      reviewBtn.disabled = true;
      setStatus("agentChatStatus", t("uxp.agent.runtime.running_self_review", "Running self-review..."));
      try {
        const resp = await BackendClient.post("/agent/chat/review", {
          session_id: activeSessionId,
        });
        if (!resp || resp.error || resp.ok === false) {
          setStatus("agentChatStatus", formatI18n("uxp.agent.runtime.review_failed", "Review failed: {error}", { error: responseError(resp) }));
          return;
        }
        const data = responseData(resp);
        renderReview(data);
        setStatus(
          "agentChatStatus",
          formatI18n("uxp.agent.runtime.reviewed_status", "Reviewed ({source}). Drift score {score}/100.", {
            source: data.source || t("common.unknown", "unknown"),
            score: data.drift_score ?? 100,
          })
        );
      } catch (err) {
        setStatus("agentChatStatus", formatI18n("uxp.agent.runtime.review_error", "Review error: {error}", { error: err?.message || err }));
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
      setStatus("agentChatStatus", t("uxp.agent.runtime.cleared", "Cleared."));
    });
  }

  // --- Enhance wiring -------------------------------------------
  async function runEnhance(dryRun) {
    const filepath = ($("enhanceClipPath")?.value || "").trim();
    const style = $("enhanceStyle")?.value || "social";
    if (!filepath) {
      setStatus("enhanceStatus", t("uxp.agent.runtime.enter_clip_path_first", "Enter a clip path first."));
      return;
    }
    const btnId = dryRun ? "enhanceDryRunBtn" : "enhanceRunBtn";
    const btn = $(btnId);
    if (btn) btn.disabled = true;
    setStatus(
      "enhanceStatus",
      dryRun
        ? t("uxp.agent.runtime.building_plan", "Building plan...")
        : t("uxp.agent.runtime.running_enhance", "Running Enhance...")
    );
    try {
      const endpoint = dryRun ? "/enhance/auto/dry-run" : "/enhance/auto";
      const resp = await BackendClient.post(endpoint, { filepath, style });
      if (!resp || resp.error || resp.ok === false) {
        setStatus("enhanceStatus", formatI18n("uxp.agent.runtime.failed", "Failed: {error}", { error: responseError(resp) }));
        return;
      }
      const data = responseData(resp);
      if (dryRun) {
        const steps = data.steps || [];
        const skipped = steps.filter((s) => s.status === "skipped").length;
        setStatus(
          "enhanceStatus",
          formatI18n("uxp.agent.runtime.enhance_plan_ready", "Plan: {runCount} step(s) will run, {skipped} skipped.", {
            runCount: steps.length - skipped,
            skipped,
          })
        );
      } else {
        setStatus("enhanceStatus", t("uxp.agent.runtime.job_queued", "Job queued - watch the progress bar above."));
      }
    } catch (err) {
      setStatus("enhanceStatus", formatI18n("uxp.agent.runtime.error", "Error: {error}", { error: err?.message || err }));
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
      setStatus("variantsStatus", t("uxp.agent.runtime.enter_clip_path_first", "Enter a clip path first."));
      return;
    }
    if (end <= start) {
      setStatus("variantsStatus", t("uxp.agent.runtime.end_after_start", "End must be greater than start."));
      return;
    }
    const btnId = dryRun ? "variantsDryRunBtn" : "variantsRunBtn";
    const btn = $(btnId);
    if (btn) btn.disabled = true;
    setStatus(
      "variantsStatus",
      dryRun
        ? t("uxp.agent.runtime.planning_variants", "Planning variants...")
        : t("uxp.agent.runtime.rendering_variants", "Rendering variants...")
    );
    try {
      const endpoint = dryRun ? "/shorts/variants/dry-run" : "/shorts/variants";
      const resp = await BackendClient.post(endpoint, {
        filepath, start, end, n_variants,
      });
      if (!resp || resp.error || resp.ok === false) {
        setStatus("variantsStatus", formatI18n("uxp.agent.runtime.failed", "Failed: {error}", { error: responseError(resp) }));
        return;
      }
      const variants = responseData(resp).variants || [];
      setStatus("variantsStatus",
        dryRun
          ? formatI18n("uxp.agent.runtime.variant_plan_ready", "Plan: {count} variant(s) will be generated.", { count: variants.length })
          : formatI18n("uxp.agent.runtime.variants_generated", "Generated {count} variant(s).", { count: variants.length }));
    } catch (err) {
      setStatus("variantsStatus", formatI18n("uxp.agent.runtime.error", "Error: {error}", { error: err?.message || err }));
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
    setStatus("sequenceIndexStatus", t("uxp.agent.runtime.reading_active_sequence", "Reading active sequence..."));
    try {
      // PProBridge.getSequenceInfo() returns the same JSON shape that
      // host/index.jsx::ocGetSequenceInfo() produces, which is exactly
      // what /timeline/sequence-index expects.
      const sequence = await PProBridge.getSequenceInfo();
      if (!sequence || sequence.error) {
        setStatus("sequenceIndexStatus", formatI18n("uxp.agent.runtime.no_active_sequence", "No active sequence: {error}", { error: sequence?.error || t("common.unknown", "unknown") }));
        return;
      }
      const resp = await BackendClient.post("/timeline/sequence-index", { sequence });
      if (!resp || resp.error || resp.ok === false) {
        setStatus("sequenceIndexStatus", formatI18n("uxp.agent.runtime.failed", "Failed: {error}", { error: responseError(resp) }));
        return;
      }
      const data = responseData(resp);
      const summary = $("sequenceIndexSummary");
      const box = $("sequenceIndexBox");
      if (summary && box) {
        box.hidden = false;
        summary.textContent =
          formatI18n("uxp.agent.runtime.sequence_index_summary", "{name} - {rows} clips ({duration}s @ {fps}fps), {markers} markers.", {
            name: data.sequence_name || t("uxp.agent.runtime.sequence", "Sequence"),
            rows: data.total_rows ?? 0,
            duration: data.duration_s?.toFixed?.(1) || 0,
            fps: data.fps ?? 0,
            markers: data.marker_count ?? 0,
          });
      }
      setStatus("sequenceIndexStatus",
        formatI18n("uxp.agent.runtime.sequence_index_built", "Index built: {rows} rows across {width}x{height} sequence.", {
          rows: data.total_rows ?? 0,
          width: data.width ?? 0,
          height: data.height ?? 0,
        }));
    } catch (err) {
      setStatus("sequenceIndexStatus", formatI18n("uxp.agent.runtime.error", "Error: {error}", { error: err?.message || err }));
    } finally {
      if (btn) btn.disabled = false;
    }
  });

  $("sequenceIndexInfoBtn")?.addEventListener("click", async () => {
    try {
      const info = await BackendClient.get("/timeline/sequence-index/info");
      const data = responseData(info);
      setStatus("sequenceIndexStatus",
        data?.available
          ? formatI18n("uxp.agent.runtime.sequence_index_capability_ok", "Capability OK - sort keys: {keys}", { keys: (data.sort_keys || []).join(", ") })
          : t("uxp.agent.runtime.sequence_index_unavailable", "Sequence Index unavailable"));
    } catch (err) {
      setStatus("sequenceIndexStatus", formatI18n("uxp.agent.runtime.capability_check_failed", "Capability check failed: {error}", { error: err?.message || err }));
    }
  });

  // --- MCP Bridge wiring -----------------------------------------
  $("mcpBridgeInfoBtn")?.addEventListener("click", async () => {
    try {
      const info = await BackendClient.get("/mcp/info");
      const data = responseData(info);
      setStatus("mcpBridgeStatus",
        formatI18n("uxp.agent.runtime.mcp_bridge_info", "v{version}: {curated} curated + {extended} extended (extended {state} by default).", {
          version: data.version || t("common.unknown", "unknown"),
          curated: data.curated_count ?? 0,
          extended: data.extended_count ?? 0,
          state: data.extended_enabled_by_default ? t("uxp.agent.runtime.on", "on") : t("uxp.agent.runtime.off", "off"),
        }));
    } catch (err) {
      setStatus("mcpBridgeStatus", formatI18n("uxp.agent.runtime.capability_check_failed", "Capability check failed: {error}", { error: err?.message || err }));
    }
  });
  $("mcpBridgeListBtn")?.addEventListener("click", async () => {
    try {
      const resp = await BackendClient.get("/mcp/tools?include_extended=false");
      const data = responseData(resp);
      const names = (data.tools || []).map((tool) => tool.name).slice(0, 5);
      setStatus("mcpBridgeStatus",
        formatI18n("uxp.agent.runtime.mcp_tools_summary", "{count} tool(s). First 5: {names}", { count: data.count ?? names.length, names: names.join(", ") }));
    } catch (err) {
      setStatus("mcpBridgeStatus", formatI18n("uxp.agent.runtime.list_failed", "List failed: {error}", { error: err?.message || err }));
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

  let cachedSchema = null;

  const formatMessage = (key, fallback, params = {}) => {
    let message = t(key, fallback);
    for (const [name, value] of Object.entries(params)) {
      message = message.replace(new RegExp(`\\{${name}\\}`, "g"), String(value));
    }
    return message;
  };

  const setStatus = (key, fallback, params = {}) => {
    const el = $("capDispStatus");
    if (el) el.textContent = formatMessage(key, fallback, params);
  };

  const fontOptionLabel = (opt) => {
    const source = opt?.font_resolution?.source || "preferred_file";
    const status = source === "preferred_file" ? "resolved" : "fallback";
    return `${opt.id} (${opt.font_family}, ${status})`;
  };

  const fontOptionTitle = (opt) => {
    const resolution = opt?.font_resolution || {};
    return resolution.warning || `Font source: ${resolution.source || "resolved"}`;
  };

  const responseData = (response) => response?.data ?? response ?? {};
  const responseError = (response) => response?.error || response?.data?.error || t("common.unknown", "unknown");

  // Selector -> token-category mapping (mirrors token_schema()).
  const SELECT_MAP = [
    { id: "capDispFont", category: "font", labelFn: fontOptionLabel, titleFn: fontOptionTitle },
    { id: "capDispSize", category: "size", labelFn: (o) => `${o.id} (${o.font_size})` },
    { id: "capDispTextColor", category: "color", labelFn: (o) => `${o.id} (${o.hex})`, key: "text_color" },
    { id: "capDispTextOpacity", category: "opacity", labelFn: (o) => `${o.id} (${o.alpha})`, key: "text_opacity" },
    { id: "capDispBgColor", category: "color", labelFn: (o) => `${o.id} (${o.hex})`, key: "background_color" },
    { id: "capDispBgOpacity", category: "opacity", labelFn: (o) => `${o.id} (${o.alpha})`, key: "background_opacity" },
    { id: "capDispEdge", category: "edge_style", labelFn: (o) => `${o.id}` },
  ];

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
        if (spec.titleFn) option.title = spec.titleFn(opt);
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
    sample.textContent = payload.sample_text || t("uxp.fcc.caption_preview_sample", "Caption preview");
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
    setStatus("uxp.fcc.rendering_preview", "Rendering preview...");
    try {
      const resp = await BackendClient.post("/captions/display-settings/preview", {
        settings: readSettings(),
        sample_text: t("uxp.fcc.preview_sample_text", "The quick brown fox jumps over the lazy dog."),
      });
      const data = responseData(resp);
      if ((!resp?.ok && !data.preview_css) || data.error) {
        setStatus("uxp.fcc.preview_failed", "Preview failed: {error}", { error: responseError(resp) });
        return;
      }
      applyPreviewStyles(data);
      setStatus("uxp.fcc.preview_updated", "Preview updated. These are the FCC display tokens applied to burn-in.");
    } catch (err) {
      setStatus("uxp.fcc.preview_error", "Preview error: {error}", { error: err?.message || err });
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
    setStatus("uxp.fcc.reset_defaults_status", "Reset to FCC defaults. Click Preview to re-render.");
  }

  // Lazy-load the token catalogue once the card is first painted.
  // (The Captions tab may not be the user's first stop, so don't block
  // initApp() on this network call.)
  async function lazyLoadTokens() {
    try {
      const resp = await BackendClient.get("/captions/display-settings/tokens");
      const data = responseData(resp);
      if ((!resp?.ok && !data.tokens) || data.error) {
        setStatus("uxp.fcc.schema_unavailable", "Could not load FCC token schema. The card will stay empty.");
        return;
      }
      populateSelects(data);
      // Surface the compliance-date string in the hint if the backend supplies one.
      const complianceDate = document.getElementById("fccComplianceDate");
      if (complianceDate && data.compliance_date) {
        complianceDate.textContent = data.compliance_date;
      }
      setStatus("uxp.fcc.defaults_loaded", "Defaults loaded. Adjust tokens then Preview.");
    } catch (err) {
      setStatus("uxp.fcc.token_schema_failed", "Token-schema fetch failed: {error}", { error: err?.message || err });
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
