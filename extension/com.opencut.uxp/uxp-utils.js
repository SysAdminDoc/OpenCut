export function escapeHtml(value) {
  if (value === undefined || value === null) return "";
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

export function safeDomIdSegment(value) {
  return String(value ?? "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "-")
    .replace(/^-+|-+$/g, "") || "item";
}

// ── Batch rename / smart bins pure logic ─────────────────────────────
// Extracted from the timeline controller so it can be unit-tested without a
// live Premiere/UXP host. The UI passes real project items in; these helpers
// build the payloads the PProBridge host actions consume, plus an inverse map
// for undo and a human preview.

function _applyRenameToken(pattern, name, index) {
  const stem = String(name ?? "").replace(/\.[^./\\]+$/, "");
  const extMatch = String(name ?? "").match(/(\.[^./\\]+)$/);
  const ext = extMatch ? extMatch[1] : "";
  return String(pattern).replace(/\{(name|stem|ext|index(?::0(\d+)d)?)\}/g, (_, token, width) => {
    if (token === "name") return String(name ?? "");
    if (token === "stem") return stem;
    if (token === "ext") return ext;
    // index or index:0Nd
    const oneBased = index + 1;
    if (width) return String(oneBased).padStart(Number(width), "0");
    return String(oneBased);
  });
}

/**
 * Expand a rename pattern (e.g. "{stem}_{index:03d}{ext}") across items.
 * Returns rename entries the ocBatchRenameProjectItems host action consumes.
 * Items whose name is unchanged, empty, or would collide are skipped.
 */
export function expandRenamePattern(items, pattern) {
  const list = Array.isArray(items) ? items : [];
  const effectivePattern = String(pattern || "").trim() || "{stem}_{index:03d}{ext}";
  const renames = [];
  const usedNames = new Set();
  list.forEach((item, index) => {
    const oldName = String(item?.name ?? "");
    if (!oldName) return;
    const newName = _applyRenameToken(effectivePattern, oldName, index).trim();
    if (!newName || newName === oldName) return;
    if (usedNames.has(newName)) return; // avoid collisions within the batch
    usedNames.add(newName);
    renames.push({
      oldName,
      newName,
      path: item?.path ?? item?.mediaPath ?? "",
      nodeId: item?.nodeId,
    });
  });
  return renames;
}

/** Inverse rename entries for a one-click undo / journal checkpoint.
 * Canonical journal inverse shape shared with the CEP host restore
 * (ocUnrenameItems in host/index.jsx): oldName is the ORIGINAL name to
 * restore, currentName is the name the rename applied. Consumers detect
 * legacy pre-canonical entries by the absence of `currentName` (those
 * stored {oldName: applied, newName: original}). */
export function computeInverseRenames(renames) {
  return (Array.isArray(renames) ? renames : []).map((r) => ({
    nodeId: r.nodeId,
    path: r.path,
    oldName: r.oldName,
    currentName: r.newName,
  }));
}

/** Build smart-bin rules from a strategy label and the current items. */
export function buildSmartBinRules(items, strategy) {
  const list = Array.isArray(items) ? items : [];
  const strat = String(strategy || "").toLowerCase();
  const names = new Set();
  const add = (name) => {
    const clean = String(name || "").trim();
    if (clean) names.add(clean);
  };
  for (const item of list) {
    const rawName = String(item?.name ?? "");
    if (strat.includes("type") || strat.includes("extension")) {
      const ext = (rawName.match(/\.([^./\\]+)$/)?.[1] || "other").toUpperCase();
      add(ext);
    } else if (strat.includes("resolution")) {
      add(item?.resolution || "Unsorted");
    } else if (strat.includes("date")) {
      add(item?.date || "Undated");
    } else {
      // Default: parent-folder grouping from the media path.
      const seg = String(item?.path ?? "").replace(/\\/g, "/").split("/").slice(-2, -1)[0];
      add(seg || "Media");
    }
  }
  return [...names].map((name) => ({ name }));
}

/** Human-readable preview of a pending rename batch. */
export function summarizeRenamePreview(renames, limit = 5) {
  const list = Array.isArray(renames) ? renames : [];
  const sample = list.slice(0, limit).map((r) => `${r.oldName} -> ${r.newName}`);
  return { count: list.length, sample };
}

// ── Pure formatting / normalization helpers ──────────────────────────
// Extracted from the UXP controller so they can be unit-tested without a
// live Premiere/UXP host. All are deterministic (argument-only) with no DOM,
// host API, i18n, or Date/locale access.

function pad(n) {
  return String(n).padStart(2, "0");
}

/** Seconds → HH:MM:SS.ff timecode, or "—" for non-numbers. */
export function formatTimecode(seconds) {
  if (typeof seconds !== "number" || isNaN(seconds)) return "—";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  const f = Math.floor((seconds % 1) * 100);
  return `${pad(h)}:${pad(m)}:${pad(s)}.${pad(f)}`;
}

/** Seconds → compact human duration ("450 ms", "12.5 s", "3m 4s"). */
export function formatCompactDuration(seconds) {
  if (typeof seconds !== "number" || isNaN(seconds) || seconds <= 0) return "0 s";
  if (seconds < 1) return `${Math.round(seconds * 1000)} ms`;
  if (seconds < 60) return `${seconds.toFixed(seconds >= 10 ? 1 : 2)} s`;
  const minutes = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60);
  return `${minutes}m ${secs}s`;
}

/** Byte count → human size ("0 B", "512 KB", "1.5 GB"). */
export function formatBytes(bytes) {
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

/** Title-case an underscore-delimited domain ("silence_cut" → "Silence Cut"). */
export function humanizeDomain(domain) {
  return String(domain || "")
    .split("_")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

/** Basename of a path across / or \\ separators, defaulting to "output". */
export function shortsBundleFileNameUxp(path) {
  return String(path || "").split(/[\\/]/).pop() || "output";
}

/** Return a validated https:// href, or null for anything else. */
export function normalizeHttpsExternalUrl(value) {
  const raw = String(value || "").trim();
  if (!raw) return null;

  try {
    const parsed = new URL(raw);
    return parsed.protocol === "https:" ? parsed.href : null;
  } catch (_) {
    return null;
  }
}

/** True when an error represents an abort/timeout. */
export function isTimeoutError(err) {
  const message = String(err?.message || err || "");
  return err?.name === "AbortError" || /timed out|abort/i.test(message);
}

/** First non-empty path-like field on a search result item. */
export function getSearchResultPath(item) {
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

/** First non-empty preview/snippet field, whitespace-collapsed and truncated. */
export function getSearchResultPreview(item) {
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

// ── Pure catalog / classifier / locale helpers ───────────────────────
// Deterministic, host-independent logic extracted from the UXP controller.
// The locale helpers take the default locale as a parameter (its "en"
// default matches the controller's UXP_DEFAULT_LOCALE), so call sites need
// no changes.

/** Dedupe + coerce a caption-style catalog to a fixed display shape. */
export function normalizeCaptionStyleCatalog(styles) {
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

/** Map a migration row's risk/status to a trust-badge class. */
export function migrationStateClass(row) {
  if (row.risk === "high" || row.status === "cep_only") return "warning";
  if (row.status === "partial_uxp" || row.risk === "medium") return "manual";
  return "auto";
}

/** Map a plugin's trust/load state to a trust-badge class. */
export function pluginTrustStateClass(plugin) {
  const trust = plugin?.trust || {};
  if (plugin?.load_status === "failed" || trust.errors?.length) return "warning";
  if (trust.unsigned_allowed || trust.lock_missing) return "manual";
  if (plugin?.load_status === "loaded") return "auto";
  return "manual";
}

/** Normalize a locale tag to lowercase BCP-47-ish form (e.g. "en-US" → "en-us"). */
export function normalizeLocaleTag(lang, defaultLocale = "en") {
  return String(lang || defaultLocale).trim().toLowerCase().replace(/_/g, "-") || defaultLocale;
}

/** Ordered fallback locale candidates: [full, base, default]. */
export function getLocaleCandidates(lang, defaultLocale = "en") {
  const normalized = normalizeLocaleTag(lang, defaultLocale);
  const base = normalized.split("-")[0];
  const candidates = [normalized];
  if (base && base !== normalized) candidates.push(base);
  if (defaultLocale !== normalized && defaultLocale !== base) candidates.push(defaultLocale);
  return candidates;
}

// ── i18n lookup + interpolation (pure core) ──────────────────────────
// The controller's `t()` / `formatI18n()` bind these to the live translation
// map; the resolution and placeholder-substitution logic itself is pure and
// unit-tested here.

/** Look up a key in a translation map, falling back to `fallback` then `key`. */
export function translate(map, key, fallback) {
  return (map && map[key]) || fallback || key;
}

/**
 * Replace `{name}` placeholders in `text` with values[name] (all occurrences,
 * treated literally so `$`-sequences in values are not special).
 */
export function interpolateI18n(text, values = {}) {
  let out = String(text ?? "");
  Object.keys(values || {}).forEach((name) => {
    const val = String(values[name]);
    out = out.replace(new RegExp(`\\{${name}\\}`, "g"), () => val);
  });
  return out;
}
