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

/** Inverse rename entries (newName → oldName) for a one-click undo. */
export function computeInverseRenames(renames) {
  return (Array.isArray(renames) ? renames : []).map((r) => ({
    oldName: r.newName,
    newName: r.oldName,
    path: r.path,
    nodeId: r.nodeId,
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
