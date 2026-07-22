function applyRenameToken(pattern, name, index) {
  const stem = String(name ?? "").replace(/\.[^./\\]+$/, "");
  const extMatch = String(name ?? "").match(/(\.[^./\\]+)$/);
  const ext = extMatch ? extMatch[1] : "";
  return String(pattern).replace(/\{(name|stem|ext|index(?::0(\d+)d)?)\}/g, (_, token, width) => {
    if (token === "name") return String(name ?? "");
    if (token === "stem") return stem;
    if (token === "ext") return ext;
    const oneBased = index + 1;
    return width ? String(oneBased).padStart(Number(width), "0") : String(oneBased);
  });
}

export function expandRenamePattern(items, pattern) {
  const list = Array.isArray(items) ? items : [];
  const effectivePattern = String(pattern || "").trim() || "{stem}_{index:03d}{ext}";
  const renames = [];
  const usedNames = new Set();
  list.forEach((item, index) => {
    const oldName = String(item?.name ?? "");
    if (!oldName) return;
    const newName = applyRenameToken(effectivePattern, oldName, index).trim();
    if (!newName || newName === oldName || usedNames.has(newName)) return;
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

export function computeInverseRenames(renames) {
  return (Array.isArray(renames) ? renames : []).map((item) => ({
    oldName: item.newName,
    newName: item.oldName,
    path: item.path,
    nodeId: item.nodeId,
  }));
}

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
      add((rawName.match(/\.([^./\\]+)$/)?.[1] || "other").toUpperCase());
    } else if (strat.includes("resolution")) {
      add(item?.resolution || "Unsorted");
    } else if (strat.includes("date")) {
      add(item?.date || "Undated");
    } else {
      const segment = String(item?.path ?? "").replace(/\\/g, "/").split("/").slice(-2, -1)[0];
      add(segment || "Media");
    }
  }
  return [...names].map((name) => ({ name }));
}

export function summarizeRenamePreview(renames, limit = 5) {
  const list = Array.isArray(renames) ? renames : [];
  return {
    count: list.length,
    sample: list.slice(0, limit).map((item) => `${item.oldName} -> ${item.newName}`),
  };
}

export function normalizeTimelineCuts(cuts) {
  return (Array.isArray(cuts) ? cuts : [])
    .map((cut) => ({ start: Number(cut?.start), end: Number(cut?.end) }))
    .filter((cut) => Number.isFinite(cut.start) && Number.isFinite(cut.end) && cut.end > cut.start)
    .sort((a, b) => b.start - a.start);
}

export function buildMarkerPayload(markers, defaultType = "Comment") {
  return (Array.isArray(markers) ? markers : [])
    .map((marker, index) => ({
      time: Number(marker?.time ?? marker?.seconds),
      name: String(marker?.name || `Marker ${index + 1}`),
      type: String(marker?.type || defaultType),
    }))
    .filter((marker) => Number.isFinite(marker.time) && marker.time >= 0);
}
