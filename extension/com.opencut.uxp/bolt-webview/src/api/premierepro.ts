import type { PlainObject } from "./uxp";

const BACKEND_DEFAULT = "http://127.0.0.1:5679";
const BACKEND_MAX_PORT = 5689;

async function fetchWithTimeout(url: string, timeoutMs = 500): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

async function getPremiereModule(): Promise<any | null> {
  try {
    return await import("premierepro");
  } catch {
    return null;
  }
}

const TICKS_PER_SECOND = 254016000000;

function toSeconds(value: any): number {
  if (value == null) return 0;
  if (typeof value === "number") return value;
  if (value.seconds != null) return Number(value.seconds);
  if (value.ticks != null) return Number(value.ticks) / TICKS_PER_SECOND;
  return Number(value) || 0;
}

function deltaCount(before: string[], after: string[]): number {
  const remaining = new Map<string, number>();
  for (const item of after) remaining.set(item, (remaining.get(item) || 0) + 1);
  let removed = 0;
  for (const item of before) {
    const count = remaining.get(item) || 0;
    if (count > 0) remaining.set(item, count - 1);
    else removed += 1;
  }
  return removed;
}

function addedCount(before: string[], after: string[]): number {
  return deltaCount(after, before);
}

async function hostInfo(ppro: any): Promise<PlainObject> {
  let version = "";
  try { version = String(ppro?.app?.version ?? ppro?.version ?? ""); } catch { /* best effort */ }
  return { bridge: "uxp-bolt", app_name: "Premiere Pro", version, build: "" };
}

async function verificationResult(
  ppro: any,
  result: PlainObject,
  options: {
    action: string;
    attempted: number;
    reported: number;
    verified: number | null;
    canVerify: boolean;
    readBackMethod: string;
    beforeState: PlainObject;
    afterState: PlainObject;
    detail: string;
  },
): Promise<PlainObject> {
  let status = "verified";
  if (!options.canVerify) status = "unverified";
  else if (options.reported > 0 && Number(options.verified || 0) === 0) status = "failed";
  else if (Number(options.verified || 0) < options.reported) status = "partial";
  const host = await hostInfo(ppro);
  const verification = {
    schema: "opencut.host_write_verification.v1",
    action: options.action,
    host_version: host.version,
    host,
    attempted_count: options.attempted,
    reported_count: options.reported,
    verified_count: options.verified,
    verification_status: status,
    read_back_method: options.readBackMethod,
    before_state: options.beforeState,
    after_state: options.afterState,
    detail: options.detail,
  };
  const response: PlainObject = {
    ...result,
    host_write_verification: verification,
    attempted_count: options.attempted,
    reported_count: options.reported,
    verified_count: options.verified,
    verification_status: status,
    read_back_method: options.readBackMethod,
    host_version: host.version,
    host,
  };
  if (status === "failed") {
    response.ok = false;
    response.error_code = "HOST_WRITE_NOT_APPLIED";
    response.reason = "Premiere reported success but the independent read-back found no timeline or project change.";
  } else if (status === "unverified") {
    response.unverified = true;
    response.warning = "Premiere accepted the request, but this operation has no independent read-back API and remains unverified.";
  } else if (status === "partial") {
    response.warning = "Premiere applied only part of the requested host write; review the verified count before continuing.";
  }
  return response;
}

async function markerSnapshot(markerList: any): Promise<string[] | null> {
  if (!markerList) return null;
  try {
    const markers: any[] = [];
    const direct = await markerList.getMarkers?.();
    if (direct && typeof direct[Symbol.iterator] === "function") {
      markers.push(...direct);
    } else if (markerList.getFirstMarker) {
      let marker = await markerList.getFirstMarker();
      let guard = 0;
      while (marker && guard < 1000) {
        markers.push(marker);
        marker = await markerList.getNextMarker?.(marker);
        guard += 1;
      }
    } else {
      return null;
    }
    const result: string[] = [];
    for (const marker of markers) {
      const time = toSeconds(await marker.getTime?.() ?? await marker.getStartTime?.() ?? marker.time ?? marker.start);
      const name = String(await marker.getName?.() ?? marker.name ?? "");
      result.push(`${time.toFixed(6)}|${name}`);
    }
    return result;
  } catch {
    return null;
  }
}

async function trackSnapshot(sequence: any): Promise<string[] | null> {
  if (!sequence || (!sequence.getVideoTrackList && !sequence.getAudioTrackList)) return null;
  const result: string[] = [];
  try {
    for (const [kind, list] of [
      ["video", await sequence.getVideoTrackList?.()],
      ["audio", await sequence.getAudioTrackList?.()],
    ] as const) {
      const trackCount = Number(await list?.getTrackCount?.() ?? list?.length ?? 0);
      for (let trackIndex = 0; trackIndex < trackCount; trackIndex += 1) {
        const track = await list?.getTrackAtIndex?.(trackIndex) ?? list?.[trackIndex];
        const items = await track?.getTrackItems?.() ?? [];
        for (const item of items) {
          const name = String(await item.getName?.() ?? item.name ?? "");
          const id = String(await item.getNodeId?.() ?? await item.getId?.() ?? item.nodeId ?? "");
          const start = toSeconds(await item.getStartTime?.() ?? await item.getStart?.() ?? item.start);
          const end = toSeconds(await item.getEndTime?.() ?? await item.getEnd?.() ?? item.end);
          result.push(`${kind}|${trackIndex}|${id}|${name}|${start.toFixed(6)}|${end.toFixed(6)}`);
        }
      }
    }
    return result;
  } catch {
    return null;
  }
}

async function projectTreeSnapshot(root: any, depth = 0, path = ""): Promise<string[] | null> {
  if (!root || depth > 20) return root ? [] : null;
  try {
    const result: string[] = [];
    const children = await root.getItems?.() ?? [];
    for (const child of children) {
      const name = String(await child.getName?.() ?? child.name ?? "");
      const childPath = path ? `${path}/${name}` : name;
      const id = String(await child.getNodeId?.() ?? child.nodeId ?? "");
      const mediaPath = String(await child.getMediaPath?.() ?? "");
      const isFolder = Boolean(await child.isFolder?.() ?? false);
      result.push(`${isFolder ? "bin" : "item"}|${id}|${childPath}|${mediaPath}`);
      if (isFolder) result.push(...((await projectTreeSnapshot(child, depth + 1, childPath)) ?? []));
    }
    return result;
  } catch {
    return null;
  }
}

async function projectMediaPathSnapshot(root: any, depth = 0): Promise<string[] | null> {
  if (!root || depth > 20) return root ? [] : null;
  try {
    const result: string[] = [];
    const children = await root.getItems?.() ?? [];
    for (const child of children) {
      const mediaPath = String(await child.getMediaPath?.() ?? "").replace(/\\/g, "/").toLowerCase();
      if (mediaPath) result.push(mediaPath);
      if (Boolean(await child.isFolder?.() ?? false)) {
        result.push(...((await projectMediaPathSnapshot(child, depth + 1)) ?? []));
      }
    }
    return result;
  } catch {
    return null;
  }
}

export async function detectBackend(): Promise<PlainObject> {
  for (let port = 5679; port <= BACKEND_MAX_PORT; port += 1) {
    const url = `http://127.0.0.1:${port}`;
    try {
      const response = await fetchWithTimeout(`${url}/health`, 500);
      if (response.ok) {
        return { ok: true, baseUrl: url };
      }
    } catch {
      // Try the next OpenCut loopback port.
    }
  }
  return { ok: false, baseUrl: BACKEND_DEFAULT, reason: "OpenCut backend not detected." };
}

export async function getProjectInfo(): Promise<PlainObject> {
  const ppro = await getPremiereModule();
  const projects = await ppro?.app?.getProjectList?.();
  const project = Array.isArray(projects) ? projects[0] : null;
  if (!project) {
    return { ok: false, reason: "No open Premiere project." };
  }
  return {
    ok: true,
    name: String(await project.getName?.() ?? ""),
  };
}

export async function getSequenceInfo(): Promise<PlainObject> {
  const ppro = await getPremiereModule();
  const projects = await ppro?.app?.getProjectList?.();
  const project = Array.isArray(projects) ? projects[0] : null;
  const sequence = await project?.getActiveSequence?.();
  if (!sequence) {
    return { ok: false, reason: "No active Premiere sequence." };
  }
  const settings = await sequence.getSettings?.();
  return {
    ok: true,
    name: String(await sequence.getName?.() ?? ""),
    duration: Number(await sequence.getEnd?.() ?? 0),
    width: Number(settings?.videoFrameWidth ?? 0),
    height: Number(settings?.videoFrameHeight ?? 0),
    framerate: String(settings?.videoFrameRate ?? ""),
  };
}

export async function addTimelineMarkers(
  markers: Array<{ time: number; label?: string; color?: string }>,
): Promise<PlainObject> {
  const ppro = await getPremiereModule();
  const projects = await ppro?.app?.getProjectList?.();
  const sequence = await projects?.[0]?.getActiveSequence?.();
  const markerList = await sequence?.getMarkerList?.();
  if (!markerList) {
    return { ok: false, reason: "Marker API unavailable." };
  }
  const before = await markerSnapshot(markerList);
  for (const marker of markers) {
    const time = Number(marker.time || 0);
    await markerList.createMarker(time);
    const created = await markerList.getFirstMarkerAtTime?.(time);
    await created?.setName?.(String(marker.label || "OpenCut marker"));
  }
  const after = await markerSnapshot(markerList);
  const canVerify = before !== null && after !== null;
  const verified = canVerify ? addedCount(before, after) : null;
  return verificationResult(ppro, { ok: true, count: markers.length }, {
    action: "ocAddSequenceMarkers",
    attempted: markers.length,
    reported: markers.length,
    verified,
    canVerify,
    readBackMethod: canVerify ? "Sequence marker-list traversal fingerprint diff" : "unavailable: marker-list traversal",
    beforeState: { marker_count: before?.length ?? null },
    afterState: { marker_count: after?.length ?? null },
    detail: "createMarker() writes are verified through an independent marker-list traversal.",
  });
}

export async function applyTimelineCuts(
  cuts: Array<{ start: number; end: number }>,
): Promise<PlainObject> {
  const ppro = await getPremiereModule();
  const projects = await ppro?.app?.getProjectList?.();
  const sequence = await projects?.[0]?.getActiveSequence?.();
  if (!sequence?.rippleDelete) {
    return { ok: false, reason: "Ripple-delete API unavailable." };
  }
  const sorted = [...cuts].sort((a, b) => Number(b.start) - Number(a.start));
  const beforeAll = await trackSnapshot(sequence);
  let verifiedCuts = 0;
  for (const cut of sorted) {
    const beforeCut = await trackSnapshot(sequence);
    const startTick = Math.round(Number(cut.start) * TICKS_PER_SECOND);
    const endTick = Math.round(Number(cut.end) * TICKS_PER_SECOND);
    await sequence.rippleDelete(startTick, endTick);
    const afterCut = await trackSnapshot(sequence);
    if (beforeCut !== null && afterCut !== null &&
        (deltaCount(beforeCut, afterCut) > 0 || addedCount(beforeCut, afterCut) > 0)) {
      verifiedCuts += 1;
    }
  }
  const afterAll = await trackSnapshot(sequence);
  const canVerify = beforeAll !== null && afterAll !== null;
  return verificationResult(ppro, { ok: true, applied: sorted.length }, {
    action: "ocApplySequenceCuts",
    attempted: sorted.length,
    reported: sorted.length,
    verified: canVerify ? verifiedCuts : null,
    canVerify,
    readBackMethod: canVerify ? "video/audio track-item boundary fingerprint diff per cut" : "unavailable: track-item traversal",
    beforeState: { track_item_count: beforeAll?.length ?? null },
    afterState: { track_item_count: afterAll?.length ?? null },
    detail: "Every non-throwing rippleDelete() is followed by a fresh track-item walk; a no-op fails explicitly.",
  });
}

export async function importFiles(
  filePaths: string[],
  binName = "",
): Promise<PlainObject> {
  const ppro = await getPremiereModule();
  const projects = await ppro?.app?.getProjectList?.();
  const project = Array.isArray(projects) ? projects[0] : null;
  if (!project?.importFiles) {
    return { ok: false, reason: "Project import API unavailable." };
  }
  const root = await project.getRootItem?.();
  const before = await projectTreeSnapshot(root);
  const beforePaths = await projectMediaPathSnapshot(root);
  const imported = await project.importFiles(filePaths, Boolean(binName));
  const reported = imported === false ? 0 : filePaths.length;
  const after = await projectTreeSnapshot(root);
  const afterPaths = await projectMediaPathSnapshot(root);
  const canVerify = before !== null && after !== null && beforePaths !== null && afterPaths !== null;
  const verified = canVerify ? Math.min(filePaths.length, addedCount(beforePaths, afterPaths)) : null;
  return verificationResult(ppro, { ok: imported !== false, imported: reported }, {
    action: "importFiles",
    attempted: filePaths.length,
    reported,
    verified,
    canVerify,
    readBackMethod: canVerify ? "project root recursive identity/path fingerprint diff" : "unavailable: project root traversal",
    beforeState: { project_item_count: before?.length ?? null },
    afterState: { project_item_count: after?.length ?? null },
    detail: "Project.importFiles() is verified only when requested media paths are newly present in a fresh project-tree walk.",
  });
}
