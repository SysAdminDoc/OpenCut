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
  for (const marker of markers) {
    const time = Number(marker.time || 0);
    await markerList.createMarker(time);
    const created = await markerList.getFirstMarkerAtTime?.(time);
    await created?.setName?.(String(marker.label || "OpenCut marker"));
  }
  return { ok: true, count: markers.length };
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
  for (const cut of sorted) {
    const startTick = Math.round(Number(cut.start) * 254016000000);
    const endTick = Math.round(Number(cut.end) * 254016000000);
    await sequence.rippleDelete(startTick, endTick);
  }
  return { ok: true, applied: sorted.length };
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
  const imported = await project.importFiles(filePaths, Boolean(binName));
  return { ok: Boolean(imported), imported: imported ? filePaths.length : 0 };
}
