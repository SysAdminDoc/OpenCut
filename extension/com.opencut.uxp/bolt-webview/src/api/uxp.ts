type Primitive = string | number | boolean | null;

export type PlainObject = {
  [key: string]: Primitive | Primitive[] | PlainObject | PlainObject[];
};

export type UXPInfo = {
  pluginId: string;
  pluginVersion: string;
  hostName: string;
  hostVersion: string;
  uxpVersion: string;
};

export type HostColorScheme = {
  theme: string;
  colors: Record<string, string>;
};

const HOST_COLOR_DEFAULTS: Record<string, string> = {
  "--uxp-host-background-color": "#323232",
  "--uxp-host-text-color": "#ffffff",
  "--uxp-host-border-color": "#292929",
  "--uxp-host-link-text-color": "#77a7ff",
  "--uxp-host-widget-hover-background-color": "#3d3d3d",
  "--uxp-host-widget-hover-text-color": "#ffffff",
  "--uxp-host-widget-hover-border-color": "#3d3d3d",
  "--uxp-host-text-color-secondary": "#b5bfca",
  "--uxp-host-link-hover-text-color": "#ffffff",
  "--uxp-host-label-text-color": "#ffffff",
};

function getPluginInfo(): { id?: string; version?: string } {
  const uxpGlobal = (globalThis as any).uxp;
  return uxpGlobal?.entrypoints?._pluginInfo ?? {};
}

export async function getUXPInfo(): Promise<UXPInfo> {
  const uxpGlobal = (globalThis as any).uxp;
  const pluginInfo = getPluginInfo();
  return {
    pluginId: String(pluginInfo.id ?? "com.opencut.uxp"),
    pluginVersion: String(pluginInfo.version ?? "unknown"),
    hostName: String(uxpGlobal?.host?.name ?? "premierepro").toLowerCase(),
    hostVersion: String(uxpGlobal?.host?.version ?? "unknown"),
    uxpVersion: String(uxpGlobal?.versions?.uxp ?? "unknown"),
  };
}

export async function getColorScheme(): Promise<HostColorScheme> {
  const themeApi = (globalThis as any).document?.theme;
  const theme = String(themeApi?.getCurrent?.() ?? "darkest");
  return {
    theme,
    colors: HOST_COLOR_DEFAULTS,
  };
}

export async function openURL(url: string): Promise<PlainObject> {
  const trimmed = String(url || "").trim();
  let parsed: URL;
  try {
    parsed = new URL(trimmed);
  } catch {
    return { ok: false, reason: "Only https URLs can be opened from UXP." };
  }
  if (parsed.protocol !== "https:") {
    return { ok: false, reason: "Only https URLs can be opened from UXP." };
  }
  const uxpGlobal = (globalThis as any).uxp;
  const result = await uxpGlobal?.shell?.openExternal?.(
    parsed.href,
    "Opening a secure web page in your browser",
  );
  if (typeof result === "string" && result.trim()) {
    return { ok: false, reason: result };
  }
  return { ok: true };
}
