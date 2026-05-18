export type HostToast = {
  message: string;
  state?: "info" | "success" | "warning" | "error";
};

export function pingWebview(): string {
  return "opencut-webview-ready";
}

export function showToastFromHost(payload: HostToast): string {
  const message = String(payload?.message || "").trim();
  if (!message) {
    return "ignored-empty-toast";
  }
  const event = new CustomEvent("opencut:webview-toast", {
    detail: {
      message,
      state: payload.state || "info",
    },
  });
  window.dispatchEvent(event);
  return "toast-dispatched";
}

export function updateColorScheme(payload: { theme: string; colors: Record<string, string> }): string {
  const root = document.documentElement;
  for (const [key, value] of Object.entries(payload.colors || {})) {
    if (key.startsWith("--uxp-host-")) {
      root.style.setProperty(key, String(value));
    }
  }
  root.dataset.uxpTheme = String(payload.theme || "darkest");
  return "color-scheme-updated";
}
