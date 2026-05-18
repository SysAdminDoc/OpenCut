type WebviewHost = {
  postMessage: (message: unknown) => void;
  addEventListener?: (type: "message", handler: (event: MessageEvent) => void) => void;
  removeEventListener?: (type: "message", handler: (event: MessageEvent) => void) => void;
};

type UXPWebviewWindow = Window & {
  uxpHost?: WebviewHost;
};

type PendingCall = {
  resolve: (value: unknown) => void;
  reject: (reason: Error) => void;
  timer: ReturnType<typeof setTimeout>;
};

const pending = new Map<string, PendingCall>();

function nextId(): string {
  return `opencut-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export function initWebview(webviewAPI: Record<string, unknown> = {}) {
  const host = (window as UXPWebviewWindow).uxpHost;
  const hasHost = Boolean(host?.postMessage);

  function handleMessage(event: MessageEvent) {
    const data = event.data || {};
    if (data.type === "opencut:host-response" && pending.has(data.id)) {
      const call = pending.get(data.id);
      pending.delete(data.id);
      if (!call) return;
      clearTimeout(call.timer);
      if (data.ok === false) {
        call.reject(new Error(String(data.error || "UXP host call failed.")));
      } else {
        call.resolve(data.result);
      }
    }
    if (data.type === "opencut:host-call") {
      const fn = webviewAPI[String(data.method || "")];
      if (typeof fn === "function") {
        Promise.resolve((fn as (...args: unknown[]) => unknown)(...(data.args || [])))
          .then((result) => host?.postMessage({
            type: "opencut:webview-response",
            id: data.id,
            ok: true,
            result,
          }))
          .catch((error: Error) => host?.postMessage({
            type: "opencut:webview-response",
            id: data.id,
            ok: false,
            error: error.message,
          }));
      }
    }
  }

  host?.addEventListener?.("message", handleMessage);
  window.addEventListener("message", handleMessage);

  function callHost(method: string, args: unknown[] = [], timeoutMs = 30000): Promise<unknown> {
    if (!host?.postMessage) {
      return Promise.reject(new Error("UXP WebView message bridge is unavailable."));
    }
    const id = nextId();
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        pending.delete(id);
        reject(new Error(`UXP host call timed out after ${timeoutMs}ms.`));
      }, timeoutMs);
      pending.set(id, { resolve, reject, timer });
      host.postMessage({
        type: "opencut:webview-call",
        id,
        method,
        args,
      });
    });
  }

  return {
    hasHost,
    callHost,
    dispose() {
      host?.removeEventListener?.("message", handleMessage);
      window.removeEventListener("message", handleMessage);
      for (const call of pending.values()) {
        clearTimeout(call.timer);
      }
      pending.clear();
    },
  };
}
