import { isTimeoutError as defaultIsTimeoutError } from "./uxp-utils.js";

// F471: /health hands out the CSRF token only to a caller holding the
// per-user bootstrap secret. The server copies it into this plugin's data
// folder, which UXP lets the panel read without a file-system permission.
export const PANEL_BOOTSTRAP_HEADER = "X-OpenCut-Panel-Bootstrap";
export const PANEL_BOOTSTRAP_FILENAME = "panel_bootstrap.token";

const LOOPBACK_BACKEND = /^https?:\/\/(?:127(?:\.\d{1,3}){3}|localhost|\[::1\])(?::\d+)?(?:\/|$)/i;

/** Read the bootstrap secret from the plugin data folder, or "" when it isn't there yet. */
export async function readPluginDataBootstrapSecret(getLocalFileSystem) {
  try {
    const localFileSystem = await getLocalFileSystem();
    if (!localFileSystem || typeof localFileSystem.getDataFolder !== "function") return "";
    const folder = await localFileSystem.getDataFolder();
    if (!folder || typeof folder.getEntry !== "function") return "";
    const entry = await folder.getEntry(PANEL_BOOTSTRAP_FILENAME);
    if (!entry || entry.isFolder || typeof entry.read !== "function") return "";
    const text = await entry.read();
    return typeof text === "string" ? text.trim() : "";
  } catch (_) {
    return "";
  }
}

/** Create the UXP HTTP boundary with CSRF refresh and one-retry semantics. */
export function createBackendClient({
  state,
  fetchWithTimeout,
  isTimeoutError = defaultIsTimeoutError,
  onCapabilities = () => {},
  requestTimeoutMs = 120000,
  getLocalFileSystem = null,
} = {}) {
  if (!state) throw new TypeError("createBackendClient requires a state store");
  if (typeof fetchWithTimeout !== "function") {
    throw new TypeError("createBackendClient requires fetchWithTimeout");
  }

  let capabilities = {};
  let bootstrapSecret = "";
  let bootstrapLoaded = false;

  async function loadBootstrapSecret(force = false) {
    if (bootstrapLoaded && !force) return bootstrapSecret;
    bootstrapSecret = typeof getLocalFileSystem === "function"
      ? await readPluginDataBootstrapSecret(getLocalFileSystem)
      : "";
    bootstrapLoaded = true;
    return bootstrapSecret;
  }

  async function doFetch(method, endpoint, body) {
    const headers = { "Content-Type": "application/json" };
    if (state.csrfToken) headers["X-OpenCut-Token"] = state.csrfToken;
    // Only a loopback backend gets the secret; a remote one has its own.
    if (endpoint === "/health" && bootstrapSecret && LOOPBACK_BACKEND.test(state.backendUrl || "")) {
      headers[PANEL_BOOTSTRAP_HEADER] = bootstrapSecret;
    }
    const opts = { method, headers };
    if (body && method !== "GET") opts.body = JSON.stringify(body);

    let resp;
    try {
      resp = await fetchWithTimeout(
        `${state.backendUrl}${endpoint}`,
        opts,
        requestTimeoutMs,
      );
    } catch (err) {
      if (isTimeoutError(err)) {
        throw new Error("Backend request timed out. Check that OpenCut Server is still running, then try again.");
      }
      throw err;
    }

    const newToken = resp.headers.get("X-OpenCut-Token");
    if (newToken) state.csrfToken = newToken;

    let data;
    const contentType = resp.headers.get("Content-Type") || "";
    if (contentType.includes("application/json")) {
      try {
        data = await resp.json();
      } catch (_) {
        data = null;
      }
    } else {
      data = await resp.text();
    }
    return { resp, data };
  }

  async function call(method, endpoint, body = null) {
    try {
      // The panel's own health polls get the secret too, not just fetchCsrf.
      if (endpoint === "/health" && !bootstrapLoaded) await loadBootstrapSecret();
      let { resp, data } = await doFetch(method, endpoint, body);
      if (resp.status === 403 && endpoint !== "/health") {
        await fetchCsrf();
        ({ resp, data } = await doFetch(method, endpoint, body));
      }
      if (!resp.ok) {
        return {
          ok: false,
          error: data?.error || `HTTP ${resp.status}`,
          status: resp.status,
          data,
        };
      }
      return { ok: true, data, status: resp.status };
    } catch (err) {
      const fallback = isTimeoutError(err)
        ? "Backend request timed out. Check that OpenCut Server is still running, then try again."
        : "Network error";
      return { ok: false, error: err?.message ?? fallback };
    }
  }

  const get = (endpoint) => call("GET", endpoint);
  const post = (endpoint, body) => call("POST", endpoint, body);
  const del = (endpoint) => call("DELETE", endpoint);

  async function checkHealth() {
    const result = await get("/health");
    if (result.ok && result.data?.capabilities) {
      capabilities = result.data.capabilities;
      onCapabilities(capabilities);
    }
    return result.ok;
  }

  async function fetchCsrf() {
    await loadBootstrapSecret();
    let result = await get("/health");
    if (result.ok && !result.data?.csrf_token) {
      // A refused /health makes the server refresh the panel's copy (the data
      // folder may not have existed when it started), so read it once more.
      const previous = bootstrapSecret;
      await loadBootstrapSecret(true);
      if (bootstrapSecret && bootstrapSecret !== previous) result = await get("/health");
    }
    if (result.ok && result.data?.csrf_token) state.csrfToken = result.data.csrf_token;
    return state.csrfToken;
  }

  return {
    call,
    get,
    post,
    del,
    checkHealth,
    fetchCsrf,
    getCapabilities: () => capabilities,
  };
}
