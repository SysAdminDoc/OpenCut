import { isTimeoutError as defaultIsTimeoutError } from "./uxp-utils.js";

/** Create the UXP HTTP boundary with CSRF refresh and one-retry semantics. */
export function createBackendClient({
  state,
  fetchWithTimeout,
  isTimeoutError = defaultIsTimeoutError,
  onCapabilities = () => {},
  requestTimeoutMs = 120000,
} = {}) {
  if (!state) throw new TypeError("createBackendClient requires a state store");
  if (typeof fetchWithTimeout !== "function") {
    throw new TypeError("createBackendClient requires fetchWithTimeout");
  }

  let capabilities = {};

  async function doFetch(method, endpoint, body) {
    const headers = { "Content-Type": "application/json" };
    if (state.csrfToken) headers["X-OpenCut-Token"] = state.csrfToken;
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
    const result = await get("/health");
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
