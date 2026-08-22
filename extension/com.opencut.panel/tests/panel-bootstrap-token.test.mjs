import { createRequire } from "node:module";

import { describe, expect, it, vi } from "vitest";

const require = createRequire(import.meta.url);
const { createBootstrapTokenLoader } = require("../client/panel-bootstrap-token.js");
const { createBackendClient } = require("../client/backend-client.js");

/* The panel builds its loader at script-eval time, when the CEP bridge does not
 * exist yet. F389: the option used to be a wrapper captured against a null
 * `cs`, which pinned `evalScript` to null for the whole session. These tests
 * pin the late-binding contract that replaced it. */
function lateBoundBridge() {
  const bridge = { cs: null };
  // Mirrors main.js jsx(): resolve the host on call, never at capture.
  bridge.evalScript = (script, cb) => {
    if (!bridge.cs || typeof bridge.cs.evalScript !== "function") { cb(null); return; }
    bridge.cs.evalScript(script, cb);
  };
  return bridge;
}

function hostReturning(payload) {
  return { evalScript: (_script, cb) => cb(JSON.stringify(payload)) };
}

describe("panel bootstrap token loader", () => {
  it("reads the secret through a bridge that only appears after construction", () => {
    const bridge = lateBoundBridge();
    const onReady = vi.fn();
    const loader = createBootstrapTokenLoader({ evalScript: bridge.evalScript, onReady });

    // Construction-time state: no host, so a captured wrapper would be dead.
    expect(loader.get()).toBe("");

    const first = vi.fn();
    loader.load(first);
    expect(first).toHaveBeenCalledWith(false);
    expect(loader.get()).toBe("");

    bridge.cs = hostReturning({ ok: true, token: "secret-abc" });

    const second = vi.fn();
    loader.load(second);
    expect(second).toHaveBeenCalledWith(true);
    expect(loader.get()).toBe("secret-abc");
    expect(onReady).toHaveBeenCalledWith("secret-abc");
  });

  it("warns instead of throwing when the host bridge is absent", () => {
    const onWarn = vi.fn();
    const loader = createBootstrapTokenLoader({ evalScript: null, onWarn });

    const done = vi.fn();
    loader.load(done);

    expect(done).toHaveBeenCalledWith(false);
    expect(onWarn).toHaveBeenCalledWith(expect.stringContaining("Host bridge unavailable"));
  });

  it("reload reports true only when the re-read produced a different secret", () => {
    const bridge = lateBoundBridge();
    const loader = createBootstrapTokenLoader({ evalScript: bridge.evalScript });

    // Backend has not written the file yet.
    const cold = vi.fn();
    loader.reload(cold);
    expect(cold).toHaveBeenCalledWith(false);

    bridge.cs = hostReturning({ ok: true, token: "secret-abc" });
    const warm = vi.fn();
    loader.reload(warm);
    expect(warm).toHaveBeenCalledWith(true);

    // Same secret again: a caller must not keep retrying on it.
    const repeat = vi.fn();
    loader.reload(repeat);
    expect(repeat).toHaveBeenCalledWith(false);
  });
});

/* Minimal XHR double: enough for refreshCsrfToken's header + onload path. */
function fakeXhrFactory(responder) {
  const sent = [];
  const factory = () => {
    const xhr = {
      headers: {},
      status: 200,
      responseText: "{}",
      open() {},
      setRequestHeader(name, value) { this.headers[name] = value; },
      send() {
        sent.push({ headers: this.headers });
        const reply = responder(sent.length, this.headers);
        this.status = reply.status;
        this.responseText = reply.body;
        this.onload();
      },
    };
    return xhr;
  };
  factory.sent = sent;
  return factory;
}

describe("backend client CSRF bootstrap", () => {
  it("presents the bootstrap secret on /health once the loader holds one", () => {
    const createRequestFactory = fakeXhrFactory(() => ({
      status: 200,
      body: JSON.stringify({ status: "ok", csrf_token: "csrf-1" }),
    }));
    const setToken = vi.fn();
    const client = createBackendClient({
      getBaseUrl: () => "http://127.0.0.1:5679",
      getBootstrapToken: () => "secret-abc",
      setToken,
      createRequest: createRequestFactory,
    });

    const done = vi.fn();
    client.refreshCsrfToken(done);

    expect(createRequestFactory.sent[0].headers["X-OpenCut-Panel-Bootstrap"]).toBe("secret-abc");
    expect(setToken).toHaveBeenCalledWith("csrf-1");
    expect(done).toHaveBeenCalledWith(true);
  });

  it("re-reads the secret and retries once when /health withholds the token", () => {
    let secret = "";
    const createRequestFactory = fakeXhrFactory((attempt, headers) => {
      // The backend only hands over a token to a request carrying the secret.
      const body = headers["X-OpenCut-Panel-Bootstrap"]
        ? JSON.stringify({ status: "ok", csrf_token: "csrf-1" })
        : JSON.stringify({ status: "ok" });
      return { status: 200, body };
    });
    const setToken = vi.fn();
    const client = createBackendClient({
      getBaseUrl: () => "http://127.0.0.1:5679",
      getBootstrapToken: () => secret,
      // The backend wrote the file between the panel loading and this refusal.
      onBootstrapRefused: (report) => { secret = "secret-abc"; report(true); },
      setToken,
      createRequest: createRequestFactory,
      onTransportError: () => {},
    });

    const done = vi.fn();
    client.refreshCsrfToken(done);

    expect(createRequestFactory.sent).toHaveLength(2);
    expect(createRequestFactory.sent[0].headers["X-OpenCut-Panel-Bootstrap"]).toBeUndefined();
    expect(createRequestFactory.sent[1].headers["X-OpenCut-Panel-Bootstrap"]).toBe("secret-abc");
    expect(setToken).toHaveBeenCalledWith("csrf-1");
    expect(done).toHaveBeenCalledWith(true);
  });

  it("gives up after one retry rather than looping on a stale secret", () => {
    const createRequestFactory = fakeXhrFactory(() => ({
      status: 200,
      body: JSON.stringify({ status: "ok" }),
    }));
    const onBootstrapRefused = vi.fn((report) => report(true));
    const client = createBackendClient({
      getBaseUrl: () => "http://127.0.0.1:5679",
      getBootstrapToken: () => "stale",
      onBootstrapRefused,
      createRequest: createRequestFactory,
      onTransportError: () => {},
    });

    const done = vi.fn();
    client.refreshCsrfToken(done);

    expect(createRequestFactory.sent).toHaveLength(2);
    expect(onBootstrapRefused).toHaveBeenCalledTimes(1);
    expect(done).toHaveBeenCalledWith(false);
  });

  it("does not retry when the re-read found nothing new", () => {
    const createRequestFactory = fakeXhrFactory(() => ({
      status: 200,
      body: JSON.stringify({ status: "ok" }),
    }));
    const client = createBackendClient({
      getBaseUrl: () => "http://127.0.0.1:5679",
      getBootstrapToken: () => "",
      onBootstrapRefused: (report) => report(false),
      createRequest: createRequestFactory,
      onTransportError: () => {},
    });

    const done = vi.fn();
    client.refreshCsrfToken(done);

    expect(createRequestFactory.sent).toHaveLength(1);
    expect(done).toHaveBeenCalledWith(false);
  });
});
