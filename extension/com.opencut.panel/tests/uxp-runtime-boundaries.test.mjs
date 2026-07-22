import { describe, expect, it, vi } from "vitest";

import { createBackendClient } from "../../com.opencut.uxp/backend-client.js";
import { bootstrapApplication, runBootstrapSteps } from "../../com.opencut.uxp/uxp-bootstrap.js";
import { escapeHtml, safeDomIdSegment, setButtonBusy } from "../../com.opencut.uxp/uxp-components.js";
import { createI18nRuntime } from "../../com.opencut.uxp/uxp-i18n.js";
import { createJobController } from "../../com.opencut.uxp/job-controller.js";
import { createUxpState } from "../../com.opencut.uxp/uxp-state.js";
import {
  buildMarkerPayload,
  expandRenamePattern,
  normalizeTimelineCuts,
} from "../../com.opencut.uxp/uxp-timeline.js";

function response(status, data, headers = {}) {
  const normalized = new Map(Object.entries({ "Content-Type": "application/json", ...headers }));
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: { get: (name) => normalized.get(name) || null },
    json: async () => data,
    text: async () => JSON.stringify(data),
  };
}

describe("UXP runtime state", () => {
  it("owns job transitions without permitting overlap", () => {
    const state = createUxpState();
    expect(state.markJobStarting()).toBe(true);
    expect(state.markJobStarting()).toBe(false);
    state.trackJob("job-1");
    expect(state.hasActiveJob()).toBe(true);
    expect(state.clearJob("other-job")).toBe(false);
    expect(state.clearJob("job-1")).toBe(true);
  });

  it("returns the previous SSE handle when replacing it", () => {
    const state = createUxpState();
    const first = { close: vi.fn() };
    expect(state.replaceSse(first)).toBe(null);
    expect(state.replaceSse(null)).toBe(first);
  });
});

describe("UXP backend client", () => {
  it("refreshes a stale CSRF token and retries once", async () => {
    const state = createUxpState({ backendUrl: "http://127.0.0.1:5679", csrfToken: "stale" });
    const fetchWithTimeout = vi.fn()
      .mockResolvedValueOnce(response(403, { error: "stale token" }))
      .mockResolvedValueOnce(response(200, { status: "ok", csrf_token: "fresh" }))
      .mockResolvedValueOnce(response(200, { saved: true }));
    const client = createBackendClient({ state, fetchWithTimeout });

    const result = await client.post("/settings/test", { enabled: true });

    expect(result).toEqual({ ok: true, data: { saved: true }, status: 200 });
    expect(state.csrfToken).toBe("fresh");
    expect(fetchWithTimeout.mock.calls[2][1].headers["X-OpenCut-Token"]).toBe("fresh");
  });

  it("normalizes invalid JSON and HTTP errors", async () => {
    const state = createUxpState();
    const invalid = response(500, null);
    invalid.json = async () => { throw new Error("bad json"); };
    const client = createBackendClient({ state, fetchWithTimeout: async () => invalid });
    expect(await client.get("/broken")).toEqual({
      ok: false,
      error: "HTTP 500",
      status: 500,
      data: null,
    });
  });
});

describe("UXP job controller", () => {
  it("handles synchronous job responses and unlocks the UI", async () => {
    const state = createUxpState();
    const locked = [];
    const progress = vi.fn();
    const complete = vi.fn();
    const controller = createJobController({
      state,
      client: { post: async () => ({ ok: true, data: { output: "done.mp4" } }) },
      setLocked: (value) => locked.push(value),
    });

    await controller.start("/render", {}, progress, complete, vi.fn());

    expect(progress).toHaveBeenCalledWith(100, "Done");
    expect(complete).toHaveBeenCalledWith({ output: "done.mp4" });
    expect(locked).toEqual([true, false]);
    expect(state.hasActiveJob()).toBe(false);
  });

  it("rejects a second submission while a job is active", async () => {
    const state = createUxpState({ activeJobId: "job-1" });
    const error = vi.fn();
    const controller = createJobController({ state, client: {} });
    await controller.start("/render", {}, vi.fn(), vi.fn(), error);
    expect(error).toHaveBeenCalledWith("Another OpenCut job is already running.");
  });
});

describe("UXP i18n runtime", () => {
  it("merges the requested locale over English and interpolates values", async () => {
    const locales = {
      "locales/en.json": { greeting: "Hello {name}", shared: "English" },
      "locales/es.json": { greeting: "Hola {name}" },
    };
    const runtime = createI18nRuntime({
      fetchJson: async (path) => locales[path] || null,
      documentRef: null,
      navigatorRef: { languages: ["es-MX"] },
    });
    await runtime.load();
    expect(runtime.currentLang).toBe("es");
    expect(runtime.format("greeting", "fallback", { name: "Ava" })).toBe("Hola Ava");
    expect(runtime.t("shared", "fallback")).toBe("English");
  });
});

describe("UXP timeline and component boundaries", () => {
  it("normalizes cut ordering and marker payloads", () => {
    expect(normalizeTimelineCuts([
      { start: "1", end: "2" },
      { start: 4, end: 3 },
      { start: 5, end: 7 },
    ])).toEqual([{ start: 5, end: 7 }, { start: 1, end: 2 }]);
    expect(buildMarkerPayload([{ seconds: "2.5" }])).toEqual([
      { time: 2.5, name: "Marker 1", type: "Comment" },
    ]);
  });

  it("keeps the host rename contract stable", () => {
    expect(expandRenamePattern([{ name: "take.mov", nodeId: "7" }], "{stem}_{index:02d}{ext}"))
      .toEqual([{ oldName: "take.mov", newName: "take_01.mov", path: "", nodeId: "7" }]);
  });

  it("provides XSS-safe DOM identifiers and button state", () => {
    expect(escapeHtml('<img onerror="x">')).toBe("&lt;img onerror=&quot;x&quot;&gt;");
    expect(safeDomIdSegment("Hello / World")).toBe("hello-world");
    const label = { textContent: "Run" };
    const button = {
      dataset: {},
      classList: { toggle: vi.fn() },
      querySelector: () => label,
      setAttribute: vi.fn(),
      disabled: false,
    };
    expect(setButtonBusy(button, true, "Working")).toBe("Working");
    expect(setButtonBusy(button, false)).toBe("Run");
  });
});

describe("UXP bootstrap boundary", () => {
  it("runs named steps in order and reports the failing boundary", async () => {
    const order = [];
    await runBootstrapSteps([
      { name: "locale", run: async () => order.push("locale") },
      { name: "events", run: async () => order.push("events") },
    ]);
    expect(order).toEqual(["locale", "events"]);

    const failure = new Error("offline");
    await expect(runBootstrapSteps([{ name: "backend", run: async () => { throw failure; } }]))
      .rejects.toMatchObject({ bootstrapStep: "backend" });
  });

  it("contains bootstrap failures at the entrypoint", async () => {
    const onError = vi.fn();
    await expect(bootstrapApplication(async () => { throw new Error("boom"); }, onError))
      .resolves.toBeNull();
    expect(onError).toHaveBeenCalledOnce();
  });
});
