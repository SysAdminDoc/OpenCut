import { readFileSync } from "node:fs";

import { describe, expect, it, vi } from "vitest";

import { createBackendClient } from "../../com.opencut.uxp/backend-client.js";
import { bootstrapApplication, runBootstrapSteps } from "../../com.opencut.uxp/uxp-bootstrap.js";
import { escapeHtml, safeDomIdSegment, setButtonBusy } from "../../com.opencut.uxp/uxp-components.js";
import { createI18nRuntime } from "../../com.opencut.uxp/uxp-i18n.js";
import { createJobController } from "../../com.opencut.uxp/job-controller.js";
import { createUxpState } from "../../com.opencut.uxp/uxp-state.js";
import {
  applyPremiereTheme,
  createPremiereThemeSync,
  normalizePremiereTheme,
} from "../../com.opencut.uxp/uxp-theme.js";
import {
  buildMarkerPayload,
  computeInverseRenames,
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

describe("UXP host theme boundary", () => {
  function themeDocument(initial = "darkest") {
    const classes = new Set();
    const listeners = new Set();
    const root = {
      classList: {
        add: (...values) => values.forEach((value) => classes.add(value)),
        remove: (...values) => values.forEach((value) => classes.delete(value)),
        contains: (value) => classes.has(value),
      },
      dataset: {},
    };
    return {
      classes,
      listeners,
      documentRef: {
        documentElement: root,
        theme: {
          getCurrent: vi.fn(() => initial),
          onUpdated: {
            addListener: vi.fn((listener) => listeners.add(listener)),
            removeListener: vi.fn((listener) => listeners.delete(listener)),
          },
        },
      },
    };
  }

  it("normalizes Premiere Light, Dark, and Darkest values", () => {
    expect(normalizePremiereTheme("light")).toBe("light");
    expect(normalizePremiereTheme("Dark")).toBe("dark");
    expect(normalizePremiereTheme("darkest")).toBe("darkest");
    expect(normalizePremiereTheme("unknown")).toBe("darkest");
  });

  it("applies exactly one root theme class", () => {
    const harness = themeDocument();
    applyPremiereTheme("dark", harness.documentRef);
    expect([...harness.classes]).toEqual(["theme-dark"]);
    applyPremiereTheme("light", harness.documentRef);
    expect([...harness.classes]).toEqual(["theme-light"]);
    expect(harness.documentRef.documentElement.dataset.premiereTheme).toBe("light");
  });

  it("tracks live host changes and unregisters on teardown", () => {
    const harness = themeDocument("dark");
    const runtime = createPremiereThemeSync({ documentRef: harness.documentRef });
    const dispose = runtime.start();

    expect(runtime.currentTheme).toBe("dark");
    expect(harness.documentRef.theme.onUpdated.addListener).toHaveBeenCalledOnce();
    expect(harness.listeners.size).toBe(1);
    [...harness.listeners][0]("light");
    expect(runtime.currentTheme).toBe("light");
    expect([...harness.classes]).toEqual(["theme-light"]);

    dispose();
    expect(harness.documentRef.theme.onUpdated.removeListener).toHaveBeenCalledOnce();
    expect(harness.listeners.size).toBe(0);
  });
});

describe("UXP journal restore contract", () => {
  const mainSource = () =>
    readFileSync(new URL("../../com.opencut.uxp/main.js", import.meta.url), "utf8");

  it("stores marker fingerprints with both name and comment labels", () => {
    // The UXP bridge writes labels to the marker NAME (setName) while the
    // CEP host writes COMMENTS; the checkpoint fingerprint must carry both
    // so either panel can match markers written by the other.
    const source = mainSource();
    const start = source.indexOf("const inverseMarkers");
    expect(start).toBeGreaterThan(-1);
    const slice = source.slice(start, start + 700);
    expect(slice).toContain("name: marker.label");
    expect(slice).toContain("comment: marker.label");
  });

  it("matches marker fingerprints against either the marker name or comment", () => {
    const source = mainSource();
    const start = source.indexOf("function _markerMatches");
    const end = source.indexOf("async function removeSequenceMarkers", start);
    const slice = source.slice(start, end);
    expect(slice).toContain("fingerprint.name, fingerprint.label, fingerprint.comment");
    expect(slice).toContain("[info.name, info.comment]");
  });

  it("treats a zero-removal marker restore as failed instead of recovered", () => {
    const source = mainSource();
    const start = source.indexOf("async function recoverJournalCheckpointUxp");
    const end = source.indexOf("async function copyJournalDiagnosticsUxp", start);
    const slice = source.slice(start, end);
    expect(slice).toContain("fingerprints.length > 0 && Number(result.removed || 0) === 0");
    expect(slice).toContain("uxp.journal.restore_markers_missing");
    // The failure path posts /recovery-failed; only verified restores may
    // reach the /recovered mark further down.
    expect(slice.indexOf("/recovery-failed")).toBeGreaterThan(-1);
    expect(slice.indexOf("/recovery-failed")).toBeLessThan(slice.indexOf("/recovered`"));
    // batch_rename restores consume the canonical inverse shape.
    expect(slice).toContain("renamesFromCanonicalInverse(inverse.renames");
  });

  it("computes canonical inverse renames and converts them for the host bridge", () => {
    expect(
      computeInverseRenames([
        { oldName: "a.mp4", newName: "a_001.mp4", path: "/p/a.mp4", nodeId: "9" },
      ]),
    ).toEqual([{ nodeId: "9", path: "/p/a.mp4", oldName: "a.mp4", currentName: "a_001.mp4" }]);

    const source = mainSource();
    const start = source.indexOf("function renamesFromCanonicalInverse");
    expect(start).toBeGreaterThan(-1);
    // Shape heuristic: canonical entries carry currentName; legacy entries
    // ({oldName: applied, newName: original}) pass through unchanged.
    const slice = source.slice(start, start + 900);
    expect(slice).toContain("item.currentName != null");
    expect(slice).toContain("oldName: item.currentName, newName: item.oldName");
  });

  it("re-runs connect-time loaders on reconnect without stacking OTIO listeners", () => {
    const source = mainSource();
    const start = source.indexOf("if (alive && wasAlive === false)");
    expect(start).toBeGreaterThan(-1);
    const slice = source.slice(start, start + 700);
    expect(slice).toContain("loadOtioCapabilities()");
    expect(slice).toContain("loadJournalRecoveryUxp()");
    expect(source).toContain("_otioAdapterListenerBound");
    expect(source.match(/_otioAdapterListenerBound = true/g)).toHaveLength(1);
  });
});

describe("UXP source ownership", () => {
  it("checkpoints direct UXP host writes and exposes restart recovery", () => {
    const main = readFileSync(new URL("../../com.opencut.uxp/main.js", import.meta.url), "utf8");
    const index = readFileSync(new URL("../../com.opencut.uxp/index.html", import.meta.url), "utf8");
    for (const name of ["applyTimelineCuts", "addSequenceMarkers", "runBatchRename", "runSmartBins"]) {
      const start = main.indexOf(`function ${name}`);
      const next = main.indexOf("\nasync function ", start + 1);
      const source = main.slice(start, next > start ? next : start + 5000);
      expect(source, name).toContain("runCheckpointedUxpHostWrite");
    }
    expect(main).toContain('BackendClient.post("/journal/checkpoints"');
    expect(main).toContain("loadJournalRecoveryUxp");
    expect(index).toContain('id="uxpRecoveryList"');
    expect(index).toContain('id="uxpRefreshRecoveryBtn"');
  });

  it("keeps extracted runtime implementations out of main.js", () => {
    const main = readFileSync(new URL("../../com.opencut.uxp/main.js", import.meta.url), "utf8");
    expect(main).toContain("createBackendClient");
    expect(main).toContain("createJobController");
    expect(main).toContain("bootstrapApplication");
    expect(main).toContain("createPremiereThemeSync");
    expect(main).not.toContain("class BackendClient");
    expect(main).not.toContain("class JobPoller");
    expect(main).not.toContain("const I18n = {");
    expect(main).not.toContain("function expandRenamePattern(");
    expect(main).not.toContain("function buildSmartBinRules(");
  });

  it("keeps token and workspace layout rules outside component CSS", () => {
    const root = "../../com.opencut.uxp/";
    const index = readFileSync(new URL(`${root}index.html`, import.meta.url), "utf8");
    const tokens = readFileSync(new URL(`${root}command-center-tokens.css`, import.meta.url), "utf8");
    const layout = readFileSync(new URL(`${root}command-center-layout.css`, import.meta.url), "utf8");
    const components = readFileSync(new URL(`${root}command-center.css`, import.meta.url), "utf8");
    expect(tokens).toContain(":root,");
    expect(tokens).toContain("html.theme-darkest");
    expect(tokens).toContain("html.theme-dark");
    expect(tokens).toContain("html.theme-light");
    expect(layout).toContain(".oc-header {");
    expect(components).not.toContain(":root {");
    expect(index.indexOf("command-center-tokens.css")).toBeLessThan(index.indexOf("command-center-layout.css"));
    expect(index.indexOf("command-center-layout.css")).toBeLessThan(index.indexOf("command-center.css"));
  });
});
