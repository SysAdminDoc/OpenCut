import { readFileSync } from "node:fs";

import { describe, expect, it, vi } from "vitest";

import { createBackendClient } from "../../com.opencut.uxp/backend-client.js";
import { bootstrapApplication, runBootstrapSteps } from "../../com.opencut.uxp/uxp-bootstrap.js";
import { escapeHtml, safeDomIdSegment, setButtonBusy } from "../../com.opencut.uxp/uxp-components.js";
import { createI18nRuntime } from "../../com.opencut.uxp/uxp-i18n.js";
import { createJobController } from "../../com.opencut.uxp/job-controller.js";
import { createUxpState } from "../../com.opencut.uxp/uxp-state.js";
import { createUxpUiController } from "../../com.opencut.uxp/uxp-ui-controller.js";
import { createUxpUpdateController } from "../../com.opencut.uxp/uxp-update-controller.js";
import { createUxpSettingsController } from "../../com.opencut.uxp/uxp-settings-controller.js";
import { createSequenceIndexFilterController } from "../../com.opencut.uxp/uxp-sequence-index-controller.js";
import {
  buildChatActionRequest,
  buildLoudnessMatchPayload,
} from "../../com.opencut.uxp/uxp-utils.js";
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

function fakeElement({ id = "", tab = "", classes = [] } = {}) {
  const listeners = new Map();
  const classNames = new Set(classes);
  const element = {
    id,
    dataset: tab ? { tab } : {},
    classList: {
      add: (...names) => names.forEach((name) => classNames.add(name)),
      remove: (...names) => names.forEach((name) => classNames.delete(name)),
      contains: (name) => classNames.has(name),
      toggle: (name, force) => {
        const next = force === undefined ? !classNames.has(name) : force;
        if (next) classNames.add(name);
        else classNames.delete(name);
        return next;
      },
    },
    style: {},
    children: [],
    hidden: false,
    disabled: false,
    scrollLeft: 0,
    scrollWidth: 0,
    clientWidth: 0,
    focus: vi.fn(),
    setAttribute: vi.fn((name, value) => { element[name] = value; }),
    addEventListener: vi.fn((type, listener) => {
      const handlers = listeners.get(type) || [];
      handlers.push(listener);
      listeners.set(type, handlers);
    }),
    removeEventListener: vi.fn((type, listener) => {
      const handlers = listeners.get(type) || [];
      listeners.set(type, handlers.filter((candidate) => candidate !== listener));
    }),
    dispatch: (type, event = {}) => {
      (listeners.get(type) || []).forEach((listener) => listener({ currentTarget: element, preventDefault: vi.fn(), ...event }));
    },
    getBoundingClientRect: () => ({ left: 0, right: 100 }),
    querySelectorAll: (selector) => selector === ".oc-tab" ? element.children : [],
    querySelector: () => null,
    appendChild: (child) => {
      child.parentNode = element;
      element.children.push(child);
    },
    remove: vi.fn(),
  };
  Object.defineProperty(element, "firstElementChild", { get: () => element.children[0] || null });
  element.dataset = { ...element.dataset };
  return element;
}

function fakeUiDocument() {
  const tabs = [
    fakeElement({ tab: "cut", classes: ["oc-tab", "active"] }),
    fakeElement({ tab: "audio", classes: ["oc-tab"] }),
  ];
  const panels = [
    fakeElement({ id: "tab-cut", classes: ["oc-tab-panel", "active"] }),
    fakeElement({ id: "tab-audio", classes: ["oc-tab-panel"] }),
  ];
  const nav = fakeElement({ id: "tabNav" });
  nav.children.push(...tabs);
  nav.scrollWidth = 200;
  nav.clientWidth = 100;
  const ids = new Map([
    ["tabNav", nav],
    ["tabNavShell", fakeElement({ id: "tabNavShell" })],
    ["tabScrollPrev", fakeElement({ id: "tabScrollPrev" })],
    ["tabScrollNext", fakeElement({ id: "tabScrollNext" })],
    ["mainContent", fakeElement({ id: "mainContent" })],
    ["processingBanner", fakeElement({ id: "processingBanner" })],
    ["processingMsg", fakeElement({ id: "processingMsg" })],
    ["progressFill", fakeElement({ id: "progressFill" })],
    ["processingElapsed", fakeElement({ id: "processingElapsed" })],
    ["statusText", fakeElement({ id: "statusText" })],
    ["statusBar", fakeElement({ id: "statusBar" })],
    ["connDot", fakeElement({ id: "connDot" })],
    ["connLabel", fakeElement({ id: "connLabel" })],
    ["connectionStatus", fakeElement({ id: "connectionStatus" })],
  ]);
  return {
    tabs,
    panels,
    nav,
    ids,
    documentRef: {
      getElementById: (id) => ids.get(id) || null,
      querySelectorAll: (selector) => selector === ".oc-tab" ? tabs : selector === ".oc-tab-panel" ? panels : [],
      createElement: () => fakeElement(),
    },
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

describe("UXP feature control contracts", () => {
  it("keeps only the latest Sequence Index filter response", async () => {
    const pending = [];
    const rows = [];
    const busy = [];
    const controller = createSequenceIndexFilterController({
      request: vi.fn(() => new Promise((resolve) => pending.push(resolve))),
      onRows: (value) => rows.push(value),
      setBusy: (value) => busy.push(value),
    });

    const first = controller.apply({ query: "old" });
    const second = controller.apply({ query: "new" });
    pending[1]({ ok: true, data: { rows: ["new-result"] } });
    await second;
    pending[0]({ ok: true, data: { rows: ["old-result"] } });
    await first;

    expect(rows).toEqual([["new-result"]]);
    expect(busy).toEqual([true, true, false]);
  });

  it("builds allowlisted chat requests with the active clip", () => {
    expect(buildChatActionRequest({
      action: "/audio/normalize",
      params: { preset: "youtube" },
    }, "C:/clips/interview.mp4")).toEqual({
      endpoint: "/audio/normalize",
      payload: { preset: "youtube", filepath: "C:/clips/interview.mp4" },
    });
    expect(buildChatActionRequest({ action: "/settings/delete-all", params: {} }, "clip.mp4")).toBeNull();
  });

  it("normalizes only the input clip to the measured reference target", () => {
    expect(buildLoudnessMatchPayload("C:/clips/input.wav", -18.25)).toEqual({
      files: ["C:/clips/input.wav"],
      target_lufs: -18.25,
    });
    expect(buildLoudnessMatchPayload("C:/clips/input.wav", "not-a-number")).toBeNull();
  });

  it("removes unsupported auto-zoom aspect state and keeps OTIO fallback aligned", () => {
    const main = readFileSync(new URL("../../com.opencut.uxp/main.js", import.meta.url), "utf8");
    const index = readFileSync(new URL("../../com.opencut.uxp/index.html", import.meta.url), "utf8");
    expect(main).not.toContain('getElementById("zoomAspect")');
    expect(index).not.toContain('id="zoomAspect"');

    const otioStart = main.indexOf('document.getElementById("exportOtioBtn")?.addEventListener');
    const otioSource = main.slice(otioStart, otioStart + 1800);
    expect(otioSource).toContain('document.getElementById("clipPathCut")?.value?.trim()');
    expect(otioSource).toContain('|| document.getElementById("clipPathVideo")?.value?.trim()');
    expect(otioSource).not.toContain('?? document.getElementById("clipPathVideo")');
  });

  it("dispatches supported chat actions through the job controller", () => {
    const main = readFileSync(new URL("../../com.opencut.uxp/main.js", import.meta.url), "utf8");
    expect(main).toContain("void executeChatActions(actions, clipPath);");
    const start = main.indexOf("async function executeChatActions");
    const source = main.slice(start, start + 1800);
    expect(source).toContain("JobPoller.start(");
    expect(source).toContain("request.endpoint");
    expect(source).toContain("request.payload");
  });

  it("keeps first-run, support export, and portability seams wired", () => {
    const main = readFileSync(new URL("../../com.opencut.uxp/main.js", import.meta.url), "utf8");
    const settings = readFileSync(new URL("../../com.opencut.uxp/uxp-settings-controller.js", import.meta.url), "utf8");
    const index = readFileSync(new URL("../../com.opencut.uxp/index.html", import.meta.url), "utf8");
    expect(main).toContain('import { createUxpSettingsController } from "./uxp-settings-controller.js";');
    expect(main).toContain("UxpSettingsController.initOnboardingEvents();");
    expect(main).toContain("UxpSettingsController.dispose();");
    expect(settings).toContain('BackendClient.post("/system/support-bundle"');
    expect(settings).toContain('BackendClient.post("/settings/onboarding"');
    expect(settings).toContain('BackendClient.get("/settings/onboarding")');
    expect(settings).toContain("opencut.support_bundle.v1");
    expect(settings).toContain("function dispose()");
    expect(main).toContain("getUxpLocalFileSystem");
    expect(index).toContain('id="uxpExportSupportBundleBtn"');
    expect(index).toContain('id="uxpOnboardingOverlay"');
    expect(index).toContain('id="uxpRestartOnboardingBtn"');
  });
});

describe("UXP UI controller", () => {
  it("owns navigation listeners and clears processing resources on dispose", () => {
    const { documentRef, tabs, panels } = fakeUiDocument();
    const windowRef = fakeElement();
    const intervalHandle = {};
    const setIntervalFn = vi.fn(() => intervalHandle);
    const clearIntervalFn = vi.fn();
    const invalidateCache = vi.fn();
    const workspaceChanged = vi.fn();
    const observer = { observe: vi.fn(), disconnect: vi.fn() };
    const controller = createUxpUiController({
      documentRef,
      windowRef,
      requestAnimationFrameFn: (callback) => callback(),
      setIntervalFn,
      clearIntervalFn,
      ResizeObserverCtor: class {
        constructor() {
          return observer;
        }
      },
      isBackendConnected: () => false,
      onInvalidatePProCache: invalidateCache,
      onWorkspaceTabChange: workspaceChanged,
    });

    controller.bindNavigation();
    tabs[1].dispatch("click");
    expect(tabs[1].classList.contains("active")).toBe(true);
    expect(panels[1].hidden).toBe(false);
    expect(invalidateCache).toHaveBeenCalledOnce();
    expect(workspaceChanged).toHaveBeenCalledWith("audio");

    controller.showProcessing("Working");
    expect(setIntervalFn).toHaveBeenCalledWith(expect.any(Function), 1000);
    expect(documentRef.getElementById("processingMsg").textContent).toBe("Working");
    controller.dispose();
    expect(clearIntervalFn).toHaveBeenCalledWith(intervalHandle);
    expect(observer.disconnect).toHaveBeenCalledOnce();
    expect(windowRef.removeEventListener).toHaveBeenCalled();
  });

  it("keeps the controller implementation and teardown in the extracted module", () => {
    const main = readFileSync(new URL("../../com.opencut.uxp/main.js", import.meta.url), "utf8");
    const ui = readFileSync(new URL("../../com.opencut.uxp/uxp-ui-controller.js", import.meta.url), "utf8");
    expect(main).toContain('import { createUxpUiController } from "./uxp-ui-controller.js";');
    expect(main).toContain("UIController.bindNavigation();");
    expect(main).toContain("UIController.dispose();");
    expect(main).not.toContain("const UIController = (() =>");
    expect(ui).toContain("export function createUxpUiController");
    expect(ui).toContain("function dispose()");
  });

  it("closes the UXP WebSocket and reconnect timer during unload", () => {
    const main = readFileSync(new URL("../../com.opencut.uxp/main.js", import.meta.url), "utf8");
    const unloadStart = main.indexOf('window.addEventListener("beforeunload"');
    const unloadEnd = main.indexOf("}, { once: true });", unloadStart);
    const unload = main.slice(unloadStart, unloadEnd);
    const disconnectStart = main.indexOf("function uxpWsDisconnect()");
    const disconnectEnd = main.indexOf("async function uxpUpdateWsStatus", disconnectStart);
    const disconnect = main.slice(disconnectStart, disconnectEnd);

    expect(unload).toContain("JobPoller.closeSse();");
    expect(unload).toContain("uxpWsDisconnect();");
    expect(disconnect).toContain("uxpWsClearReconnectTimer();");
    expect(disconnect).toContain("_uxpWs = null;");
    expect(disconnect).toContain("socket.close();");
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

  it("ignores a status response that resolves after cancellation", async () => {
    const state = createUxpState();
    let resolveStatus;
    const client = {
      get: vi.fn(() => new Promise((resolve) => { resolveStatus = resolve; })),
      post: vi.fn(async () => ({ ok: true, data: {} })),
    };
    const hook = vi.fn();
    const locked = [];
    const controller = createJobController({
      state,
      client,
      setLocked: (value) => locked.push(value),
    });
    controller.onJobFinished(hook);

    let settled = false;
    const pollPromise = controller.poll("job-1");
    void pollPromise.then(
      () => { settled = true; },
      () => { settled = true; },
    );
    await vi.waitFor(() => expect(client.get).toHaveBeenCalledOnce());
    await controller.cancel();

    resolveStatus({
      ok: true,
      data: { status: "complete", result: { output: "late.mp4" } },
    });
    await Promise.resolve();
    await Promise.resolve();

    expect(settled).toBe(false);
    expect(hook).toHaveBeenCalledOnce();
    expect(client.post).toHaveBeenCalledWith("/cancel/job-1", {});
    expect(locked).toEqual([true, false]);
    expect(state.hasActiveJob()).toBe(false);
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

describe("UXP update-check boundary", () => {
  it("renders a failed check, recovers on retry, and opens only the normalized release", async () => {
    const ids = new Map([
      ["uxpUpdateNoticeCard", fakeElement()],
      ["uxpUpdateStatusText", fakeElement()],
      ["uxpUpdateSummary", fakeElement()],
      ["uxpUpdateCurrentVersion", fakeElement()],
      ["uxpUpdateAvailableVersion", fakeElement()],
      ["uxpUpdateReleaseName", fakeElement()],
      ["uxpUpdateReleaseNotes", fakeElement()],
      ["uxpUpdateNotesDetails", fakeElement()],
      ["uxpUpdateReleaseDetails", fakeElement()],
      ["uxpUpdateRetryBtn", fakeElement()],
      ["uxpUpdateOpenBtn", fakeElement()],
      ["uxpUpdateDismissBtn", fakeElement()],
    ]);
    const client = {
      get: vi.fn()
        .mockResolvedValueOnce({ ok: false, error: "offline", data: { error: "offline" } })
        .mockResolvedValueOnce({
          ok: true,
          data: {
            current_version: "1.46.0",
            latest_version: "1.47.0",
            update_available: true,
            release_url: "https://github.com/SysAdminDoc/OpenCut/releases/tag/v1.47.0",
            release_name: "OpenCut 1.47.0",
            release_notes: "Fixes",
          },
        }),
    };
    const showToast = vi.fn();
    const openExternalUrl = vi.fn().mockResolvedValue(true);
    const controller = createUxpUpdateController({
      documentRef: { getElementById: (id) => ids.get(id) || null },
      client,
      showToast,
      openExternalUrl,
      normalizeReleaseUrl: (value) => value?.startsWith("https://") ? value : null,
    });

    await expect(controller.checkForUpdates()).resolves.toBe(false);
    expect(ids.get("uxpUpdateNoticeCard").dataset.state).toBe("error");
    await expect(controller.checkForUpdates({ force: true })).resolves.toBe(true);
    expect(ids.get("uxpUpdateNoticeCard").dataset.state).toBe("available");
    await expect(controller.openRelease()).resolves.toBe(true);
    expect(openExternalUrl).toHaveBeenCalledWith(
      "https://github.com/SysAdminDoc/OpenCut/releases/tag/v1.47.0",
      expect.any(String),
    );
    controller.dispose();
  });

  it("surfaces failed checks and retries from the visible refresh controls", () => {
    const main = readFileSync(new URL("../../com.opencut.uxp/main.js", import.meta.url), "utf8");
    const update = readFileSync(new URL("../../com.opencut.uxp/uxp-update-controller.js", import.meta.url), "utf8");
    const index = readFileSync(new URL("../../com.opencut.uxp/index.html", import.meta.url), "utf8");

    expect(main).toContain('import { createUxpUpdateController } from "./uxp-update-controller.js";');
    expect(main).toContain("UxpUpdateController.bind");
    expect(main).toContain("await UxpUpdateController.checkForUpdates({ force: true });");
    expect(main).toContain("UxpUpdateController.dispose();");
    expect(update).toContain("async function checkForUpdates");
    expect(update).toContain("function renderNotice");
    expect(update).toContain("normalizeReleaseUrl(latestUpdate?.release_url)");
    expect(update).toContain("function dismissNotice()");
    expect(update).toContain("data.error || !data.latest_version");
    expect(update).toContain("uxp.status.update_check_failed");
    expect(index).toContain('id="refreshBtn"');
    expect(index).toContain('data-workspace-command="refresh-backend"');
    for (const id of ["uxpUpdateNoticeCard", "uxpUpdateOpenBtn", "uxpUpdateDismissBtn", "uxpUpdateRetryBtn"]) {
      expect(index).toContain(`id="${id}"`);
    }
  });
});

describe("UXP settings controller", () => {
  it("owns settings navigation and onboarding listeners behind one teardown", async () => {
    const nav = fakeElement();
    const settingsButton = fakeElement({ classes: ["oc-settings-nav-item", "active"] });
    settingsButton.dataset.settingsSection = "workspace";
    const settingsPane = fakeElement();
    settingsPane.dataset.settingsPane = "workspace";
    nav.querySelectorAll = (selector) => selector === ".oc-settings-nav-item" ? [settingsButton] : [];
    const ids = new Map([
      ["uxpOnboardingOverlay", fakeElement()],
      ["uxpOnboardingTitle", fakeElement()],
      ["uxpOnboardingBody", fakeElement()],
      ["uxpOnboardingStep", fakeElement()],
      ["uxpOnboardingActionBtn", fakeElement()],
      ["uxpOnboardingBackBtn", fakeElement()],
      ["uxpOnboardingNextBtn", fakeElement()],
      ["uxpOnboardingSkipBtn", fakeElement()],
    ]);
    ids.get("uxpOnboardingOverlay").hidden = true;
    const documentRef = {
      activeElement: settingsButton,
      body: fakeElement(),
      querySelector: (selector) => selector === ".oc-settings-nav" ? nav : null,
      querySelectorAll: (selector) => selector === "#tab-settings [data-settings-pane]" ? [settingsPane] : [],
      getElementById: (id) => ids.get(id) || null,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    };
    const controller = createUxpSettingsController({
      documentRef,
      requestAnimationFrameFn: (callback) => callback(),
      client: {
        get: vi.fn().mockResolvedValue({ ok: true, data: { seen: false, step: 0 } }),
        post: vi.fn(),
      },
      isBackendConnected: () => true,
    });

    controller.initSettingsNavigation();
    controller.initOnboardingEvents();
    await controller.loadOnboarding();
    expect(ids.get("uxpOnboardingOverlay").hidden).toBe(false);
    controller.dispose();
    expect(nav.removeEventListener).toHaveBeenCalled();
    expect(documentRef.removeEventListener).toHaveBeenCalled();
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
    const checkpointStart = main.indexOf("async function runCheckpointedUxpHostWrite");
    const checkpointSource = main.slice(checkpointStart, checkpointStart + 2200);
    expect(checkpointSource.indexOf("isVersionAtLeast")).toBeGreaterThanOrEqual(0);
    expect(checkpointSource.indexOf("isVersionAtLeast")).toBeLessThan(
      checkpointSource.indexOf('BackendClient.post("/journal/checkpoints"'),
    );
    expect(main).toContain('runtimeState.backendVersion = String(r.data?.version || "")');
  });

  it("imports oversized cut passes as one requested-count interchange", () => {
    const main = readFileSync(new URL("../../com.opencut.uxp/main.js", import.meta.url), "utf8");
    const start = main.indexOf("async function applyTimelineCutsViaInterchange");
    const end = main.indexOf("/** ── ADD SEQUENCE MARKERS", start);
    const source = main.slice(start, end);

    expect(source).toContain('"/timeline/export-premiere-interchange"');
    expect(source).toContain('action: "import_sequence"');
    expect(source).toContain("PProBridge.importTimelineInterchange");
    expect(source).toContain("requested_cuts");
    expect(source).toContain("cutsToApply.length");
    expect(source).toContain("interchange_imported");
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

  it("sends walked UXP clips to deliverables and reports empty sheets", () => {
    const main = readFileSync(new URL("../../com.opencut.uxp/main.js", import.meta.url), "utf8");
    const start = main.indexOf("async function getDeliverablesSequenceData");
    const end = main.indexOf("// ─────────────────────────────────────────────────────────────\n// AI B-Roll Generation", start);
    const source = main.slice(start, end);

    expect(source).toContain("getSequenceIndexPayload");
    expect(source).toContain("video_tracks: Array.isArray(sequence.videoTracks)");
    expect(source).toContain("audio_tracks: Array.isArray(sequence.audioTracks)");
    expect(source).toContain("sequence_data: seqData");
    expect(source).toContain("getDeliverableRowCount(result)");
    expect(source).toContain("getDeliverableRowCount(r.data)");
    expect(source).toContain('"uxp.deliverables.runtime.no_clips_found_toast"');
    expect(source).toContain('"uxp.deliverables.runtime.package_generated_no_clips"');
    expect(source).toContain("empty: emptyRows > 0");
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
