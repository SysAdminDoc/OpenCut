import { createRequire } from "node:module";
import { readFileSync } from "node:fs";

import { describe, expect, it, vi } from "vitest";

const require = createRequire(import.meta.url);
const { createPanelState } = require("../client/panel-state.js");
const { createBackendClient } = require("../client/backend-client.js");
const { createJobRuntime, isTerminalStatus } = require("../client/job-runtime.js");
const { createJobLifecycleRegistry } = require("../client/job-lifecycle.js");
const components = require("../client/component-utils.js");
const timeline = require("../client/timeline-utils.js");
const onboarding = require("../client/onboarding-state.js");
const bootstrap = require("../client/bootstrap.js");
const { createUpdateController } = require("../client/update-controller.js");
const { createResultsController } = require("../client/results-controller.js");
const { createSettingsDiagnosticsController } = require("../client/settings-diagnostics-controller.js");
const { createNavigationController } = require("../client/navigation-controller.js");

function requestHarness() {
  const requests = [];
  function createRequest() {
    const xhr = {
      headers: {},
      open: vi.fn((method, url) => {
        xhr.method = method;
        xhr.url = url;
      }),
      setRequestHeader: vi.fn((name, value) => {
        xhr.headers[name] = value;
      }),
      send: vi.fn((body) => {
        xhr.body = body;
      }),
      status: 0,
      responseText: "",
    };
    requests.push(xhr);
    return xhr;
  }
  function respond(index, status, body) {
    requests[index].status = status;
    requests[index].responseText = JSON.stringify(body);
    requests[index].onload();
  }
  return { requests, createRequest, respond };
}

describe("CEP panel state", () => {
  it("publishes immutable connection snapshots", () => {
    const store = createPanelState({ backendUrl: "http://127.0.0.1:5679" });
    const listener = vi.fn();
    const unsubscribe = store.subscribe(listener);
    store.setConnected(true);
    store.setCsrfToken("token");
    unsubscribe();
    store.setConnected(false);

    expect(listener).toHaveBeenCalledTimes(2);
    expect(listener.mock.calls[1][0]).toEqual({
      backendUrl: "http://127.0.0.1:5679",
      csrfToken: "token",
      connected: true,
    });
    expect(store.isConnected()).toBe(false);
  });
});

describe("CEP backend client", () => {
  it("deduplicates in-flight GETs and fans out their result", () => {
    const harness = requestHarness();
    const first = vi.fn();
    const second = vi.fn();
    const client = createBackendClient({
      getBaseUrl: () => "http://local",
      createRequest: harness.createRequest,
    });

    client.request("GET", "/health", null, first);
    client.request("GET", "/health", null, second);
    expect(harness.requests).toHaveLength(1);
    harness.respond(0, 200, { status: "ok" });
    expect(first).toHaveBeenCalledWith(null, { status: "ok" });
    expect(second).toHaveBeenCalledWith(null, { status: "ok" });
  });

  it("refreshes CSRF and retries one rejected mutation", () => {
    const harness = requestHarness();
    let token = "stale";
    const callback = vi.fn();
    const client = createBackendClient({
      getBaseUrl: () => "http://local",
      getToken: () => token,
      setToken: (value) => { token = value; },
      createRequest: harness.createRequest,
    });

    client.request("POST", "/jobs", { name: "cut" }, callback);
    harness.respond(0, 403, { error: "CSRF token expired" });
    expect(harness.requests[1].url).toBe("http://local/health");
    harness.respond(1, 200, { csrf_token: "fresh" });
    expect(harness.requests[2].headers["X-OpenCut-Token"]).toBe("fresh");
    harness.respond(2, 200, { job_id: "job-1" });
    expect(callback).toHaveBeenCalledOnce();
    expect(callback).toHaveBeenCalledWith(null, { job_id: "job-1" });
  });

  it("surfaces HTTP and timeout failures through the callback contract", () => {
    const harness = requestHarness();
    const httpCallback = vi.fn();
    const timeoutCallback = vi.fn();
    const client = createBackendClient({
      getBaseUrl: () => "http://local",
      translate: (_key, fallback) => fallback,
      createRequest: harness.createRequest,
    });
    client.request("POST", "/jobs", {}, httpCallback);
    harness.respond(0, 500, { error: "boom" });
    expect(httpCallback.mock.calls[0][0].message).toBe("boom");

    client.request("POST", "/slow", {}, timeoutCallback);
    harness.requests[1].ontimeout();
    expect(timeoutCallback.mock.calls[0][0].message).toBe("Timeout");
  });
});

describe("CEP job runtime", () => {
  it("locks start and active phases as one exclusive lifecycle", () => {
    const runtime = createJobRuntime();
    expect(runtime.beginStart()).toBe(true);
    expect(runtime.beginStart()).toBe(false);
    runtime.activate("job-7");
    expect(runtime.isCurrent("job-7")).toBe(true);
    expect(runtime.beginStart()).toBe(false);
    expect(runtime.finish({ status: "running" })).toBe(false);
    expect(runtime.finish({ status: "complete" })).toBe(true);
    expect(runtime.isIdle()).toBe(true);
  });

  it("recognizes all backend terminal statuses", () => {
    expect(["complete", "error", "cancelled"].every(isTerminalStatus)).toBe(true);
    expect(isTerminalStatus("running")).toBe(false);
  });
});

describe("CEP job lifecycle", () => {
  it("settles cancellation hooks once and clears the registry", () => {
    const registry = createJobLifecycleRegistry();
    const onCancel = vi.fn();
    const onFinally = vi.fn();
    registry.register("job-cancel", { onCancel, onFinally });

    registry.settle({ id: "job-cancel", status: "cancelled" });
    registry.settle({ id: "job-cancel", status: "cancelled" });

    expect(onCancel).toHaveBeenCalledOnce();
    expect(onCancel).toHaveBeenCalledWith({ id: "job-cancel", status: "cancelled" });
    expect(onFinally).toHaveBeenCalledOnce();
    expect(registry.pendingCount()).toBe(0);
  });
});

describe("CEP component helpers", () => {
  it("updates a nested label without replacing button structure", () => {
    const label = { textContent: "Run" };
    const attributes = new Map();
    const button = {
      disabled: false,
      textContent: "container",
      querySelector: () => label,
      getAttribute: (name) => attributes.has(name) ? attributes.get(name) : null,
      setAttribute: (name, value) => attributes.set(name, value),
    };
    components.setButtonBusy(button, true, "Working…");
    expect(button.disabled).toBe(true);
    expect(label.textContent).toBe("Working…");
    components.setButtonBusy(button, false, "unused");
    expect(label.textContent).toBe("Run");
  });
});

describe("CEP timeline payloads", () => {
  it("preserves cut order while cloning host payload rows", () => {
    const cuts = [{ start: 5, end: 7 }, { start: 1, end: 2 }];
    const cloned = timeline.cloneCuts(cuts);
    expect(cloned).toEqual(cuts);
    expect(cloned[0]).not.toBe(cuts[0]);
  });

  it("builds marker, rename, smart-bin, and OTIO route shapes", () => {
    expect(timeline.buildBeatMarkers([1.5], "Beat", "Chapter")).toEqual([
      { time: 1.5, name: "Beat", type: "Chapter" },
    ]);
    expect(timeline.buildRenameOperations(
      [{ nodeId: "n1", name: "Old" }, { id: "n2", name: "Same" }],
      [{ index: 0, value: "New" }, { index: 1, value: "Same" }],
    )).toEqual([{ nodeId: "n1", newName: "New" }]);
    expect(timeline.buildSmartBinHostRules([
      { bin_name: "Wide", rule_type: "contains", field: "name", value: "16x9" },
    ])).toEqual([{ binName: "Wide", rule: "contains", field: "name", value: "16x9" }]);
    expect(timeline.buildOtioPayload({
      filepath: "clip.mp4",
      outputDir: "out",
      mode: "markers",
      beatTimes: [2],
      beatLabel: "Beat",
    })).toEqual({
      filepath: "clip.mp4",
      output_dir: "out",
      mode: "markers",
      adapter_name: "otio_json",
      schema_target: "current",
      accept_lossy: false,
      markers: [{ time: 2, name: "Beat" }],
    });
  });
});

describe("CEP onboarding state machine", () => {
  it("loads server state and clamps resume steps", () => {
    const machine = onboarding.createOnboardingState({ stepCount: 5 });
    expect(machine.transition("load").status).toBe(onboarding.STATUS.LOADING);
    expect(machine.transition("loaded", { seen: false, step: 99 })).toEqual({
      status: onboarding.STATUS.ACTIVE,
      step: 4,
      seen: false,
      error: "",
      stepCount: 5,
    });
    expect(machine.transition("back").step).toBe(3);
    expect(machine.transition("next").step).toBe(4);
  });

  it("models unavailable, completed, and restart recovery states", () => {
    const changes = [];
    const machine = onboarding.createOnboardingState({
      stepCount: 3,
      onChange: (state) => changes.push(state),
    });
    expect(machine.transition("failed", { error: "offline" }).status)
      .toBe(onboarding.STATUS.UNAVAILABLE);
    expect(machine.transition("completed").seen).toBe(true);
    expect(machine.transition("restart")).toMatchObject({
      status: onboarding.STATUS.ACTIVE,
      step: 0,
      seen: false,
    });
    expect(Object.isFrozen(changes[0])).toBe(true);
  });

  it("gates the startup tour probe on connectivity and a local seen cache", () => {
    const main = readFileSync(new URL("../client/main.js", import.meta.url), "utf8");
    expect(main).toContain('var ONBOARDING_SEEN_KEY = "opencut_onboarding_seen"');

    const start = main.indexOf("function maybeRunOnboarding(");
    const end = main.indexOf("function onBackendConnected(");
    expect(start).toBeGreaterThan(-1);
    expect(end).toBeGreaterThan(start);
    const probe = main.slice(start, end);
    // Silent short-circuits: a cached seen flag or a disconnected backend
    // must never open the focus-trapped modal from the startup timer.
    expect(probe).toContain("if (hasLocalOnboardingSeen()) return;");
    expect(probe).toContain("onboardingAutoPending = true;");
    // The offline card is only rendered when the backend is reachable but
    // the onboarding endpoint itself errors (and on explicit restarts).
    expect(probe).toContain("if (!connected) {");

    // Deferred probes re-run once a backend health check succeeds again.
    expect(main).toContain("function notifyBackendReconnectHooks()");
    expect(main).toContain("if (wasDisconnected) notifyBackendReconnectHooks();");
    expect(main).toContain("onBackendConnected: onBackendConnected");

    // Server seen=true responses and complete/skip both cache locally, and
    // an explicit restart clears the cache so the tour can run again.
    expect(main.match(/setLocalOnboardingSeen\(true\)/g)?.length).toBeGreaterThanOrEqual(2);
    expect(main).toContain("setLocalOnboardingSeen(false)");
  });
});

describe("CEP bootstrap", () => {
  it("keeps ordered steps moving after an isolated failure", () => {
    const order = [];
    const onError = vi.fn();
    bootstrap.runSteps([
      () => order.push("first"),
      () => { throw new Error("broken"); },
      () => order.push("third"),
    ], onError);
    expect(order).toEqual(["first", "third"]);
    expect(onError.mock.calls[0][1]).toBe(1);
  });

  it("runs immediately for an already-ready document", () => {
    const initialize = vi.fn();
    bootstrap.onReady({ readyState: "complete", addEventListener: vi.fn() }, initialize);
    expect(initialize).toHaveBeenCalledOnce();
  });
});

describe("CEP update-check boundary", () => {
  it("surfaces failed checks and retries from the header refresh control", () => {
    const main = readFileSync(new URL("../client/main.js", import.meta.url), "utf8");
    const controller = readFileSync(new URL("../client/update-controller.js", import.meta.url), "utf8");
    const index = readFileSync(new URL("../client/index.html", import.meta.url), "utf8");

    expect(main).toContain("OpenCutUpdateController.createUpdateController");
    expect(main).toContain("UpdateController.check(false)");
    expect(main).toContain("UpdateController.check(true)");
    expect(main).toContain("UpdateController.bind()");
    expect(main).toContain("UpdateController.dispose()");
    expect(main).not.toContain("function checkForUpdateNotice(force)");
    expect(main).not.toContain("function renderUpdateNotice(result, checking)");
    expect(controller).toContain("function createUpdateController(options)");
    expect(controller).toContain('"/system/update-check"');
    expect(controller).toContain("updateCheckFailed");
    expect(controller).toContain("function dispose()");
    const refreshStart = main.indexOf("function refreshAll()");
    expect(refreshStart).toBeGreaterThan(-1);
    expect(main.slice(refreshStart, refreshStart + 500)).toContain("UpdateController.check(true)");
    expect(index).toContain('id="refreshAllBtn"');
    for (const id of ["updateNoticeCard", "updateOpenBtn", "updateDismissBtn", "updateRetryBtn"]) {
      expect(index).toContain(`id="${id}"`);
    }
  });

  it("owns retry, release, dismiss, and teardown behavior", () => {
    const ids = [
      "updateNoticeCard",
      "updateStatusText",
      "updateSummary",
      "updateCurrentVersion",
      "updateAvailableVersion",
      "updateReleaseName",
      "updateReleaseNotes",
      "updateNotesDetails",
      "updateReleaseDetails",
      "updateRetryBtn",
      "updateOpenBtn",
      "updateDismissBtn",
    ];
    const elements = new Map();
    for (const id of ids) {
      const listeners = new Map();
      elements.set(id, {
        textContent: id === "updateCurrentVersion" ? "1.46.0" : "",
        hidden: false,
        disabled: false,
        attributes: {},
        classList: { toggle: vi.fn() },
        setAttribute(name, value) { this.attributes[name] = value; },
        addEventListener(type, listener) { listeners.set(type, listener); },
        removeEventListener(type, listener) {
          if (listeners.get(type) === listener) listeners.delete(type);
        },
        dispatch(type) {
          if (listeners.has(type)) listeners.get(type)({ type, target: this });
        },
      });
    }
    const storage = new Map();
    const requests = [];
    const toasts = vi.fn();
    const openExternalUrl = vi.fn(() => true);
    const controller = createUpdateController({
      documentRef: { getElementById: (id) => elements.get(id) || null },
      storage: {
        getItem: (key) => storage.get(key) || null,
        setItem: (key, value) => storage.set(key, value),
      },
      request: (method, path, body, callback) => requests.push({ method, path, callback }),
      translate: (_key, fallback) => fallback,
      showToast: toasts,
      normalizeReleaseUrl: (value) => value === "https://github.com/SysAdminDoc/OpenCut/releases/tag/v1.47.0" ? value : "",
      openExternalUrl,
    });

    controller.bind();
    expect(controller.check(false)).toBe(true);
    expect(requests[0].path).toBe("/system/update-check");
    expect(elements.get("updateNoticeCard").attributes["data-state"]).toBe("checking");
    requests[0].callback(new Error("offline"));
    expect(elements.get("updateNoticeCard").attributes["data-state"]).toBe("error");
    expect(controller.check(false)).toBe(false);

    expect(controller.check(true)).toBe(true);
    requests[1].callback(null, {
      current_version: "1.46.0",
      latest_version: "1.47.0",
      update_available: true,
      release_name: "OpenCut 1.47.0",
      release_url: "https://github.com/SysAdminDoc/OpenCut/releases/tag/v1.47.0",
      release_notes: "Notes",
    });
    expect(elements.get("updateNoticeCard").attributes["data-state"]).toBe("available");
    elements.get("updateOpenBtn").dispatch("click");
    expect(openExternalUrl).toHaveBeenCalledWith("https://github.com/SysAdminDoc/OpenCut/releases/tag/v1.47.0");
    elements.get("updateDismissBtn").dispatch("click");
    expect(storage.get("opencut_update_dismissed_version")).toBe("1.47.0");
    expect(elements.get("updateNoticeCard").attributes["data-state"]).toBe("dismissed");

    controller.dispose();
    elements.get("updateRetryBtn").dispatch("click");
    expect(requests).toHaveLength(2);
    expect(controller.check(true)).toBe(false);
  });
});

describe("CEP shipped control wiring", () => {
  function sourceBetween(source, startMarker, endMarker) {
    const start = source.indexOf(startMarker);
    const end = source.indexOf(endMarker, start + startMarker.length);
    expect(start, startMarker).toBeGreaterThanOrEqual(0);
    expect(end, endMarker).toBeGreaterThan(start);
    return source.slice(start, end);
  }

  it("opens deliverable folders through a guarded host function and reports failures", () => {
    const main = readFileSync(new URL("../client/main.js", import.meta.url), "utf8");
    const index = readFileSync(new URL("../client/index.html", import.meta.url), "utf8");
    const host = readFileSync(new URL("../host/index.jsx", import.meta.url), "utf8");
    const source = sourceBetween(main, "function initDeliverablesFeatures()", "function initNlpFeatures()");

    expect(index).toContain('id="openDeliverablesFolder"');
    expect(source).toContain("openFolderInFinder");
    expect(source).toContain("JSON.parse(result || \"{}\")");
    expect(source).toContain("parsed.error || parsed.success === false");
    expect(main).toContain('openDeliverablesFolderBtn.classList.toggle("hidden", !inPremiere)');
    expect(host).toContain("function openFolderInFinder(path)");
    expect(host).toContain("new File(String(path))");
    expect(host).toContain("file.parent.execute()");
    expect(host).toContain('JSON.stringify({ error: "No project open" })');
  });

  it("previews silence with the same controls used by the run payload", () => {
    const main = readFileSync(new URL("../client/main.js", import.meta.url), "utf8");
    const source = sourceBetween(main, "function initAudioPreviewButtons()", "function renderAudioPreview(");

    expect(source).toContain("el.threshold.value");
    expect(source).toContain("el.minDuration.value");
    expect(source).not.toContain("el.silenceThreshold");
    expect(source).not.toContain("el.silenceMinDur");
  });

  it("loads demo footage through the real selection lifecycle", () => {
    const main = readFileSync(new URL("../client/main.js", import.meta.url), "utf8");
    const source = sourceBetween(main, "function tryDemo()", "function gistPush()");

    expect(source).toContain("selectFile(data.path)");
    expect(source).not.toContain("selectedPath = data.path");
    expect(source).not.toContain("selectedClipLabel");
  });

  it("journals multicam writes and treats host error payloads as failures", () => {
    const main = readFileSync(new URL("../client/main.js", import.meta.url), "utf8");
    const source = sourceBetween(main, "function applyMulticamCuts()", "function renderMulticamTrackMap()");

    expect(source).toContain("journalCheckpointedHostWrite");
    expect(source).toContain("ocApplySequenceCuts");
    expect(source).toContain("if (!r || r.error)");
    expect(source.indexOf("journalCheckpointedHostWrite")).toBeLessThan(source.indexOf("ocApplySequenceCuts"));
    expect(source).toContain("timeline.action_failed");
  });
});

describe("CEP source ownership", () => {
  it("persists recovery checkpoints before every journalled host mutation", () => {
    const main = readFileSync(new URL("../client/main.js", import.meta.url), "utf8");
    const functionSlice = (name, nextName) => main.slice(
      main.indexOf(`function ${name}`),
      main.indexOf(`function ${nextName}`, main.indexOf(`function ${name}`) + 1),
    );
    for (const [name, nextName, hostCall] of [
      ["applySequenceCuts", "runBeatMarkers", "ocApplySequenceCuts"],
      ["addBeatMarkersToSequence", "runMulticamCuts", "ocAddSequenceMarkers"],
      ["renameAll", "createSmartBins", "ocBatchRenameProjectItems"],
      ["createSmartBins", "runRepeatDetect", "ocCreateSmartBins"],
      ["addChaptersAsMarkers", "runSrtImport", "ocAddSequenceMarkers"],
      ["runSrtImport", "runLoudMatch", "ocAddNativeCaptionTrack"],
    ]) {
      const source = functionSlice(name, nextName);
      expect(source.indexOf("journalCheckpointedHostWrite"), name).toBeGreaterThanOrEqual(0);
      expect(source.indexOf("journalCheckpointedHostWrite"), name).toBeLessThan(source.indexOf(hostCall));
    }
    expect(main).toContain('"/journal/checkpoints"');
    expect(main).toContain('"/recovery-failed"');
    expect(main).toContain('"/complete"');
    const checkpointStart = main.indexOf("function journalCheckpointedHostWrite");
    const checkpointSource = main.slice(checkpointStart, checkpointStart + 2400);
    expect(checkpointSource.indexOf("isVersionAtLeast")).toBeGreaterThanOrEqual(0);
    expect(checkpointSource.indexOf("isVersionAtLeast")).toBeLessThan(
      checkpointSource.indexOf('api("POST", "/journal/checkpoints"'),
    );
    expect(main).toContain('backendVersion = String(data.version || "")');
  });

  it("imports oversized cut passes as one requested-count interchange", () => {
    const main = readFileSync(new URL("../client/main.js", import.meta.url), "utf8");
    const source = main.slice(
      main.indexOf("function applySequenceCutsViaInterchange"),
      main.indexOf("function runBeatMarkers()"),
    );

    expect(source).toContain("getInterchangeCutThreshold");
    expect(source).toContain('"/timeline/export-premiere-interchange"');
    expect(source).toContain('action: "import_sequence"');
    expect(source).toContain("PremiereBridge.importXML");
    expect(source).toContain("requestedCount");
    expect(source).toContain("timeline.interchange_imported");
    expect(source).toContain('.replace("{count}", cutPlan.length)');

    const normalApply = main.slice(
      main.indexOf("function applySequenceCuts(cuts)"),
      main.indexOf("function runBeatMarkers()"),
    );
    expect(normalApply).toContain('.replace("{count}", cutPlan.length)');
  });

  it("keeps panel copy on the backend term and removes the unreachable wizard body", () => {
    const index = readFileSync(new URL("../client/index.html", import.meta.url), "utf8");
    const locale = JSON.parse(readFileSync(new URL("../client/locales/en.json", import.meta.url), "utf8"));
    const descriptionEntries = Object.entries(locale).filter(([key]) => key.endsWith("_desc"));
    expect(descriptionEntries.filter(([, value]) => /\s--\s/.test(value))).toEqual([]);

    const serverReferences = Object.entries(locale).filter(([, value]) => /\bserver\b/i.test(value));
    expect(serverReferences.every(([, value]) => value.includes("opencut-server") || value.includes("opencut.server"))).toBe(true);

    expect(index).toContain('<div class="wizard-overlay hidden" id="wizardOverlay" role="dialog" aria-modal="true">');
    expect(index).toContain('<div class="wizard-card" tabindex="-1"></div>');
    for (const id of ["wizardDontShow", "wizardCloseBtn", "wizardSteps"]) {
      expect(index).not.toContain(`id="${id}"`);
    }
  });

  it("keeps extracted responsibilities out of the orchestration entrypoint", () => {
    const main = readFileSync(new URL("../client/main.js", import.meta.url), "utf8");
    expect(main).toContain("OpenCutBackendClient.createBackendClient");
    expect(main).toContain("OpenCutJobRuntime.createJobRuntime");
    expect(main).toContain("OpenCutOnboardingState.createOnboardingState");
    expect(main).toContain("OpenCutBootstrap.onReady");
    expect(main).not.toContain("var _inflightRequests");
    expect(main).not.toContain("function rememberButtonText(");
    expect(main).not.toContain("document.addEventListener(\"DOMContentLoaded\"");
    expect(main).not.toContain("var currentJob =");
    expect(main).not.toContain("var jobStarting =");
    expect(main).not.toContain("wizardDismissed");
    expect(main).not.toContain("function initWizard(");
  });

  it("announces every terminal job result through the live regions", () => {
    const main = readFileSync(new URL("../client/main.js", import.meta.url), "utf8");
    const results = readFileSync(new URL("../client/results-controller.js", import.meta.url), "utf8");
    const index = readFileSync(new URL("../client/index.html", import.meta.url), "utf8");

    // Two regions, not one with a swapped politeness: assistive technology
    // reads aria-live when it first sees the node, so flipping it is unreliable.
    expect(index).toContain(
      '<p class="results-announce" id="resultsAnnouncePolite" role="status" aria-live="polite" aria-atomic="true"></p>',
    );
    expect(index).toContain(
      '<p class="results-announce" id="resultsAnnounceAssertive" role="alert" aria-live="assertive" aria-atomic="true"></p>',
    );
    expect(index).toContain('id="resultsSection" tabindex="-1"');
    expect(index.indexOf("announce-utils.js")).toBeLessThan(index.indexOf("main.js"));

    // Success is polite, failure is assertive and names the recovery route.
    expect(main).toContain("ResultsController.showSuccess(job, lastJobPayload)");
    expect(results).toContain('onAnnounce("polite", t("progress.announce_finished"');
    expect(main).toContain("ResultsController.showFailure(job, enhanceError(");
    expect(main).toContain('announceJobResult("error", replaceTemplateValue(failureTemplate, "{reason}", failureReason));');
    expect(main).not.toMatch(/\.replace\("\{(?:error|reason)\}"/);
    expect(main).toContain('t("progress.announce_failed_retry"');
    expect(main).toContain('announceJobResult("polite", t("progress.announce_cancelled"');

    // Focus is rescued only when finishing the job stranded it — moving it
    // on every result would yank the user out of wherever they were.
    const start = main.indexOf("function announceJobResult(");
    const end = main.indexOf("function settleJobLifecycle(");
    expect(start).toBeGreaterThan(-1);
    expect(end).toBeGreaterThan(start);
    const helper = main.slice(start, end);
    expect(helper).toContain("Announce.focusWasStranded(document.activeElement, document)");
    expect(helper).toContain("Announce.focusResultsRegion(el.resultsSection)");

    // A dismissed card must not leave a stale result to be re-read.
    expect(main.match(/clearResultAnnouncement\(\);/g).length).toBeGreaterThanOrEqual(3);
  });

  it("keeps terminal result rendering behind an injected controller boundary", () => {
    const main = readFileSync(new URL("../client/main.js", import.meta.url), "utf8");
    const results = readFileSync(new URL("../client/results-controller.js", import.meta.url), "utf8");

    expect(main).toContain("OpenCutResultsController.createResultsController");
    expect(main).toContain("ResultsController.dispose()");
    expect(main).not.toContain("function showResults(job)");
    expect(results).toContain("function showSuccess(job, sourcePayload)");
    expect(results).toContain("function showFailure(job, message, canRetry)");
    expect(results).toContain("function dispose()");
  });

  it("keeps settings diagnostics requests behind a disposable controller", () => {
    const main = readFileSync(new URL("../client/main.js", import.meta.url), "utf8");
    const diagnostics = readFileSync(new URL("../client/settings-diagnostics-controller.js", import.meta.url), "utf8");

    expect(main).toContain("OpenCutSettingsDiagnosticsController.createSettingsDiagnosticsController");
    expect(main).toContain("SettingsDiagnosticsController.load()");
    expect(main).toContain("SettingsDiagnosticsController.dispose()");
    expect(main).not.toContain("function loadSettingsInfo()");
    expect(diagnostics).toContain('"/system/gpu"');
    expect(diagnostics).toContain('"/health"');
    expect(diagnostics).toContain("function dispose()");
  });

  it("keeps CEP navigation listener ownership behind a teardown boundary", () => {
    const main = readFileSync(new URL("../client/main.js", import.meta.url), "utf8");
    const navigation = readFileSync(new URL("../client/navigation-controller.js", import.meta.url), "utf8");

    expect(main).toContain("OpenCutNavigationController.createNavigationController");
    expect(main).toContain("NavigationController.bind()");
    expect(main).toContain("NavigationController.dispose()");
    expect(main).not.toContain("function setupNavTabs()");
    expect(navigation).toContain("function bind()");
    expect(navigation).toContain("function dispose()");
  });

  it("removes CEP navigation listeners on panel teardown", () => {
    function button(name, active) {
      const listeners = new Map();
      const attributes = { "data-nav": name };
      return {
        id: "",
        tabIndex: 0,
        parentElement: null,
        classList: {
          contains: (value) => value === "active" && active,
          toggle: vi.fn(),
        },
        getAttribute: (key) => attributes[key] || null,
        setAttribute: (key, value) => { attributes[key] = value; },
        removeAttribute: (key) => { delete attributes[key]; },
        addEventListener: (type, listener) => listeners.set(type, listener),
        removeEventListener: (type, listener) => {
          if (listeners.get(type) === listener) listeners.delete(type);
        },
        focus: vi.fn(),
        click: vi.fn(),
        dispatch: (type, event = {}) => listeners.get(type)?.({ target: this, ...event }),
        listenerCount: () => listeners.size,
      };
    }
    const first = button("cut", true);
    const second = button("settings", false);
    first.parentElement = { querySelectorAll: () => [first, second] };
    second.parentElement = first.parentElement;
    const windowListeners = new Map();
    const windowRef = {
      addEventListener: (type, listener) => windowListeners.set(type, listener),
      removeEventListener: (type, listener) => {
        if (windowListeners.get(type) === listener) windowListeners.delete(type);
      },
    };
    const documentRef = {
      querySelectorAll: (selector) => selector === ".nav-tab" ? [first, second] : [],
      querySelector: () => first,
    };
    const activateNavTab = vi.fn();
    const updateWorkspaceClipStatus = vi.fn();
    const controller = createNavigationController({
      documentRef,
      windowRef,
      getElement: () => null,
      getVisibleTabButtons: (container) => container.querySelectorAll(".nav-tab"),
      moveFocusAndActivate: vi.fn(),
      activateNavTab,
      activateSubTab: vi.fn(),
      updateWorkspaceClipStatus,
      onResize: vi.fn(),
    });

    expect(controller.bind()).toBe(true);
    expect(activateNavTab).toHaveBeenCalledWith("cut", { remember: false, scroll: false });
    expect(updateWorkspaceClipStatus).toHaveBeenCalledOnce();
    expect(windowListeners.has("resize")).toBe(true);
    const clicksBeforeDispose = first.listenerCount();
    controller.dispose();
    expect(first.listenerCount()).toBe(clicksBeforeDispose - 2);
    expect(windowListeners.has("resize")).toBe(false);
    expect(controller.bind()).toBe(false);
  });

  it("prevents late health and GPU callbacks after settings teardown", () => {
    const requests = [];
    const health = vi.fn();
    const gpu = vi.fn();
    const controller = createSettingsDiagnosticsController({
      request: (method, path, body, callback) => requests.push({ path, callback }),
      renderOverview: vi.fn(),
      syncBackendSummary: vi.fn(),
      updateWhisperState: health,
      renderGpuState: gpu,
      loadLlmSettings: vi.fn(),
      updateBridgeStatus: vi.fn(),
      refreshDependencies: vi.fn(),
      refreshModels: vi.fn(),
      loadEngineRegistry: vi.fn(),
      loadPluginTrust: vi.fn(),
    });

    expect(controller.load()).toBe(true);
    expect(requests.map((request) => request.path)).toEqual(["/health", "/system/gpu"]);
    controller.dispose();
    requests[0].callback(null, { status: "ok", capabilities: {} });
    requests[1].callback(null, { available: true, name: "GPU" });
    expect(health).not.toHaveBeenCalled();
    expect(gpu).not.toHaveBeenCalled();
    expect(controller.load()).toBe(false);
  });

  it("renders success and failure states without owning the job lifecycle", () => {
    function node() {
      return {
        classList: { add: vi.fn(), remove: vi.fn() },
        textContent: "",
        innerHTML: "",
        title: "",
        removeAttribute: vi.fn(),
        setAttribute: vi.fn(),
      };
    }
    const elements = {
      resultsSection: node(),
      resultsTitle: node(),
      resultsStats: node(),
      resultsPath: node(),
      retryJobBtn: node(),
    };
    const boundaryReview = vi.fn();
    const announce = vi.fn();
    const controller = createResultsController({
      elements,
      translate: (_key, fallback) => fallback,
      escapeHtml: (value) => String(value),
      safeFixed: (value, digits) => Number(value).toFixed(digits),
      onBoundaryReview: boundaryReview,
      onAnnounce: announce,
    });

    expect(controller.showSuccess({
      result: {
        summary: "Finished safely",
        segments: 2,
        boundary_review: { required: true, review_hits: 1, items: [] },
        output_path: "C:/exports/result.wav",
      },
    }, { filepath: "clip.mov" })).toBe(true);
    expect(elements.resultsTitle.textContent).toBe("Finished");
    expect(elements.resultsPath.textContent).toBe("C:/exports/result.wav");
    expect(boundaryReview).toHaveBeenCalledWith(
      { required: true, review_hits: 1, items: [] },
      { filepath: "clip.mov" },
    );
    expect(announce).toHaveBeenCalledWith("polite", expect.stringContaining("Run finished"));

    expect(controller.showFailure({ error: "bad media" }, "Readable failure", true)).toBe(true);
    expect(elements.resultsTitle.textContent).toBe("Run failed");
    expect(elements.resultsStats.textContent).toBe("Readable failure");
    expect(controller.dispose()).toBeUndefined();
    expect(controller.showSuccess({ result: {} })).toBe(false);
  });

  it("settles the registered lifecycle before sending a user cancel", () => {
    const main = readFileSync(new URL("../client/main.js", import.meta.url), "utf8");
    const cancelStart = main.indexOf("function cancelJob()");
    const settled = main.indexOf(
      'settleJobLifecycle({ id: cancellingJob, status: "cancelled" });',
      cancelStart,
    );
    const backendCancel = main.indexOf('api("POST", "/cancel/" + cancellingJob', cancelStart);

    expect(cancelStart).toBeGreaterThan(-1);
    expect(settled).toBeGreaterThan(cancelStart);
    expect(settled).toBeLessThan(backendCancel);
    expect(main).toContain("jobLifecycleHandlers.register(data.job_id, opts);");
  });

  it("binds Auto theme to the Premiere host skin rather than the OS", () => {
    const main = readFileSync(new URL("../client/main.js", import.meta.url), "utf8");
    const index = readFileSync(new URL("../client/index.html", import.meta.url), "utf8");

    expect(index.indexOf("cep-theme.js")).toBeLessThan(index.indexOf("main.js"));

    // Auto resolves against the host skin; the OS is only the no-host fallback.
    expect(main).toContain("CepTheme.readHostTheme(cs)");
    expect(main).toContain("CepTheme.resolveTheme(pref, hostTheme, _osPrefersLight())");
    expect(main).toContain("startHostThemeSync();");

    // The OS media query must not fight the host event while docked.
    expect(main).toContain("if (window.matchMedia && !inPremiere) {");

    // The listener is unregistered with the rest of the panel teardown.
    expect(main).toContain("stopHostThemeSync();");
    const teardown = main.indexOf('window.addEventListener("beforeunload"');
    expect(teardown).toBeGreaterThan(-1);
    expect(main.slice(teardown, teardown + 200)).toContain("stopHostThemeSync();");

    // An explicit Light/Dark choice must survive a host skin change.
    const start = main.indexOf("function startHostThemeSync(");
    const end = main.indexOf("function stopHostThemeSync(");
    expect(end).toBeGreaterThan(start);
    expect(main.slice(start, end)).toContain("_applyTheme(_currentThemePref(), hostTheme);");
  });

  it("keeps token and shell layout rules in their ordered CSS owners", () => {
    const index = readFileSync(new URL("../client/index.html", import.meta.url), "utf8");
    const tokens = readFileSync(new URL("../client/command-center-tokens.css", import.meta.url), "utf8");
    const layout = readFileSync(new URL("../client/command-center-layout.css", import.meta.url), "utf8");
    const components = readFileSync(new URL("../client/command-center.css", import.meta.url), "utf8");
    expect(tokens).toContain(":root {");
    expect(layout).toContain(".app {");
    expect(components).not.toContain(":root {");
    expect(index.match(/id="wizardOverlay"/g)).toHaveLength(1);
    expect(index).not.toContain("ocOnboardingOverlay");
    expect(index.indexOf("onboarding-state.js")).toBeLessThan(index.indexOf("main.js"));
    expect(index.indexOf("command-center-tokens.css")).toBeLessThan(index.indexOf("command-center-layout.css"));
    expect(index.indexOf("command-center-layout.css")).toBeLessThan(index.indexOf("command-center.css"));
  });
});
