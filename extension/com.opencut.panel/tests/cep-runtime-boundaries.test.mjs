import { createRequire } from "node:module";
import { readFileSync } from "node:fs";

import { describe, expect, it, vi } from "vitest";

const require = createRequire(import.meta.url);
const { createPanelState } = require("../client/panel-state.js");
const { createBackendClient } = require("../client/backend-client.js");
const { createJobRuntime, isTerminalStatus } = require("../client/job-runtime.js");
const components = require("../client/component-utils.js");
const timeline = require("../client/timeline-utils.js");
const onboarding = require("../client/onboarding-state.js");
const bootstrap = require("../client/bootstrap.js");

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

describe("CEP source ownership", () => {
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
