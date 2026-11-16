import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { describe, expect, it } from "vitest";

const require = createRequire(import.meta.url);
const { fillTemplate, createCapabilityStatus } = require("../client/capability-status.js");

const clientDir = join(dirname(fileURLToPath(import.meta.url)), "..", "client");
const mainSource = readFileSync(join(clientDir, "main.js"), "utf8");

const HINT_STATE_CLASSES = ["is-info", "is-error", "is-success", "is-warning"];

// Plain stand-ins for the few element methods the hints touch.
function fakeClassList(initial) {
  const names = new Set(initial || []);
  return {
    contains: (name) => names.has(name),
    toggle: (name, force) => {
      const on = force === undefined ? !names.has(name) : !!force;
      if (on) names.add(name);
      else names.delete(name);
      return on;
    },
    has: (name) => names.has(name),
  };
}

function fakeElement({ text = "", attrs = {}, classes = [] } = {}) {
  const attributes = { ...attrs };
  return {
    textContent: text,
    disabled: false,
    classList: fakeClassList(classes),
    hasAttribute: (name) => Object.prototype.hasOwnProperty.call(attributes, name),
    getAttribute: (name) => (name in attributes ? attributes[name] : null),
    setAttribute: (name, value) => {
      attributes[name] = String(value);
    },
  };
}

function fakeHint({ copyText, i18nKey, classes = [] }) {
  const copy = fakeElement({ text: copyText, attrs: i18nKey ? { "data-i18n": i18nKey } : {} });
  const alt = fakeElement();
  const hint = fakeElement({ classes });
  hint.querySelector = (selector) => (selector === ".install-alt" ? alt : null);
  return { hint, copy, alt };
}

function depthPayload(overrides = {}) {
  return {
    model_weights: {
      local_only: false,
      components: {
        depth_effects: { label: "Depth Effects", packages: true, state: "missing", download_mb: 1340, ...overrides },
        emotion_highlights: { label: "Emotion highlights", packages: false, state: "missing", download_mb: 6 },
        crisper_whisper: { label: "CrisperWhisper", packages: true, state: "ready", download_mb: 3100 },
      },
    },
  };
}

function statusFor(getCapabilities, copyNodes = new Map()) {
  return createCapabilityStatus({
    getCapabilities,
    t: (key, fallback) => fallback,
    getHintCopyNode: (hint) => copyNodes.get(hint) || null,
    hintStateClasses: HINT_STATE_CLASSES,
  });
}

describe("capability status: probe results", () => {
  it("warns once per feature whose check crashed or hung", () => {
    const status = statusFor(() => ({}));
    const discovery = {
      state: "partial",
      failed: {
        captions: { state: "crashed", reason: "access violation", exit_code: 3221225477 },
        depth_effects: { state: "timeout", reason: "no answer in 30 s" },
        otio: { state: "unknown", reason: "child never reached it" },
      },
    };
    expect(status.probeFailureNotice(discovery)).toBe(
      "Optional feature check failed for captions, depth_effects. Those stay off, and the backend is still running. crash.log has the details.",
    );
    expect(status.probeFailureNotice(discovery)).toBe("");

    discovery.failed.silero_vad = { state: "crashed" };
    expect(status.probeFailureNotice(discovery)).toBe(
      "Optional feature check failed for silero_vad. Those stay off, and the backend is still running. crash.log has the details.",
    );
    expect(status.probeFailureNotice(null)).toBe("");
    expect(status.probeFailureNotice({ failed: "nope" })).toBe("");
  });

  it("takes the GPU name from the video AI probe once it reports", () => {
    let caps = { captions: true };
    const status = statusFor(() => caps);
    expect(status.probedGpuName()).toBe("");
    caps = { video_ai: { gpu_name: "NVIDIA GeForce RTX 4070" } };
    expect(status.probedGpuName()).toBe("NVIDIA GeForce RTX 4070");
    caps = null;
    expect(status.probedGpuName()).toBe("");
  });

  it("reads Whisper as checking until the probe reports or fails", () => {
    const status = statusFor(() => ({}));
    expect(status.whisperProbe({ capabilities: {} })).toEqual({ checking: true, failureReason: "" });
    expect(status.whisperProbe({ capabilities: { captions: false } })).toEqual({ checking: false, failureReason: "" });
    expect(status.whisperProbe({ capabilities: { captions: true } })).toEqual({ checking: false, failureReason: "" });
    expect(status.whisperProbe({
      capabilities: {},
      capability_discovery: { failed: { captions: { state: "timeout", reason: "no answer in 30 s" } } },
    })).toEqual({ checking: false, failureReason: "no answer in 30 s" });
    expect(status.whisperProbe({
      capabilities: {},
      capability_discovery: { failed: { captions: { state: "crashed" } } },
    })).toEqual({ checking: false, failureReason: "" });
    expect(status.whisperProbe(null)).toEqual({ checking: true, failureReason: "" });
  });

  it("asks for a Whisper refresh only after Settings showed it and only on a change", () => {
    const status = statusFor(() => ({}));
    const checking = { status: "ok", capabilities: {} };
    const ready = { status: "ok", capabilities: { captions: true, whisper_backend: "faster-whisper" } };
    expect(status.whisperNeedsRefresh(ready)).toBe(false);
    status.noteWhisperShown(checking);
    expect(status.whisperNeedsRefresh({ status: "ok", capabilities: {} })).toBe(false);
    expect(status.whisperNeedsRefresh(ready)).toBe(true);
    status.noteWhisperShown(ready);
    expect(status.whisperNeedsRefresh({ status: "ok", capabilities: { captions: true, whisper_backend: "faster-whisper" } })).toBe(false);
    expect(status.whisperNeedsRefresh({ status: "ok", capabilities: { captions: true, whisper_backend: "faster-whisper", whisper_cpu_mode: true } })).toBe(true);
    expect(status.whisperNeedsRefresh({
      status: "ok",
      capabilities: {},
      capability_discovery: { failed: { captions: { state: "timeout", reason: "no answer in 30 s" } } },
    })).toBe(true);
    expect(status.whisperNeedsRefresh(null)).toBe(true);
    status.noteWhisperShown({ status: "error" });
    expect(status.whisperNeedsRefresh(null)).toBe(false);
  });

  it("leaves main.js the wiring only", () => {
    expect(mainSource).toContain("capabilityStatus.probeFailureNotice(data.capability_discovery)");
    expect(mainSource).toContain("capabilityStatus.whisperProbe(healthData)");
    expect(mainSource).toContain("capabilityStatus.whisperNeedsRefresh(data)");
    expect(mainSource).toContain("capabilityStatus.noteWhisperShown(healthData)");
    expect(mainSource).not.toContain("function whisperStateKey(");
    expect(mainSource).not.toContain("function reportCapabilityProbeFailures(");
  });
});

describe("capability status: model weights", () => {
  it("reports a model as missing only when its packages are installed", () => {
    const caps = depthPayload();
    const status = statusFor(() => caps);
    expect(status.weightsMissing("depth_effects")?.label).toBe("Depth Effects");
    expect(status.weightsMissing("emotion_highlights")).toBeNull();
    expect(status.weightsMissing("crisper_whisper")).toBeNull();
    expect(status.weightsMissing("not_a_component")).toBeNull();
    expect(statusFor(() => ({})).weightsMissing("depth_effects")).toBeNull();
    expect(statusFor(() => null).weightsMissing("depth_effects")).toBeNull();
  });

  it("reads the capabilities the panel holds now, not the ones it started with", () => {
    let caps = {};
    const status = statusFor(() => caps);
    expect(status.weightsMissing("depth_effects")).toBeNull();
    caps = depthPayload();
    expect(status.weightsMissing("depth_effects")?.label).toBe("Depth Effects");
  });

  it("names the download size and explains a local-only block", () => {
    const caps = depthPayload();
    const status = statusFor(() => caps);
    const entry = status.weightsMissing("depth_effects");
    expect(status.weightsHintMessage(entry)).toBe(
      "Depth Effects is installed, but its model (1,340 MB) isn't downloaded yet.",
    );
    expect(status.weightsDownloadBlocked()).toBe(false);

    caps.model_weights.local_only = true;
    expect(status.weightsDownloadBlocked()).toBe(true);
    expect(status.weightsHintMessage(entry)).toBe(
      "Depth Effects is installed, but its model (1,340 MB) isn't downloaded yet. "
        + "Local-only mode blocks the download. Turn it off in Settings to get the model, then turn it back on.",
    );
  });

  it("fills every token in a template and never reads a value as a pattern", () => {
    expect(fillTemplate("{a} and {a}, not {b}", { a: "x" })).toBe("x and x, not {b}");
    expect(fillTemplate("label: {label}", { label: "$& $1" })).toBe("label: $& $1");
  });
});

describe("capability status: hints", () => {
  it("puts a hint into the download state and back to its markup copy", () => {
    let caps = depthPayload();
    const { hint, copy, alt } = fakeHint({ copyText: "Install Depth Effects to use this.", i18nKey: "hint.depth" });
    const button = fakeElement({ text: "Install Now" });
    const status = statusFor(() => caps, new Map([[hint, copy]]));

    status.syncWeightsHint(hint, button, status.weightsMissing("depth_effects"), "common.install_now", "Install Now");
    expect(copy.textContent).toBe("Depth Effects is installed, but its model (1,340 MB) isn't downloaded yet.");
    expect(alt.classList.has("hidden")).toBe(true);
    expect(button.textContent).toBe("Download model");
    expect(button.classList.has("hidden")).toBe(false);
    expect(button.disabled).toBe(false);

    caps = depthPayload({ state: "ready" });
    status.syncWeightsHint(hint, button, status.weightsMissing("depth_effects"), "common.install_now", "Install Now");
    // The test's t() returns the fallback, which is the copy the markup had.
    expect(copy.textContent).toBe("Install Depth Effects to use this.");
    expect(alt.classList.has("hidden")).toBe(false);
    expect(button.textContent).toBe("Install Now");
  });

  it("hides the download button while local-only mode blocks it", () => {
    const caps = depthPayload();
    caps.model_weights.local_only = true;
    const { hint, copy } = fakeHint({ copyText: "Install Depth Effects to use this." });
    const button = fakeElement({ text: "Install Now" });
    const status = statusFor(() => caps, new Map([[hint, copy]]));

    status.syncWeightsHint(hint, button, status.weightsMissing("depth_effects"), "common.install_now", "Install Now");
    expect(button.classList.has("hidden")).toBe(true);
    expect(button.disabled).toBe(true);
  });

  it("leaves a hint alone while it shows install progress or an error", () => {
    const caps = depthPayload();
    const { hint, copy } = fakeHint({ copyText: "Installing Depth Effects (40%)", classes: ["is-info"] });
    const button = fakeElement({ text: "Installing…" });
    const status = statusFor(() => caps, new Map([[hint, copy]]));

    expect(status.hintHasState(hint)).toBe(true);
    status.syncWeightsHint(hint, button, status.weightsMissing("depth_effects"), "common.install_now", "Install Now");
    expect(copy.textContent).toBe("Installing Depth Effects (40%)");
    expect(button.textContent).toBe("Installing…");
  });
});

describe("capability status: weights-only installs", () => {
  it("asks for the model alone and marks it ready on the capabilities current at success", () => {
    let caps = depthPayload();
    const status = statusFor(() => caps);
    let forwarded = null;
    const config = status.weightsInstallConfig("depth_effects", {
      startMessage: "Installing Depth Effects…",
      onSuccess: (result, job) => {
        forwarded = { result, job };
      },
    });
    expect(config.payload).toEqual({ weights_only: true });
    expect(config.startMessage).toBe("Downloading the Depth Effects model… This may take a while.");

    // A poll replaced the payload while the download ran.
    caps = depthPayload();
    config.onSuccess({ ok: true }, { id: "j1" });
    expect(caps.model_weights.components.depth_effects.state).toBe("ready");
    expect(status.weightsMissing("depth_effects")).toBeNull();
    expect(forwarded).toEqual({ result: { ok: true }, job: { id: "j1" } });
  });

  it("leaves a full install alone when the packages are missing", () => {
    const caps = depthPayload();
    const status = statusFor(() => caps);
    const config = status.weightsInstallConfig("emotion_highlights", { startMessage: "Installing emotion tools…" });
    expect(config.payload).toBeUndefined();
    expect(config.startMessage).toBe("Installing emotion tools…");
    config.onSuccess({}, {});
    expect(caps.model_weights.components.emotion_highlights.packages).toBe(true);
    expect(caps.model_weights.components.emotion_highlights.state).toBe("ready");
  });

  it("is the only home of the weights logic", () => {
    expect(mainSource).toContain("OpenCutCapabilityStatus.createCapabilityStatus({");
    for (const moved of [
      "function weightsMissing(",
      "function weightsHintMessage(",
      "function syncWeightsHint(",
      "function weightsInstallConfig(",
    ]) {
      expect(mainSource).not.toContain(moved);
    }
  });
});
