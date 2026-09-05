import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { describe, expect, it } from "vitest";

const require = createRequire(import.meta.url);
const { DEFAULT_PORT, portFromPayload, createBridgePortTracker } = require("../client/ws-bridge-port.js");

const clientDir = join(dirname(fileURLToPath(import.meta.url)), "..", "client");
const mainSource = readFileSync(join(clientDir, "main.js"), "utf8");

describe("live-updates bridge port", () => {
  it("takes the port the backend reported", () => {
    const tracker = createBridgePortTracker();
    expect(tracker.needsResolve()).toBe(true);
    expect(tracker.current()).toBeNull();

    tracker.remember({ port: 5684 });

    expect(tracker.needsResolve()).toBe(false);
    expect(tracker.effective()).toBe(5684);
  });

  it("falls back to 5680 only until something answers", () => {
    const tracker = createBridgePortTracker();
    expect(tracker.effective()).toBe(DEFAULT_PORT);
    tracker.remember({ port: 5689 });
    expect(tracker.effective()).toBe(5689);
  });

  it("ignores a payload that carries no usable port", () => {
    const tracker = createBridgePortTracker();
    tracker.remember({ port: 5685 });

    for (const junk of [null, undefined, {}, { port: null }, { port: "" }, { port: "abc" }]) {
      tracker.remember(junk);
      expect(tracker.effective(), JSON.stringify(junk)).toBe(5685);
    }
  });

  it("refuses ports outside the usable range", () => {
    expect(portFromPayload({ port: 80 })).toBeNull();
    expect(portFromPayload({ port: 70000 })).toBeNull();
    expect(portFromPayload({ port: 5680.5 })).toBeNull();
    expect(portFromPayload({ port: 1024 })).toBe(1024);
    expect(portFromPayload({ port: 65535 })).toBe(65535);
  });

  it("accepts a numeric string, which is what JSON round-trips sometimes give", () => {
    expect(portFromPayload({ port: "5683" })).toBe(5683);
  });

  it("forgets the port so a reconnect asks again", () => {
    const tracker = createBridgePortTracker();
    tracker.remember({ port: 5686 });
    tracker.forget();
    expect(tracker.needsResolve()).toBe(true);
  });
});

describe("the panel uses the tracker rather than a literal", () => {
  it("does not dial a hardcoded port", () => {
    // The original bug: `var port = 5680;` regardless of what the backend said.
    expect(mainSource).not.toContain("var port = 5680;");
  });

  it("resolves the port before opening the socket", () => {
    // Reading the tracker synchronously was not enough: on a fresh panel load
    // nothing had answered yet, so the first connect still used the fallback.
    const start = mainSource.indexOf("function wsConnect(");
    expect(start, "wsConnect was renamed; this test no longer reads it").toBeGreaterThan(-1);
    const body = mainSource.slice(start, start + 1400);
    expect(body).toContain("needsResolve()");
    expect(body).toContain('"/ws/status"');
    expect(body).toContain("effective()");
  });

  it("records the port from both endpoints", () => {
    const start = mainSource.slice(mainSource.indexOf('api("POST", "/ws/start"'), mainSource.indexOf('api("POST", "/ws/start"') + 700);
    const status = mainSource.slice(mainSource.indexOf('api("GET", "/ws/status"'), mainSource.indexOf('api("GET", "/ws/status"') + 500);
    expect(start).toContain("_wsBridgePort.remember(");
    expect(status).toContain("_wsBridgePort.remember(");
  });
});
