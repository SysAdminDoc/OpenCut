import { createRequire } from "node:module";

import { describe, expect, it } from "vitest";

const require = createRequire(import.meta.url);
const cl = require("../client/classify-utils.js");

describe("humanizeEngineDomain", () => {
  it("title-cases underscore-separated domains", () => {
    expect(cl.humanizeEngineDomain("silence_detection")).toBe("Silence Detection");
    expect(cl.humanizeEngineDomain("tts")).toBe("Tts");
    expect(cl.humanizeEngineDomain("")).toBe("");
    expect(cl.humanizeEngineDomain(null)).toBe("");
  });
});

describe("engineTemplateText", () => {
  it("substitutes {key} placeholders from a values object", () => {
    expect(cl.engineTemplateText("Use {name} ({vram})", { name: "CLIP", vram: "512MB" })).toBe("Use CLIP (512MB)");
    expect(cl.engineTemplateText("no placeholders", {})).toBe("no placeholders");
  });
});

describe("pluginTrustStateClass", () => {
  it("maps trust/load state to a css class", () => {
    expect(cl.pluginTrustStateClass({ load_status: "failed" })).toBe("warning");
    expect(cl.pluginTrustStateClass({ trust: { errors: ["x"] } })).toBe("warning");
    expect(cl.pluginTrustStateClass({ trust: { unsigned_allowed: true } })).toBe("manual");
    expect(cl.pluginTrustStateClass({ trust: { lock_missing: true } })).toBe("manual");
    expect(cl.pluginTrustStateClass({ load_status: "loaded" })).toBe("auto");
    expect(cl.pluginTrustStateClass({})).toBe("manual");
    expect(cl.pluginTrustStateClass(null)).toBe("manual");
  });
});
