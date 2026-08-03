import { describe, expect, it } from "vitest";

import {
  escapeHtml,
  isVersionAtLeast,
  normalizeReleaseUrl,
  safeDomIdSegment,
} from "../../com.opencut.uxp/uxp-utils.js";

describe("UXP utility escaping", () => {
  it("escapes HTML consistently with the CEP panel helper", () => {
    expect(escapeHtml("<span data-x=\"1 & 2\">Bob's</span>")).toBe(
      "&lt;span data-x=&quot;1 &amp; 2&quot;&gt;Bob&#39;s&lt;/span&gt;",
    );
    expect(escapeHtml(undefined)).toBe("");
  });
});

describe("UXP safe DOM id segments", () => {
  it("normalizes dynamic domain names for generated settings controls", () => {
    expect(safeDomIdSegment(" Video / AI Quality ")).toBe("video-ai-quality");
    expect(safeDomIdSegment("speech_to-text")).toBe("speech_to-text");
    expect(safeDomIdSegment("!!!")).toBe("item");
  });
});

describe("UXP backend version comparison", () => {
  it("accepts checkpoint-capable releases and common version prefixes", () => {
    expect(isVersionAtLeast("1.42.0", "1.42.0")).toBe(true);
    expect(isVersionAtLeast("v1.43.0-beta.1", "1.42.0")).toBe(true);
    expect(isVersionAtLeast("2.0.0", "1.42.0")).toBe(true);
  });

  it("rejects older and malformed releases", () => {
    expect(isVersionAtLeast("1.41.9", "1.42.0")).toBe(false);
    expect(isVersionAtLeast("", "1.42.0")).toBe(false);
    expect(isVersionAtLeast("development", "1.42.0")).toBe(false);
  });
});

describe("UXP release URL boundary", () => {
  it("accepts only canonical HTTPS GitHub release pages", () => {
    expect(normalizeReleaseUrl("https://github.com/SysAdminDoc/OpenCut/releases/tag/v1.46.0"))
      .toBe("https://github.com/SysAdminDoc/OpenCut/releases/tag/v1.46.0");
    expect(normalizeReleaseUrl("https://github.com/SysAdminDoc/OpenCut/releases/"))
      .toBe("https://github.com/SysAdminDoc/OpenCut/releases");
  });

  it("rejects alternate hosts, redirects, and non-HTTPS schemes", () => {
    expect(normalizeReleaseUrl("https://evil.example/releases/tag/v1.46.0")).toBeNull();
    expect(normalizeReleaseUrl("https://github.com.evil/SysAdminDoc/OpenCut/releases")).toBeNull();
    expect(normalizeReleaseUrl("http://github.com/SysAdminDoc/OpenCut/releases")).toBeNull();
    expect(normalizeReleaseUrl("https://github.com/SysAdminDoc/OpenCut/issues/1")).toBeNull();
  });
});
