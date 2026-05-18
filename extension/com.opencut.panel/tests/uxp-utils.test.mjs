import { describe, expect, it } from "vitest";

import { escapeHtml, safeDomIdSegment } from "../../com.opencut.uxp/uxp-utils.js";

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
