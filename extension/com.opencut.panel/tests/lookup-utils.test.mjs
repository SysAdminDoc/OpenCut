import { createRequire } from "node:module";

import { describe, expect, it } from "vitest";

const require = createRequire(import.meta.url);
const lu = require("../client/lookup-utils.js");

describe("getTranscriptCacheKey", () => {
  it("slugifies non-alphanumerics behind a stable prefix", () => {
    expect(lu.getTranscriptCacheKey("C:/a b/clip.mp4")).toBe("opencut_transcript_C__a_b_clip_mp4");
    expect(lu.getTranscriptCacheKey("")).toBe("opencut_transcript_");
  });
});

describe("parseToUnixSeconds", () => {
  it("passes through numbers and downscales ms/µs magnitudes", () => {
    expect(lu.parseToUnixSeconds(1_700_000_000)).toBe(1_700_000_000);
    expect(lu.parseToUnixSeconds(1_700_000_000_000)).toBe(1_700_000_000);
    expect(lu.parseToUnixSeconds(170_000_000_000)).toBe(170_000_000); // >1e10 (ms) → /1000
    expect(lu.parseToUnixSeconds("2023-01-01T00:00:00Z")).toBe(1672531200);
    expect(lu.parseToUnixSeconds("")).toBe(0);
    expect(lu.parseToUnixSeconds(null)).toBe(0);
    expect(lu.parseToUnixSeconds("not a date")).toBe(0);
  });
});

describe("getOutputItemPath", () => {
  it("returns the path or empty string", () => {
    expect(lu.getOutputItemPath({ path: "/x.mp4" })).toBe("/x.mp4");
    expect(lu.getOutputItemPath({})).toBe("");
    expect(lu.getOutputItemPath(null)).toBe("");
  });
});

describe("normalizeJobOptions", () => {
  it("wraps a bare callback and passes objects through", () => {
    const fn = () => {};
    expect(lu.normalizeJobOptions(fn)).toEqual({ onComplete: fn });
    expect(lu.normalizeJobOptions({ poll: 5 })).toEqual({ poll: 5 });
    expect(lu.normalizeJobOptions(undefined)).toEqual({});
  });
});

describe("matchesShortcut", () => {
  const ev = (over) => ({ ctrlKey: false, shiftKey: false, altKey: false, metaKey: false, key: "", ...over });
  it("matches modifier+key combos exactly", () => {
    expect(lu.matchesShortcut(ev({ ctrlKey: true, shiftKey: true, key: "K" }), "ctrl+shift+k")).toBe(true);
    expect(lu.matchesShortcut(ev({ ctrlKey: true, key: "K" }), "ctrl+shift+k")).toBe(false);
    expect(lu.matchesShortcut(ev({ key: "Escape" }), "escape")).toBe(true);
    expect(lu.matchesShortcut(ev({ metaKey: true, key: "s" }), "cmd+s")).toBe(true);
    expect(lu.matchesShortcut(ev({ key: "a" }), "b")).toBe(false);
  });
});

describe("findSelectOptionByValue", () => {
  const select = { options: [{ value: "a" }, { value: "b" }] };
  it("finds an option by value or returns null", () => {
    expect(lu.findSelectOptionByValue(select, "b")).toBe(select.options[1]);
    expect(lu.findSelectOptionByValue(select, "z")).toBeNull();
    expect(lu.findSelectOptionByValue(null, "a")).toBeNull();
    expect(lu.findSelectOptionByValue(select, "")).toBeNull();
  });
});

describe("getPanelTabName", () => {
  it("strips the panel- prefix", () => {
    expect(lu.getPanelTabName({ id: "panel-cut" })).toBe("cut");
    expect(lu.getPanelTabName({ id: "settings" })).toBe("settings");
    expect(lu.getPanelTabName({})).toBe("");
    expect(lu.getPanelTabName(null)).toBe("");
  });
});

describe("getSelectOptionLabel", () => {
  it("returns the selected option's text or a fallback", () => {
    const sel = { selectedIndex: 1, options: [{ textContent: "A" }, { textContent: "B" }] };
    expect(lu.getSelectOptionLabel(sel, "?")).toBe("B");
    expect(lu.getSelectOptionLabel({ selectedIndex: -1, options: [] }, "none")).toBe("none");
    expect(lu.getSelectOptionLabel(null, "none")).toBe("none");
  });
});
