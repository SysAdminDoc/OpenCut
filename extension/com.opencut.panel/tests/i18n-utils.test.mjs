import { createRequire } from "node:module";

import { describe, expect, it } from "vitest";

const require = createRequire(import.meta.url);
const i18n = require("../client/i18n-utils.js");

describe("translate", () => {
  it("resolves key from map, then fallback, then key", () => {
    const map = { "a.b": "Hello", empty: "" };
    expect(i18n.translate(map, "a.b", "fb")).toBe("Hello");
    expect(i18n.translate(map, "missing", "fb")).toBe("fb");
    expect(i18n.translate(map, "missing")).toBe("missing");
    expect(i18n.translate(null, "k", "fb")).toBe("fb");
    expect(i18n.translate(map, "empty", "fb")).toBe("fb"); // empty value is falsy
  });
});

describe("mergeLocale", () => {
  it("overlays own keys onto the base without mutating inputs", () => {
    const base = { a: "1", b: "2" };
    const overlay = { b: "two", c: "3" };
    const merged = i18n.mergeLocale(base, overlay);
    expect(merged).toEqual({ a: "1", b: "two", c: "3" });
    // inputs untouched
    expect(base).toEqual({ a: "1", b: "2" });
    expect(overlay).toEqual({ b: "two", c: "3" });
  });

  it("handles missing base/overlay", () => {
    expect(i18n.mergeLocale(null, { a: "1" })).toEqual({ a: "1" });
    expect(i18n.mergeLocale({ a: "1" }, null)).toEqual({ a: "1" });
    expect(i18n.mergeLocale(null, null)).toEqual({});
  });

  it("copies only own enumerable keys", () => {
    const base = Object.create({ inherited: "x" });
    base.own = "y";
    expect(i18n.mergeLocale(base, null)).toEqual({ own: "y" });
  });
});

describe("createSearchCopy", () => {
  const t = (key, fallback) => fallback;
  const copy = i18n.createSearchCopy(t);

  it("singularises one and pluralises everything else", () => {
    expect(copy.filesIndexed(1)).toBe("1 file indexed");
    expect(copy.filesIndexed(0)).toBe("0 files indexed");
    expect(copy.filesIndexed(2)).toBe("2 files indexed");
    expect(copy.segments(1)).toBe("1 segment");
    expect(copy.projectClips(3)).toBe("3 project clips");
  });

  it("drops the segment clause when there are no segments", () => {
    expect(copy.indexCount(4, 0)).toBe("4 files indexed");
    expect(copy.indexCount(4, 9)).toBe("4 files indexed • 9 segments");
  });

  it("builds the across and progress lines from the same parts", () => {
    expect(copy.indexedAcross(2, 5)).toBe("2 files indexed indexed across 5 segments.");
    expect(copy.indexingProgress(1, 4)).toBe("Indexed 1 of 4 project clips.");
  });

  it("mentions issues only when some occurred", () => {
    expect(copy.indexingToast(3, 3, 0)).toBe("Indexed 3 of 3 project clips.");
    expect(copy.indexingToast(3, 4, 1)).toBe("Indexed 3 of 4 project clips with 1 issue.");
    expect(copy.indexingToast(3, 4, 2)).toBe("Indexed 3 of 4 project clips with 2 issues.");
  });

  it("reads every string through the bound translator", () => {
    const seen = [];
    const bound = i18n.createSearchCopy((key, fallback) => { seen.push(key); return fallback; });
    bound.indexingToast(1, 2, 1);

    expect(seen).toContain("search.indexing_complete_toast");
    expect(seen).toContain("search.indexing_toast_issues");
    expect(seen).toContain("search.project_clip_count");
  });
});
