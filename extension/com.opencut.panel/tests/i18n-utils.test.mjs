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
