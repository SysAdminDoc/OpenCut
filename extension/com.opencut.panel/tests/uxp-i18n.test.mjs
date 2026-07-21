import { describe, expect, it } from "vitest";

import { translate, interpolateI18n } from "../../com.opencut.uxp/uxp-utils.js";

describe("translate", () => {
  it("resolves key from map, then fallback, then key", () => {
    const map = { "a.b": "Hello", empty: "" };
    expect(translate(map, "a.b", "fb")).toBe("Hello");
    expect(translate(map, "missing", "fb")).toBe("fb");
    expect(translate(map, "missing")).toBe("missing");
    expect(translate(null, "k", "fb")).toBe("fb");
    // empty string value is falsy → falls back
    expect(translate(map, "empty", "fb")).toBe("fb");
  });
});

describe("interpolateI18n", () => {
  it("replaces every {name} occurrence literally", () => {
    expect(interpolateI18n("Hi {name}, {name}!", { name: "Ada" })).toBe("Hi Ada, Ada!");
    expect(interpolateI18n("{count} of {total}", { count: 2, total: 5 })).toBe("2 of 5");
  });

  it("leaves unknown placeholders intact and coerces values", () => {
    expect(interpolateI18n("{a}-{b}", { a: 1 })).toBe("1-{b}");
    expect(interpolateI18n("v={v}", { v: null })).toBe("v=null");
  });

  it("treats $-sequences in values as literal (no regex replacement specials)", () => {
    expect(interpolateI18n("path {p}", { p: "$1$&\\0" })).toBe("path $1$&\\0");
  });

  it("handles missing/empty values and non-string text", () => {
    expect(interpolateI18n("plain")).toBe("plain");
    expect(interpolateI18n("plain", {})).toBe("plain");
    expect(interpolateI18n(null)).toBe("");
    expect(interpolateI18n(42, {})).toBe("42");
  });
});
