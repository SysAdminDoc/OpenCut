import { createRequire } from "node:module";

import { describe, expect, it } from "vitest";

const require = createRequire(import.meta.url);
const fmt = require("../client/format-utils.js");

describe("safeFixed", () => {
  it("fixes finite numbers", () => {
    expect(fmt.safeFixed(3.14159, 2)).toBe("3.14");
    expect(fmt.safeFixed("2.5", 1)).toBe("2.5");
  });
  it("returns '0' for non-finite / non-numeric", () => {
    expect(fmt.safeFixed("nope", 2)).toBe("0");
    expect(fmt.safeFixed(Infinity, 2)).toBe("0");
    expect(fmt.safeFixed(NaN, 0)).toBe("0");
  });
});

describe("escSingleQuote (ExtendScript single-quoted injection guard)", () => {
  it("escapes backslash first, then quote", () => {
    expect(fmt.escSingleQuote("a\\b")).toBe("a\\\\b");
    expect(fmt.escSingleQuote("it's")).toBe("it\\'s");
  });
  it("escapes newlines, CR, tab and Unicode line separators", () => {
    expect(fmt.escSingleQuote("a\nb")).toBe("a\\nb");
    expect(fmt.escSingleQuote("a\rb")).toBe("a\\rb");
    expect(fmt.escSingleQuote("a\tb")).toBe("a\\tb");
    expect(fmt.escSingleQuote("a\u2028b")).toBe("a\\u2028b");
    expect(fmt.escSingleQuote("a\u2029b")).toBe("a\\u2029b");
  });
  it("handles nullish input", () => {
    expect(fmt.escSingleQuote(null)).toBe("");
    expect(fmt.escSingleQuote(undefined)).toBe("");
  });
  it("does not double-escape independently ordered replacements", () => {
    // backslash-n literal (two chars) stays a literal escaped backslash + n
    expect(fmt.escSingleQuote("\\n")).toBe("\\\\n");
  });
});

describe("fmtDur", () => {
  it("formats seconds as M:SS", () => {
    expect(fmt.fmtDur(0)).toBe("0:00");
    expect(fmt.fmtDur(65)).toBe("1:05");
    expect(fmt.fmtDur(3599)).toBe("59:59");
  });
  it("returns '--' for nullish", () => {
    expect(fmt.fmtDur(null)).toBe("--");
    expect(fmt.fmtDur(undefined)).toBe("--");
  });
});

describe("formatTimecode", () => {
  it("formats as MM:SS.s", () => {
    expect(fmt.formatTimecode(83.44)).toBe("01:23.4");
    expect(fmt.formatTimecode(0)).toBe("00:00.0");
    expect(fmt.formatTimecode()).toBe("00:00.0");
  });
});

describe("getStepPrecision / formatNumberForInput", () => {
  it("counts decimal places of a step", () => {
    expect(fmt.getStepPrecision("0.01")).toBe(2);
    expect(fmt.getStepPrecision("1")).toBe(0);
    expect(fmt.getStepPrecision("")).toBe(0);
  });
  it("formats numbers trimming trailing zeros and -0", () => {
    expect(fmt.formatNumberForInput(5.1, 3)).toBe("5.1");
    expect(fmt.formatNumberForInput(-0, 0)).toBe("0");
    expect(fmt.formatNumberForInput(2.0, 2)).toBe("2");
    expect(fmt.formatNumberForInput(Infinity, 2)).toBe("");
  });
});

describe("extractWordSegments", () => {
  it("flattens words across segments, skipping segments without words", () => {
    expect(fmt.extractWordSegments([{ words: [1, 2] }, {}, { words: [3] }])).toEqual([1, 2, 3]);
    expect(fmt.extractWordSegments([])).toEqual([]);
  });
});
