import { describe, expect, it } from "vitest";

import {
  formatTimecode,
  formatCompactDuration,
  formatBytes,
  humanizeDomain,
  shortsBundleFileNameUxp,
  normalizeHttpsExternalUrl,
  isTimeoutError,
  getSearchResultPath,
  getSearchResultPreview,
} from "../../com.opencut.uxp/uxp-utils.js";

describe("formatTimecode", () => {
  it("formats seconds as HH:MM:SS.ff and guards non-numbers", () => {
    expect(formatTimecode(0)).toBe("00:00:00.00");
    expect(formatTimecode(3661.5)).toBe("01:01:01.50");
    expect(formatTimecode(NaN)).toBe("—");
    expect(formatTimecode("12")).toBe("—");
  });
});

describe("formatCompactDuration", () => {
  it("scales units by magnitude", () => {
    expect(formatCompactDuration(0)).toBe("0 s");
    expect(formatCompactDuration(-5)).toBe("0 s");
    expect(formatCompactDuration(0.45)).toBe("450 ms");
    expect(formatCompactDuration(5)).toBe("5.00 s");
    expect(formatCompactDuration(12.5)).toBe("12.5 s");
    expect(formatCompactDuration(184)).toBe("3m 4s");
  });
});

describe("formatBytes", () => {
  it("scales to B/KB/MB/GB", () => {
    expect(formatBytes(0)).toBe("0 B");
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(2048)).toBe("2.0 KB");
    expect(formatBytes(15 * 1024)).toBe("15 KB");
    expect(formatBytes(1024 * 1024 * 1.5)).toBe("1.5 MB");
    expect(formatBytes(Infinity)).toBe("0 B");
  });
});

describe("humanizeDomain", () => {
  it("title-cases underscore-delimited domains", () => {
    expect(humanizeDomain("silence_cut")).toBe("Silence Cut");
    expect(humanizeDomain("tts")).toBe("Tts");
    expect(humanizeDomain("")).toBe("");
    expect(humanizeDomain(null)).toBe("");
  });
});

describe("shortsBundleFileNameUxp", () => {
  it("returns the basename with an 'output' fallback", () => {
    expect(shortsBundleFileNameUxp("C:/a/b/short.mp4")).toBe("short.mp4");
    expect(shortsBundleFileNameUxp("a\\b\\c.mov")).toBe("c.mov");
    expect(shortsBundleFileNameUxp("")).toBe("output");
    expect(shortsBundleFileNameUxp(null)).toBe("output");
  });
});

describe("normalizeHttpsExternalUrl", () => {
  it("accepts only https urls", () => {
    expect(normalizeHttpsExternalUrl("https://example.com/x")).toBe("https://example.com/x");
    expect(normalizeHttpsExternalUrl("http://example.com")).toBeNull();
    expect(normalizeHttpsExternalUrl("javascript:alert(1)")).toBeNull();
    expect(normalizeHttpsExternalUrl("not a url")).toBeNull();
    expect(normalizeHttpsExternalUrl("")).toBeNull();
  });
});

describe("isTimeoutError", () => {
  it("detects abort/timeout errors", () => {
    expect(isTimeoutError({ name: "AbortError" })).toBe(true);
    expect(isTimeoutError(new Error("request timed out"))).toBe(true);
    expect(isTimeoutError("connection abort")).toBe(true);
    expect(isTimeoutError(new Error("nope"))).toBe(false);
    expect(isTimeoutError(null)).toBe(false);
  });
});

describe("getSearchResultPath / getSearchResultPreview", () => {
  it("coalesces the first non-empty path field", () => {
    expect(getSearchResultPath({ file: "  /a/b.mp4  " })).toBe("/a/b.mp4");
    expect(getSearchResultPath({ clip_path: "c.mov" })).toBe("c.mov");
    expect(getSearchResultPath({})).toBe("");
  });

  it("coalesces, collapses, and truncates the preview text", () => {
    expect(getSearchResultPreview({ snippet: "  a\n\n b  " })).toBe("a b");
    expect(getSearchResultPreview({})).toBe("");
    const long = "x".repeat(200);
    const out = getSearchResultPreview({ text: long });
    expect(out.length).toBe(154);
    expect(out.endsWith("…")).toBe(true);
  });
});
