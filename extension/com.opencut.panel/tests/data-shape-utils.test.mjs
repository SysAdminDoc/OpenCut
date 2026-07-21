import { createRequire } from "node:module";

import { describe, expect, it } from "vitest";

const require = createRequire(import.meta.url);
const ds = require("../client/data-shape-utils.js");

describe("normalizeWorkspaceState", () => {
  it("fills defaults for missing/mistyped fields", () => {
    expect(ds.normalizeWorkspaceState(null)).toEqual({
      activeNav: "cut",
      activeSubs: {},
      selectedPath: "",
      selectedName: "",
    });
    expect(
      ds.normalizeWorkspaceState({
        activeNav: "captions",
        activeSubs: { cut: "trim" },
        selectedPath: "C:/a.mp4",
        selectedName: "a.mp4",
        extra: 1,
      }),
    ).toEqual({
      activeNav: "captions",
      activeSubs: { cut: "trim" },
      selectedPath: "C:/a.mp4",
      selectedName: "a.mp4",
    });
  });

  it("drops non-object activeSubs and non-string scalars", () => {
    const out = ds.normalizeWorkspaceState({ activeNav: 3, activeSubs: [], selectedPath: 5 });
    expect(out.activeNav).toBe("cut");
    expect(out.selectedPath).toBe("");
    // arrays are objects, so activeSubs (an array) is kept as-is
    expect(Array.isArray(out.activeSubs)).toBe(true);
  });
});

describe("normalizeNavScrollState", () => {
  it("keeps only finite non-negative numbers, rounded", () => {
    expect(ds.normalizeNavScrollState({ a: 12.4, b: -1, c: "x", d: Infinity, e: 0 })).toEqual({
      a: 12,
      e: 0,
    });
    expect(ds.normalizeNavScrollState(null)).toEqual({});
    expect(ds.normalizeNavScrollState("nope")).toEqual({});
  });
});

describe("languageOptionLabel", () => {
  it("prefers name > label > native_name > code > fallback", () => {
    expect(ds.languageOptionLabel({ name: "English", code: "en" }, "x")).toBe("English");
    expect(ds.languageOptionLabel({ native_name: "Deutsch" }, "de")).toBe("Deutsch");
    expect(ds.languageOptionLabel({ code: "fr" }, "x")).toBe("fr");
    expect(ds.languageOptionLabel({}, "fallback")).toBe("fallback");
    expect(ds.languageOptionLabel(null, "auto")).toBe("auto");
    expect(ds.languageOptionLabel("es")).toBe("es");
  });
});

describe("normalizeLanguageOptions", () => {
  it("builds a code->label map from arrays", () => {
    expect(
      ds.normalizeLanguageOptions([
        { code: "en", name: "English" },
        { id: "fr", label: "French" },
        "es",
        null,
      ]),
    ).toEqual({ en: "English", fr: "French", es: "es" });
  });

  it("builds a map from an object and returns {} for junk", () => {
    expect(ds.normalizeLanguageOptions({ en: { name: "English" }, de: "German" })).toEqual({
      en: "English",
      de: "German",
    });
    expect(ds.normalizeLanguageOptions(null)).toEqual({});
  });
});

describe("getTranscriptTotalDuration", () => {
  it("returns the max segment end, falling back to start", () => {
    expect(
      ds.getTranscriptTotalDuration({ segments: [{ start: 0, end: 3 }, { start: 5, end: 9 }] }),
    ).toBe(9);
    expect(ds.getTranscriptTotalDuration({ segments: [{ start: 4 }] })).toBe(4);
    expect(ds.getTranscriptTotalDuration(null)).toBe(0);
    expect(ds.getTranscriptTotalDuration({})).toBe(0);
  });
});

describe("polishStepsFromResult", () => {
  it("indexes steps by key", () => {
    expect(
      ds.polishStepsFromResult({ steps: [{ key: "a", ok: true }, { key: "b", ok: false }] }),
    ).toEqual({ a: { key: "a", ok: true }, b: { key: "b", ok: false } });
    expect(ds.polishStepsFromResult(null)).toEqual({});
  });
});
