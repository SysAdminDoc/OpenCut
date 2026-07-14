import { describe, expect, it } from "vitest";

import {
  buildSmartBinRules,
  computeInverseRenames,
  expandRenamePattern,
  summarizeRenamePreview,
} from "../../com.opencut.uxp/uxp-utils.js";

const ITEMS = [
  { name: "clip a.mp4", path: "/proj/raw/clip a.mp4" },
  { name: "clip b.mov", path: "/proj/raw/clip b.mov" },
  { name: "song.wav", path: "/proj/audio/song.wav" },
];

describe("expandRenamePattern", () => {
  it("expands stem/ext/index tokens with zero padding", () => {
    const renames = expandRenamePattern(ITEMS, "{stem}_{index:03d}{ext}");
    expect(renames.map((r) => r.newName)).toEqual([
      "clip a_001.mp4",
      "clip b_002.mov",
      "song_003.wav",
    ]);
    expect(renames[0]).toMatchObject({ oldName: "clip a.mp4", path: "/proj/raw/clip a.mp4" });
  });

  it("defaults the pattern and carries a bare index token", () => {
    const renames = expandRenamePattern(ITEMS, "");
    expect(renames).toHaveLength(3);
    expect(expandRenamePattern([{ name: "x.mp4" }], "take_{index}")[0].newName).toBe("take_1");
  });

  it("skips unchanged names and within-batch collisions", () => {
    // A pattern that ignores the index collapses everything to one name.
    const renames = expandRenamePattern(ITEMS, "fixed{ext}");
    expect(renames.map((r) => r.newName)).toEqual(["fixed.mp4", "fixed.mov", "fixed.wav"]);
    // Identical output name => only the first survives.
    const collide = expandRenamePattern(
      [{ name: "a.mp4" }, { name: "b.mp4" }],
      "same.mp4",
    );
    expect(collide).toHaveLength(1);
  });

  it("ignores empty and non-array input", () => {
    expect(expandRenamePattern(null, "{stem}")).toEqual([]);
    expect(expandRenamePattern([{ name: "" }], "{stem}_{index}")).toEqual([]);
  });
});

describe("computeInverseRenames (undo)", () => {
  it("swaps old and new names so applying it reverts the batch", () => {
    const renames = expandRenamePattern(ITEMS, "{stem}_{index:03d}{ext}");
    const inverse = computeInverseRenames(renames);
    expect(inverse[0]).toMatchObject({ oldName: "clip a_001.mp4", newName: "clip a.mp4" });
    expect(inverse).toHaveLength(renames.length);
  });
});

describe("buildSmartBinRules", () => {
  it("groups by file type (unique, uppercased)", () => {
    const rules = buildSmartBinRules(ITEMS, "File Type");
    expect(rules.map((r) => r.name).sort()).toEqual(["MOV", "MP4", "WAV"]);
  });

  it("groups by parent folder for the default strategy", () => {
    const rules = buildSmartBinRules(ITEMS, "Folder");
    expect(rules.map((r) => r.name).sort()).toEqual(["audio", "raw"]);
  });

  it("returns [] for empty input", () => {
    expect(buildSmartBinRules([], "File Type")).toEqual([]);
  });
});

describe("summarizeRenamePreview", () => {
  it("reports count and a bounded sample", () => {
    const renames = expandRenamePattern(ITEMS, "{stem}_{index}{ext}");
    const preview = summarizeRenamePreview(renames, 2);
    expect(preview.count).toBe(3);
    expect(preview.sample).toHaveLength(2);
    expect(preview.sample[0]).toContain("->");
  });
});
