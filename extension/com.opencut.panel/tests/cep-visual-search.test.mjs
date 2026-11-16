import { createRequire } from "node:module";

import { describe, expect, it } from "vitest";

const require = createRequire(import.meta.url);
const fmt = require("../client/format-utils.js");

const JPEG_URI = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ==";

describe("CEP visual search timestamps", () => {
  it("formats seconds as mm:ss and adds hours only when needed", () => {
    expect(fmt.formatClipTimestamp(0)).toBe("00:00");
    expect(fmt.formatClipTimestamp(5.9)).toBe("00:05");
    expect(fmt.formatClipTimestamp(65)).toBe("01:05");
    expect(fmt.formatClipTimestamp(3599)).toBe("59:59");
    expect(fmt.formatClipTimestamp(3723)).toBe("1:02:03");
  });

  it("treats missing or bad values as the start of the clip", () => {
    expect(fmt.formatClipTimestamp(undefined)).toBe("00:00");
    expect(fmt.formatClipTimestamp("nope")).toBe("00:00");
    expect(fmt.formatClipTimestamp(-4)).toBe("00:00");
    expect(fmt.formatClipTimestamp(Infinity)).toBe("00:00");
  });
});

describe("CEP visual search scores", () => {
  it("turns a 0-1 similarity into a whole percent", () => {
    expect(fmt.scoreToPercent(0.2843)).toBe(28);
    expect(fmt.scoreToPercent(1)).toBe(100);
    expect(fmt.scoreToPercent(0)).toBe(0);
  });

  it("clamps out-of-range scores and skips missing ones", () => {
    expect(fmt.scoreToPercent(1.4)).toBe(100);
    expect(fmt.scoreToPercent(-0.2)).toBe(0);
    expect(fmt.scoreToPercent(undefined)).toBeNull();
    expect(fmt.scoreToPercent("0.5")).toBeNull();
    expect(fmt.scoreToPercent(NaN)).toBeNull();
  });
});

describe("CEP visual search thumbnails", () => {
  it("keeps inline base64 JPEG data URIs", () => {
    expect(fmt.safeThumbnailDataUri(JPEG_URI)).toBe(JPEG_URI);
  });

  it("drops anything that isn't an inline JPEG", () => {
    expect(fmt.safeThumbnailDataUri("")).toBe("");
    expect(fmt.safeThumbnailDataUri(null)).toBe("");
    expect(fmt.safeThumbnailDataUri("C:/Users/someone/.opencut/thumb_0000.jpg")).toBe("");
    expect(fmt.safeThumbnailDataUri("file:///C:/secret.jpg")).toBe("");
    expect(fmt.safeThumbnailDataUri("https://example.com/a.jpg")).toBe("");
    expect(fmt.safeThumbnailDataUri("data:image/svg+xml;base64,PHN2Zz4=")).toBe("");
    expect(fmt.safeThumbnailDataUri('data:image/jpeg;base64,AAAA" onerror="alert(1)')).toBe("");
    expect(fmt.safeThumbnailDataUri(`data:image/jpeg;base64,${"A".repeat(100 * 1024)}`)).toBe("");
  });
});

describe("CEP visual search inputs", () => {
  it("keeps video paths from project media in order without duplicates", () => {
    expect(
      fmt.collectVisualSearchClipPaths([
        { name: "A", path: "C:\\media\\a.mp4", type: "video" },
        "D:/media/b.MOV",
        { path: "C:\\media\\a.mp4" },
        { path: "C:\\media\\voice.wav" },
        { path: "C:\\media\\still.png" },
        { path: "  " },
        { path: 42 },
        null,
        { path: " C:\\media\\c.mxf " },
      ]),
    ).toEqual(["C:\\media\\a.mp4", "D:/media/b.MOV", "C:\\media\\c.mxf"]);
    expect(fmt.collectVisualSearchClipPaths(undefined)).toEqual([]);
  });

  it("sends the project folder only for a saved, local project", () => {
    expect(fmt.visualSearchProjectDir("C:\\Projects\\Doc", "project_path")).toBe("C:\\Projects\\Doc");
    expect(fmt.visualSearchProjectDir(" /Users/me/Edit ", "project_path")).toBe("/Users/me/Edit");
  });

  it("ignores the host's fallback folders and network shares", () => {
    expect(fmt.visualSearchProjectDir("C:\\media", "first_media")).toBe("");
    expect(fmt.visualSearchProjectDir("D:\\Scratch", "scratch_disk")).toBe("");
    expect(fmt.visualSearchProjectDir("C:\\Projects\\Doc", undefined)).toBe("");
    expect(fmt.visualSearchProjectDir("", "project_path")).toBe("");
    expect(fmt.visualSearchProjectDir(undefined, "project_path")).toBe("");
    expect(fmt.visualSearchProjectDir("\\\\nas\\share\\Doc", "project_path")).toBe("");
    expect(fmt.visualSearchProjectDir("//nas/share/Doc", "project_path")).toBe("");
  });
});

describe("CEP visual search cards", () => {
  it("builds a card from a /search/semantic hit", () => {
    expect(
      fmt.buildVisualSearchCard(
        {
          clip_path: "C:\\media\\beach.mp4",
          clip_name: "beach.mp4",
          frame_idx: 3,
          timestamp: 83.4,
          score: 0.3127,
          thumbnail_path: "C:\\Users\\me\\.opencut\\clip_cache\\thumbs\\thumb_0003.jpg",
          thumbnail: JPEG_URI,
        },
        0,
      ),
    ).toEqual({
      index: 0,
      path: "C:\\media\\beach.mp4",
      name: "beach.mp4",
      timestamp: "01:23",
      percent: 31,
      thumbnail: JPEG_URI,
    });
  });

  it("falls back to the file name and leaves bad thumbnails out", () => {
    expect(
      fmt.buildVisualSearchCard({ clip_path: "/media/city/dusk.mov", score: 0.2, thumbnail: "/tmp/x.jpg" }, 2),
    ).toEqual({
      index: 2,
      path: "/media/city/dusk.mov",
      name: "dusk.mov",
      timestamp: "00:00",
      percent: 20,
      thumbnail: "",
    });
    expect(fmt.buildVisualSearchCard(null, 1)).toEqual({
      index: 1,
      path: "",
      name: "",
      timestamp: "00:00",
      percent: null,
      thumbnail: "",
    });
  });
});
