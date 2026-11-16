import { describe, expect, it } from "vitest";

import {
  buildVisualSearchCard,
  collectVisualSearchClipPaths,
  formatClipTimestamp,
  projectFolderFromPath,
  safeThumbnailDataUri,
  scoreToPercent,
} from "../../com.opencut.uxp/uxp-utils.js";

const JPEG_URI = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ==";

describe("UXP visual search timestamps", () => {
  it("formats seconds as mm:ss and adds hours only when needed", () => {
    expect(formatClipTimestamp(0)).toBe("00:00");
    expect(formatClipTimestamp(5.9)).toBe("00:05");
    expect(formatClipTimestamp(65)).toBe("01:05");
    expect(formatClipTimestamp(3599)).toBe("59:59");
    expect(formatClipTimestamp(3723)).toBe("1:02:03");
  });

  it("treats missing or bad values as the start of the clip", () => {
    expect(formatClipTimestamp(undefined)).toBe("00:00");
    expect(formatClipTimestamp("nope")).toBe("00:00");
    expect(formatClipTimestamp(-4)).toBe("00:00");
    expect(formatClipTimestamp(Infinity)).toBe("00:00");
  });
});

describe("UXP visual search scores", () => {
  it("turns a 0-1 similarity into a whole percent", () => {
    expect(scoreToPercent(0.2843)).toBe(28);
    expect(scoreToPercent(1)).toBe(100);
    expect(scoreToPercent(0)).toBe(0);
  });

  it("clamps out-of-range scores and skips missing ones", () => {
    expect(scoreToPercent(1.4)).toBe(100);
    expect(scoreToPercent(-0.2)).toBe(0);
    expect(scoreToPercent(undefined)).toBeNull();
    expect(scoreToPercent("0.5")).toBeNull();
    expect(scoreToPercent(NaN)).toBeNull();
  });
});

describe("UXP visual search thumbnails", () => {
  it("keeps inline base64 JPEG data URIs", () => {
    expect(safeThumbnailDataUri(JPEG_URI)).toBe(JPEG_URI);
  });

  it("drops anything that isn't an inline JPEG", () => {
    expect(safeThumbnailDataUri("")).toBe("");
    expect(safeThumbnailDataUri(undefined)).toBe("");
    expect(safeThumbnailDataUri("C:/Users/someone/.opencut/thumb_0000.jpg")).toBe("");
    expect(safeThumbnailDataUri("file:///C:/secret.jpg")).toBe("");
    expect(safeThumbnailDataUri("https://example.com/a.jpg")).toBe("");
    expect(safeThumbnailDataUri("data:image/svg+xml;base64,PHN2Zz4=")).toBe("");
    expect(safeThumbnailDataUri('data:image/jpeg;base64,AAAA" onerror="alert(1)')).toBe("");
    expect(safeThumbnailDataUri(`data:image/jpeg;base64,${"A".repeat(100 * 1024)}`)).toBe("");
  });
});

describe("UXP visual search inputs", () => {
  it("keeps video paths from project items in order without duplicates", () => {
    expect(
      collectVisualSearchClipPaths([
        { name: "A", path: "C:\\media\\a.mp4" },
        "D:/media/b.MOV",
        { path: "C:\\media\\a.mp4" },
        { path: "C:\\media\\voice.wav" },
        { path: "C:\\media\\still.png" },
        { path: "  " },
        null,
        { path: " C:\\media\\c.mxf " },
      ]),
    ).toEqual(["C:\\media\\a.mp4", "D:/media/b.MOV", "C:\\media\\c.mxf"]);
    expect(collectVisualSearchClipPaths(undefined)).toEqual([]);
  });

  it("finds the folder of a saved project file", () => {
    expect(projectFolderFromPath("C:\\Projects\\Doc\\Doc.prproj")).toBe("C:\\Projects\\Doc");
    expect(projectFolderFromPath("/Users/me/Edit/cut.PRPROJ")).toBe("/Users/me/Edit");
    expect(projectFolderFromPath("D:\\cut.prproj")).toBe("D:\\");
  });

  it("sends no project folder for unsaved, odd or network projects", () => {
    expect(projectFolderFromPath("")).toBe("");
    expect(projectFolderFromPath(undefined)).toBe("");
    expect(projectFolderFromPath("Untitled")).toBe("");
    expect(projectFolderFromPath("C:\\Projects\\notes.txt")).toBe("");
    expect(projectFolderFromPath("cut.prproj")).toBe("");
    expect(projectFolderFromPath("\\\\nas\\share\\cut.prproj")).toBe("");
    expect(projectFolderFromPath("//nas/share/cut.prproj")).toBe("");
  });
});

describe("UXP visual search cards", () => {
  it("builds a card from a /search/semantic hit", () => {
    expect(
      buildVisualSearchCard(
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
      buildVisualSearchCard({ clip_path: "/media/city/dusk.mov", score: 0.2, thumbnail: "/tmp/x.jpg" }, 2),
    ).toEqual({
      index: 2,
      path: "/media/city/dusk.mov",
      name: "dusk.mov",
      timestamp: "00:00",
      percent: 20,
      thumbnail: "",
    });
    expect(buildVisualSearchCard(null, 1)).toEqual({
      index: 1,
      path: "",
      name: "",
      timestamp: "00:00",
      percent: null,
      thumbnail: "",
    });
  });
});
