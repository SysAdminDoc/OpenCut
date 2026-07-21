import { createRequire } from "node:module";

import { describe, expect, it } from "vitest";

const require = createRequire(import.meta.url);
const jm = require("../client/job-meta-utils.js");

describe("sessionCtxOpText", () => {
  it("title-cases the job type and replaces separators with spaces", () => {
    expect(jm.sessionCtxOpText({ type: "auto_subtitle" })).toBe("Auto subtitle");
    expect(jm.sessionCtxOpText({ type: "silence-removal" })).toBe("Silence removal");
    expect(jm.sessionCtxOpText({})).toBe("Unknown");
  });
});

describe("sessionCtxResultPath", () => {
  it("prefers output_path, then falls back through known keys", () => {
    expect(jm.sessionCtxResultPath({ result: { output_path: "/a.mp4" } })).toBe("/a.mp4");
    expect(jm.sessionCtxResultPath({ result: { srt_path: "/a.srt" } })).toBe("/a.srt");
    expect(jm.sessionCtxResultPath({ result: { output_paths: ["/first.mp4", "/second.mp4"] } })).toBe("/first.mp4");
  });
  it("returns '' when there is no result object", () => {
    expect(jm.sessionCtxResultPath({})).toBe("");
    expect(jm.sessionCtxResultPath({ result: null })).toBe("");
    expect(jm.sessionCtxResultPath({ result: "nope" })).toBe("");
  });
});

describe("getJobHistorySourcePath / getJobHistorySourceName", () => {
  it("resolves the source path from entry fallbacks", () => {
    expect(jm.getJobHistorySourcePath({ sourcePath: "/a.mp4" })).toBe("/a.mp4");
    expect(jm.getJobHistorySourcePath({ payload: { filepath: "/b.mp4" } })).toBe("/b.mp4");
    expect(jm.getJobHistorySourcePath({ filepath: "/c.mp4" })).toBe("/c.mp4");
    expect(jm.getJobHistorySourcePath(null)).toBe("");
    expect(jm.getJobHistorySourcePath({})).toBe("");
  });
  it("returns the basename across both path separators", () => {
    expect(jm.getJobHistorySourceName({ sourcePath: "/x/y/clip.mp4" })).toBe("clip.mp4");
    expect(jm.getJobHistorySourceName({ sourcePath: "C:\\a\\b\\shot.mov" })).toBe("shot.mov");
    expect(jm.getJobHistorySourceName({})).toBe("");
  });
});
