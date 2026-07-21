import { createRequire } from "node:module";

import { describe, expect, it } from "vitest";

const require = createRequire(import.meta.url);
const su = require("../client/string-utils.js");

describe("humanizeControlId", () => {
  it("splits camelCase and separators into title case", () => {
    expect(su.humanizeControlId("autoSilenceCut")).toBe("Auto Silence Cut");
    expect(su.humanizeControlId("min_gap-ms")).toBe("Min Gap Ms");
    expect(su.humanizeControlId("")).toBe("");
    expect(su.humanizeControlId(null)).toBe("");
  });
});

describe("journalClipName / shortsBundleFileName", () => {
  it("returns the basename across separators", () => {
    expect(su.journalClipName("C:/a/b/clip.mp4")).toBe("clip.mp4");
    expect(su.journalClipName("a\\b\\c.mov")).toBe("c.mov");
    expect(su.journalClipName("")).toBe("");
    expect(su.journalClipName(null)).toBe("");
  });

  it("shortsBundleFileName falls back to 'output'", () => {
    expect(su.shortsBundleFileName("/x/y/short.mp4")).toBe("short.mp4");
    expect(su.shortsBundleFileName("")).toBe("output");
    expect(su.shortsBundleFileName(null)).toBe("output");
  });
});

describe("parseTimeToSec", () => {
  it("parses HH:MM:SS, MM:SS and SS forms", () => {
    expect(su.parseTimeToSec("01:02:03")).toBe(3723);
    expect(su.parseTimeToSec("02:30")).toBe(150);
    expect(su.parseTimeToSec("45")).toBe(45);
    expect(su.parseTimeToSec("")).toBe(0);
    expect(su.parseTimeToSec("garbage")).toBe(0);
  });
});

describe("captionDisplayOptionLabel", () => {
  it("annotates font/size/color/opacity categories", () => {
    expect(
      su.captionDisplayOptionLabel(
        { category: "font" },
        { id: "Inter", font_family: "Inter", font_resolution: { source: "preferred_file" } },
      ),
    ).toBe("Inter (Inter, resolved)");
    expect(
      su.captionDisplayOptionLabel(
        { category: "font" },
        { id: "Inter", font_family: "Inter", font_resolution: { source: "system" } },
      ),
    ).toBe("Inter (Inter, fallback)");
    expect(su.captionDisplayOptionLabel({ category: "size" }, { id: "L", font_size: 32 })).toBe("L (32)");
    expect(su.captionDisplayOptionLabel({ category: "color" }, { id: "red", hex: "#f00" })).toBe("red (#f00)");
    expect(su.captionDisplayOptionLabel({ category: "opacity" }, { id: "half", alpha: 0.5 })).toBe("half (0.5)");
    expect(su.captionDisplayOptionLabel({ category: "other" }, { id: "x" })).toBe("x");
  });
});

describe("inferNotificationTone", () => {
  it("honors explicit type, then error data, then message heuristics", () => {
    expect(su.inferNotificationTone("anything", null, "warning")).toBe("warning");
    expect(su.inferNotificationTone("all good", { code: "E1" })).toBe("error");
    expect(su.inferNotificationTone("Export failed")).toBe("error");
    expect(su.inferNotificationTone("Project saved")).toBe("success");
    expect(su.inferNotificationTone("Please select a clip")).toBe("warning");
    expect(su.inferNotificationTone("Just a status")).toBe("info");
    expect(su.inferNotificationTone("", null, "bogus")).toBe("info");
  });
});

describe("getNotificationIconSvg", () => {
  it("returns a distinct svg per tone with info as default", () => {
    const success = su.getNotificationIconSvg("success");
    const info = su.getNotificationIconSvg("nonsense");
    expect(success).toContain("<svg");
    expect(su.getNotificationIconSvg("warning")).toContain("<svg");
    expect(su.getNotificationIconSvg("error")).toContain("<svg");
    expect(info).toContain("<svg");
    expect(success).not.toBe(info);
  });
});

describe("wsFormatListenerCount", () => {
  it("fills {count} and pluralizes {plural}", () => {
    expect(su.wsFormatListenerCount(1, "{count} client{plural}")).toBe("1 client");
    expect(su.wsFormatListenerCount(3, "{count} client{plural}")).toBe("3 clients");
    expect(su.wsFormatListenerCount(0, "{count} listener{plural} online")).toBe("0 listeners online");
  });
});
