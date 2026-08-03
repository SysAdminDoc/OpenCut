import { createRequire } from "node:module";

import { describe, expect, it, vi } from "vitest";

const require = createRequire(import.meta.url);
const utils = require("../client/panel-utils.js");

const COMMANDS = [
  { name: "Denoise", tab: "audio", sub: "aud-denoise", keywords: "denoise noise reduce clean" },
  { name: "Silence Removal", tab: "cut", sub: "silence", keywords: "silence remove cut clean" },
  { name: "Export Presets", tab: "export", sub: "exp-platform", keywords: "export youtube tiktok instagram" },
  {
    name: "Future Tool",
    tab: "export",
    sub: "exp-future",
    keywords: "future unavailable",
    runnable: false,
    readiness: "missing_route",
    route_valid: false,
    readiness_reason: "No live route is registered.",
  },
];

function paletteOptions(overrides = {}) {
  const favoriteMap = { "fav-export": COMMANDS[2], "fav-denoise": COMMANDS[0] };
  return {
    items: COMMANDS,
    activeTab: "audio",
    historyKeys: [],
    favoriteIds: [],
    getTabLabel: (tab) => ({ audio: "Audio", cut: "Cut", export: "Export" })[tab] || tab,
    getSubLabel: (sub) => ({
      "aud-denoise": "Denoise",
      silence: "Silence",
      "exp-platform": "Platform",
    })[sub] || sub,
    getFavoriteId: (item) => {
      if (item.name === "Export Presets") return "fav-export";
      if (item.name === "Denoise") return "fav-denoise";
      return "";
    },
    getItemForFavorite: (id) => favoriteMap[id] || null,
    ...overrides,
  };
}

describe("CEP panel utility escaping", () => {
  it("escapes HTML text for dynamic panel markup", () => {
    expect(utils.escapeHtml("<button title=\"O'Reilly & Co\">Run</button>")).toBe(
      "&lt;button title=&quot;O&#39;Reilly &amp; Co&quot;&gt;Run&lt;/button&gt;",
    );
    expect(utils.escapeHtml(null)).toBe("");
  });

  it("escapes Windows paths for double-quoted ExtendScript calls", () => {
    const value = "C:\\Projects\\clip \"A\"\nnext\tline\r";
    expect(utils.escapeJsxDoubleQuotedString(value)).toBe(
      "C:\\\\Projects\\\\clip \\\"A\\\"\\nnext\\tline\\r",
    );
    expect(utils.escapeJsxDoubleQuotedString("")).toBe("");
  });
});

describe("CEP OAuth URL boundary", () => {
  it("preserves secure authorization URLs byte-for-byte", () => {
    const url = "https://accounts.example.test/oauth?scope=read&state=a^b%20c%22d#return";
    expect(utils.normalizeOAuthUrl(url)).toBe(url);
  });

  it("permits only explicit loopback hosts over HTTP", () => {
    const localhost = "http://localhost:5679/oauth/callback?code=a&state=b";
    const ipv4 = "http://127.0.0.1:5679/oauth/callback";
    const ipv6 = "http://[::1]:5679/oauth/callback";

    expect(utils.normalizeOAuthUrl(localhost)).toBe(localhost);
    expect(utils.normalizeOAuthUrl(ipv4)).toBe(ipv4);
    expect(utils.normalizeOAuthUrl(ipv6)).toBe(ipv6);
    expect(utils.normalizeOAuthUrl("http://example.test/oauth")).toBe("");
    expect(utils.normalizeOAuthUrl("http://localhost.example.test/oauth")).toBe("");
  });

  it("rejects malformed, non-web, and control-character URLs", () => {
    expect(utils.normalizeOAuthUrl("javascript:alert(1)")).toBe("");
    expect(utils.normalizeOAuthUrl("file:///C:/secrets.txt")).toBe("");
    expect(utils.normalizeOAuthUrl("https://example.test/\ncmd")).toBe("");
    expect(utils.normalizeOAuthUrl("not a url")).toBe("");
  });
});

describe("CEP release URL boundary", () => {
  it("accepts only canonical HTTPS GitHub release pages", () => {
    expect(utils.normalizeReleaseUrl("https://github.com/SysAdminDoc/OpenCut/releases/tag/v1.46.0"))
      .toBe("https://github.com/SysAdminDoc/OpenCut/releases/tag/v1.46.0");
    expect(utils.normalizeReleaseUrl("https://github.com/SysAdminDoc/OpenCut/releases/"))
      .toBe("https://github.com/SysAdminDoc/OpenCut/releases");
  });

  it("rejects redirects, alternate hosts, and non-HTTPS schemes", () => {
    expect(utils.normalizeReleaseUrl("https://evil.example/releases/tag/v1.46.0")).toBe("");
    expect(utils.normalizeReleaseUrl("https://github.com.evil/SysAdminDoc/OpenCut/releases")).toBe("");
    expect(utils.normalizeReleaseUrl("http://github.com/SysAdminDoc/OpenCut/releases")).toBe("");
    expect(utils.normalizeReleaseUrl("https://github.com/SysAdminDoc/OpenCut/issues/1")).toBe("");
  });
});

describe("CEP backend version comparison", () => {
  it("accepts checkpoint-capable releases and common version prefixes", () => {
    expect(utils.isVersionAtLeast("1.42.0", "1.42.0")).toBe(true);
    expect(utils.isVersionAtLeast("v1.43.0-beta.1", "1.42.0")).toBe(true);
    expect(utils.isVersionAtLeast("2.0.0", "1.42.0")).toBe(true);
  });

  it("rejects older and malformed releases", () => {
    expect(utils.isVersionAtLeast("1.41.9", "1.42.0")).toBe(false);
    expect(utils.isVersionAtLeast("", "1.42.0")).toBe(false);
    expect(utils.isVersionAtLeast("development", "1.42.0")).toBe(false);
  });
});

describe("CEP lazy DOM proxy", () => {
  it("caches successful element lookups by id", () => {
    const node = { id: "runButton" };
    const doc = { getElementById: vi.fn((id) => (id === "runButton" ? node : null)) };
    const proxy = utils.createLazyDomProxy(doc);

    expect(proxy.runButton).toBe(node);
    expect(proxy.runButton).toBe(node);
    expect(doc.getElementById).toHaveBeenCalledTimes(1);
  });

  it("does not cache missing nodes so late-rendered ids can appear", () => {
    const doc = { getElementById: vi.fn(() => null) };
    const proxy = utils.createLazyDomProxy(doc);

    expect(proxy.lateNode).toBe(null);
    expect(proxy.lateNode).toBe(null);
    expect(doc.getElementById).toHaveBeenCalledTimes(2);
  });
});

describe("CEP local settings normalization", () => {
  it("keeps only supported local panel preference keys and values", () => {
    expect(utils.normalizeLocalSettings({
      autoImport: true,
      autoOpen: false,
      showNotifications: true,
      outputDir: "project",
      defaultModel: "large-v3",
      lang: "en",
      theme: "light",
      unexpected: "<script>",
      nested: { value: "custom" },
    })).toEqual({
      autoImport: true,
      autoOpen: false,
      showNotifications: true,
      outputDir: "project",
      defaultModel: "large-v3",
      lang: "en",
      theme: "light",
    });
  });

  it("drops malformed or unsupported imported panel preference values", () => {
    expect(utils.normalizeLocalSettings({
      autoImport: "yes",
      autoOpen: 1,
      showNotifications: null,
      outputDir: "C:/Users/me",
      defaultModel: "malicious",
      lang: "es",
      theme: "solarized",
    })).toEqual({});
    expect(utils.normalizeLocalSettings(null)).toEqual({});
    expect(utils.normalizeLocalSettings([])).toEqual({});
  });
});

describe("CEP command palette indexer", () => {
  it("builds deduplicated browse sections from recent, favorite, and current commands", () => {
    const sections = utils.buildCommandPaletteSections(paletteOptions({
      historyKeys: [utils.getCommandPaletteItemKey(COMMANDS[0])],
      favoriteIds: ["fav-export"],
    }));

    expect(sections.map((section) => section.label)).toEqual(["Recent", "Favorites", "Browse All"]);
    expect(sections[0].entries.map((entry) => entry.item.name)).toEqual(["Denoise"]);
    expect(sections[1].entries[0]).toMatchObject({
      key: utils.getCommandPaletteItemKey(COMMANDS[2]),
      favoriteId: "fav-export",
      isFavorite: true,
    });
    expect(sections.flatMap((section) => section.entries).map((entry) => entry.item.name)).toEqual([
      "Denoise",
      "Export Presets",
      "Silence Removal",
      "Future Tool",
    ]);
  });

  it("ranks query matches by direct name, favorite state, and active workspace", () => {
    const sections = utils.buildCommandPaletteSections(paletteOptions({
      query: "noise",
      historyKeys: [utils.getCommandPaletteItemKey(COMMANDS[0])],
      favoriteIds: ["fav-denoise"],
    }));

    expect(sections).toHaveLength(1);
    expect(sections[0].label).toBe("Matching Tools");
    expect(sections[0].entries[0]).toMatchObject({
      item: COMMANDS[0],
      isCurrent: true,
      isFavorite: true,
      isRecent: true,
    });
    expect(sections[0].entries[0].score).toBeGreaterThan(50);
  });

  it("preserves readiness metadata for disabled command entries", () => {
    const sections = utils.buildCommandPaletteSections(paletteOptions({
      query: "future",
    }));

    expect(sections).toHaveLength(1);
    expect(sections[0].entries[0]).toMatchObject({
      isRunnable: false,
      readiness: "missing_route",
      routeValid: false,
      readinessReason: "No live route is registered.",
    });
  });

  it("resolves localized command descriptions supplied as lazy functions", () => {
    const denoiseKey = utils.getCommandPaletteItemKey(COMMANDS[0]);
    const sections = utils.buildCommandPaletteSections(paletteOptions({
      query: "noise",
      descriptionMap: {
        [denoiseKey]: () => "Clean dialogue while preserving speech detail.",
        _tabDescriptions: {
          audio: () => "Polish dialogue and stems.",
          _default: () => "Open this tool.",
        },
      },
    }));

    expect(sections[0].entries[0].description).toBe(
      "Clean dialogue while preserving speech detail.",
    );
  });
});
