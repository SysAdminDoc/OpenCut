import { describe, expect, it, vi } from "vitest";
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";

const require = createRequire(import.meta.url);
const theme = require("../client/cep-theme.js");

// Approximate panel greys Premiere reports for its four skins.
const SKINS = {
  darkest: 35,
  dark: 50,
  light: 180,
  lightest: 210,
};

function skinInfo(grey) {
  return {
    panelBackgroundColor: {
      color: { red: grey, green: grey, blue: grey, alpha: 255 },
    },
  };
}

function fakeCsInterface(grey, { throwOnEnv = false } = {}) {
  const listeners = new Map();
  return {
    listeners,
    getHostEnvironment: () => {
      if (throwOnEnv) throw new Error("host unavailable");
      return { appSkinInfo: skinInfo(grey) };
    },
    addEventListener: vi.fn((name, fn) => listeners.set(name, fn)),
    removeEventListener: vi.fn((name) => listeners.delete(name)),
    fire: (name) => listeners.get(name)?.(),
  };
}

function fakeRoot() {
  const classes = new Set();
  return {
    dataset: {},
    classList: {
      toggle: (name, on) => (on ? classes.add(name) : classes.delete(name)),
      contains: (name) => classes.has(name),
    },
  };
}

describe("hostThemeFromSkin", () => {
  it("classifies every Premiere skin", () => {
    expect(theme.hostThemeFromSkin(skinInfo(SKINS.darkest))).toBe("darkest");
    expect(theme.hostThemeFromSkin(skinInfo(SKINS.dark))).toBe("dark");
    expect(theme.hostThemeFromSkin(skinInfo(SKINS.light))).toBe("light");
    expect(theme.hostThemeFromSkin(skinInfo(SKINS.lightest))).toBe("light");
  });

  it("returns null when the skin cannot be read", () => {
    // null, not a guess — the caller must be able to fall back.
    expect(theme.hostThemeFromSkin(null)).toBeNull();
    expect(theme.hostThemeFromSkin({})).toBeNull();
    expect(theme.hostThemeFromSkin({ panelBackgroundColor: {} })).toBeNull();
  });

  it("accepts float channels from builds that report 0-1", () => {
    const floaty = {
      panelBackgroundColor: { color: { red: 0.8, green: 0.8, blue: 0.8 } },
    };
    expect(theme.hostThemeFromSkin(floaty)).toBe("light");
  });

  it("weights channels perceptually", () => {
    // Mid green reads brighter than mid blue at the same numeric value.
    const green = theme.luminanceOf({ red: 0, green: 200, blue: 0 });
    const blue = theme.luminanceOf({ red: 0, green: 0, blue: 200 });
    expect(green).toBeGreaterThan(blue);
  });
});

describe("readHostTheme", () => {
  it("reads the live host skin", () => {
    expect(theme.readHostTheme(fakeCsInterface(SKINS.light))).toBe("light");
  });

  it("returns null when the host throws or is absent", () => {
    expect(theme.readHostTheme(null)).toBeNull();
    expect(theme.readHostTheme({})).toBeNull();
    expect(theme.readHostTheme(fakeCsInterface(35, { throwOnEnv: true }))).toBeNull();
  });
});

describe("resolveTheme", () => {
  it("lets an explicit choice override the host", () => {
    // The whole point of Light/Dark: the host must not overrule the user.
    expect(theme.resolveTheme("light", "darkest", false)).toMatchObject({
      isLight: true,
      source: "user",
    });
    expect(theme.resolveTheme("dark", "light", true)).toMatchObject({
      isLight: false,
      source: "user",
    });
  });

  it("follows the host skin in auto mode", () => {
    expect(theme.resolveTheme("auto", "light", false)).toMatchObject({
      isLight: true,
      premiereTheme: "light",
      source: "host",
    });
    expect(theme.resolveTheme("auto", "darkest", true)).toMatchObject({
      isLight: false,
      premiereTheme: "darkest",
      source: "host",
    });
  });

  it("falls back to the OS only when there is no host skin", () => {
    expect(theme.resolveTheme("auto", null, true)).toMatchObject({
      isLight: true,
      source: "os",
    });
    expect(theme.resolveTheme("auto", null, false)).toMatchObject({
      isLight: false,
      source: "os",
    });
  });
});

describe("applyTheme", () => {
  it("publishes the binary class and the finer-grained host name", () => {
    const root = fakeRoot();
    theme.applyTheme(root, theme.resolveTheme("auto", "darkest", false));
    expect(root.classList.contains("theme-light")).toBe(false);
    expect(root.dataset.premiereTheme).toBe("darkest");
    expect(root.dataset.themeSource).toBe("host");

    theme.applyTheme(root, theme.resolveTheme("auto", "light", false));
    expect(root.classList.contains("theme-light")).toBe(true);
    expect(root.dataset.premiereTheme).toBe("light");
  });

  it("tolerates a missing root", () => {
    expect(() => theme.applyTheme(null, theme.resolveTheme("auto", "light", false))).not.toThrow();
  });
});

describe("createHostThemeSync", () => {
  it("reports host changes and unregisters on dispose", () => {
    const cs = fakeCsInterface(SKINS.darkest);
    const seen = [];
    const sync = theme.createHostThemeSync({
      csInterface: cs,
      onChange: (value) => seen.push(value),
    });

    sync.start();
    expect(sync.isListening()).toBe(true);
    expect(cs.addEventListener).toHaveBeenCalledWith(theme.THEME_COLOR_CHANGED, expect.any(Function));

    cs.fire(theme.THEME_COLOR_CHANGED);
    expect(seen).toEqual(["darkest"]);

    sync.dispose();
    expect(sync.isListening()).toBe(false);
    expect(cs.removeEventListener).toHaveBeenCalledWith(theme.THEME_COLOR_CHANGED, expect.any(Function));
  });

  it("does not stack listeners across repeated starts", () => {
    // Panel reconnects re-run initialisation; duplicates would multiply work
    // and leave handlers behind that dispose() no longer knows about.
    const cs = fakeCsInterface(SKINS.dark);
    const sync = theme.createHostThemeSync({ csInterface: cs });
    sync.start();
    sync.start();
    sync.start();
    expect(cs.addEventListener).toHaveBeenCalledTimes(1);
    sync.dispose();
    expect(cs.listeners.size).toBe(0);
  });

  it("stays inert without a CSInterface", () => {
    const sync = theme.createHostThemeSync({ csInterface: null });
    expect(() => sync.start()).not.toThrow();
    expect(sync.isListening()).toBe(false);
    expect(sync.current()).toBeNull();
    expect(() => sync.dispose()).not.toThrow();
  });

  it("survives a host that rejects listener registration", () => {
    const cs = fakeCsInterface(SKINS.dark);
    cs.addEventListener = vi.fn(() => {
      throw new Error("nope");
    });
    const warn = vi.fn();
    const sync = theme.createHostThemeSync({ csInterface: cs, logger: { warn } });
    expect(() => sync.start()).not.toThrow();
    expect(sync.isListening()).toBe(false);
    expect(warn).toHaveBeenCalled();
  });
});

describe("shipped theme default", () => {
  /* F405: the panel shipped with `selected` on the Dark option. resolveTheme
   * short-circuits on an explicit dark/light preference and only consults the
   * host when the preference is "auto", so on a fresh install none of the
   * host-skin machinery this module exists for ever fired: a user running
   * Premiere's light skin got a dark panel that clashed with the application
   * until they found Settings and changed it themselves. */
  const markup = readFileSync(new URL("../client/index.html", import.meta.url), "utf8");

  function themeSelectOptions() {
    const select = markup.slice(
      markup.indexOf('<select id="settingsTheme">'),
      markup.indexOf("</select>", markup.indexOf('<select id="settingsTheme">')),
    );
    return [...select.matchAll(/<option value="(\w+)"([^>]*)>/g)]
      .map(([, value, attrs]) => ({ value, selected: attrs.includes("selected") }));
  }

  it("defaults the appearance control to following the host", () => {
    const options = themeSelectOptions();
    expect(options.map((option) => option.value)).toEqual(["auto", "dark", "light"]);
    expect(options.filter((option) => option.selected).map((option) => option.value))
      .toEqual(["auto"]);
  });

  it("follows the host skin under that default, in both directions", () => {
    expect(theme.resolveTheme("auto", "light", false)).toMatchObject({
      isLight: true,
      source: "host",
    });
    expect(theme.resolveTheme("auto", "darkest", true)).toMatchObject({
      isLight: false,
      source: "host",
    });
  });

  it("still lets an explicit choice outrank the host", () => {
    expect(theme.resolveTheme("dark", "light", true)).toMatchObject({
      isLight: false,
      source: "user",
    });
    expect(theme.resolveTheme("light", "darkest", false)).toMatchObject({
      isLight: true,
      source: "user",
    });
  });
});
