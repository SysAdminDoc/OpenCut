/* OpenCut CEP host-theme sync. Classic script + CommonJS for tests.
 *
 * "Auto" used to read `prefers-color-scheme`, which is the operating system's
 * setting, not Premiere's. A user running a dark OS with Premiere's light skin
 * (or the reverse) got a panel that clashed with the host, and changing the
 * skin inside Premiere did nothing until the panel reloaded.
 *
 * CEP reports the host skin through `getHostEnvironment().appSkinInfo` and
 * fires `CSInterface.THEME_COLOR_CHANGED_EVENT` when the user switches it, so
 * both the initial value and later changes come from the host.
 *
 * The panel stylesheet is binary (`html.theme-light` versus the dark default),
 * but Premiere exposes four skins. The finer-grained name is still published
 * as `data-premiere-theme` so it matches the UXP vocabulary in
 * `com.opencut.uxp/uxp-theme.js` and is visible to diagnostics and tests.
 */
(function (root, factory) {
    "use strict";
    var api = factory();
    if (typeof module === "object" && module.exports) module.exports = api;
    if (root) root.OpenCutCepTheme = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
    "use strict";

    var THEME_COLOR_CHANGED = "com.adobe.csxs.events.ThemeColorChanged";

    // Premiere's four skins report panel greys near #232323 (35), #323232
    // (50), #B4B4B4 (180) and #D2D2D2 (210). The thresholds sit midway between
    // those values rather than on them, so a build that shifts a shade by a
    // few points does not flip category.
    var LIGHT_MIN_LUMINANCE = 115;
    var DARK_MIN_LUMINANCE = 42;

    function _channel(value) {
        var n = Number(value);
        if (!isFinite(n)) return 0;
        if (n < 0) return 0;
        // CEP reports 0-255, but some builds report 0-1 floats.
        if (n <= 1 && n > 0 && !Number.isInteger(n)) return n * 255;
        return n > 255 ? 255 : n;
    }

    /* Rec. 601 luma — matches how the eye weights the channels, so a mid
     * green does not read as darker than a mid blue. */
    function luminanceOf(color) {
        if (!color) return null;
        var r = _channel(color.red);
        var g = _channel(color.green);
        var b = _channel(color.blue);
        return 0.299 * r + 0.587 * g + 0.114 * b;
    }

    /* Classify the host skin. Returns null when the skin cannot be read, so
     * callers can fall back rather than guessing a theme. */
    function hostThemeFromSkin(appSkinInfo) {
        if (!appSkinInfo) return null;
        var panel = appSkinInfo.panelBackgroundColor;
        var color = panel && panel.color ? panel.color : null;
        var luma = luminanceOf(color);
        if (luma === null) return null;
        if (luma >= LIGHT_MIN_LUMINANCE) return "light";
        if (luma >= DARK_MIN_LUMINANCE) return "dark";
        return "darkest";
    }

    function readHostTheme(csInterface) {
        if (!csInterface || typeof csInterface.getHostEnvironment !== "function") return null;
        try {
            var env = csInterface.getHostEnvironment();
            return hostThemeFromSkin(env && env.appSkinInfo);
        } catch (err) {
            return null;
        }
    }

    /* An explicit Light/Dark choice always wins — Auto is the only mode that
     * defers to the host. `osPrefersLight` is the last resort for panels
     * running outside Premiere (browser development). */
    function resolveTheme(pref, hostTheme, osPrefersLight) {
        if (pref === "light") return { isLight: true, premiereTheme: "light", source: "user" };
        if (pref === "dark") return { isLight: false, premiereTheme: "dark", source: "user" };
        if (hostTheme) {
            return {
                isLight: hostTheme === "light",
                premiereTheme: hostTheme,
                source: "host"
            };
        }
        return {
            isLight: !!osPrefersLight,
            premiereTheme: osPrefersLight ? "light" : "darkest",
            source: "os"
        };
    }

    function applyTheme(root, resolved) {
        if (!root || !root.classList || !resolved) return resolved;
        root.classList.toggle("theme-light", !!resolved.isLight);
        if (root.dataset) {
            root.dataset.premiereTheme = resolved.premiereTheme;
            root.dataset.themeSource = resolved.source;
        } else if (typeof root.setAttribute === "function") {
            root.setAttribute("data-premiere-theme", resolved.premiereTheme);
            root.setAttribute("data-theme-source", resolved.source);
        }
        return resolved;
    }

    /* Owns the host-theme listener.
     *
     * `start()` is idempotent so panel reconnects cannot stack duplicate
     * listeners, and `dispose()` unregisters so a torn-down panel does not
     * keep a dead handler wired to the host. */
    function createHostThemeSync(options) {
        var opts = options || {};
        var csInterface = opts.csInterface || null;
        var onChange = typeof opts.onChange === "function" ? opts.onChange : null;
        var logger = opts.logger || null;
        var handler = null;

        function current() {
            return readHostTheme(csInterface);
        }

        function dispose() {
            if (!handler) return;
            try {
                if (csInterface && typeof csInterface.removeEventListener === "function") {
                    csInterface.removeEventListener(THEME_COLOR_CHANGED, handler);
                }
            } catch (err) {
                if (logger && logger.warn) logger.warn("[OpenCut] Could not remove host theme listener:", err);
            }
            handler = null;
        }

        function start() {
            if (handler) return dispose;
            if (!csInterface || typeof csInterface.addEventListener !== "function") return dispose;
            handler = function () {
                var theme = current();
                if (onChange) onChange(theme);
            };
            try {
                csInterface.addEventListener(THEME_COLOR_CHANGED, handler);
            } catch (err) {
                if (logger && logger.warn) logger.warn("[OpenCut] Could not observe host theme:", err);
                handler = null;
            }
            return dispose;
        }

        return {
            start: start,
            dispose: dispose,
            current: current,
            isListening: function () { return !!handler; }
        };
    }

    return {
        THEME_COLOR_CHANGED: THEME_COLOR_CHANGED,
        LIGHT_MIN_LUMINANCE: LIGHT_MIN_LUMINANCE,
        DARK_MIN_LUMINANCE: DARK_MIN_LUMINANCE,
        luminanceOf: luminanceOf,
        hostThemeFromSkin: hostThemeFromSkin,
        readHostTheme: readHostTheme,
        resolveTheme: resolveTheme,
        applyTheme: applyTheme,
        createHostThemeSync: createHostThemeSync
    };
});
