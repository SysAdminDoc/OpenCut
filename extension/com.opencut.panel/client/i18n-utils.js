/* ============================================================
   OpenCut CEP Panel - Pure i18n helpers
   Extracted from main.js so they can be unit-tested in isolation.
   Loaded as a classic script (window.OpenCutI18n) and as a
   CommonJS module (vitest). Pure: the controller binds these to
   the live translation map and locale JSON — no DOM, XHR, or
   shared state lives here.
   ============================================================ */
(function (root, factory) {
    "use strict";
    var api = factory();
    if (typeof module === "object" && module.exports) module.exports = api;
    if (root) root.OpenCutI18n = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
    "use strict";

    // Look up a key in a translation map, falling back to `fallback` then `key`.
    function translate(map, key, fallback) {
        return (map && map[key]) || fallback || key;
    }

    // Overlay a locale's own keys onto an English base, returning a new object
    // so missing keys fall back to the base. Mirrors the loadLocale merge.
    function mergeLocale(base, overlay) {
        var merged = {};
        var k;
        if (base) {
            for (k in base) {
                if (Object.prototype.hasOwnProperty.call(base, k)) merged[k] = base[k];
            }
        }
        if (overlay) {
            for (k in overlay) {
                if (Object.prototype.hasOwnProperty.call(overlay, k)) merged[k] = overlay[k];
            }
        }
        return merged;
    }

    return {
        translate: translate,
        mergeLocale: mergeLocale
    };
});
