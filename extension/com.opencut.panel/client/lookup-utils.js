/* ============================================================
   OpenCut CEP Panel - Pure lookup / parse helpers
   Extracted from main.js so they can be unit-tested in isolation.
   Loaded as a classic script (window.OpenCutLookup) and as a
   CommonJS module (vitest). Reads only their arguments (including
   passed-in DOM objects/events) — no document, globals, i18n,
   shared state, or mutation.
   ============================================================ */
(function (root, factory) {
    "use strict";
    var api = factory();
    if (typeof module === "object" && module.exports) module.exports = api;
    if (root) root.OpenCutLookup = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
    "use strict";

    function getTranscriptCacheKey(filepath) {
        return "opencut_transcript_" + filepath.replace(/[^a-zA-Z0-9]/g, "_");
    }

    function parseToUnixSeconds(value) {
        if (value == null || value === "") return 0;
        if (typeof value === "number") {
            if (value > 1000000000000) return Math.floor(value / 1000);
            if (value > 10000000000) return Math.floor(value / 1000);
            return Math.floor(value);
        }
        var parsed = Date.parse(value);
        return isNaN(parsed) ? 0 : Math.floor(parsed / 1000);
    }

    function getOutputItemPath(item) {
        return (item && item.path) || "";
    }

    function normalizeJobOptions(options) {
        if (typeof options === "function") {
            return { onComplete: options };
        }
        return options || {};
    }

    function matchesShortcut(e, keysStr) {
        var parts = keysStr.split("+");
        var needCtrl = false, needShift = false, needAlt = false, needMeta = false;
        var keyPart = "";
        for (var i = 0; i < parts.length; i++) {
            var p = parts[i].trim().toLowerCase();
            if (p === "ctrl") needCtrl = true;
            else if (p === "shift") needShift = true;
            else if (p === "alt") needAlt = true;
            else if (p === "meta" || p === "cmd") needMeta = true;
            else keyPart = p;
        }
        if (e.ctrlKey !== needCtrl) return false;
        if (e.shiftKey !== needShift) return false;
        if (e.altKey !== needAlt) return false;
        if (e.metaKey !== needMeta) return false;
        var eventKey = e.key.toLowerCase();
        if (keyPart === "escape") return eventKey === "escape";
        return eventKey === keyPart;
    }

    function findSelectOptionByValue(select, value) {
        if (!select || !value) return null;
        for (var i = 0; i < select.options.length; i++) {
            if (select.options[i].value === value) return select.options[i];
        }
        return null;
    }

    function getPanelTabName(panel) {
        if (!panel || !panel.id) return "";
        return panel.id.indexOf("panel-") === 0 ? panel.id.substring(6) : panel.id;
    }

    function getSelectOptionLabel(selectEl, fallback) {
        if (!selectEl) return fallback || "";
        var opt = selectEl.selectedIndex >= 0 ? selectEl.options[selectEl.selectedIndex] : null;
        return opt ? opt.textContent : (fallback || "");
    }

    return {
        getTranscriptCacheKey: getTranscriptCacheKey,
        parseToUnixSeconds: parseToUnixSeconds,
        getOutputItemPath: getOutputItemPath,
        normalizeJobOptions: normalizeJobOptions,
        matchesShortcut: matchesShortcut,
        findSelectOptionByValue: findSelectOptionByValue,
        getPanelTabName: getPanelTabName,
        getSelectOptionLabel: getSelectOptionLabel
    };
});
