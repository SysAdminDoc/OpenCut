/* ============================================================
   OpenCut CEP Panel - Pure data-shape / normalization helpers
   Extracted from main.js so they can be unit-tested in isolation.
   Loaded as a classic script (window.OpenCutDataShape) and as a
   CommonJS module (vitest). No DOM, i18n, or shared-state access.
   ============================================================ */
(function (root, factory) {
    "use strict";
    var api = factory();
    if (typeof module === "object" && module.exports) module.exports = api;
    if (root) root.OpenCutDataShape = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
    "use strict";

    function normalizeWorkspaceState(saved) {
        return {
            activeNav: saved && typeof saved.activeNav === "string" ? saved.activeNav : "cut",
            activeSubs: saved && saved.activeSubs && typeof saved.activeSubs === "object" ? saved.activeSubs : {},
            selectedPath: saved && typeof saved.selectedPath === "string" ? saved.selectedPath : "",
            selectedName: saved && typeof saved.selectedName === "string" ? saved.selectedName : ""
        };
    }

    function normalizeNavScrollState(saved) {
        var normalized = {};
        var key;
        if (!saved || typeof saved !== "object") return normalized;
        for (key in saved) {
            if (Object.prototype.hasOwnProperty.call(saved, key) &&
                typeof saved[key] === "number" &&
                isFinite(saved[key]) &&
                saved[key] >= 0) {
                normalized[key] = Math.round(saved[key]);
            }
        }
        return normalized;
    }

    function languageOptionLabel(value, fallback) {
        if (value && typeof value === "object") {
            return String(value.name || value.label || value.native_name || value.code || fallback || "");
        }
        return String(value == null ? (fallback || "") : value);
    }

    function normalizeLanguageOptions(languages) {
        var out = {};
        var i;
        var item;
        var code;
        if (!languages || typeof languages !== "object") return out;
        if (Array.isArray(languages)) {
            for (i = 0; i < languages.length; i++) {
                item = languages[i];
                if (item && typeof item === "object") {
                    code = item.code || item.id || item.language;
                    if (code) out[String(code)] = languageOptionLabel(item, code);
                } else if (typeof item === "string" && item) {
                    out[item] = item;
                }
            }
            return out;
        }
        var keys = Object.keys(languages);
        for (i = 0; i < keys.length; i++) {
            code = keys[i];
            out[code] = languageOptionLabel(languages[code], code);
        }
        return out;
    }

    function getTranscriptTotalDuration(data) {
        var segments = data && data.segments ? data.segments : [];
        var maxEnd = 0;
        for (var i = 0; i < segments.length; i++) {
            maxEnd = Math.max(maxEnd, Number(segments[i].end || segments[i].start || 0));
        }
        return maxEnd;
    }

    function polishStepsFromResult(result) {
        var map = {};
        var steps = (result && result.steps) || [];
        for (var i = 0; i < steps.length; i++) {
            map[steps[i].key] = steps[i];
        }
        return map;
    }

    return {
        normalizeWorkspaceState: normalizeWorkspaceState,
        normalizeNavScrollState: normalizeNavScrollState,
        languageOptionLabel: languageOptionLabel,
        normalizeLanguageOptions: normalizeLanguageOptions,
        getTranscriptTotalDuration: getTranscriptTotalDuration,
        polishStepsFromResult: polishStepsFromResult
    };
});
