/* ============================================================
   OpenCut CEP Panel - Pure string / path / template helpers
   Extracted from main.js so they can be unit-tested in isolation.
   Loaded as a classic script (window.OpenCutStringUtils) and as a
   CommonJS module (vitest). No DOM, i18n, or shared-state access.
   ============================================================ */
(function (root, factory) {
    "use strict";
    var api = factory();
    if (typeof module === "object" && module.exports) module.exports = api;
    if (root) root.OpenCutStringUtils = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
    "use strict";

    function humanizeControlId(id) {
        return String(id || "")
            .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
            .replace(/[_-]+/g, " ")
            .replace(/\b\w/g, function (letter) { return letter.toUpperCase(); })
            .trim();
    }

    function journalClipName(path) {
        return path ? String(path).split(/[/\\]/).pop() : "";
    }

    function shortsBundleFileName(path) {
        return String(path || "").split(/[\\/]/).pop() || "output";
    }

    function parseTimeToSec(t) {
        var parts = (t || "0").split(":");
        var result;
        if (parts.length === 3) result = (+parts[0]) * 3600 + (+parts[1]) * 60 + (+parts[2]);
        else if (parts.length === 2) result = (+parts[0]) * 60 + (+parts[1]);
        else result = +parts[0];
        return isNaN(result) ? 0 : result;
    }

    function captionDisplayOptionLabel(spec, opt) {
        var id = String(opt.id || "");
        if (spec.category === "font" && opt.font_family) {
            var source = opt.font_resolution && opt.font_resolution.source;
            var status = source && source !== "preferred_file" ? "fallback" : "resolved";
            return id + " (" + opt.font_family + ", " + status + ")";
        }
        if (spec.category === "size" && opt.font_size) return id + " (" + opt.font_size + ")";
        if (spec.category === "color" && opt.hex) return id + " (" + opt.hex + ")";
        if (spec.category === "opacity" && opt.alpha !== undefined) return id + " (" + opt.alpha + ")";
        return id;
    }

    function inferNotificationTone(message, errorData, explicitType) {
        if (explicitType && /^(success|error|warning|info)$/.test(explicitType)) {
            return explicitType;
        }
        var lower = String(message || "").toLowerCase();
        if (errorData && (errorData.error || errorData.message || errorData.code)) {
            return "error";
        }
        if (/(^error\b|failed|failure|couldn't|could not|invalid|unable|not configured|unexpected|import error|oauth error)/.test(lower)) {
            return "error";
        }
        if (/(success|saved|loaded|opened|exported|installed successfully|complete|completed|deleted|enabled|disabled|cleared|copied|refreshed|queue cleared|succeeded)/.test(lower)) {
            return "success";
        }
        if (/(select|choose|enter|make sure|no clip|no clips|no cuts|no markers|no items|no project media|required|another task is in progress|connection required)/.test(lower)) {
            return "warning";
        }
        return "info";
    }

    function getNotificationIconSvg(tone) {
        switch (tone) {
            case "success":
                return '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.85" stroke-linecap="round" stroke-linejoin="round"><circle cx="10" cy="10" r="7.25"/><path d="M6.8 10.2l2.1 2.15 4.3-4.45"/></svg>';
            case "warning":
                return '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M10 3.4 16.2 14a1.15 1.15 0 0 1-.99 1.72H4.79A1.15 1.15 0 0 1 3.8 14L10 3.4Z"/><path d="M10 7.35v3.9"/><circle cx="10" cy="13.45" r="0.9" fill="currentColor" stroke="none"/></svg>';
            case "error":
                return '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="10" cy="10" r="7.25"/><path d="m7.35 7.35 5.3 5.3"/><path d="m12.65 7.35-5.3 5.3"/></svg>';
            default:
                return '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="10" cy="10" r="7.25"/><path d="M10 6.4v4.45"/><circle cx="10" cy="13.85" r="0.9" fill="currentColor" stroke="none"/></svg>';
        }
    }

    function wsFormatListenerCount(count, template) {
        return template
            .replace("{count}", count)
            .replace("{plural}", count === 1 ? "" : "s");
    }

    return {
        humanizeControlId: humanizeControlId,
        journalClipName: journalClipName,
        shortsBundleFileName: shortsBundleFileName,
        parseTimeToSec: parseTimeToSec,
        captionDisplayOptionLabel: captionDisplayOptionLabel,
        inferNotificationTone: inferNotificationTone,
        getNotificationIconSvg: getNotificationIconSvg,
        wsFormatListenerCount: wsFormatListenerCount
    };
});
