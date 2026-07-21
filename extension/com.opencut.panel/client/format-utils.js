/* ============================================================
   OpenCut CEP Panel - Pure formatting / escaping helpers
   Extracted from main.js so they can be unit-tested in isolation
   and reused without dragging in the panel controller closure.
   Loaded as a classic script (window.OpenCutFormat) and as a
   CommonJS module (vitest). No DOM, i18n, or shared-state access.
   ============================================================ */
(function (root, factory) {
    "use strict";
    var api = factory();
    if (typeof module === "object" && module.exports) module.exports = api;
    if (root) root.OpenCutFormat = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
    "use strict";

    function safeFixed(v, digits) { var n = Number(v); return isFinite(n) ? n.toFixed(digits) : "0"; }

    // Escape for embedding inside an ExtendScript *single-quoted* string.
    // Used by evalScript("ocX('" + escSingleQuote(payload) + "')") call sites
    // that previously did a narrow replace(/\\/g,"\\\\").replace(/'/g,"\\'")
    // which failed on literal newlines/CR/Unicode line separators — breaking
    // the JSX string and raising a cs.evalScript error on markers/chapters
    // whose names contained those chars.
    function escSingleQuote(s) {
        if (s === undefined || s === null) return "";
        return String(s)
            .replace(/\\/g, "\\\\")
            .replace(/'/g, "\\'")
            .replace(/\n/g, "\\n")
            .replace(/\r/g, "\\r")
            .replace(/\t/g, "\\t")
            .replace(/\u2028/g, "\\u2028")
            .replace(/\u2029/g, "\\u2029");
    }

    function extractWordSegments(segments) {
        var words = [];
        for (var i = 0; i < segments.length; i++) {
            if (segments[i].words) {
                for (var j = 0; j < segments[i].words.length; j++) {
                    words.push(segments[i].words[j]);
                }
            }
        }
        return words;
    }

    function fmtDur(s) {
        if (!s && s !== 0) return "--";
        var m = Math.floor(s / 60);
        var sec = Math.floor(s % 60);
        return m + ":" + (sec < 10 ? "0" : "") + sec;
    }

    // Format seconds as MM:SS.s (e.g. "01:23.4")
    function formatTimecode(seconds) {
        if (!seconds && seconds !== 0) return "00:00.0";
        var totalTenths = Math.round(Math.abs(Number(seconds) || 0) * 10);
        var totalSec = Math.floor(totalTenths / 10);
        var tenths = totalTenths % 10;
        var m = Math.floor(totalSec / 60);
        var sec = totalSec % 60;
        return (m < 10 ? "0" : "") + m + ":" + (sec < 10 ? "0" : "") + sec + "." + tenths;
    }

    function getStepPrecision(step) {
        var stepText = String(step || "");
        var decimalPoint = stepText.indexOf(".");
        return decimalPoint === -1 ? 0 : (stepText.length - decimalPoint - 1);
    }

    function formatNumberForInput(value, precision) {
        var text;
        if (!isFinite(value)) return "";
        if (precision > 0) {
            text = value.toFixed(precision).replace(/\.?0+$/, "");
            return text === "-0" ? "0" : text;
        }
        text = String(Math.round(value));
        return text === "-0" ? "0" : text;
    }

    return {
        safeFixed: safeFixed,
        escSingleQuote: escSingleQuote,
        extractWordSegments: extractWordSegments,
        fmtDur: fmtDur,
        formatTimecode: formatTimecode,
        getStepPrecision: getStepPrecision,
        formatNumberForInput: formatNumberForInput
    };
});
