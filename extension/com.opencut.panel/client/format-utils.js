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

    // ---- Visual (CLIP) footage search -----------------------------------
    // Video containers /search/semantic can pull key frames from. Audio and
    // stills in the project bin would only fail the whole request.
    var VISUAL_SEARCH_VIDEO_EXT_RE = /\.(?:mp4|m4v|mov|mxf|avi|mkv|webm|mts|m2ts|mpg|mpeg|wmv|flv|3gp)$/i;
    // The backend caps inline thumbnails at 64 KB of JPEG; leave headroom for
    // base64 growth and refuse anything larger outright.
    var MAX_THUMBNAIL_DATA_URI_LENGTH = 96 * 1024;
    var JPEG_DATA_URI_RE = /^data:image\/jpeg;base64,[A-Za-z0-9+/]+={0,2}$/;

    function pad2(n) { return (n < 10 ? "0" : "") + n; }

    // Seconds -> "mm:ss", or "h:mm:ss" past the hour. Bad input reads as 0.
    function formatClipTimestamp(seconds) {
        var n = Number(seconds);
        var total = isFinite(n) && n > 0 ? Math.floor(n) : 0;
        var h = Math.floor(total / 3600);
        var m = Math.floor((total % 3600) / 60);
        var s = total % 60;
        return h > 0 ? h + ":" + pad2(m) + ":" + pad2(s) : pad2(m) + ":" + pad2(s);
    }

    // CLIP cosine similarity (0-1) -> whole percent, or null when missing.
    function scoreToPercent(score) {
        if (typeof score !== "number" || !isFinite(score)) return null;
        return Math.round(Math.min(1, Math.max(0, score)) * 100);
    }

    // Only inline base64 JPEGs of bounded size are safe to drop into <img src>.
    function safeThumbnailDataUri(value) {
        if (typeof value !== "string" || value.length > MAX_THUMBNAIL_DATA_URI_LENGTH) return "";
        return JPEG_DATA_URI_RE.test(value) ? value : "";
    }

    // Project media ({path} objects or bare strings) -> unique video paths.
    function collectVisualSearchClipPaths(items) {
        var out = [];
        var seen = {};
        if (!items || typeof items.length !== "number") return out;
        for (var i = 0; i < items.length; i++) {
            var item = items[i];
            var raw = typeof item === "string" ? item : (item && typeof item.path === "string" ? item.path : "");
            var path = raw.replace(/^\s+|\s+$/g, "");
            if (!path || !VISUAL_SEARCH_VIDEO_EXT_RE.test(path)) continue;
            if (Object.prototype.hasOwnProperty.call(seen, path)) continue;
            seen[path] = true;
            out.push(path);
        }
        return out;
    }

    // project_dir for the portable sidecar: only a saved project's own folder.
    // The host's first-media and scratch-disk fallbacks aren't the project, and
    // the backend refuses UNC paths, which would fail the whole search.
    function visualSearchProjectDir(folder, source) {
        if (source !== "project_path" || typeof folder !== "string") return "";
        var dir = folder.replace(/^\s+|\s+$/g, "");
        if (!dir || /^[\\/]{2}/.test(dir)) return "";
        return dir;
    }

    // One /search/semantic hit -> the fields a result row shows.
    function buildVisualSearchCard(item, index) {
        var hit = item && typeof item === "object" ? item : {};
        var path = typeof hit.clip_path === "string" ? hit.clip_path : "";
        var clipName = typeof hit.clip_name === "string" ? hit.clip_name.replace(/^\s+|\s+$/g, "") : "";
        return {
            index: index,
            path: path,
            name: clipName || path.split(/[/\\]/).pop() || "",
            timestamp: formatClipTimestamp(hit.timestamp),
            percent: scoreToPercent(hit.score),
            thumbnail: safeThumbnailDataUri(hit.thumbnail)
        };
    }

    return {
        safeFixed: safeFixed,
        escSingleQuote: escSingleQuote,
        extractWordSegments: extractWordSegments,
        fmtDur: fmtDur,
        formatTimecode: formatTimecode,
        getStepPrecision: getStepPrecision,
        formatNumberForInput: formatNumberForInput,
        formatClipTimestamp: formatClipTimestamp,
        scoreToPercent: scoreToPercent,
        safeThumbnailDataUri: safeThumbnailDataUri,
        collectVisualSearchClipPaths: collectVisualSearchClipPaths,
        visualSearchProjectDir: visualSearchProjectDir,
        buildVisualSearchCard: buildVisualSearchCard
    };
});
