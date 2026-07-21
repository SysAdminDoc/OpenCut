/* ============================================================
   OpenCut CEP Panel - Pure job / session metadata accessors
   Extracted from main.js so they can be unit-tested in isolation.
   Loaded as a classic script (window.OpenCutJobMeta) and as a
   CommonJS module (vitest). No DOM, i18n, or shared-state access.
   ============================================================ */
(function (root, factory) {
    "use strict";
    var api = factory();
    if (typeof module === "object" && module.exports) module.exports = api;
    if (root) root.OpenCutJobMeta = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
    "use strict";

    // NOTE: the local `var t` intentionally shadows the panel's i18n accessor —
    // this is a plain string transform, not a translation lookup.
    function sessionCtxOpText(job) {
        var t = (job.type || "unknown").replace(/[_-]/g, " ");
        return t.charAt(0).toUpperCase() + t.slice(1);
    }

    function sessionCtxResultPath(job) {
        var r = job.result;
        if (!r || typeof r !== "object") return "";
        return r.output_path || r.xml_path || r.overlay_path || r.srt_path ||
               (Array.isArray(r.output_paths) && r.output_paths[0]) || "";
    }

    function getJobHistorySourcePath(entry) {
        if (!entry) return "";
        return entry.sourcePath || (entry.payload && entry.payload.filepath) || entry.filepath || "";
    }

    function getJobHistorySourceName(entry) {
        var path = getJobHistorySourcePath(entry);
        if (!path) return "";
        var parts = path.split(/[\\/]/);
        return parts[parts.length - 1] || path;
    }

    return {
        sessionCtxOpText: sessionCtxOpText,
        sessionCtxResultPath: sessionCtxResultPath,
        getJobHistorySourcePath: getJobHistorySourcePath,
        getJobHistorySourceName: getJobHistorySourceName
    };
});
