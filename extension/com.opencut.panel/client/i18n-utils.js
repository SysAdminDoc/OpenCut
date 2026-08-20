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

    // Search status copy: pluralisation and template substitution over
    // translated strings, which is i18n formatting rather than panel logic.
    // Bind it once to the controller's `t` and the whole family comes with it.
    function createSearchCopy(t) {
        function plural(count) { return count === 1 ? "" : "s"; }

        function filesIndexed(count) {
            return t("search.files_indexed", "{count} file{plural} indexed")
                .replace("{count}", count)
                .replace("{plural}", plural(count));
        }

        function segments(count) {
            return t("search.segments_count", "{count} segment{plural}")
                .replace("{count}", count)
                .replace("{plural}", plural(count));
        }

        function projectClips(count) {
            return t("search.project_clip_count", "{count} project clip{plural}")
                .replace("{count}", count)
                .replace("{plural}", plural(count));
        }

        function indexCount(totalFiles, totalSegments) {
            var countLabel = filesIndexed(totalFiles);
            if (!totalSegments) return countLabel;
            return t("search.files_with_segments", "{files} • {segments}")
                .replace("{files}", countLabel)
                .replace("{segments}", segments(totalSegments));
        }

        function indexedAcross(totalFiles, totalSegments) {
            return t("search.indexed_across", "{files} indexed across {segments}.")
                .replace("{files}", filesIndexed(totalFiles))
                .replace("{segments}", segments(totalSegments));
        }

        function indexingProgress(indexed, total) {
            return t("search.indexing_progress", "Indexed {indexed} of {total}.")
                .replace("{indexed}", indexed)
                .replace("{total}", projectClips(total));
        }

        function indexingToast(indexed, total, errorCount) {
            var issues = errorCount
                ? t("search.indexing_toast_issues", " with {count} issue{plural}")
                    .replace("{count}", errorCount)
                    .replace("{plural}", plural(errorCount))
                : "";
            return t("search.indexing_complete_toast", "Indexed {indexed} of {total}{issues}.")
                .replace("{indexed}", indexed)
                .replace("{total}", projectClips(total))
                .replace("{issues}", issues);
        }

        return {
            filesIndexed: filesIndexed,
            segments: segments,
            projectClips: projectClips,
            indexCount: indexCount,
            indexedAcross: indexedAcross,
            indexingProgress: indexingProgress,
            indexingToast: indexingToast
        };
    }

    return {
        translate: translate,
        mergeLocale: mergeLocale,
        createSearchCopy: createSearchCopy
    };
});
