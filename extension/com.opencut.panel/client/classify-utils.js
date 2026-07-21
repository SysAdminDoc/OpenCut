/* ============================================================
   OpenCut CEP Panel - Pure engine/plugin classifier helpers
   Extracted from main.js so they can be unit-tested in isolation.
   Loaded as a classic script (window.OpenCutClassify) and as a
   CommonJS module (vitest). No DOM, i18n, or shared-state access.
   ============================================================ */
(function (root, factory) {
    "use strict";
    var api = factory();
    if (typeof module === "object" && module.exports) module.exports = api;
    if (root) root.OpenCutClassify = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
    "use strict";

    function humanizeEngineDomain(domain) {
        return String(domain || "")
            .split("_")
            .filter(Boolean)
            .map(function (part) { return part.charAt(0).toUpperCase() + part.slice(1); })
            .join(" ");
    }

    function engineTemplateText(template, values) {
        var text = template;
        for (var key in values) {
            if (Object.prototype.hasOwnProperty.call(values, key)) {
                text = text.replace("{" + key + "}", values[key]);
            }
        }
        return text;
    }

    function pluginTrustStateClass(plugin) {
        var trust = plugin && plugin.trust ? plugin.trust : {};
        if ((plugin && plugin.load_status === "failed") || trust.errors && trust.errors.length) return "warning";
        if (trust.unsigned_allowed || trust.lock_missing) return "manual";
        if (plugin && plugin.load_status === "loaded") return "auto";
        return "manual";
    }

    return {
        humanizeEngineDomain: humanizeEngineDomain,
        engineTemplateText: engineTemplateText,
        pluginTrustStateClass: pluginTrustStateClass
    };
});
