/* OpenCut CEP disabled-control reasons. Classic script + CommonJS for tests.
 *
 * A greyed-out button with no tooltip tells a user nothing. Controls opt in
 * with `data-disabled-title` (translated through `data-i18n-disabled-title`),
 * and this module copies it into `title` for as long as the control is
 * disabled, putting back whatever title it had once it becomes usable.
 */
(function (root, factory) {
    "use strict";
    var api = factory();
    if (typeof module === "object" && module.exports) module.exports = api;
    if (root) root.OpenCutDisabledReasons = api;
    if (typeof document !== "undefined" && typeof MutationObserver !== "undefined") {
        // Self-starting: the panel entrypoint is on a hard line budget, and
        // there is nothing to configure here.
        if (document.readyState === "loading") {
            document.addEventListener("DOMContentLoaded", function () {
                api.observeDisabledReasons(document);
            });
        } else {
            api.observeDisabledReasons(document);
        }
    }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
    "use strict";

    var REASON_ATTR = "data-disabled-title";
    var SAVED_ATTR = "data-enabled-title";

    function syncControl(el) {
        if (!el || typeof el.getAttribute !== "function") return false;
        var reason = el.getAttribute(REASON_ATTR);
        if (!reason) return false;
        if (el.disabled) {
            if (!el.hasAttribute(SAVED_ATTR)) {
                el.setAttribute(SAVED_ATTR, el.getAttribute("title") || "");
            }
            el.setAttribute("title", reason);
            return true;
        }
        var restored = el.hasAttribute(SAVED_ATTR) ? el.getAttribute(SAVED_ATTR) : null;
        if (restored) el.setAttribute("title", restored);
        else el.removeAttribute("title");
        el.removeAttribute(SAVED_ATTR);
        return true;
    }

    function syncDisabledReasons(scope) {
        var root = scope || (typeof document !== "undefined" ? document : null);
        if (!root || typeof root.querySelectorAll !== "function") return 0;
        var nodes = root.querySelectorAll("[" + REASON_ATTR + "]");
        var synced = 0;
        for (var i = 0; i < nodes.length; i++) {
            if (syncControl(nodes[i])) synced++;
        }
        return synced;
    }

    function observeDisabledReasons(scope, ObserverCtor) {
        var root = scope || (typeof document !== "undefined" ? document : null);
        var Ctor = ObserverCtor ||
            (typeof MutationObserver !== "undefined" ? MutationObserver : null);
        if (!root || !Ctor) return null;
        syncDisabledReasons(root);
        var observer = new Ctor(function (records) {
            for (var i = 0; i < records.length; i++) syncControl(records[i].target);
        });
        // The reason text arrives translated after the locale loads, so watch
        // it as well as the disabled flag it explains.
        observer.observe(root, {
            subtree: true,
            attributes: true,
            attributeFilter: ["disabled", REASON_ATTR]
        });
        return observer;
    }

    return {
        REASON_ATTR: REASON_ATTR,
        SAVED_ATTR: SAVED_ATTR,
        syncControl: syncControl,
        syncDisabledReasons: syncDisabledReasons,
        observeDisabledReasons: observeDisabledReasons
    };
});
