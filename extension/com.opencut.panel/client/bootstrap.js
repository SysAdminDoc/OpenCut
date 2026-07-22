/* OpenCut CEP DOM-ready/bootstrap boundary. Classic script + CommonJS for tests. */
(function (root, factory) {
    "use strict";
    var api = factory();
    if (typeof module === "object" && module.exports) module.exports = api;
    if (root) root.OpenCutBootstrap = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
    "use strict";

    function runSteps(steps, onError) {
        steps = Array.isArray(steps) ? steps : [];
        onError = typeof onError === "function" ? onError : function () {};
        for (var i = 0; i < steps.length; i++) {
            try {
                steps[i]();
            } catch (error) {
                onError(error, i);
            }
        }
    }

    function onReady(doc, initialize, onError) {
        if (!doc || typeof doc.addEventListener !== "function") {
            throw new TypeError("onReady requires a document");
        }
        if (typeof initialize !== "function") {
            throw new TypeError("onReady requires an initializer");
        }
        function start() {
            try { initialize(); }
            catch (error) {
                if (typeof onError === "function") onError(error);
                else throw error;
            }
        }
        if (doc.readyState === "loading") doc.addEventListener("DOMContentLoaded", start);
        else start();
    }

    return { runSteps: runSteps, onReady: onReady };
});
