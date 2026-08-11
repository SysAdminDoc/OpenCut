/* OpenCut panel bootstrap-secret reader. Classic script + CommonJS for tests.
 *
 * F303: the panel document is a file:// origin, so /health refuses it the CSRF
 * bootstrap token — a hostile local page presents the same opaque origin. Only
 * ExtendScript can read the backend's 0600 secret file, so the host bridge is
 * how the real panel proves it is not a web page.
 */
(function (root, factory) {
    "use strict";
    var api = factory();
    if (typeof module === "object" && module.exports) module.exports = api;
    if (root) root.OpenCutPanelBootstrapToken = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
    "use strict";

    function createBootstrapTokenLoader(options) {
        options = options || {};
        var evalScript = options.evalScript || null;
        var onWarn = options.onWarn || function () {};
        var onReady = options.onReady || function () {};
        var token = "";

        function get() {
            return token;
        }

        function load(callback) {
            callback = typeof callback === "function" ? callback : function () {};
            if (typeof evalScript !== "function") {
                onWarn("Host bridge unavailable; cannot read the panel bootstrap secret.");
                callback(false);
                return;
            }
            try {
                evalScript("ocReadPanelBootstrapToken()", function (raw) {
                    var parsed = null;
                    try { parsed = JSON.parse(raw); } catch (e) { parsed = null; }
                    if (parsed && parsed.ok && parsed.token) {
                        token = String(parsed.token);
                        onReady(token);
                        callback(true);
                        return;
                    }
                    onWarn(
                        parsed && parsed.reason
                            ? String(parsed.reason)
                            : "Host bridge returned no panel bootstrap secret."
                    );
                    callback(false);
                });
            } catch (e) {
                onWarn("Panel bootstrap read failed: " + (e && e.message ? e.message : e));
                callback(false);
            }
        }

        return { get: get, load: load };
    }

    return { createBootstrapTokenLoader: createBootstrapTokenLoader };
});
