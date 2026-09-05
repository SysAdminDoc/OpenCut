/* OpenCut live-updates bridge port. Classic script + CommonJS for tests.
 *
 * F415: the panel dialled a hardcoded 5680 while /ws/start walks 5680-5689 for
 * a free port and skips the HTTP port, so whenever the first choice was taken
 * the panel opened a socket to nothing. Recording the port the backend reports
 * is the fix, and it lives here rather than inside main.js's IIFE so the
 * behaviour can be tested instead of grepped for.
 */
(function (root, factory) {
    "use strict";
    var api = factory();
    if (typeof module === "object" && module.exports) module.exports = api;
    if (root) root.OpenCutWsBridgePort = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
    "use strict";

    var DEFAULT_PORT = 5680;
    var MIN_PORT = 1024;
    var MAX_PORT = 65535;

    /* Return a usable port from a /ws/status or /ws/start payload, or null.
     *
     * Null rather than the default on purpose: the caller has to be able to
     * tell "the backend has not answered yet" from "the backend said 5680",
     * because only the first case is worth waiting for. */
    function portFromPayload(payload) {
        if (!payload) return null;
        var value = payload.port;
        if (typeof value === "string" && value.trim() !== "") value = Number(value);
        if (typeof value !== "number") return null;
        if (!isFinite(value) || Math.floor(value) !== value) return null;
        if (value < MIN_PORT || value > MAX_PORT) return null;
        return value;
    }

    function createBridgePortTracker(options) {
        options = options || {};
        var fallback = portFromPayload({ port: options.defaultPort }) || DEFAULT_PORT;
        var known = null;

        return {
            /* Record a port the backend reported. Ignores anything unusable so
             * a malformed payload cannot replace a good value with junk. */
            remember: function (payload) {
                var port = portFromPayload(payload);
                if (port !== null) known = port;
                return known;
            },
            /* The port to dial, or null while nothing has answered yet. */
            current: function () {
                return known;
            },
            /* The port to dial once we have given up waiting. */
            effective: function () {
                return known === null ? fallback : known;
            },
            /* True while the caller should still ask the backend first. */
            needsResolve: function () {
                return known === null;
            },
            forget: function () {
                known = null;
            },
        };
    }

    return {
        DEFAULT_PORT: DEFAULT_PORT,
        portFromPayload: portFromPayload,
        createBridgePortTracker: createBridgePortTracker,
    };
});
