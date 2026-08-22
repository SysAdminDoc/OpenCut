/* OpenCut CEP XHR/CSRF transport boundary. Classic script + CommonJS for tests. */
(function (root, factory) {
    "use strict";
    var api = factory();
    if (typeof module === "object" && module.exports) module.exports = api;
    if (root) root.OpenCutBackendClient = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
    "use strict";

    function createBackendClient(options) {
        options = options || {};
        var getBaseUrl = options.getBaseUrl || function () { return ""; };
        var getToken = options.getToken || function () { return ""; };
        var setToken = options.setToken || function () {};
        var translate = options.translate || function (_key, fallback) { return fallback; };
        var createRequest = options.createRequest || function () { return new XMLHttpRequest(); };
        // F303: proves to /health that this is the host-embedded panel and not
        // a web page sharing its opaque `Origin: null`. Empty until the host
        // bridge has read the local secret; a `file://` panel has no
        // same-origin fallback, so without it every mutation 403s.
        var getBootstrapToken = options.getBootstrapToken || function () { return ""; };
        // F389: the secret is written by the backend at startup, so a panel
        // that loaded first reads nothing and would stay broken for the whole
        // session. Called once per refresh when /health refuses the bootstrap;
        // it re-reads the secret and reports whether a retry is worth making.
        var onBootstrapRefused = options.onBootstrapRefused || null;
        // F308: transport faults used to vanish here. Reporting is on by
        // default so a caller cannot silently opt out of diagnosability.
        var onTransportError = options.onTransportError || function (info) {
            if (typeof console !== "undefined" && console.error) {
                console.error(
                    "OpenCut transport error [" + info.stage + "]: " + info.detail,
                    info.context
                );
            }
        };
        var inflightRequests = {};

        function reportTransportError(stage, detail, context) {
            try {
                onTransportError({ stage: stage, detail: String(detail), context: context || {} });
            } catch (e) {
                // The reporter itself must never break the request path.
                if (typeof console !== "undefined" && console.error) {
                    console.error("OpenCut transport reporter failed:", e);
                }
            }
        }

        function safeParse(text, context) {
            try {
                return { ok: true, data: JSON.parse(text) };
            } catch (e) {
                reportTransportError("parse", e && e.message ? e.message : e, context);
                return { ok: false, data: null, error: e };
            }
        }

        function invoke(fn, err, data, context) {
            try {
                fn(err, data);
            } catch (e) {
                reportTransportError("callback", e && e.message ? e.message : e, context);
            }
        }

        function formatHttpStatusError(status) {
            return translate("error.http_status", "Request failed (HTTP {status}).").replace("{status}", status);
        }

        function refreshCsrfToken(callback, timeout, retriedBootstrap) {
            callback = typeof callback === "function" ? callback : function () {};
            var xhr = createRequest();
            xhr.open("GET", getBaseUrl() + "/health", true);
            xhr.timeout = timeout || 10000;
            var bootstrap = "";
            try { bootstrap = getBootstrapToken() || ""; } catch (e) { bootstrap = ""; }
            if (bootstrap) xhr.setRequestHeader("X-OpenCut-Panel-Bootstrap", bootstrap);
            xhr.onload = function () {
                var parsed = safeParse(xhr.responseText, { path: "/health", status: xhr.status });
                var data = parsed.data;
                if (xhr.status >= 200 && xhr.status < 300 && data && data.csrf_token) {
                    setToken(data.csrf_token);
                    callback(true);
                    return;
                }
                if (xhr.status >= 200 && xhr.status < 300 && data && !data.csrf_token) {
                    // Reached the backend but was refused the bootstrap token.
                    // Without this the panel reports "connected" and then fails
                    // every mutation with an unexplained CSRF error.
                    reportTransportError(
                        "csrf_bootstrap",
                        bootstrap
                            ? "Backend withheld the CSRF token despite a panel bootstrap secret."
                            : "Backend withheld the CSRF token and no panel bootstrap secret was available.",
                        { path: "/health", status: xhr.status }
                    );
                    // F389: the backend writes the secret at startup, so a
                    // panel that loaded first had nothing to read. Re-read once
                    // and retry — otherwise the session never recovers.
                    if (onBootstrapRefused && !retriedBootstrap) {
                        onBootstrapRefused(function (readAgain) {
                            if (!readAgain) { callback(false); return; }
                            refreshCsrfToken(callback, timeout, true);
                        });
                        return;
                    }
                }
                callback(false);
            };
            xhr.onerror = function () { callback(false); };
            xhr.ontimeout = function () { callback(false); };
            xhr.send(null);
        }

        function request(method, path, body, callback, timeout) {
            callback = typeof callback === "function" ? callback : function () {};
            var key = method + " " + path;
            if (method === "GET" && inflightRequests[key]) {
                var existing = inflightRequests[key];
                if (existing._pendingCallbacks) existing._pendingCallbacks.push(callback);
                return;
            }

            function send(retriedCsrf) {
                var xhr = createRequest();
                xhr.open(method, getBaseUrl() + path, true);
                xhr.timeout = timeout || 120000;
                xhr.setRequestHeader("Content-Type", "application/json");
                var token = getToken();
                if (token) xhr.setRequestHeader("X-OpenCut-Token", token);
                if (method === "GET") {
                    xhr._pendingCallbacks = [];
                    inflightRequests[key] = xhr;
                }

                function notifyPending(err, data) {
                    var callbacks = xhr._pendingCallbacks || [];
                    for (var i = 0; i < callbacks.length; i++) {
                        invoke(callbacks[i], err, data, { path: path, method: method, deduped: true });
                    }
                }

                xhr.onload = function () {
                    delete inflightRequests[key];
                    var err = null;
                    var data = null;
                    var parsed = safeParse(xhr.responseText, {
                        path: path,
                        method: method,
                        status: xhr.status
                    });
                    if (parsed.ok) {
                        data = parsed.data;
                    } else {
                        err = parsed.error;
                    }
                    if (!err && xhr.status === 403 && method !== "GET" && !retriedCsrf) {
                        var message = data && data.error ? String(data.error) : "";
                        if (/csrf|token/i.test(message)) {
                            refreshCsrfToken(function (ok) {
                                if (ok) {
                                    send(true);
                                    return;
                                }
                                err = new Error(message || formatHttpStatusError(403));
                                err.status = xhr.status;
                                callback(err, data);
                            }, timeout);
                            return;
                        }
                    }
                    if (!err && xhr.status >= 400) {
                        err = new Error((data && data.error) ? data.error : formatHttpStatusError(xhr.status));
                        err.status = xhr.status;
                    }
                    invoke(callback, err, data, { path: path, method: method, status: xhr.status });
                    notifyPending(err, data);
                };
                xhr.onerror = function () {
                    delete inflightRequests[key];
                    var err = new Error(translate("error.network", "Network error"));
                    reportTransportError("network", err.message, { path: path, method: method });
                    invoke(callback, err, null, { path: path, method: method });
                    notifyPending(err, null);
                };
                xhr.ontimeout = function () {
                    delete inflightRequests[key];
                    var err = new Error(translate("error.timeout", "Timeout"));
                    reportTransportError("timeout", err.message, { path: path, method: method });
                    invoke(callback, err, null, { path: path, method: method });
                    notifyPending(err, null);
                };
                xhr.send(body ? JSON.stringify(body) : null);
            }

            send(false);
        }

        return {
            request: request,
            refreshCsrfToken: refreshCsrfToken,
            formatHttpStatusError: formatHttpStatusError
        };
    }

    return { createBackendClient: createBackendClient };
});
