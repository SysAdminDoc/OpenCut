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
        var inflightRequests = {};

        function formatHttpStatusError(status) {
            return translate("error.http_status", "HTTP {status}").replace("{status}", status);
        }

        function refreshCsrfToken(callback, timeout) {
            callback = typeof callback === "function" ? callback : function () {};
            var xhr = createRequest();
            xhr.open("GET", getBaseUrl() + "/health", true);
            xhr.timeout = timeout || 10000;
            xhr.onload = function () {
                var data = null;
                try { data = JSON.parse(xhr.responseText); } catch (e) {}
                if (xhr.status >= 200 && xhr.status < 300 && data && data.csrf_token) {
                    setToken(data.csrf_token);
                    callback(true);
                    return;
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
                        try { callbacks[i](err, data); } catch (e) {}
                    }
                }

                xhr.onload = function () {
                    delete inflightRequests[key];
                    var err = null;
                    var data = null;
                    try { data = JSON.parse(xhr.responseText); } catch (e) { err = e; }
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
                    callback(err, data);
                    notifyPending(err, data);
                };
                xhr.onerror = function () {
                    delete inflightRequests[key];
                    var err = new Error(translate("error.network", "Network error"));
                    callback(err, null);
                    notifyPending(err, null);
                };
                xhr.ontimeout = function () {
                    delete inflightRequests[key];
                    var err = new Error(translate("error.timeout", "Timeout"));
                    callback(err, null);
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
