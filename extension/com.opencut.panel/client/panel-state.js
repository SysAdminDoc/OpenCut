/* OpenCut CEP panel state boundary. Classic script + CommonJS for tests. */
(function (root, factory) {
    "use strict";
    var api = factory();
    if (typeof module === "object" && module.exports) module.exports = api;
    if (root) root.OpenCutPanelState = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
    "use strict";

    function createPanelState(initial) {
        initial = initial || {};
        var state = {
            backendUrl: initial.backendUrl || "http://127.0.0.1:5679",
            csrfToken: initial.csrfToken || "",
            connected: !!initial.connected
        };
        var listeners = [];

        function snapshot() {
            return {
                backendUrl: state.backendUrl,
                csrfToken: state.csrfToken,
                connected: state.connected
            };
        }

        function notify() {
            var next = snapshot();
            for (var i = 0; i < listeners.length; i++) listeners[i](next);
        }

        function update(values) {
            values = values || {};
            var changed = false;
            for (var key in values) {
                if (Object.prototype.hasOwnProperty.call(values, key) && state[key] !== values[key]) {
                    state[key] = values[key];
                    changed = true;
                }
            }
            if (changed) notify();
            return snapshot();
        }

        function subscribe(listener) {
            if (typeof listener !== "function") return function () {};
            listeners.push(listener);
            return function () {
                var index = listeners.indexOf(listener);
                if (index >= 0) listeners.splice(index, 1);
            };
        }

        return {
            snapshot: snapshot,
            update: update,
            subscribe: subscribe,
            getBackendUrl: function () { return state.backendUrl; },
            getCsrfToken: function () { return state.csrfToken; },
            isConnected: function () { return state.connected; },
            setBackendUrl: function (value) { return update({ backendUrl: value }); },
            setCsrfToken: function (value) { return update({ csrfToken: value || "" }); },
            setConnected: function (value) { return update({ connected: !!value }); }
        };
    }

    return { createPanelState: createPanelState };
});
