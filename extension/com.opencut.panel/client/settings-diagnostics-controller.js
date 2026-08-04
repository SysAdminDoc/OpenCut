/* OpenCut CEP settings/diagnostics lifecycle controller. Classic script + CommonJS. */
(function (root, factory) {
    "use strict";
    var api = factory();
    if (typeof module === "object" && module.exports) module.exports = api;
    if (root) root.OpenCutSettingsDiagnosticsController = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
    "use strict";

    function createSettingsDiagnosticsController(options) {
        options = options || {};
        var request = typeof options.request === "function"
            ? options.request
            : function (method, path, body, callback) { callback(new Error("Settings request is unavailable.")); };
        var isConnected = typeof options.isConnected === "function"
            ? options.isConnected
            : function () { return false; };
        var renderOverview = typeof options.renderOverview === "function" ? options.renderOverview : function () {};
        var syncBackendSummary = typeof options.syncBackendSummary === "function" ? options.syncBackendSummary : function () {};
        var updateWhisperState = typeof options.updateWhisperState === "function" ? options.updateWhisperState : function () {};
        var renderGpuState = typeof options.renderGpuState === "function" ? options.renderGpuState : function () {};
        var onHealthUnavailable = typeof options.onHealthUnavailable === "function" ? options.onHealthUnavailable : function () {};
        var setCpuMode = typeof options.setCpuMode === "function" ? options.setCpuMode : function () {};
        var onFirstLoad = typeof options.onFirstLoad === "function" ? options.onFirstLoad : function () {};
        var loadLlmSettings = typeof options.loadLlmSettings === "function" ? options.loadLlmSettings : function () {};
        var updateBridgeStatus = typeof options.updateBridgeStatus === "function" ? options.updateBridgeStatus : function () {};
        var refreshDependencies = typeof options.refreshDependencies === "function" ? options.refreshDependencies : function () {};
        var refreshModels = typeof options.refreshModels === "function" ? options.refreshModels : function () {};
        var loadEngineRegistry = typeof options.loadEngineRegistry === "function" ? options.loadEngineRegistry : function () {};
        var loadPluginTrust = typeof options.loadPluginTrust === "function" ? options.loadPluginTrust : function () {};
        var loaded = false;
        var disposed = false;

        function load() {
            if (disposed) return false;
            var firstLoad = !loaded;
            loaded = true;
            renderOverview();
            if (!isConnected()) syncBackendSummary(false);

            request("GET", "/health", null, function (err, data) {
                if (disposed) return;
                if (err || !data || data.status !== "ok") {
                    syncBackendSummary(false);
                    updateWhisperState(null);
                    onHealthUnavailable();
                    return;
                }
                syncBackendSummary(true);
                setCpuMode(!!(data.capabilities && data.capabilities.whisper_cpu_mode));
                updateWhisperState(data);
                if (firstLoad) onFirstLoad();
            });

            request("GET", "/system/gpu", null, function (err, data) {
                if (disposed) return;
                renderGpuState(!err && data ? data : null);
            });

            loadLlmSettings();
            updateBridgeStatus();
            refreshDependencies();
            refreshModels();
            loadEngineRegistry();
            loadPluginTrust();
            return true;
        }

        function dispose() {
            disposed = true;
        }

        function reset() {
            if (!disposed) loaded = false;
        }

        return {
            dispose: dispose,
            load: load,
            reset: reset,
            getState: function () { return { loaded: loaded, disposed: disposed }; }
        };
    }

    return { createSettingsDiagnosticsController: createSettingsDiagnosticsController };
});
