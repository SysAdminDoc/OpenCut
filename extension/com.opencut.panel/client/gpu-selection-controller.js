/* OpenCut GPU adapter selection controller. Classic script + CommonJS. */
(function (root, factory) {
    "use strict";
    var api = factory();
    if (typeof module === "object" && module.exports) module.exports = api;
    if (root) root.OpenCutGpuSelectionController = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
    "use strict";

    function createGpuSelectionController(options) {
        options = options || {};
        var request = typeof options.request === "function"
            ? options.request
            : function (method, path, body, callback) { callback(new Error("GPU request is unavailable.")); };
        var select = options.selectElement || null;
        var status = options.statusElement || null;
        var documentRef = options.documentRef || (typeof document !== "undefined" ? document : null);
        var translate = typeof options.translate === "function" ? options.translate : function (_key, fallback) { return fallback; };
        var notify = typeof options.notify === "function" ? options.notify : function () {};
        var state = null;
        var disposed = false;

        function t(key, fallback) {
            return translate(key, fallback);
        }

        function setStatus(message, stateName) {
            if (!status) return;
            status.textContent = message;
            status.setAttribute("data-state", stateName || "idle");
        }

        function render(data) {
            if (disposed) return;
            state = data && typeof data === "object" ? data : null;
            if (select) {
                while (select.firstChild) select.removeChild(select.firstChild);
                var auto = documentRef.createElement("option");
                auto.value = "auto";
                auto.textContent = t("settings.gpu_adapter_auto", "Auto (recommended)");
                select.appendChild(auto);
                var devices = Array.isArray(state && state.devices) ? state.devices : [];
                for (var i = 0; i < devices.length; i++) {
                    var device = devices[i] || {};
                    var option = documentRef.createElement("option");
                    option.value = String(device.index);
                    option.textContent = String(device.index) + " — " + String(device.name || t("settings.gpu_adapter_device", "CUDA device"));
                    select.appendChild(option);
                }
                var configured = state && state.configured_index != null
                    ? String(state.configured_index)
                    : "auto";
                select.value = configured;
                if (select.value !== configured) select.value = "auto";
                select.disabled = devices.length === 0;
            }
            if (state && state.selection_error) {
                setStatus(
                    t("settings.gpu_adapter_invalid", "The configured GPU is unavailable. Choose another adapter."),
                    "error"
                );
            } else if (state && Array.isArray(state.devices) && state.devices.length) {
                var selected = state.selected_index != null ? String(state.selected_index) : "auto";
                setStatus(
                    t("settings.gpu_adapter_status", "Using GPU adapter {index}.").replace("{index}", selected),
                    "success"
                );
            } else {
                setStatus(t("settings.gpu_adapter_none", "No CUDA adapters detected; GPU work will use CPU."), "warning");
            }
        }

        function selectGpu(value) {
            if (disposed) return;
            var requested = value === "auto" ? null : Number(value);
            setStatus(t("settings.gpu_adapter_saving", "Saving GPU adapter selection…"), "working");
            request("POST", "/system/gpu", { gpu_index: requested }, function (err, data) {
                if (disposed) return;
                if (err || !data || data.success === false) {
                    render(state);
                    notify(err || data, "error");
                    return;
                }
                render(data);
                notify(t("settings.gpu_adapter_saved", "GPU adapter selection saved."), "success");
            });
        }

        function bind() {
            if (!select) return;
            select.addEventListener("change", function () { selectGpu(select.value); });
        }

        function dispose() {
            disposed = true;
        }

        return { bind: bind, render: render, dispose: dispose };
    }

    return { createGpuSelectionController: createGpuSelectionController };
});
