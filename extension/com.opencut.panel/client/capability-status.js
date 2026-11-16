/* OpenCut capability status. Classic script + CommonJS for tests.
 *
 * Reads what /health reports about optional features: the probe the backend
 * runs in a child process (capability_discovery) and add-on model weights.
 * Some add-ons (depth, emotion, CrisperWhisper, multimodal diarization) also
 * load a model, and the server reports each one's weights under
 * capabilities.model_weights. When
 * the packages are installed and the model isn't, the hint says so and its
 * button downloads only the model. This lives here rather than inside
 * main.js's IIFE so the behaviour can be tested instead of grepped for.
 */
(function (root, factory) {
    "use strict";
    var api = factory();
    if (typeof module === "object" && module.exports) module.exports = api;
    if (root) root.OpenCutCapabilityStatus = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
    "use strict";

    /* Replace every {name} in `template` with values[name]. Unknown tokens
     * stay as they are, and a value is never read as a replace pattern. */
    function fillTemplate(template, values) {
        return String(template).replace(/\{(\w+)\}/g, function (match, key) {
            return Object.prototype.hasOwnProperty.call(values, key) ? String(values[key]) : match;
        });
    }

    function createCapabilityStatus(options) {
        options = options || {};
        var getCapabilities = options.getCapabilities || function () { return {}; };
        var t = options.t || function (key, fallback) { return fallback; };
        var getHintCopyNode = options.getHintCopyNode || function (hintEl) {
            return hintEl ? hintEl.querySelector(".hint-copy") : null;
        };
        var hintStateClasses = options.hintStateClasses || [];

        /* Read through the getter every time: main.js replaces its
         * capabilities object on each poll. */
        function modelWeights() {
            var caps = getCapabilities();
            return caps && caps.model_weights ? caps.model_weights : null;
        }

        function weightsEntry(component) {
            var mw = modelWeights();
            return mw && mw.components ? mw.components[component] || null : null;
        }

        /* The weights entry for `component` when its packages are installed
         * and its model isn't on disk, else null. */
        function weightsMissing(component) {
            var entry = weightsEntry(component);
            return entry && entry.packages && entry.state === "missing" ? entry : null;
        }

        function weightsDownloadBlocked() {
            var mw = modelWeights();
            return !!(mw && mw.local_only);
        }

        function weightsHintMessage(entry) {
            var size = Number(entry.download_mb) || 0;
            var message = fillTemplate(
                t("install.weights_missing", "{label} is installed, but its model ({size} MB) isn't downloaded yet."),
                { label: entry.label || "", size: size.toLocaleString("en-US") }
            );
            if (weightsDownloadBlocked()) {
                message += " " + t(
                    "install.weights_local_only",
                    "Local-only mode blocks the download. Turn it off in Settings to get the model, then turn it back on."
                );
            }
            return message;
        }

        function hintHasState(hintEl) {
            for (var i = 0; i < hintStateClasses.length; i++) {
                if (hintEl.classList.contains(hintStateClasses[i])) return true;
            }
            return false;
        }

        /* Puts a hint in the weights-missing state for `entry`, or back to its
         * markup copy when `entry` is null. A hint showing install progress or
         * an install error keeps that text. */
        function syncWeightsHint(hintEl, actionBtn, entry, labelKey, labelFallback) {
            if (!hintEl || hintHasState(hintEl)) return;
            var copy = getHintCopyNode(hintEl);
            if (copy) {
                if (!copy.hasAttribute("data-default-copy")) copy.setAttribute("data-default-copy", copy.textContent);
                var copyKey = copy.getAttribute("data-i18n");
                var defaultCopy = copy.getAttribute("data-default-copy");
                copy.textContent = entry ? weightsHintMessage(entry) : (copyKey ? t(copyKey, defaultCopy) : defaultCopy);
            }
            var alt = hintEl.querySelector(".install-alt");
            if (alt) alt.classList.toggle("hidden", !!entry);
            if (actionBtn) {
                actionBtn.textContent = entry ? t("install.download_model", "Download model") : t(labelKey, labelFallback);
                var blocked = !!entry && weightsDownloadBlocked();
                actionBtn.classList.toggle("hidden", blocked);
                actionBtn.disabled = blocked;
            }
        }

        /* An install of an add-on whose packages are already there fetches
         * only its model, and success marks the model ready in the cached
         * payload so the hint clears before the next poll. */
        function weightsInstallConfig(component, config) {
            var entry = weightsMissing(component);
            if (entry) {
                config.payload = { weights_only: true };
                config.startMessage = fillTemplate(
                    t("install.weights_start", "Downloading the {label} model… This may take a while."),
                    { label: entry.label || "" }
                );
            }
            var onSuccess = config.onSuccess;
            config.onSuccess = function (result, job) {
                var current = weightsEntry(component);
                if (current) {
                    current.packages = true;
                    current.state = "ready";
                }
                if (typeof onSuccess === "function") onSuccess(result, job);
            };
            return config;
        }

        /* Optional features are checked in a separate backend process. When
         * one of those checks crashes or hangs, the backend keeps running and
         * turns just that feature off. This returns the warning for features
         * not reported yet in this panel session, or "" when there are none. */
        var reportedProbeFailures = {};
        function probeFailureNotice(discovery) {
            var failed = discovery && discovery.failed;
            if (!failed || typeof failed !== "object") return "";
            var names = Object.keys(failed).filter(function (name) {
                var entry = failed[name] || {};
                if (entry.state !== "crashed" && entry.state !== "timeout") return false;
                if (reportedProbeFailures[name]) return false;
                reportedProbeFailures[name] = true;
                return true;
            });
            if (!names.length) return "";
            return fillTemplate(
                t("toast.capability_probe_failed", "Optional feature check failed for {features}. Those stay off, and the backend is still running. crash.log has the details."),
                { features: names.join(", ") }
            );
        }

        /* The GPU name arrives with the video AI probe, which can finish
         * after the first poll. */
        function probedGpuName() {
            var caps = getCapabilities();
            var videoAi = caps && caps.video_ai;
            return videoAi && videoAi.gpu_name ? String(videoAi.gpu_name) : "";
        }

        /* Whisper's probe state from a /health payload. `captions` stays
         * absent until the probe child reports, so that reads as checking
         * unless the check itself crashed or hung, which gives a reason. */
        function whisperProbe(healthData) {
            var caps = (healthData && healthData.capabilities) || {};
            var discovery = healthData && healthData.capability_discovery;
            var failure = discovery && discovery.failed ? discovery.failed.captions || null : null;
            return {
                checking: caps.captions === undefined && !failure,
                failureReason: failure && failure.reason ? String(failure.reason) : "",
            };
        }

        return {
            probeFailureNotice: probeFailureNotice,
            probedGpuName: probedGpuName,
            whisperProbe: whisperProbe,
            weightsMissing: weightsMissing,
            weightsDownloadBlocked: weightsDownloadBlocked,
            weightsHintMessage: weightsHintMessage,
            hintHasState: hintHasState,
            syncWeightsHint: syncWeightsHint,
            weightsInstallConfig: weightsInstallConfig,
        };
    }

    return {
        fillTemplate: fillTemplate,
        createCapabilityStatus: createCapabilityStatus,
    };
});
