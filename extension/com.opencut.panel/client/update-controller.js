/* OpenCut CEP update lifecycle controller. Classic script + CommonJS for tests. */
(function (root, factory) {
    "use strict";
    var api = factory();
    if (typeof module === "object" && module.exports) module.exports = api;
    if (root) root.OpenCutUpdateController = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
    "use strict";

    var DISMISSED_VERSION_KEY = "opencut_update_dismissed_version";

    function createUpdateController(options) {
        options = options || {};
        var documentRef = options.documentRef
            || (typeof document !== "undefined" ? document : null);
        var storage = options.storage;
        if (!storage && typeof localStorage !== "undefined") storage = localStorage;
        var request = typeof options.request === "function"
            ? options.request
            : function (method, path, body, callback) { callback(new Error("Update request is unavailable.")); };
        var translate = typeof options.translate === "function"
            ? options.translate
            : function (key, fallback) { return fallback || key; };
        var showToast = typeof options.showToast === "function"
            ? options.showToast
            : function () {};
        var normalizeReleaseUrl = typeof options.normalizeReleaseUrl === "function"
            ? options.normalizeReleaseUrl
            : function () { return ""; };
        var openExternalUrl = typeof options.openExternalUrl === "function"
            ? options.openExternalUrl
            : function () { return false; };
        var currentVersion = options.currentVersion || "";
        var updateCheckDone = false;
        var updateCheckFailed = false;
        var updateCheckInFlight = false;
        var latestUpdate = null;
        var bound = false;
        var disposed = false;
        var cleanupCallbacks = [];

        function t(key, fallback) {
            return translate(key, fallback);
        }

        function getElement(id) {
            return documentRef && typeof documentRef.getElementById === "function"
                ? documentRef.getElementById(id)
                : null;
        }

        function getDismissedVersion() {
            try {
                return storage && typeof storage.getItem === "function"
                    ? storage.getItem(DISMISSED_VERSION_KEY) || ""
                    : "";
            } catch (error) {
                return "";
            }
        }

        function setDismissedVersion(version) {
            try {
                if (storage && typeof storage.setItem === "function") {
                    storage.setItem(DISMISSED_VERSION_KEY, String(version || ""));
                }
            } catch (error) {}
        }

        function setElementHidden(id, hidden) {
            var node = getElement(id);
            if (!node) return;
            node.hidden = !!hidden;
            if (node.classList && typeof node.classList.toggle === "function") {
                node.classList.toggle("hidden", !!hidden);
            }
        }

        function formatPublishedAt(value) {
            if (!value) return "";
            try {
                var date = new Date(value);
                if (!isNaN(date.getTime())) return date.toLocaleDateString();
            } catch (error) {}
            return String(value);
        }

        function render(result, checking) {
            var card = getElement("updateNoticeCard");
            var status = getElement("updateStatusText");
            var summary = getElement("updateSummary");
            var current = getElement("updateCurrentVersion");
            var available = getElement("updateAvailableVersion");
            var releaseName = getElement("updateReleaseName");
            var notes = getElement("updateReleaseNotes");
            var notesDetails = getElement("updateNotesDetails");
            var retry = getElement("updateRetryBtn");
            if (!card || !status || !summary) return;

            var displayedVersion = (result && result.current_version)
                || (current && current.textContent)
                || currentVersion
                || "—";
            if (current) current.textContent = displayedVersion;
            if (retry) retry.disabled = !!checking;

            if (checking) {
                card.setAttribute("data-state", "checking");
                status.setAttribute("data-state", "working");
                status.textContent = t("settings.update_checking", "Checking…");
                summary.textContent = t("settings.update_checking_summary", "Checking GitHub for the latest OpenCut release.");
                setElementHidden("updateReleaseDetails", true);
                return;
            }

            if (!result || result.error || !result.latest_version) {
                card.setAttribute("data-state", "error");
                status.setAttribute("data-state", "error");
                status.textContent = t("settings.update_unavailable_status", "Unavailable");
                summary.textContent = t("settings.update_check_failed", "Couldn't check for updates. Click Check again to retry.");
                if (available) available.textContent = "—";
                setElementHidden("updateReleaseDetails", true);
                return;
            }

            var latestVersion = String(result.latest_version);
            if (available) available.textContent = latestVersion;
            if (!result.update_available) {
                card.setAttribute("data-state", "current");
                status.setAttribute("data-state", "success");
                status.textContent = t("settings.update_current_status", "Up to date");
                summary.textContent = t("settings.update_up_to_date", "You're up to date on v{version}.").replace("{version}", displayedVersion);
                setElementHidden("updateReleaseDetails", true);
                return;
            }

            if (getDismissedVersion() === latestVersion) {
                card.setAttribute("data-state", "dismissed");
                status.setAttribute("data-state", "neutral");
                status.textContent = t("settings.update_dismissed_status", "Dismissed");
                summary.textContent = t("settings.update_dismissed", "Update v{version} is dismissed for this panel. A newer release will appear here.").replace("{version}", latestVersion);
                setElementHidden("updateReleaseDetails", true);
                return;
            }

            card.setAttribute("data-state", "available");
            status.setAttribute("data-state", "warning");
            status.textContent = t("settings.update_available_status", "Update available");
            var published = formatPublishedAt(result.published_at);
            var availableSummary = t(
                "settings.update_available_summary",
                "OpenCut v{version} is available. Review the release notes before opening GitHub."
            ).replace("{version}", latestVersion);
            if (published) availableSummary += " " + t("settings.update_published_at", "Published {date}.").replace("{date}", published);
            summary.textContent = availableSummary;
            if (releaseName) releaseName.textContent = result.release_name || ("OpenCut " + latestVersion);
            if (notes) notes.textContent = result.release_notes || t("settings.update_no_release_notes", "No release notes were published.");
            if (notesDetails) notesDetails.hidden = !result.release_notes;
            setElementHidden("updateReleaseDetails", false);
        }

        function open() {
            var raw = latestUpdate && latestUpdate.release_url;
            var releaseUrl = normalizeReleaseUrl(raw);
            if (!releaseUrl) {
                showToast(t("settings.update_invalid_release", "The release link could not be verified."), "error");
                return false;
            }
            try {
                if (!openExternalUrl(releaseUrl)) throw new Error("No browser launch API is available.");
                showToast(t("settings.update_opened", "Release page opened."), "success");
                return true;
            } catch (error) {
                showToast(t("settings.update_invalid_release", "The release link could not be verified."), "error");
                return false;
            }
        }

        function dismiss() {
            if (!latestUpdate || !latestUpdate.latest_version) return false;
            setDismissedVersion(latestUpdate.latest_version);
            render(latestUpdate, false);
            showToast(
                t("settings.update_dismissed_toast", "Update v{version} dismissed until a newer release is available.")
                    .replace("{version}", latestUpdate.latest_version),
                "info"
            );
            return true;
        }

        function check(force) {
            force = !!force;
            if (disposed || updateCheckInFlight || updateCheckDone || (updateCheckFailed && !force)) return false;
            updateCheckInFlight = true;
            render(null, true);
            var settled = false;
            function complete(error, data) {
                if (settled) return;
                settled = true;
                updateCheckInFlight = false;
                if (disposed) return;
                if (error || !data || data.error || !data.latest_version) {
                    updateCheckDone = false;
                    updateCheckFailed = true;
                    render(data || { error: "offline" }, false);
                    showToast(t("toast.update_check_failed", "Couldn't check for updates. Click Refresh to try again."), "warning");
                    return;
                }
                updateCheckDone = true;
                updateCheckFailed = false;
                latestUpdate = data;
                render(data, false);
                if (data.update_available && getDismissedVersion() !== String(data.latest_version)) {
                    var template = t("toast.update_available", "OpenCut v{version} available — visit GitHub to update");
                    showToast(template.replace("{version}", data.latest_version || ""), "info");
                }
            }
            try {
                request("GET", "/system/update-check", null, complete);
            } catch (error) {
                complete(error, null);
            }
            return true;
        }

        function listen(target, type, handler) {
            if (!target || typeof target.addEventListener !== "function") return;
            target.addEventListener(type, handler);
            cleanupCallbacks.push(function () {
                if (target && typeof target.removeEventListener === "function") target.removeEventListener(type, handler);
            });
        }

        function bind() {
            if (disposed || bound) return;
            bound = true;
            listen(getElement("updateRetryBtn"), "click", function () { check(true); });
            listen(getElement("updateOpenBtn"), "click", open);
            listen(getElement("updateDismissBtn"), "click", dismiss);
        }

        function dispose() {
            if (disposed) return;
            disposed = true;
            for (var i = cleanupCallbacks.length - 1; i >= 0; i--) cleanupCallbacks[i]();
            cleanupCallbacks = [];
            bound = false;
            updateCheckInFlight = false;
        }

        return {
            bind: bind,
            check: check,
            dismiss: dismiss,
            dispose: dispose,
            getState: function () {
                return {
                    done: updateCheckDone,
                    failed: updateCheckFailed,
                    inFlight: updateCheckInFlight,
                    latest: latestUpdate
                };
            },
            open: open,
            render: render
        };
    }

    return {
        DISMISSED_VERSION_KEY: DISMISSED_VERSION_KEY,
        createUpdateController: createUpdateController
    };
});
