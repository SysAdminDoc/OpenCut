/* OpenCut CEP host-write verification normalizer. Classic script + CommonJS. */
(function (root, factory) {
    "use strict";
    var api = factory();
    if (typeof module === "object" && module.exports) module.exports = api;
    if (root) root.OpenCutHostWriteVerification = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
    "use strict";

    var latestVerification = null;
    var HOST_WRITE_COPY = {
        statusKey: "journal.host_write_no_change",
        hintKey: "journal.host_write_unverified"
    };

    function reportedCount(result) {
        var keys = ["applied", "added", "count", "created", "renamed", "removed", "imported"];
        for (var i = 0; i < keys.length; i++) {
            if (result[keys[i]] != null) return Number(result[keys[i]]);
        }
        return null;
    }

    function translate(t, key, fallback) {
        return typeof t === "function" ? t(key, fallback) : fallback;
    }

    function ensure(result, spec, t, notify) {
        result = result || {};
        spec = spec || {};
        if (!result.error && !result.host_write_verification) {
            result.host_write_verification = {
                schema: "opencut.host_write_verification.v1",
                action: spec.action || "unknown",
                host_version: result.host_version || "",
                host: result.host || { bridge: "cep", app_name: "Premiere Pro", version: result.host_version || "" },
                attempted_count: null,
                reported_count: reportedCount(result),
                verified_count: null,
                verification_status: "unverified",
                read_back_method: "unavailable: host operation returned no verification contract",
                before_state: null,
                after_state: null,
                detail: "The bridge accepted this write but did not expose an independent read-back; it is intentionally not counted as applied."
            };
            result.attempted_count = null;
            result.reported_count = result.host_write_verification.reported_count;
            result.verified_count = null;
            result.verification_status = "unverified";
            result.read_back_method = result.host_write_verification.read_back_method;
            result.unverified = true;
            result.warning = translate(t, HOST_WRITE_COPY.hintKey, "Premiere accepted this request, but independent read-back is unavailable. Review recovery diagnostics before relying on it.");
        }

        var verification = result.host_write_verification;
        if (verification) latestVerification = verification;
        if (!result.error && verification && verification.verification_status === "failed") {
            result.error_code = "HOST_WRITE_NOT_APPLIED";
            result.error = translate(t, HOST_WRITE_COPY.statusKey, "Premiere reported success but independent read-back found no timeline or project change.");
            if (typeof notify === "function") notify(result.error, "error");
        }
        if (!result.error && verification &&
                (verification.verification_status === "unverified" || verification.verification_status === "partial") &&
                typeof notify === "function") {
            notify(result.warning || translate(t, HOST_WRITE_COPY.hintKey, "Premiere accepted this request, but independent read-back is unavailable. Review recovery diagnostics before relying on it."), "warning");
        }
        return result;
    }

    function parse(rawResult, spec, t, notify) {
        var parsed;
        try { parsed = JSON.parse(rawResult || "{}"); }
        catch (parseErr) { parsed = { error: rawResult || parseErr.message }; }
        return ensure(parsed, spec, t, notify);
    }

    return {
        ensure: ensure,
        latest: function () { return latestVerification; },
        parse: parse
    };
});
