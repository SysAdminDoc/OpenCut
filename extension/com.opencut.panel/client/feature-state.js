/*
 * OpenCut feature readiness helper (F100).
 *
 * Fetches `GET /system/feature-state` once on panel boot and exposes a
 * tiny API that any tab can use to grey out buttons whose backend isn't
 * `available`. The naming is intentionally narrow — this is plumbing,
 * not a router. Per-button adoption is opt-in via:
 *
 *     <button class="oc-btn" data-feature-id="audio.demucs">Separate stems</button>
 *
 * On boot we call `applyGating(root)` once; tabs can call it again on
 * dynamic content to re-evaluate. The helper never blocks UI: if the
 * fetch fails (panel offline, server down) buttons stay clickable and
 * the existing 503 error flow handles the click.
 */
(function (global) {
    "use strict";

    var STATE_BADGES = {
        stub: {
            chip: "Coming soon",
            tone: "warning",
            tooltip:
                "Backend stub. The route exists for MCP/script clients but the implementation is on the roadmap.",
        },
        missing_dependency: {
            chip: "Install required",
            tone: "warning",
            tooltip: "Optional dependency is not installed locally. Open Settings → Models for instructions.",
        },
        experimental: {
            chip: "Experimental",
            tone: "info",
            tooltip:
                "Works but isn't covered by the standard release-gate tests yet. Treat output as preview-grade.",
        },
    };

    var state = {
        loaded: false,
        loading: null,
        manifest: null,
        byFeatureId: Object.create(null),
    };

    function _registerFeature(record) {
        if (!record || !record.feature_id) return;
        state.byFeatureId[record.feature_id] = record;
    }

    function _absorbManifest(manifest) {
        state.manifest = manifest;
        state.byFeatureId = Object.create(null);
        var features = (manifest && manifest.features) || [];
        for (var i = 0; i < features.length; i++) {
            _registerFeature(features[i]);
        }
        state.loaded = true;
    }

    function fetchManifest(backendUrl) {
        if (state.loading) return state.loading;
        var base = (backendUrl || "").replace(/\/$/, "");
        var url = base + "/system/feature-state";

        state.loading = fetch(url, { credentials: "same-origin" })
            .then(function (resp) {
                if (!resp.ok) {
                    throw new Error("feature-state HTTP " + resp.status);
                }
                return resp.json();
            })
            .then(function (manifest) {
                _absorbManifest(manifest);
                return manifest;
            })
            .catch(function (err) {
                state.loaded = false;
                state.manifest = null;
                state.byFeatureId = Object.create(null);
                throw err;
            })
            .finally(function () {
                state.loading = null;
            });

        return state.loading;
    }

    function getFeature(featureId) {
        return state.byFeatureId[featureId] || null;
    }

    function isAvailable(featureId) {
        var rec = getFeature(featureId);
        if (!rec) return true; // optimistic: unknown ids stay enabled
        return rec.state === "available";
    }

    function badgeFor(featureId) {
        var rec = getFeature(featureId);
        if (!rec || rec.state === "available") return null;
        var badge = STATE_BADGES[rec.state] || null;
        if (!badge) return null;
        return Object.assign({}, badge, { record: rec });
    }

    function _styleElement(el, badge) {
        if (!el || !badge) return;
        el.classList.add("oc-feature-gated");
        el.classList.add("oc-feature-state-" + (badge.record.state || "unknown"));
        if (el.tagName === "BUTTON" || el.tagName === "INPUT") {
            el.disabled = true;
        } else {
            el.setAttribute("aria-disabled", "true");
        }
        var hint = badge.tooltip;
        if (badge.record.install_hint) {
            hint += "\nHint: " + badge.record.install_hint;
        }
        if (badge.record.docs) {
            hint += "\nDocs: " + badge.record.docs;
        }
        el.title = hint;
        el.setAttribute("data-feature-state", badge.record.state);
        el.setAttribute("data-feature-chip", badge.chip);
    }

    function applyGating(root) {
        if (!state.loaded) return 0;
        var scope = root || document;
        var nodes = scope.querySelectorAll
            ? scope.querySelectorAll("[data-feature-id]")
            : [];
        var gated = 0;
        for (var i = 0; i < nodes.length; i++) {
            var el = nodes[i];
            var featureId = el.getAttribute("data-feature-id");
            var badge = badgeFor(featureId);
            if (!badge) continue;
            _styleElement(el, badge);
            gated += 1;
        }
        return gated;
    }

    var api = {
        STATE_BADGES: STATE_BADGES,
        fetchManifest: fetchManifest,
        getFeature: getFeature,
        isAvailable: isAvailable,
        badgeFor: badgeFor,
        applyGating: applyGating,
        // Test helpers — never call from production code.
        _absorbManifest: _absorbManifest,
        _reset: function () {
            state.loaded = false;
            state.loading = null;
            state.manifest = null;
            state.byFeatureId = Object.create(null);
        },
        _state: state,
    };

    if (typeof module !== "undefined" && module.exports) {
        module.exports = api;
    } else {
        global.OpenCutFeatureState = api;
    }
})(typeof window !== "undefined" ? window : globalThis);
