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

    function hardwareFor(featureId) {
        var rec = getFeature(featureId);
        if (!rec) return null;
        return {
            hardware: rec.hardware || "",
            requiresGpu: !!rec.requires_gpu,
            minimumVramMb: Number(rec.minimum_vram_mb || 0),
        };
    }

    function privacyFor(featureId) {
        var rec = getFeature(featureId);
        if (!rec) return null;
        return {
            privacy: rec.privacy || "",
            license: rec.license || "",
            advisoryNotes: rec.advisory_notes || [],
        };
    }

    var PRIVACY_BADGES = {
        "local-only": { chip: "Local", tone: "success" },
        "cloud": { chip: "Cloud", tone: "caution" },
    };

    function _privacyChip(privacy) {
        if (!privacy) return null;
        var key = privacy.toLowerCase();
        if (key === "local-only") return PRIVACY_BADGES["local-only"];
        if (key.indexOf("cloud") !== -1) return PRIVACY_BADGES["cloud"];
        return null;
    }

    function _hardwareSummary(record) {
        if (!record) return "";
        var parts = [];
        if (record.hardware) {
            parts.push(record.hardware);
        }
        var minVram = Number(record.minimum_vram_mb || 0);
        if (minVram > 0) {
            parts.push("minimum " + Math.round(minVram / 1024) + " GB VRAM");
        }
        return parts.join("; ");
    }

    function badgeFor(featureId) {
        var rec = getFeature(featureId);
        if (!rec || rec.state === "available") return null;
        var badge = STATE_BADGES[rec.state] || null;
        if (!badge) return null;
        return Object.assign({}, badge, { record: rec });
    }

    function _annotateHardware(el, record) {
        if (!el || !record) return;
        var summary = _hardwareSummary(record);
        if (summary) {
            el.setAttribute("data-feature-hardware", record.hardware || "");
            if (record.minimum_vram_mb) {
                el.setAttribute("data-feature-min-vram-mb", String(record.minimum_vram_mb));
            }
            if (record.requires_gpu) {
                el.setAttribute("data-feature-requires-gpu", "true");
            }
        }
        if (record.privacy) {
            el.setAttribute("data-feature-privacy", record.privacy);
            var pChip = _privacyChip(record.privacy);
            if (pChip) {
                el.setAttribute("data-feature-privacy-chip", pChip.chip);
            }
        }
        if (record.license) {
            el.setAttribute("data-feature-license", record.license);
        }
        var parts = [];
        if (summary) parts.push("Hardware: " + summary);
        if (record.privacy) parts.push("Privacy: " + record.privacy);
        if (record.license) parts.push("License: " + record.license);
        if (parts.length) {
            var title = el.title || "";
            var newInfo = parts.join("\n");
            if (title.indexOf("Hardware:") === -1 && title.indexOf("Privacy:") === -1) {
                el.title = title ? title + "\n" + newInfo : newInfo;
            }
        }
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
        var hardware = _hardwareSummary(badge.record);
        if (hardware) {
            hint += "\nHardware: " + hardware;
        }
        if (badge.record.privacy) {
            hint += "\nPrivacy: " + badge.record.privacy;
        }
        if (badge.record.license) {
            hint += "\nLicense: " + badge.record.license;
        }
        var notes = badge.record.advisory_notes || [];
        if (notes.length) {
            hint += "\nAdvisory: " + notes[0];
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
            var record = getFeature(featureId);
            if (record) {
                _annotateHardware(el, record);
            }
            var badge = badgeFor(featureId);
            if (!badge) continue;
            _styleElement(el, badge);
            gated += 1;
        }
        return gated;
    }

    var api = {
        STATE_BADGES: STATE_BADGES,
        PRIVACY_BADGES: PRIVACY_BADGES,
        fetchManifest: fetchManifest,
        getFeature: getFeature,
        isAvailable: isAvailable,
        hardwareFor: hardwareFor,
        privacyFor: privacyFor,
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
