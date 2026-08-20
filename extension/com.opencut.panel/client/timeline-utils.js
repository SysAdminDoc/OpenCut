/* OpenCut CEP timeline payload helpers. Classic script + CommonJS for tests. */
(function (root, factory) {
    "use strict";
    var api = factory();
    if (typeof module === "object" && module.exports) module.exports = api;
    if (root) root.OpenCutTimeline = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
    "use strict";

    function cloneCuts(cuts) {
        if (!Array.isArray(cuts)) return [];
        return cuts.map(function (cut) {
            var copy = {};
            for (var key in cut) {
                if (Object.prototype.hasOwnProperty.call(cut, key)) copy[key] = cut[key];
            }
            return copy;
        });
    }

    function buildBeatMarkers(times, name, type) {
        return (Array.isArray(times) ? times : []).map(function (time) {
            return { time: time, name: name, type: type };
        });
    }

    function buildRenameOperations(items, edits) {
        var operations = [];
        items = Array.isArray(items) ? items : [];
        edits = Array.isArray(edits) ? edits : [];
        for (var i = 0; i < edits.length; i++) {
            var edit = edits[i] || {};
            var index = parseInt(edit.index, 10);
            var item = items[index];
            if (item && edit.value !== item.name) {
                operations.push({
                    nodeId: item.nodeId || item.id || item.path,
                    newName: edit.value
                });
            }
        }
        return operations;
    }

    function buildSmartBinHostRules(rules) {
        return (Array.isArray(rules) ? rules : []).map(function (rule) {
            return {
                binName: rule.bin_name,
                rule: rule.rule_type,
                field: rule.field,
                value: rule.value
            };
        });
    }

    function buildOtioPayload(options) {
        options = options || {};
        var payload = {
            filepath: options.filepath,
            output_dir: options.outputDir,
            mode: options.mode,
            adapter_name: options.adapterName || "otio_json",
            schema_target: options.schemaTarget || "current",
            accept_lossy: options.acceptLossy === true
        };
        if (options.mode === "cuts") {
            payload.cuts = cloneCuts(options.cuts);
        } else if (options.mode === "markers") {
            if (options.beatTimes && options.beatTimes.length) {
                payload.markers = buildBeatMarkers(options.beatTimes, options.beatLabel, undefined).map(function (marker) {
                    delete marker.type;
                    return marker;
                });
            } else {
                payload.markers = (options.chapters || []).map(function (chapter) {
                    return {
                        time: chapter.seconds || chapter.start || chapter.time || 0,
                        name: chapter.title || options.chapterLabel
                    };
                });
            }
        }
        return payload;
    }

    // Repeat detection returns clusters of attempts at the same line, each
    // naming the take to keep. Turn them into cut ranges for everything else,
    // carrying the kept take's text so review can show what survives.
    function buildRepeatCutsFromClusters(clusters) {
        if (!clusters || !clusters.length) return null;
        var cuts = [];
        for (var ci = 0; ci < clusters.length; ci++) {
            var cluster = clusters[ci] || {};
            var takes = cluster.takes || [];
            var byIndex = {};
            for (var ti = 0; ti < takes.length; ti++) byIndex[takes[ti].index] = takes[ti];
            var keep = byIndex[cluster.keep_index];
            var cutIndices = cluster.cut_indices || [];
            for (var ki = 0; ki < cutIndices.length; ki++) {
                var take = byIndex[cutIndices[ki]];
                if (!take) continue;
                cuts.push({
                    start: take.start,
                    end: take.end,
                    text: take.text,
                    keep_text: keep ? keep.text : "",
                    decision_source: cluster.decision_source || "heuristic"
                });
            }
        }
        return cuts.length ? cuts : null;
    }

    return {
        cloneCuts: cloneCuts,
        buildBeatMarkers: buildBeatMarkers,
        buildRenameOperations: buildRenameOperations,
        buildSmartBinHostRules: buildSmartBinHostRules,
        buildOtioPayload: buildOtioPayload,
        buildRepeatCutsFromClusters: buildRepeatCutsFromClusters
    };
});
