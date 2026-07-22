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
            mode: options.mode
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

    return {
        cloneCuts: cloneCuts,
        buildBeatMarkers: buildBeatMarkers,
        buildRenameOperations: buildRenameOperations,
        buildSmartBinHostRules: buildSmartBinHostRules,
        buildOtioPayload: buildOtioPayload
    };
});
