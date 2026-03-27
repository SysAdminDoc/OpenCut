/**
 * OpenCut ExtendScript Mock Harness
 * ==================================
 * Provides a fake Premiere Pro DOM so the ExtendScript host functions
 * (extension/com.opencut.panel/host/index.jsx) can be executed and tested
 * inside Node.js without Premiere Pro.
 *
 * Run:  node tests/jsx_mock.js
 */

// ---------------------------------------------------------------------------
// 1. Simple test runner
// ---------------------------------------------------------------------------

var _tests = [];
var _passed = 0;
var _failed = 0;
var _errors = [];

function describe(name, fn) {
    console.log("\n--- " + name + " ---");
    fn();
}

function it(name, fn) {
    _tests.push(name);
    try {
        fn();
        _passed++;
        console.log("  PASS  " + name);
    } catch (e) {
        _failed++;
        _errors.push({ test: name, error: e });
        console.log("  FAIL  " + name);
        console.log("        " + (e.message || e));
    }
}

function assert(condition, msg) {
    if (!condition) {
        throw new Error("Assertion failed: " + (msg || ""));
    }
}

function assertEqual(actual, expected, msg) {
    if (actual !== expected) {
        throw new Error(
            (msg ? msg + " -- " : "") +
            "expected " + JSON.stringify(expected) +
            " but got " + JSON.stringify(actual)
        );
    }
}

function assertDeepEqual(actual, expected, msg) {
    var a = JSON.stringify(actual);
    var b = JSON.stringify(expected);
    if (a !== b) {
        throw new Error(
            (msg ? msg + " -- " : "") +
            "expected " + b + " but got " + a
        );
    }
}

// ---------------------------------------------------------------------------
// 2. Fake Premiere Pro DOM
// ---------------------------------------------------------------------------

/** Time object used by Premiere for start/end/duration values */
function Time(seconds) {
    this.seconds = seconds;
    this.ticks = String(Math.round(seconds * 254016000000));
}

/** ProjectItem mock */
function ProjectItem(opts) {
    opts = opts || {};
    this.name = opts.name || "Untitled";
    this.nodeId = opts.nodeId || ("node-" + Math.random().toString(36).substr(2, 8));
    this.type = opts.type !== undefined ? opts.type : 1; // 1 = clip, 2 = bin
    this.treePath = opts.treePath || "";
    this.children = opts.children || _makeItemCollection([]);
    this._mediaPath = opts.mediaPath || "";
}
ProjectItem.prototype.getMediaPath = function () { return this._mediaPath; };
ProjectItem.prototype.setOverrideFrameRate = function (fps) { this._fps = fps; };
ProjectItem.prototype.createBin = function (binName) {
    var bin = new ProjectItem({ name: binName, type: 2 });
    this.children._items.push(bin);
    this.children.numItems = this.children._items.length;
    return bin;
};

/** Collection helper -- mimics the Premiere numItems / [index] pattern */
function _makeItemCollection(arr) {
    var col = {
        _items: arr,
        numItems: arr.length
    };
    // Proxy-like index access via defineProperty (works in Node)
    for (var i = 0; i < 500; i++) {
        (function (idx) {
            Object.defineProperty(col, idx, {
                get: function () { return col._items[idx]; },
                configurable: true,
                enumerable: false
            });
        })(i);
    }
    return col;
}

/** TrackItem (clip on a timeline track) */
function TrackItem(opts) {
    opts = opts || {};
    this.name = opts.name || "Clip";
    this.start = new Time(opts.startSeconds || 0);
    this.end = new Time(opts.endSeconds || 0);
    this.duration = new Time((opts.endSeconds || 0) - (opts.startSeconds || 0));
    this.inPoint = new Time(opts.inPoint || 0);
    this.outPoint = new Time(opts.outPoint || (opts.endSeconds || 0));
    this.projectItem = opts.projectItem || null;
    this._removed = false;
}
TrackItem.prototype.remove = function (ripple, align) {
    this._removed = true;
};

/** Track mock */
function Track(clips) {
    this.clips = _makeItemCollection(clips || []);
}

/** TrackCollection: videoTracks / audioTracks container */
function TrackCollection(tracks) {
    this._tracks = tracks || [];
    this.numTracks = this._tracks.length;
    for (var i = 0; i < this._tracks.length; i++) {
        this[i] = this._tracks[i];
    }
}

/** Marker mock */
function MarkerMock(time) {
    this.start = new Time(time);
    this.end = new Time(time);
    this.name = "";
    this.type = 0;
    this.comments = "";
    this.colorByteArray = [0, 0, 0, 255];
}

/** Markers collection */
function MarkersCollection() {
    this._markers = [];
    this.numMarkers = 0;
}
MarkersCollection.prototype.createMarker = function (time) {
    var m = new MarkerMock(time);
    this._markers.push(m);
    this.numMarkers = this._markers.length;
    return m;
};

/** Sequence mock */
function Sequence(opts) {
    opts = opts || {};
    this.name = opts.name || "Test Sequence";
    this.sequenceID = opts.sequenceID || "seq-001";
    this.videoTracks = new TrackCollection(opts.videoTracks || []);
    this.audioTracks = new TrackCollection(opts.audioTracks || []);
    this.markers = new MarkersCollection();
    this.end = opts.end ? new Time(opts.end) : new Time(60);
    this.frameSizeHorizontal = opts.width || 1920;
    this.frameSizeVertical = opts.height || 1080;
}
Sequence.prototype.getPlayerPosition = function () {
    return new Time(0);
};

/** Fake CSInterface / CSEvent for CEP communication */
function CSInterface() {
    this._listeners = {};
}
CSInterface.prototype.addEventListener = function (type, fn) {
    this._listeners[type] = this._listeners[type] || [];
    this._listeners[type].push(fn);
};
CSInterface.prototype.dispatchEvent = function (evt) {
    var fns = this._listeners[evt.type] || [];
    for (var i = 0; i < fns.length; i++) fns[i](evt);
};
CSInterface.prototype.evalScript = function (script, cb) {
    if (cb) cb("mock");
};
CSInterface.prototype.getSystemPath = function () { return "/tmp"; };

function CSEvent(type, scope) {
    this.type = type;
    this.scope = scope || "APPLICATION";
    this.data = "";
}

/** Fake File / Folder used by ocAddNativeCaptionTrack */
function FakeFile(path) {
    this.fsName = path;
    this._content = "";
    this._open = false;
}
FakeFile.prototype.open = function (mode) { this._open = true; return true; };
FakeFile.prototype.writeln = function (line) { this._content += line + "\n"; };
FakeFile.prototype.close = function () { this._open = false; };
FakeFile.prototype.remove = function () {};

// ---------------------------------------------------------------------------
// 3. Global environment (mimics ExtendScript globals)
// ---------------------------------------------------------------------------

// $ object (ExtendScript built-in)
var $ = { writeln: function () {} };

// Folder.temp
var Folder = { temp: { fsName: "/tmp" } };

// File constructor override for caption test
var _lastTempFile = null;
var _OrigFile = FakeFile;
// We will override `File` inside the global scope injected into the JSX code
var File = function (path) {
    _lastTempFile = new FakeFile(path);
    return _lastTempFile;
};

// ---------------------------------------------------------------------------
// 4. Build default mock app state
// ---------------------------------------------------------------------------

function buildMockApp(opts) {
    opts = opts || {};

    // Default project items
    var clip1 = new ProjectItem({ name: "Interview.mp4", nodeId: "node-001", type: 1, mediaPath: "/media/Interview.mp4" });
    var clip2 = new ProjectItem({ name: "Broll_001.mp4", nodeId: "node-002", type: 1, mediaPath: "/media/Broll_001.mp4" });
    var clip3 = new ProjectItem({ name: "Music.wav", nodeId: "node-003", type: 1, mediaPath: "/media/Music.wav" });
    var bin1 = new ProjectItem({ name: "Footage", nodeId: "node-bin-001", type: 2 });

    var rootItem = new ProjectItem({
        name: "Root",
        nodeId: "root",
        type: 2,
        children: _makeItemCollection(opts.rootChildren || [clip1, clip2, clip3, bin1])
    });

    // Default sequence with clips on tracks
    var vClip1 = new TrackItem({ name: "V-Interview", startSeconds: 0, endSeconds: 10, projectItem: clip1 });
    var vClip2 = new TrackItem({ name: "V-Broll", startSeconds: 10, endSeconds: 20, projectItem: clip2 });
    var vClip3 = new TrackItem({ name: "V-Outro", startSeconds: 20, endSeconds: 30, projectItem: clip2 });
    var aClip1 = new TrackItem({ name: "A-Interview", startSeconds: 0, endSeconds: 10 });
    var aClip2 = new TrackItem({ name: "A-Music", startSeconds: 10, endSeconds: 25 });

    var videoTrack1 = new Track([vClip1, vClip2, vClip3]);
    var audioTrack1 = new Track([aClip1, aClip2]);

    var seq = new Sequence({
        name: "Main Edit",
        videoTracks: opts.videoTracks || [videoTrack1],
        audioTracks: opts.audioTracks || [audioTrack1],
        end: opts.sequenceEnd || 30
    });

    return {
        project: {
            rootItem: rootItem,
            activeSequence: seq,
            name: "TestProject",
            path: "/projects/TestProject.prproj",
            importFiles: function (files, suppressUI, targetBin, asNumbered) {
                // Fake import: just add a ProjectItem to the target bin
                for (var f = 0; f < files.length; f++) {
                    var pi = new ProjectItem({ name: files[f].split("/").pop(), type: 1, mediaPath: files[f] });
                    if (targetBin && targetBin.children) {
                        targetBin.children._items.push(pi);
                        targetBin.children.numItems = targetBin.children._items.length;
                    }
                }
            }
        }
    };
}

// ---------------------------------------------------------------------------
// 5. Load the ExtendScript host code
// ---------------------------------------------------------------------------
//    We read the file as text and eval it so all functions land in our scope
//    where `app`, `$`, `File`, `Folder` are already defined.

var fs = require("fs");
var path = require("path");

var jsxPath = path.resolve(__dirname, "..", "extension", "com.opencut.panel", "host", "index.jsx");
var jsxCode = fs.readFileSync(jsxPath, "utf-8");

// Make `app` a global that tests can swap out
var app;

// Eval the JSX in current scope -- all function declarations become global
eval(jsxCode);

// ---------------------------------------------------------------------------
// 6. Tests
// ---------------------------------------------------------------------------

describe("ocApplySequenceCuts", function () {

    it("should remove clips fully within the cut range", function () {
        app = buildMockApp();
        var cuts = [{ start: 0, end: 10 }]; // Should remove first video clip (0-10) and first audio clip (0-10)
        var result = JSON.parse(ocApplySequenceCuts(JSON.stringify(cuts)));
        assert(!result.error, "should not error: " + result.error);
        assertEqual(result.applied, 2, "should remove 2 clips (1 video + 1 audio)");
        assertEqual(result.errors.length, 0, "no per-clip errors");
    });

    it("should handle multiple cuts sorted in reverse", function () {
        app = buildMockApp();
        // cut1 removes 0-10, cut2 removes 20-30
        var cuts = [
            { start: 0, end: 10 },
            { start: 20, end: 30 }
        ];
        var result = JSON.parse(ocApplySequenceCuts(JSON.stringify(cuts)));
        assert(!result.error, "should not error");
        // Video: clip 0-10 removed by cut1, clip 20-30 removed by cut2 => 2 video
        // Audio: clip 0-10 removed by cut1 => 1 audio
        assertEqual(result.applied, 3, "should remove 3 clips total");
    });

    it("should NOT remove clips that only partially overlap", function () {
        app = buildMockApp();
        // Cut range 5-15 partially overlaps clip 0-10 and 10-20, but neither is fully contained
        var cuts = [{ start: 5, end: 15 }];
        var result = JSON.parse(ocApplySequenceCuts(JSON.stringify(cuts)));
        assertEqual(result.applied, 0, "no clips fully within 5-15");
    });

    it("should return error for no project", function () {
        app = null;
        var result = JSON.parse(ocApplySequenceCuts("[]"));
        assert(result.error, "should return error when no project");
    });

    it("should return error for invalid JSON", function () {
        app = buildMockApp();
        var result = JSON.parse(ocApplySequenceCuts("{bad json"));
        assert(result.error && result.error.indexOf("Invalid JSON") !== -1, "should flag bad JSON");
    });

    it("should skip cuts where start >= end", function () {
        app = buildMockApp();
        var cuts = [{ start: 10, end: 5 }]; // invalid range
        var result = JSON.parse(ocApplySequenceCuts(JSON.stringify(cuts)));
        assertEqual(result.applied, 0, "skip reversed range");
    });

    it("should handle empty cuts array gracefully", function () {
        app = buildMockApp();
        var result = JSON.parse(ocApplySequenceCuts(JSON.stringify([])));
        assertEqual(result.applied, 0);
    });

    it("should respect 0.01s tolerance for floating-point matching", function () {
        app = buildMockApp();
        // Clip is exactly 0-10. Cut is 0.005-10.005 => within tolerance
        var cuts = [{ start: 0.005, end: 10.005 }];
        var result = JSON.parse(ocApplySequenceCuts(JSON.stringify(cuts)));
        // clip 0-10 video + clip 0-10 audio should both match within tolerance
        assertEqual(result.applied, 2, "clips within 0.01s tolerance should be removed");
    });
});

describe("ocBatchRenameProjectItems", function () {

    it("should rename items by nodeId", function () {
        app = buildMockApp();
        var renames = [
            { nodeId: "node-001", newName: "Main_Interview.mp4" },
            { nodeId: "node-003", newName: "Background_Music.wav" }
        ];
        var result = JSON.parse(ocBatchRenameProjectItems(JSON.stringify(renames)));
        assertEqual(result.renamed, 2, "should rename 2 items");
        assertEqual(result.errors.length, 0, "no errors");
        // Verify actual name change on the mock objects
        assertEqual(app.project.rootItem.children[0].name, "Main_Interview.mp4");
        assertEqual(app.project.rootItem.children[2].name, "Background_Music.wav");
    });

    it("should report error for missing nodeId", function () {
        app = buildMockApp();
        var renames = [{ nodeId: "node-nonexistent", newName: "Ghost.mp4" }];
        var result = JSON.parse(ocBatchRenameProjectItems(JSON.stringify(renames)));
        assertEqual(result.renamed, 0);
        assertEqual(result.errors.length, 1, "one error for missing nodeId");
    });

    it("should skip entries with empty newName", function () {
        app = buildMockApp();
        var renames = [{ nodeId: "node-001", newName: "" }];
        var result = JSON.parse(ocBatchRenameProjectItems(JSON.stringify(renames)));
        assertEqual(result.renamed, 0, "empty newName should be rejected");
        assertEqual(result.errors.length, 1);
    });

    it("should skip entries with missing nodeId field", function () {
        app = buildMockApp();
        var renames = [{ newName: "Orphan.mp4" }];
        var result = JSON.parse(ocBatchRenameProjectItems(JSON.stringify(renames)));
        assertEqual(result.renamed, 0);
        assertEqual(result.errors.length, 1);
    });

    it("should return error for no project", function () {
        app = null;
        var result = JSON.parse(ocBatchRenameProjectItems("[]"));
        assert(result.error, "should error when no project");
    });

    it("should return error for invalid JSON", function () {
        app = buildMockApp();
        var result = JSON.parse(ocBatchRenameProjectItems("not json"));
        assert(result.error && result.error.indexOf("Invalid JSON") !== -1);
    });
});

describe("ocAddSequenceMarkers", function () {

    it("should add markers to the active sequence", function () {
        app = buildMockApp();
        var markers = [
            { time: 5, name: "Intro", type: "chapter" },
            { time: 15, name: "Main Content", type: "comment" },
            { time: 25, name: "Outro", type: "segmentation" }
        ];
        var result = JSON.parse(ocAddSequenceMarkers(JSON.stringify(markers)));
        assertEqual(result.added, 3);
        assertEqual(result.errors.length, 0);
        // Check the markers were created on the sequence mock
        var seq = app.project.activeSequence;
        assertEqual(seq.markers.numMarkers, 3);
        assertEqual(seq.markers._markers[0].name, "Intro");
        assertEqual(seq.markers._markers[0].type, 1, "chapter type = 1");
        assertEqual(seq.markers._markers[1].type, 0, "comment type = 0");
        assertEqual(seq.markers._markers[2].type, 2, "segmentation type = 2");
    });

    it("should skip markers beyond sequence duration", function () {
        app = buildMockApp();
        var markers = [
            { time: 5, name: "OK" },
            { time: 999, name: "Way past end" }
        ];
        var result = JSON.parse(ocAddSequenceMarkers(JSON.stringify(markers)));
        assertEqual(result.added, 1, "only one marker within duration");
        assertEqual(result.errors.length, 1, "one error for out-of-range marker");
    });

    it("should clamp negative times to 0", function () {
        app = buildMockApp();
        var markers = [{ time: -5, name: "Negative" }];
        var result = JSON.parse(ocAddSequenceMarkers(JSON.stringify(markers)));
        assertEqual(result.added, 1);
        assertEqual(app.project.activeSequence.markers._markers[0].start.seconds, 0);
    });

    it("should return error for no active sequence", function () {
        app = buildMockApp();
        app.project.activeSequence = null;
        var result = JSON.parse(ocAddSequenceMarkers(JSON.stringify([{ time: 1, name: "X" }])));
        assert(result.error && result.error.indexOf("No active sequence") !== -1);
    });

    it("should handle empty markers array", function () {
        app = buildMockApp();
        var result = JSON.parse(ocAddSequenceMarkers(JSON.stringify([])));
        assertEqual(result.added, 0);
        assertEqual(result.errors.length, 0);
    });
});

describe("ocAddNativeCaptionTrack", function () {

    it("should generate SRT content and import captions", function () {
        app = buildMockApp();
        var segments = [
            { start: 0, end: 3, text: "Hello world" },
            { start: 3.5, end: 7, text: "Second line" }
        ];
        var result = JSON.parse(ocAddNativeCaptionTrack(JSON.stringify(segments)));
        assert(!result.error, "should not error: " + result.error);
        assertEqual(result.success, true);
        assertEqual(result.captions_added, 2);
    });

    it("should skip invalid segments (end <= start)", function () {
        app = buildMockApp();
        var segments = [
            { start: 5, end: 3, text: "Backwards" },   // invalid
            { start: 0, end: 2, text: "Valid" }
        ];
        var result = JSON.parse(ocAddNativeCaptionTrack(JSON.stringify(segments)));
        assertEqual(result.captions_added, 1, "only valid segment counted");
    });

    it("should skip segments with negative start", function () {
        app = buildMockApp();
        var segments = [
            { start: -1, end: 2, text: "Neg start" }, // segS < 0 => skip
            { start: 2, end: 5, text: "OK" }
        ];
        var result = JSON.parse(ocAddNativeCaptionTrack(JSON.stringify(segments)));
        assertEqual(result.captions_added, 1);
    });

    it("should return error for empty segments", function () {
        app = buildMockApp();
        var result = JSON.parse(ocAddNativeCaptionTrack(JSON.stringify([])));
        assert(result.error && result.error.indexOf("No caption segments") !== -1);
    });

    it("should return error for no project", function () {
        app = null;
        var result = JSON.parse(ocAddNativeCaptionTrack(JSON.stringify([{ start: 0, end: 1, text: "X" }])));
        assert(result.error);
    });

    it("should return error for invalid JSON", function () {
        app = buildMockApp();
        var result = JSON.parse(ocAddNativeCaptionTrack("{nope"));
        assert(result.error && result.error.indexOf("Invalid JSON") !== -1);
    });

    it("should create 'OpenCut Captions' bin if it does not exist", function () {
        app = buildMockApp();
        var rootBefore = app.project.rootItem.children.numItems;
        var segments = [{ start: 0, end: 1, text: "Test" }];
        ocAddNativeCaptionTrack(JSON.stringify(segments));
        // _findOrCreateBin should have created a new bin
        var rootAfter = app.project.rootItem.children.numItems;
        assert(rootAfter > rootBefore, "should have created a new bin");
        // Find the bin
        var found = false;
        for (var i = 0; i < app.project.rootItem.children.numItems; i++) {
            if (app.project.rootItem.children[i] &&
                app.project.rootItem.children[i].name === "OpenCut Captions") {
                found = true;
                break;
            }
        }
        assert(found, "OpenCut Captions bin should exist");
    });
});

describe("_findByNodeId (helper)", function () {

    it("should find a top-level item by nodeId", function () {
        app = buildMockApp();
        var found = _findByNodeId(app.project.rootItem, "node-002", 0);
        assert(found, "should find node-002");
        assertEqual(found.name, "Broll_001.mp4");
    });

    it("should return null for non-existent nodeId", function () {
        app = buildMockApp();
        var found = _findByNodeId(app.project.rootItem, "does-not-exist", 0);
        assertEqual(found, null);
    });

    it("should find items inside nested bins", function () {
        // Put a clip inside a bin
        var nestedClip = new ProjectItem({ name: "NestedClip.mp4", nodeId: "nested-001", type: 1 });
        var subBin = new ProjectItem({ name: "SubBin", nodeId: "sub-bin", type: 2, children: _makeItemCollection([nestedClip]) });
        app = buildMockApp({ rootChildren: [subBin] });
        var found = _findByNodeId(app.project.rootItem, "nested-001", 0);
        assert(found, "should find nested item");
        assertEqual(found.name, "NestedClip.mp4");
    });
});

describe("_findOrCreateBin (helper)", function () {

    it("should return existing bin if name matches", function () {
        app = buildMockApp();
        // "Footage" bin already exists in default mock
        var bin = _findOrCreateBin("Footage");
        assert(bin, "should find existing bin");
        assertEqual(bin.name, "Footage");
        assertEqual(bin.nodeId, "node-bin-001");
    });

    it("should create a new bin if none exists", function () {
        app = buildMockApp();
        var countBefore = app.project.rootItem.children.numItems;
        var bin = _findOrCreateBin("NewBin");
        assert(bin, "should create bin");
        assertEqual(bin.name, "NewBin");
        assertEqual(app.project.rootItem.children.numItems, countBefore + 1);
    });
});

describe("ocPing", function () {
    it("should return 'pong'", function () {
        assertEqual(ocPing(), "pong");
    });
});

describe("Edge cases and robustness", function () {

    it("ocApplySequenceCuts coerces string numbers in cuts", function () {
        app = buildMockApp();
        var cuts = [{ start: "0", end: "10" }]; // strings, not numbers
        var result = JSON.parse(ocApplySequenceCuts(JSON.stringify(cuts)));
        assertEqual(result.applied, 2, "string numbers should be coerced");
    });

    it("ocAddSequenceMarkers defaults name to empty string when missing", function () {
        app = buildMockApp();
        var markers = [{ time: 5 }]; // no name field
        var result = JSON.parse(ocAddSequenceMarkers(JSON.stringify(markers)));
        assertEqual(result.added, 1);
        assertEqual(app.project.activeSequence.markers._markers[0].name, "");
    });

    it("ocBatchRenameProjectItems handles mixed valid and invalid entries", function () {
        app = buildMockApp();
        var renames = [
            { nodeId: "node-001", newName: "Renamed.mp4" },       // valid
            { nodeId: "missing", newName: "Ghost.mp4" },          // nodeId not found
            { nodeId: "node-002", newName: "" },                   // empty newName
        ];
        var result = JSON.parse(ocBatchRenameProjectItems(JSON.stringify(renames)));
        assertEqual(result.renamed, 1, "only 1 valid rename");
        assertEqual(result.errors.length, 2, "2 errors");
    });
});

// ---------------------------------------------------------------------------
// 7. Report
// ---------------------------------------------------------------------------

console.log("\n===================================");
console.log("  Results: " + _passed + " passed, " + _failed + " failed, " + (_passed + _failed) + " total");
console.log("===================================");

if (_errors.length > 0) {
    console.log("\nFailures:\n");
    for (var i = 0; i < _errors.length; i++) {
        console.log("  " + (i + 1) + ") " + _errors[i].test);
        console.log("     " + (_errors[i].error.message || _errors[i].error));
    }
}

process.exit(_failed > 0 ? 1 : 0);
