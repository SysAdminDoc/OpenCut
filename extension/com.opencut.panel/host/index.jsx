/*
 * OpenCut ExtendScript Host
 * Runs inside Premiere Pro's ExtendScript engine.
 *
 * IMPORTANT: ExtendScript is ES3 -- no let/const, no arrow functions,
 * no template literals, no default params. Use var everywhere.
 */


/**
 * Log errors to the ExtendScript console for debugging.
 */
function _ocLog(msg) {
    try { $.writeln("[OpenCut] " + msg); } catch (e) {}
}


/**
 * Simple ping to verify ExtendScript host is loaded and running.
 * Returns "pong" if everything is OK.
 */
function ocPing() {
    return "pong";
}


/**
 * Get all media items in the project as a JSON array.
 * This is the primary way the panel discovers available clips.
 * Returns JSON string: [{name, path, duration, type, nodeId}, ...]
 */
function getAllProjectMedia() {
    var items = [];
    try {
        if (!app || !app.project) {
            return JSON.stringify({ error: "No project open" });
        }
        var root = app.project.rootItem;
        if (!root) {
            return JSON.stringify({ error: "Cannot access project root" });
        }
        _walkProjectItems(root, items, 0);
    } catch (e) {
        _ocLog("getAllProjectMedia error: " + e.toString());
        return JSON.stringify({ error: "ExtendScript: " + e.toString() });
    }
    return JSON.stringify(items);
}

function _walkProjectItems(parent, items, depth) {
    if (depth > 10) return;
    var numChildren = 0;
    try { numChildren = parent.children.numItems; } catch (e) { return; }

    for (var i = 0; i < numChildren; i++) {
        var item = null;
        try { item = parent.children[i]; } catch (e) { continue; }
        if (!item) continue;

        try {
            // Check if this is a bin/folder - type 2 or has children
            var isBin = false;
            try { isBin = (item.type === 2); } catch (e2) {}
            if (!isBin) {
                try { isBin = (item.children && item.children.numItems > 0 && !item.getMediaPath); } catch (e3) {}
            }

            if (isBin) {
                _walkProjectItems(item, items, depth + 1);
            } else {
                var mediaPath = "";
                try { mediaPath = item.getMediaPath(); } catch (e4) { mediaPath = ""; }

                if (mediaPath && mediaPath.length > 0) {
                    var dur = 0;
                    try {
                        var outPt = item.getOutPoint();
                        dur = outPt ? outPt.seconds : 0;
                    } catch (e5) { dur = 0; }

                    var hasVideo = false;
                    var hasAudio = false;
                    try { hasVideo = item.hasVideo(); } catch (e6) {}
                    try { hasAudio = item.hasAudio(); } catch (e7) {}

                    var mediaType = "unknown";
                    if (hasVideo && hasAudio) mediaType = "av";
                    else if (hasVideo) mediaType = "video";
                    else if (hasAudio) mediaType = "audio";

                    items.push({
                        name: item.name || "",
                        path: mediaPath,
                        duration: dur,
                        type: mediaType,
                        nodeId: item.nodeId || ""
                    });
                }
            }
        } catch (e) { _ocLog("walkItem[" + i + "] error: " + e.toString()); }
    }
}


/**
 * Get selected clips from timeline or project panel.
 * Tries multiple methods for maximum compatibility.
 * Returns JSON string: [{name, path}] or {error: "..."}
 */
function getSelectedClips() {
    var results = [];

    // Method 1: Timeline selection
    try {
        var seq = app.project.activeSequence;
        if (seq) {
            for (var t = 0; t < seq.videoTracks.numTracks; t++) {
                var track = seq.videoTracks[t];
                for (var c = 0; c < track.clips.numItems; c++) {
                    var clip = track.clips[c];
                    var selected = false;
                    try { selected = clip.isSelected(); } catch (e) { _ocLog(e.toString()); }
                    if (selected) {
                        try {
                            var pi = clip.projectItem;
                            if (pi) {
                                var p = pi.getMediaPath();
                                if (p) results.push({ name: clip.name || pi.name || "", path: p });
                            }
                        } catch (e2) {}
                    }
                }
            }
            for (var t2 = 0; t2 < seq.audioTracks.numTracks; t2++) {
                var atrack = seq.audioTracks[t2];
                for (var c2 = 0; c2 < atrack.clips.numItems; c2++) {
                    var aclip = atrack.clips[c2];
                    var asel = false;
                    try { asel = aclip.isSelected(); } catch (e3) {}
                    if (asel) {
                        try {
                            var api = aclip.projectItem;
                            if (api) {
                                var ap = api.getMediaPath();
                                if (ap) {
                                    var dup = false;
                                    for (var d = 0; d < results.length; d++) {
                                        if (results[d].path === ap) { dup = true; break; }
                                    }
                                    if (!dup) results.push({ name: aclip.name || api.name || "", path: ap });
                                }
                            }
                        } catch (e4) {}
                    }
                }
            }
        }
    } catch (e5) {}

    // Method 2: Project panel selection
    if (results.length === 0) {
        try {
            var viewIDs = app.getProjectViewIDs ? app.getProjectViewIDs() : null;
            if (viewIDs && viewIDs.length > 0) {
                var selItems = app.getProjectViewSelection(viewIDs[0]);
                if (selItems && selItems.length > 0) {
                    for (var s = 0; s < selItems.length; s++) {
                        try {
                            var sp = selItems[s].getMediaPath();
                            if (sp) results.push({ name: selItems[s].name || "", path: sp });
                        } catch (e6) {}
                    }
                }
            }
        } catch (e7) {}
    }

    // Method 3: Source monitor
    if (results.length === 0) {
        try {
            var srcMon = app.sourceMonitor;
            if (srcMon && srcMon.projectItem) {
                var smp = srcMon.projectItem.getMediaPath();
                if (smp) results.push({ name: srcMon.projectItem.name || "", path: smp });
            }
        } catch (e8) {}
    }

    if (results.length === 0) {
        return JSON.stringify({ error: "nothing_selected" });
    }
    return JSON.stringify(results);
}


/**
 * Get all clips in the active sequence.
 */
function getSequenceClips() {
    var clips = [];
    try {
        var seq = app.project.activeSequence;
        if (!seq) return JSON.stringify({ error: "No active sequence" });

        for (var t = 0; t < seq.videoTracks.numTracks; t++) {
            var track = seq.videoTracks[t];
            for (var c = 0; c < track.clips.numItems; c++) {
                var clip = track.clips[c];
                var path = "";
                try { path = clip.projectItem.getMediaPath(); } catch (e) { _ocLog(e.toString()); }
                clips.push({
                    name: clip.name || "",
                    path: path,
                    inPoint: clip.start ? clip.start.seconds : 0,
                    outPoint: clip.end ? clip.end.seconds : 0,
                    trackIndex: t,
                    trackType: "video"
                });
            }
        }
    } catch (e3) {
        return JSON.stringify({ error: e3.toString() });
    }
    return JSON.stringify(clips);
}


/**
 * Browse for a media file.
 */
function browseForFile() {
    var filter = "Media Files:*.mp4;*.mov;*.avi;*.mkv;*.wmv;*.webm;*.mxf;*.wav;*.mp3;*.aac;*.m4a;*.flac,All Files:*.*";
    var file = File.openDialog("Select Media File", filter, false);
    if (file) return file.fsName;
    return "null";
}


/**
 * Import a file and open it as a sequence.
 */
function importAndOpenXml(filePath) {
    try {
        var file = new File(filePath);
        if (!file.exists) return JSON.stringify({ error: "File not found: " + filePath });

        var seqCountBefore = app.project.sequences.numSequences;

        var success = app.project.importFiles(
            [filePath], true, app.project.rootItem, false
        );

        if (!success) return JSON.stringify({ error: "Import returned false" });

        var seqCountAfter = app.project.sequences.numSequences;
        if (seqCountAfter > seqCountBefore) {
            var newSeq = app.project.sequences[seqCountAfter - 1];
            if (newSeq) {
                app.project.activeSequence = newSeq;
                return JSON.stringify({ success: true, sequenceName: newSeq.name });
            }
        }
        return JSON.stringify({ success: true, sequenceName: "" });
    } catch (e) {
        return JSON.stringify({ error: e.toString() });
    }
}


/**
 * Get project info.
 */
function getProjectInfo() {
    var info = {
        projectName: "",
        projectPath: "",
        sequenceName: "",
        sequenceFrameRate: 0,
        numSequences: 0
    };
    try {
        info.projectName = app.project.name || "";
        info.projectPath = app.project.path || "";
        info.numSequences = app.project.sequences.numSequences;
        var seq = app.project.activeSequence;
        if (seq) {
            info.sequenceName = seq.name || "";
            try {
                var settings = seq.getSettings();
                if (settings && settings.videoFrameRate) {
                    info.sequenceFrameRate = 1 / settings.videoFrameRate.seconds;
                }
            } catch (e) { _ocLog(e.toString()); }
        }
    } catch (e2) {}
    return JSON.stringify(info);
}


/**
 * Get the project folder path.
 */
function getProjectFolder() {
    try {
        var p = app.project.path;
        if (p) {
            var f = new File(p);
            return f.parent.fsName;
        }
    } catch (e) { _ocLog(e.toString()); }
    return "";
}


/**
 * Apply silence-removed edits directly to the timeline.
 *
 * Creates a new sequence and inserts the clip multiple times,
 * once per speech segment, with proper in/out points.
 * This bypasses XML import entirely -- no Locate Media issues.
 *
 * @param {string} segmentsJson - JSON array of {start, end} objects (seconds)
 * @param {string} mediaPath    - Full path to the source media file
 * @returns {string} JSON result: {success, sequenceName, segments} or {error}
 */
function applyEditsToTimeline(segmentsJson, mediaPath) {
    var TICKS_PER_SECOND = 254016000000;

    var segments;
    try {
        segments = JSON.parse(segmentsJson);
    } catch (e) {
        return JSON.stringify({ error: "Invalid segments data: " + e.toString() });
    }

    if (!segments || segments.length === 0) {
        return JSON.stringify({ error: "No speech segments found" });
    }

    // Find the project item by matching its media path
    var projectItem = _findProjectItemByPath(app.project.rootItem, mediaPath, 0);

    if (!projectItem) {
        return JSON.stringify({ error: "Media not found in project. Import the file first, then try again." });
    }

    // Build sequence name from the clip
    var clipName = projectItem.name || "Edit";
    // Remove file extension for cleaner name
    clipName = clipName.replace(/\.[^.]+$/, "");
    var seqName = "OpenCut - " + clipName;

    // Create a new sequence
    try {
        app.project.createNewSequence(seqName);
    } catch (e) {
        return JSON.stringify({ error: "Could not create sequence: " + e.toString() });
    }

    var seq = app.project.activeSequence;
    if (!seq) {
        return JSON.stringify({ error: "Sequence creation failed" });
    }

    var videoTrack = seq.videoTracks[0];
    if (!videoTrack) {
        return JSON.stringify({ error: "No video track in sequence" });
    }

    // Insert each speech segment as a clip with proper in/out points
    var timelinePos = 0;
    var insertedCount = 0;

    for (var i = 0; i < segments.length; i++) {
        var seg = segments[i];
        var segStart = seg.start;
        var segEnd = seg.end;
        var segDuration = segEnd - segStart;

        if (segDuration <= 0.01) continue;

        // Set the project item's in/out points to this segment
        // This controls what portion of the clip gets inserted
        try {
            projectItem.setInPoint(segStart, 4);  // 4 = all media types
            projectItem.setOutPoint(segEnd, 4);
        } catch (e) {
            // If setInPoint/setOutPoint not available, skip this segment
            continue;
        }

        // Calculate timeline position in ticks
        var ticks = String(Math.round(timelinePos * TICKS_PER_SECOND));

        // Insert the clip at this position
        try {
            videoTrack.insertClip(projectItem, ticks);
            insertedCount++;
            timelinePos += segDuration;
        } catch (e) {
            // Try overwriteClip as fallback
            try {
                videoTrack.overwriteClip(projectItem, ticks);
                insertedCount++;
                timelinePos += segDuration;
            } catch (e2) {
                // Skip this segment
            }
        }
    }

    // Reset the project item's in/out points so it appears normal in the project panel
    try {
        projectItem.clearInPoint(4);
        projectItem.clearOutPoint(4);
    } catch (e) {
        // clearInPoint may not exist in all versions; try setting to extremes
        try {
            projectItem.setInPoint(0, 4);
            // setOutPoint to a very large value to effectively clear it
            projectItem.setOutPoint(86400, 4);  // 24 hours
        } catch (e2) {
            // Best effort -- don't fail the whole operation
        }
    }

    if (insertedCount === 0) {
        return JSON.stringify({ error: "Could not insert any clips. Your Premiere version may not support this method. Use Import XML instead." });
    }

    return JSON.stringify({
        success: true,
        sequenceName: seqName,
        segments: insertedCount,
        duration: timelinePos
    });
}


/**
 * Find a project item by its media file path (recursive search).
 */
function _findProjectItemByPath(parent, targetPath, depth) {
    if (depth > 10) return null;
    for (var i = 0; i < parent.children.numItems; i++) {
        var item = parent.children[i];
        try {
            if (item.type === 2) {
                // Bin -- recurse into it
                var found = _findProjectItemByPath(item, targetPath, depth + 1);
                if (found) return found;
            } else {
                var p = "";
                try { p = item.getMediaPath(); } catch (e) { _ocLog(e.toString()); }
                if (p && p === targetPath) return item;
            }
        } catch (e) { _ocLog(e.toString()); }
    }
    return null;
}


/**
 * Import a caption file (SRT/VTT) into the project and optionally
 * add it to the active sequence's caption track.
 *
 * Premiere Pro 2021+ (v15+) supports native SRT import.
 * Older versions can import but may not have caption tracks.
 *
 * @param {string} captionPath - Full file path to the .srt or .vtt file
 * @returns {string} JSON: {success, imported, addedToTimeline, message} or {error}
 */
function importCaptions(captionPath) {
    _ocLog("importCaptions: " + captionPath);

    // Validate file exists
    var captionFile = new File(captionPath);
    if (!captionFile.exists) {
        return JSON.stringify({ error: "Caption file not found: " + captionPath });
    }

    // Check for active sequence
    var seq = null;
    try { seq = app.project.activeSequence; } catch (e) {}

    // Step 1: Import the caption file into the project
    var importSuccess = false;
    var captionItem = null;
    try {
        // Check if already imported
        captionItem = _findProjectItemByPath(app.project.rootItem, captionPath, 0);

        if (!captionItem) {
            // Find or create an "OpenCut Captions" bin for organization
            var captionBin = _findOrCreateBin("OpenCut Captions");
            var targetBin = captionBin || app.project.rootItem;

            importSuccess = app.project.importFiles(
                [captionPath], false, targetBin, false
            );

            if (importSuccess) {
                // Find the newly imported item
                captionItem = _findProjectItemByPath(app.project.rootItem, captionPath, 0);
            }
        } else {
            importSuccess = true;
            _ocLog("Caption file already imported");
        }
    } catch (e) {
        _ocLog("Import error: " + e.toString());
        return JSON.stringify({ error: "Failed to import caption file: " + e.toString() });
    }

    if (!importSuccess && !captionItem) {
        return JSON.stringify({ error: "Import failed. Your Premiere version may not support SRT import (requires v15+)." });
    }

    // Step 2: Try to add to the active sequence's caption track
    var addedToTimeline = false;
    var timelineMessage = "";

    if (seq && captionItem) {
        // Method 1: Try addCaptionTrack (Premiere 2021+ / v15+)
        try {
            if (seq.captionTracks) {
                // Ensure there's at least one caption track
                var numCaptionTracks = 0;
                try { numCaptionTracks = seq.captionTracks.numTracks; } catch (e) {}

                if (numCaptionTracks === 0) {
                    // Try creating a caption track
                    try { seq.addCaptionTrack(); } catch (e2) {
                        _ocLog("Could not create caption track: " + e2.toString());
                    }
                }

                // Insert caption item into the first caption track
                try {
                    var captionTrack = seq.captionTracks[0];
                    if (captionTrack) {
                        captionTrack.insertClip(captionItem, "0");
                        addedToTimeline = true;
                        timelineMessage = "Captions added to timeline caption track";
                    }
                } catch (e3) {
                    _ocLog("Caption track insert failed: " + e3.toString());
                    timelineMessage = "Imported to project (drag to timeline manually)";
                }
            }
        } catch (e) {
            _ocLog("captionTracks not available: " + e.toString());
        }

        // Method 2: If caption track method failed, try inserting on a video track
        // (older Premiere versions treat SRT as a regular graphic/clip)
        if (!addedToTimeline) {
            try {
                // Insert at the start of the timeline on the topmost empty video track
                var targetTrackIdx = seq.videoTracks.numTracks - 1;
                for (var t = seq.videoTracks.numTracks - 1; t >= 0; t--) {
                    if (seq.videoTracks[t].clips.numItems === 0) {
                        targetTrackIdx = t;
                        break;
                    }
                }
                var vTrack = seq.videoTracks[targetTrackIdx];
                if (vTrack) {
                    vTrack.insertClip(captionItem, "0");
                    addedToTimeline = true;
                    timelineMessage = "Captions added to video track V" + (targetTrackIdx + 1);
                }
            } catch (e) {
                _ocLog("Video track insert also failed: " + e.toString());
                timelineMessage = "Imported to project panel. Drag onto your timeline to use.";
            }
        }
    } else if (!seq) {
        timelineMessage = "No active sequence. Open a sequence and drag the caption file from the project panel.";
    } else {
        timelineMessage = "Caption file imported to project panel.";
    }

    return JSON.stringify({
        success: true,
        imported: true,
        addedToTimeline: addedToTimeline,
        message: timelineMessage
    });
}


/**
 * Find or create a bin (folder) in the project panel.
 */
function _findOrCreateBin(binName) {
    try {
        var root = app.project.rootItem;
        // Search existing bins
        for (var i = 0; i < root.children.numItems; i++) {
            var item = root.children[i];
            try {
                if (item.type === 2 && item.name === binName) {
                    return item;
                }
            } catch (e) {}
        }
        // Create new bin
        return root.createBin(binName);
    } catch (e) {
        _ocLog("_findOrCreateBin error: " + e.toString());
        return null;
    }
}


/**
 * Import a styled caption overlay video (.mov with alpha) and place it
 * on a video track above the existing content in the active sequence.
 *
 * @param {string} overlayPath - Full path to the transparent .mov file
 * @returns {string} JSON: {success, trackIndex, message} or {error}
 */
function importCaptionOverlay(overlayPath) {
    _ocLog("importCaptionOverlay: " + overlayPath);

    var overlayFile = new File(overlayPath);
    if (!overlayFile.exists) {
        return JSON.stringify({ error: "Overlay file not found: " + overlayPath });
    }

    var seq = null;
    try { seq = app.project.activeSequence; } catch (e) {}
    if (!seq) {
        return JSON.stringify({ error: "No active sequence. Open or create a sequence first." });
    }

    // Import the overlay into the project
    var overlayItem = _findProjectItemByPath(app.project.rootItem, overlayPath, 0);
    if (!overlayItem) {
        var captionBin = _findOrCreateBin("OpenCut Captions");
        var targetBin = captionBin || app.project.rootItem;
        try {
            app.project.importFiles([overlayPath], false, targetBin, false);
            overlayItem = _findProjectItemByPath(app.project.rootItem, overlayPath, 0);
        } catch (e) {
            return JSON.stringify({ error: "Import failed: " + e.toString() });
        }
    }

    if (!overlayItem) {
        return JSON.stringify({ error: "Could not import overlay video." });
    }

    // Find the topmost available video track (or create one above existing content)
    var numTracks = seq.videoTracks.numTracks;
    var targetTrack = null;
    var targetIdx = -1;

    // Look for an empty track above track 0
    for (var t = numTracks - 1; t >= 1; t--) {
        if (seq.videoTracks[t].clips.numItems === 0) {
            targetTrack = seq.videoTracks[t];
            targetIdx = t;
            break;
        }
    }

    // If no empty track, use the highest numbered track
    if (!targetTrack) {
        targetIdx = numTracks - 1;
        if (targetIdx < 1) targetIdx = 1;
        // Prefer track above any content
        if (targetIdx < numTracks) {
            targetTrack = seq.videoTracks[targetIdx];
        }
    }

    if (!targetTrack) {
        return JSON.stringify({ error: "No video track available for overlay." });
    }

    // Insert at the beginning of the timeline
    try {
        targetTrack.insertClip(overlayItem, "0");
        return JSON.stringify({
            success: true,
            trackIndex: targetIdx + 1,
            message: "Caption overlay added to V" + (targetIdx + 1)
        });
    } catch (e) {
        // Try overwriteClip as fallback
        try {
            targetTrack.overwriteClip(overlayItem, "0");
            return JSON.stringify({
                success: true,
                trackIndex: targetIdx + 1,
                message: "Caption overlay added to V" + (targetIdx + 1)
            });
        } catch (e2) {
            return JSON.stringify({
                error: "Could not place overlay on timeline: " + e2.toString()
                    + ". Drag it from the project panel manually."
            });
        }
    }
}
