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


/**
 * Get project media with folder path - wrapper for panel use.
 * Returns JSON string: {media: [...], projectFolder: "..."}
 */
function getProjectMedia() {
    var items = [];
    var projectFolder = "";
    try {
        if (!app || !app.project) {
            return JSON.stringify({ error: "No project open", media: [], projectFolder: "" });
        }
        var root = app.project.rootItem;
        if (!root) {
            return JSON.stringify({ error: "Cannot access project root", media: [], projectFolder: "" });
        }
        
        // Get project folder path
        try {
            var projPath = app.project.path;
            if (projPath) {
                var projFile = new File(projPath);
                projectFolder = projFile.parent ? projFile.parent.fsName : "";
            }
        } catch (e2) {
            _ocLog("getProjectMedia projectFolder error: " + e2.toString());
        }
        
        _walkProjectItems(root, items, 0);
    } catch (e) {
        _ocLog("getProjectMedia error: " + e.toString());
        return JSON.stringify({ error: "ExtendScript: " + e.toString(), media: [], projectFolder: "" });
    }
    return JSON.stringify({ media: items, projectFolder: projectFolder });
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
 * Alias for getSelectedClips - used by the panel for timeline selection.
 * Returns the first selected clip's path and name.
 */
function getTimelineSelection() {
    var result = getSelectedClips();
    try {
        var parsed = JSON.parse(result);
        if (parsed.error) {
            return "null";
        }
        if (parsed.length > 0) {
            return JSON.stringify({ path: parsed[0].path, name: parsed[0].name });
        }
    } catch (e) {
        _ocLog("getTimelineSelection error: " + e.toString());
    }
    return "null";
}


/**
 * Import XML file into project (for FCP XML edit lists).
 */
function importXMLToProject(xmlPath) {
    _ocLog("importXMLToProject: " + xmlPath);
    try {
        var xmlFile = new File(xmlPath);
        if (!xmlFile.exists) {
            return JSON.stringify({ error: "XML file not found: " + xmlPath });
        }
        
        // Track how many sequences we had before import
        var seqCountBefore = 0;
        try {
            seqCountBefore = app.project.sequences.numSequences;
        } catch (e) {}
        
        // Import the XML file into the project root
        // Premiere Pro will create a sequence from FCP XML
        // importFiles() returns undefined in many Premiere versions,
        // so we don't rely on the return value.  If it throws, we catch it.
        app.project.importFiles([xmlPath], false, app.project.rootItem, false);
        
        // Wait a moment for import to process
        $.sleep(500);
        
        // Try to find and open the newly created sequence
        var seqCountAfter = 0;
        try {
            seqCountAfter = app.project.sequences.numSequences;
        } catch (e) {}
        
        _ocLog("Sequences before: " + seqCountBefore + ", after: " + seqCountAfter);
        
        // If a new sequence was created, open it
        if (seqCountAfter > seqCountBefore) {
            try {
                // Get the last sequence (most recently added)
                var newSeq = app.project.sequences[seqCountAfter - 1];
                if (newSeq) {
                    app.project.openSequence(newSeq.sequenceID);
                    _ocLog("Opened sequence: " + newSeq.name);
                    return JSON.stringify({ 
                        success: true, 
                        message: "Imported and opened sequence: " + newSeq.name,
                        sequence_name: newSeq.name
                    });
                }
            } catch (e2) {
                _ocLog("Could not open new sequence: " + e2.toString());
            }
        }
        
        // Fallback: Try to find sequence by searching project items
        try {
            var root = app.project.rootItem;
            for (var i = root.children.numItems - 1; i >= 0; i--) {
                var item = root.children[i];
                try {
                    // type 1 = sequence
                    if (item.type === 1) {
                        // Check if it's an OpenCut sequence by name
                        if (item.name && item.name.indexOf("OpenCut") >= 0) {
                            // Open this sequence
                            app.project.openSequence(item.sequenceID);
                            _ocLog("Found and opened: " + item.name);
                            return JSON.stringify({
                                success: true,
                                message: "Opened sequence: " + item.name,
                                sequence_name: item.name
                            });
                        }
                    }
                } catch (e3) {}
            }
        } catch (e4) {
            _ocLog("Search error: " + e4.toString());
        }
        
        return JSON.stringify({ 
            success: true, 
            message: "XML imported. Look for the new sequence in Project panel." 
        });
    } catch (e) {
        _ocLog("importXMLToProject error: " + e.toString());
        return JSON.stringify({ error: "Import failed: " + e.toString() });
    }
}


/**
 * Import overlay video into project (alias for importCaptionOverlay).
 */
function importOverlayToProject(overlayPath) {
    return importCaptionOverlay(overlayPath);
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

        for (var a = 0; a < seq.audioTracks.numTracks; a++) {
            var aTrack = seq.audioTracks[a];
            for (var ac = 0; ac < aTrack.clips.numItems; ac++) {
                var aClip = aTrack.clips[ac];
                var aPath = "";
                try { aPath = aClip.projectItem.getMediaPath(); } catch (e2) { _ocLog(e2.toString()); }
                clips.push({
                    name: aClip.name || "",
                    path: aPath,
                    inPoint: aClip.start ? aClip.start.seconds : 0,
                    outPoint: aClip.end ? aClip.end.seconds : 0,
                    trackIndex: a,
                    trackType: "audio"
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

        var seqCountBefore = 0;
        try { seqCountBefore = app.project.sequences.numSequences; } catch (e) {}

        // importFiles() returns undefined in many Premiere versions
        app.project.importFiles(
            [filePath], true, app.project.rootItem, false
        );

        var seqCountAfter = 0;
        try { seqCountAfter = app.project.sequences.numSequences; } catch (e) {}
        if (seqCountAfter > seqCountBefore) {
            var newSeq = app.project.sequences[seqCountAfter - 1];
            if (newSeq) {
                app.project.openSequence(newSeq.sequenceID);
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
                if (settings && settings.videoFrameRate && settings.videoFrameRate.seconds > 0) {
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
            if (f.parent) {
                return f.parent.fsName;
            }
        }
    } catch (e) { _ocLog(e.toString()); }
    return "";
}


/**
 * Check if the current project has been saved (has a file path).
 * Returns JSON: {saved: true/false, path: "..."}
 */
function isProjectSaved() {
    try {
        var p = app.project.path;
        if (p && p.length > 0) {
            return '{"saved":true,"path":"' + p.replace(/\\/g, "\\\\").replace(/"/g, '\\"').replace(/\n/g, "\\n").replace(/\r/g, "\\r") + '"}';
        }
        return '{"saved":false,"path":""}';
    } catch (e) {
        _ocLog("isProjectSaved error: " + e.toString());
        return '{"saved":false,"path":""}';
    }
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
    var TICKS_PER_SECOND = 254016000000; // Premiere Pro internal time base

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
        var t = new Time();
        t.ticks = String(Math.round(timelinePos * TICKS_PER_SECOND));

        // Insert the clip at this position
        try {
            videoTrack.insertClip(projectItem, t);
            insertedCount++;
            timelinePos += segDuration;
        } catch (e) {
            // Try overwriteClip as fallback
            try {
                videoTrack.overwriteClip(projectItem, t);
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
    if (depth > 50) return null;
    var numChildren = 0;
    try { numChildren = parent.children.numItems; } catch (e) { return null; }
    for (var i = 0; i < numChildren; i++) {
        var item = parent.children[i];
        try {
            if (item.type === 2) {
                // Bin -- recurse into it
                var found = _findProjectItemByPath(item, targetPath, depth + 1);
                if (found) return found;
            } else {
                var p = "";
                try { p = item.getMediaPath(); } catch (e) { _ocLog(e.toString()); }
                if (p) {
                    var dp, dt;
                    try { dp = decodeURI(p); } catch (e) { dp = p; }
                    try { dt = decodeURI(targetPath); } catch (e) { dt = targetPath; }
                    if (dp.toLowerCase() === dt.toLowerCase()) return item;
                }
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

            // importFiles() returns undefined in many Premiere versions —
            // check for the item in the project instead of relying on return value
            app.project.importFiles(
                [captionPath], false, targetBin, false
            );

            // Find the newly imported item
            captionItem = _findProjectItemByPath(app.project.rootItem, captionPath, 0);
            importSuccess = !!captionItem;
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
                        captionTrack.insertClip(captionItem, new Time());
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
                    vTrack.insertClip(captionItem, new Time());
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
 * Generic file import - imports any media file into the project.
 * Used for denoised audio, normalized audio, stems, watermark-removed video, etc.
 *
 * @param {string} filePath  - Full path to the file to import
 * @param {string} binName   - Optional bin name to organize imports (default: "OpenCut Output")
 * @returns {string} JSON: {success, message, name} or {error}
 */
function importFileToProject(filePath, binName) {
    _ocLog("importFileToProject: " + filePath + " -> bin: " + binName);

    if (!binName) binName = "OpenCut Output";

    var f = new File(filePath);
    if (!f.exists) {
        return JSON.stringify({ error: "File not found: " + filePath });
    }

    // Check if already imported
    var existing = _findProjectItemByPath(app.project.rootItem, filePath, 0);
    if (existing) {
        return JSON.stringify({
            success: true,
            message: existing.name + " already in project.",
            name: existing.name
        });
    }

    var targetBin = _findOrCreateBin(binName);
    if (!targetBin) targetBin = app.project.rootItem;

    try {
        app.project.importFiles([filePath], false, targetBin, false);
    } catch (e) {
        return JSON.stringify({ error: "Import failed: " + e.toString() });
    }

    // Poll for the imported item instead of fixed sleep
    var imported = null;
    for (var attempt = 0; attempt < 20; attempt++) {
        $.sleep(50);
        imported = _findProjectItemByPath(app.project.rootItem, filePath, 0);
        if (imported) break;
    }
    var displayName = f.displayName || filePath.split(/[\/\\]/).pop();

    if (imported) {
        return JSON.stringify({
            success: true,
            message: "Imported: " + imported.name,
            name: imported.name
        });
    }

    return JSON.stringify({
        success: true,
        message: "Imported: " + displayName,
        name: displayName
    });
}


/**
 * Import multiple files into the project in a single batch.
 *
 * @param {string} filePathsJson - JSON array of file paths
 * @param {string} binName       - Bin name (default: "OpenCut Output")
 * @returns {string} JSON: {success, message, imported, failed}
 */
function importFilesToProject(filePathsJson, binName) {
    _ocLog("importFilesToProject: " + filePathsJson);

    if (!binName) binName = "OpenCut Output";

    var paths = [];
    try { paths = JSON.parse(filePathsJson); } catch (e) {
        return JSON.stringify({ error: "Invalid JSON: " + e.toString() });
    }

    if (!paths || paths.length === 0) {
        return JSON.stringify({ error: "No files to import" });
    }

    var validPaths = [];
    var failed = [];
    for (var i = 0; i < paths.length; i++) {
        var f = new File(paths[i]);
        if (f.exists) {
            validPaths.push(paths[i]);
        } else {
            failed.push(paths[i]);
        }
    }

    if (validPaths.length === 0) {
        return JSON.stringify({ error: "No valid files found" });
    }

    var targetBin = _findOrCreateBin(binName);
    if (!targetBin) targetBin = app.project.rootItem;

    var itemsBefore = 0;
    try { itemsBefore = targetBin.children.numItems; } catch (e) { /* ignore */ }

    try {
        app.project.importFiles(validPaths, false, targetBin, false);
    } catch (e) {
        return JSON.stringify({ error: "Batch import failed: " + e.toString() });
    }

    var itemsAfter = 0;
    try { itemsAfter = targetBin.children.numItems; } catch (e) { /* ignore */ }
    var actualImported = itemsAfter - itemsBefore;

    return JSON.stringify({
        success: actualImported > 0,
        message: "Imported " + actualImported + " of " + validPaths.length + " file(s) to " + binName,
        imported: actualImported,
        requested: validPaths.length,
        failed: failed.length
    });
}


/**
 * Import a styled caption overlay video (.mov with alpha) into the
 * project panel so the user can drag it onto V2+ above their video.
 *
 * @param {string} overlayPath - Full path to the transparent .mov file
 * @returns {string} JSON: {success, message} or {error}
 */
function importCaptionOverlay(overlayPath) {
    _ocLog("importCaptionOverlay: " + overlayPath);

    var overlayFile = new File(overlayPath);
    if (!overlayFile.exists) {
        return JSON.stringify({ error: "Overlay file not found: " + overlayPath });
    }

    // Check if already imported
    var overlayItem = _findProjectItemByPath(app.project.rootItem, overlayPath, 0);
    if (overlayItem) {
        return JSON.stringify({
            success: true,
            message: "Caption overlay ready in project panel (OpenCut Overlays bin). Drag it onto V2 above your video."
        });
    }

    // Import into an OpenCut Overlays bin
    var overlayBin = _findOrCreateBin("OpenCut Overlays");
    var targetBin = overlayBin || app.project.rootItem;
    try {
        // importFiles returns undefined in many Premiere versions, so
        // we don't rely on the return value. If it throws, we catch it.
        app.project.importFiles([overlayPath], false, targetBin, false);
    } catch (e) {
        return JSON.stringify({ error: "Import failed: " + e.toString() });
    }

    // Verify it actually imported by searching for it
    overlayItem = _findProjectItemByPath(app.project.rootItem, overlayPath, 0);
    if (overlayItem) {
        return JSON.stringify({
            success: true,
            message: "Caption overlay imported! Find it in the OpenCut Overlays bin and drag it onto V2 above your video."
        });
    }

    // If we can't find it by path, check the bin for the most recent item
    if (overlayBin) {
        try {
            var numItems = overlayBin.children.numItems;
            if (numItems > 0) {
                return JSON.stringify({
                    success: true,
                    message: "Caption overlay imported to OpenCut Overlays bin. Drag it onto V2 above your video."
                });
            }
        } catch (e2) {}
    }

    return JSON.stringify({
        error: "Import may have failed. Check the OpenCut Overlays bin, or try File > Import and select: " + overlayPath
    });
}


/**
 * Attempt to start the OpenCut backend server from Premiere.
 * Priority order:
 *   1. Installed exe (from OpenCut installer) via registry path
 *   2. Exe in known install location (%LOCALAPPDATA%\OpenCut\)
 *   3. Fall back to python -m opencut.server (dev mode)
 *
 * Kills any existing server first via PID file + port kill.
 *
 * @returns {string} JSON: {success, message} or {error}
 */
function startOpenCutBackend() {
    _ocLog("startOpenCutBackend called");

    var isWindows = ($.os.indexOf("Windows") !== -1);

    // --- Try to find the installed exe ---
    var exePath = "";

    if (isWindows) {
        // Check registry for install path (set by installer)
        try {
            var wsh = new ActiveXObject("WScript.Shell");
            var regPath = wsh.RegRead("HKCU\\Software\\OpenCut\\InstallPath");
            if (regPath) {
                // Custom installer puts exe in server\ subdir
                var candidate = regPath + "\\server\\OpenCut-Server.exe";
                if (new File(candidate).exists) {
                    exePath = candidate;
                    _ocLog("Found exe via registry: " + exePath);
                }
                // Fallback: exe in install root (legacy layout)
                if (!exePath) {
                    candidate = regPath + "\\OpenCut-Server.exe";
                    if (new File(candidate).exists) {
                        exePath = candidate;
                        _ocLog("Found exe via registry (root): " + exePath);
                    }
                }
            }
        } catch (e) {
            _ocLog("Registry lookup failed (normal if not installed): " + e.toString());
        }

        // Check default install location
        if (!exePath) {
            try {
                var progFiles = $.getenv("ProgramFiles");
                if (progFiles) {
                    var candidate2 = progFiles + "\\OpenCut\\server\\OpenCut-Server.exe";
                    if (new File(candidate2).exists) {
                        exePath = candidate2;
                        _ocLog("Found exe at Program Files: " + exePath);
                    }
                }
            } catch (e2) {}
        }
    }

    var launched = false;

    if (isWindows) {
        // ---- WINDOWS ----
        // Build a batch file that kills old server, then launches new
        var bat = new File(Folder.temp.fsName + "/opencut_start.bat");
        bat.open("w");
        try {
            bat.writeln("@echo off");
            bat.writeln("setlocal");
            // Kill via PID file
            bat.writeln('set "PIDFILE=%USERPROFILE%\\.opencut\\server.pid"');
            bat.writeln('if exist "%PIDFILE%" (');
            bat.writeln('    set /p OLDPID=<"%PIDFILE%"');
            bat.writeln('    if defined OLDPID (');
            bat.writeln('        taskkill /F /T /PID %OLDPID% >nul 2>&1');
            bat.writeln('    )');
            bat.writeln('    del "%PIDFILE%" >nul 2>&1');
            bat.writeln(')');
            // Kill anything holding port 5679
            bat.writeln('for /f "tokens=5" %%a in (\'netstat -ano -p TCP ^| findstr ":5679 " ^| findstr "LISTENING"\') do (');
            bat.writeln('    taskkill /F /T /PID %%a >nul 2>&1');
            bat.writeln(')');
            bat.writeln("timeout /t 1 /nobreak >nul 2>&1");

            if (exePath) {
                // Launch the installed exe
                bat.writeln('"' + exePath + '"');
            } else {
                // Fall back to python -m (dev mode)
                var pythonCmds = ["python", "python3", "py"];
                for (var i = 0; i < pythonCmds.length; i++) {
                    bat.writeln(pythonCmds[i] + ' -m opencut.server 2>nul && goto :done');
                }
                bat.writeln(":done");
            }
        } finally {
            bat.close();
        }

        try {
            bat.execute();
            launched = true;
            _ocLog("Launched via bat" + (exePath ? " (exe)" : " (python fallback)"));
        } catch (e3) {
            _ocLog("Bat launch failed: " + e3.toString());
        }
    } else {
        // ---- macOS / Linux ----
        // Write the startup script
        var shPath = Folder.temp.fsName + "/opencut_start.sh";
        var sh = new File(shPath);
        sh.open("w");
        try {
            sh.writeln("#!/bin/bash");
            // Kill via PID file
            sh.writeln('PIDFILE="$HOME/.opencut/server.pid"');
            sh.writeln('if [ -f "$PIDFILE" ]; then');
            sh.writeln('    OLDPID=$(head -1 "$PIDFILE" 2>/dev/null)');
            sh.writeln('    [ -n "$OLDPID" ] && kill -9 "$OLDPID" 2>/dev/null');
            sh.writeln('    rm -f "$PIDFILE"');
            sh.writeln('fi');
            // Kill anything on port 5679
            sh.writeln('for PID in $(lsof -ti :5679 2>/dev/null); do kill -9 "$PID" 2>/dev/null; done');
            sh.writeln("sleep 1");
            // Launch
            sh.writeln("nohup python3 -m opencut.server > /dev/null 2>&1 &");
        } finally {
            sh.close();
        }

        try {
            // File.execute() on macOS opens .sh files in a text editor.
            // Use system.callSystem (or app.system) to run bash directly.
            var cmd = '/bin/bash -c \'chmod +x "' + shPath + '" && "' + shPath + '"\'';
            if (typeof app.system === "function") {
                app.system(cmd);
                launched = true;
            } else {
                // Fallback: use a .command file which Terminal.app can run
                var cmdFile = new File(Folder.temp.fsName + "/opencut_start.command");
                cmdFile.open("w");
                try {
                    cmdFile.writeln("#!/bin/bash");
                    cmdFile.writeln('chmod +x "' + shPath + '" && "' + shPath + '"');
                    cmdFile.writeln("exit 0");
                } finally {
                    cmdFile.close();
                }
                cmdFile.execute();
                launched = true;
            }
            _ocLog("Launched via sh (python)");
        } catch (e4) {
            _ocLog("Sh launch failed: " + e4.toString());
        }
    }

    if (launched) {
        return JSON.stringify({ success: true, message: "Backend launch attempted" + (exePath ? " (installed)" : " (dev)") });
    }
    return JSON.stringify({ error: "Could not launch backend. Start manually: python -m opencut.server" });
}


/**
 * Get Premiere Pro's current UI brightness for theme syncing.
 * Returns JSON with brightness level (0-255 range).
 *
 * @returns {string} JSON: {brightness: number, isDark: boolean}
 */
function getPremiereThemeInfo() {
    try {
        // CSInterface handles this on the panel side, but for ExtendScript
        // we can check the app display name / version for context
        var info = {
            appName: app.name || "Premiere Pro",
            appVersion: app.version || "unknown",
            projectName: "",
            projectPath: ""
        };
        if (app.project) {
            info.projectName = app.project.name || "";
            info.projectPath = app.project.path || "";
        }
        return JSON.stringify(info);
    } catch (e) {
        return JSON.stringify({ error: e.toString() });
    }
}


/**
 * Auto-import any output file by detecting its type and placing it
 * in the appropriate project bin. Universal import handler.
 *
 * @param {string} filePath  - Full path to the output file
 * @param {string} jobType   - The type of job that produced this file (e.g., "denoise", "export-preset")
 * @returns {string} JSON result
 */
function autoImportResult(filePath, jobType) {
    _ocLog("autoImportResult: " + filePath + " (type: " + jobType + ")");

    if (!filePath) {
        return JSON.stringify({ error: "No file path provided" });
    }

    var f = new File(filePath);
    if (!f.exists) {
        return JSON.stringify({ error: "File not found: " + filePath });
    }

    // Determine the bin name based on job type
    var binName = "OpenCut Output";
    if (jobType) {
        var t = jobType.toLowerCase();
        if (t.indexOf("caption") !== -1 || t.indexOf("subtitle") !== -1 || t.indexOf("srt") !== -1) {
            binName = "OpenCut Captions";
        } else if (t.indexOf("audio") !== -1 || t.indexOf("denoise") !== -1 || t.indexOf("normalize") !== -1 || t.indexOf("stem") !== -1 || t.indexOf("tts") !== -1 || t.indexOf("sfx") !== -1 || t.indexOf("music") !== -1) {
            binName = "OpenCut Audio";
        } else if (t.indexOf("export") !== -1) {
            binName = "OpenCut Exports";
        } else if (t.indexOf("thumbnail") !== -1) {
            binName = "OpenCut Thumbnails";
        }
    }

    // Check if already imported
    var existing = _findProjectItemByPath(app.project.rootItem, filePath, 0);
    if (existing) {
        return JSON.stringify({ success: true, message: "Already in project (" + binName + ")" });
    }

    var targetBin = _findOrCreateBin(binName);
    if (!targetBin) targetBin = app.project.rootItem;

    try {
        app.project.importFiles([filePath], false, targetBin, false);
    } catch (e) {
        return JSON.stringify({ error: "Import failed: " + e.toString() });
    }

    return JSON.stringify({
        success: true,
        message: "Imported to " + binName
    });
}
