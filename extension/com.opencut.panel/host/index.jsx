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
    var scan = null;
    try {
        if (!app || !app.project) {
            return JSON.stringify({ error: "No project open" });
        }
        var root = app.project.rootItem;
        if (!root) {
            return JSON.stringify({ error: "Cannot access project root" });
        }
        scan = _collectProjectMedia(root, 5, 80);
    } catch (e) {
        _ocLog("getAllProjectMedia error: " + e.toString());
        return JSON.stringify({ error: "ExtendScript: " + e.toString() });
    }
    return JSON.stringify(scan ? scan.items : []);
}


/**
 * Get project media with folder path - wrapper for panel use.
 * Returns JSON string: {media: [...], projectFolder: "..."}
 */
function getProjectMedia() {
    var scan = null;
    var projectFolder = "";
    try {
        if (!app || !app.project) {
            return JSON.stringify({ error: "No project open", media: [], projectFolder: "", rootChildren: 0, scanAttempts: 0 });
        }
        var root = app.project.rootItem;
        if (!root) {
            return JSON.stringify({ error: "Cannot access project root", media: [], projectFolder: "", rootChildren: 0, scanAttempts: 0 });
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
        
        scan = _collectProjectMedia(root, 6, 100);
    } catch (e) {
        _ocLog("getProjectMedia error: " + e.toString());
        return JSON.stringify({ error: "ExtendScript: " + e.toString(), media: [], projectFolder: "", rootChildren: 0, scanAttempts: 0 });
    }
    return JSON.stringify({
        media: scan ? scan.items : [],
        projectFolder: projectFolder,
        rootChildren: scan ? scan.rootChildren : 0,
        scanAttempts: scan ? scan.attempts : 0
    });
}

function _getProjectChildCount(parent) {
    try {
        if (parent && parent.children) return parent.children.numItems || 0;
    } catch (e) {}
    return 0;
}

function _normalizeMediaPath(path) {
    if (!path || typeof path !== "string") return "";
    var normalized = path;
    try { normalized = decodeURI(normalized); } catch (e) {}
    normalized = normalized.replace(/^file:\/*/i, "");
    normalized = normalized.replace(/^([A-Za-z])\|/, "$1:");
    try {
        if (Folder && Folder.fs === "Windows") {
            if (/^\/[A-Za-z]:/.test(normalized)) normalized = normalized.substring(1);
            normalized = normalized.replace(/\//g, "\\");
        }
    } catch (e2) {}
    return normalized;
}

function _collectProjectMedia(root, maxAttempts, delayMs) {
    var attempts = 0;
    var rootChildren = 0;
    var items = [];
    if (!maxAttempts || maxAttempts < 1) maxAttempts = 1;
    if (!delayMs || delayMs < 0) delayMs = 0;

    while (attempts < maxAttempts) {
        attempts++;
        rootChildren = _getProjectChildCount(root);
        items = [];
        _walkProjectItems(root, items, {}, 0);
        if (items.length > 0) break;
        if (rootChildren === 0 || attempts >= maxAttempts) break;
        if (delayMs > 0) $.sleep(delayMs);
    }

    return {
        items: items,
        rootChildren: rootChildren,
        attempts: attempts
    };
}

function _walkProjectItems(parent, items, seenPaths, depth) {
    if (depth > 20) return;
    var numChildren = _getProjectChildCount(parent);
    if (!numChildren) return;

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
                _walkProjectItems(item, items, seenPaths, depth + 1);
            } else {
                var mediaPath = "";
                try { mediaPath = _normalizeMediaPath(item.getMediaPath()); } catch (e4) { mediaPath = ""; }

                if (mediaPath && mediaPath.length > 0) {
                    var dedupeKey = mediaPath.toLowerCase();
                    if (seenPaths && seenPaths[dedupeKey]) continue;
                    if (seenPaths) seenPaths[dedupeKey] = true;

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
                                var p = _normalizeMediaPath(pi.getMediaPath());
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
                                var ap = _normalizeMediaPath(api.getMediaPath());
                                if (ap) {
                                    var dup = false;
                                    for (var d = 0; d < results.length; d++) {
                                        if (results[d].path.toLowerCase() === ap.toLowerCase()) { dup = true; break; }
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
                            var sp = _normalizeMediaPath(selItems[s].getMediaPath());
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
                var smp = _normalizeMediaPath(srcMon.projectItem.getMediaPath());
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

    if (!app || !app.project || !app.project.rootItem) {
        return JSON.stringify({ error: "No project open" });
    }
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
        
        // Poll for import to complete (avoids blocking Premiere UI with long sleep)
        for (var _poll = 0; _poll < 20; _poll++) {
            $.sleep(50);
            try {
                if (app.project.sequences.numSequences > seqCountBefore) break;
            } catch (e) {}
        }
        
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
        if (!app || !app.project) return JSON.stringify({ error: "No project open" });
        var seq = app.project.activeSequence;
        if (!seq) return JSON.stringify({ error: "No active sequence" });

        for (var t = 0; t < seq.videoTracks.numTracks; t++) {
            var track = seq.videoTracks[t];
            for (var c = 0; c < track.clips.numItems; c++) {
                var clip = track.clips[c];
                var path = "";
                try { path = _normalizeMediaPath(clip.projectItem.getMediaPath()); } catch (e) { _ocLog(e.toString()); }
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
                try { aPath = _normalizeMediaPath(aClip.projectItem.getMediaPath()); } catch (e2) { _ocLog(e2.toString()); }
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
    if (!app || !app.project || !app.project.rootItem) {
        return JSON.stringify({ error: "No project open" });
    }
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
    if (!app || !app.project) return JSON.stringify(info);
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
    if (!app || !app.project) return "";
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
    if (!app || !app.project) return '{"saved":false,"path":""}';
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

    if (!app || !app.project || !app.project.rootItem) {
        return JSON.stringify({ error: "No project open" });
    }

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
        var segStart = Number(seg.start);
        var segEnd = Number(seg.end);
        if (isNaN(segStart) || isNaN(segEnd) || segStart < 0) continue;
        var segDuration = segEnd - segStart;

        // Skip zero-length, negative, or near-zero segments
        if (segDuration <= 0.01) continue;

        // Set the project item's in/out points to this segment
        // This controls what portion of the clip gets inserted
        var needsReset = false;
        try {
            projectItem.setInPoint(segStart, 4);  // 4 = all media types
            projectItem.setOutPoint(segEnd, 4);
            needsReset = true;
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

    // Always reset the project item's in/out points (even if loop threw)
    try {
        projectItem.clearInPoint(4);
        projectItem.clearOutPoint(4);
    } catch (e) {
        try {
            projectItem.setInPoint(0, 4);
            projectItem.setOutPoint(86400, 4);
        } catch (e2) {}
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
    // Normalize target once at the top-level call (depth 0), reuse on recursion
    var normalizedTarget = (depth === 0) ? _normalizeMediaPath(targetPath).toLowerCase() : targetPath;
    var numChildren = _getProjectChildCount(parent);
    if (!numChildren) return null;
    for (var i = 0; i < numChildren; i++) {
        try {
            var item = parent.children[i];
            if (!item) continue;
            var isBin = false;
            try { isBin = (item.type === 2); } catch (e2) {}
            if (isBin) {
                // Bin -- recurse with already-normalized target
                var found = _findProjectItemByPath(item, normalizedTarget, depth + 1);
                if (found) return found;
            } else {
                var p = "";
                try { p = _normalizeMediaPath(item.getMediaPath()); } catch (e3) { _ocLog(e3.toString()); }
                if (p && p.toLowerCase() === normalizedTarget) return item;
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

    if (!app || !app.project || !app.project.rootItem) {
        return JSON.stringify({ error: "No project open" });
    }

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

            // Poll for the imported item — Premiere may need time to register it
            var pollAttempts = 0;
            while (!captionItem && pollAttempts < 20) {
                captionItem = _findProjectItemByPath(app.project.rootItem, captionPath, 0);
                if (captionItem) { break; }
                $.sleep(50);
                pollAttempts++;
            }
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
        message: timelineMessage,
        warning: addedToTimeline ? "" : "Captions imported to project panel but not yet on the timeline."
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

    if (!app || !app.project || !app.project.rootItem) {
        return JSON.stringify({ error: "No project open" });
    }

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

    if (!app || !app.project || !app.project.rootItem) {
        return JSON.stringify({ error: "No project open" });
    }

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

    if (!app || !app.project || !app.project.rootItem) {
        return JSON.stringify({ error: "No project open" });
    }

    var overlayFile = new File(overlayPath);
    if (!overlayFile.exists) {
        return JSON.stringify({ error: "Overlay file not found: " + overlayPath });
    }

    // Check if already imported
    try {
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
            app.project.importFiles([overlayPath], false, targetBin, false);
        } catch (e) {
            return JSON.stringify({ error: "Import failed: " + e.toString() });
        }

        // Poll for import to complete
        for (var _poll = 0; _poll < 20; _poll++) {
            $.sleep(50);
            overlayItem = _findProjectItemByPath(app.project.rootItem, overlayPath, 0);
            if (overlayItem) break;
        }

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
    } catch (e) {
        return JSON.stringify({ error: "importCaptionOverlay: " + e.toString() });
    }
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
        if (!bat.open("w")) {
            return JSON.stringify({ error: "Cannot write startup script to temp folder" });
        }
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
                // Launch the installed exe (sanitize path against injection)
                var safePath = exePath.replace(/[&|<>^%"]/g, "");
                bat.writeln('"' + safePath + '"');
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
        if (!sh.open("w")) {
            return JSON.stringify({ error: "Cannot write startup script to temp folder" });
        }
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
                if (!cmdFile.open("w")) {
                    return JSON.stringify({ error: "Cannot write startup script to temp folder" });
                }
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

    if (!app || !app.project || !app.project.rootItem) {
        return JSON.stringify({ error: "No project open" });
    }

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


/**
 * Get rich information about the active sequence for deliverables generation.
 * Returns JSON: {name, duration, fps, width, height, video_tracks, audio_tracks, markers}
 */
function ocGetSequenceInfo() {
    try {
        if (!app || !app.project) {
            return JSON.stringify({ error: "No project open" });
        }
        var seq = app.project.activeSequence;
        if (!seq) {
            return JSON.stringify({ error: "No active sequence" });
        }

        // Duration
        var duration = 0;
        try { duration = seq.end ? seq.end.seconds : 0; } catch (e) {}

        // FPS
        var fps = 24.0;
        try {
            var settings = seq.getSettings();
            if (settings && settings.videoFrameRate && settings.videoFrameRate.seconds > 0) {
                fps = 1.0 / settings.videoFrameRate.seconds;
            }
        } catch (e) { _ocLog("ocGetSequenceInfo fps error: " + e.toString()); }

        // Dimensions
        var width = 0;
        var height = 0;
        try { width = seq.frameSizeHorizontal || 0; } catch (e) {}
        try { height = seq.frameSizeVertical || 0; } catch (e) {}

        // Video tracks
        var videoTracks = [];
        try {
            for (var vt = 0; vt < seq.videoTracks.numTracks; vt++) {
                var vTrack = seq.videoTracks[vt];
                var vClips = [];
                for (var vc = 0; vc < vTrack.clips.numItems; vc++) {
                    var vClip = vTrack.clips[vc];
                    var vPath = "";
                    try { vPath = _normalizeMediaPath(vClip.projectItem.getMediaPath()); } catch (e) {}
                    var vStart = 0;
                    var vEnd = 0;
                    try { vStart = vClip.start ? vClip.start.seconds : 0; } catch (e) {}
                    try { vEnd = vClip.end ? vClip.end.seconds : 0; } catch (e) {}

                    // Effects / components
                    var effects = [];
                    try {
                        if (vClip.components) {
                            for (var ci = 0; ci < vClip.components.numItems; ci++) {
                                try {
                                    var comp = vClip.components[ci];
                                    if (comp && comp.displayName) {
                                        effects.push(comp.displayName);
                                    }
                                } catch (ce) {}
                            }
                        }
                    } catch (e) {}

                    vClips.push({
                        name: vClip.name || "",
                        path: vPath,
                        start: vStart,
                        end: vEnd,
                        effects: effects
                    });
                }
                videoTracks.push({ index: vt, clips: vClips });
            }
        } catch (e) { _ocLog("ocGetSequenceInfo videoTracks error: " + e.toString()); }

        // Audio tracks
        var audioTracks = [];
        try {
            for (var at = 0; at < seq.audioTracks.numTracks; at++) {
                var aTrack = seq.audioTracks[at];
                var aClips = [];
                for (var ac = 0; ac < aTrack.clips.numItems; ac++) {
                    var aClip = aTrack.clips[ac];
                    var aPath = "";
                    try { aPath = _normalizeMediaPath(aClip.projectItem.getMediaPath()); } catch (e) {}
                    var aStart = 0;
                    var aEnd = 0;
                    try { aStart = aClip.start ? aClip.start.seconds : 0; } catch (e) {}
                    try { aEnd = aClip.end ? aClip.end.seconds : 0; } catch (e) {}
                    aClips.push({
                        name: aClip.name || "",
                        path: aPath,
                        start: aStart,
                        end: aEnd
                    });
                }
                audioTracks.push({ index: at, clips: aClips });
            }
        } catch (e) { _ocLog("ocGetSequenceInfo audioTracks error: " + e.toString()); }

        // Markers — use getFirstMarker/getNextMarker iterator (indexed access is unreliable)
        var markers = [];
        try {
            var seqMarkers = seq.markers;
            if (seqMarkers && seqMarkers.numMarkers > 0) {
                var m = seqMarkers.getFirstMarker();
                while (m) {
                    try {
                        var mTime = 0;
                        try { mTime = m.time ? m.time.seconds : 0; } catch (e) {}
                        var mName = "";
                        try { mName = m.name || m.comments || ""; } catch (e) {}
                        var mType = "";
                        try { mType = m.type || ""; } catch (e) {}
                        var mColor = 0;
                        try { mColor = m.colorByteArray ? m.colorByteArray[0] : 0; } catch (e) {}
                        markers.push({
                            time: mTime,
                            name: mName,
                            type: mType,
                            color: mColor
                        });
                    } catch (e) {}
                    try { m = seqMarkers.getNextMarker(m); } catch (e) { m = null; }
                }
            }
        } catch (e) { _ocLog("ocGetSequenceInfo markers error: " + e.toString()); }

        return JSON.stringify({
            name: seq.name || "",
            duration: duration,
            fps: fps,
            width: width,
            height: height,
            video_tracks: videoTracks,
            audio_tracks: audioTracks,
            markers: markers
        });
    } catch (e) {
        _ocLog("ocGetSequenceInfo error: " + e.toString());
        return JSON.stringify({ error: e.toString() });
    }
}


/**
 * Add markers to the active sequence from a JSON string array.
 * markersJSON: '[{"time": 30.5, "name": "Chapter 1", "color": 0, "type": "comment", "duration": 0}]'
 * Returns: '{"added": N, "errors": [...]}'
 */
function ocAddSequenceMarkers(markersJSON) {
    try {
        if (!app || !app.project) {
            return JSON.stringify({ error: "No project open" });
        }
        var seq = app.project.activeSequence;
        if (!seq) {
            return JSON.stringify({ error: "No active sequence" });
        }

        var items = [];
        try {
            items = JSON.parse(markersJSON);
        } catch (e) {
            return JSON.stringify({ error: "Invalid JSON: " + e.toString() });
        }

        var added = 0;
        var errors = [];

        var seqDuration = 0;
        try { seqDuration = seq.end ? Number(seq.end.seconds) : 0; } catch (e) {}

        for (var i = 0; i < items.length; i++) {
            var item = items[i];
            try {
                var markerTime = Number(item.time) || 0;
                if (markerTime < 0) markerTime = 0;
                if (seqDuration > 0 && markerTime > seqDuration) {
                    errors.push("Marker " + i + " time " + markerTime + "s exceeds sequence duration " + seqDuration + "s");
                    continue;
                }
                var marker = seq.markers.createMarker(markerTime);

                try { marker.name = item.name || ""; } catch (e) {}

                // Type: 0 = comment, 1 = chapter, 2 = segmentation
                try {
                    var typeStr = (item.type || "comment").toLowerCase();
                    if (typeStr === "chapter") {
                        marker.type = 1;
                    } else if (typeStr === "segmentation") {
                        marker.type = 2;
                    } else {
                        marker.type = 0;
                    }
                } catch (e) {}

                // Color — API varies by version, wrap tightly
                try {
                    var colorVal = item.color || 0;
                    marker.colorByteArray = [colorVal, colorVal, colorVal, 255];
                } catch (e) {}

                added++;
            } catch (e) {
                errors.push("Marker " + i + ": " + e.toString());
                _ocLog("ocAddSequenceMarkers item " + i + " error: " + e.toString());
            }
        }

        return JSON.stringify({ added: added, errors: errors });
    } catch (e) {
        _ocLog("ocAddSequenceMarkers error: " + e.toString());
        return JSON.stringify({ error: e.toString() });
    }
}


/**
 * Get all markers on the active sequence.
 * Returns: '[{"time": 30.5, "name": "Chapter 1", "type": "comment", "duration": 0, "color": 0}]'
 */
function ocGetSequenceMarkers() {
    try {
        if (!app || !app.project) {
            return JSON.stringify({ error: "No project open" });
        }
        var seq = app.project.activeSequence;
        if (!seq) {
            return JSON.stringify({ error: "No active sequence" });
        }

        var result = [];
        var seqMarkers = seq.markers;
        if (!seqMarkers) {
            return JSON.stringify(result);
        }

        var numM = 0;
        try { numM = seqMarkers.numMarkers; } catch (e) {}

        if (numM > 0) {
            var m = null;
            try { m = seqMarkers.getFirstMarker(); } catch (e) {}
            while (m) {
                try {
                    var mTime = 0;
                    try { mTime = m.time ? m.time.seconds : 0; } catch (e) {}
                    var mName = "";
                    try { mName = m.name || m.comments || ""; } catch (e) {}
                    var mType = "";
                    try { mType = m.type !== undefined ? String(m.type) : ""; } catch (e) {}
                    var mDuration = 0;
                    try { mDuration = m.duration ? m.duration.seconds : 0; } catch (e) {}
                    var mColor = 0;
                    try { mColor = m.colorByteArray ? m.colorByteArray[0] : 0; } catch (e) {}

                    result.push({
                        time: mTime,
                        name: mName,
                        type: mType,
                        duration: mDuration,
                        color: mColor
                    });
                } catch (e) {
                    _ocLog("ocGetSequenceMarkers item error: " + e.toString());
                }
                try { m = seqMarkers.getNextMarker(m); } catch (e) { m = null; }
            }
        }

        return JSON.stringify(result);
    } catch (e) {
        _ocLog("ocGetSequenceMarkers error: " + e.toString());
        return JSON.stringify({ error: e.toString() });
    }
}


/**
 * Delete regions from the active sequence timeline (ripple delete).
 * Removes clips FULLY contained within each cut range.
 * cutsJSON: '[{"start": 10.5, "end": 12.3}, ...]'
 * Returns: '{"applied": N, "errors": [...]}'
 */
// NOTE: Premiere Pro ExtendScript does not support undo groups (unlike After Effects).
// Each clip.remove() creates a separate undo step. This is a Premiere limitation.
// When UXP API supports undo groups, this should be wrapped in one.
function ocApplySequenceCuts(cutsJSON) {
    try {
        if (!app || !app.project) {
            return JSON.stringify({ error: "No project open" });
        }
        var seq = app.project.activeSequence;
        if (!seq) {
            return JSON.stringify({ error: "No active sequence" });
        }

        var cuts = [];
        try {
            cuts = JSON.parse(cutsJSON);
        } catch (e) {
            return JSON.stringify({ error: "Invalid JSON: " + e.toString() });
        }

        if (!cuts || cuts.length === 0) {
            return JSON.stringify({ applied: 0, errors: [] });
        }

        // Coerce cut times to numbers before sorting so NaN doesn't corrupt sort order
        for (var si = 0; si < cuts.length; si++) {
            cuts[si].start = Number(cuts[si].start) || 0;
            cuts[si].end = Number(cuts[si].end) || 0;
        }
        // Sort cuts in reverse order so earlier ripple deletes don't shift later timecodes
        cuts.sort(function (a, b) { return b.start - a.start; });

        var applied = 0;
        var errors = [];

        for (var ci = 0; ci < cuts.length; ci++) {
            var cut = cuts[ci];
            var cutStart = Number(cut.start);
            var cutEnd = Number(cut.end);

            // Validate numeric cut range before deleting anything
            if (isNaN(cutStart) || isNaN(cutEnd) || cutStart >= cutEnd) continue;

            // Iterate video tracks
            for (var vt = 0; vt < seq.videoTracks.numTracks; vt++) {
                var vTrack = seq.videoTracks[vt];
                // Walk backwards so removing an item doesn't shift the index
                for (var vc = vTrack.clips.numItems - 1; vc >= 0; vc--) {
                    try {
                        var vClip = vTrack.clips[vc];
                        var clipStart = vClip.start ? Number(vClip.start.seconds) : 0;
                        var clipEnd = vClip.end ? Number(vClip.end.seconds) : 0;
                        // Remove clips fully contained within cut range (with 0.01s tolerance for floating-point)
                        if (clipStart >= cutStart - 0.01 && clipEnd <= cutEnd + 0.01) {
                            vClip.remove(false, true);
                            applied++;
                        }
                    } catch (e) {
                        errors.push("Video track " + vt + " clip " + vc + ": " + e.toString());
                        _ocLog("ocApplySequenceCuts vTrack error: " + e.toString());
                    }
                }
            }

            // Iterate audio tracks
            for (var at = 0; at < seq.audioTracks.numTracks; at++) {
                var aTrack = seq.audioTracks[at];
                for (var ac = aTrack.clips.numItems - 1; ac >= 0; ac--) {
                    try {
                        var aClip = aTrack.clips[ac];
                        var aClipStart = aClip.start ? Number(aClip.start.seconds) : 0;
                        var aClipEnd = aClip.end ? Number(aClip.end.seconds) : 0;
                        if (aClipStart >= cutStart - 0.01 && aClipEnd <= cutEnd + 0.01) {
                            aClip.remove(false, true);
                            applied++;
                        }
                    } catch (e) {
                        errors.push("Audio track " + at + " clip " + ac + ": " + e.toString());
                        _ocLog("ocApplySequenceCuts aTrack error: " + e.toString());
                    }
                }
            }
        }

        return JSON.stringify({ applied: applied, errors: errors });
    } catch (e) {
        _ocLog("ocApplySequenceCuts error: " + e.toString());
        return JSON.stringify({ error: e.toString() });
    }
}


/**
 * Apply scale and position keyframes to a clip on a video track.
 * trackIndex: int (0-based video track index)
 * clipStartTime: float (clip's start time in seconds, used to identify the clip)
 * keyframesJSON: '[{"time": 0.0, "scale": 100, "x": 0, "y": 0}, ...]'
 * Returns: '{"success": true}' or '{"error": "..."}'
 */
function ocApplyClipKeyframes(trackIndex, clipStartTime, keyframesJSON) {
    try {
        if (!app || !app.project) {
            return JSON.stringify({ error: "No project open" });
        }
        var seq = app.project.activeSequence;
        if (!seq) {
            return JSON.stringify({ error: "No active sequence" });
        }

        var keyframes = [];
        try {
            keyframes = JSON.parse(keyframesJSON);
        } catch (e) {
            return JSON.stringify({ error: "Invalid JSON: " + e.toString() });
        }

        trackIndex = Number(trackIndex) || 0;
        clipStartTime = Number(clipStartTime);
        if (isNaN(clipStartTime)) {
            return JSON.stringify({ error: "Invalid clipStartTime" });
        }

        var vTrack = seq.videoTracks[trackIndex];
        if (!vTrack) {
            return JSON.stringify({ error: "Video track " + trackIndex + " not found" });
        }

        // Find the clip whose start time is within 0.1s of clipStartTime
        var targetClip = null;
        for (var ci = 0; ci < vTrack.clips.numItems; ci++) {
            try {
                var c = vTrack.clips[ci];
                var cs = (c.start && c.start.seconds !== undefined) ? Number(c.start.seconds) : NaN;
                if (!isNaN(cs) && Math.abs(cs - clipStartTime) < 0.1) {
                    targetClip = c;
                    break;
                }
            } catch (e) {}
        }

        if (!targetClip) {
            return JSON.stringify({ error: "Clip not found at time " + clipStartTime + " on track " + trackIndex });
        }

        // Find the Motion component
        var motionComponent = null;
        try {
            if (targetClip.components) {
                for (var compIdx = 0; compIdx < targetClip.components.numItems; compIdx++) {
                    try {
                        var comp = targetClip.components[compIdx];
                        if (comp && (comp.displayName === "Motion" || comp.matchName === "AE.ADBE Motion")) {
                            motionComponent = comp;
                            break;
                        }
                    } catch (e) {}
                }
                // Fallback: index 0 is typically Motion
                if (!motionComponent && targetClip.components.numItems > 0) {
                    motionComponent = targetClip.components[0];
                }
            }
        } catch (e) {
            return JSON.stringify({ error: "Cannot access clip components: " + e.toString() });
        }

        if (!motionComponent) {
            return JSON.stringify({ error: "Motion component not found on clip" });
        }

        // Find Scale and Position properties
        var scaleProp = null;
        var posProp = null;
        try {
            if (motionComponent.properties) {
                for (var pi = 0; pi < motionComponent.properties.numItems; pi++) {
                    try {
                        var prop = motionComponent.properties[pi];
                        if (!prop) continue;
                        var pName = prop.displayName || "";
                        if (pName === "Scale" || pName === "Uniform Scale") {
                            scaleProp = prop;
                        } else if (pName === "Position") {
                            posProp = prop;
                        }
                    } catch (e) {}
                }
            }
        } catch (e) {
            return JSON.stringify({ error: "Cannot access motion properties: " + e.toString() });
        }

        // Apply keyframes
        for (var ki = 0; ki < keyframes.length; ki++) {
            var kf = keyframes[ki];
            var kfTime = Number(kf.time) || 0;

            if (scaleProp) {
                try {
                    scaleProp.addKey(kfTime);
                    scaleProp.setValueAtKey(kfTime, kf.scale !== undefined ? kf.scale : 100);
                } catch (e) {
                    _ocLog("ocApplyClipKeyframes scale key error: " + e.toString());
                }
            }

            if (posProp) {
                try {
                    posProp.addKey(kfTime);
                    posProp.setValueAtKey(kfTime, [kf.x || 0, kf.y || 0]);
                } catch (e) {
                    _ocLog("ocApplyClipKeyframes position key error: " + e.toString());
                }
            }
        }

        return JSON.stringify({ success: true });
    } catch (e) {
        _ocLog("ocApplyClipKeyframes error: " + e.toString());
        return JSON.stringify({ error: e.toString() });
    }
}


/**
 * Rename project panel items by nodeId.
 * renamesJSON: '[{"nodeId": "abc123", "newName": "Interview_01"}, ...]'
 * Returns: '{"renamed": N, "errors": [...]}'
 */
function ocBatchRenameProjectItems(renamesJSON) {
    try {
        if (!app || !app.project) {
            return JSON.stringify({ error: "No project open" });
        }

        var renames = [];
        try {
            renames = JSON.parse(renamesJSON);
        } catch (e) {
            return JSON.stringify({ error: "Invalid JSON: " + e.toString() });
        }

        var renamed = 0;
        var errors = [];

        for (var i = 0; i < renames.length; i++) {
            var rename = renames[i];
            try {
                if (!rename || !rename.nodeId || typeof rename.newName !== "string" || rename.newName.length === 0) {
                    errors.push("Rename " + i + ": missing nodeId or newName");
                    continue;
                }
                var found = _findByNodeId(app.project.rootItem, rename.nodeId, 0);
                if (found) {
                    found.name = rename.newName;
                    renamed++;
                } else {
                    errors.push("nodeId not found: " + rename.nodeId);
                }
            } catch (e) {
                errors.push("Rename " + i + " (" + rename.nodeId + "): " + e.toString());
                _ocLog("ocBatchRenameProjectItems item " + i + " error: " + e.toString());
            }
        }

        return JSON.stringify({ renamed: renamed, errors: errors });
    } catch (e) {
        _ocLog("ocBatchRenameProjectItems error: " + e.toString());
        return JSON.stringify({ error: e.toString() });
    }
}

/**
 * Recursive helper: find a project item by nodeId.
 */
function _findByNodeId(parent, nodeId, depth) {
    if (depth > 20) return null;
    var numChildren = 0;
    try { numChildren = parent.children.numItems; } catch (e) { return null; }
    for (var i = 0; i < numChildren; i++) {
        try {
            var item = parent.children[i];
            if (!item) continue;
            if (item.nodeId === nodeId) return item;
            // Recurse into bins
            var isBin = false;
            try { isBin = (item.type === 2); } catch (e) {}
            if (isBin) {
                var found = _findByNodeId(item, nodeId, depth + 1);
                if (found) return found;
            }
        } catch (e) {}
    }
    return null;
}


/**
 * Create project bins and auto-sort items into them by rules.
 * rulesJSON: '[{"binName": "B-Roll", "rule": "contains", "field": "name", "value": "broll"}, ...]'
 * Returns: '{"bins_created": N, "items_moved": M}'
 */
function ocCreateSmartBins(rulesJSON) {
    try {
        if (!app || !app.project) {
            return JSON.stringify({ error: "No project open" });
        }

        var rules = [];
        try {
            rules = JSON.parse(rulesJSON);
        } catch (e) {
            return JSON.stringify({ error: "Invalid JSON: " + e.toString() });
        }

        var binsCreated = 0;
        var itemsMoved = 0;

        // Collect all media items from the project
        var mediaItems = [];
        _collectMediaItems(app.project.rootItem, mediaItems, 0);

        for (var ri = 0; ri < rules.length; ri++) {
            var rule = rules[ri];
            var binName = rule.binName || "Unnamed Bin";
            var ruleType = (rule.rule || "contains").toLowerCase();
            var ruleField = (rule.field || "name").toLowerCase();
            var ruleValue = (rule.value !== undefined ? String(rule.value) : "").toLowerCase();

            // Find or create the target bin
            var targetBin = null;
            try {
                // Check root for existing bin with this name
                var root = app.project.rootItem;
                for (var bi = 0; bi < root.children.numItems; bi++) {
                    try {
                        var child = root.children[bi];
                        if (child.type === 2 && child.name === binName) {
                            targetBin = child;
                            break;
                        }
                    } catch (e) {}
                }
                if (!targetBin) {
                    targetBin = root.createBin(binName);
                    binsCreated++;
                }
            } catch (e) {
                _ocLog("ocCreateSmartBins bin create error: " + e.toString());
                continue;
            }

            if (!targetBin) continue;

            // Test each media item against the rule
            for (var ii = 0; ii < mediaItems.length; ii++) {
                var mediaItem = mediaItems[ii];
                try {
                    var matches = false;

                    if (ruleField === "name") {
                        var itemNameLower = (mediaItem.name || "").toLowerCase();
                        if (ruleType === "contains") {
                            matches = itemNameLower.indexOf(ruleValue) >= 0;
                        } else if (ruleType === "starts_with") {
                            matches = itemNameLower.indexOf(ruleValue) === 0;
                        } else if (ruleType === "ends_with") {
                            matches = itemNameLower.length >= ruleValue.length &&
                                      itemNameLower.indexOf(ruleValue, itemNameLower.length - ruleValue.length) !== -1;
                        }
                    } else if (ruleField === "type") {
                        var hasVid = false;
                        var hasAud = false;
                        try { hasVid = mediaItem.hasVideo(); } catch (e) {}
                        try { hasAud = mediaItem.hasAudio(); } catch (e) {}
                        if (ruleValue === "video") {
                            matches = hasVid;
                        } else if (ruleValue === "audio") {
                            matches = hasAud && !hasVid;
                        } else if (ruleValue === "av") {
                            matches = hasVid && hasAud;
                        }
                    } else if (ruleField === "duration") {
                        var dur = 0;
                        try {
                            var op = mediaItem.getOutPoint();
                            dur = op ? op.seconds : 0;
                        } catch (e) {}
                        var ruleNum = parseFloat(ruleValue) || 0;
                        if (ruleType === "duration_gt") {
                            matches = dur > ruleNum;
                        } else if (ruleType === "duration_lt") {
                            matches = dur < ruleNum;
                        }
                    }

                    if (matches) {
                        try {
                            mediaItem.moveBin(targetBin);
                            itemsMoved++;
                        } catch (e) {
                            _ocLog("ocCreateSmartBins moveBin error: " + e.toString());
                        }
                    }
                } catch (e) {
                    _ocLog("ocCreateSmartBins item test error: " + e.toString());
                }
            }
        }

        return JSON.stringify({ bins_created: binsCreated, items_moved: itemsMoved });
    } catch (e) {
        _ocLog("ocCreateSmartBins error: " + e.toString());
        return JSON.stringify({ error: e.toString() });
    }
}

/**
 * Helper: collect all non-bin media items from the project tree into an array.
 */
function _collectMediaItems(parent, items, depth) {
    if (depth > 20) return;
    var numChildren = 0;
    try { numChildren = parent.children.numItems; } catch (e) { return; }
    for (var i = 0; i < numChildren; i++) {
        try {
            var item = parent.children[i];
            if (!item) continue;
            var isBin = false;
            try { isBin = (item.type === 2); } catch (e) {}
            if (isBin) {
                _collectMediaItems(item, items, depth + 1);
            } else {
                items.push(item);
            }
        } catch (e) {}
    }
}


/**
 * Create a native Premiere Pro caption track from SRT-style segment data.
 * Writes an SRT file to temp and imports it for maximum cross-version compatibility.
 * srtJSON: '[{"start": 0.5, "end": 2.3, "text": "Hello world"}, ...]'
 * Returns: '{"success": true, "captions_added": N}' or '{"error": "..."}'
 */
function ocAddNativeCaptionTrack(srtJSON) {
    try {
        if (!app || !app.project) {
            return JSON.stringify({ error: "No project open" });
        }

        var segments = [];
        try {
            segments = JSON.parse(srtJSON);
        } catch (e) {
            return JSON.stringify({ error: "Invalid JSON: " + e.toString() });
        }

        if (!segments || segments.length === 0) {
            return JSON.stringify({ error: "No caption segments provided" });
        }

        // Helper: format seconds as SRT timecode HH:MM:SS,mmm
        function _secondsToSrtTime(secs) {
            var totalMs = Math.round(secs * 1000);
            var ms = totalMs % 1000;
            var totalSec = Math.floor(totalMs / 1000);
            var s = totalSec % 60;
            var totalMin = Math.floor(totalSec / 60);
            var m = totalMin % 60;
            var h = Math.floor(totalMin / 60);

            var hh = h < 10 ? "0" + h : String(h);
            var mm = m < 10 ? "0" + m : String(m);
            var ss = s < 10 ? "0" + s : String(s);
            var msStr = ms < 10 ? "00" + ms : (ms < 100 ? "0" + ms : String(ms));
            return hh + ":" + mm + ":" + ss + "," + msStr;
        }

        // Write SRT file to temp
        var tempPath = Folder.temp.fsName + "/opencut_captions.srt";
        var tempFile = new File(tempPath);
        if (!tempFile.open("w")) {
            return JSON.stringify({ error: "Cannot write temp SRT file" });
        }

        try {
            var srtIndex = 0;
            for (var i = 0; i < segments.length; i++) {
                var seg = segments[i];
                var segS = Number(seg.start);
                var segE = Number(seg.end);
                // Skip invalid segments
                if (isNaN(segS) || isNaN(segE) || segS < 0 || segE <= segS) continue;
                srtIndex++;
                tempFile.writeln(String(srtIndex));
                tempFile.writeln(_secondsToSrtTime(segS) + " --> " + _secondsToSrtTime(segE));
                tempFile.writeln(seg.text || "");
                tempFile.writeln("");
            }
        } finally {
            tempFile.close();
        }

        // Import the SRT into the project
        var captionBin = _findOrCreateBin("OpenCut Captions");
        var targetBin = captionBin || app.project.rootItem;

        try {
            app.project.importFiles([tempFile.fsName], false, targetBin, false);
        } catch (e) {
            return JSON.stringify({ error: "SRT import failed: " + e.toString() });
        } finally {
            try { tempFile.remove(); } catch (e2) {}
        }

        return JSON.stringify({ success: true, captions_added: srtIndex });
    } catch (e) {
        _ocLog("ocAddNativeCaptionTrack error: " + e.toString());
        return JSON.stringify({ error: e.toString() });
    }
}


/**
 * Get all bins (folders) in the project with their contents.
 * Returns: '[{"name": "BinName", "path": "BinName/SubBin", "item_count": N}]'
 */
function ocGetProjectBins() {
    try {
        if (!app || !app.project) {
            return JSON.stringify({ error: "No project open" });
        }

        var bins = [];
        _collectBins(app.project.rootItem, bins, "", 0);
        return JSON.stringify(bins);
    } catch (e) {
        _ocLog("ocGetProjectBins error: " + e.toString());
        return JSON.stringify({ error: e.toString() });
    }
}

/**
 * Helper: recursively collect bins into an array with path tracking.
 */
function _collectBins(parent, bins, parentPath, depth) {
    if (depth > 20) return;
    var numChildren = 0;
    try { numChildren = parent.children.numItems; } catch (e) { return; }
    for (var i = 0; i < numChildren; i++) {
        try {
            var item = parent.children[i];
            if (!item) continue;
            var isBin = false;
            try { isBin = (item.type === 2); } catch (e) {}
            if (isBin) {
                var binName = item.name || "";
                var binPath = parentPath ? (parentPath + "/" + binName) : binName;
                var itemCount = 0;
                try { itemCount = item.children.numItems; } catch (e) {}
                bins.push({
                    name: binName,
                    path: binPath,
                    item_count: itemCount
                });
                // Recurse into sub-bins
                _collectBins(item, bins, binPath, depth + 1);
            }
        } catch (e) {
            _ocLog("_collectBins item " + i + " error: " + e.toString());
        }
    }
}


/**
 * Queue a sequence range export to Adobe Media Encoder.
 * outputPath: full path to the output file
 * startSeconds: in-point in seconds
 * endSeconds: out-point in seconds
 * Returns: '{"success": true}' or '{"error": "..."}'
 */
function ocExportSequenceRange(outputPath, startSeconds, endSeconds) {
    try {
        if (!app || !app.project) {
            return JSON.stringify({ error: "No project open" });
        }
        var seq = app.project.activeSequence;
        if (!seq) {
            return JSON.stringify({ error: "No active sequence" });
        }

        // Validate time parameters
        var start = Number(startSeconds);
        var end = Number(endSeconds);
        if (isNaN(start) || isNaN(end) || start < 0 || end <= start) {
            return JSON.stringify({ error: "Invalid time range: start=" + startSeconds + " end=" + endSeconds });
        }
        startSeconds = start;
        endSeconds = end;

        // Save original in/out points so we can restore after export
        var origIn = -1, origOut = -1;
        try { origIn = seq.getInPoint(); } catch (e) {}
        try { origOut = seq.getOutPoint(); } catch (e) {}

        // Set in/out points on the sequence — abort if either fails
        try {
            seq.setInPoint(startSeconds);
        } catch (e) {
            _ocLog("ocExportSequenceRange setInPoint error: " + e.toString());
            return JSON.stringify({ error: "Failed to set in-point: " + e.toString() });
        }
        try {
            seq.setOutPoint(endSeconds);
        } catch (e) {
            _ocLog("ocExportSequenceRange setOutPoint error: " + e.toString());
            // Restore in-point before returning
            try { if (origIn >= 0) seq.setInPoint(origIn); } catch (e2) {}
            return JSON.stringify({ error: "Failed to set out-point: " + e.toString() });
        }

        // Queue to AME
        var encodeError = null;
        try {
            app.encoder.encodeSequence(seq, outputPath, "", 1, 2);
        } catch (e) {
            encodeError = e.toString();
        }

        // Restore original in/out points regardless of success/failure
        try { if (origIn >= 0) seq.setInPoint(origIn); else seq.setInPoint(0); } catch (e) {}
        try { if (origOut >= 0) seq.setOutPoint(origOut); else seq.setOutPoint(seq.end); } catch (e) {}

        if (encodeError) {
            return JSON.stringify({ error: "AME encode failed: " + encodeError });
        }
        return JSON.stringify({ success: true });
    } catch (e) {
        _ocLog("ocExportSequenceRange error: " + e.toString());
        return JSON.stringify({ error: e.toString() });
    }
}


// ============================================================================
// Journal inverse helpers (v1.9.28) — called by the panel's rollback UI.
//
// Each inverse function takes a JSON payload that the panel previously stored
// in the journal at the time of the forward operation, and tries to undo it
// on a best-effort basis. On ES3 runtimes we cannot use JSON.parse safely in
// every host, so we fall back to eval when JSON is missing.
// ============================================================================
function _ocParse(jsonStr) {
    if (typeof JSON !== "undefined" && JSON.parse) {
        return JSON.parse(jsonStr);
    }
    // Legacy ExtendScript fallback: validate that the payload only contains
    // JSON tokens before using eval() on older hosts with no JSON.parse().
    var sanitized = String(jsonStr || "")
        .replace(/\\(?:["\\\/bfnrt]|u[0-9a-fA-F]{4})/g, "@")
        .replace(/"[^"\\\n\r]*"|true|false|null|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?/g, "]")
        .replace(/(?:^|:|,)(?:\s*\[)+/g, "");
    if (!/^[\],:{}\s]*$/.test(sanitized)) {
        throw new Error("Unsafe JSON payload");
    }
    return eval("(" + jsonStr + ")");  // eslint-disable-line no-eval
}

// Remove markers matching {time, comment} fingerprints from the active sequence.
// Accepts either {markers: [{time, comment}, ...]} or a bare array.
function ocRemoveSequenceMarkers(fingerprintsJSON) {
    try {
        if (!app || !app.project || !app.project.activeSequence) {
            return JSON.stringify({ error: "No active sequence" });
        }
        var payload = _ocParse(fingerprintsJSON);
        var fingerprints = payload && payload.markers ? payload.markers : payload;
        if (!fingerprints || typeof fingerprints.length !== "number") {
            return JSON.stringify({ error: "Invalid fingerprints" });
        }

        var seq = app.project.activeSequence;
        var markers = seq.markers;
        if (!markers) return JSON.stringify({ error: "Sequence has no markers collection" });

        // Build a quick lookup by "time|comment" so we don't do O(N*M).
        var keyFor = function (t, c) {
            return Number(t).toFixed(4) + "|" + (c || "");
        };
        var targetSet = {};
        var wanted = 0;
        for (var i = 0; i < fingerprints.length; i++) {
            var fp = fingerprints[i];
            if (fp == null) continue;
            targetSet[keyFor(fp.time, fp.comment || fp.name || "")] = true;
            wanted++;
        }

        var removed = 0;
        var m = markers.getFirstMarker();
        while (m) {
            var nextM;
            try { nextM = markers.getNextMarker(m); } catch (e0) { nextM = null; }
            var mtime = null;
            try { mtime = m.start && m.start.seconds != null ? m.start.seconds : m.start; } catch (e1) {}
            var mcomment = "";
            try { mcomment = m.comments || m.name || ""; } catch (e2) {}
            if (mtime != null && targetSet[keyFor(mtime, mcomment)]) {
                try { markers.deleteMarker(m); removed++; } catch (e3) {}
            }
            m = nextM;
        }

        return JSON.stringify({
            success: true,
            removed: removed,
            wanted: wanted
        });
    } catch (e) {
        _ocLog("ocRemoveSequenceMarkers error: " + e.toString());
        return JSON.stringify({ error: e.toString() });
    }
}

// Restore previous names for project items. Payload: {renames:[{nodeId,oldName}]}.
function ocUnrenameItems(mapJSON) {
    try {
        if (!app || !app.project) return JSON.stringify({ error: "No project" });
        var payload = _ocParse(mapJSON);
        var renames = payload && payload.renames ? payload.renames : payload;
        if (!renames || typeof renames.length !== "number") {
            return JSON.stringify({ error: "Invalid renames" });
        }

        var restored = 0, missed = 0;
        for (var i = 0; i < renames.length; i++) {
            var entry = renames[i];
            if (!entry || !entry.oldName) { missed++; continue; }
            var item = null;
            if (entry.nodeId) {
                try { item = _findByNodeId(entry.nodeId); } catch (e1) { item = null; }
            }
            if (!item && entry.currentName) {
                // Fall back to name lookup
                var all = _collectMediaItems(app.project.rootItem);
                for (var j = 0; j < all.length; j++) {
                    if (all[j].name === entry.currentName) { item = all[j]; break; }
                }
            }
            if (item) {
                try {
                    item.name = entry.oldName;
                    restored++;
                } catch (e2) {
                    _ocLog("ocUnrenameItems rename error: " + e2.toString());
                    missed++;
                }
            } else {
                missed++;
            }
        }

        return JSON.stringify({ success: true, restored: restored, missed: missed });
    } catch (e) {
        _ocLog("ocUnrenameItems error: " + e.toString());
        return JSON.stringify({ error: e.toString() });
    }
}

// Remove a previously-imported sequence by name. Payload: {name:"..."}.
function ocRemoveImportedSequence(payloadJSON) {
    try {
        if (!app || !app.project) return JSON.stringify({ error: "No project" });
        var payload = _ocParse(payloadJSON);
        var name = payload && payload.name ? payload.name : "";
        if (!name) return JSON.stringify({ error: "Missing sequence name" });

        var target = null;
        var seqs = app.project.sequences;
        if (seqs) {
            for (var i = 0; i < seqs.numSequences; i++) {
                var s = seqs[i];
                if (s && s.name === name) { target = s; break; }
            }
        }
        if (!target) return JSON.stringify({ error: "Sequence '" + name + "' not found" });

        // ProjectItem.deleteItem() for sequences isn't universal across
        // Premiere versions, so try the rootItem walk first.
        var items = _collectMediaItems(app.project.rootItem);
        for (var j = 0; j < items.length; j++) {
            if (items[j].name === name && items[j].type === ProjectItemType.CLIP) {
                try {
                    items[j].deleteAsset ? items[j].deleteAsset() : app.project.deleteSequence(target);
                    return JSON.stringify({ success: true, removed: name });
                } catch (e0) {}
            }
        }
        try {
            app.project.deleteSequence(target);
            return JSON.stringify({ success: true, removed: name });
        } catch (e1) {
            return JSON.stringify({ error: "Could not delete: " + e1.toString() });
        }
    } catch (e) {
        _ocLog("ocRemoveImportedSequence error: " + e.toString());
        return JSON.stringify({ error: e.toString() });
    }
}

// v1.9.34 (H) — Move the active sequence's playhead to *seconds*. One-arg
// form for simpler panel invocation. Uses ticks (254016000000 per second)
// when setPlayerPosition expects a tick string, or falls back to seconds.
function ocSetSequencePlayhead(seconds) {
    try {
        if (!app || !app.project || !app.project.activeSequence) {
            return JSON.stringify({ error: "No active sequence" });
        }
        var secs = Number(seconds);
        if (!isFinite(secs) || secs < 0) {
            return JSON.stringify({ error: "Invalid seconds" });
        }
        var seq = app.project.activeSequence;

        // Try the modern setPlayerPosition(ticks_as_string) signature first.
        var ticks = Math.floor(secs * 254016000000);
        try {
            seq.setPlayerPosition(ticks.toString());
            return JSON.stringify({ success: true, seconds: secs });
        } catch (e1) {}

        // Fallback: some Premiere builds accept a Number
        try {
            seq.setPlayerPosition(ticks);
            return JSON.stringify({ success: true, seconds: secs });
        } catch (e2) {}

        // Last resort: cursor via Time object
        try {
            var t = {};
            t.ticks = ticks.toString();
            seq.setPlayerPosition(t);
            return JSON.stringify({ success: true, seconds: secs });
        } catch (e3) {
            return JSON.stringify({ error: "setPlayerPosition failed on this Premiere version" });
        }
    } catch (e) {
        _ocLog("ocSetSequencePlayhead error: " + e.toString());
        return JSON.stringify({ error: e.toString() });
    }
}

// Remove an imported project item by node id. Payload: {nodeId:"..."}.
function ocRemoveImportedItem(payloadJSON) {
    try {
        if (!app || !app.project) return JSON.stringify({ error: "No project" });
        var payload = _ocParse(payloadJSON);
        var nodeId = payload && payload.nodeId ? payload.nodeId : "";
        if (!nodeId) return JSON.stringify({ error: "Missing nodeId" });

        var item = _findByNodeId(nodeId);
        if (!item) return JSON.stringify({ error: "Item not found" });

        try {
            if (item.deleteAsset) { item.deleteAsset(); }
            else if (app.project.deleteAsset) { app.project.deleteAsset(item); }
            else { return JSON.stringify({ error: "No delete API on this item" }); }
        } catch (eDel) {
            return JSON.stringify({ error: "deleteAsset failed: " + eDel.toString() });
        }
        return JSON.stringify({ success: true, removed: nodeId });
    } catch (e) {
        _ocLog("ocRemoveImportedItem error: " + e.toString());
        return JSON.stringify({ error: e.toString() });
    }
}
