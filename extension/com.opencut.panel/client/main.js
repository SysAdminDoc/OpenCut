/* ============================================================
   OpenCut CEP Panel - Main Controller v0.3.0
   - SSE job tracking (polling fallback)
   - Waveform visualization + segment toggling
   - Undo/Redo for segment edits
   - Keyboard shortcuts
   - Merge/Split segments
   - Audio preview playback
   - Audio-only export
   - Batch processing
   ============================================================ */
(function () {
    "use strict";

    var BACKEND = "http://127.0.0.1:5679";
    var POLL_MS = 700;
    var HEALTH_MS = 4000;
    var SSE_OK = typeof EventSource !== "undefined";
    var UNDO_MAX = 50;

    // ---- Core State ----
    var cs = null;
    var inPremiere = false;
    var connected = false;
    var capabilities = {};
    var selectedPath = "";
    var selectedName = "";
    var currentJob = null;
    var pollTimer = null;
    var healthTimer = null;
    var lastXmlPath = "";
    var lastCaptionPath = "";
    var lastOverlayPath = "";
    var projectMedia = [];
    var projectFolder = "";
    var backendStartAttempted = false;
    var jobStartTime = 0;
    var elapsedTimer = null;
    var activeStream = null;

    // ---- Waveform / Segment State ----
    var allSegments = [];
    var waveformPeaks = null;
    var waveformDuration = 0;
    var waveformMaxRms = 0;
    var segmentsModified = false;
    var selectedSegIdx = -1;

    // ---- Undo / Redo ----
    var undoStack = [];
    var redoStack = [];

    // ---- Batch State ----
    var batchRunning = false;
    var batchCancel = false;
    var batchResults = [];

    // ---- DOM ----
    var el = {};
    function $(id) { return document.getElementById(id); }

    function initDOM() {
        el.connDot = $("connDot");
        el.connLabel = $("connLabel");
        el.alertBanner = $("alertBanner");
        el.alertText = $("alertText");
        el.alertDismiss = $("alertDismiss");
        el.clipSelect = $("clipSelect");
        el.fileInfoBox = $("fileInfoBox");
        el.fileNameDisplay = $("fileNameDisplay");
        el.fileMetaDisplay = $("fileMetaDisplay");
        el.refreshClipsBtn = $("refreshClipsBtn");
        el.useSelectionBtn = $("useSelectionBtn");
        el.browseFileBtn = $("browseFileBtn");
        el.batchBtn = $("batchBtn");
        el.actionSection = $("actionSection");
        el.silencePreset = $("silencePreset");
        el.customSilenceSettings = $("customSilenceSettings");
        el.threshold = $("threshold");
        el.thresholdVal = $("thresholdVal");
        el.minDuration = $("minDuration");
        el.minDurationVal = $("minDurationVal");
        el.padBefore = $("padBefore");
        el.padAfter = $("padAfter");
        el.captionModel = $("captionModel");
        el.captionLang = $("captionLang");
        el.captionFormat = $("captionFormat");
        el.captionMode = $("captionMode");
        el.captionStyle = $("captionStyle");
        el.srtOptions = $("srtOptions");
        el.styledOptions = $("styledOptions");
        el.stylePreview = $("stylePreview");
        el.captionWordHighlight = $("captionWordHighlight");
        el.captionAutoAction = $("captionAutoAction");
        el.actionWordsInput = $("actionWordsInput");
        el.captionsHint = $("captionsHint");
        el.captionsBadge = $("captionsBadge");
        el.fullPreset = $("fullPreset");
        el.fullZoom = $("fullZoom");
        el.fullCaptions = $("fullCaptions");
        el.fullFillers = $("fullFillers");
        el.fillerModel = $("fillerModel");
        el.fillerChecks = $("fillerChecks");
        el.fillerCustom = $("fillerCustom");
        el.fillerSilence = $("fillerSilence");
        el.fillersHint = $("fillersHint");
        el.runSilenceBtn = $("runSilenceBtn");
        el.runFillersBtn = $("runFillersBtn");
        el.runCaptionsBtn = $("runCaptionsBtn");
        el.runFullBtn = $("runFullBtn");
        el.progressSection = $("progressSection");
        el.progressBar = $("progressBar");
        el.progressLabel = $("progressLabel");
        el.progressElapsed = $("progressElapsed");
        el.cancelBtn = $("cancelBtn");
        el.resultsSection = $("resultsSection");
        el.resultsCard = $("resultsCard");
        el.resultsTitle = $("resultsTitle");
        el.resultsStats = $("resultsStats");
        el.resultsPath = $("resultsPath");
        el.editTimelineBtn = $("editTimelineBtn");
        el.importBtn = $("importBtn");
        el.importCaptionsBtn = $("importCaptionsBtn");
        el.newJobBtn = $("newJobBtn");
        el.installWhisperBtn = $("installWhisperBtn");
        // Waveform
        el.waveformSection = $("waveformSection");
        el.waveformCanvas = $("waveformCanvas");
        el.waveformWrap = $("waveformWrap");
        el.waveformLoading = $("waveformLoading");
        el.waveformHint = $("waveformHint");
        el.previewIndicator = $("previewIndicator");
        el.segmentCount = $("segmentCount");
        el.segmentList = $("segmentList");
        el.enableAllBtn = $("enableAllBtn");
        el.disableAllBtn = $("disableAllBtn");
        // Toolbar
        el.undoBtn = $("undoBtn");
        el.redoBtn = $("redoBtn");
        el.mergeBtn = $("mergeBtn");
        el.splitBtn = $("splitBtn");
        // Export
        el.exportVideoSection = $("exportVideoSection");
        el.exportVideoBtn = $("exportVideoBtn");
        el.exportAudioBtn = $("exportAudioBtn");
        el.exportQuality = $("exportQuality");
        el.exportFormat = $("exportFormat");
        el.audioExportFormat = $("audioExportFormat");
        // Batch
        el.batchSection = $("batchSection");
        el.batchProgressBar = $("batchProgressBar");
        el.batchCounter = $("batchCounter");
        el.batchStatusLabel = $("batchStatusLabel");
        el.batchList = $("batchList");
        el.cancelBatchBtn = $("cancelBatchBtn");
        el.closeBatchBtn = $("closeBatchBtn");
        // Shortcuts
        el.shortcutsOverlay = $("shortcutsOverlay");
        el.shortcutsHelpBtn = $("shortcutsHelpBtn");
        el.closeShortcutsBtn = $("closeShortcutsBtn");
        // Audio preview
        el.previewAudio = $("previewAudio");
    }

    // ================================================================
    // CSInterface / ExtendScript Bridge
    // ================================================================
    function initCSInterface() {
        try { cs = new CSInterface(); inPremiere = true; }
        catch (e) { cs = null; inPremiere = false; }
    }

    function jsx(script, cb) {
        if (cs) cs.evalScript(script, function (r) { if (cb) cb(r); });
        else if (cb) cb("null");
    }

    // ================================================================
    // Backend Communication
    // ================================================================
    function api(method, path, data, cb, timeout) {
        var xhr = new XMLHttpRequest();
        xhr.open(method, BACKEND + path, true);
        xhr.setRequestHeader("Content-Type", "application/json");
        xhr.timeout = timeout || 30000;
        xhr.onload = function () {
            try { cb(null, JSON.parse(xhr.responseText), xhr.status); }
            catch (e) { cb(e, null, xhr.status); }
        };
        xhr.onerror = function () { cb(new Error("Connection failed"), null, 0); };
        xhr.ontimeout = function () { cb(new Error("Timeout"), null, 0); };
        xhr.send(data ? JSON.stringify(data) : null);
    }

    function checkHealth() {
        api("GET", "/health", null, function (err, data) {
            var wasConnected = connected;
            connected = !err && data && data.status === "ok";
            updateConnectionUI();
            updateButtons();
            if (connected && !wasConnected) {
                api("GET", "/capabilities", null, function (e2, caps) {
                    if (!e2 && caps) { capabilities = caps; updateCapabilities(); }
                });
                hideAlert();
            }
            if (!connected && !backendStartAttempted) {
                backendStartAttempted = true;
                tryStartBackend();
            }
        });
    }

    function updateConnectionUI() {
        el.connDot.className = "conn-dot" + (connected ? " ok" : (!backendStartAttempted ? "" : " wait"));
        el.connLabel.textContent = connected ? "Connected" : "Connecting...";
    }

    function updateCapabilities() {
        if (!capabilities.captions) {
            el.captionsHint.classList.remove("hidden");
            if (el.captionsBadge) el.captionsBadge.classList.remove("hidden");
        } else {
            el.captionsHint.classList.add("hidden");
            if (el.captionsBadge) el.captionsBadge.classList.add("hidden");
        }
    }

    function tryStartBackend() {
        if (!inPremiere) {
            showAlert("Backend not running. Start with Start-OpenCut.bat or run: opencut server");
            return;
        }
        try {
            var child_process = require("child_process");
            var cmds = ["python", "python3", "py"];
            var ok = false;
            for (var i = 0; i < cmds.length && !ok; i++) {
                try {
                    var proc = child_process.spawn(cmds[i], ["-m", "opencut.server"], {
                        detached: true, stdio: "ignore", windowsHide: true
                    });
                    proc.unref(); ok = true;
                } catch (e) {}
            }
            if (!ok) showAlert("Could not auto-start backend. Run Start-OpenCut.bat manually.");
        } catch (e) {
            showAlert("Start the backend server: double-click Start-OpenCut.bat");
        }
    }

    // ================================================================
    // Project Media Scanning
    // ================================================================
    function scanProjectMedia() {
        if (!inPremiere) {
            el.clipSelect.innerHTML = '<option value="" disabled selected>Not in Premiere - use Browse</option>';
            return;
        }
        el.clipSelect.innerHTML = '<option value="" disabled selected>Scanning project...</option>';
        jsx("getProjectFolder()", function (folder) {
            if (folder && folder !== "null" && folder !== "undefined" && folder !== "") projectFolder = folder;
        });

        // First ping to check if ExtendScript host is loaded
        jsx("ocPing()", function (pong) {
            if (pong === "pong") {
                // Host is loaded, scan normally
                jsx("getAllProjectMedia()", _parseMediaResult);
            } else {
                // Host not loaded - try to reload then scan
                _tryReloadHost(function () {
                    jsx("ocPing()", function (pong2) {
                        if (pong2 === "pong") {
                            jsx("getAllProjectMedia()", _parseMediaResult);
                        } else {
                            el.clipSelect.innerHTML = '<option value="" disabled selected>ExtendScript not loaded (restart panel)</option>';
                            showAlert("Host script failed to load. Try: Window > Extensions > OpenCut to re-open.");
                        }
                    });
                });
            }
        });
    }

    function _parseMediaResult(result) {
        if (!result || result === "null" || result === "undefined") {
            el.clipSelect.innerHTML = '<option value="" disabled selected>No project open</option>';
            return;
        }

        // Check for EvalScript error (function not found)
        if (typeof result === "string" && result.indexOf("EvalScript error") !== -1) {
            _tryReloadHost(function () {
                jsx("getAllProjectMedia()", function (retryResult) {
                    _parseMediaResultInner(retryResult);
                });
            });
            return;
        }

        _parseMediaResultInner(result);
    }

    function _parseMediaResultInner(result) {
        if (!result || result === "null" || result === "undefined") {
            el.clipSelect.innerHTML = '<option value="" disabled selected>No project open</option>';
            return;
        }
        try {
            var data = JSON.parse(result);
            if (data.error) {
                el.clipSelect.innerHTML = '<option value="" disabled selected>' + esc(data.error) + '</option>';
                return;
            }
            projectMedia = data;
            if (data.length === 0) {
                el.clipSelect.innerHTML = '<option value="" disabled selected>No media in project (import files first)</option>';
                return;
            }
            var html = '<option value="" disabled selected>Select a clip (' + data.length + ' found)</option>';
            for (var i = 0; i < data.length; i++) {
                var item = data[i];
                var label = item.name;
                if (item.duration > 0) label += "  (" + fmtDur(item.duration) + ")";
                html += '<option value="' + item.path.replace(/&/g, "&amp;").replace(/"/g, "&quot;") + '" data-name="' + esc(item.name) + '">' + esc(label) + '</option>';
            }
            el.clipSelect.innerHTML = html;
            hideAlert();
        } catch (e) {
            var truncated = String(result).substring(0, 120);
            el.clipSelect.innerHTML = '<option value="" disabled selected>Scan error</option>';
            showAlert("Media scan returned unexpected data: " + truncated);
        }
    }

    /** Try to manually reload the host script if it failed to auto-load */
    function _tryReloadHost(cb) {
        if (!cs) { if (cb) cb(); return; }
        try {
            var extPath = cs.getSystemPath(SystemPath.EXTENSION);
            var hostPath = extPath + "/host/index.jsx";
            cs.evalScript('$.evalFile("' + hostPath.replace(/\\/g, "/") + '")', function () {
                if (cb) setTimeout(cb, 300);
            });
        } catch (e) {
            if (cb) cb();
        }
    }

    function useTimelineSelection() {
        if (!inPremiere) { showAlert("Not running inside Premiere Pro"); return; }
        jsx("getSelectedClips()", function (result) {
            if (!result || result === "null" || result === "undefined") { showAlert("Could not read selection."); return; }
            try {
                var data = JSON.parse(result);
                if (data.error) { showAlert(data.error === "nothing_selected" ? "No clip selected." : "Error: " + data.error); return; }
                if (data.length > 0) {
                    selectFile(data[0].path, data[0].name);
                    for (var i = 0; i < el.clipSelect.options.length; i++) {
                        if (el.clipSelect.options[i].value === data[0].path) { el.clipSelect.selectedIndex = i; break; }
                    }
                    hideAlert();
                }
            } catch (e) { showAlert("Error reading selection: " + e.message); }
        });
    }

    function browseForFile() {
        if (inPremiere) {
            jsx("browseForFile()", function (r) {
                if (r && r !== "null" && r !== "undefined") selectFile(r, r.split(/[/\\]/).pop());
            });
        } else {
            var path = prompt("Enter file path:");
            if (path) selectFile(path, path.split(/[/\\]/).pop());
        }
    }

    // ================================================================
    // File Selection
    // ================================================================
    function selectFile(path, name) {
        if (!path) return;
        path = path.replace(/^["'\s]+|["'\s]+$/g, "").trim();
        if (!path) return;
        selectedPath = path;
        selectedName = name || path.split(/[/\\]/).pop();
        el.fileNameDisplay.textContent = selectedName;
        el.fileMetaDisplay.textContent = "Loading info...";
        el.fileInfoBox.classList.remove("hidden");
        updateButtons();
        if (connected) {
            api("POST", "/info", { filepath: path }, function (err, data) {
                if (!err && data && !data.error) {
                    var meta = "";
                    if (data.duration) meta += fmtDur(data.duration);
                    if (data.video) meta += " | " + data.video.width + "x" + data.video.height + " @ " + data.video.fps.toFixed(2) + " fps";
                    if (data.audio) meta += " | " + (data.audio.sample_rate / 1000).toFixed(1) + " kHz";
                    if (meta) el.fileMetaDisplay.textContent = meta;
                } else { el.fileMetaDisplay.textContent = path; }
            });
        } else { el.fileMetaDisplay.textContent = path; }
    }

    function updateButtons() {
        var ready = connected && selectedPath && !currentJob && !batchRunning;
        el.runSilenceBtn.disabled = !ready;
        el.runFillersBtn.disabled = !ready || !capabilities.captions;
        el.runCaptionsBtn.disabled = !ready || !capabilities.captions;
        el.runFullBtn.disabled = !ready;
        el.batchBtn.disabled = !connected || batchRunning || !!currentJob;
        // Show/hide filler hint based on whisper availability
        if (capabilities.captions) {
            el.fillersHint.classList.add("hidden");
        } else {
            el.fillersHint.classList.remove("hidden");
        }
        updateToolbar();
    }

    function updateToolbar() {
        el.undoBtn.disabled = undoStack.length === 0;
        el.redoBtn.disabled = redoStack.length === 0;
        var hasSel = selectedSegIdx >= 0 && selectedSegIdx < allSegments.length;
        el.mergeBtn.disabled = !hasSel || selectedSegIdx >= allSegments.length - 1;
        el.splitBtn.disabled = !hasSel;
    }

    // ================================================================
    // SSE Job Tracking (with polling fallback)
    // ================================================================
    function trackJob(jobId, callbacks) {
        var cb = callbacks || {};
        stopTracking();

        function handleUpdate(job) {
            if (cb.onProgress) cb.onProgress(job);
            if (job.status === "complete") { stopTracking(); if (cb.onComplete) cb.onComplete(job); }
            else if (job.status === "error") { stopTracking(); if (cb.onError) cb.onError(job); }
            else if (job.status === "cancelled") { stopTracking(); if (cb.onCancelled) cb.onCancelled(job); }
        }

        if (SSE_OK) {
            try {
                var es = new EventSource(BACKEND + "/stream/" + jobId);
                activeStream = es;
                es.onmessage = function (event) {
                    try { handleUpdate(JSON.parse(event.data)); } catch (e) {}
                };
                es.onerror = function () {
                    es.close(); activeStream = null;
                    _pollFallback(jobId, handleUpdate);
                };
                return;
            } catch (e) {}
        }
        _pollFallback(jobId, handleUpdate);
    }

    function _pollFallback(jobId, handler) {
        if (pollTimer) clearInterval(pollTimer);
        pollTimer = setInterval(function () {
            api("GET", "/status/" + jobId, null, function (err, job) {
                if (!err && job) handler(job);
            });
        }, POLL_MS);
    }

    function stopTracking() {
        if (activeStream) { try { activeStream.close(); } catch (e) {} activeStream = null; }
        if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
        if (elapsedTimer) { clearInterval(elapsedTimer); elapsedTimer = null; }
    }

    // ================================================================
    // Job Execution
    // ================================================================
    function runJob(endpoint, payload) {
        if (projectFolder) payload.output_dir = projectFolder;
        currentJob = null;
        el.resultsSection.classList.add("hidden");
        el.waveformSection.classList.add("hidden");
        el.progressSection.classList.remove("hidden");
        el.progressBar.style.width = "0%";
        el.progressBar.className = "progress-fill";
        el.progressLabel.textContent = "Starting...";
        el.cancelBtn.classList.remove("hidden");
        jobStartTime = Date.now();
        updateElapsed();
        elapsedTimer = setInterval(updateElapsed, 1000);
        updateButtons();

        api("POST", endpoint, payload, function (err, data) {
            if (err || !data || data.error) {
                jobError(data ? data.error : (err ? err.message : "Unknown error"));
                return;
            }
            currentJob = data.job_id;
            trackJob(data.job_id, {
                onProgress: function (job) {
                    el.progressBar.style.width = (job.progress || 0) + "%";
                    el.progressLabel.textContent = job.message || "";
                },
                onComplete: function (job) {
                    currentJob = null;
                    el.progressBar.style.width = "100%";
                    el.progressBar.classList.add("done");
                    el.cancelBtn.classList.add("hidden");
                    showResults(job.result);
                    updateButtons();
                },
                onError: function (job) {
                    currentJob = null;
                    el.cancelBtn.classList.add("hidden");
                    jobError(job.error || "Processing failed");
                    updateButtons();
                },
                onCancelled: function () {
                    currentJob = null;
                    el.cancelBtn.classList.add("hidden");
                    el.progressBar.style.width = "0%";
                    el.progressLabel.textContent = "Cancelled";
                    updateButtons();
                }
            });
        });
    }

    function updateElapsed() {
        if (!jobStartTime) return;
        var sec = Math.floor((Date.now() - jobStartTime) / 1000);
        el.progressElapsed.textContent = Math.floor(sec / 60) + ":" + (sec % 60 < 10 ? "0" : "") + (sec % 60);
    }

    function cancelJob() {
        if (!currentJob) return;
        api("POST", "/cancel/" + currentJob, null, function () {});
    }

    function jobError(msg) {
        el.progressBar.style.width = "100%";
        el.progressBar.classList.add("err");
        el.progressLabel.textContent = "Error: " + msg;
        el.cancelBtn.classList.add("hidden");
        stopTracking();
        el.resultsSection.classList.remove("hidden");
        el.resultsCard.className = "results-card fail";
        el.resultsTitle.className = "results-title fail";
        el.resultsTitle.textContent = "Error";
        el.resultsStats.innerHTML = '<span class="stat-k" style="grid-column:1/-1;color:var(--error)">' + esc(msg) + '</span>';
        el.resultsPath.textContent = "";
        el.editTimelineBtn.style.display = "none";
        el.importBtn.style.display = "none";
        el.importCaptionsBtn.style.display = "none";
        el.newJobBtn.style.display = "";
        currentJob = null;
        updateButtons();
    }

    // ================================================================
    // Results Display
    // ================================================================
    function showResults(result) {
        if (!result) return;
        el.resultsSection.classList.remove("hidden");
        el.resultsCard.className = "results-card ok";
        el.resultsTitle.className = "results-title ok";

        lastXmlPath = result.xml_path || "";

        // Track caption output path (from /captions or /full-edit with captions)
        var outPath = result.output_path || "";
        if (outPath && /\.(srt|vtt)$/i.test(outPath)) {
            lastCaptionPath = outPath;
        } else if (result.srt_path) {
            lastCaptionPath = result.srt_path;
        } else {
            lastCaptionPath = "";
        }

        // Track styled caption overlay path
        lastOverlayPath = result.overlay_path || "";

        var segsRaw = result.segments_data || [];
        allSegments = [];
        for (var i = 0; i < segsRaw.length; i++) {
            allSegments.push({ start: segsRaw[i].start, end: segsRaw[i].end, enabled: true });
        }
        segmentsModified = false;
        selectedSegIdx = -1;
        undoStack = [];
        redoStack = [];

        renderResultsStats(result);

        var outFile = lastXmlPath || result.overlay_path || result.output_path || result.srt_path || "";
        el.resultsPath.textContent = outFile;
        el.resultsPath.title = outFile;

        el.editTimelineBtn.style.display = (allSegments.length > 0 && inPremiere) ? "" : "none";
        el.importBtn.style.display = lastXmlPath ? "" : "none";
        el.importCaptionsBtn.style.display = ((lastCaptionPath || lastOverlayPath) && inPremiere) ? "" : "none";
        // Update button text based on what type of caption
        if (lastOverlayPath) {
            el.importCaptionsBtn.textContent = "Import Overlay";
        } else {
            el.importCaptionsBtn.textContent = "Import Captions";
        }
        el.newJobBtn.style.display = "";

        if (allSegments.length > 0 && selectedPath) {
            fetchWaveform();
        } else {
            el.waveformSection.classList.add("hidden");
        }
    }

    function renderResultsStats(result) {
        var html = "";
        if (result && result.summary) {
            el.resultsTitle.textContent = result.summary.reduction_percent.toFixed(0) + "% shorter";
            html += stat("Original", fmtDur(result.summary.original_duration));
            html += stat("Kept", fmtDur(result.summary.kept_duration));
            html += stat("Removed", fmtDur(result.summary.removed_duration));
            html += stat("Segments", getEnabledSegments().length + " / " + allSegments.length);
        } else if (result) {
            el.resultsTitle.textContent = "Processing complete";
        }
        if (result && result.zoom_events) html += stat("Zoom points", result.zoom_events);
        if (result && result.caption_segments) html += stat("Captions", result.caption_segments + " segments");
        if (result && result.language) html += stat("Language", result.language);
        if (result && result.words) html += stat("Words", result.words);
        // Styled overlay stats
        if (result && result.overlay_path) {
            if (result.style) html += stat("Style", result.style.replace(/_/g, " "));
            if (result.action_words_found) html += stat("Action words", result.action_words_found);
            if (result.frames_rendered) html += stat("Frames", result.frames_rendered);
        }
        // Filler word stats
        if (result && result.filler_stats) {
            var fs = result.filler_stats;
            html += stat("Fillers found", fs.total_fillers + " (" + fs.filler_percentage + "% of words)");
            html += stat("Filler time", fs.total_filler_time.toFixed(1) + "s removed");
            // Show top filler breakdown
            if (fs.breakdown && fs.breakdown.length > 0) {
                var topFillers = [];
                for (var fi = 0; fi < Math.min(fs.breakdown.length, 5); fi++) {
                    var fb = fs.breakdown[fi];
                    topFillers.push('"' + fb.word + '" x' + fb.count);
                }
                html += stat("Top fillers", topFillers.join(", "));
            }
        }
        if (jobStartTime) {
            var elapsed = Math.floor((Date.now() - jobStartTime) / 1000);
            html += stat("Processed in", Math.floor(elapsed / 60) + "m " + (elapsed % 60) + "s");
        }
        el.resultsStats.innerHTML = html;
    }

    function refreshResultsStatsFromSegments() {
        if (!allSegments.length) return;
        var enabled = getEnabledSegments();
        var kept = 0;
        for (var i = 0; i < enabled.length; i++) kept += enabled[i].end - enabled[i].start;
        el.resultsTitle.textContent = enabled.length + " of " + allSegments.length + " segments kept";
        var html = "";
        html += stat("Kept duration", fmtDur(kept));
        html += stat("Segments", enabled.length + " / " + allSegments.length);
        if (segmentsModified) html += '<span class="stat-k" style="grid-column:1/-1;color:var(--warning);font-size:9px;margin-top:4px">Modified - re-export to apply</span>';
        el.resultsStats.innerHTML = html;
    }

    function stat(k, v) {
        return '<span class="stat-k">' + esc(k) + '</span><span class="stat-v">' + esc(String(v)) + '</span>';
    }

    function getEnabledSegments() {
        var out = [];
        for (var i = 0; i < allSegments.length; i++) {
            if (allSegments[i].enabled) out.push({ start: allSegments[i].start, end: allSegments[i].end });
        }
        return out;
    }

    function resetForNewJob() {
        el.resultsSection.classList.add("hidden");
        el.waveformSection.classList.add("hidden");
        el.progressSection.classList.add("hidden");
        el.progressBar.style.width = "0%";
        el.progressBar.className = "progress-fill";
        el.progressLabel.textContent = "";
        el.progressElapsed.textContent = "";
        lastXmlPath = "";
        lastCaptionPath = "";
        lastOverlayPath = "";
        allSegments = [];
        waveformPeaks = null;
        segmentsModified = false;
        selectedSegIdx = -1;
        undoStack = [];
        redoStack = [];
        currentJob = null;
        jobStartTime = 0;
        updateButtons();
    }

    // ================================================================
    // Undo / Redo
    // ================================================================
    function pushUndo() {
        undoStack.push(JSON.stringify(allSegments));
        if (undoStack.length > UNDO_MAX) undoStack.shift();
        redoStack = [];
        updateToolbar();
    }

    function doUndo() {
        if (undoStack.length === 0) return;
        redoStack.push(JSON.stringify(allSegments));
        allSegments = JSON.parse(undoStack.pop());
        segmentsModified = true;
        afterSegmentChange();
    }

    function doRedo() {
        if (redoStack.length === 0) return;
        undoStack.push(JSON.stringify(allSegments));
        allSegments = JSON.parse(redoStack.pop());
        segmentsModified = true;
        afterSegmentChange();
    }

    function afterSegmentChange() {
        if (selectedSegIdx >= allSegments.length) selectedSegIdx = allSegments.length - 1;
        renderWaveform();
        buildSegmentList();
        refreshResultsStatsFromSegments();
        updateToolbar();
    }

    // ================================================================
    // Waveform Visualization
    // ================================================================
    function fetchWaveform() {
        el.waveformSection.classList.remove("hidden");
        el.waveformLoading.classList.remove("hidden");
        el.waveformHint.textContent = "Loading...";
        var numPeaks = Math.min(Math.max(200, el.waveformWrap.clientWidth || 300), 600);

        api("POST", "/waveform", { filepath: selectedPath, peaks: numPeaks }, function (err, data) {
            el.waveformLoading.classList.add("hidden");
            if (err || !data || data.error) {
                el.waveformHint.textContent = "Could not load waveform";
                return;
            }
            waveformPeaks = data.peaks || [];
            waveformDuration = data.duration || 0;
            waveformMaxRms = data.max_rms || 1;
            el.waveformHint.textContent = "Click to toggle | Space to preview";
            renderWaveform();
            buildSegmentList();
        }, 60000);
    }

    function renderWaveform() {
        var canvas = el.waveformCanvas;
        var ctx = canvas.getContext("2d");
        var wrap = el.waveformWrap;
        var w = wrap.clientWidth || 300;
        var h = 72;
        var dpr = window.devicePixelRatio || 1;

        canvas.width = w * dpr;
        canvas.height = h * dpr;
        canvas.style.width = w + "px";
        canvas.style.height = h + "px";
        ctx.scale(dpr, dpr);

        ctx.fillStyle = "#111114";
        ctx.fillRect(0, 0, w, h);
        if (!waveformPeaks || waveformPeaks.length === 0) return;

        var barW = w / waveformPeaks.length;
        var centerY = h / 2;

        for (var i = 0; i < waveformPeaks.length; i++) {
            var pk = waveformPeaks[i];
            var norm = waveformMaxRms > 0 ? pk.r / waveformMaxRms : 0;
            var barH = Math.max(1, norm * (h * 0.88));
            var x = i * barW;
            var t = pk.t;

            var inSeg = false, segEnabled = true, segIdx = -1;
            for (var s = 0; s < allSegments.length; s++) {
                if (t >= allSegments[s].start && t < allSegments[s].end) {
                    inSeg = true;
                    segEnabled = allSegments[s].enabled;
                    segIdx = s;
                    break;
                }
            }

            if (inSeg && segEnabled && segIdx === selectedSegIdx)
                ctx.fillStyle = "#80d8ff";
            else if (inSeg && segEnabled)
                ctx.fillStyle = "#4fc3f7";
            else if (inSeg && !segEnabled)
                ctx.fillStyle = "#333340";
            else
                ctx.fillStyle = "#1e1012";

            ctx.fillRect(x, centerY - barH / 2, Math.max(barW - 0.3, 0.5), barH);
        }

        // Segment boundary lines
        if (waveformDuration > 0) {
            ctx.lineWidth = 1;
            for (var s2 = 0; s2 < allSegments.length; s2++) {
                var seg = allSegments[s2];
                var sx = (seg.start / waveformDuration) * w;
                ctx.strokeStyle = seg.enabled ? "rgba(79,195,247,0.25)" : "rgba(80,80,80,0.25)";
                ctx.beginPath(); ctx.moveTo(sx, 0); ctx.lineTo(sx, h); ctx.stroke();
            }
        }
    }

    function buildSegmentList() {
        var html = "";
        for (var i = 0; i < allSegments.length; i++) {
            var seg = allSegments[i];
            var dur = (seg.end - seg.start).toFixed(2);
            var cls = seg.enabled ? "" : " disabled";
            if (i === selectedSegIdx) cls += " selected";
            html += '<label class="seg-item' + cls + '" data-idx="' + i + '">' +
                '<input type="checkbox"' + (seg.enabled ? " checked" : "") + ' data-idx="' + i + '">' +
                '<span class="seg-time">' + fmtTime(seg.start) + ' - ' + fmtTime(seg.end) + '</span>' +
                '<span class="seg-dur">' + dur + 's</span>' +
                '</label>';
        }
        el.segmentList.innerHTML = html;
        el.segmentCount.textContent = getEnabledSegments().length + " / " + allSegments.length + " segments";
    }

    // ================================================================
    // Segment Editing
    // ================================================================
    function toggleSegment(index) {
        if (index < 0 || index >= allSegments.length) return;
        pushUndo();
        allSegments[index].enabled = !allSegments[index].enabled;
        segmentsModified = true;
        afterSegmentChange();
    }

    function setAllSegments(enabled) {
        pushUndo();
        for (var i = 0; i < allSegments.length; i++) allSegments[i].enabled = enabled;
        segmentsModified = true;
        afterSegmentChange();
    }

    function selectSegment(index) {
        selectedSegIdx = (index === selectedSegIdx) ? -1 : index;
        renderWaveform();
        buildSegmentList();
        updateToolbar();
        // Scroll into view
        if (selectedSegIdx >= 0) {
            var item = el.segmentList.querySelector('[data-idx="' + selectedSegIdx + '"]');
            if (item) item.scrollIntoView({ block: "nearest" });
        }
    }

    function mergeWithNext() {
        if (selectedSegIdx < 0 || selectedSegIdx >= allSegments.length - 1) return;
        pushUndo();
        var a = allSegments[selectedSegIdx];
        var b = allSegments[selectedSegIdx + 1];
        // Merge: keep from a.start to b.end, inherit enabled from either
        allSegments[selectedSegIdx] = {
            start: a.start,
            end: b.end,
            enabled: a.enabled || b.enabled
        };
        allSegments.splice(selectedSegIdx + 1, 1);
        segmentsModified = true;
        afterSegmentChange();
    }

    function splitSegment() {
        if (selectedSegIdx < 0 || selectedSegIdx >= allSegments.length) return;
        var seg = allSegments[selectedSegIdx];
        var mid = (seg.start + seg.end) / 2;
        if (seg.end - seg.start < 0.1) return; // too short to split
        pushUndo();
        var first = { start: seg.start, end: mid, enabled: seg.enabled };
        var second = { start: mid, end: seg.end, enabled: seg.enabled };
        allSegments.splice(selectedSegIdx, 1, first, second);
        segmentsModified = true;
        afterSegmentChange();
    }

    // ================================================================
    // Audio Preview
    // ================================================================
    function previewSegment(index) {
        if (index < 0 || index >= allSegments.length) return;
        if (!connected || !selectedPath) return;

        var seg = allSegments[index];
        var centerTime = seg.start;
        var duration = Math.min(3, seg.end - seg.start + 0.5);

        el.previewIndicator.classList.remove("hidden");

        api("POST", "/preview-audio", {
            filepath: selectedPath,
            time: centerTime + duration / 2,
            duration: duration
        }, function (err, data) {
            if (err || !data || data.error) {
                el.previewIndicator.classList.add("hidden");
                return;
            }
            el.previewAudio.src = BACKEND + data.url;
            el.previewAudio.play().catch(function () {});
        }, 15000);

        el.previewAudio.onended = function () {
            el.previewIndicator.classList.add("hidden");
        };
        el.previewAudio.onerror = function () {
            el.previewIndicator.classList.add("hidden");
        };
        // Auto-hide after timeout
        setTimeout(function () {
            el.previewIndicator.classList.add("hidden");
        }, 5000);
    }

    // ================================================================
    // Export (Video + Audio-Only)
    // ================================================================
    function exportVideo() { _doExport(false); }
    function exportAudio() { _doExport(true); }

    function _doExport(audioOnly) {
        var segments = getEnabledSegments();
        if (!connected || !selectedPath || segments.length === 0) {
            showAlert("No segments available to export.");
            return;
        }

        var payload = {
            filepath: selectedPath,
            segments: segments,
        };
        if (projectFolder) payload.output_dir = projectFolder;

        if (audioOnly) {
            payload.audio_only = true;
            payload.audio_format = el.audioExportFormat.value;
        } else {
            payload.quality = el.exportQuality.value;
            payload.output_format = el.exportFormat.value;
        }

        var btn = audioOnly ? el.exportAudioBtn : el.exportVideoBtn;
        var label = audioOnly ? "Audio" : "Video";
        btn.disabled = true;
        btn.textContent = "Exporting...";
        el.progressSection.classList.remove("hidden");
        el.progressBar.style.width = "0%";
        el.progressBar.className = "progress-fill";
        el.progressLabel.textContent = "Starting " + label.toLowerCase() + " export...";
        el.cancelBtn.classList.remove("hidden");
        jobStartTime = Date.now();
        updateElapsed();
        elapsedTimer = setInterval(updateElapsed, 1000);

        api("POST", "/export-video", payload, function (err, data) {
            if (err || !data || data.error) {
                _resetExBtn(btn, label);
                el.progressSection.classList.add("hidden");
                stopTracking();
                showAlert("Export failed: " + (data ? data.error : (err ? err.message : "Unknown error")));
                return;
            }
            currentJob = data.job_id;
            trackJob(data.job_id, {
                onProgress: function (job) {
                    el.progressBar.style.width = (job.progress || 0) + "%";
                    el.progressLabel.textContent = job.message || "";
                },
                onComplete: function (job) {
                    currentJob = null;
                    el.progressBar.style.width = "100%";
                    el.progressBar.classList.add("done");
                    el.cancelBtn.classList.add("hidden");
                    _resetExBtn(btn, label);
                    updateButtons();
                    if (job.result) {
                        var r = job.result;
                        el.resultsTitle.textContent = label + " exported!";
                        var h = "";
                        h += stat("Output", r.output_path ? r.output_path.split(/[/\\]/).pop() : "");
                        if (r.duration) h += stat("Duration", fmtDur(r.duration));
                        if (r.file_size_mb) h += stat("File size", r.file_size_mb + " MB");
                        if (r.segments) h += stat("Segments", r.segments);
                        if (jobStartTime) {
                            var elapsed = Math.floor((Date.now() - jobStartTime) / 1000);
                            h += stat("Rendered in", Math.floor(elapsed / 60) + "m " + (elapsed % 60) + "s");
                        }
                        el.resultsStats.innerHTML = h;
                        el.resultsPath.textContent = r.output_path || "";
                    }
                },
                onError: function (job) {
                    currentJob = null; _resetExBtn(btn, label);
                    el.cancelBtn.classList.add("hidden");
                    el.progressBar.style.width = "100%";
                    el.progressBar.classList.add("err");
                    el.progressLabel.textContent = "Export error: " + (job.error || "Failed");
                    updateButtons();
                },
                onCancelled: function () {
                    currentJob = null; _resetExBtn(btn, label);
                    el.cancelBtn.classList.add("hidden");
                    el.progressBar.style.width = "0%";
                    el.progressLabel.textContent = "Cancelled";
                    updateButtons();
                }
            });
        });
    }

    function _resetExBtn(btn, label) {
        btn.disabled = false;
        btn.textContent = "Export " + label;
    }

    // ================================================================
    // Edit Timeline / Import XML
    // ================================================================
    function editTimeline() {
        var segments = getEnabledSegments();
        if (!segments.length || !selectedPath) return;
        if (!inPremiere) { showAlert("Not in Premiere Pro."); return; }

        var segJson = JSON.stringify(segments).replace(/\\/g, "\\\\").replace(/'/g, "\\'");
        var pathEsc = selectedPath.replace(/\\/g, "\\\\").replace(/'/g, "\\'");

        el.editTimelineBtn.disabled = true;
        el.editTimelineBtn.textContent = "Creating sequence...";

        jsx("applyEditsToTimeline('" + segJson + "', '" + pathEsc + "')", function (result) {
            el.editTimelineBtn.disabled = false;
            el.editTimelineBtn.textContent = "Edit in Timeline";
            try {
                var data = JSON.parse(result);
                if (data.success) {
                    el.resultsTitle.textContent = data.sequenceName || "Timeline created!";
                    el.editTimelineBtn.style.display = "none";
                    el.importBtn.style.display = "none";
                    hideAlert();
                } else {
                    showAlert("Direct edit failed: " + (data.error || "Unknown") + ". Try Import XML.");
                }
            } catch (e) {
                showAlert("Direct edit failed: " + e.message);
            }
        });
    }

    function importXml() {
        if (!lastXmlPath) return;
        if (!inPremiere) { showAlert("Not in Premiere Pro. XML saved to: " + lastXmlPath); return; }

        function doImport(xmlPath) {
            var escaped = xmlPath.replace(/\\/g, "\\\\").replace(/'/g, "\\'");
            jsx("importAndOpenXml('" + escaped + "')", function (result) {
                try {
                    var data = JSON.parse(result);
                    if (data.success) {
                        el.resultsTitle.textContent = "Imported!" + (data.sequenceName ? " " + data.sequenceName : "");
                        el.importBtn.style.display = "none";
                    } else { showAlert("Import failed: " + (data.error || "Unknown")); }
                } catch (e) {
                    showAlert("Import failed. Try File > Import and select: " + xmlPath);
                }
            });
        }

        if (segmentsModified) {
            el.importBtn.disabled = true;
            el.importBtn.textContent = "Regenerating...";
            api("POST", "/regenerate", { filepath: selectedPath, segments: getEnabledSegments(), output_dir: projectFolder || "" }, function (err, data) {
                el.importBtn.disabled = false;
                el.importBtn.textContent = "Import XML";
                if (!err && data && data.xml_path) {
                    lastXmlPath = data.xml_path;
                    segmentsModified = false;
                    doImport(data.xml_path);
                } else {
                    showAlert("XML regeneration failed. Importing original.");
                    doImport(lastXmlPath);
                }
            });
        } else {
            doImport(lastXmlPath);
        }
    }

    // ================================================================
    // Import Captions to Timeline
    // ================================================================
    function importCaptions() {
        // Determine what to import: overlay video or SRT/VTT
        var importPath = lastOverlayPath || lastCaptionPath;
        var isOverlay = !!lastOverlayPath;

        if (!importPath) return;
        if (!inPremiere) {
            showAlert("Not in Premiere Pro. File saved to: " + importPath);
            return;
        }

        var btnLabel = isOverlay ? "Import Overlay" : "Import Captions";
        el.importCaptionsBtn.disabled = true;
        el.importCaptionsBtn.textContent = "Importing...";

        var escaped = importPath.replace(/\\/g, "\\\\").replace(/'/g, "\\'");
        var jsxFunc = isOverlay
            ? "importCaptionOverlay('" + escaped + "')"
            : "importCaptions('" + escaped + "')";

        jsx(jsxFunc, function (result) {
            el.importCaptionsBtn.disabled = false;

            if (!result || result === "null" || result === "undefined") {
                el.importCaptionsBtn.textContent = btnLabel;
                showAlert("Could not import. Try File > Import manually.");
                return;
            }

            try {
                var data = JSON.parse(result);
                if (data.success) {
                    el.importCaptionsBtn.textContent = "Imported!";
                    el.importCaptionsBtn.style.background = "#4caf50";
                    var msg = data.message || "";
                    if (data.addedToTimeline || data.trackIndex) {
                        hideAlert();
                        if (msg) showAlert(msg);
                    } else {
                        showAlert(msg || "Imported to project. Drag onto timeline to use.");
                    }
                    setTimeout(function () {
                        el.importCaptionsBtn.textContent = btnLabel;
                        el.importCaptionsBtn.style.background = "";
                    }, 3000);
                } else {
                    el.importCaptionsBtn.textContent = btnLabel;
                    showAlert(data.error || "Import failed.");
                }
            } catch (e) {
                el.importCaptionsBtn.textContent = btnLabel;
                showAlert("Import error. File saved to: " + importPath);
            }
        });
    }

    // ================================================================
    // Install Whisper
    // ================================================================
    function installWhisper() {
        if (!connected) { showAlert("Backend not connected."); return; }
        el.installWhisperBtn.disabled = true;
        el.installWhisperBtn.textContent = "Installing...";

        api("POST", "/install-whisper", { backend: "faster-whisper" }, function (err, data) {
            if (err || !data || data.error) {
                el.installWhisperBtn.disabled = false;
                el.installWhisperBtn.textContent = "Install Whisper Now";
                showAlert("Failed: " + (data ? data.error : (err ? err.message : "Unknown error")));
                return;
            }
            trackJob(data.job_id, {
                onProgress: function (job) {
                    // Truncate long messages to fit button
                    var msg = job.message || "Installing...";
                    if (msg.length > 40) msg = msg.substring(0, 37) + "...";
                    el.installWhisperBtn.textContent = msg;
                },
                onComplete: function (job) {
                    el.installWhisperBtn.textContent = "Installed!";
                    el.installWhisperBtn.style.background = "#4caf50";
                    // Check if it used a fallback backend
                    if (job.result && job.result.backend && job.result.backend !== "faster-whisper") {
                        showAlert("Installed " + job.result.backend + " as fallback (faster-whisper unavailable)");
                    }
                    api("GET", "/capabilities", null, function (e2, caps) {
                        if (!e2 && caps) { capabilities = caps; updateCapabilities(); updateButtons(); }
                    });
                    setTimeout(function () {
                        el.captionsHint.classList.add("hidden");
                        if (el.captionsBadge) el.captionsBadge.classList.add("hidden");
                    }, 2000);
                },
                onError: function (job) {
                    el.installWhisperBtn.disabled = false;
                    el.installWhisperBtn.textContent = "Retry Install";
                    el.installWhisperBtn.style.background = "";
                    // Show truncated error in alert (the full error can be very long)
                    var errMsg = job.error || "Unknown error";
                    if (errMsg.length > 300) errMsg = errMsg.substring(0, 300) + "...";
                    showAlert(errMsg);
                }
            });
        }, 300000); // 5 minute timeout for install
    }

    // ================================================================
    // Batch Processing
    // ================================================================
    function startBatch() {
        var clips = [];
        for (var i = 0; i < projectMedia.length; i++) {
            if (projectMedia[i].path) clips.push(projectMedia[i]);
        }
        if (clips.length === 0) { showAlert("No clips in project to process."); return; }

        batchRunning = true; batchCancel = false; batchResults = [];
        updateButtons();

        el.actionSection.classList.add("hidden");
        el.resultsSection.classList.add("hidden");
        el.waveformSection.classList.add("hidden");
        el.progressSection.classList.add("hidden");
        el.batchSection.classList.remove("hidden");
        el.closeBatchBtn.style.display = "none";
        el.cancelBatchBtn.style.display = "";

        var html = "";
        for (var j = 0; j < clips.length; j++) {
            html += '<div class="batch-item pending" id="batch-' + j + '">' +
                '<span class="batch-icon">-</span>' +
                '<span class="batch-name">' + esc(clips[j].name) + '</span>' +
                '<span class="batch-info"></span></div>';
        }
        el.batchList.innerHTML = html;
        el.batchCounter.textContent = "0/" + clips.length;
        el.batchProgressBar.style.width = "0%";
        el.batchProgressBar.className = "progress-fill";

        var idx = 0;
        function processNext() {
            if (batchCancel || idx >= clips.length) { batchDone(clips.length); return; }
            var clip = clips[idx];
            var row = $("batch-" + idx);
            if (row) { row.className = "batch-item active"; row.querySelector(".batch-icon").textContent = "~"; }
            el.batchStatusLabel.textContent = "Processing: " + clip.name;
            el.batchCounter.textContent = idx + "/" + clips.length;
            el.batchProgressBar.style.width = ((idx / clips.length) * 100).toFixed(0) + "%";

            var payload = { filepath: clip.path };
            var preset = el.silencePreset.value;
            if (preset) payload.preset = preset;
            else {
                payload.threshold = parseFloat(el.threshold.value);
                payload.min_duration = parseFloat(el.minDuration.value);
                payload.padding_before = parseFloat(el.padBefore.value);
                payload.padding_after = parseFloat(el.padAfter.value);
            }
            if (projectFolder) payload.output_dir = projectFolder;

            api("POST", "/silence", payload, function (err, data) {
                if (err || !data || data.error) {
                    batchResults.push({ name: clip.name, error: true });
                    if (row) { row.className = "batch-item error"; row.querySelector(".batch-icon").textContent = "x"; row.querySelector(".batch-info").textContent = "Failed"; }
                    idx++; processNext();
                    return;
                }
                trackJob(data.job_id, {
                    onProgress: function (job) { if (row) row.querySelector(".batch-info").textContent = (job.progress || 0) + "%"; },
                    onComplete: function (job) {
                        var info = "";
                        if (job.result && job.result.summary) info = job.result.summary.reduction_percent.toFixed(0) + "% shorter";
                        batchResults.push({ name: clip.name, info: info });
                        if (row) { row.className = "batch-item done"; row.querySelector(".batch-icon").textContent = "ok"; row.querySelector(".batch-info").textContent = info; }
                        idx++; processNext();
                    },
                    onError: function () {
                        batchResults.push({ name: clip.name, error: true });
                        if (row) { row.className = "batch-item error"; row.querySelector(".batch-icon").textContent = "x"; row.querySelector(".batch-info").textContent = "Error"; }
                        idx++; processNext();
                    },
                    onCancelled: function () { batchCancel = true; idx++; processNext(); }
                });
            });
        }
        processNext();
    }

    function batchDone(total) {
        batchRunning = false; updateButtons();
        el.cancelBatchBtn.style.display = "none";
        el.closeBatchBtn.style.display = "";
        var ok = 0, fail = 0;
        for (var i = 0; i < batchResults.length; i++) { if (batchResults[i].error) fail++; else ok++; }
        el.batchProgressBar.style.width = "100%";
        el.batchProgressBar.className = "progress-fill" + (fail > 0 ? "" : " done");
        el.batchCounter.textContent = batchResults.length + "/" + total;
        el.batchStatusLabel.textContent = batchCancel
            ? "Cancelled. " + ok + " done, " + fail + " failed."
            : "Done! " + ok + " completed" + (fail > 0 ? ", " + fail + " failed" : "");
    }

    function cancelBatch() {
        batchCancel = true;
        if (currentJob) api("POST", "/cancel/" + currentJob, null, function () {});
    }

    function closeBatch() {
        el.batchSection.classList.add("hidden");
        el.actionSection.classList.remove("hidden");
        batchRunning = false;
        updateButtons();
    }

    // ================================================================
    // Keyboard Shortcuts
    // ================================================================
    function handleKeyDown(e) {
        // Ignore if typing in an input/select
        var tag = (e.target.tagName || "").toLowerCase();
        if (tag === "input" || tag === "textarea" || tag === "select") return;

        var key = e.key;
        var ctrl = e.ctrlKey || e.metaKey;

        if (ctrl && key === "z") { e.preventDefault(); doUndo(); return; }
        if (ctrl && (key === "y" || (e.shiftKey && key === "Z"))) { e.preventDefault(); doRedo(); return; }

        if (key === "Escape") {
            if (!el.shortcutsOverlay.classList.contains("hidden")) {
                el.shortcutsOverlay.classList.add("hidden");
            } else {
                selectedSegIdx = -1;
                renderWaveform();
                buildSegmentList();
                updateToolbar();
            }
            return;
        }

        if (key === "?" || key === "/") {
            e.preventDefault();
            el.shortcutsOverlay.classList.toggle("hidden");
            return;
        }

        // The rest only apply when waveform section is visible
        if (el.waveformSection.classList.contains("hidden")) return;

        if (key === " ") {
            e.preventDefault();
            if (selectedSegIdx >= 0) previewSegment(selectedSegIdx);
            return;
        }
        if (key === "e" || key === "E") {
            if (selectedSegIdx >= 0) toggleSegment(selectedSegIdx);
            return;
        }
        if (key === "a" || key === "A") {
            if (allSegments.length > 0) selectSegment(Math.max(0, selectedSegIdx - 1));
            return;
        }
        if (key === "d" || key === "D") {
            if (allSegments.length > 0) selectSegment(Math.min(allSegments.length - 1, selectedSegIdx + 1));
            return;
        }
        if (key === "m" || key === "M") { mergeWithNext(); return; }
        if (key === "s" || key === "S") {
            if (!ctrl) { splitSegment(); return; }
        }
    }

    // ================================================================
    // Shortcuts Overlay
    // ================================================================
    function toggleShortcuts() {
        el.shortcutsOverlay.classList.toggle("hidden");
    }

    // ================================================================
    // Alerts
    // ================================================================
    function showAlert(msg) { el.alertText.textContent = msg; el.alertBanner.classList.remove("hidden"); }
    function hideAlert() { el.alertBanner.classList.add("hidden"); }

    // ================================================================
    // Utilities
    // ================================================================
    function fmtDur(sec) {
        if (!sec || sec <= 0) return "0:00";
        var h = Math.floor(sec / 3600);
        var m = Math.floor((sec % 3600) / 60);
        var s = (sec % 60).toFixed(1);
        if (h > 0) return h + ":" + (m < 10 ? "0" : "") + m + ":" + (parseFloat(s) < 10 ? "0" : "") + s;
        return m + ":" + (parseFloat(s) < 10 ? "0" : "") + s;
    }

    function fmtTime(sec) {
        var m = Math.floor(sec / 60);
        var s = (sec % 60).toFixed(2);
        return m + ":" + (parseFloat(s) < 10 ? "0" : "") + s;
    }

    function esc(str) {
        return String(str).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }

    // ================================================================
    // Event Binding
    // ================================================================
    function bindEvents() {
        // Action tabs
        var tabs = document.querySelectorAll(".tab");
        for (var i = 0; i < tabs.length; i++) {
            tabs[i].addEventListener("click", function () {
                for (var j = 0; j < tabs.length; j++) tabs[j].classList.remove("active");
                this.classList.add("active");
                var panels = document.querySelectorAll(".tab-panel");
                for (var k = 0; k < panels.length; k++) panels[k].classList.remove("active");
                $("panel-" + this.getAttribute("data-tab")).classList.add("active");
            });
        }

        // Export tabs
        var etabs = document.querySelectorAll(".etab");
        for (var ei = 0; ei < etabs.length; ei++) {
            etabs[ei].addEventListener("click", function () {
                for (var j = 0; j < etabs.length; j++) etabs[j].classList.remove("active");
                this.classList.add("active");
                var ep = document.querySelectorAll(".export-panel");
                for (var k = 0; k < ep.length; k++) ep[k].classList.remove("active");
                $("epanel-" + this.getAttribute("data-etype")).classList.add("active");
            });
        }

        el.alertDismiss.addEventListener("click", hideAlert);
        el.clipSelect.addEventListener("change", function () {
            var opt = this.options[this.selectedIndex];
            if (opt && opt.value) selectFile(opt.value, opt.getAttribute("data-name") || opt.textContent);
        });

        el.refreshClipsBtn.addEventListener("click", scanProjectMedia);
        el.useSelectionBtn.addEventListener("click", useTimelineSelection);
        el.browseFileBtn.addEventListener("click", browseForFile);

        el.threshold.addEventListener("input", function () { el.thresholdVal.textContent = this.value + " dB"; });
        el.minDuration.addEventListener("input", function () { el.minDurationVal.textContent = parseFloat(this.value).toFixed(1) + "s"; });
        el.silencePreset.addEventListener("change", function () {
            if (this.value === "") el.customSilenceSettings.classList.remove("hidden");
            else el.customSilenceSettings.classList.add("hidden");
        });

        el.runSilenceBtn.addEventListener("click", function () {
            var payload = { filepath: selectedPath };
            var preset = el.silencePreset.value;
            if (preset) payload.preset = preset;
            else {
                payload.threshold = parseFloat(el.threshold.value);
                payload.min_duration = parseFloat(el.minDuration.value);
                payload.padding_before = parseFloat(el.padBefore.value);
                payload.padding_after = parseFloat(el.padAfter.value);
            }
            runJob("/silence", payload);
        });

        el.runCaptionsBtn.addEventListener("click", function () {
            if (el.captionMode.value === "styled") {
                // Styled overlay mode
                var actionWords = [];
                var awStr = (el.actionWordsInput.value || "").trim();
                if (awStr) {
                    var awParts = awStr.split(",");
                    for (var k = 0; k < awParts.length; k++) {
                        var aw = awParts[k].trim();
                        if (aw) actionWords.push(aw);
                    }
                }
                runJob("/styled-captions", {
                    filepath: selectedPath,
                    model: el.captionModel.value,
                    language: el.captionLang.value || null,
                    style: el.captionStyle.value,
                    auto_detect_energy: el.captionAutoAction.checked,
                    action_words: actionWords
                });
            } else {
                // SRT/VTT text file mode
                runJob("/captions", {
                    filepath: selectedPath,
                    model: el.captionModel.value,
                    language: el.captionLang.value || null,
                    format: el.captionFormat.value,
                    word_timestamps: true
                });
            }
        });

        // Caption mode toggle: show/hide styled vs SRT options
        el.captionMode.addEventListener("change", function () {
            if (this.value === "styled") {
                el.styledOptions.classList.remove("hidden");
                el.srtOptions.classList.add("hidden");
            } else {
                el.styledOptions.classList.add("hidden");
                el.srtOptions.classList.remove("hidden");
            }
        });

        // Style preview update
        el.captionStyle.addEventListener("change", updateStylePreview);
        function updateStylePreview() {
            var previewBg = el.stylePreview.querySelector(".style-preview-bg");
            // Remove all sp- classes
            var classes = previewBg.className.split(" ");
            for (var ci = classes.length - 1; ci >= 0; ci--) {
                if (classes[ci].indexOf("sp-") === 0 && classes[ci] !== "sp-word"
                    && classes[ci] !== "sp-highlight" && classes[ci] !== "sp-action") {
                    previewBg.classList.remove(classes[ci]);
                }
            }
            previewBg.className = "style-preview-bg sp-" + el.captionStyle.value;
        }
        updateStylePreview();

        el.runFillersBtn.addEventListener("click", function () {
            // Collect selected filler types
            var removeFillers = [];
            var checks = el.fillerChecks.querySelectorAll("input[data-filler]");
            for (var i = 0; i < checks.length; i++) {
                if (checks[i].checked) removeFillers.push(checks[i].getAttribute("data-filler"));
            }
            // Custom words
            var customWords = [];
            var customStr = (el.fillerCustom.value || "").trim();
            if (customStr) {
                var parts = customStr.split(",");
                for (var j = 0; j < parts.length; j++) {
                    var w = parts[j].trim();
                    if (w) customWords.push(w);
                }
            }

            runJob("/fillers", {
                filepath: selectedPath,
                model: el.fillerModel.value,
                include_context_fillers: true,
                remove_fillers: removeFillers,
                custom_words: customWords,
                remove_silence: el.fillerSilence.checked,
                silence_preset: "youtube"
            });
        });

        el.runFullBtn.addEventListener("click", function () {
            var payload = {
                filepath: selectedPath,
                preset: el.fullPreset.value,
                skip_captions: !el.fullCaptions.checked,
                skip_zoom: !el.fullZoom.checked,
                remove_fillers: el.fullFillers.checked
            };
            runJob("/full", payload);
        });

        el.editTimelineBtn.addEventListener("click", editTimeline);
        el.importBtn.addEventListener("click", importXml);
        el.importCaptionsBtn.addEventListener("click", importCaptions);
        el.cancelBtn.addEventListener("click", cancelJob);
        el.newJobBtn.addEventListener("click", resetForNewJob);
        el.installWhisperBtn.addEventListener("click", installWhisper);
        el.exportVideoBtn.addEventListener("click", exportVideo);
        el.exportAudioBtn.addEventListener("click", exportAudio);

        // Waveform click: select or toggle
        el.waveformCanvas.addEventListener("click", function (e) {
            if (!waveformPeaks || waveformDuration <= 0) return;
            var rect = el.waveformCanvas.getBoundingClientRect();
            var x = e.clientX - rect.left;
            var t = (x / rect.width) * waveformDuration;
            for (var i = 0; i < allSegments.length; i++) {
                if (t >= allSegments[i].start && t < allSegments[i].end) {
                    if (e.shiftKey) {
                        toggleSegment(i);
                    } else {
                        selectSegment(i);
                    }
                    break;
                }
            }
        });

        // Double-click waveform: preview
        el.waveformCanvas.addEventListener("dblclick", function (e) {
            if (!waveformPeaks || waveformDuration <= 0) return;
            var rect = el.waveformCanvas.getBoundingClientRect();
            var t = ((e.clientX - rect.left) / rect.width) * waveformDuration;
            for (var i = 0; i < allSegments.length; i++) {
                if (t >= allSegments[i].start && t < allSegments[i].end) {
                    previewSegment(i);
                    break;
                }
            }
        });

        // Segment list
        el.segmentList.addEventListener("change", function (e) {
            var idx = e.target.getAttribute("data-idx");
            if (idx !== null) toggleSegment(parseInt(idx, 10));
        });
        el.segmentList.addEventListener("click", function (e) {
            var item = e.target.closest(".seg-item");
            if (!item) return;
            var idx = parseInt(item.getAttribute("data-idx"), 10);
            // Only select if not clicking the checkbox
            if (e.target.tagName !== "INPUT") {
                e.preventDefault();
                selectSegment(idx);
            }
        });

        el.enableAllBtn.addEventListener("click", function () { setAllSegments(true); });
        el.disableAllBtn.addEventListener("click", function () { setAllSegments(false); });

        // Toolbar
        el.undoBtn.addEventListener("click", doUndo);
        el.redoBtn.addEventListener("click", doRedo);
        el.mergeBtn.addEventListener("click", mergeWithNext);
        el.splitBtn.addEventListener("click", splitSegment);

        // Batch
        el.batchBtn.addEventListener("click", startBatch);
        el.cancelBatchBtn.addEventListener("click", cancelBatch);
        el.closeBatchBtn.addEventListener("click", closeBatch);

        // Shortcuts
        el.shortcutsHelpBtn.addEventListener("click", toggleShortcuts);
        el.closeShortcutsBtn.addEventListener("click", toggleShortcuts);
        el.shortcutsOverlay.addEventListener("click", function (e) {
            if (e.target === el.shortcutsOverlay) toggleShortcuts();
        });

        // Global keyboard
        document.addEventListener("keydown", handleKeyDown);
    }

    // ================================================================
    // Init
    // ================================================================
    function init() {
        initDOM();
        initCSInterface();
        bindEvents();
        checkHealth();
        healthTimer = setInterval(checkHealth, HEALTH_MS);
        setTimeout(scanProjectMedia, 800);
    }

    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
    else init();
})();
