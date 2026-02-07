/* ============================================================
   OpenCut CEP Panel - Main Controller v1.0.0
   7-Tab Professional Toolkit
   ============================================================ */
(function () {
    "use strict";

    var BACKEND = "http://127.0.0.1:5679";
    var BACKEND_BASE_PORT = 5679;
    var BACKEND_MAX_PORT = 5689;
    var POLL_MS = 700;
    var HEALTH_MS = 4000;
    var SSE_OK = typeof EventSource !== "undefined";

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
    var transcriptData = null; // stored transcript for editing/export
    var voiceInstalled = false; // whether Qwen3-TTS is available
    var rembgInstalled = false;
    var rifeInstalled = false;
    var esrganInstalled = false;
    var batchFiles = [];
    var batchJobId = null;
    var batchPollTimer = null;
    var workflowSteps = [];
    var activeWatcherId = null;
    var cloneRefAudioPath = ""; // reference audio path for cloning
    var voiceProfiles = []; // cached voice profiles list

    // ---- Style Preview CSS Map (loaded from backend) ----
    var stylePreviewMap = {};

    // ---- DOM ----
    var el = {};
    function $(id) { return document.getElementById(id); }

    function initDOM() {
        // Header
        el.connDot = $("connDot");
        el.connLabel = $("connLabel");
        el.alertBanner = $("alertBanner");
        el.alertText = $("alertText");
        el.alertDismiss = $("alertDismiss");

        // Clip
        el.clipSelect = $("clipSelect");
        el.fileInfoBox = $("fileInfoBox");
        el.fileNameDisplay = $("fileNameDisplay");
        el.fileMetaDisplay = $("fileMetaDisplay");
        el.refreshClipsBtn = $("refreshClipsBtn");
        el.useSelectionBtn = $("useSelectionBtn");
        el.browseFileBtn = $("browseFileBtn");

        // Cut tab
        el.silencePreset = $("silencePreset");
        el.customSilenceSettings = $("customSilenceSettings");
        el.threshold = $("threshold");
        el.thresholdVal = $("thresholdVal");
        el.minDuration = $("minDuration");
        el.minDurationVal = $("minDurationVal");
        el.padBefore = $("padBefore");
        el.padAfter = $("padAfter");
        el.runSilenceBtn = $("runSilenceBtn");
        el.fillerModel = $("fillerModel");
        el.fillerChecks = $("fillerChecks");
        el.fillerCustom = $("fillerCustom");
        el.fillerSilence = $("fillerSilence");
        el.fillersHint = $("fillersHint");
        el.runFillersBtn = $("runFillersBtn");
        el.fullPreset = $("fullPreset");
        el.fullZoom = $("fullZoom");
        el.fullCaptions = $("fullCaptions");
        el.fullFillers = $("fullFillers");
        el.runFullBtn = $("runFullBtn");

        // Captions tab
        el.captionModel = $("captionModel");
        el.captionLang = $("captionLang");
        el.captionStyle = $("captionStyle");
        el.stylePreview = $("stylePreview");
        el.captionWordHighlight = $("captionWordHighlight");
        el.captionAutoAction = $("captionAutoAction");
        el.captionAutoEmoji = $("captionAutoEmoji");
        el.actionWordsInput = $("actionWordsInput");
        el.runStyledCaptionsBtn = $("runStyledCaptionsBtn");
        el.captionsHint = $("captionsHint");
        el.installWhisperBtn = $("installWhisperBtn");
        el.subModel = $("subModel");
        el.subLang = $("subLang");
        el.subFormat = $("subFormat");
        el.runSubtitleBtn = $("runSubtitleBtn");
        el.transcriptModel = $("transcriptModel");
        el.runTranscriptBtn = $("runTranscriptBtn");
        el.transcriptEditor = $("transcriptEditor");
        el.transcriptInfo = $("transcriptInfo");
        el.transcriptSegments = $("transcriptSegments");
        el.transcriptExportFormat = $("transcriptExportFormat");
        el.exportTranscriptBtn = $("exportTranscriptBtn");

        // Audio tab
        el.denoiseMethod = $("denoiseMethod");
        el.denoiseMethodDesc = $("denoiseMethodDesc");
        el.denoiseStrength = $("denoiseStrength");
        el.denoiseStrengthVal = $("denoiseStrengthVal");
        el.runDenoiseBtn = $("runDenoiseBtn");
        el.isolateMethod = $("isolateMethod");
        el.isolateMethodDesc = $("isolateMethodDesc");
        el.stemOptions = $("stemOptions");
        el.stemVocals = $("stemVocals");
        el.stemDrums = $("stemDrums");
        el.stemBass = $("stemBass");
        el.stemOther = $("stemOther");
        el.runIsolateBtn = $("runIsolateBtn");
        el.eqPreset = $("eqPreset");
        el.eqPresetDesc = $("eqPresetDesc");
        el.runEqBtn = $("runEqBtn");
        el.musicSelect = $("musicSelect");
        el.duckLevel = $("duckLevel");
        el.duckLevelVal = $("duckLevelVal");
        el.duckAttack = $("duckAttack");
        el.duckAttackVal = $("duckAttackVal");
        el.duckRelease = $("duckRelease");
        el.duckReleaseVal = $("duckReleaseVal");
        el.runDuckBtn = $("runDuckBtn");
        el.normalizePreset = $("normalizePreset");
        el.loudnessMeter = $("loudnessMeter");
        el.meterLUFS = $("meterLUFS");
        el.meterTP = $("meterTP");
        el.meterLRA = $("meterLRA");
        el.measureLoudnessBtn = $("measureLoudnessBtn");
        el.runNormalizeBtn = $("runNormalizeBtn");
        el.beatSensitivity = $("beatSensitivity");
        el.beatSensitivityVal = $("beatSensitivityVal");
        el.runBeatsBtn = $("runBeatsBtn");
        el.beatResults = $("beatResults");
        el.bpmValue = $("bpmValue");
        el.beatCount = $("beatCount");
        el.beatConfidence = $("beatConfidence");
        el.audioEffect = $("audioEffect");
        el.runEffectBtn = $("runEffectBtn");

        // Video tab
        el.sceneThreshold = $("sceneThreshold");
        el.sceneThresholdVal = $("sceneThresholdVal");
        el.minSceneLen = $("minSceneLen");
        el.minSceneLenVal = $("minSceneLenVal");
        el.runScenesBtn = $("runScenesBtn");
        el.sceneResults = $("sceneResults");
        el.sceneCount = $("sceneCount");
        el.avgSceneLen = $("avgSceneLen");
        el.ytChapters = $("ytChapters");
        el.ytChaptersText = $("ytChaptersText");
        el.copyChaptersBtn = $("copyChaptersBtn");
        // Speed
        el.speedPreset = $("speedPreset");
        el.speedMode = $("speedMode");
        el.speedQuality = $("speedQuality");
        el.runSpeedRampBtn = $("runSpeedRampBtn");
        // Reframe
        el.reframeAspect = $("reframeAspect");
        el.reframeFaceDetect = $("reframeFaceDetect");
        el.runReframeBtn = $("runReframeBtn");
        // Color
        el.colorPreset = $("colorPreset");
        el.colorIntensity = $("colorIntensity");
        el.colorIntensityVal = $("colorIntensityVal");
        el.runColorBtn = $("runColorBtn");
        // Chroma
        el.chromaColor = $("chromaColor");
        el.chromaSimilarity = $("chromaSimilarity");
        el.chromaSimilarityVal = $("chromaSimilarityVal");
        el.chromaBlend = $("chromaBlend");
        el.chromaBlendVal = $("chromaBlendVal");
        el.chromaBgMode = $("chromaBgMode");
        el.runChromaBtn = $("runChromaBtn");
        // BG Removal
        el.bgInstallHint = $("bgInstallHint");
        el.bgModel = $("bgModel");
        el.bgMode = $("bgMode");
        el.runBgRemoveBtn = $("runBgRemoveBtn");
        el.installRembgBtn = $("installRembgBtn");
        // Slow-Mo
        el.slowmoInstallHint = $("slowmoInstallHint");
        el.slowmoMultiplier = $("slowmoMultiplier");
        el.slowmoScale = $("slowmoScale");
        el.runSlowmoBtn = $("runSlowmoBtn");
        el.installRifeBtn = $("installRifeBtn");
        // Upscale
        el.upscaleInstallHint = $("upscaleInstallHint");
        el.upscaleModel = $("upscaleModel");
        el.upscaleScale = $("upscaleScale");
        el.runUpscaleBtn = $("runUpscaleBtn");
        el.installEsrganBtn = $("installEsrganBtn");

        // Voice tab - TTS
        el.ttsSpeaker = $("ttsSpeaker");
        el.ttsText = $("ttsText");
        el.ttsLanguage = $("ttsLanguage");
        el.ttsEmotion = $("ttsEmotion");
        el.ttsSpeed = $("ttsSpeed");
        el.ttsSpeedVal = $("ttsSpeedVal");
        el.ttsQuality = $("ttsQuality");
        el.runTtsBtn = $("runTtsBtn");
        el.voiceHint = $("voiceHint");
        el.installVoiceBtn = $("installVoiceBtn");
        el.voicePreviewCard = $("voicePreviewCard");
        el.voicePreviewAudio = $("voicePreviewAudio");
        el.voicePreviewDuration = $("voicePreviewDuration");
        el.voicePreviewModel = $("voicePreviewModel");

        // Voice tab - Clone
        el.cloneSource = $("cloneSource");
        el.cloneSourceInfo = $("cloneSourceInfo");
        el.cloneRefText = $("cloneRefText");
        el.cloneText = $("cloneText");
        el.cloneLanguage = $("cloneLanguage");
        el.runCloneBtn = $("runCloneBtn");
        el.voiceProfilesList = $("voiceProfilesList");
        el.profileName = $("profileName");
        el.saveProfileBtn = $("saveProfileBtn");

        // Voice tab - Design
        el.designDescription = $("designDescription");
        el.designText = $("designText");
        el.designLanguage = $("designLanguage");
        el.runDesignBtn = $("runDesignBtn");

        // Voice tab - Replace
        el.replaceOriginal = $("replaceOriginal");
        el.replaceNew = $("replaceNew");
        el.replaceLanguage = $("replaceLanguage");
        el.runReplaceBtn = $("runReplaceBtn");

        // Export tab
        el.expTranscriptFormat = $("expTranscriptFormat");
        el.expModel = $("expModel");
        el.runExpTranscriptBtn = $("runExpTranscriptBtn");
        // Platform export
        el.platformPreset = $("platformPreset");
        el.platformInfo = $("platformInfo");
        el.platformRes = $("platformRes");
        el.platformAspect = $("platformAspect");
        el.platformLimit = $("platformLimit");
        el.runPlatformExportBtn = $("runPlatformExportBtn");
        // Custom render
        el.renderVideoCodec = $("renderVideoCodec");
        el.renderResolution = $("renderResolution");
        el.renderCrf = $("renderCrf");
        el.renderCrfVal = $("renderCrfVal");
        el.renderAudioCodec = $("renderAudioCodec");
        el.runCustomRenderBtn = $("runCustomRenderBtn");
        // Thumbnail
        el.thumbMode = $("thumbMode");
        el.thumbTimestampGroup = $("thumbTimestampGroup");
        el.thumbTimestamp = $("thumbTimestamp");
        el.thumbCountGroup = $("thumbCountGroup");
        el.thumbCount = $("thumbCount");
        el.thumbCountVal = $("thumbCountVal");
        el.runThumbBtn = $("runThumbBtn");
        // Burn-in
        el.burninSubPath = $("burninSubPath");
        el.browseBurninSubBtn = $("browseBurninSubBtn");
        el.burninFontSize = $("burninFontSize");
        el.burninFontSizeVal = $("burninFontSizeVal");
        el.burninPosition = $("burninPosition");
        el.burninFontColor = $("burninFontColor");
        el.runBurninBtn = $("runBurninBtn");
        // Extras - Watermark
        el.watermarkText = $("watermarkText");
        el.watermarkPosition = $("watermarkPosition");
        el.watermarkOpacity = $("watermarkOpacity");
        el.watermarkOpacityVal = $("watermarkOpacityVal");
        el.runWatermarkBtn = $("runWatermarkBtn");
        // Extras - GIF
        el.gifStart = $("gifStart");
        el.gifDuration = $("gifDuration");
        el.gifWidth = $("gifWidth");
        el.runGifBtn = $("runGifBtn");
        // Extras - Audio Extract
        el.audioExtractCodec = $("audioExtractCodec");
        el.audioExtractNormalize = $("audioExtractNormalize");
        el.runAudioExtractBtn = $("runAudioExtractBtn");
        // Batch - Queue
        el.batchFolder = $("batchFolder");
        el.browseBatchFolderBtn = $("browseBatchFolderBtn");
        el.batchFileCount = $("batchFileCount");
        el.batchFileList = $("batchFileList");
        el.batchPreset = $("batchPreset");
        el.runBatchBtn = $("runBatchBtn");
        el.batchProgress = $("batchProgress");
        el.batchProgressFill = $("batchProgressFill");
        el.batchProgressText = $("batchProgressText");
        el.cancelBatchBtn = $("cancelBatchBtn");
        el.batchResults = $("batchResults");
        el.batchResultsList = $("batchResultsList");
        // Batch - Workflow
        el.workflowOpSelect = $("workflowOpSelect");
        el.addWorkflowStepBtn = $("addWorkflowStepBtn");
        el.workflowStepsList = $("workflowStepsList");
        el.runWorkflowBtn = $("runWorkflowBtn");
        el.clearWorkflowBtn = $("clearWorkflowBtn");
        // Batch - Watch
        el.watchFolder = $("watchFolder");
        el.browseWatchFolderBtn = $("browseWatchFolderBtn");
        el.watchPreset = $("watchPreset");
        el.startWatchBtn = $("startWatchBtn");
        el.stopWatchBtn = $("stopWatchBtn");
        el.watchStatus = $("watchStatus");
        el.watchStatusText = $("watchStatusText");
        el.watchProcessedCount = $("watchProcessedCount");
        // Batch - Inspect
        el.runInspectBtn = $("runInspectBtn");
        el.inspectResults = $("inspectResults");
        el.inspectFile = $("inspectFile");
        el.inspectContainer = $("inspectContainer");
        el.inspectVideo = $("inspectVideo");
        el.inspectAudio = $("inspectAudio");
        el.inspectChapters = $("inspectChapters");

        // Settings tab
        el.whisperStatusText = $("whisperStatusText");
        el.settingsInstallWhisperBtn = $("settingsInstallWhisperBtn");
        el.gpuName = $("gpuName");
        el.gpuVram = $("gpuVram");
        el.backendPort = $("backendPort");

        // Progress / Results
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
        el.newJobBtn = $("newJobBtn");
    }

    // ================================================================
    // CEP / Premiere Interface
    // ================================================================
    function initCSInterface() {
        try { cs = new CSInterface(); inPremiere = true; }
        catch (e) { cs = null; inPremiere = false; }
    }

    function jsx(script, callback) {
        if (!cs) { if (callback) callback(null); return; }
        cs.evalScript(script, function (result) { if (callback) callback(result); });
    }

    // ================================================================
    // Backend Communication
    // ================================================================
    function api(method, path, body, callback, timeout) {
        var xhr = new XMLHttpRequest();
        xhr.open(method, BACKEND + path, true);
        xhr.timeout = timeout || 15000;
        xhr.setRequestHeader("Content-Type", "application/json");
        xhr.onload = function () {
            try { callback(null, JSON.parse(xhr.responseText)); }
            catch (e) { callback(e, null); }
        };
        xhr.onerror = function () { callback(new Error("Network error"), null); };
        xhr.ontimeout = function () { callback(new Error("Timeout"), null); };
        xhr.send(body ? JSON.stringify(body) : null);
    }

    // ================================================================
    // Health Check
    // ================================================================
    var portScanPending = false;

    function checkHealth() {
        api("GET", "/health", null, function (err, data) {
            var ok = !err && data && data.status === "ok";
            if (ok) {
                connected = true;
                el.connDot.className = "conn-dot on";
                el.connLabel.textContent = "Connected";
                if (data.capabilities) capabilities = data.capabilities;
                el.backendPort.textContent = BACKEND.replace("http://127.0.0.1:", "Port ");
                updateButtons();
                return;
            }
            if (!portScanPending) { portScanPending = true; scanForServer(); }
        }, 2000);
    }

    function scanForServer() {
        var found = false;
        var checked = 0;
        var total = BACKEND_MAX_PORT - BACKEND_BASE_PORT + 1;

        for (var p = BACKEND_BASE_PORT; p <= BACKEND_MAX_PORT; p++) {
            (function (port) {
                var testUrl = "http://127.0.0.1:" + port;
                var xhr = new XMLHttpRequest();
                xhr.open("GET", testUrl + "/health", true);
                xhr.timeout = 1500;
                xhr.onload = function () {
                    checked++;
                    if (found) return;
                    try {
                        var data = JSON.parse(xhr.responseText);
                        if (data.status === "ok") {
                            found = true;
                            BACKEND = testUrl;
                            connected = true;
                            el.connDot.className = "conn-dot on";
                            el.connLabel.textContent = "Connected" + (port !== BACKEND_BASE_PORT ? " (:" + port + ")" : "");
                            el.backendPort.textContent = "Port " + port;
                            if (data.capabilities) capabilities = data.capabilities;
                            updateButtons();
                            portScanPending = false;
                        }
                    } catch (e) {}
                    if (checked >= total && !found) finishScan();
                };
                xhr.onerror = xhr.ontimeout = function () {
                    checked++;
                    if (checked >= total && !found) finishScan();
                };
                xhr.send();
            })(p);
        }

        function finishScan() {
            portScanPending = false;
            connected = false;
            el.connDot.className = "conn-dot off";
            el.connLabel.textContent = "Disconnected";
            updateButtons();
            if (!backendStartAttempted && inPremiere) {
                backendStartAttempted = true;
                jsx("startOpenCutBackend()", function () {});
            }
        }
    }

    // ================================================================
    // Project Media
    // ================================================================
    function scanProjectMedia() {
        if (!inPremiere) return;
        jsx("getProjectMedia()", function (result) {
            if (!result || result === "null" || result === "undefined") return;
            try {
                var data = JSON.parse(result);
                projectMedia = data.media || [];
                projectFolder = data.projectFolder || "";
                var html = '<option value="">-- Select a clip --</option>';
                for (var i = 0; i < projectMedia.length; i++) {
                    var m = projectMedia[i];
                    html += '<option value="' + esc(m.path) + '" data-name="' + esc(m.name) + '">' + esc(m.name) + '</option>';
                }
                el.clipSelect.innerHTML = html;
                populateMusicSelect();
            } catch (e) {}
        });
    }

    function useTimelineSelection() {
        if (!inPremiere) return;
        jsx("getTimelineSelection()", function (result) {
            if (!result || result === "null") { showAlert("No clip selected in timeline."); return; }
            try {
                var data = JSON.parse(result);
                if (data.path) selectFile(data.path, data.name || data.path.split(/[/\\]/).pop());
                else showAlert("Could not get clip path.");
            } catch (e) { showAlert("Could not read selection."); }
        });
    }

    function browseForFile() {
        if (inPremiere) {
            jsx("browseForFile()", function (result) {
                if (result && result !== "null" && result !== "undefined" && result.length > 3) {
                    selectFile(result, result.split(/[/\\]/).pop());
                }
            });
        }
    }

    function selectFile(path, name) {
        selectedPath = path;
        selectedName = name || path.split(/[/\\]/).pop();
        el.fileInfoBox.classList.remove("hidden");
        el.fileNameDisplay.textContent = selectedName;
        el.fileMetaDisplay.textContent = "Loading...";
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
        }
    }

    // ================================================================
    // Tab Navigation
    // ================================================================
    function setupNavTabs() {
        // Main nav tabs
        var navBtns = document.querySelectorAll(".nav-tab");
        for (var i = 0; i < navBtns.length; i++) {
            navBtns[i].addEventListener("click", function () {
                var target = this.getAttribute("data-nav");
                // Deactivate all
                var all = document.querySelectorAll(".nav-tab");
                for (var j = 0; j < all.length; j++) all[j].classList.remove("active");
                var panels = document.querySelectorAll(".nav-panel");
                for (var j = 0; j < panels.length; j++) panels[j].classList.remove("active");
                // Activate target
                this.classList.add("active");
                var panel = $("panel-" + target);
                if (panel) panel.classList.add("active");
                // Load settings info on first visit
                if (target === "settings") loadSettingsInfo();
                if (target === "voice") { checkVoiceStatus(); loadVoiceProfiles(); }
                if (target === "video") { checkRembgStatus(); checkRifeStatus(); checkEsrganStatus(); }
            });
        }

        // Sub-tabs (generic handler for all sub-tab groups)
        var subTabContainers = document.querySelectorAll(".sub-tabs");
        for (var i = 0; i < subTabContainers.length; i++) {
            (function (container) {
                var btns = container.querySelectorAll(".sub-tab");
                var parent = container.parentElement;
                for (var j = 0; j < btns.length; j++) {
                    btns[j].addEventListener("click", function () {
                        var target = this.getAttribute("data-sub");
                        // Deactivate siblings
                        var siblings = container.querySelectorAll(".sub-tab");
                        for (var k = 0; k < siblings.length; k++) siblings[k].classList.remove("active");
                        var panels = parent.querySelectorAll(".sub-panel");
                        for (var k = 0; k < panels.length; k++) panels[k].classList.remove("active");
                        // Activate
                        this.classList.add("active");
                        var panel = $("sub-" + target);
                        if (panel) panel.classList.add("active");
                    });
                }
            })(subTabContainers[i]);
        }
    }

    // ================================================================
    // Button State
    // ================================================================
    function updateButtons() {
        var canRun = connected && selectedPath;

        // Cut tab
        el.runSilenceBtn.disabled = !canRun;
        el.runFillersBtn.disabled = !canRun;
        el.runFullBtn.disabled = !canRun;

        // Captions tab
        el.runStyledCaptionsBtn.disabled = !canRun;
        el.runSubtitleBtn.disabled = !canRun;
        el.runTranscriptBtn.disabled = !canRun;

        // Audio tab
        el.runDenoiseBtn.disabled = !canRun;
        el.runIsolateBtn.disabled = !canRun;
        el.runEqBtn.disabled = !canRun;
        el.runDuckBtn.disabled = !(canRun && el.musicSelect && el.musicSelect.value);
        el.measureLoudnessBtn.disabled = !canRun;
        el.runNormalizeBtn.disabled = !canRun;
        el.runBeatsBtn.disabled = !canRun;
        el.runEffectBtn.disabled = !canRun;

        // Video tab
        el.runScenesBtn.disabled = !canRun;
        el.runSpeedRampBtn.disabled = !canRun;
        el.runReframeBtn.disabled = !canRun;
        el.runColorBtn.disabled = !canRun;
        el.runChromaBtn.disabled = !canRun;
        el.runBgRemoveBtn.disabled = !(canRun && rembgInstalled);
        el.runSlowmoBtn.disabled = !(canRun && rifeInstalled);
        el.runUpscaleBtn.disabled = !(canRun && esrganInstalled);

        // AI install hints
        if (connected && !rembgInstalled) { el.bgInstallHint.classList.remove("hidden"); }
        else { el.bgInstallHint.classList.add("hidden"); }
        if (connected && !rifeInstalled) { el.slowmoInstallHint.classList.remove("hidden"); }
        else { el.slowmoInstallHint.classList.add("hidden"); }
        if (connected && !esrganInstalled) { el.upscaleInstallHint.classList.remove("hidden"); }
        else { el.upscaleInstallHint.classList.add("hidden"); }

        // Voice tab
        var canVoice = connected && voiceInstalled;
        var hasText = canVoice;  // text check happens in run functions
        el.runTtsBtn.disabled = !canVoice;
        el.runCloneBtn.disabled = !canVoice;
        el.runDesignBtn.disabled = !canVoice;
        el.runReplaceBtn.disabled = !(canRun && voiceInstalled);

        // Voice hints
        if (connected && !voiceInstalled) {
            el.voiceHint.classList.remove("hidden");
        } else {
            el.voiceHint.classList.add("hidden");
        }

        // Export tab
        el.runExpTranscriptBtn.disabled = !canRun;
        el.runPlatformExportBtn.disabled = !canRun;
        el.runCustomRenderBtn.disabled = !canRun;
        el.runThumbBtn.disabled = !canRun;
        el.runBurninBtn.disabled = !(canRun && el.burninSubPath.value);
        el.runWatermarkBtn.disabled = !canRun;
        el.runGifBtn.disabled = !canRun;
        el.runAudioExtractBtn.disabled = !canRun;
        // Batch tab
        el.runBatchBtn.disabled = !(batchFiles.length > 0);
        el.runWorkflowBtn.disabled = !(canRun && workflowSteps.length > 0);
        el.startWatchBtn.disabled = !el.watchFolder.value;
        el.runInspectBtn.disabled = !canRun;

        // Whisper hints
        if (capabilities.captions === false) {
            el.captionsHint.classList.remove("hidden");
            el.fillersHint.classList.remove("hidden");
        } else {
            el.captionsHint.classList.add("hidden");
            el.fillersHint.classList.add("hidden");
        }
    }

    // ================================================================
    // Job Execution & Tracking
    // ================================================================
    function startJob(endpoint, payload) {
        if (currentJob) return;

        el.progressSection.classList.remove("hidden");
        el.resultsSection.classList.add("hidden");
        el.progressBar.style.width = "0%";
        el.progressLabel.textContent = "Starting...";
        el.cancelBtn.classList.remove("hidden");
        jobStartTime = Date.now();
        el.progressElapsed.textContent = "0s";
        elapsedTimer = setInterval(function () {
            var s = Math.floor((Date.now() - jobStartTime) / 1000);
            el.progressElapsed.textContent = s + "s";
        }, 1000);

        api("POST", endpoint, payload, function (err, data) {
            if (err || !data || data.error) {
                showAlert(data ? data.error : "Failed to start job");
                hideProgress();
                return;
            }
            currentJob = data.job_id;

            if (SSE_OK) {
                trackJobSSE(data.job_id);
            } else {
                trackJobPoll(data.job_id);
            }
        });
    }

    function trackJobSSE(jobId) {
        if (activeStream) { activeStream.close(); activeStream = null; }
        var es = new EventSource(BACKEND + "/stream/" + jobId);
        activeStream = es;

        es.onmessage = function (e) {
            try {
                var job = JSON.parse(e.data);
                updateProgress(job);
                if (job.status === "complete" || job.status === "error" || job.status === "cancelled") {
                    es.close();
                    activeStream = null;
                    onJobDone(job);
                }
            } catch (ex) {}
        };
        es.onerror = function () {
            es.close();
            activeStream = null;
            // Fallback to polling
            trackJobPoll(jobId);
        };
    }

    function trackJobPoll(jobId) {
        pollTimer = setInterval(function () {
            api("GET", "/status/" + jobId, null, function (err, job) {
                if (err || !job) return;
                updateProgress(job);
                if (job.status === "complete" || job.status === "error" || job.status === "cancelled") {
                    clearInterval(pollTimer);
                    pollTimer = null;
                    onJobDone(job);
                }
            });
        }, POLL_MS);
    }

    function updateProgress(job) {
        el.progressBar.style.width = (job.progress || 0) + "%";
        el.progressLabel.textContent = job.message || "Processing...";
    }

    function onJobDone(job) {
        currentJob = null;
        if (elapsedTimer) { clearInterval(elapsedTimer); elapsedTimer = null; }

        if (job.status === "error") {
            el.progressBar.style.width = "100%";
            el.progressBar.style.background = "var(--error-dim)";
            el.progressLabel.textContent = "Error: " + (job.error || "Unknown error");
            el.cancelBtn.classList.add("hidden");
            setTimeout(function () {
                el.progressBar.style.background = "";
                hideProgress();
            }, 5000);
            return;
        }

        if (job.status === "cancelled") {
            hideProgress();
            return;
        }

        // Success
        hideProgress();
        showResults(job);

        // Auto-import XML into Premiere
        if (job.result) {
            var xmlPath = job.result.xml_path;
            if (xmlPath && inPremiere) {
                jsx('importXMLToProject("' + esc(xmlPath) + '")', function () {});
                lastXmlPath = xmlPath;
            }
            var overlayPath = job.result.overlay_path;
            if (overlayPath && inPremiere) {
                jsx('importOverlayToProject("' + esc(overlayPath) + '")', function () {});
                lastOverlayPath = overlayPath;
            }
        }
    }

    function hideProgress() {
        el.progressSection.classList.add("hidden");
        el.cancelBtn.classList.add("hidden");
        if (elapsedTimer) { clearInterval(elapsedTimer); elapsedTimer = null; }
    }

    function showResults(job) {
        el.resultsSection.classList.remove("hidden");
        el.resultsTitle.textContent = "Complete";
        el.resultsTitle.style.color = "var(--success)";

        var stats = "";
        var r = job.result || {};

        if (r.summary) {
            stats += r.summary + "<br>";
        }
        if (r.segments !== undefined) {
            stats += r.segments + " segments";
        }
        if (r.filler_stats) {
            stats += " | " + r.filler_stats.removed_fillers + " fillers removed (" + r.filler_stats.total_filler_time.toFixed(1) + "s)";
        }
        if (r.caption_segments !== undefined) {
            stats += " | " + r.caption_segments + " captions, " + (r.words || 0) + " words";
        }
        if (r.style) {
            stats += " | Style: " + r.style;
        }
        if (r.bpm) {
            stats += "BPM: " + r.bpm.toFixed(0) + " | " + r.total_beats + " beats";
        }
        if (r.total_scenes) {
            stats += "Scenes: " + r.total_scenes + " | Avg: " + r.avg_scene_length + "s";
        }

        el.resultsStats.innerHTML = stats || "Processing complete.";
        el.resultsPath.textContent = r.xml_path || r.output_path || r.overlay_path || "";
    }

    function cancelJob() {
        if (currentJob) {
            api("POST", "/cancel/" + currentJob, {}, function () {});
        }
    }

    // ================================================================
    // Run Functions (one per action button)
    // ================================================================

    // --- CUT TAB ---
    function runSilence() {
        var preset = el.silencePreset.value;
        var payload = { filepath: selectedPath, output_dir: projectFolder };
        if (preset) {
            payload.preset = preset;
        } else {
            payload.threshold = parseFloat(el.threshold.value);
            payload.min_duration = parseFloat(el.minDuration.value);
            payload.padding_before = parseFloat(el.padBefore.value);
            payload.padding_after = parseFloat(el.padAfter.value);
        }
        startJob("/silence", payload);
    }

    function runFillers() {
        var checks = el.fillerChecks.querySelectorAll("input:checked");
        var removeKeys = [];
        for (var i = 0; i < checks.length; i++) removeKeys.push(checks[i].getAttribute("data-filler"));
        var customRaw = el.fillerCustom.value.trim();
        var custom = customRaw ? customRaw.split(",").map(function (s) { return s.trim(); }).filter(Boolean) : [];

        startJob("/fillers", {
            filepath: selectedPath,
            output_dir: projectFolder,
            model: el.fillerModel.value,
            remove_fillers: removeKeys,
            custom_words: custom,
            remove_silence: el.fillerSilence.checked,
        });
    }

    function runFull() {
        startJob("/full", {
            filepath: selectedPath,
            output_dir: projectFolder,
            preset: el.fullPreset.value,
            skip_zoom: !el.fullZoom.checked,
            skip_captions: !el.fullCaptions.checked,
            remove_fillers: el.fullFillers.checked,
        });
    }

    // --- CAPTIONS TAB ---
    function runStyledCaptions() {
        var actionWords = el.actionWordsInput.value.trim();
        var custom = actionWords ? actionWords.split(",").map(function (s) { return s.trim(); }).filter(Boolean) : [];

        startJob("/styled-captions", {
            filepath: selectedPath,
            output_dir: projectFolder,
            style: el.captionStyle.value,
            model: el.captionModel.value,
            language: el.captionLang.value || null,
            action_words: custom,
            auto_detect_energy: el.captionAutoAction.checked,
        });
    }

    function runSubtitle() {
        startJob("/captions", {
            filepath: selectedPath,
            output_dir: projectFolder,
            model: el.subModel.value,
            language: el.subLang.value || null,
            format: el.subFormat.value,
            word_timestamps: true,
        });
    }

    function runTranscript() {
        startJob("/transcript", {
            filepath: selectedPath,
            model: el.transcriptModel.value,
        });
    }

    function exportEditedTranscript() {
        if (!transcriptData) return;

        api("POST", "/transcript/export", {
            filepath: selectedPath,
            output_dir: projectFolder,
            segments: transcriptData.segments,
            format: el.transcriptExportFormat.value,
            language: transcriptData.language || "en",
        }, function (err, data) {
            if (!err && data && data.output_path) {
                showAlert("Exported to: " + data.output_path.split(/[/\\]/).pop());
            } else {
                showAlert("Export failed: " + (data ? data.error : "Unknown error"));
            }
        });
    }

    // --- AUDIO TAB ---
    function runDenoise() {
        startJob("/audio/denoise", {
            filepath: selectedPath,
            output_dir: projectFolder,
            method: el.denoiseMethod.value,
            strength: parseFloat(el.denoiseStrength.value),
        });
    }

    function runIsolate() {
        var method = el.isolateMethod.value;
        var payload = {
            filepath: selectedPath,
            output_dir: projectFolder,
            method: method,
        };
        
        if (method === "demucs") {
            // Collect selected stems
            var stems = [];
            if (el.stemVocals.checked) stems.push("vocals");
            if (el.stemDrums.checked) stems.push("drums");
            if (el.stemBass.checked) stems.push("bass");
            if (el.stemOther.checked) stems.push("other");
            if (stems.length > 0) payload.stems = stems;
        }
        
        startJob("/audio/isolate", payload);
    }

    function runEq() {
        startJob("/audio/eq", {
            filepath: selectedPath,
            output_dir: projectFolder,
            preset: el.eqPreset.value,
        });
    }

    function runDuck() {
        if (!el.musicSelect.value) {
            showAlert("Please select a music track");
            return;
        }
        startJob("/audio/duck", {
            filepath: selectedPath,
            music_filepath: el.musicSelect.value,
            output_dir: projectFolder,
            duck_level: parseFloat(el.duckLevel.value),
            attack_ms: parseFloat(el.duckAttack.value),
            release_ms: parseFloat(el.duckRelease.value),
        });
    }

    function measureLoudness() {
        el.loudnessMeter.classList.remove("hidden");
        el.meterLUFS.textContent = "Measuring...";
        el.meterTP.textContent = "--";
        el.meterLRA.textContent = "--";

        api("POST", "/audio/measure", { filepath: selectedPath }, function (err, data) {
            if (!err && data && !data.error) {
                el.meterLUFS.textContent = data.integrated_lufs.toFixed(1) + " LUFS";
                el.meterTP.textContent = data.true_peak_dbtp.toFixed(1) + " dBTP";
                el.meterLRA.textContent = data.loudness_range_lu.toFixed(1) + " LU";
            } else {
                el.meterLUFS.textContent = "Error";
            }
        });
    }

    function runNormalize() {
        startJob("/audio/normalize", {
            filepath: selectedPath,
            output_dir: projectFolder,
            preset: el.normalizePreset.value,
        });
    }

    function runBeats() {
        el.beatResults.classList.add("hidden");
        startJob("/audio/beats", {
            filepath: selectedPath,
            sensitivity: parseFloat(el.beatSensitivity.value),
        });
    }

    function runEffect() {
        startJob("/audio/effects/apply", {
            filepath: selectedPath,
            output_dir: projectFolder,
            effect: el.audioEffect.value,
        });
    }

    // --- VOICE TAB ---
    function runTts() {
        var text = el.ttsText.value.trim();
        if (!text) { showAlert("Please enter text to generate speech."); return; }

        startJob("/voice/generate", {
            mode: "tts",
            text: text,
            speaker: el.ttsSpeaker.value,
            language: el.ttsLanguage.value,
            emotion: el.ttsEmotion.value,
            speed: parseFloat(el.ttsSpeed.value),
            quality: el.ttsQuality.value,
            output_dir: projectFolder,
        });
    }

    function runClone() {
        var text = el.cloneText.value.trim();
        if (!text) { showAlert("Please enter text to generate."); return; }
        if (!cloneRefAudioPath) { showAlert("Please select a reference audio file."); return; }

        startJob("/voice/generate", {
            mode: "clone",
            text: text,
            ref_audio: cloneRefAudioPath,
            ref_text: el.cloneRefText.value.trim(),
            language: el.cloneLanguage.value,
            quality: el.ttsQuality.value,
            output_dir: projectFolder,
        });
    }

    function runDesign() {
        var text = el.designText.value.trim();
        var desc = el.designDescription.value.trim();
        if (!text) { showAlert("Please enter text to generate."); return; }
        if (!desc) { showAlert("Please describe the voice you want."); return; }

        startJob("/voice/generate", {
            mode: "design",
            text: text,
            voice_description: desc,
            language: el.designLanguage.value,
            output_dir: projectFolder,
        });
    }

    function runReplace() {
        var orig = el.replaceOriginal.value.trim();
        var repl = el.replaceNew.value.trim();
        if (!orig || !repl) { showAlert("Both original and replacement text are required."); return; }

        startJob("/voice/replace", {
            filepath: selectedPath,
            original_text: orig,
            replacement_text: repl,
            language: el.replaceLanguage.value,
            quality: el.ttsQuality.value,
            output_dir: projectFolder,
        });
    }

    function installVoice() {
        showAlert("Installing Qwen3-TTS... This may take several minutes.");
        startJob("/voice/install", {});
    }

    function checkVoiceStatus() {
        api("GET", "/voice/check", null, function (err, data) {
            if (!err && data) {
                voiceInstalled = !!data.installed;
                updateButtons();
            }
        });
    }

    function loadVoiceProfiles() {
        api("GET", "/voice/profiles", null, function (err, data) {
            if (!err && data && data.profiles) {
                voiceProfiles = data.profiles;
                renderVoiceProfiles(data.profiles);
                // Add profiles to clone source dropdown
                updateCloneSourceOptions(data.profiles);
            }
        });
    }

    function renderVoiceProfiles(profiles) {
        if (!profiles || profiles.length === 0) {
            el.voiceProfilesList.innerHTML = '<p class="card-desc">No saved profiles yet.</p>';
            return;
        }
        var html = "";
        for (var i = 0; i < profiles.length; i++) {
            var p = profiles[i];
            html += '<div class="voice-profile-item" data-id="' + esc(p.id) + '">'
                + '<div class="vp-info">'
                + '<span class="vp-name">' + esc(p.name) + '</span>'
                + '<span class="vp-meta">' + esc(p.language) + ' | ' + (p.duration_seconds || 0).toFixed(1) + 's</span>'
                + '</div>'
                + '<div class="vp-actions">'
                + '<button class="btn-xs vp-use" data-id="' + esc(p.id) + '" data-audio="' + esc(p.ref_audio_path) + '" data-text="' + esc(p.ref_text) + '">Use</button>'
                + '<button class="btn-xs btn-danger vp-delete" data-id="' + esc(p.id) + '">Del</button>'
                + '</div>'
                + '</div>';
        }
        el.voiceProfilesList.innerHTML = html;

        // Bind profile actions
        var useBtns = el.voiceProfilesList.querySelectorAll(".vp-use");
        for (var i = 0; i < useBtns.length; i++) {
            useBtns[i].addEventListener("click", function () {
                cloneRefAudioPath = this.getAttribute("data-audio");
                el.cloneRefText.value = this.getAttribute("data-text") || "";
                el.cloneSourceInfo.textContent = "Using profile: " + this.closest(".voice-profile-item").querySelector(".vp-name").textContent;
                updateButtons();
            });
        }
        var delBtns = el.voiceProfilesList.querySelectorAll(".vp-delete");
        for (var i = 0; i < delBtns.length; i++) {
            delBtns[i].addEventListener("click", function () {
                var pid = this.getAttribute("data-id");
                api("DELETE", "/voice/profiles/" + pid, null, function (err) {
                    if (!err) loadVoiceProfiles();
                });
            });
        }
    }

    function updateCloneSourceOptions(profiles) {
        var html = '<option value="" disabled selected>Select source...</option>'
            + '<option value="browse">Browse for audio file...</option>'
            + '<option value="selected">Use selected clip</option>';
        for (var i = 0; i < profiles.length; i++) {
            html += '<option value="profile:' + esc(profiles[i].id) + '">Profile: ' + esc(profiles[i].name) + '</option>';
        }
        el.cloneSource.innerHTML = html;
    }

    function saveVoiceProfile() {
        var name = el.profileName.value.trim();
        if (!name) { showAlert("Please enter a profile name."); return; }
        if (!cloneRefAudioPath) { showAlert("No reference audio selected."); return; }

        api("POST", "/voice/profiles", {
            name: name,
            ref_audio: cloneRefAudioPath,
            ref_text: el.cloneRefText.value.trim(),
            language: el.cloneLanguage.value,
        }, function (err, data) {
            if (!err && data && data.profile) {
                showAlert("Profile saved: " + name);
                el.profileName.value = "";
                loadVoiceProfiles();
            } else {
                showAlert("Failed to save profile: " + (data ? data.error : "Error"));
            }
        });
    }

    function showVoicePreview(outputPath, duration, model) {
        el.voicePreviewCard.classList.remove("hidden");
        el.voicePreviewAudio.src = BACKEND + "/voice/preview/" + encodeURIComponent(outputPath);
        el.voicePreviewDuration.textContent = (duration || 0).toFixed(1) + "s";
        el.voicePreviewModel.textContent = model || "";
    }

    // --- VIDEO TAB ---
    function runScenes() {
        el.sceneResults.classList.add("hidden");
        startJob("/video/scenes", {
            filepath: selectedPath,
            threshold: parseFloat(el.sceneThreshold.value),
            min_scene_length: parseFloat(el.minSceneLen.value),
        });
    }

    function runSpeedRamp() {
        startJob("/video/speed-ramp", {
            input: selectedPath,
            preset: el.speedPreset.value,
            mode: el.speedMode.value,
            quality: el.speedQuality.value,
            output_dir: projectFolder,
        });
    }

    function runReframe() {
        startJob("/video/reframe", {
            input: selectedPath,
            target_aspect: el.reframeAspect.value,
            face_detection: el.reframeFaceDetect.checked,
            output_dir: projectFolder,
        });
    }

    function runColor() {
        startJob("/video/lut", {
            input: selectedPath,
            preset: el.colorPreset.value,
            intensity: parseFloat(el.colorIntensity.value),
            output_dir: projectFolder,
        });
    }

    function runChromaKey() {
        startJob("/video/chroma-key", {
            input: selectedPath,
            key_color: el.chromaColor.value,
            similarity: parseFloat(el.chromaSimilarity.value),
            blend: parseFloat(el.chromaBlend.value),
            bg_mode: el.chromaBgMode.value,
            output_dir: projectFolder,
        });
    }

    function runBgRemove() {
        startJob("/video/bg-remove", {
            input: selectedPath,
            model: el.bgModel.value,
            bg_mode: el.bgMode.value,
            output_dir: projectFolder,
        });
    }

    function installRembg() {
        startJob("/video/bg-install", {});
    }

    function runSlowmo() {
        startJob("/video/slowmo", {
            input: selectedPath,
            multiplier: parseInt(el.slowmoMultiplier.value),
            scale: parseFloat(el.slowmoScale.value),
            output_dir: projectFolder,
        });
    }

    function installRife() {
        startJob("/video/slowmo-install", {});
    }

    function runUpscale() {
        startJob("/video/upscale", {
            input: selectedPath,
            model: el.upscaleModel.value,
            scale: parseInt(el.upscaleScale.value),
            output_dir: projectFolder,
        });
    }

    function installEsrgan() {
        startJob("/video/upscale-install", {});
    }

    function checkRembgStatus() {
        fetch(BACKEND + "/video/bg-check").then(function (r) { return r.json(); }).then(function (d) {
            rembgInstalled = d.installed;
            updateButtons();
        }).catch(function () {});
    }

    function checkRifeStatus() {
        fetch(BACKEND + "/video/slowmo-check").then(function (r) { return r.json(); }).then(function (d) {
            rifeInstalled = d.installed;
            updateButtons();
        }).catch(function () {});
    }

    function checkEsrganStatus() {
        fetch(BACKEND + "/video/upscale-check").then(function (r) { return r.json(); }).then(function (d) {
            esrganInstalled = d.installed;
            updateButtons();
        }).catch(function () {});
    }

    // --- EXPORT TAB ---
    function runExpTranscript() {
        var fmt = el.expTranscriptFormat.value;
        if (fmt === "plain" || fmt === "timestamped") {
            startJob("/transcript", {
                filepath: selectedPath,
                model: el.expModel.value,
            });
        } else {
            startJob("/captions", {
                filepath: selectedPath,
                output_dir: projectFolder,
                model: el.expModel.value,
                format: fmt,
                word_timestamps: true,
            });
        }
    }

    // --- EXPORT TAB: Platform ---
    function runPlatformExport() {
        startJob("/export/render", {
            input: selectedPath,
            preset: el.platformPreset.value,
            output_dir: projectFolder,
        });
    }

    function updatePlatformInfo() {
        var presetInfo = {
            youtube_1080: { res: "1920x1080", aspect: "16:9", limit: "" },
            youtube_4k:   { res: "3840x2160", aspect: "16:9", limit: "" },
            shorts:       { res: "1080x1920", aspect: "9:16", limit: "60s max" },
            tiktok:       { res: "1080x1920", aspect: "9:16", limit: "10m max" },
            instagram_reels: { res: "1080x1920", aspect: "9:16", limit: "90s max" },
            instagram_feed:  { res: "1080x1350", aspect: "4:5", limit: "60s max" },
            instagram_square: { res: "1080x1080", aspect: "1:1", limit: "60s max" },
            twitter:      { res: "1280x720", aspect: "16:9", limit: "2m20s max" },
            linkedin:     { res: "1920x1080", aspect: "16:9", limit: "10m max" },
            podcast_video: { res: "1920x1080", aspect: "16:9", limit: "" },
            podcast_audio: { res: "Audio only", aspect: "MP3", limit: "" },
        };
        var info = presetInfo[el.platformPreset.value] || {};
        el.platformRes.textContent = info.res || "";
        el.platformAspect.textContent = info.aspect || "";
        el.platformLimit.textContent = info.limit || "";
    }

    // --- EXPORT TAB: Custom Render ---
    function runCustomRender() {
        startJob("/export/render", {
            input: selectedPath,
            video_codec: el.renderVideoCodec.value,
            audio_codec: el.renderAudioCodec.value,
            resolution: el.renderResolution.value,
            crf: parseInt(el.renderCrf.value),
            output_dir: projectFolder,
        });
    }

    // --- EXPORT TAB: Thumbnail ---
    function runThumbnail() {
        var mode = el.thumbMode.value;
        var actualMode = mode === "single_ts" ? "single" : mode;
        var ts = mode === "single_ts" ? parseFloat(el.thumbTimestamp.value) : -1;
        startJob("/export/thumbnail", {
            input: selectedPath,
            mode: actualMode,
            timestamp: ts,
            count: parseInt(el.thumbCount.value),
            output_dir: projectFolder,
        });
    }

    function updateThumbMode() {
        var mode = el.thumbMode.value;
        el.thumbTimestampGroup.classList.toggle("hidden", mode !== "single_ts");
        el.thumbCountGroup.classList.toggle("hidden", mode !== "multi");
    }

    // --- EXPORT TAB: Burn-In ---
    function runBurnin() {
        if (!el.burninSubPath.value) { showAlert("Select a subtitle file first"); return; }
        startJob("/export/burn-subs", {
            input: selectedPath,
            subtitle_path: el.burninSubPath.value,
            font_size: parseInt(el.burninFontSize.value),
            position: el.burninPosition.value,
            font_color: el.burninFontColor.value,
            output_dir: projectFolder,
        });
    }

    function browseBurninSub() {
        if (inPremiere) {
            jsx("browseForFile()", function (result) {
                if (result && result !== "null" && result.length > 3) {
                    el.burninSubPath.value = result;
                    updateButtons();
                }
            });
        }
    }

    // --- EXPORT TAB: Watermark ---
    function runWatermark() {
        startJob("/export/watermark", {
            input: selectedPath,
            watermark_type: "text",
            text: el.watermarkText.value || "WATERMARK",
            position: el.watermarkPosition.value,
            opacity: parseFloat(el.watermarkOpacity.value),
            output_dir: projectFolder,
        });
    }

    // --- EXPORT TAB: GIF ---
    function runGif() {
        startJob("/export/gif", {
            input: selectedPath,
            start: parseFloat(el.gifStart.value),
            duration: parseFloat(el.gifDuration.value),
            width: parseInt(el.gifWidth.value),
            output_dir: projectFolder,
        });
    }

    // --- EXPORT TAB: Audio Extract ---
    function runAudioExtract() {
        startJob("/export/audio-extract", {
            input: selectedPath,
            codec: el.audioExtractCodec.value,
            normalize: el.audioExtractNormalize.checked,
            output_dir: projectFolder,
        });
    }

    // ================================================================
    // BATCH & WORKFLOW
    // ================================================================

    function browseBatchFolder() {
        if (inPremiere) {
            jsx("browseForFolder()", function (result) {
                if (result && result !== "null" && result.length > 2) {
                    el.batchFolder.value = result;
                    scanBatchFolder(result);
                }
            });
        }
    }

    function scanBatchFolder(folder) {
        fetch(API + "/batch/scan", {
            method: "POST", headers: {"Content-Type": "application/json"},
            body: JSON.stringify({folder: folder}),
        })
        .then(function (r) { return r.json(); })
        .then(function (data) {
            batchFiles = (data.files || []).map(function (f) { return f.path; });
            el.batchFileCount.textContent = batchFiles.length;
            el.batchFileList.innerHTML = batchFiles.slice(0, 50).map(function (fp) {
                var name = fp.split(/[\\/]/).pop();
                return '<div class="batch-file-item">' + name + '</div>';
            }).join("");
            if (batchFiles.length > 50) {
                el.batchFileList.innerHTML += '<div class="batch-file-item dim">...and ' + (batchFiles.length - 50) + ' more</div>';
            }
            updateButtons();
        })
        .catch(function (e) { showAlert("Scan failed: " + e.message); });
    }

    function runBatch() {
        if (batchFiles.length === 0) return;
        el.batchProgress.classList.remove("hidden");
        el.batchResults.classList.add("hidden");
        el.batchProgressFill.style.width = "0%";
        el.batchProgressText.textContent = "0 / " + batchFiles.length;

        fetch(API + "/batch/start", {
            method: "POST", headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                files: batchFiles,
                preset: el.batchPreset.value,
                output_dir: projectFolder,
            }),
        })
        .then(function (r) { return r.json(); })
        .then(function (data) {
            if (data.error) { showAlert(data.error); return; }
            batchJobId = data.job_id;
            pollBatchStatus();
        })
        .catch(function (e) { showAlert("Batch start failed: " + e.message); });
    }

    function pollBatchStatus() {
        if (!batchJobId) return;
        batchPollTimer = setInterval(function () {
            fetch(API + "/batch/status/" + batchJobId)
            .then(function (r) { return r.json(); })
            .then(function (data) {
                var pct = data.progress_pct || 0;
                el.batchProgressFill.style.width = pct + "%";
                el.batchProgressText.textContent = data.completed + " / " + data.total;

                if (data.status === "done" || data.status === "error" || data.status === "cancelled") {
                    clearInterval(batchPollTimer);
                    batchPollTimer = null;
                    showBatchResults(data);
                }
            })
            .catch(function () {});
        }, 2000);
    }

    function showBatchResults(data) {
        el.batchResults.classList.remove("hidden");
        var html = "";
        (data.items || []).forEach(function (item) {
            var icon = item.status === "done" ? "OK" : "ERR";
            var cls = item.status === "done" ? "batch-item-ok" : "batch-item-err";
            html += '<div class="batch-result-item ' + cls + '">';
            html += '<span class="batch-result-icon">' + icon + '</span>';
            html += '<span class="batch-result-file">' + item.file + '</span>';
            html += '<span class="batch-result-msg">' + (item.message || item.error || "") + '</span>';
            html += '</div>';
        });
        el.batchResultsList.innerHTML = html;
        showAlert("Batch complete: " + data.completed + "/" + data.total + " files processed");
    }

    function cancelBatch() {
        if (batchJobId) {
            fetch(API + "/batch/cancel/" + batchJobId, {method: "POST"});
            if (batchPollTimer) { clearInterval(batchPollTimer); batchPollTimer = null; }
        }
    }

    // --- Workflow Builder ---
    function addWorkflowStep() {
        var op = el.workflowOpSelect.value;
        workflowSteps.push({operation: op, params: {}});
        renderWorkflowSteps();
        updateButtons();
    }

    function removeWorkflowStep(index) {
        workflowSteps.splice(index, 1);
        renderWorkflowSteps();
        updateButtons();
    }

    function clearWorkflow() {
        workflowSteps = [];
        renderWorkflowSteps();
        updateButtons();
    }

    function renderWorkflowSteps() {
        var html = "";
        workflowSteps.forEach(function (step, i) {
            html += '<div class="workflow-step-item">';
            html += '<span class="workflow-step-num">' + (i + 1) + '</span>';
            html += '<span class="workflow-step-name">' + step.operation.replace(/_/g, " ") + '</span>';
            html += '<button class="btn-xs" onclick="window._removeWfStep(' + i + ')">X</button>';
            html += '</div>';
        });
        el.workflowStepsList.innerHTML = html || '<div class="dim">No steps added yet</div>';
    }
    window._removeWfStep = removeWorkflowStep;

    function runWorkflow() {
        if (!selectedPath || workflowSteps.length === 0) return;
        startJob("/batch/start", {
            files: [selectedPath],
            steps: workflowSteps,
            output_dir: projectFolder,
        });
    }

    // --- Watch Folder ---
    function browseWatchFolder() {
        if (inPremiere) {
            jsx("browseForFolder()", function (result) {
                if (result && result !== "null" && result.length > 2) {
                    el.watchFolder.value = result;
                    updateButtons();
                }
            });
        }
    }

    function startWatch() {
        var folder = el.watchFolder.value;
        if (!folder) return;
        fetch(API + "/batch/watch/start", {
            method: "POST", headers: {"Content-Type": "application/json"},
            body: JSON.stringify({
                folder: folder,
                preset: el.watchPreset.value,
                output_dir: "",
            }),
        })
        .then(function (r) { return r.json(); })
        .then(function (data) {
            if (data.error) { showAlert(data.error); return; }
            activeWatcherId = data.watcher_id;
            el.watchStatus.classList.remove("hidden");
            el.stopWatchBtn.classList.remove("hidden");
            el.startWatchBtn.disabled = true;
            el.watchStatusText.textContent = "Watching: " + folder.split(/[\\/]/).pop();
        })
        .catch(function (e) { showAlert("Watch failed: " + e.message); });
    }

    function stopWatch() {
        if (activeWatcherId) {
            fetch(API + "/batch/watch/stop/" + activeWatcherId, {method: "POST"});
            activeWatcherId = null;
            el.watchStatus.classList.add("hidden");
            el.stopWatchBtn.classList.add("hidden");
            el.startWatchBtn.disabled = false;
        }
    }

    // --- Media Inspector ---
    function runInspect() {
        if (!selectedPath) return;
        el.inspectResults.classList.add("hidden");
        fetch(API + "/batch/inspect", {
            method: "POST", headers: {"Content-Type": "application/json"},
            body: JSON.stringify({filepath: selectedPath}),
        })
        .then(function (r) { return r.json(); })
        .then(function (data) {
            if (data.error) { showAlert(data.error); return; }
            el.inspectResults.classList.remove("hidden");
            // File info
            var f = data.file || {};
            el.inspectFile.innerHTML = '<div class="inspect-title">File</div>' +
                inspRow("Name", f.name) + inspRow("Size", f.size_mb + " MB") + inspRow("Extension", f.extension);
            // Container
            var c = data.container || {};
            el.inspectContainer.innerHTML = '<div class="inspect-title">Container</div>' +
                inspRow("Format", c.format_long || c.format) + inspRow("Duration", formatTime(c.duration)) +
                inspRow("Bitrate", c.bitrate_str) + inspRow("Streams", c.streams);
            // Video
            if (data.video) {
                var v = data.video;
                el.inspectVideo.innerHTML = '<div class="inspect-title">Video</div>' +
                    inspRow("Codec", v.codec_long_name || v.codec_name) + inspRow("Resolution", v.resolution) +
                    inspRow("FPS", v.fps) + inspRow("Bitrate", v.bitrate_str) +
                    inspRow("Pixel Format", v.pixel_format) + inspRow("Color Space", v.color_space || "N/A") +
                    inspRow("Bit Depth", v.bit_depth || "N/A") + inspRow("DAR", v.display_aspect_ratio);
            } else {
                el.inspectVideo.innerHTML = "";
            }
            // Audio
            if (data.audio) {
                var a = data.audio;
                el.inspectAudio.innerHTML = '<div class="inspect-title">Audio</div>' +
                    inspRow("Codec", a.codec_long_name || a.codec_name) + inspRow("Sample Rate", a.sample_rate + " Hz") +
                    inspRow("Channels", a.channels + " (" + (a.channel_layout || "?") + ")") +
                    inspRow("Bitrate", a.bitrate_str) + inspRow("Sample Format", a.sample_format || "N/A");
            } else {
                el.inspectAudio.innerHTML = "";
            }
            // Chapters
            if (data.chapters && data.chapters.length > 0) {
                var chHtml = '<div class="inspect-title">Chapters (' + data.chapters.length + ')</div>';
                data.chapters.forEach(function (ch) {
                    chHtml += inspRow(formatTime(ch.start) + " - " + formatTime(ch.end), ch.title);
                });
                el.inspectChapters.innerHTML = chHtml;
            } else {
                el.inspectChapters.innerHTML = "";
            }
        })
        .catch(function (e) { showAlert("Inspect failed: " + e.message); });
    }

    function inspRow(label, value) {
        return '<div class="inspect-row"><span class="inspect-label">' + label + '</span><span class="inspect-value">' + (value || "") + '</span></div>';
    }

    function formatTime(sec) {
        if (!sec) return "0:00";
        var m = Math.floor(sec / 60);
        var s = Math.floor(sec % 60);
        var h = Math.floor(m / 60);
        if (h > 0) return h + ":" + String(m % 60).padStart(2, "0") + ":" + String(s).padStart(2, "0");
        return m + ":" + String(s).padStart(2, "0");
    }
    // Override onJobDone to handle special result types
    var _origOnJobDone = onJobDone;
    onJobDone = function (job) {
        // Handle transcript result for editor
        if (job.type === "transcript" && job.status === "complete" && job.result) {
            transcriptData = job.result;
            renderTranscriptEditor(job.result);
        }

        // Handle beat results
        if (job.type === "beats" && job.status === "complete" && job.result) {
            el.beatResults.classList.remove("hidden");
            el.bpmValue.textContent = job.result.bpm.toFixed(0);
            el.beatCount.textContent = job.result.total_beats;
            el.beatConfidence.textContent = (job.result.confidence * 100).toFixed(0) + "%";
        }

        // Handle scene results
        if (job.type === "scenes" && job.status === "complete" && job.result) {
            el.sceneResults.classList.remove("hidden");
            el.sceneCount.textContent = job.result.total_scenes;
            el.avgSceneLen.textContent = job.result.avg_scene_length + "s";
            if (job.result.youtube_chapters) {
                el.ytChapters.classList.remove("hidden");
                el.ytChaptersText.value = job.result.youtube_chapters;
            }
        }

        // Handle voice generation results - show preview player
        if (job.type === "voice-generate" && job.status === "complete" && job.result) {
            showVoicePreview(
                job.result.output_path,
                job.result.duration_seconds,
                job.result.model_used
            );
        }

        // Handle voice replace results
        if (job.type === "voice-replace" && job.status === "complete" && job.result) {
            showVoicePreview(
                job.result.output_path,
                job.result.duration_seconds,
                ""
            );
        }

        // Handle voice install completion
        if (job.type === "install-voice" && job.status === "complete") {
            voiceInstalled = true;
            updateButtons();
            showAlert("Qwen3-TTS installed successfully!");
        }

        // Handle video AI install completions
        if (job.type === "install-rembg" && job.status === "complete") {
            rembgInstalled = true;
            updateButtons();
            showAlert("rembg installed successfully!");
        }
        if (job.type === "install-rife" && job.status === "complete") {
            rifeInstalled = true;
            updateButtons();
            showAlert("RIFE installed successfully!");
        }
        if (job.type === "install-esrgan" && job.status === "complete") {
            esrganInstalled = true;
            updateButtons();
            showAlert("Real-ESRGAN installed successfully!");
        }

        _origOnJobDone(job);
    };

    // ================================================================
    // Transcript Editor
    // ================================================================
    function renderTranscriptEditor(data) {
        el.transcriptEditor.classList.remove("hidden");
        el.transcriptInfo.textContent = data.word_count + " words | " + data.segments.length + " segments | " + (data.language || "en");

        var html = "";
        for (var i = 0; i < data.segments.length; i++) {
            var seg = data.segments[i];
            var timeStr = fmtDur(seg.start) + " - " + fmtDur(seg.end);
            html += '<div class="transcript-seg" data-idx="' + i + '">'
                + '<div class="transcript-seg-time">' + timeStr + '</div>'
                + '<textarea class="transcript-seg-text" data-idx="' + i + '" rows="1">' + esc(seg.text) + '</textarea>'
                + '</div>';
        }
        el.transcriptSegments.innerHTML = html;

        // Auto-resize textareas
        var textareas = el.transcriptSegments.querySelectorAll(".transcript-seg-text");
        for (var i = 0; i < textareas.length; i++) {
            autoResize(textareas[i]);
            textareas[i].addEventListener("input", function () {
                autoResize(this);
                var idx = parseInt(this.getAttribute("data-idx"));
                if (transcriptData && transcriptData.segments[idx]) {
                    transcriptData.segments[idx].text = this.value;
                }
            });
        }
    }

    function autoResize(textarea) {
        textarea.style.height = "auto";
        textarea.style.height = textarea.scrollHeight + "px";
    }

    // ================================================================
    // Style Preview
    // ================================================================
    function loadStylePreview() {
        api("GET", "/caption-styles", null, function (err, data) {
            if (!err && data && data.styles) {
                for (var i = 0; i < data.styles.length; i++) {
                    var s = data.styles[i];
                    stylePreviewMap[s.name] = s.preview_css || "";
                }
                updateStylePreview();
            }
        });
    }

    function updateStylePreview() {
        var styleName = el.captionStyle.value;
        var css = stylePreviewMap[styleName] || "";
        var previewBg = el.stylePreview.querySelector(".style-preview-bg");
        if (previewBg) {
            previewBg.style.cssText = ""; // Reset
            if (css) {
                // Apply CSS to each word
                var words = previewBg.querySelectorAll(".sp-word");
                for (var i = 0; i < words.length; i++) {
                    words[i].style.cssText = css;
                }
                // Highlight color for current word
                var hlWord = previewBg.querySelector(".sp-highlight");
                if (hlWord) {
                    hlWord.style.cssText = css;
                    hlWord.style.color = "#ffe600";
                    hlWord.style.transform = "scale(1.1)";
                }
                // Action color
                var actWord = previewBg.querySelector(".sp-action");
                if (actWord) {
                    actWord.style.cssText = css;
                    actWord.style.color = "#ff3232";
                    actWord.style.transform = "scale(1.05)";
                }
            }
        }
    }

    // ================================================================
    // Settings Info
    // ================================================================
    var settingsLoaded = false;

    function loadSettingsInfo() {
        if (settingsLoaded) return;
        settingsLoaded = true;

        // Whisper status
        api("GET", "/health", null, function (err, data) {
            if (!err && data) {
                if (data.capabilities && data.capabilities.captions) {
                    el.whisperStatusText.textContent = "Installed (" + (data.capabilities.whisper_backend || "unknown") + ")";
                    el.whisperStatusText.style.color = "var(--success)";
                } else {
                    el.whisperStatusText.textContent = "Not installed";
                    el.whisperStatusText.style.color = "var(--error)";
                }
            }
        });

        // GPU info
        api("GET", "/system/gpu", null, function (err, data) {
            if (!err && data) {
                el.gpuName.textContent = data.available ? data.name : "None detected";
                el.gpuVram.textContent = data.available ? (data.vram_mb / 1024).toFixed(1) + " GB" : "--";
            }
        });
    }

    function installWhisper() {
        showAlert("Installing faster-whisper... This may take a minute.");
        api("POST", "/install-whisper", { backend: "faster-whisper" }, function (err, data) {
            if (!err && data && data.success) {
                showAlert("Whisper installed successfully! Refresh the panel.");
                setTimeout(checkHealth, 2000);
            } else {
                showAlert("Install failed: " + (data ? data.error : "Network error"));
            }
        }, 120000);
    }

    // ================================================================
    // Slider Handlers
    // ================================================================
    function setupSliders() {
        // Silence sliders
        el.threshold.addEventListener("input", function () { el.thresholdVal.textContent = this.value + " dB"; });
        el.minDuration.addEventListener("input", function () { el.minDurationVal.textContent = this.value + "s"; });
        el.silencePreset.addEventListener("change", function () {
            el.customSilenceSettings.classList.toggle("hidden", this.value !== "");
        });

        // Audio sliders
        el.denoiseStrength.addEventListener("input", function () { el.denoiseStrengthVal.textContent = this.value; });
        el.beatSensitivity.addEventListener("input", function () { el.beatSensitivityVal.textContent = this.value; });
        
        // Ducking sliders
        el.duckLevel.addEventListener("input", function () { el.duckLevelVal.textContent = this.value + " dB"; });
        el.duckAttack.addEventListener("input", function () { el.duckAttackVal.textContent = this.value; });
        el.duckRelease.addEventListener("input", function () { el.duckReleaseVal.textContent = this.value; });

        // Video sliders
        el.sceneThreshold.addEventListener("input", function () { el.sceneThresholdVal.textContent = parseFloat(this.value).toFixed(2); });
        el.minSceneLen.addEventListener("input", function () { el.minSceneLenVal.textContent = this.value + "s"; });

        // Voice sliders
        el.ttsSpeed.addEventListener("input", function () { el.ttsSpeedVal.textContent = parseFloat(this.value).toFixed(1) + "x"; });
        
        // Denoise method descriptions
        el.denoiseMethod.addEventListener("change", function () {
            var descs = {
                "afftdn": "Adaptive FFT-based noise reduction using FFmpeg.",
                "highpass": "Simple bandpass filter for rumble and hiss removal.",
                "deepfilter": "AI-powered DeepFilterNet3 for high-quality noise reduction. Downloads model on first use (~40MB)."
            };
            el.denoiseMethodDesc.textContent = descs[this.value] || "";
        });
        
        // Isolate method UI toggle
        el.isolateMethod.addEventListener("change", function () {
            var isDemucs = this.value === "demucs";
            el.stemOptions.classList.toggle("hidden", !isDemucs);
            var descs = {
                "bandpass": "Emphasize voice frequencies and reduce background using bandpass filtering.",
                "demucs": "AI-powered Demucs v4 for high-quality stem separation. Downloads model on first use (~80MB)."
            };
            el.isolateMethodDesc.textContent = descs[this.value] || "";
        });
        
        // EQ preset descriptions
        el.eqPreset.addEventListener("change", function () {
            var descs = {
                "voice_enhance": "Clarity and presence for speech",
                "podcast_voice": "Warm, clear voice for podcasts",
                "broadcast": "Professional broadcast EQ",
                "bass_boost": "Enhanced low-end frequencies",
                "treble_boost": "Enhanced high-end frequencies",
                "warmth": "Add analog warmth to audio",
                "presence": "Boost vocal presence and clarity",
                "telephone": "Lo-fi telephone/radio effect"
            };
            el.eqPresetDesc.textContent = descs[this.value] || "";
        });
        
        // Music select for ducking
        el.musicSelect.addEventListener("change", function () {
            updateButtons();
        });
    }
    
    function populateMusicSelect() {
        // Populate the music track select with project media
        if (!projectMedia || projectMedia.length === 0) return;
        
        var html = '<option value="" disabled selected>Select music file...</option>';
        for (var i = 0; i < projectMedia.length; i++) {
            var m = projectMedia[i];
            if (m.path && m.path !== selectedPath) {
                var name = m.name || m.path.split(/[/\\]/).pop();
                html += '<option value="' + esc(m.path) + '">' + esc(name) + '</option>';
            }
        }
        el.musicSelect.innerHTML = html;
    }

    // ================================================================
    // Utility
    // ================================================================
    function showAlert(msg) {
        el.alertText.textContent = msg;
        el.alertBanner.classList.remove("hidden");
        setTimeout(function () { el.alertBanner.classList.add("hidden"); }, 6000);
    }

    function esc(s) {
        if (!s) return "";
        return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
    }

    function fmtDur(s) {
        if (!s && s !== 0) return "--";
        var m = Math.floor(s / 60);
        var sec = Math.floor(s % 60);
        return m + ":" + (sec < 10 ? "0" : "") + sec;
    }

    // ================================================================
    // Init
    // ================================================================
    document.addEventListener("DOMContentLoaded", function () {
        initCSInterface();
        initDOM();
        setupNavTabs();
        setupSliders();

        // Event listeners - Clip selection
        el.clipSelect.addEventListener("change", function () {
            var opt = this.options[this.selectedIndex];
            if (opt.value) selectFile(opt.value, opt.getAttribute("data-name") || opt.value.split(/[/\\]/).pop());
        });
        el.refreshClipsBtn.addEventListener("click", scanProjectMedia);
        el.useSelectionBtn.addEventListener("click", useTimelineSelection);
        el.browseFileBtn.addEventListener("click", browseForFile);

        // Cut tab buttons
        el.runSilenceBtn.addEventListener("click", runSilence);
        el.runFillersBtn.addEventListener("click", runFillers);
        el.runFullBtn.addEventListener("click", runFull);

        // Captions tab buttons
        el.runStyledCaptionsBtn.addEventListener("click", runStyledCaptions);
        el.runSubtitleBtn.addEventListener("click", runSubtitle);
        el.runTranscriptBtn.addEventListener("click", runTranscript);
        el.exportTranscriptBtn.addEventListener("click", exportEditedTranscript);
        el.installWhisperBtn.addEventListener("click", installWhisper);
        el.captionStyle.addEventListener("change", updateStylePreview);

        // Audio tab buttons
        el.runDenoiseBtn.addEventListener("click", runDenoise);
        el.runIsolateBtn.addEventListener("click", runIsolate);
        el.runEqBtn.addEventListener("click", runEq);
        el.runDuckBtn.addEventListener("click", runDuck);
        el.measureLoudnessBtn.addEventListener("click", measureLoudness);
        el.runNormalizeBtn.addEventListener("click", runNormalize);
        el.runBeatsBtn.addEventListener("click", runBeats);
        el.runEffectBtn.addEventListener("click", runEffect);

        // Video tab buttons
        el.runScenesBtn.addEventListener("click", runScenes);
        el.copyChaptersBtn.addEventListener("click", function () {
            var text = el.ytChaptersText.value;
            if (navigator.clipboard) {
                navigator.clipboard.writeText(text).then(function () { showAlert("Copied to clipboard!"); });
            } else {
                el.ytChaptersText.select();
                document.execCommand("copy");
                showAlert("Copied to clipboard!");
            }
        });
        el.runSpeedRampBtn.addEventListener("click", runSpeedRamp);
        el.runReframeBtn.addEventListener("click", runReframe);
        el.runColorBtn.addEventListener("click", runColor);
        el.runChromaBtn.addEventListener("click", runChromaKey);
        el.runBgRemoveBtn.addEventListener("click", runBgRemove);
        el.installRembgBtn.addEventListener("click", installRembg);
        el.runSlowmoBtn.addEventListener("click", runSlowmo);
        el.installRifeBtn.addEventListener("click", installRife);
        el.runUpscaleBtn.addEventListener("click", runUpscale);
        el.installEsrganBtn.addEventListener("click", installEsrgan);

        // Video slider feedback
        el.colorIntensity.addEventListener("input", function () { el.colorIntensityVal.textContent = parseFloat(this.value).toFixed(2); });
        el.chromaSimilarity.addEventListener("input", function () { el.chromaSimilarityVal.textContent = parseFloat(this.value).toFixed(2); });
        el.chromaBlend.addEventListener("input", function () { el.chromaBlendVal.textContent = parseFloat(this.value).toFixed(2); });

        // Voice tab buttons
        el.runTtsBtn.addEventListener("click", runTts);
        el.runCloneBtn.addEventListener("click", runClone);
        el.runDesignBtn.addEventListener("click", runDesign);
        el.runReplaceBtn.addEventListener("click", runReplace);
        el.installVoiceBtn.addEventListener("click", installVoice);
        el.saveProfileBtn.addEventListener("click", saveVoiceProfile);

        // Clone source selection
        el.cloneSource.addEventListener("change", function () {
            var val = this.value;
            if (val === "browse") {
                if (inPremiere) {
                    jsx("browseForFile()", function (result) {
                        if (result && result !== "null" && result.length > 3) {
                            cloneRefAudioPath = result;
                            el.cloneSourceInfo.textContent = "File: " + result.split(/[/\\]/).pop();
                            updateButtons();
                        }
                    });
                }
            } else if (val === "selected") {
                if (selectedPath) {
                    cloneRefAudioPath = selectedPath;
                    el.cloneSourceInfo.textContent = "Using: " + selectedName;
                    updateButtons();
                } else {
                    showAlert("No clip selected.");
                }
            } else if (val.indexOf("profile:") === 0) {
                var pid = val.replace("profile:", "");
                for (var i = 0; i < voiceProfiles.length; i++) {
                    if (voiceProfiles[i].id === pid) {
                        cloneRefAudioPath = voiceProfiles[i].ref_audio_path;
                        el.cloneRefText.value = voiceProfiles[i].ref_text || "";
                        el.cloneSourceInfo.textContent = "Profile: " + voiceProfiles[i].name;
                        updateButtons();
                        break;
                    }
                }
            }
        });

        // Export tab buttons
        el.runExpTranscriptBtn.addEventListener("click", runExpTranscript);
        el.runPlatformExportBtn.addEventListener("click", runPlatformExport);
        el.runCustomRenderBtn.addEventListener("click", runCustomRender);
        el.runThumbBtn.addEventListener("click", runThumbnail);
        el.runBurninBtn.addEventListener("click", runBurnin);
        el.browseBurninSubBtn.addEventListener("click", browseBurninSub);
        el.runWatermarkBtn.addEventListener("click", runWatermark);
        el.runGifBtn.addEventListener("click", runGif);
        el.runAudioExtractBtn.addEventListener("click", runAudioExtract);

        // Export slider/select feedback
        el.platformPreset.addEventListener("change", updatePlatformInfo);
        el.renderCrf.addEventListener("input", function () { el.renderCrfVal.textContent = this.value; });
        el.thumbMode.addEventListener("change", updateThumbMode);
        el.thumbCount.addEventListener("input", function () { el.thumbCountVal.textContent = this.value; });
        el.burninFontSize.addEventListener("input", function () { el.burninFontSizeVal.textContent = this.value; });
        el.watermarkOpacity.addEventListener("input", function () { el.watermarkOpacityVal.textContent = parseFloat(this.value).toFixed(2); });
        updatePlatformInfo();
        updateThumbMode();

        // Batch tab buttons
        el.browseBatchFolderBtn.addEventListener("click", browseBatchFolder);
        el.runBatchBtn.addEventListener("click", runBatch);
        el.cancelBatchBtn.addEventListener("click", cancelBatch);
        el.addWorkflowStepBtn.addEventListener("click", addWorkflowStep);
        el.clearWorkflowBtn.addEventListener("click", clearWorkflow);
        el.runWorkflowBtn.addEventListener("click", runWorkflow);
        el.browseWatchFolderBtn.addEventListener("click", browseWatchFolder);
        el.startWatchBtn.addEventListener("click", startWatch);
        el.stopWatchBtn.addEventListener("click", stopWatch);
        el.runInspectBtn.addEventListener("click", runInspect);
        renderWorkflowSteps();

        // Settings tab buttons
        el.settingsInstallWhisperBtn.addEventListener("click", installWhisper);

        // Progress / Results
        el.cancelBtn.addEventListener("click", cancelJob);
        el.newJobBtn.addEventListener("click", function () {
            el.resultsSection.classList.add("hidden");
        });

        // Alert dismiss
        el.alertDismiss.addEventListener("click", function () {
            el.alertBanner.classList.add("hidden");
        });

        // Health check loop
        checkHealth();
        healthTimer = setInterval(checkHealth, HEALTH_MS);

        // Scan project media
        scanProjectMedia();

        // Load style preview data
        loadStylePreview();

        // Check voice lab status
        checkVoiceStatus();
        loadVoiceProfiles();

        // Check video AI tool status
        checkRembgStatus();
        checkRifeStatus();
        checkEsrganStatus();
    });

})();
