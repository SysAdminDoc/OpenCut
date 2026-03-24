/* ============================================================
   OpenCut CEP Panel - Main Controller v1.4.0
   6-Tab Professional Toolkit
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
    var csrfToken = "";
    var lastXmlPath = "";
    var lastCaptionPath = "";
    var lastOverlayPath = "";
    var projectMedia = [];
    var projectFolder = "";
    var backendStartAttempted = false;
    var jobStartTime = 0;
    var elapsedTimer = null;
    var activeStream = null;
    var batchPollTimer = null;
    var transcriptData = null; // stored transcript for editing/export
    var lastJobEndpoint = "";  // for retry
    var lastJobPayload = null; // for retry

    // ---- Style Preview CSS Map (loaded from backend) ----
    var stylePreviewMap = {};

    // ---- New Feature State (v1.5.0) ----
    var lastTimelineCuts = null;    // stores last cuts result for timeline apply
    var sequenceInfo = null;        // stores loaded sequence info for deliverables
    var footageIndex = {};          // stores local copy of index stats
    var beatMarkerTimes = null;     // beat times for marker insertion
    var seqMarkersData = null;      // markers from sequence for export
    var renameItemsData = [];       // project items for rename
    var multicamCutsData = null;    // multicam cut result
    var repeatCutsData = null;      // repeat-detect cuts data
    var chaptersData = null;        // generated chapters

    // ============================================================
    // CUSTOM DROPDOWN SYSTEM - Inline Panel Dropdowns
    // ============================================================
    var dropdownGlobalListenersAdded = false;

    function initCustomDropdowns() {
        var selects = document.querySelectorAll('select:not(.no-custom)');
        for (var i = 0; i < selects.length; i++) {
            var select = selects[i];
            if (select.dataset.customized) continue;
            createCustomDropdown(select);
        }

        // Register global listeners only once
        if (!dropdownGlobalListenersAdded) {
            dropdownGlobalListenersAdded = true;
            document.addEventListener('click', function(e) {
                if (!e.target.closest('.custom-dropdown')) {
                    closeAllDropdowns();
                }
            });
            document.addEventListener('keydown', function(e) {
                if (e.key === 'Escape') {
                    closeAllDropdowns();
                }
            });
        }
    }
    
    function createCustomDropdown(select) {
        select.dataset.customized = 'true';
        select.style.display = 'none';
        
        var wrapper = document.createElement('div');
        wrapper.className = 'custom-dropdown';
        if (select.id) wrapper.dataset.for = select.id;
        
        var trigger = document.createElement('div');
        trigger.className = 'custom-dropdown-trigger';
        trigger.tabIndex = 0;
        trigger.setAttribute("role", "combobox");
        trigger.setAttribute("aria-expanded", "false");
        trigger.setAttribute("aria-haspopup", "listbox");
        
        var selectedText = document.createElement('span');
        selectedText.className = 'custom-dropdown-text';
        
        var arrow = document.createElement('span');
        arrow.className = 'custom-dropdown-arrow';
        arrow.innerHTML = '<svg width="10" height="10" viewBox="0 0 16 16" fill="currentColor"><path d="M8 11L3 6h10l-5 5z"/></svg>';
        
        trigger.appendChild(selectedText);
        trigger.appendChild(arrow);
        
        var dropdown = document.createElement('div');
        dropdown.className = 'custom-dropdown-menu';
        dropdown.setAttribute("role", "listbox");
        
        function buildOptions() {
            dropdown.innerHTML = '';
            var hasOptgroups = select.querySelector('optgroup');
            var i, j, child, opt;
            
            if (hasOptgroups) {
                for (i = 0; i < select.children.length; i++) {
                    child = select.children[i];
                    if (child.tagName === 'OPTGROUP') {
                        var groupLabel = document.createElement('div');
                        groupLabel.className = 'custom-dropdown-group';
                        groupLabel.textContent = child.label;
                        dropdown.appendChild(groupLabel);
                        
                        for (j = 0; j < child.children.length; j++) {
                            dropdown.appendChild(createOption(child.children[j]));
                        }
                    } else if (child.tagName === 'OPTION') {
                        dropdown.appendChild(createOption(child));
                    }
                }
            } else {
                for (i = 0; i < select.options.length; i++) {
                    dropdown.appendChild(createOption(select.options[i]));
                }
            }
            updateSelectedText();
        }
        
        function createOption(opt) {
            var item = document.createElement('div');
            item.className = 'custom-dropdown-item';
            item.setAttribute("role", "option");
            if (opt.disabled) item.classList.add('disabled');
            if (opt.selected) item.classList.add('selected');
            item.dataset.value = opt.value;
            item.textContent = opt.textContent;
            
            item.addEventListener('click', function(e) {
                e.stopPropagation();
                if (item.classList.contains('disabled')) return;
                
                // Set the select value
                select.value = item.dataset.value;
                
                // Fire change event (compatible with older browsers/CEP)
                var evt;
                try {
                    evt = new Event('change', { bubbles: true });
                } catch (err) {
                    evt = document.createEvent('Event');
                    evt.initEvent('change', true, true);
                }
                select.dispatchEvent(evt);
                
                // Update visual selection
                var allItems = dropdown.querySelectorAll('.custom-dropdown-item');
                for (var i = 0; i < allItems.length; i++) {
                    allItems[i].classList.remove('selected');
                }
                item.classList.add('selected');
                updateSelectedText();
                closeDropdown();
            });
            
            return item;
        }
        
        function updateSelectedText() {
            var selected = select.options[select.selectedIndex];
            if (selected) {
                selectedText.textContent = selected.textContent;
                selectedText.classList.toggle('placeholder', selected.disabled);
            }
        }
        
        function toggleDropdown(e) {
            e.stopPropagation();
            var isOpen = wrapper.classList.contains('open');
            closeAllDropdowns();
            if (!isOpen) {
                wrapper.classList.add('open');
                trigger.setAttribute("aria-expanded", "true");
                positionDropdown();

                // Scroll to selected item
                var selectedItem = dropdown.querySelector('.custom-dropdown-item.selected');
                if (selectedItem) {
                    selectedItem.scrollIntoView({ block: 'nearest' });
                }
            }
        }
        
        function closeDropdown() {
            wrapper.classList.remove('open');
            trigger.setAttribute("aria-expanded", "false");
            var focused = dropdown.querySelector('.custom-dropdown-item.focused');
            if (focused) focused.classList.remove('focused');
        }
        
        function positionDropdown() {
            // Reset position
            dropdown.style.top = '';
            dropdown.style.bottom = '';
            dropdown.style.maxHeight = '';
            
            var rect = wrapper.getBoundingClientRect();
            var menuHeight = dropdown.scrollHeight;
            var viewportHeight = window.innerHeight;
            var spaceBelow = viewportHeight - rect.bottom - 10;
            var spaceAbove = rect.top - 10;
            
            // Determine if dropdown should open upward
            if (spaceBelow < menuHeight && spaceAbove > spaceBelow) {
                dropdown.classList.add('dropup');
                dropdown.style.maxHeight = Math.min(menuHeight, spaceAbove) + 'px';
            } else {
                dropdown.classList.remove('dropup');
                dropdown.style.maxHeight = Math.min(menuHeight, spaceBelow, 250) + 'px';
            }
        }
        
        trigger.addEventListener('click', toggleDropdown);

        var typeSearchBuffer = '';
        var typeSearchTimer = null;

        trigger.addEventListener('keydown', function(e) {
            var isOpen = wrapper.classList.contains('open');

            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                if (isOpen) {
                    // Select the focused item
                    var focused = dropdown.querySelector('.custom-dropdown-item.focused');
                    if (focused) focused.click();
                    else closeDropdown();
                } else {
                    toggleDropdown(e);
                }
                return;
            }

            if (e.key === 'Escape') {
                e.preventDefault();
                closeDropdown();
                return;
            }

            if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
                e.preventDefault();
                if (!isOpen) {
                    toggleDropdown(e);
                    return;
                }
                var items = dropdown.querySelectorAll('.custom-dropdown-item:not(.disabled)');
                if (!items.length) return;
                var focusedIdx = -1;
                for (var fi = 0; fi < items.length; fi++) {
                    if (items[fi].classList.contains('focused')) { focusedIdx = fi; break; }
                }
                // Clear old focus
                if (focusedIdx >= 0) items[focusedIdx].classList.remove('focused');
                // Calculate new index
                if (e.key === 'ArrowDown') {
                    focusedIdx = (focusedIdx + 1) % items.length;
                } else {
                    focusedIdx = focusedIdx <= 0 ? items.length - 1 : focusedIdx - 1;
                }
                items[focusedIdx].classList.add('focused');
                items[focusedIdx].scrollIntoView({ block: 'nearest' });
                return;
            }

            // Type-to-search: match visible option text
            if (e.key.length === 1 && !e.ctrlKey && !e.altKey && !e.metaKey) {
                e.preventDefault();
                typeSearchBuffer += e.key.toLowerCase();
                if (typeSearchTimer) clearTimeout(typeSearchTimer);
                typeSearchTimer = setTimeout(function () { typeSearchBuffer = ''; }, 600);

                if (!isOpen) toggleDropdown(e);

                var items = dropdown.querySelectorAll('.custom-dropdown-item:not(.disabled)');
                for (var ti = 0; ti < items.length; ti++) {
                    if (items[ti].textContent.toLowerCase().indexOf(typeSearchBuffer) === 0) {
                        // Clear old focus
                        var oldFocus = dropdown.querySelector('.custom-dropdown-item.focused');
                        if (oldFocus) oldFocus.classList.remove('focused');
                        items[ti].classList.add('focused');
                        items[ti].scrollIntoView({ block: 'nearest' });
                        break;
                    }
                }
            }
        });
        
        wrapper.appendChild(trigger);
        wrapper.appendChild(dropdown);
        select.parentNode.insertBefore(wrapper, select.nextSibling);
        
        buildOptions();
        
        // Watch for external changes to select
        var observer = new MutationObserver(function() {
            buildOptions();
        });
        observer.observe(select, { childList: true, subtree: true, attributes: true });
        
        // Store reference for updating and cleanup
        select._customDropdown = {
            wrapper: wrapper,
            update: buildOptions,
            updateText: updateSelectedText,
            observer: observer
        };
    }
    
    function closeAllDropdowns() {
        var openDropdowns = document.querySelectorAll('.custom-dropdown.open');
        for (var i = 0; i < openDropdowns.length; i++) {
            openDropdowns[i].classList.remove('open');
            var trig = openDropdowns[i].querySelector('.custom-dropdown-trigger');
            if (trig) trig.setAttribute("aria-expanded", "false");
        }
    }
    
    function updateCustomDropdown(selectId) {
        var select = document.getElementById(selectId);
        if (select && select._customDropdown) {
            select._customDropdown.update();
        }
    }

    // ---- DOM (Lazy Proxy — elements are cached on first access) ----
    var _elCache = {};
    var el = new Proxy(_elCache, {
        get: function (target, id) {
            if (id in target) return target[id];
            var node = document.getElementById(id);
            if (node) target[id] = node;
            return node;
        }
    });
    function $(id) { return document.getElementById(id); }

    function initDOM() {
        // Content header
        el.contentTitle = $("contentTitle");
        el.connDot = $("connDot");
        el.connLabel = $("connLabel");
        el.refreshAllBtn = $("refreshAllBtn");
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
        el.transcriptUndoBtn = $("transcriptUndoBtn");
        el.transcriptRedoBtn = $("transcriptRedoBtn");

        // Audio tab - Separation
        el.separateModel = $("separateModel");
        el.stemVocals = $("stemVocals");
        el.stemInstrumental = $("stemInstrumental");
        el.stemDrums = $("stemDrums");
        el.stemBass = $("stemBass");
        el.stemOther = $("stemOther");
        el.separateFormat = $("separateFormat");
        el.separateImport = $("separateImport");
        el.runSeparateBtn = $("runSeparateBtn");
        el.separateHint = $("separateHint");
        el.installDemucsBtn = $("installDemucsBtn");
        
        // Audio tab - Other
        el.denoiseMethod = $("denoiseMethod");
        el.denoiseStrength = $("denoiseStrength");
        el.denoiseStrengthVal = $("denoiseStrengthVal");
        el.runDenoiseBtn = $("runDenoiseBtn");
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

        // Video tab - Watermark Removal
        el.wmMaxBbox = $("wmMaxBbox");
        el.wmMaxBboxVal = $("wmMaxBboxVal");
        el.wmPrompt = $("wmPrompt");
        el.wmVideoOptions = $("wmVideoOptions");
        el.wmFrameSkip = $("wmFrameSkip");
        el.wmFrameSkipVal = $("wmFrameSkipVal");
        el.wmTransparent = $("wmTransparent");
        el.wmPreview = $("wmPreview");
        el.wmAutoImport = $("wmAutoImport");
        el.runWatermarkBtn = $("runWatermarkBtn");
        el.watermarkHint = $("watermarkHint");
        el.installWatermarkBtn = $("installWatermarkBtn");
        
        // Video tab - Scene Detection
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

        // Video tab - Effects (FFmpeg)
        el.vfxSelect = $("vfxSelect");
        el.vfxAutoImport = $("vfxAutoImport");
        el.runVfxBtn = $("runVfxBtn");
        el.vfxStabSmoothing = $("vfxStabSmoothing");
        el.vfxStabSmoothingVal = $("vfxStabSmoothingVal");
        el.vfxStabZoom = $("vfxStabZoom");
        el.vfxStabZoomVal = $("vfxStabZoomVal");
        el.vfxVignetteIntensity = $("vfxVignetteIntensity");
        el.vfxVignetteIntensityVal = $("vfxVignetteIntensityVal");
        el.vfxGrainIntensity = $("vfxGrainIntensity");
        el.vfxGrainIntensityVal = $("vfxGrainIntensityVal");
        el.vfxLetterboxAspect = $("vfxLetterboxAspect");
        el.vfxChromakeyColor = $("vfxChromakeyColor");
        el.vfxChromakeySim = $("vfxChromakeySim");
        el.vfxChromakeySimVal = $("vfxChromakeySimVal");
        el.vfxChromakeyBlend = $("vfxChromakeyBlend");
        el.vfxChromakeyBlendVal = $("vfxChromakeyBlendVal");
        el.vfxLutPath = $("vfxLutPath");
        el.vfxLutIntensity = $("vfxLutIntensity");
        el.vfxLutIntensityVal = $("vfxLutIntensityVal");

        // Video tab - AI Tools
        el.vidAiTool = $("vidAiTool");
        el.vidAiAutoImport = $("vidAiAutoImport");
        el.runVidAiBtn = $("runVidAiBtn");
        el.vidAiHint = $("vidAiHint");
        el.vidAiHintText = $("vidAiHintText");
        el.installVidAiBtn = $("installVidAiBtn");
        el.vidAiUpscaleScale = $("vidAiUpscaleScale");
        el.vidAiUpscaleModel = $("vidAiUpscaleModel");
        el.vidAiRembgModel = $("vidAiRembgModel");
        el.vidAiRembgBg = $("vidAiRembgBg");
        el.vidAiRembgAlpha = $("vidAiRembgAlpha");
        el.vidAiInterpMultiplier = $("vidAiInterpMultiplier");
        el.vidAiDenoiseMethod = $("vidAiDenoiseMethod");
        el.vidAiDenoiseStrength = $("vidAiDenoiseStrength");
        el.vidAiDenoiseStrengthVal = $("vidAiDenoiseStrengthVal");

        // Audio tab - Pro FX
        el.proFxCategory = $("proFxCategory");
        el.proFxEffect = $("proFxEffect");
        el.proFxParams = $("proFxParams");
        el.proFxAutoImport = $("proFxAutoImport");
        el.runProFxBtn = $("runProFxBtn");
        el.proFxHint = $("proFxHint");
        el.installPedalboardBtn = $("installPedalboardBtn");
        el.runDeepFilterBtn = $("runDeepFilterBtn");
        el.deepFilterAutoImport = $("deepFilterAutoImport");
        el.deepFilterHint = $("deepFilterHint");
        el.installDeepFilterBtn = $("installDeepFilterBtn");

        // Video tab - Face Blur
        el.faceBlurMethod = $("faceBlurMethod");
        el.faceBlurStrength = $("faceBlurStrength");
        el.faceBlurStrengthVal = $("faceBlurStrengthVal");
        el.faceDetector = $("faceDetector");
        el.faceBlurAutoImport = $("faceBlurAutoImport");
        el.runFaceBlurBtn = $("runFaceBlurBtn");
        el.faceHint = $("faceHint");
        el.installMediapipeBtn = $("installMediapipeBtn");

        // Video tab - Style Transfer
        el.styleModel = $("styleModel");
        el.styleIntensity = $("styleIntensity");
        el.styleIntensityVal = $("styleIntensityVal");
        el.styleAutoImport = $("styleAutoImport");
        el.runStyleBtn = $("runStyleBtn");

        // Captions tab - Translate
        el.translateModel = $("translateModel");
        el.translateSourceLang = $("translateSourceLang");
        el.translateTargetLang = $("translateTargetLang");
        el.translateFormat = $("translateFormat");
        el.runTranslateBtn = $("runTranslateBtn");
        el.translateHint = $("translateHint");
        el.installNllbBtn = $("installNllbBtn");

        // Captions tab - Karaoke
        el.karaokeModel = $("karaokeModel");
        el.karaokeFont = $("karaokeFont");
        el.karaokeFontSize = $("karaokeFontSize");
        el.karaokeFontSizeVal = $("karaokeFontSizeVal");
        el.karaokeDiarize = $("karaokeDiarize");
        el.runKaraokeBtn = $("runKaraokeBtn");
        el.karaokeHint = $("karaokeHint");
        el.installWhisperxBtn = $("installWhisperxBtn");

        // Export tab - Presets
        el.exportPresetCategory = $("exportPresetCategory");
        el.exportPresetSelect = $("exportPresetSelect");
        el.exportPresetDesc = $("exportPresetDesc");
        el.exportPresetAutoImport = $("exportPresetAutoImport");
        el.runExportPresetBtn = $("runExportPresetBtn");

        // Export tab - Thumbnails
        el.thumbCount = $("thumbCount");
        el.thumbWidth = $("thumbWidth");
        el.thumbUseFaces = $("thumbUseFaces");
        el.runThumbBtn = $("runThumbBtn");

        // Export tab - Batch
        el.batchOperation = $("batchOperation");
        el.runBatchBtn = $("runBatchBtn");
        el.batchResults = $("batchResults");
        el.batchStatusText = $("batchStatusText");

        // Workflow presets
        el.workflowPreset = $("workflowPreset");
        el.runWorkflowBtn = $("runWorkflowBtn");

        // Audio tab - TTS
        el.ttsEngine = $("ttsEngine");
        el.ttsVoice = $("ttsVoice");
        el.ttsRate = $("ttsRate");
        el.ttsRateVal = $("ttsRateVal");
        el.ttsText = $("ttsText");
        el.ttsAutoImport = $("ttsAutoImport");
        el.runTtsBtn = $("runTtsBtn");
        el.ttsHint = $("ttsHint");
        el.installEdgeTtsBtn = $("installEdgeTtsBtn");

        // Audio tab - SFX
        el.sfxType = $("sfxType");
        el.sfxPreset = $("sfxPreset");
        el.sfxPresetParams = $("sfxPresetParams");
        el.sfxToneParams = $("sfxToneParams");
        el.toneWaveform = $("toneWaveform");
        el.toneFreq = $("toneFreq");
        el.toneFreqVal = $("toneFreqVal");
        el.sfxDuration = $("sfxDuration");
        el.sfxDurationVal = $("sfxDurationVal");
        el.sfxAutoImport = $("sfxAutoImport");
        el.runSfxBtn = $("runSfxBtn");

        // Captions tab - Burn-in
        el.burninStyle = $("burninStyle");
        el.burninModel = $("burninModel");
        el.burninAutoImport = $("burninAutoImport");
        el.runBurninBtn = $("runBurninBtn");

        // Video tab - Speed
        el.speedMode = $("speedMode");
        el.speedConstantParams = $("speedConstantParams");
        el.speedRampParams = $("speedRampParams");
        el.speedMultiplier = $("speedMultiplier");
        el.speedMultiplierVal = $("speedMultiplierVal");
        el.speedMaintainPitch = $("speedMaintainPitch");
        el.speedRampPreset = $("speedRampPreset");
        el.speedAutoImport = $("speedAutoImport");
        el.runSpeedBtn = $("runSpeedBtn");

        // Video tab - LUT
        el.lutCategory = $("lutCategory");
        el.lutSelect = $("lutSelect");
        el.lutIntensity = $("lutIntensity");
        el.lutIntensityVal = $("lutIntensityVal");
        el.lutAutoImport = $("lutAutoImport");
        el.runLutBtn = $("runLutBtn");

        // Audio tab - Duck
        el.duckMusicPath = $("duckMusicPath");
        el.duckMusicVol = $("duckMusicVol");
        el.duckMusicVolVal = $("duckMusicVolVal");
        el.duckAmount = $("duckAmount");
        el.duckAmountVal = $("duckAmountVal");
        el.duckAutoImport = $("duckAutoImport");
        el.runDuckBtn = $("runDuckBtn");

        // Chromakey
        el.chromaMode = $("chromaMode");
        el.chromakeyParams = $("chromakeyParams");
        el.pipParams = $("pipParams");
        el.blendParams = $("blendParams");
        el.chromaColor = $("chromaColor");
        el.chromaBgPath = $("chromaBgPath");
        el.chromaTol = $("chromaTol");
        el.chromaTolVal = $("chromaTolVal");
        el.pipPath = $("pipPath");
        el.pipPosition = $("pipPosition");
        el.pipScale = $("pipScale");
        el.pipScaleVal = $("pipScaleVal");
        el.blendOverlay = $("blendOverlay");
        el.blendMode = $("blendMode");
        el.blendOpacity = $("blendOpacity");
        el.blendOpacityVal = $("blendOpacityVal");
        el.runChromaBtn = $("runChromaBtn");

        // Transitions
        el.transClipB = $("transClipB");
        el.transType = $("transType");
        el.transDur = $("transDur");
        el.transDurVal = $("transDurVal");
        el.runTransBtn = $("runTransBtn");

        // Particles
        el.particlePreset = $("particlePreset");
        el.particleDensity = $("particleDensity");
        el.particleDensityVal = $("particleDensityVal");
        el.runParticlesBtn = $("runParticlesBtn");

        // Titles
        el.titleText = $("titleText");
        el.titleSubtext = $("titleSubtext");
        el.titlePreset = $("titlePreset");
        el.titleDur = $("titleDur");
        el.titleDurVal = $("titleDurVal");
        el.titleFontSize = $("titleFontSize");
        el.titleFontSizeVal = $("titleFontSizeVal");
        el.runTitleOverlayBtn = $("runTitleOverlayBtn");
        el.runTitleCardBtn = $("runTitleCardBtn");

        // Upscale
        el.upscalePreset = $("upscalePreset");
        el.upscaleScale = $("upscaleScale");
        el.runUpscaleBtn = $("runUpscaleBtn");

        // Reframe
        el.reframePreset = $("reframePreset");
        el.reframeCustomDims = $("reframeCustomDims");
        el.reframeCustomW = $("reframeCustomW");
        el.reframeCustomH = $("reframeCustomH");
        el.reframeMode = $("reframeMode");
        el.reframeCropPosGroup = $("reframeCropPosGroup");
        el.reframeCropPos = $("reframeCropPos");
        el.reframePadColorGroup = $("reframePadColorGroup");
        el.reframePadColor = $("reframePadColor");
        el.reframeQuality = $("reframeQuality");
        el.reframeInfo = $("reframeInfo");
        el.runReframeBtn = $("runReframeBtn");

        // Color Correction
        el.ccExposure = $("ccExposure"); el.ccExposureVal = $("ccExposureVal");
        el.ccContrast = $("ccContrast"); el.ccContrastVal = $("ccContrastVal");
        el.ccSaturation = $("ccSaturation"); el.ccSaturationVal = $("ccSaturationVal");
        el.ccTemp = $("ccTemp"); el.ccTempVal = $("ccTempVal");
        el.ccShadows = $("ccShadows"); el.ccShadowsVal = $("ccShadowsVal");
        el.ccHighlights = $("ccHighlights"); el.ccHighlightsVal = $("ccHighlightsVal");
        el.runColorBtn = $("runColorBtn");

        // Object Removal
        el.removeMethod = $("removeMethod");
        el.removeX = $("removeX"); el.removeY = $("removeY");
        el.removeW = $("removeW"); el.removeH = $("removeH");
        el.runRemoveBtn = $("runRemoveBtn");

        // Face AI
        el.faceAiMode = $("faceAiMode");
        el.faceSwapParams = $("faceSwapParams");
        el.faceRefPath = $("faceRefPath");
        el.runFaceAiBtn = $("runFaceAiBtn");

        // Animated Captions
        el.animCapPreset = $("animCapPreset");
        el.animCapFontSize = $("animCapFontSize"); el.animCapFontSizeVal = $("animCapFontSizeVal");
        el.animCapWpl = $("animCapWpl"); el.animCapWplVal = $("animCapWplVal");
        el.animCapModel = $("animCapModel");
        el.runAnimCapBtn = $("runAnimCapBtn");

        // Music AI
        el.musicAiPrompt = $("musicAiPrompt");
        el.musicAiModel = $("musicAiModel");
        el.musicAiDur = $("musicAiDur"); el.musicAiDurVal = $("musicAiDurVal");
        el.musicAiTemp = $("musicAiTemp"); el.musicAiTempVal = $("musicAiTempVal");
        el.musicAiAutoImport = $("musicAiAutoImport");
        el.runMusicAiBtn = $("runMusicAiBtn");

        // Export tab
        el.expTranscriptFormat = $("expTranscriptFormat");
        el.expModel = $("expModel");
        el.runExpTranscriptBtn = $("runExpTranscriptBtn");

        // Settings tab
        el.whisperStatusText = $("whisperStatusText");
        el.whisperDeviceText = $("whisperDeviceText");
        el.whisperCpuMode = $("whisperCpuMode");
        el.settingsDefaultModel = $("settingsDefaultModel");
        el.settingsInstallWhisperBtn = $("settingsInstallWhisperBtn");
        el.settingsReinstallWhisperBtn = $("settingsReinstallWhisperBtn");
        el.settingsClearCacheBtn = $("settingsClearCacheBtn");
        el.settingsAutoImport = $("settingsAutoImport");
        el.settingsTheme = $("settingsTheme");
        el.settingsAutoOpen = $("settingsAutoOpen");
        el.settingsShowNotifications = $("settingsShowNotifications");
        el.settingsOutputDir = $("settingsOutputDir");
        el.restartBackendBtn = $("restartBackendBtn");
        el.openLogsBtn = $("openLogsBtn");
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
        el.retryJobBtn = $("retryJobBtn");

        // Processing banner (persistent top bar)
        el.processingBanner = $("processingBanner");
        el.processingMsg = $("processingMsg");
        el.processingElapsed = $("processingElapsed");
        el.processingCancel = $("processingCancel");
        el.processingFill = $("processingFill");

        // Drop zone
        el.dropZone = $("dropZone");

        // Theme toggle
        el.themeToggleBtn = $("themeToggleBtn");
        el.themeMenu = $("themeMenu");

        // Job history
        el.jobHistoryToggle = $("jobHistoryToggle");
        el.jobHistory = $("jobHistory");

        // Presets
        el.presetNameInput = $("presetNameInput");
        el.savePresetBtn = $("savePresetBtn");
        el.presetSelect = $("presetSelect");
        el.loadPresetBtn = $("loadPresetBtn");
        el.deletePresetBtn = $("deletePresetBtn");

        // Model management
        el.modelList = $("modelList");
        el.modelsTotalSize = $("modelsTotalSize");
        el.refreshModelsBtn = $("refreshModelsBtn");

        // GPU recommendation
        el.getGpuRecBtn = $("getGpuRecBtn");
        el.gpuRecResults = $("gpuRecResults");
        el.gpuRecModel = $("gpuRecModel");
        el.gpuRecQuality = $("gpuRecQuality");
        el.gpuRecDevice = $("gpuRecDevice");
        el.gpuRecNotes = $("gpuRecNotes");
        el.applyGpuRecBtn = $("applyGpuRecBtn");

        // Job queue
        el.jobQueueBar = $("jobQueueBar");
        el.queueStatusText = $("queueStatusText");
        el.clearQueueBtn = $("clearQueueBtn");

        // Transcript search
        el.transcriptSearchInput = $("transcriptSearchInput");
        el.transcriptSearchCount = $("transcriptSearchCount");
        el.transcriptSearchPrev = $("transcriptSearchPrev");
        el.transcriptSearchNext = $("transcriptSearchNext");

        // v1.2.0 elements
        // Waveform
        el.waveformContainer = $("waveformContainer");
        el.waveformCanvas = $("waveformCanvas");
        el.waveformThreshold = $("waveformThreshold");
        el.loadWaveformBtn = $("loadWaveformBtn");
        // Favorites
        el.favoritesBar = $("favoritesBar");
        el.favoritesItems = $("favoritesItems");
        // Preview modal
        el.previewModal = $("previewModal");
        el.previewModalClose = $("previewModalClose");
        el.previewOriginal = $("previewOriginal");
        el.previewProcessed = $("previewProcessed");
        el.previewRefreshBtn = $("previewRefreshBtn");
        el.previewTimestamp = $("previewTimestamp");
        el.previewVfxBtn = $("previewVfxBtn");
        // Audio preview
        el.audioPreview = $("audioPreview");
        el.audioPreviewClose = $("audioPreviewClose");
        el.audioPreviewPlayer = $("audioPreviewPlayer");
        // Context menu
        el.contextMenu = $("contextMenu");
        // Wizard
        el.wizardOverlay = $("wizardOverlay");
        el.wizardCloseBtn = $("wizardCloseBtn");
        el.wizardDontShow = $("wizardDontShow");
        // Output browser
        el.outputBrowser = $("outputBrowser");
        el.outputBrowserToggle = $("outputBrowserToggle");
        el.outputBrowserClose = $("outputBrowserClose");
        el.outputBrowserList = $("outputBrowserList");
        el.refreshOutputsBtn = $("refreshOutputsBtn");
        // Batch multi-select
        el.batchFileList = $("batchFileList");
        el.batchAddSelectedBtn = $("batchAddSelectedBtn");
        el.batchAddAllBtn = $("batchAddAllBtn");
        el.batchClearBtn = $("batchClearBtn");
        // Dep dashboard
        el.depGrid = $("depGrid");
        el.refreshDepsBtn = $("refreshDepsBtn");
        // Settings import/export
        el.exportSettingsBtn = $("exportSettingsBtn");
        el.importSettingsBtn = $("importSettingsBtn");
        el.importSettingsFile = $("importSettingsFile");
        // Workflow builder
        el.customWorkflowName = $("customWorkflowName");
        el.workflowStepList = $("workflowStepList");
        el.workflowStepSelect = $("workflowStepSelect");
        el.workflowAddStepBtn = $("workflowAddStepBtn");
        el.saveCustomWorkflowBtn = $("saveCustomWorkflowBtn");
        el.runCustomWorkflowBtn = $("runCustomWorkflowBtn");
        el.savedWorkflowSelect = $("savedWorkflowSelect");
        el.loadCustomWorkflowBtn = $("loadCustomWorkflowBtn");
        el.deleteCustomWorkflowBtn = $("deleteCustomWorkflowBtn");
        // i18n
        el.settingsLang = $("settingsLang");
        // Time estimate
        el.processingEstimate = $("processingEstimate");

        // v1.3.0 - Clip Preview
        el.clipPreviewRow = $("clipPreviewRow");
        el.clipThumb = $("clipThumb");
        el.clipMeta = $("clipMeta");
        el.clipMetaRes = $("clipMetaRes");
        el.clipMetaDur = $("clipMetaDur");
        el.clipMetaSize = $("clipMetaSize");

        // v1.3.0 - Recent Clips
        el.recentClipsBtn = $("recentClipsBtn");
        el.recentClipsDropdown = $("recentClipsDropdown");

        // v1.3.0 - Command Palette
        el.commandPaletteOverlay = $("commandPaletteOverlay");
        el.commandPaletteInput = $("commandPaletteInput");
        el.commandPaletteResults = $("commandPaletteResults");

        // v1.3.0 - Trim
        el.trimStart = $("trimStart");
        el.trimEnd = $("trimEnd");
        el.trimMode = $("trimMode");
        el.trimQuality = $("trimQuality");
        el.trimQualityGroup = $("trimQualityGroup");
        el.runTrimBtn = $("runTrimBtn");

        // v1.3.0 - Merge
        el.mergeFileList = $("mergeFileList");
        el.mergeAddCurrentBtn = $("mergeAddCurrentBtn");
        el.mergeAddAllBtn = $("mergeAddAllBtn");
        el.mergeClearBtn = $("mergeClearBtn");
        el.mergeMode = $("mergeMode");
        el.mergeQuality = $("mergeQuality");
        el.runMergeBtn = $("runMergeBtn");

        // v1.3.0 - Server Status
        el.serverStatusBanner = $("serverStatusBanner");
        el.serverStatusMsg = $("serverStatusMsg");
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
    // UXP Bridge Abstraction Layer
    // ================================================================
    // Wraps all ExtendScript/CSInterface calls. When CEP is replaced by UXP,
    // only this object needs to change — all call sites use PremiereBridge.
    var PremiereBridge = {
        startBackend: function () {
            jsx("startOpenCutBackend()", function () {});
        },
        getProjectMedia: function (cb) {
            jsx("getProjectMedia()", cb);
        },
        getTimelineSelection: function (cb) {
            jsx("getTimelineSelection()", cb);
        },
        browseForFile: function (cb) {
            jsx("browseForFile()", cb);
        },
        importXML: function (path, cb) {
            jsx('importXMLToProject("' + escPath(path) + '")', cb);
        },
        importOverlay: function (path, cb) {
            jsx('importOverlayToProject("' + escPath(path) + '")', cb);
        },
        importFiles: function (paths, bin, cb) {
            var pathsJson = JSON.stringify(paths);
            jsx("importFilesToProject('" + pathsJson.replace(/\\/g, "\\\\").replace(/'/g, "\\'") + "', \"" + (bin || "OpenCut Output") + "\")", cb);
        },
        importCaptions: function (path, cb) {
            jsx('importCaptions("' + escPath(path) + '")', cb);
        },
        importFile: function (path, bin, cb) {
            jsx('importFileToProject("' + escPath(path) + '", "' + (bin || "OpenCut Output") + '")', cb);
        },
        autoImport: function (path, type) {
            if (!cs) return;
            cs.evalScript('autoImportResult("' + path.replace(/\\/g, "\\\\").replace(/"/g, '\\"') + '", "' + (type || "output") + '")');
        },
        isProjectSaved: function (cb) {
            jsx("isProjectSaved()", cb);
        }
    };

    // ================================================================
    // Backend Communication
    // ================================================================
    var _inflightRequests = {};
    function api(method, path, body, callback, timeout) {
        var key = method + " " + path;
        // Deduplicate in-flight GET requests (F6)
        if (method === "GET" && _inflightRequests[key]) {
            // Queue callback to be called when the in-flight request completes
            var existing = _inflightRequests[key];
            if (callback && existing._pendingCallbacks) {
                existing._pendingCallbacks.push(callback);
            }
            return;
        }
        var xhr = new XMLHttpRequest();
        xhr.open(method, BACKEND + path, true);
        xhr.timeout = timeout || 120000;
        xhr.setRequestHeader("Content-Type", "application/json");
        if (csrfToken) xhr.setRequestHeader("X-OpenCut-Token", csrfToken);
        if (method === "GET") {
            xhr._pendingCallbacks = [];
            _inflightRequests[key] = xhr;
        }
        function _notifyPending(err, data) {
            var cbs = xhr._pendingCallbacks || [];
            for (var i = 0; i < cbs.length; i++) {
                try { cbs[i](err, data); } catch (e) { /* swallow */ }
            }
        }
        xhr.onload = function () {
            delete _inflightRequests[key];
            var err = null, data = null;
            try { data = JSON.parse(xhr.responseText); }
            catch (e) { err = e; }
            callback(err, data);
            _notifyPending(err, data);
        };
        xhr.onerror = function () {
            delete _inflightRequests[key];
            var err = new Error("Network error");
            callback(err, null);
            _notifyPending(err, null);
        };
        xhr.ontimeout = function () {
            delete _inflightRequests[key];
            var err = new Error("Timeout");
            callback(err, null);
            _notifyPending(err, null);
        };
        xhr.send(body ? JSON.stringify(body) : null);
    }

    // Wrapper: api call with button spinner feedback
    function apiWithSpinner(btn, method, path, body, callback, timeout) {
        var origText = btn.textContent;
        btn.disabled = true;
        btn.textContent = "Working...";
        api(method, path, body, function (err, data) {
            btn.disabled = false;
            btn.textContent = origText;
            callback(err, data);
        }, timeout);
    }

    // ================================================================
    // Health Check (exponential backoff on failure)
    // ================================================================
    var portScanPending = false;
    var healthBackoff = HEALTH_MS;
    var HEALTH_MAX_MS = 60000;

    function checkHealth() {
        api("GET", "/health", null, function (err, data) {
            var ok = !err && data && data.status === "ok";
            if (ok) {
                // Reset backoff on success
                if (healthBackoff !== HEALTH_MS) {
                    healthBackoff = HEALTH_MS;
                    clearInterval(healthTimer);
                    healthTimer = setInterval(checkHealth, HEALTH_MS);
                }
                if (!connected && el.serverStatusBanner) {
                    el.serverStatusBanner.classList.add("hidden");
                    showToast("Server reconnected", "success");
                }
                connected = true;
                el.connDot.className = "conn-dot on";
                el.connLabel.textContent = "Connected";
                if (data.csrf_token) csrfToken = data.csrf_token;
                if (data.capabilities) capabilities = data.capabilities;
                el.backendPort.textContent = BACKEND.replace("http://127.0.0.1:", "Port ");
                updateButtons();
                loadCapabilities();
                return;
            }
            if (connected && el.serverStatusBanner) {
                el.serverStatusBanner.classList.remove("hidden");
                if (el.serverStatusMsg) el.serverStatusMsg.textContent = "Server disconnected. Reconnecting...";
            }
            connected = false;
            // Exponential backoff: double interval on failure, cap at 60s
            healthBackoff = Math.min(healthBackoff * 2, HEALTH_MAX_MS);
            clearInterval(healthTimer);
            healthTimer = setInterval(checkHealth, healthBackoff);
            if (!portScanPending) { portScanPending = true; scanForServer(); }
        }, 10000);
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
                            if (data.csrf_token) csrfToken = data.csrf_token;
                            el.connDot.className = "conn-dot on";
                            el.connLabel.textContent = "Connected" + (port !== BACKEND_BASE_PORT ? " (:" + port + ")" : "");
                            el.backendPort.textContent = "Port " + port;
                            if (data.capabilities) capabilities = data.capabilities;
                            healthBackoff = HEALTH_MS;
                            clearInterval(healthTimer);
                            healthTimer = setInterval(checkHealth, HEALTH_MS);
                            updateButtons();
                            loadCapabilities();
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
                PremiereBridge.startBackend();
            }
        }
    }

    // ================================================================
    // Project Media
    // ================================================================
    var _projectSaveWarned = false;

    function scanProjectMedia() {
        if (!inPremiere) return;
        // Warn once if project hasn't been saved
        if (!_projectSaveWarned) {
            PremiereBridge.isProjectSaved(function (res) {
                try {
                    var info = JSON.parse(res);
                    if (!info.saved) {
                        showToast("Save your project first for best results", "warning");
                    }
                    _projectSaveWarned = true;
                } catch (e) {}
            });
        }
        PremiereBridge.getProjectMedia(function (result) {
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
                populateRecentFiles();
                refreshClipDropdown();
            } catch (e) {
                console.error("scanProjectMedia parse error:", e, result);
                showAlert("Failed to read project media. Check console for details.");
            }
        });
    }

    function useTimelineSelection() {
        if (!inPremiere) return;
        PremiereBridge.getTimelineSelection(function (result) {
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
            PremiereBridge.browseForFile(function (result) {
                if (result && result !== "null" && result !== "undefined" && result.length > 3) {
                    selectFile(result, result.split(/[/\\]/).pop());
                }
            });
        }
    }

    function browseForInput(targetId) {
        if (inPremiere) {
            PremiereBridge.browseForFile(function (result) {
                if (result && result !== "null" && result !== "undefined" && result.length > 3) {
                    var input = document.getElementById(targetId);
                    if (input) input.value = result;
                }
            });
        }
    }

    function getTranscriptCacheKey(filepath) {
        return "opencut_transcript_" + filepath.replace(/[^a-zA-Z0-9]/g, "_");
    }

    function cacheTranscriptSegments(filepath, segments) {
        try {
            var key = getTranscriptCacheKey(filepath);
            localStorage.setItem(key, JSON.stringify(segments));
        } catch (e) { /* quota exceeded or unavailable */ }
    }

    function loadCachedTranscript(filepath) {
        try {
            var key = getTranscriptCacheKey(filepath);
            var data = localStorage.getItem(key);
            if (data) return JSON.parse(data);
        } catch (e) {}
        return null;
    }

    var RECENT_FILES_KEY = "opencut_recent_files";
    var MAX_RECENT_FILES = 10;

    function addRecentFile(path, name) {
        try {
            var recent = JSON.parse(localStorage.getItem(RECENT_FILES_KEY) || "[]");
            // Remove if already exists
            recent = recent.filter(function (r) { return r.path !== path; });
            recent.unshift({ path: path, name: name });
            if (recent.length > MAX_RECENT_FILES) recent = recent.slice(0, MAX_RECENT_FILES);
            localStorage.setItem(RECENT_FILES_KEY, JSON.stringify(recent));
        } catch (e) {}
    }

    function getRecentFiles() {
        try {
            return JSON.parse(localStorage.getItem(RECENT_FILES_KEY) || "[]");
        } catch (e) { return []; }
    }

    function populateRecentFiles() {
        var recent = getRecentFiles();
        if (!recent.length) return;
        // Check if optgroup already exists
        var existing = el.clipSelect.querySelector('optgroup[label="Recent Files"]');
        if (existing) existing.parentNode.removeChild(existing);
        var group = document.createElement("optgroup");
        group.label = "Recent Files";
        for (var i = 0; i < recent.length; i++) {
            var opt = document.createElement("option");
            opt.value = recent[i].path;
            opt.textContent = recent[i].name;
            opt.setAttribute("data-name", recent[i].name);
            group.appendChild(opt);
        }
        el.clipSelect.appendChild(group);
        refreshClipDropdown();
    }

    function refreshClipDropdown() {
        if (el.clipSelect.parentNode) {
            // Disconnect old observer to prevent leak
            if (el.clipSelect._customDropdown && el.clipSelect._customDropdown.observer) {
                el.clipSelect._customDropdown.observer.disconnect();
            }
            var oldDd = el.clipSelect.parentNode.querySelector(".custom-dropdown");
            if (oldDd) {
                oldDd.parentNode.removeChild(oldDd);
                delete el.clipSelect.dataset.customized;
                createCustomDropdown(el.clipSelect);
            }
        }
    }

    function selectFile(path, name) {
        selectedPath = path;
        selectedName = name || path.split(/[/\\]/).pop();
        lastTranscriptSegments = loadCachedTranscript(path);
        transcriptData = null;
        addRecentFile(path, selectedName);
        addRecentClip(path);
        el.fileInfoBox.classList.remove("hidden");
        el.fileNameDisplay.textContent = selectedName;
        el.fileMetaDisplay.innerHTML = '<span class="skeleton skeleton-wide"></span>';
        // CSS-driven button state: body.has-clip enables .requires-clip buttons
        if (path) document.body.classList.add("has-clip");
        else document.body.classList.remove("has-clip");
        updateButtons();
        updateClipPreview();

        if (connected) {
            api("POST", "/info", { filepath: path }, function (err, data) {
                if (!err && data && !data.error) {
                    var meta = "";
                    if (data.duration) meta += fmtDur(data.duration);
                    if (data.video) {
                        meta += " | " + data.video.width + "x" + data.video.height + " @ " + safeFixed(data.video.fps, 2) + " fps";
                        if (data.video.codec) meta += " (" + data.video.codec + ")";
                    }
                    if (data.audio) {
                        meta += " | " + safeFixed(data.audio.sample_rate / 1000, 1) + " kHz";
                        if (data.audio.codec) meta += " (" + data.audio.codec + ")";
                    }
                    if (data.file_size_mb) meta += " | " + safeFixed(data.file_size_mb, 1) + " MB";
                    if (lastTranscriptSegments) meta += " | Transcript cached";
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
                for (var j = 0; j < all.length; j++) {
                    all[j].classList.remove("active");
                    all[j].setAttribute("aria-selected", "false");
                }
                var panels = document.querySelectorAll(".nav-panel");
                for (var j = 0; j < panels.length; j++) panels[j].classList.remove("active");
                // Activate target
                this.classList.add("active");
                this.setAttribute("aria-selected", "true");
                var panel = $("panel-" + target);
                if (panel) panel.classList.add("active");
                // Update content header title
                if (el.contentTitle) {
                    el.contentTitle.textContent = this.getAttribute("title") || target;
                }
                // Check sub-tab overflow after tab switch
                checkSubTabOverflow();
                // Load settings info on first visit
                if (target === "settings") loadSettingsInfo();
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

    function checkSubTabOverflow() {
        var containers = document.querySelectorAll(".sub-tabs");
        for (var i = 0; i < containers.length; i++) {
            if (containers[i].scrollWidth > containers[i].clientWidth) {
                containers[i].classList.add("has-overflow");
            } else {
                containers[i].classList.remove("has-overflow");
            }
        }
    }

    // ================================================================
    // Button State
    // ================================================================
    // All buttons that require a clip to be selected
    var _clipButtons = [
        "runSilenceBtn", "runFillersBtn", "runFullBtn",
        "runStyledCaptionsBtn", "runSubtitleBtn", "runTranscriptBtn",
        "runSeparateBtn", "runDenoiseBtn", "measureLoudnessBtn",
        "runNormalizeBtn", "runBeatsBtn", "runEffectBtn",
        "runWatermarkBtn", "runScenesBtn", "runVfxBtn", "runVidAiBtn",
        "runProFxBtn", "runDeepFilterBtn",
        "runFaceBlurBtn", "runStyleBtn",
        "runTranslateBtn", "runKaraokeBtn",
        "runExportPresetBtn", "runThumbBtn", "runBatchBtn", "runWorkflowBtn",
        "runBurninBtn",
        "runSpeedBtn", "runLutBtn", "runDuckBtn",
        "runChromaBtn", "runTransBtn", "runParticlesBtn",
        "runTitleOverlayBtn", "runReframeBtn", "runUpscaleBtn",
        "runColorBtn", "runRemoveBtn", "runFaceAiBtn", "runAnimCapBtn",
        "runExpTranscriptBtn", "loadWaveformBtn", "previewVfxBtn", "runTrimBtn",
        "runAutoEditBtn", "runHighlightsBtn", "runEnhanceBtn", "runShortsBtn",
        "runRepeatDetectBtn", "runChaptersBtn", "runBeatMarkersBtn", "runMulticamBtn",
        "runLoudMatchBtn", "runFootageSearchBtn"
    ];

    function updateButtons() {
        var canRun = connected && selectedPath;

        // Batch-disable all clip-dependent buttons
        for (var i = 0; i < _clipButtons.length; i++) {
            var btn = el[_clipButtons[i]];
            if (btn) btn.disabled = !canRun;
        }

        // Merge has special logic
        if (el.runMergeBtn) el.runMergeBtn.disabled = _mergeFiles.length < 2;

        // Whisper hints
        if (capabilities.captions === false) {
            el.captionsHint.classList.remove("hidden");
            el.fillersHint.classList.remove("hidden");
        } else {
            el.captionsHint.classList.add("hidden");
            el.fillersHint.classList.add("hidden");
        }
        
        // Demucs hint
        if (capabilities.separation === false) {
            el.separateHint.classList.remove("hidden");
            el.runSeparateBtn.disabled = true;
        } else {
            el.separateHint.classList.add("hidden");
        }
        
        // Watermark removal hint
        if (capabilities.watermark_removal === false) {
            el.watermarkHint.classList.remove("hidden");
            el.runWatermarkBtn.disabled = true;
        } else {
            el.watermarkHint.classList.add("hidden");
        }

        // Pedalboard hint
        if (capabilities.pedalboard === false) {
            el.proFxHint.classList.remove("hidden");
            el.runProFxBtn.disabled = true;
        } else {
            el.proFxHint.classList.add("hidden");
        }

        // DeepFilterNet hint
        if (capabilities.deepfilter === false) {
            el.deepFilterHint.classList.remove("hidden");
            el.runDeepFilterBtn.disabled = true;
        } else {
            el.deepFilterHint.classList.add("hidden");
        }

        // Video AI hints (upscale/rembg need install, interp/denoise always work)
        var aiCaps = capabilities.video_ai || {};
        var tool = el.vidAiTool ? el.vidAiTool.value : "upscale";
        if (tool === "upscale" && aiCaps.upscale === false) {
            el.vidAiHint.classList.remove("hidden");
            el.vidAiHintText.textContent = "Real-ESRGAN not installed.";
        } else if (tool === "rembg" && aiCaps.rembg === false) {
            el.vidAiHint.classList.remove("hidden");
            el.vidAiHintText.textContent = "rembg not installed.";
        } else {
            el.vidAiHint.classList.add("hidden");
        }

        // Face tools hint
        var faceCaps = capabilities.face_tools || {};
        if (faceCaps.mediapipe === false) {
            el.faceHint.classList.remove("hidden");
        } else {
            el.faceHint.classList.add("hidden");
        }

        // WhisperX / karaoke hint
        if (capabilities.whisperx === false) {
            el.karaokeHint.classList.remove("hidden");
            el.runKaraokeBtn.disabled = true;
        } else {
            el.karaokeHint.classList.add("hidden");
        }

        // NLLB translation hint
        if (capabilities.nllb === false) {
            el.translateHint.classList.remove("hidden");
        } else {
            el.translateHint.classList.add("hidden");
        }

        // Edge TTS hint
        if (capabilities.edge_tts === false) {
            el.ttsHint.classList.remove("hidden");
        } else {
            el.ttsHint.classList.add("hidden");
        }
    }

    // ================================================================
    // Dynamic Capability Loading
    // ================================================================
    var capabilitiesLoaded = false;

    function loadCapabilities() {
        if (capabilitiesLoaded) return;
        capabilitiesLoaded = true;

        // Fetch translation languages
        api("GET", "/captions/enhanced/capabilities", null, function (err, data) {
            if (err || !data || data.error) return;
            if (data.languages && typeof data.languages === "object") {
                var keys = Object.keys(data.languages);
                if (keys.length > 0) {
                    populateDropdown(el.translateSourceLang, data.languages, "en");
                    populateDropdown(el.translateTargetLang, data.languages, "es");
                }
            }
        });

        // Fetch video AI capabilities
        api("GET", "/video/ai/capabilities", null, function (err, data) {
            if (err || !data || data.error) return;
            if (data.gpu_name) {
                el.connLabel.textContent = "Connected (" + data.gpu_name + ")";
            }
        });
    }

    function populateDropdown(selectEl, langMap, defaultVal) {
        var currentVal = selectEl.value || defaultVal;
        selectEl.innerHTML = "";
        var codes = Object.keys(langMap).sort(function (a, b) {
            return langMap[a].localeCompare(langMap[b]);
        });
        for (var i = 0; i < codes.length; i++) {
            var opt = document.createElement("option");
            opt.value = codes[i];
            opt.textContent = langMap[codes[i]];
            if (codes[i] === currentVal) opt.selected = true;
            selectEl.appendChild(opt);
        }
        // Re-init custom dropdown if it was already created
        if (selectEl.parentNode) {
            // Disconnect old observer to prevent leak
            if (selectEl._customDropdown && selectEl._customDropdown.observer) {
                selectEl._customDropdown.observer.disconnect();
            }
            var oldDropdown = selectEl.parentNode.querySelector(".custom-dropdown");
            if (oldDropdown) {
                oldDropdown.parentNode.removeChild(oldDropdown);
                delete selectEl.dataset.customized;
                createCustomDropdown(selectEl);
            }
        }
    }

    // ================================================================
    // Workflow Queue (multi-step job chains)
    // ================================================================
    var workflowQueue = [];
    var workflowActive = false;

    function runWorkflow(steps) {
        // steps: [{endpoint, payload, label}, ...]
        if (!steps || !steps.length) return;
        workflowQueue = steps.slice();
        workflowActive = true;
        jobStepTotal = workflowQueue.length;
        jobStepCurrent = 0;
        runNextWorkflowStep();
    }

    function runNextWorkflowStep() {
        if (!workflowQueue.length) {
            workflowActive = false;
            jobStepCurrent = 0;
            jobStepTotal = 0;
            return;
        }
        var step = workflowQueue.shift();
        jobStepCurrent++;
        if (step.label) showAlert("Step " + jobStepCurrent + "/" + jobStepTotal + ": " + step.label);
        startJob(step.endpoint, step.payload);
    }

    // Job done listener registry (must be declared before any addJobDoneListener calls)
    var jobDoneListeners = [];
    function addJobDoneListener(fn) { jobDoneListeners.push(fn); }

    // Listener: auto-advance workflow queue on job completion
    addJobDoneListener(function (job) {
        if (workflowActive && job.status === "complete" && workflowQueue.length > 0) {
            runNextWorkflowStep();
            return true; // handled — skip default result display until final step
        }
        if (workflowActive && (job.status === "error" || job.status === "cancelled")) {
            workflowQueue = [];
            workflowActive = false;
        }
    });

    // ================================================================
    // Job Execution & Tracking
    // ================================================================
    var jobStarting = false;

    function startJob(endpoint, payload) {
        if (currentJob || jobStarting) {
            showAlert("A job is already running. Wait for it to finish or cancel it.");
            return;
        }
        if (!selectedPath && payload && !payload.filepath && !payload.no_input) {
            showAlert("Select a clip first.");
            return;
        }

        // Lock immediately to prevent double-click race
        jobStarting = true;

        // Show persistent processing banner
        var stepPrefix = (jobStepTotal > 1) ? "Step " + jobStepCurrent + "/" + jobStepTotal + ": " : "";
        el.processingBanner.classList.remove("hidden");
        el.processingMsg.textContent = stepPrefix + "Starting...";
        el.processingFill.style.width = "0%";
        el.processingElapsed.textContent = "0s";

        // Show inline progress section too
        el.progressSection.classList.remove("hidden");
        el.resultsSection.classList.add("hidden");
        el.progressBar.style.width = "0%";
        el.progressLabel.textContent = stepPrefix + "Starting...";
        el.cancelBtn.classList.remove("hidden");

        // Lock the entire UI
        document.body.classList.add("job-active");

        // Track for retry
        lastJobEndpoint = endpoint;
        lastJobPayload = payload;

        // Show time estimate based on historical data
        fetchTimeEstimate(endpoint.replace(/^\//, "").replace(/\//g, "_"));

        jobStartTime = Date.now();
        elapsedTimer = setInterval(function () {
            var s = Math.floor((Date.now() - jobStartTime) / 1000);
            var timeStr = s < 60 ? s + "s" : Math.floor(s / 60) + "m " + (s % 60) + "s";
            el.progressElapsed.textContent = timeStr;
            el.processingElapsed.textContent = timeStr;
        }, 1000);

        try {
            api("POST", endpoint, payload, function (err, data) {
                jobStarting = false;
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
        } catch (e) {
            jobStarting = false;
            hideProgress();
            showAlert("Failed to start job: " + e.message);
        }
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
            if (!activeStream) return;
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
        var pct = (job.progress || 0) + "%";
        var msg = job.message || "Processing...";
        if (jobStepTotal > 1) {
            msg = "Step " + jobStepCurrent + "/" + jobStepTotal + ": " + msg;
        }
        el.progressBar.style.width = pct;
        el.progressLabel.textContent = msg;
        // Sync to persistent banner
        el.processingFill.style.width = pct;
        el.processingMsg.textContent = msg;
    }

    function enhanceError(msg) {
        if (!msg) return msg;
        if (/not installed|No module named/i.test(msg)) {
            return msg + " \u2014 Install from the Settings tab.";
        }
        if (/memory|CUDA out of memory/i.test(msg)) {
            return msg + " \u2014 Try reducing file size or using CPU mode.";
        }
        if (/Permission|Access denied/i.test(msg)) {
            return msg + " \u2014 Check file permissions.";
        }
        if (/No such file/i.test(msg)) {
            return msg + " \u2014 File may have been moved or deleted.";
        }
        return msg;
    }

    function onJobDone(job) {
        currentJob = null;
        if (elapsedTimer) { clearInterval(elapsedTimer); elapsedTimer = null; }

        // Dispatch to registered listeners; if any returns true, it handled the job
        for (var li = 0; li < jobDoneListeners.length; li++) {
            if (jobDoneListeners[li](job) === true) return;
        }

        if (job.status === "error") {
            hideProgress();
            // Show error in results card for better visibility
            el.resultsSection.classList.remove("hidden");
            el.resultsTitle.textContent = "Error";
            el.resultsTitle.removeAttribute("style");
            el.resultsTitle.setAttribute("data-state", "error");
            el.resultsStats.textContent = enhanceError(job.error || job.message || "Unknown error");
            el.resultsPath.textContent = "";
            // Show retry button if we have a last job to retry
            if (lastJobEndpoint) {
                el.retryJobBtn.classList.remove("hidden");
            }
            return;
        }

        if (job.status === "cancelled") {
            hideProgress();
            return;
        }

        // Success
        hideProgress();
        el.retryJobBtn.classList.add("hidden");
        showResults(job);

        // Auto-import into Premiere (respect global setting)
        var autoImportEnabled = el.settingsAutoImport ? el.settingsAutoImport.checked : true;
        if (job.result && inPremiere && autoImportEnabled) {
            // XML edit list (silence removal, filler removal, etc.)
            var xmlPath = job.result.xml_path;
            if (xmlPath) {
                PremiereBridge.importXML(xmlPath, function (result) {
                    try {
                        var r = JSON.parse(result);
                        if (r.error) {
                            showAlert("Import error: " + r.error);
                        } else if (r.sequence_name) {
                            showAlert("Opened: " + r.sequence_name);
                        }
                    } catch (e) { console.error("XML import parse error:", e); }
                });
                lastXmlPath = xmlPath;
            }
            
            // Styled caption overlay video (.mov with alpha)
            var overlayPath = job.result.overlay_path;
            if (overlayPath) {
                PremiereBridge.importOverlay(overlayPath, function (result) {
                    try {
                        var r = JSON.parse(result);
                        if (r.error) {
                            showAlert("Overlay import error: " + r.error);
                        } else if (r.message) {
                            showAlert(r.message);
                        }
                    } catch (e) { console.error("Overlay import parse error:", e, result); }
                });
                lastOverlayPath = overlayPath;
            }

            // Multiple output files (stem separation)
            var outputPaths = job.result.output_paths;
            if (outputPaths && outputPaths.length > 0) {
                PremiereBridge.importFiles(outputPaths, "OpenCut Stems", function (result) {
                    try {
                        var r = JSON.parse(result);
                        if (r.error) {
                            showAlert("Stem import error: " + r.error);
                        } else if (r.message) {
                            showAlert(r.message);
                        }
                    } catch (e) { console.error("Stem import parse error:", e, result); }
                });
            }

            // Single output file
            var outputPath = job.result.output_path;
            if (outputPath && !overlayPath && !xmlPath) {
                var ext = outputPath.toLowerCase().split(".").pop();
                // Show audio preview for generated audio files (TTS, SFX, music)
                if ((ext === "wav" || ext === "mp3" || ext === "flac" || ext === "ogg") &&
                    (lastJobEndpoint && (lastJobEndpoint.indexOf("tts") !== -1 || lastJobEndpoint.indexOf("sfx") !== -1 || lastJobEndpoint.indexOf("music") !== -1))) {
                    showAudioPreview(outputPath);
                }
                // Caption files (SRT, VTT, ASS) - import to caption track
                if (ext === "srt" || ext === "vtt" || ext === "ass") {
                    PremiereBridge.importCaptions(outputPath, function (result) {
                        try {
                            var r = JSON.parse(result);
                            if (r.error) {
                                showAlert("Caption import error: " + r.error);
                            } else if (r.message) {
                                showAlert(r.message);
                            }
                        } catch (e) { console.error("Caption import parse error:", e, result); }
                    });
                    lastCaptionPath = outputPath;
                }
                // Audio/video files - generic import to project
                else if (ext === "wav" || ext === "mp3" || ext === "flac" || ext === "aac" || ext === "ogg" ||
                         ext === "mp4" || ext === "mov" || ext === "avi" || ext === "mkv" || ext === "webm" || ext === "png" || ext === "jpg") {
                    PremiereBridge.importFile(outputPath, "OpenCut Output", function (result) {
                        try {
                            var r = JSON.parse(result);
                            if (r.error) {
                                showAlert("Import error: " + r.error);
                            } else if (r.message) {
                                showAlert(r.message);
                            }
                        } catch (e) { console.error("File import parse error:", e, result); }
                    });
                }
            }

            // SRT path from full pipeline (separate from output_path)
            var srtPath = job.result.srt_path;
            if (srtPath && srtPath !== outputPath) {
                PremiereBridge.importCaptions(srtPath, function (result) {
                    try {
                        var r = JSON.parse(result);
                        if (r.error) {
                            showAlert("Caption import error: " + r.error);
                        } else if (r.message) {
                            showAlert(r.message);
                        }
                    } catch (e) { console.error("Caption import parse error:", e, result); }
                });
                lastCaptionPath = srtPath;
            }
        }
    }

    function hideProgress() {
        el.progressSection.classList.add("hidden");
        el.cancelBtn.classList.add("hidden");
        el.cancelBtn.textContent = "Cancel";
        el.cancelBtn.disabled = false;
        el.processingBanner.classList.add("hidden");
        el.processingCancel.textContent = "Cancel";
        el.processingCancel.disabled = false;
        document.body.classList.remove("job-active");
        if (elapsedTimer) { clearInterval(elapsedTimer); elapsedTimer = null; }
    }

    function showResults(job) {
        el.resultsSection.classList.remove("hidden");
        el.resultsTitle.textContent = "Complete";
        el.resultsTitle.removeAttribute("style");
        el.resultsTitle.setAttribute("data-state", "success");

        var stats = "";
        var r = job.result || {};

        if (r.summary) {
            stats += esc(r.summary) + "<br>";
        }
        if (r.segments !== undefined) {
            stats += Number(r.segments) + " segments";
        }
        if (r.filler_stats) {
            stats += " | " + Number(r.filler_stats.removed_fillers) + " fillers removed (" + safeFixed(r.filler_stats.total_filler_time, 1) + "s)";
        }
        if (r.caption_segments !== undefined) {
            stats += (stats ? " | " : "") + Number(r.caption_segments) + " captions, " + Number(r.words || 0) + " words";
        }
        if (r.style) {
            stats += " | Style: " + esc(r.style);
        }
        // Audio results
        if (r.effect && !r.method) {
            stats += (stats ? "<br>" : "") + "Effect applied: " + esc(r.effect);
        }
        if (r.method && r.strength !== undefined) {
            stats += (stats ? "<br>" : "") + "Denoise: " + esc(r.method) + " (" + safeFixed(r.strength * 100, 0) + "% strength)";
        }
        if (r.preset && r.target_loudness !== undefined) {
            stats += (stats ? "<br>" : "") + "Normalized to " + safeFixed(r.target_loudness, 1) + " LUFS (" + esc(r.preset) + ")";
            if (r.input_loudness !== undefined) {
                stats += " | Was: " + safeFixed(r.input_loudness, 1) + " LUFS";
            }
        }
        if (r.bpm) {
            stats += (stats ? "<br>" : "") + "BPM: " + safeFixed(r.bpm, 0) + " | " + (r.total_beats != null ? Number(r.total_beats) : 0) + " beats";
            if (r.confidence !== undefined) {
                stats += " | Confidence: " + safeFixed(r.confidence * 100, 0) + "%";
            }
        }
        // Stem separation
        if (r.output_paths && r.output_paths.length > 0) {
            var stemNames = [];
            for (var i = 0; i < r.output_paths.length; i++) {
                var fname = r.output_paths[i].split(/[/\\]/).pop();
                stemNames.push(esc(fname));
            }
            stats += (stats ? "<br>" : "") + r.output_paths.length + " stems: " + stemNames.join(", ");
        }
        // Scene detection
        if (r.total_scenes) {
            stats += (stats ? "<br>" : "") + "Scenes: " + Number(r.total_scenes) + " | Avg: " + safeFixed(r.avg_scene_length, 1) + "s";
        }

        el.resultsStats.innerHTML = stats || "Processing complete.";
        el.resultsPath.textContent = r.xml_path || r.output_path || r.overlay_path || (r.output_paths ? r.output_paths.length + " files exported" : "");
    }

    function cancelJob() {
        if (currentJob) {
            el.processingCancel.textContent = "Cancelling...";
            el.processingCancel.disabled = true;
            el.cancelBtn.textContent = "Cancelling...";
            el.cancelBtn.disabled = true;
            api("POST", "/cancel/" + currentJob, {}, function (err) {
                if (err) {
                    showToast("Cancel failed — connection lost", "error");
                }
                currentJob = null;
                hideProgress();
            });
            // Clean up poll timer and SSE since job is being cancelled
            if (pollTimer) {
                clearInterval(pollTimer);
                pollTimer = null;
            }
            if (activeStream) {
                activeStream.close();
                activeStream = null;
            }
        }
    }

    // ================================================================
    // Run Functions (one per action button)
    // ================================================================

    // --- CUT TAB ---
    function runSilence() {
        var mode = el.silenceMode ? el.silenceMode.value : "remove";
        if (mode === "speedup") {
            startJob("/silence/speed-up", {
                filepath: selectedPath,
                output_dir: projectFolder,
                speed_factor: parseFloat((el.silenceSpeedFactor || {}).value || "4"),
                threshold_db: parseFloat(el.threshold.value),
                min_duration: parseFloat(el.minDuration.value),
            });
            return;
        }
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
            output_dir: projectFolder,
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

    function runSeparate() {
        // Collect selected stems
        var stems = [];
        if (el.stemVocals.checked) stems.push("vocals");
        if (el.stemInstrumental.checked) stems.push("no_vocals");
        if (el.stemDrums.checked) stems.push("drums");
        if (el.stemBass.checked) stems.push("bass");
        if (el.stemOther.checked) stems.push("other");
        
        if (stems.length === 0) {
            showAlert("Please select at least one stem to extract");
            return;
        }
        
        startJob("/audio/separate", {
            filepath: selectedPath,
            output_dir: projectFolder,
            model: el.separateModel.value,
            stems: stems,
            format: el.separateFormat.value,
            auto_import: el.separateImport.checked,
        });
    }
    
    function installDemucs() {
        el.separateHint.innerHTML = '<span style="color: var(--neon-cyan);">Installing Demucs... This may take a few minutes.</span>';
        apiWithSpinner(el.installDemucsBtn, "POST", "/demucs/install", {}, function(err, data) {
            if (err || (data && data.error)) {
                el.separateHint.innerHTML = '<span style="color: var(--neon-red);">Installation failed: ' + esc(data ? data.error : 'Unknown error') + '</span>';
            } else {
                el.separateHint.classList.add("hidden");
                capabilities.separation = true;
                updateButtons();
                showAlert("Demucs installed successfully!");
            }
        }, 300000);
    }

    function measureLoudness() {
        el.loudnessMeter.classList.remove("hidden");
        el.meterLUFS.textContent = "Measuring...";
        el.meterTP.textContent = "--";
        el.meterLRA.textContent = "--";

        apiWithSpinner(el.measureLoudnessBtn, "POST", "/audio/measure", { filepath: selectedPath }, function (err, data) {
            if (!err && data && !data.error) {
                el.meterLUFS.textContent = safeFixed(data.integrated_lufs, 1) + " LUFS";
                el.meterTP.textContent = safeFixed(data.true_peak_dbtp, 1) + " dBTP";
                el.meterLRA.textContent = safeFixed(data.loudness_range_lu, 1) + " LU";
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

    // --- VIDEO TAB ---
    function runWatermark() {
        startJob("/video/watermark", {
            filepath: selectedPath,
            output_dir: projectFolder,
            max_bbox_percent: parseInt(el.wmMaxBbox.value),
            detection_prompt: el.wmPrompt.value.trim() || "watermark",
            detection_skip: parseInt(el.wmFrameSkip.value),
            transparent: el.wmTransparent.checked,
            preview: el.wmPreview.checked,
            auto_import: el.wmAutoImport.checked,
        });
    }
    
    function installWatermark() {
        el.watermarkHint.innerHTML = '<span style="color: var(--neon-cyan);">Installing watermark remover... This may take several minutes.</span>';
        apiWithSpinner(el.installWatermarkBtn, "POST", "/watermark/install", {}, function(err, data) {
            if (err || (data && data.error)) {
                el.watermarkHint.innerHTML = '<span style="color: var(--neon-red);">Installation failed: ' + esc(data ? data.error : 'Unknown error') + '</span>';
            } else {
                el.watermarkHint.classList.add("hidden");
                capabilities.watermark_removal = true;
                updateButtons();
                showAlert("Watermark remover installed successfully!");
            }
        }, 300000);
    }
    
    function runScenes() {
        el.sceneResults.classList.add("hidden");
        startJob("/video/scenes", {
            filepath: selectedPath,
            output_dir: projectFolder,
            threshold: parseFloat(el.sceneThreshold.value),
            min_scene_length: parseFloat(el.minSceneLen.value),
            method: el.sceneMethod ? el.sceneMethod.value : "ffmpeg",
        });
    }

    // --- VIDEO EFFECTS ---
    function runVfx() {
        var effect = el.vfxSelect.value;
        var params = {};
        if (effect === "stabilize") {
            params.smoothing = parseInt(el.vfxStabSmoothing.value);
            params.zoom = parseInt(el.vfxStabZoom.value);
        } else if (effect === "vignette") {
            params.intensity = parseFloat(el.vfxVignetteIntensity.value);
        } else if (effect === "film_grain") {
            params.intensity = parseFloat(el.vfxGrainIntensity.value);
        } else if (effect === "letterbox") {
            params.aspect = el.vfxLetterboxAspect.value;
        } else if (effect === "chromakey") {
            params.color = el.vfxChromakeyColor.value;
            params.similarity = parseFloat(el.vfxChromakeySim.value);
            params.blend = parseFloat(el.vfxChromakeyBlend.value);
        } else if (effect === "lut") {
            params.lut_path = el.vfxLutPath.value.trim();
            params.intensity = parseFloat(el.vfxLutIntensity.value);
            if (!params.lut_path) { showAlert("Please enter a LUT file path"); return; }
        }
        startJob("/video/fx/apply", {
            filepath: selectedPath,
            output_dir: projectFolder,
            effect: effect,
            params: params,
        });
    }

    function showVfxParams() {
        var effect = el.vfxSelect.value;
        document.querySelectorAll(".vfx-params").forEach(function (p) { p.classList.add("hidden"); });
        var panel = document.getElementById("vfxParams-" + effect);
        if (panel) panel.classList.remove("hidden");
    }

    // --- VIDEO AI ---
    function runVidAi() {
        var tool = el.vidAiTool.value;
        if (tool === "upscale") {
            startJob("/video/ai/upscale", {
                filepath: selectedPath,
                output_dir: projectFolder,
                scale: parseInt(el.vidAiUpscaleScale.value),
                model: el.vidAiUpscaleModel.value,
            });
        } else if (tool === "rembg") {
            startJob("/video/ai/rembg", {
                filepath: selectedPath,
                output_dir: projectFolder,
                model: el.vidAiRembgModel.value,
                bg_color: el.vidAiRembgBg.value,
                alpha_only: el.vidAiRembgAlpha.checked,
            });
        } else if (tool === "interpolate") {
            startJob("/video/ai/interpolate", {
                filepath: selectedPath,
                output_dir: projectFolder,
                multiplier: parseInt(el.vidAiInterpMultiplier.value),
            });
        } else if (tool === "denoise") {
            startJob("/video/ai/denoise", {
                filepath: selectedPath,
                output_dir: projectFolder,
                method: el.vidAiDenoiseMethod.value,
                strength: parseFloat(el.vidAiDenoiseStrength.value),
            });
        }
    }

    function showVidAiParams() {
        var tool = el.vidAiTool.value;
        document.querySelectorAll(".vidai-params").forEach(function (p) { p.classList.add("hidden"); });
        var panel = document.getElementById("vidAiParams-" + tool);
        if (panel) panel.classList.remove("hidden");
        // Update install hint visibility
        updateButtons();
    }

    function installVidAi() {
        var tool = el.vidAiTool.value;
        var component = tool === "rembg" ? "rembg_cpu" : tool;
        el.vidAiHint.innerHTML = '<span style="color: var(--neon-cyan);">Installing... This may take several minutes.</span>';
        startJob("/video/ai/install", { component: component, no_input: true });
    }

    // --- AUDIO PRO (Pedalboard) ---
    var pedalboardEffectsData = [];

    function loadPedalboardEffects() {
        api("GET", "/audio/pro/effects", null, function (err, data) {
            if (!err && data && data.effects) {
                pedalboardEffectsData = data.effects;
                updateProFxEffectList();
            }
        });
    }

    function updateProFxEffectList() {
        var cat = el.proFxCategory.value;
        var filtered = pedalboardEffectsData.filter(function (e) { return e.category === cat; });
        el.proFxEffect.innerHTML = "";
        filtered.forEach(function (e) {
            var opt = document.createElement("option");
            opt.value = e.name;
            opt.textContent = e.label;
            el.proFxEffect.appendChild(opt);
        });
        updateProFxParams();
    }

    function updateProFxParams() {
        var effectName = el.proFxEffect.value;
        var effectData = pedalboardEffectsData.find(function (e) { return e.name === effectName; });
        el.proFxParams.innerHTML = "";
        if (!effectData || !effectData.params || Object.keys(effectData.params).length === 0) {
            return;
        }
        Object.keys(effectData.params).forEach(function (key) {
            var p = effectData.params[key];
            var group = document.createElement("div");
            group.className = "form-group";
            var label = document.createElement("label");
            label.textContent = p.label;
            group.appendChild(label);
            var row = document.createElement("div");
            row.className = "slider-row";
            var slider = document.createElement("input");
            slider.type = "range";
            slider.min = p.min;
            slider.max = p.max;
            slider.value = p.default;
            slider.step = p.step;
            slider.id = "proFxParam_" + key;
            slider.className = "pro-fx-slider";
            var val = document.createElement("span");
            val.className = "slider-val";
            val.textContent = p.default;
            slider.addEventListener("input", function () { val.textContent = this.value; });
            row.appendChild(slider);
            row.appendChild(val);
            group.appendChild(row);
            el.proFxParams.appendChild(group);
        });
    }

    function runProFx() {
        var effect = el.proFxEffect.value;
        var params = {};
        var sliders = el.proFxParams.querySelectorAll(".pro-fx-slider");
        sliders.forEach(function (s) {
            var key = s.id.replace("proFxParam_", "");
            params[key] = parseFloat(s.value);
        });
        startJob("/audio/pro/apply", {
            filepath: selectedPath,
            output_dir: projectFolder,
            effect: effect,
            params: params,
        });
    }

    function installPedalboard() {
        el.proFxHint.innerHTML = '<span style="color: var(--neon-cyan);">Installing Pedalboard...</span>';
        startJob("/audio/pro/install", { component: "pedalboard", no_input: true });
    }

    function runDeepFilter() {
        startJob("/audio/pro/deepfilter", {
            filepath: selectedPath,
            output_dir: projectFolder,
        });
    }

    function installDeepFilter() {
        el.deepFilterHint.innerHTML = '<span style="color: var(--neon-cyan);">Installing DeepFilterNet...</span>';
        startJob("/audio/pro/install", { component: "deepfilter", no_input: true });
    }

    // --- FACE BLUR ---
    function runFaceBlur() {
        startJob("/video/face/blur", {
            filepath: selectedPath,
            output_dir: projectFolder,
            method: el.faceBlurMethod.value,
            strength: parseInt(el.faceBlurStrength.value),
            detector: el.faceDetector.value,
        });
    }

    function installMediapipe() {
        el.faceHint.innerHTML = '<span style="color: var(--neon-cyan);">Installing MediaPipe...</span>';
        startJob("/video/face/install", { no_input: true });
    }

    // --- STYLE TRANSFER ---
    function runStyleTransfer() {
        startJob("/video/style/apply", {
            filepath: selectedPath,
            output_dir: projectFolder,
            style: el.styleModel.value,
            intensity: parseFloat(el.styleIntensity.value),
        });
    }

    // --- CAPTION TRANSLATION ---
    var lastTranscriptSegments = null;
    var pendingBurnin = false;
    var pendingAnimCap = false;
    var pendingTranslate = false;
    var jobStepCurrent = 0;
    var jobStepTotal = 0;

    function runTranslate() {
        if (lastTranscriptSegments) {
            // We have segments from a previous transcription, translate them
            startJob("/captions/translate", {
                filepath: selectedPath,
                segments: lastTranscriptSegments,
                source_lang: el.translateSourceLang.value,
                target_lang: el.translateTargetLang.value,
                format: el.translateFormat.value,
                output_dir: projectFolder,
            });
        } else {
            // Need to transcribe first, then auto-chain into translation
            showAlert("Step 1/2: Transcribing first, then translating...");
            pendingTranslate = true;
            jobStepCurrent = 1;
            jobStepTotal = 2;
            startJob("/transcript", {
                filepath: selectedPath,
                model: el.translateModel.value,
            });
        }
    }

    function installNllb() {
        el.translateHint.innerHTML = '<span style="color: var(--neon-cyan);">Installing NLLB translation...</span>';
        startJob("/captions/enhanced/install", { component: "nllb", no_input: true });
    }

    // --- KARAOKE CAPTIONS ---
    function runKaraoke() {
        startJob("/captions/whisperx", {
            filepath: selectedPath,
            output_dir: projectFolder,
            model: el.karaokeModel.value,
            diarize: el.karaokeDiarize.checked,
        });
    }

    function installWhisperx() {
        el.karaokeHint.innerHTML = '<span style="color: var(--neon-cyan);">Installing WhisperX...</span>';
        startJob("/captions/enhanced/install", { component: "whisperx", no_input: true });
    }

    // --- EXPORT PRESETS ---
    var exportPresetsData = [];

    function loadExportPresets() {
        api("GET", "/export/presets", null, function (err, data) {
            if (!err && data && data.presets) {
                exportPresetsData = data.presets;
                updateExportPresetList();
            }
        });
    }

    function updateExportPresetList() {
        var cat = el.exportPresetCategory.value;
        var filtered = exportPresetsData.filter(function (p) { return p.category === cat; });
        el.exportPresetSelect.innerHTML = "";
        filtered.forEach(function (p) {
            var opt = document.createElement("option");
            opt.value = p.name;
            opt.textContent = p.label;
            el.exportPresetSelect.appendChild(opt);
        });
        updateExportPresetDesc();
    }

    function updateExportPresetDesc() {
        var name = el.exportPresetSelect.value;
        var preset = exportPresetsData.find(function (p) { return p.name === name; });
        el.exportPresetDesc.textContent = preset ? preset.description : "";
    }

    function runExportPreset() {
        startJob("/export/preset", {
            filepath: selectedPath,
            output_dir: projectFolder,
            preset: el.exportPresetSelect.value,
        });
    }

    // --- AUTO-THUMBNAILS ---
    function runThumbnails() {
        startJob("/export/thumbnails", {
            filepath: selectedPath,
            output_dir: projectFolder,
            count: parseInt(el.thumbCount.value),
            width: parseInt(el.thumbWidth.value),
            use_faces: el.thumbUseFaces.checked,
        });
    }

    // --- BATCH PROCESSING ---
    function runBatch() {
        // Use batch file picker selection if available, otherwise fall back to clip selector
        var paths = _batchFiles && _batchFiles.length > 0 ? _batchFiles.slice() : [];
        if (paths.length === 0 && el.clipSelect && el.clipSelect.options) {
            for (var i = 0; i < el.clipSelect.options.length; i++) {
                var val = el.clipSelect.options[i].value;
                if (val) paths.push(val);
            }
        }
        if (paths.length === 0) {
            showAlert("No clips found in project. Load clips first.");
            return;
        }
        if (paths.length === 1) {
            showAlert("Only 1 clip found. Batch requires 2+ files.");
            return;
        }

        var op = el.batchOperation.value;
        el.batchResults.classList.remove("hidden");
        el.batchStatusText.textContent = "Starting batch: " + paths.length + " files...";

        api("POST", "/batch/create", {
            operation: op,
            filepaths: paths,
            params: { output_dir: projectFolder },
        }, function (err, data) {
            if (err || !data || data.error) {
                el.batchStatusText.textContent = "Batch error: " + ((data && data.error) || "Unknown");
                return;
            }
            var batchId = data.batch_id;
            el.batchStatusText.textContent = "Batch running: 0/" + data.total + " complete...";
            // Poll for status (with error limit to prevent infinite polling)
            var pollErrors = 0;
            if (batchPollTimer) { clearInterval(batchPollTimer); batchPollTimer = null; }
            batchPollTimer = setInterval(function () {
                api("GET", "/batch/" + batchId, null, function (e2, d2) {
                    if (e2 || !d2) {
                        pollErrors++;
                        if (pollErrors >= 10) {
                            clearInterval(batchPollTimer); batchPollTimer = null;
                            el.batchStatusText.textContent = "Batch poll failed after 10 errors";
                        }
                        return;
                    }
                    pollErrors = 0;
                    var res = d2.results || {};
                    el.batchStatusText.textContent =
                        "Batch " + d2.status + ": " + (d2.completed || 0) + "/" + (d2.total || 0) +
                        " (" + (res.success || 0) + " ok, " + (res.failed || 0) + " failed)";
                    if (d2.status !== "running") {
                        clearInterval(batchPollTimer); batchPollTimer = null;
                        showAlert("Batch complete: " + (res.success || 0) + " succeeded");
                    }
                });
            }, 2000);
        });
    }

    // --- WORKFLOW PRESETS ---
    var WORKFLOW_PRESETS = {
        clean_audio: [
            { endpoint: "/audio/denoise", payload: { method: "rnnoise" }, label: "Denoising audio..." },
            { endpoint: "/audio/normalize", payload: { target_lufs: -14 }, label: "Normalizing audio..." },
        ],
        subtitle_pipeline: [
            { endpoint: "/transcript", payload: { model: "base", export_format: "srt" }, label: "Transcribing & exporting subtitles..." },
        ],
        translate_pipeline: [
            { endpoint: "/transcript", payload: { model: "base" }, label: "Transcribing..." },
            // Translation is handled by chaining via pendingTranslate
        ],
        pro_video: [
            { endpoint: "/video/fx/apply", payload: { effect: "stabilize", params: { smoothing: 10, zoom: 0 } }, label: "Stabilizing video..." },
            { endpoint: "/audio/denoise", payload: { method: "rnnoise" }, label: "Denoising audio..." },
            { endpoint: "/audio/normalize", payload: { target_lufs: -14 }, label: "Normalizing audio..." },
        ],
        social_ready: [
            { endpoint: "/silence", payload: { threshold: -35, min_silence: 0.4, pad_before: 0.1, pad_after: 0.1 }, label: "Removing silence..." },
            { endpoint: "/audio/normalize", payload: { target_lufs: -14 }, label: "Normalizing audio..." },
        ],
    };

    function runWorkflowPreset() {
        var presetKey = el.workflowPreset.value;
        var preset = WORKFLOW_PRESETS[presetKey];
        if (!preset || !preset.length) {
            showAlert("Unknown workflow preset.");
            return;
        }
        // Inject filepath and output_dir into each step
        var steps = [];
        for (var i = 0; i < preset.length; i++) {
            var step = { endpoint: preset[i].endpoint, label: preset[i].label };
            var p = {};
            for (var k in preset[i].payload) { p[k] = preset[i].payload[k]; }
            p.filepath = selectedPath;
            p.output_dir = projectFolder;
            step.payload = p;
            steps.push(step);
        }
        runWorkflow(steps);
    }

    // --- TTS VOICE GENERATION ---
    function runTts() {
        var text = el.ttsText.value.trim();
        if (!text) {
            showAlert("Enter text to generate speech.");
            return;
        }
        var rateVal = parseInt(el.ttsRate.value);
        var rateStr = (rateVal >= 0 ? "+" : "") + rateVal + "%";

        startJob("/audio/tts/generate", {
            text: text,
            engine: el.ttsEngine.value,
            voice: el.ttsVoice.value,
            rate: rateStr,
            output_dir: projectFolder,
            no_input: true,
        });
    }

    function installEdgeTts() {
        el.ttsHint.innerHTML = '<span style="color: var(--neon-cyan);">Installing Edge TTS...</span>';
        startJob("/audio/tts/install", { component: "edge_tts", no_input: true });
    }

    // --- SFX GENERATOR ---
    function showSfxParams() {
        if (el.sfxType.value === "tone") {
            el.sfxPresetParams.classList.add("hidden");
            el.sfxToneParams.classList.remove("hidden");
        } else {
            el.sfxPresetParams.classList.remove("hidden");
            el.sfxToneParams.classList.add("hidden");
        }
    }

    function runSfx() {
        if (el.sfxType.value === "tone") {
            startJob("/audio/gen/tone", {
                frequency: parseInt(el.toneFreq.value),
                duration: parseFloat(el.sfxDuration.value),
                waveform: el.toneWaveform.value,
                volume: 0.5,
                output_dir: projectFolder,
                no_input: true,
            });
        } else {
            startJob("/audio/gen/sfx", {
                preset: el.sfxPreset.value,
                duration: parseFloat(el.sfxDuration.value),
                output_dir: projectFolder,
                no_input: true,
            });
        }
    }

    // --- CAPTION BURN-IN ---
    function runBurnin() {
        if (lastTranscriptSegments) {
            // We have segments, burn them in directly
            startJob("/captions/burnin/segments", {
                filepath: selectedPath,
                segments: lastTranscriptSegments,
                style: el.burninStyle.value,
                output_dir: projectFolder,
            });
        } else {
            // Transcribe first
            showAlert("Step 1/2: Transcribing first, then burning in captions...");
            pendingBurnin = true;
            jobStepCurrent = 1;
            jobStepTotal = 2;
            startJob("/transcript", {
                filepath: selectedPath,
                model: el.burninModel.value,
            });
        }
    }

    // --- SPEED / RAMP ---
    function showSpeedParams() {
        var mode = el.speedMode.value;
        if (mode === "constant") {
            el.speedConstantParams.classList.remove("hidden");
            el.speedRampParams.classList.add("hidden");
        } else if (mode === "preset") {
            el.speedConstantParams.classList.add("hidden");
            el.speedRampParams.classList.remove("hidden");
        } else {
            el.speedConstantParams.classList.add("hidden");
            el.speedRampParams.classList.add("hidden");
        }
    }

    function runSpeed() {
        var mode = el.speedMode.value;
        if (mode === "reverse") {
            startJob("/video/speed/reverse", {
                filepath: selectedPath,
                output_dir: projectFolder,
                reverse_audio: true,
            });
        } else if (mode === "preset") {
            startJob("/video/speed/ramp", {
                filepath: selectedPath,
                output_dir: projectFolder,
                preset: el.speedRampPreset.value,
            });
        } else {
            startJob("/video/speed/change", {
                filepath: selectedPath,
                output_dir: projectFolder,
                speed: parseFloat(el.speedMultiplier.value),
                maintain_pitch: el.speedMaintainPitch.checked,
            });
        }
    }

    // --- LUT LIBRARY ---
    function runLut() {
        startJob("/video/lut/apply", {
            filepath: selectedPath,
            output_dir: projectFolder,
            lut: el.lutSelect.value,
            intensity: parseFloat(el.lutIntensity.value),
        });
    }

    // --- AUDIO DUCKING ---
    function runDuck() {
        var musicPath = el.duckMusicPath.value.trim();
        if (!musicPath) {
            showAlert("Enter a music file path.");
            return;
        }
        startJob("/audio/duck-video", {
            filepath: selectedPath,
            music_path: musicPath,
            output_dir: projectFolder,
            music_volume: parseFloat(el.duckMusicVol.value),
            duck_amount: parseFloat(el.duckAmount.value),
        });
    }

    // --- CHROMAKEY / COMPOSITING ---
    function showChromaParams() {
        var m = el.chromaMode.value;
        el.chromakeyParams.classList.toggle("hidden", m !== "chromakey");
        el.pipParams.classList.toggle("hidden", m !== "pip");
        el.blendParams.classList.toggle("hidden", m !== "blend");
    }
    function runChroma() {
        var m = el.chromaMode.value;
        if (m === "pip") {
            var pp = el.pipPath.value.trim();
            if (!pp) { showAlert("Enter PiP video path."); return; }
            startJob("/video/pip", { filepath: selectedPath, pip_path: pp, output_dir: projectFolder,
                position: el.pipPosition.value, scale: parseFloat(el.pipScale.value) });
        } else if (m === "blend") {
            var ov = el.blendOverlay.value.trim();
            if (!ov) { showAlert("Enter overlay path."); return; }
            startJob("/video/blend", { filepath: selectedPath, overlay_path: ov, output_dir: projectFolder,
                mode: el.blendMode.value, opacity: parseFloat(el.blendOpacity.value) });
        } else {
            var bg = el.chromaBgPath.value.trim();
            if (!bg) { showAlert("Enter background path."); return; }
            startJob("/video/chromakey", { filepath: selectedPath, background: bg, output_dir: projectFolder,
                color: el.chromaColor.value, tolerance: parseFloat(el.chromaTol.value) });
        }
    }

    // --- TRANSITIONS ---
    function runTransition() {
        var cb = el.transClipB.value.trim();
        if (!cb) { showAlert("Enter second clip path."); return; }
        startJob("/video/transitions/apply", { clip_a: selectedPath, clip_b: cb, output_dir: projectFolder,
            transition: el.transType.value, duration: parseFloat(el.transDur.value) });
    }

    // --- PARTICLES ---
    function runParticles() {
        startJob("/video/particles/apply", { filepath: selectedPath, output_dir: projectFolder,
            preset: el.particlePreset.value, density: parseFloat(el.particleDensity.value) });
    }

    // --- TITLES ---
    function runTitleOverlay() {
        var t = el.titleText.value.trim();
        if (!t) { showAlert("Enter title text."); return; }
        startJob("/video/title/overlay", { filepath: selectedPath, text: t, output_dir: projectFolder,
            preset: el.titlePreset.value, duration: parseFloat(el.titleDur.value),
            font_size: parseInt(el.titleFontSize.value), subtitle: el.titleSubtext.value.trim() });
    }
    function runTitleCard() {
        var t = el.titleText.value.trim();
        if (!t) { showAlert("Enter title text."); return; }
        startJob("/video/title/render", { text: t, output_dir: projectFolder, no_input: true,
            preset: el.titlePreset.value, duration: parseFloat(el.titleDur.value),
            font_size: parseInt(el.titleFontSize.value), subtitle: el.titleSubtext.value.trim() });
    }

    // --- PRO UPSCALE ---
    // --- REFRAME ---
    var _reframeDims = {
        tiktok: [1080, 1920], instagram_reel: [1080, 1920], instagram_post: [1080, 1080],
        instagram_land: [1080, 566], youtube: [1920, 1080], youtube_4k: [3840, 2160],
        youtube_short: [1080, 1920], twitter: [1920, 1080], square: [1080, 1080]
    };

    function updateReframeUI() {
        var preset = el.reframePreset.value;
        var isCustom = preset === "custom";
        el.reframeCustomDims.classList.toggle("hidden", !isCustom);
        var mode = el.reframeMode.value;
        el.reframeCropPosGroup.classList.toggle("hidden", mode !== "crop");
        el.reframePadColorGroup.classList.toggle("hidden", mode !== "pad");
        // Show info
        if (!isCustom && _reframeDims[preset]) {
            var d = _reframeDims[preset];
            el.reframeInfo.textContent = "Output: " + d[0] + " × " + d[1] + " px";
        } else if (isCustom) {
            el.reframeInfo.textContent = "Output: " + (el.reframeCustomW.value || "?") + " × " + (el.reframeCustomH.value || "?") + " px";
        }
    }

    function runReframe() {
        console.log("[OpenCut] runReframe selectedPath:", selectedPath);
        var preset = el.reframePreset.value;
        var w, h;
        if (preset === "custom") {
            w = parseInt(el.reframeCustomW.value) || 1080;
            h = parseInt(el.reframeCustomH.value) || 1920;
        } else {
            var d = _reframeDims[preset] || [1080, 1920];
            w = d[0]; h = d[1];
        }
        var pos = el.reframeCropPos.value;
        if (pos === "face") {
            var smoothing = el.faceSmoothing ? parseFloat(el.faceSmoothing.value) : 0.3;
            startJob("/video/reframe/face", {
                filepath: selectedPath, output_dir: projectFolder,
                width: w, height: h,
                smoothing: smoothing, face_padding: 1.5,
            });
            return;
        }
        startJob("/video/reframe", {
            filepath: selectedPath, output_dir: projectFolder,
            width: w, height: h,
            mode: el.reframeMode.value,
            position: pos,
            bg_color: el.reframePadColor.value,
            quality: el.reframeQuality.value
        });
    }

    function runUpscale() {
        startJob("/video/upscale/run", { filepath: selectedPath, output_dir: projectFolder,
            preset: el.upscalePreset.value, scale: parseInt(el.upscaleScale.value) });
    }

    // --- COLOR CORRECTION ---
    function runColor() {
        startJob("/video/color/correct", { filepath: selectedPath, output_dir: projectFolder,
            exposure: parseFloat(el.ccExposure.value), contrast: parseFloat(el.ccContrast.value),
            saturation: parseFloat(el.ccSaturation.value), temperature: parseFloat(el.ccTemp.value),
            shadows: parseFloat(el.ccShadows.value), highlights: parseFloat(el.ccHighlights.value) });
    }

    // --- OBJECT/WATERMARK REMOVAL ---
    function runRemove() {
        startJob("/video/remove/watermark", { filepath: selectedPath, output_dir: projectFolder,
            method: el.removeMethod.value,
            region: { x: parseInt(el.removeX.value), y: parseInt(el.removeY.value),
                width: parseInt(el.removeW.value), height: parseInt(el.removeH.value) } });
    }

    // --- FACE AI ---
    function showFaceAiParams() {
        el.faceSwapParams.classList.toggle("hidden", el.faceAiMode.value !== "swap");
    }
    function runFaceAi() {
        if (el.faceAiMode.value === "swap") {
            var ref = el.faceRefPath.value.trim();
            if (!ref) { showAlert("Enter reference face image path."); return; }
            startJob("/video/face/swap", { filepath: selectedPath, reference_face: ref, output_dir: projectFolder });
        } else {
            startJob("/video/face/enhance", { filepath: selectedPath, output_dir: projectFolder });
        }
    }

    // --- ANIMATED CAPTIONS ---
    function runAnimCap() {
        if (lastTranscriptSegments && lastTranscriptSegments.length > 0 && lastTranscriptSegments[0].words) {
            // We have word-level segments, render directly
            startJob("/captions/animated/render", {
                filepath: selectedPath,
                word_segments: extractWordSegments(lastTranscriptSegments),
                animation: el.animCapPreset.value,
                font_size: parseInt(el.animCapFontSize.value),
                max_words: parseInt(el.animCapWpl.value),
                output_dir: projectFolder,
            });
        } else {
            // Transcribe first with word-level timing
            showAlert("Step 1/2: Transcribing with word-level timing first...");
            pendingAnimCap = true;
            jobStepCurrent = 1;
            jobStepTotal = 2;
            startJob("/transcript", {
                filepath: selectedPath,
                model: el.animCapModel.value,
                word_level: true,
            });
        }
    }

    // --- AI MUSIC GENERATION ---
    function runMusicAi() {
        var prompt = el.musicAiPrompt.value.trim();
        if (!prompt) { showAlert("Enter a music prompt."); return; }
        startJob("/audio/music-ai/generate", { prompt: prompt, output_dir: projectFolder, no_input: true,
            model: el.musicAiModel.value, duration: parseFloat(el.musicAiDur.value),
            temperature: parseFloat(el.musicAiTemp.value) });
    }

    // --- EXPORT TAB ---
    function runExpTranscript() {
        var fmt = el.expTranscriptFormat.value;
        if (fmt === "plain" || fmt === "timestamped") {
            // These need transcription first, then text export
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

    // ================================================================
    // Extended Job Result Handling (via addJobDoneListener)
    // ================================================================

    // Listener: Clear pending chain flags on error/cancel
    addJobDoneListener(function (job) {
        if (job.status === "error" || job.status === "cancelled") {
            pendingBurnin = false;
            pendingAnimCap = false;
            pendingTranslate = false;
            jobStepCurrent = 0;
            jobStepTotal = 0;
        }
    });

    // Listener: Handle transcript results — chaining and editor
    addJobDoneListener(function (job) {
        if (job.type !== "transcript" || job.status !== "complete" || !job.result) return;

        transcriptData = job.result;
        if (job.result.segments) {
            lastTranscriptSegments = job.result.segments;
            if (selectedPath) cacheTranscriptSegments(selectedPath, job.result.segments);
        }
        renderTranscriptEditor(job.result);

        // Chain into burn-in if pending
        if (pendingBurnin && job.result.segments) {
            pendingBurnin = false;
            jobStepCurrent = 2;
            showAlert("Step 2/2: Burning in captions...");
            startJob("/captions/burnin/segments", {
                filepath: selectedPath,
                segments: job.result.segments,
                style: el.burninStyle.value,
                output_dir: projectFolder,
            });
            return true; // handled — skip default onJobDone behavior
        }

        // Chain into animated captions if pending
        if (pendingAnimCap && job.result.segments) {
            pendingAnimCap = false;
            jobStepCurrent = 2;
            showAlert("Step 2/2: Rendering animated captions...");
            startJob("/captions/animated/render", {
                filepath: selectedPath,
                word_segments: extractWordSegments(job.result.segments),
                animation: el.animCapPreset.value,
                font_size: parseInt(el.animCapFontSize.value),
                max_words: parseInt(el.animCapWpl.value),
                output_dir: projectFolder,
            });
            return true;
        }

        // Chain into translation if pending
        if (pendingTranslate && job.result.segments) {
            pendingTranslate = false;
            jobStepCurrent = 2;
            showAlert("Step 2/2: Translating captions...");
            startJob("/captions/translate", {
                filepath: selectedPath,
                segments: job.result.segments,
                source_lang: el.translateSourceLang.value,
                target_lang: el.translateTargetLang.value,
                format: el.translateFormat.value,
                output_dir: projectFolder,
            });
            return true;
        }
    });

    // Listener: Handle beat detection results
    addJobDoneListener(function (job) {
        if (job.type === "beats" && job.status === "complete" && job.result) {
            el.beatResults.classList.remove("hidden");
            el.bpmValue.textContent = safeFixed(job.result.bpm, 0);
            el.beatCount.textContent = job.result.total_beats != null ? job.result.total_beats : "--";
            el.beatConfidence.textContent = safeFixed(job.result.confidence * 100, 0) + "%";
        }
    });

    // Listener: Handle scene detection results
    addJobDoneListener(function (job) {
        if (job.type === "scenes" && job.status === "complete" && job.result) {
            el.sceneResults.classList.remove("hidden");
            el.sceneCount.textContent = job.result.total_scenes != null ? job.result.total_scenes : "--";
            el.avgSceneLen.textContent = safeFixed(job.result.avg_scene_length, 1) + "s";
            if (job.result.youtube_chapters) {
                el.ytChapters.classList.remove("hidden");
                el.ytChaptersText.value = job.result.youtube_chapters;
            }
        }
    });

    // Listener: Reset step counters after final job (only if no chain/workflow pending)
    addJobDoneListener(function (job) {
        if (!pendingBurnin && !pendingAnimCap && !pendingTranslate && !workflowActive) {
            jobStepCurrent = 0;
            jobStepTotal = 0;
        }
    });

    // Listener: Populate summarize results panel
    addJobDoneListener(function (job) {
        if (job.type !== "summarize" || job.status !== "complete" || !job.result) return;
        var r = job.result;
        var text = "";
        if (r.bullet_points && r.bullet_points.length) {
            for (var i = 0; i < r.bullet_points.length; i++) {
                text += "\u2022 " + r.bullet_points[i] + "\n";
            }
        } else if (r.text) {
            text = r.text;
        }
        if (r.topics && r.topics.length) {
            text += "\nTopics: " + r.topics.join(", ");
        }
        if (el.summaryResult) el.summaryResult.classList.remove("hidden");
        if (el.summaryContent) el.summaryContent.textContent = text || "No summary generated.";
    });

    // ================================================================
    // Transcript Editor
    // ================================================================
    // ---- Transcript Undo/Redo ----
    var transcriptHistory = [];
    var transcriptHistoryIdx = -1;
    var MAX_TRANSCRIPT_HISTORY = 50;

    function snapshotTranscript() {
        if (!transcriptData || !transcriptData.segments) return;
        var snap = [];
        for (var i = 0; i < transcriptData.segments.length; i++) {
            snap.push(transcriptData.segments[i].text);
        }
        // Trim redo stack
        if (transcriptHistoryIdx < transcriptHistory.length - 1) {
            transcriptHistory = transcriptHistory.slice(0, transcriptHistoryIdx + 1);
        }
        transcriptHistory.push(snap);
        if (transcriptHistory.length > MAX_TRANSCRIPT_HISTORY) {
            transcriptHistory = transcriptHistory.slice(-MAX_TRANSCRIPT_HISTORY);
            transcriptHistoryIdx = Math.min(transcriptHistoryIdx, transcriptHistory.length - 1);
        }
        transcriptHistoryIdx = transcriptHistory.length - 1;
        updateUndoRedoButtons();
    }

    function restoreTranscriptSnapshot(snap) {
        if (!transcriptData || !transcriptData.segments) return;
        for (var i = 0; i < snap.length && i < transcriptData.segments.length; i++) {
            transcriptData.segments[i].text = snap[i];
        }
        // Re-render segment textareas
        var textareas = el.transcriptSegments.querySelectorAll(".transcript-seg-text");
        for (var i = 0; i < textareas.length && i < snap.length; i++) {
            textareas[i].value = snap[i];
            autoResize(textareas[i]);
        }
        if (lastTranscriptSegments) {
            for (var i = 0; i < snap.length && i < lastTranscriptSegments.length; i++) {
                lastTranscriptSegments[i].text = snap[i];
            }
        }
    }

    function undoTranscript() {
        if (transcriptHistoryIdx <= 0) return;
        transcriptHistoryIdx--;
        restoreTranscriptSnapshot(transcriptHistory[transcriptHistoryIdx]);
        updateUndoRedoButtons();
    }

    function redoTranscript() {
        if (transcriptHistoryIdx >= transcriptHistory.length - 1) return;
        transcriptHistoryIdx++;
        restoreTranscriptSnapshot(transcriptHistory[transcriptHistoryIdx]);
        updateUndoRedoButtons();
    }

    function updateUndoRedoButtons() {
        el.transcriptUndoBtn.disabled = transcriptHistoryIdx <= 0;
        el.transcriptRedoBtn.disabled = transcriptHistoryIdx >= transcriptHistory.length - 1;
    }

    var editDebounceTimer = null;

    function renderTranscriptEditor(data) {
        ensureTranscriptDelegation();
        // Clear any pending debounce from previous render
        if (editDebounceTimer) { clearTimeout(editDebounceTimer); editDebounceTimer = null; }

        el.transcriptEditor.classList.remove("hidden");
        var wordCount = data.word_count || 0;
        var segCount = data.segments ? data.segments.length : 0;
        el.transcriptInfo.textContent = wordCount + " words | " + segCount + " segments | " + (data.language || "en");
        if (!data.segments || !data.segments.length) return;

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
        }

        // Reset history and take initial snapshot
        transcriptHistory = [];
        transcriptHistoryIdx = -1;
        snapshotTranscript();
    }

    function autoResize(textarea) {
        textarea.style.height = "auto";
        textarea.style.height = textarea.scrollHeight + "px";
    }

    // Delegated input handler for transcript textareas (avoids 1000+ listeners)
    var _transcriptDelegationAdded = false;
    function ensureTranscriptDelegation() {
        if (_transcriptDelegationAdded || !el.transcriptSegments) return;
        _transcriptDelegationAdded = true;
        el.transcriptSegments.addEventListener("input", function (e) {
            var ta = e.target;
            if (!ta || !ta.classList.contains("transcript-seg-text")) return;
            autoResize(ta);
            var idx = parseInt(ta.getAttribute("data-idx"));
            if (idx >= 0 && transcriptData && idx < transcriptData.segments.length) {
                transcriptData.segments[idx].text = ta.value;
            }
            if (editDebounceTimer) clearTimeout(editDebounceTimer);
            editDebounceTimer = setTimeout(function () { snapshotTranscript(); }, 500);
        });
    }

    // ================================================================
    // Style Preview
    // ================================================================
    function loadStylePreview() {
        api("GET", "/caption-styles", null, function (err, data) {
            if (!err && data && data.styles) {
                for (var i = 0; i < data.styles.length; i++) {
                    var s = data.styles[i];
                    stylePreviewMap[s.name] = {
                        css: s.preview_css || "",
                        highlight: s.highlight_color || "#ffe600",
                        action: s.action_color || "#ff3232"
                    };
                }
                updateStylePreview();
            }
        });
    }

    function updateStylePreview() {
        var styleName = el.captionStyle.value;
        var info = stylePreviewMap[styleName] || {};
        var css = info.css || "";
        var hlColor = info.highlight || "#ffe600";
        var actColor = info.action || "#ff3232";
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
                    hlWord.style.color = hlColor;
                    hlWord.style.transform = "scale(1.1)";
                }
                // Action color
                var actWord = previewBg.querySelector(".sp-action");
                if (actWord) {
                    actWord.style.cssText = css;
                    actWord.style.color = actColor;
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

        // Whisper status from health check
        api("GET", "/health", null, function (err, data) {
            if (!err && data) {
                if (data.capabilities && data.capabilities.captions) {
                    el.whisperStatusText.textContent = "Installed (" + (data.capabilities.whisper_backend || "unknown") + ")";
                    el.whisperStatusText.style.color = "var(--success)";
                } else {
                    el.whisperStatusText.textContent = "Not installed";
                    el.whisperStatusText.style.color = "var(--error)";
                }
                // CPU mode status
                if (data.capabilities && data.capabilities.whisper_cpu_mode) {
                    el.whisperCpuMode.checked = true;
                    el.whisperDeviceText.textContent = "CPU (forced)";
                    el.whisperDeviceText.style.color = "var(--warning)";
                } else {
                    el.whisperCpuMode.checked = false;
                    el.whisperDeviceText.textContent = "Auto (GPU if available)";
                    el.whisperDeviceText.style.color = "var(--text-secondary)";
                }
            }
        });

        // GPU info
        api("GET", "/system/gpu", null, function (err, data) {
            if (!err && data) {
                el.gpuName.textContent = data.available ? data.name : "None detected";
                el.gpuVram.textContent = data.available ? safeFixed(data.vram_mb / 1024, 1) + " GB" : "--";
            }
        });
    }

    function installWhisper() {
        showAlert("Installing faster-whisper... This may take a minute.");
        startJob("/install-whisper", { backend: "faster-whisper" });
    }

    function reinstallWhisper() {
        var cpuMode = el.whisperCpuMode.checked;
        showAlert("Reinstalling Whisper" + (cpuMode ? " in CPU mode" : "") + "... Please wait.");
        startJob("/whisper/reinstall", { backend: "faster-whisper", cpu_mode: cpuMode, no_input: true });
    }

    function clearWhisperCache() {
        showAlert("Clearing Whisper cache...");
        api("POST", "/whisper/clear-cache", {}, function (err, data) {
            if (!err && data) {
                if (data.success) {
                    showAlert("Cache cleared! Cleared " + (data.cleared ? data.cleared.length : 0) + " location(s). Models will re-download on next use.");
                } else {
                    showAlert("Cache clear had errors: " + (data.errors ? data.errors.join(", ") : "unknown"));
                }
            } else {
                showAlert("Failed to clear cache.");
            }
        });
    }

    function toggleCpuMode() {
        var cpuMode = el.whisperCpuMode.checked;
        api("POST", "/whisper/settings", { cpu_mode: cpuMode }, function (err, data) {
            if (!err && data && data.success) {
                if (cpuMode) {
                    el.whisperDeviceText.textContent = "CPU (forced)";
                    el.whisperDeviceText.style.color = "var(--warning)";
                    showAlert("CPU mode enabled. Whisper will use CPU only.");
                } else {
                    el.whisperDeviceText.textContent = "Auto (GPU if available)";
                    el.whisperDeviceText.style.color = "var(--text-secondary)";
                    showAlert("CPU mode disabled. Whisper will try to use GPU.");
                }
            } else {
                showAlert("Failed to update settings.");
                // Revert checkbox
                el.whisperCpuMode.checked = !cpuMode;
            }
        });
    }

    function restartBackend() {
        showAlert("Restarting backend...");
        api("POST", "/shutdown", {}, function () {
            // Backend will shut down, then auto-restart via launcher
            setTimeout(function () {
                checkHealth();
            }, 3000);
        });
    }

    function openLogs() {
        var isWin = navigator.platform.indexOf("Win") !== -1;
        try {
            var exec = require("child_process").exec;
            if (isWin) {
                var logPath = process.env.USERPROFILE + "\\.opencut\\server.log";
                exec('start notepad "' + logPath + '"', function (err) {
                    if (err) {
                        // Fallback: open the folder
                        exec('explorer "' + process.env.USERPROFILE + '\\.opencut"');
                    }
                });
            } else {
                var home = process.env.HOME || "~";
                exec('open "' + home + '/.opencut/server.log" 2>/dev/null || xdg-open "' + home + '/.opencut/server.log"', function (err) {
                    if (err) {
                        exec('open "' + home + '/.opencut/"');
                    }
                });
            }
        } catch (e) {
            // Node not available - show path as fallback
            var fallback = isWin ? "%USERPROFILE%\\.opencut\\server.log" : "~/.opencut/server.log";
            showAlert("Log file: " + fallback);
        }
    }

    // ================================================================
    // Local Settings Persistence
    // ================================================================
    var LOCAL_SETTINGS_KEY = "opencut_settings";
    
    function saveLocalSettings() {
        var settings = {
            autoImport: el.settingsAutoImport.checked,
            autoOpen: el.settingsAutoOpen.checked,
            showNotifications: el.settingsShowNotifications.checked,
            outputDir: el.settingsOutputDir.value,
            defaultModel: el.settingsDefaultModel.value,
            theme: el.settingsTheme.value
        };
        try {
            localStorage.setItem(LOCAL_SETTINGS_KEY, JSON.stringify(settings));
            showToast("Settings saved", "success");
        } catch (e) {
            // localStorage may not be available in CEP
        }
    }
    
    function loadLocalSettings() {
        try {
            var saved = localStorage.getItem(LOCAL_SETTINGS_KEY);
            if (saved) {
                var settings = JSON.parse(saved);
                if (settings.autoImport !== undefined) el.settingsAutoImport.checked = settings.autoImport;
                if (settings.autoOpen !== undefined) el.settingsAutoOpen.checked = settings.autoOpen;
                if (settings.showNotifications !== undefined) el.settingsShowNotifications.checked = settings.showNotifications;
                if (settings.outputDir) el.settingsOutputDir.value = settings.outputDir;
                if (settings.defaultModel) el.settingsDefaultModel.value = settings.defaultModel;
                if (settings.theme) el.settingsTheme.value = settings.theme;
            }
        } catch (e) {
            // localStorage may not be available
        }
        // Always apply current theme on load
        applyTheme(el.settingsTheme.value);
    }

    function applyTheme(themeName) {
        if (themeName === "cyberpunk") {
            document.documentElement.removeAttribute("data-theme");
        } else {
            document.documentElement.setAttribute("data-theme", themeName);
        }
        // Also set on body for CSS selectors that target body
        document.body.setAttribute("data-theme", themeName === "cyberpunk" ? "" : themeName);
        closeAllDropdowns();
    }
    
    function getLocalSetting(key, defaultVal) {
        try {
            var saved = localStorage.getItem(LOCAL_SETTINGS_KEY);
            if (saved) {
                var settings = JSON.parse(saved);
                return settings[key] !== undefined ? settings[key] : defaultVal;
            }
        } catch (e) {}
        return defaultVal;
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

        // Video sliders
        el.wmMaxBbox.addEventListener("input", function () { el.wmMaxBboxVal.textContent = this.value + "%"; });
        el.wmFrameSkip.addEventListener("input", function () { el.wmFrameSkipVal.textContent = this.value; });
        el.sceneThreshold.addEventListener("input", function () { el.sceneThresholdVal.textContent = safeFixed(parseFloat(this.value), 2); });
        el.minSceneLen.addEventListener("input", function () { el.minSceneLenVal.textContent = this.value + "s"; });

        // Video FX sliders
        el.vfxStabSmoothing.addEventListener("input", function () { el.vfxStabSmoothingVal.textContent = this.value; });
        el.vfxStabZoom.addEventListener("input", function () { el.vfxStabZoomVal.textContent = this.value + "%"; });
        el.vfxVignetteIntensity.addEventListener("input", function () { el.vfxVignetteIntensityVal.textContent = this.value; });
        el.vfxGrainIntensity.addEventListener("input", function () { el.vfxGrainIntensityVal.textContent = this.value; });
        el.vfxChromakeySim.addEventListener("input", function () { el.vfxChromakeySimVal.textContent = safeFixed(parseFloat(this.value), 2); });
        el.vfxChromakeyBlend.addEventListener("input", function () { el.vfxChromakeyBlendVal.textContent = safeFixed(parseFloat(this.value), 2); });
        el.vfxLutIntensity.addEventListener("input", function () { el.vfxLutIntensityVal.textContent = this.value; });

        // Video AI sliders
        el.vidAiDenoiseStrength.addEventListener("input", function () { el.vidAiDenoiseStrengthVal.textContent = this.value; });

        // Face blur slider
        el.faceBlurStrength.addEventListener("input", function () { el.faceBlurStrengthVal.textContent = this.value; });

        // Style transfer slider
        el.styleIntensity.addEventListener("input", function () { el.styleIntensityVal.textContent = this.value; });

        // Karaoke font size slider
        el.karaokeFontSize.addEventListener("input", function () { el.karaokeFontSizeVal.textContent = this.value + "px"; });

        // TTS rate slider
        el.ttsRate.addEventListener("input", function () {
            var v = parseInt(this.value);
            el.ttsRateVal.textContent = (v >= 0 ? "+" : "") + v + "%";
        });

        // SFX sliders
        el.toneFreq.addEventListener("input", function () { el.toneFreqVal.textContent = this.value + " Hz"; });
        el.sfxDuration.addEventListener("input", function () { el.sfxDurationVal.textContent = this.value + "s"; });

        // Speed multiplier slider
        el.speedMultiplier.addEventListener("input", function () { el.speedMultiplierVal.textContent = this.value + "x"; });

        // LUT intensity slider
        el.lutIntensity.addEventListener("input", function () { el.lutIntensityVal.textContent = this.value; });

        // Duck sliders
        el.duckMusicVol.addEventListener("input", function () { el.duckMusicVolVal.textContent = this.value; });
        el.duckAmount.addEventListener("input", function () { el.duckAmountVal.textContent = this.value; });

        // Phase 6 sliders
        el.chromaTol.addEventListener("input", function () { el.chromaTolVal.textContent = this.value; });
        el.pipScale.addEventListener("input", function () { el.pipScaleVal.textContent = this.value; });
        el.blendOpacity.addEventListener("input", function () { el.blendOpacityVal.textContent = this.value; });
        el.transDur.addEventListener("input", function () { el.transDurVal.textContent = this.value + "s"; });
        el.particleDensity.addEventListener("input", function () { el.particleDensityVal.textContent = this.value; });
        el.titleDur.addEventListener("input", function () { el.titleDurVal.textContent = this.value + "s"; });
        el.titleFontSize.addEventListener("input", function () { el.titleFontSizeVal.textContent = this.value + "px"; });
        el.ccExposure.addEventListener("input", function () { el.ccExposureVal.textContent = this.value; });
        el.ccContrast.addEventListener("input", function () { el.ccContrastVal.textContent = this.value; });
        el.ccSaturation.addEventListener("input", function () { el.ccSaturationVal.textContent = this.value; });
        el.ccTemp.addEventListener("input", function () { el.ccTempVal.textContent = this.value; });
        el.ccShadows.addEventListener("input", function () { el.ccShadowsVal.textContent = this.value; });
        el.ccHighlights.addEventListener("input", function () { el.ccHighlightsVal.textContent = this.value; });
        el.animCapFontSize.addEventListener("input", function () { el.animCapFontSizeVal.textContent = this.value + "px"; });
        el.animCapWpl.addEventListener("input", function () { el.animCapWplVal.textContent = this.value; });
        el.musicAiDur.addEventListener("input", function () { el.musicAiDurVal.textContent = this.value + "s"; });
        el.musicAiTemp.addEventListener("input", function () { el.musicAiTempVal.textContent = this.value; });
    }

    // ================================================================
    // Refresh & Retry
    // ================================================================
    function refreshAll() {
        el.refreshAllBtn.classList.add("spinning");
        settingsLoaded = false;
        capabilitiesLoaded = false;
        checkHealth();
        scanProjectMedia();
        loadStylePreview();
        setTimeout(function () {
            el.refreshAllBtn.classList.remove("spinning");
            showAlert("Refreshed");
        }, 2500);
    }

    // ================================================================
    // Utility
    // ================================================================
    var _alertTimer = null;
    function showAlert(msg) {
        el.alertText.textContent = msg;
        el.alertBanner.classList.remove("hidden");
        if (_alertTimer) clearTimeout(_alertTimer);
        _alertTimer = setTimeout(function () { el.alertBanner.classList.add("hidden"); }, 15000);
    }

    function esc(s) {
        if (!s) return "";
        return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
    }

    function safeFixed(v, digits) { var n = Number(v); return isFinite(n) ? n.toFixed(digits) : "0"; }
    
    // Escape for use inside JSX string arguments (handles Windows paths)
    function escPath(s) {
        if (!s) return "";
        // Double backslashes for JavaScript string, then escape quotes
        return s.replace(/\\/g, "\\\\").replace(/"/g, '\\"');
    }

    function extractWordSegments(segments) {
        var words = [];
        for (var i = 0; i < segments.length; i++) {
            if (segments[i].words) {
                for (var j = 0; j < segments[i].words.length; j++) {
                    words.push(segments[i].words[j]);
                }
            }
        }
        return words;
    }

    function fmtDur(s) {
        if (!s && s !== 0) return "--";
        var m = Math.floor(s / 60);
        var sec = Math.floor(s % 60);
        return m + ":" + (sec < 10 ? "0" : "") + sec;
    }

    // ================================================================
    // Drop Zone
    // ================================================================
    function initDropZone() {
        if (!el.dropZone) return;
        var dz = el.dropZone;

        dz.addEventListener("dragover", function (e) {
            e.preventDefault();
            e.stopPropagation();
            dz.classList.add("drag-over");
        });
        dz.addEventListener("dragleave", function (e) {
            e.preventDefault();
            e.stopPropagation();
            dz.classList.remove("drag-over");
        });
        dz.addEventListener("drop", function (e) {
            e.preventDefault();
            e.stopPropagation();
            dz.classList.remove("drag-over");
            var files = e.dataTransfer && e.dataTransfer.files;
            if (files && files.length > 0) {
                var f = files[0];
                var path = f.path || f.name;
                if (path) selectFile(path, f.name || path.split(/[/\\]/).pop());
            }
        });
        dz.addEventListener("click", function () {
            browseForFile();
        });
    }

    // ================================================================
    // Theme Quick Toggle
    // ================================================================
    var THEME_LIST = [
        { value: "cyberpunk", label: "Cyberpunk Neon" },
        { value: "midnight", label: "Midnight OLED" },
        { value: "catppuccin", label: "Catppuccin Mocha" },
        { value: "github", label: "GitHub Dark" },
        { value: "stealth", label: "Stealth" },
        { value: "ember", label: "Ember" }
    ];

    function initThemeToggle() {
        if (!el.themeToggleBtn || !el.themeMenu) return;
        // Build menu items
        var html = "";
        for (var i = 0; i < THEME_LIST.length; i++) {
            html += '<button class="theme-menu-item" data-theme="' + THEME_LIST[i].value + '">' + THEME_LIST[i].label + '</button>';
        }
        el.themeMenu.innerHTML = html;

        el.themeToggleBtn.addEventListener("click", function (e) {
            e.stopPropagation();
            var isOpen = el.themeMenu.classList.contains("open");
            el.themeMenu.classList.toggle("open");
            if (!isOpen) updateThemeMenuActive();
        });

        el.themeMenu.addEventListener("click", function (e) {
            var item = e.target.closest(".theme-menu-item");
            if (!item) return;
            var theme = item.getAttribute("data-theme");
            applyTheme(theme);
            if (el.settingsTheme) {
                el.settingsTheme.value = theme;
                if (el.settingsTheme._customDropdown) el.settingsTheme._customDropdown.updateText();
            }
            saveLocalSettings();
            el.themeMenu.classList.remove("open");
        });

        document.addEventListener("click", function () {
            el.themeMenu.classList.remove("open");
        });
    }

    function updateThemeMenuActive() {
        var current = el.settingsTheme ? el.settingsTheme.value : "cyberpunk";
        var items = el.themeMenu.querySelectorAll(".theme-menu-item");
        for (var i = 0; i < items.length; i++) {
            if (items[i].getAttribute("data-theme") === current) {
                items[i].classList.add("active");
            } else {
                items[i].classList.remove("active");
            }
        }
    }

    // ================================================================
    // Job History
    // ================================================================
    var jobHistoryList = [];
    var MAX_JOB_HISTORY = 50;

    function addJobHistory(job) {
        if (!job || !job.type) return;
        jobHistoryList.unshift({
            type: job.type,
            status: job.status || "complete",
            message: job.message || "",
            time: new Date().toLocaleTimeString(),
            endpoint: lastJobEndpoint || "",
            payload: lastJobPayload ? JSON.parse(JSON.stringify(lastJobPayload)) : null
        });
        if (jobHistoryList.length > MAX_JOB_HISTORY) jobHistoryList.pop();
        renderJobHistory();
        // Show toast notification on completion
        if (job.status === "complete") {
            showToast(job.type + " completed", "success");
        } else if (job.status === "error") {
            showToast(job.type + " failed", "error");
        }
    }

    function renderJobHistory() {
        if (!el.jobHistory || !el.jobHistoryToggle) return;
        el.jobHistoryToggle.textContent = "History (" + jobHistoryList.length + ")";
        var html = "";
        for (var i = 0; i < jobHistoryList.length; i++) {
            var h = jobHistoryList[i];
            var statusClass = h.status === "complete" ? "complete" : (h.status === "cancelled" ? "cancelled" : "error");
            html += '<div class="job-history-item" data-idx="' + i + '">' +
                '<span style="display:flex;align-items:center;gap:4px"><span class="job-history-status ' + statusClass + '"></span>' +
                esc(h.type) + '</span>' +
                '<span style="display:flex;align-items:center;gap:6px">' +
                '<span style="font-size:10px;color:var(--text-muted)">' + esc(h.time) + '</span>' +
                (h.endpoint ? '<button type="button" class="btn-sm job-history-rerun" data-idx="' + i + '" title="Re-run this job" style="padding:1px 5px;font-size:9px">Re-run</button>' : '') +
                '</span></div>';
        }
        el.jobHistory.innerHTML = html;
    }

    // Event delegation for job history re-run buttons (avoids listener accumulation)
    var _jobHistoryDelegationAdded = false;
    function ensureJobHistoryDelegation() {
        if (_jobHistoryDelegationAdded || !el.jobHistory) return;
        _jobHistoryDelegationAdded = true;
        el.jobHistory.addEventListener("click", function (e) {
            var btn = e.target.closest(".job-history-rerun");
            if (!btn) return;
            e.stopPropagation();
            var idx = parseInt(btn.getAttribute("data-idx"));
            var entry = jobHistoryList[idx];
            if (entry && entry.endpoint && entry.payload) {
                startJob(entry.endpoint, entry.payload);
            }
        });
    }

    function initJobHistory() {
        if (!el.jobHistoryToggle || !el.jobHistory) return;
        ensureJobHistoryDelegation();
        el.jobHistoryToggle.addEventListener("click", function () {
            el.jobHistory.classList.toggle("open");
        });

        // Add listener to record finished jobs
        addJobDoneListener(function (job) {
            addJobHistory(job);
        });
    }

    // ================================================================
    // Escape to Cancel + Keyboard Shortcuts
    // ================================================================
    function initKeyboardShortcuts() {
        document.addEventListener("keydown", function (e) {
            // Escape to cancel running job
            if (e.key === "Escape" && currentJob && !e.defaultPrevented) {
                cancelJob();
                return;
            }
            // Don't handle shortcuts when typing in inputs
            var tag = e.target.tagName;
            if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || e.target.isContentEditable) return;

            // Enter to run the primary action for active tab/subtab
            if (e.key === "Enter" && !currentJob) {
                var activePanel = document.querySelector(".nav-panel.active");
                if (!activePanel) return;
                var activeSub = activePanel.querySelector(".sub-panel.active");
                var target = activeSub || activePanel;
                var primaryBtn = target.querySelector(".btn-primary:not([disabled])");
                if (primaryBtn) {
                    e.preventDefault();
                    primaryBtn.click();
                }
                return;
            }

            // Tab switching: 1-6 for main tabs
            if (e.key >= "1" && e.key <= "6" && !e.ctrlKey && !e.altKey && !e.metaKey) {
                var tabBtns = document.querySelectorAll(".nav-tab");
                var idx = parseInt(e.key) - 1;
                if (tabBtns[idx]) {
                    e.preventDefault();
                    tabBtns[idx].click();
                }
            }
        });
    }

    // ================================================================
    // Preset Save / Load
    // ================================================================
    function initPresets() {
        if (!el.savePresetBtn) return;

        el.savePresetBtn.addEventListener("click", function () {
            var name = el.presetNameInput ? el.presetNameInput.value.trim() : "";
            if (!name) { showAlert("Enter a preset name."); return; }
            var settings = collectCurrentSettings();
            api("POST", "/presets/save", { name: name, settings: settings }, function (err, data) {
                if (!err && data && data.success) {
                    showAlert("Preset saved: " + name);
                    showToast("Preset '" + name + "' saved", "success");
                    el.presetNameInput.value = "";
                    refreshPresetList();
                } else {
                    showAlert("Failed to save preset.");
                }
            });
        });

        if (el.loadPresetBtn) el.loadPresetBtn.addEventListener("click", function () {
            if (!el.presetSelect || !el.presetSelect.value) { showAlert("Select a preset first."); return; }
            api("GET", "/presets", null, function (err, data) {
                if (!err && data) {
                    var preset = data[el.presetSelect.value];
                    if (preset && preset.settings) {
                        applyPresetSettings(preset.settings);
                        showAlert("Preset loaded: " + el.presetSelect.value);
                        showToast("Preset loaded", "info");
                    }
                }
            });
        });

        if (el.deletePresetBtn) el.deletePresetBtn.addEventListener("click", function () {
            if (!el.presetSelect || !el.presetSelect.value) return;
            var name = el.presetSelect.value;
            api("POST", "/presets/delete", { name: name }, function (err, data) {
                if (!err && data && data.success) {
                    showAlert("Preset deleted: " + name);
                    showToast("Preset deleted", "success");
                    refreshPresetList();
                }
            });
        });

        refreshPresetList();
    }

    function refreshPresetList() {
        if (!el.presetSelect) return;
        api("GET", "/presets", null, function (err, data) {
            if (err || !data) return;
            var keys = Object.keys(data);
            var html = "";
            if (keys.length === 0) {
                html = '<option value="" disabled selected>No presets saved</option>';
            } else {
                html = '<option value="" disabled selected>Select preset...</option>';
                for (var i = 0; i < keys.length; i++) {
                    html += '<option value="' + esc(keys[i]) + '">' + esc(keys[i]) + '</option>';
                }
            }
            el.presetSelect.innerHTML = html;
            if (el.presetSelect._customDropdown) el.presetSelect._customDropdown.update();
        });
    }

    function collectCurrentSettings() {
        var s = {};
        // Gather values from all visible form controls
        var selects = document.querySelectorAll("select:not(.no-custom)");
        for (var i = 0; i < selects.length; i++) {
            if (selects[i].id) s["s_" + selects[i].id] = selects[i].value;
        }
        var ranges = document.querySelectorAll('input[type="range"]');
        for (var i = 0; i < ranges.length; i++) {
            if (ranges[i].id) s["r_" + ranges[i].id] = ranges[i].value;
        }
        var checks = document.querySelectorAll('input[type="checkbox"]');
        for (var i = 0; i < checks.length; i++) {
            if (checks[i].id) s["c_" + checks[i].id] = checks[i].checked;
        }
        return s;
    }

    function applyPresetSettings(s) {
        for (var key in s) {
            if (!s.hasOwnProperty(key)) continue;
            var id = key.substring(2);
            var elem = document.getElementById(id);
            if (!elem) continue;
            if (key.charAt(0) === "s") {
                elem.value = s[key];
                if (elem._customDropdown) elem._customDropdown.updateText();
                var evt = new Event("change", { bubbles: true });
                elem.dispatchEvent(evt);
            } else if (key.charAt(0) === "r") {
                elem.value = s[key];
                var evt2 = new Event("input", { bubbles: true });
                elem.dispatchEvent(evt2);
            } else if (key.charAt(0) === "c") {
                elem.checked = s[key];
            }
        }
    }

    // ================================================================
    // Model Management
    // ================================================================
    function initModelManagement() {
        if (!el.refreshModelsBtn) return;
        el.refreshModelsBtn.addEventListener("click", refreshModelList);
    }

    function refreshModelList() {
        if (!el.modelList) return;
        el.modelList.innerHTML = '<div class="hint">Scanning...</div>';
        api("GET", "/models/list", null, function (err, data) {
            if (err || !data) {
                el.modelList.innerHTML = '<div class="hint">Failed to load models.</div>';
                return;
            }
            if (!data.models || data.models.length === 0) {
                el.modelList.innerHTML = '<div class="hint">No models found.</div>';
                if (el.modelsTotalSize) el.modelsTotalSize.textContent = "0 MB";
                return;
            }
            var html = "";
            for (var i = 0; i < data.models.length; i++) {
                var m = data.models[i];
                var sizeStr = m.size_mb >= 1024 ? safeFixed(m.size_mb / 1024, 1) + " GB" : safeFixed(m.size_mb, 0) + " MB";
                html += '<div class="model-item">' +
                    '<div class="model-item-info"><span class="model-item-name">' + esc(m.name) + '</span>' +
                    '<span class="model-item-meta">' + sizeStr + ' - ' + esc(m.source) + '</span></div>' +
                    '<button class="model-item-delete" data-path="' + esc(m.path) + '" title="Delete model">Delete</button>' +
                    '</div>';
            }
            el.modelList.innerHTML = html;
            if (el.modelsTotalSize) {
                var totalStr = data.total_mb >= 1024 ? safeFixed(data.total_mb / 1024, 1) + " GB" : safeFixed(data.total_mb, 0) + " MB";
                el.modelsTotalSize.textContent = totalStr;
            }
            // Attach delete handlers
            var delBtns = el.modelList.querySelectorAll(".model-item-delete");
            for (var j = 0; j < delBtns.length; j++) {
                delBtns[j].addEventListener("click", function () {
                    var path = this.getAttribute("data-path");
                    api("POST", "/models/delete", { path: path }, function (err2, data2) {
                        if (!err2 && data2 && data2.success) {
                            showToast("Model deleted", "success");
                            refreshModelList();
                        } else {
                            showAlert("Failed to delete model.");
                        }
                    });
                });
            }
        }, 30000);
    }

    // ================================================================
    // GPU Recommendation
    // ================================================================
    function initGpuRecommendation() {
        if (!el.getGpuRecBtn) return;
        el.getGpuRecBtn.addEventListener("click", function () {
            el.getGpuRecBtn.textContent = "Checking...";
            el.getGpuRecBtn.disabled = true;
            api("GET", "/system/gpu-recommend", null, function (err, data) {
                el.getGpuRecBtn.textContent = "Get Recommendation";
                el.getGpuRecBtn.disabled = false;
                if (err || !data) { showAlert("Failed to get GPU recommendation."); return; }
                if (el.gpuRecModel) el.gpuRecModel.textContent = data.whisper_model || "N/A";
                if (el.gpuRecQuality) el.gpuRecQuality.textContent = data.caption_quality || "N/A";
                if (el.gpuRecDevice) el.gpuRecDevice.textContent = data.whisper_device || "N/A";
                if (el.gpuRecNotes) {
                    el.gpuRecNotes.textContent = (data.notes || []).join(" ");
                }
                if (el.gpuRecResults) el.gpuRecResults.classList.remove("hidden");
                _lastGpuRec = data;
            });
        });

        if (el.applyGpuRecBtn) el.applyGpuRecBtn.addEventListener("click", function () {
            if (!_lastGpuRec) return;
            // Apply the recommended model to all model selects
            var modelSelects = ["captionModel", "subModel", "fillerModel", "transcriptModel", "settingsDefaultModel"];
            for (var i = 0; i < modelSelects.length; i++) {
                var sel = document.getElementById(modelSelects[i]);
                if (sel) {
                    // Check if the recommended value exists as an option
                    for (var j = 0; j < sel.options.length; j++) {
                        if (sel.options[j].value === _lastGpuRec.whisper_model) {
                            sel.value = _lastGpuRec.whisper_model;
                            if (sel._customDropdown) sel._customDropdown.updateText();
                            break;
                        }
                    }
                }
            }
            saveLocalSettings();
            showToast("GPU recommendations applied", "success");
        });
    }
    var _lastGpuRec = null;

    // ================================================================
    // Job Queue UI
    // ================================================================
    function initQueue() {
        if (!el.clearQueueBtn) return;
        el.clearQueueBtn.addEventListener("click", function () {
            api("POST", "/queue/clear", {}, function (err, data) {
                if (!err && data) {
                    showAlert("Queue cleared: " + (data.removed || 0) + " jobs removed.");
                    refreshQueueStatus();
                }
            });
        });
    }

    function addToQueue(endpoint, payload) {
        api("POST", "/queue/add", { endpoint: endpoint, payload: payload }, function (err, data) {
            if (!err && data) {
                showToast("Added to queue (position " + data.position + ")", "info");
                refreshQueueStatus();
            }
        });
    }

    function refreshQueueStatus() {
        api("GET", "/queue/list", null, function (err, data) {
            if (err || !data) return;
            var count = data.length;
            if (el.jobQueueBar) {
                if (count > 0) {
                    el.jobQueueBar.classList.remove("hidden");
                    if (el.queueStatusText) el.queueStatusText.textContent = "Queue: " + count + " job" + (count !== 1 ? "s" : "");
                } else {
                    el.jobQueueBar.classList.add("hidden");
                }
            }
        });
    }

    // ================================================================
    // Toast Notifications
    // ================================================================
    var MAX_TOASTS = 5;
    function showToast(message, type) {
        // Only show if notifications enabled
        if (el.settingsShowNotifications && !el.settingsShowNotifications.checked) return;
        // Cap concurrent toasts — remove oldest if at limit
        var existing = document.querySelectorAll(".toast-notification");
        if (existing.length >= MAX_TOASTS) {
            for (var ti = 0; ti <= existing.length - MAX_TOASTS; ti++) {
                if (existing[ti].parentNode) existing[ti].parentNode.removeChild(existing[ti]);
            }
        }
        var toast = document.createElement("div");
        toast.className = "toast-notification " + (type || "info");
        toast.textContent = message;
        document.body.appendChild(toast);
        setTimeout(function () {
            toast.classList.add("fade-out");
            setTimeout(function () {
                if (toast.parentNode) toast.parentNode.removeChild(toast);
            }, 300);
        }, 3000);
    }

    // ================================================================
    // Enhanced Drag and Drop
    // ================================================================
    function initEnhancedDragDrop() {
        // Enable drag and drop on the whole panel, not just the drop zone
        var panel = document.querySelector(".app");
        if (!panel) return;

        var dragCounter = 0;

        panel.addEventListener("dragenter", function (e) {
            e.preventDefault();
            dragCounter++;
            if (el.dropZone) el.dropZone.classList.add("drag-active");
        });

        panel.addEventListener("dragleave", function (e) {
            dragCounter--;
            if (dragCounter <= 0) {
                dragCounter = 0;
                if (el.dropZone) el.dropZone.classList.remove("drag-active");
            }
        });

        panel.addEventListener("dragover", function (e) {
            e.preventDefault();
        });

        panel.addEventListener("drop", function (e) {
            e.preventDefault();
            dragCounter = 0;
            if (el.dropZone) el.dropZone.classList.remove("drag-active");

            var files = e.dataTransfer && e.dataTransfer.files;
            if (files && files.length > 0) {
                var file = files[0];
                // CEP environment provides the file path
                if (file.path) {
                    selectFile(file.path, file.name);
                    showToast("File loaded: " + file.name, "success");
                } else {
                    showAlert("File dropped, but path not available in this environment.");
                }
            }
        });
    }

    // ================================================================
    // Transcript Search
    // ================================================================
    var _searchMatches = [];
    var _searchIndex = -1;

    function initTranscriptSearch() {
        if (!el.transcriptSearchInput) return;

        el.transcriptSearchInput.addEventListener("input", function () {
            doTranscriptSearch(this.value.trim());
        });

        if (el.transcriptSearchNext) el.transcriptSearchNext.addEventListener("click", function () {
            if (_searchMatches.length === 0) return;
            _searchIndex = (_searchIndex + 1) % _searchMatches.length;
            highlightSearchMatch();
        });

        if (el.transcriptSearchPrev) el.transcriptSearchPrev.addEventListener("click", function () {
            if (_searchMatches.length === 0) return;
            _searchIndex = (_searchIndex - 1 + _searchMatches.length) % _searchMatches.length;
            highlightSearchMatch();
        });
    }

    function doTranscriptSearch(query) {
        _searchMatches = [];
        _searchIndex = -1;

        // Clear previous highlights
        var segments = document.querySelectorAll(".transcript-seg");
        for (var i = 0; i < segments.length; i++) {
            segments[i].classList.remove("search-highlight", "search-active");
        }

        if (!query) {
            if (el.transcriptSearchCount) el.transcriptSearchCount.textContent = "";
            return;
        }

        var lower = query.toLowerCase();
        for (var i = 0; i < segments.length; i++) {
            var text = segments[i].textContent || "";
            if (text.toLowerCase().indexOf(lower) !== -1) {
                _searchMatches.push(segments[i]);
                segments[i].classList.add("search-highlight");
            }
        }

        if (el.transcriptSearchCount) {
            el.transcriptSearchCount.textContent = _searchMatches.length + " match" + (_searchMatches.length !== 1 ? "es" : "");
        }

        if (_searchMatches.length > 0) {
            _searchIndex = 0;
            highlightSearchMatch();
        }
    }

    function highlightSearchMatch() {
        for (var i = 0; i < _searchMatches.length; i++) {
            _searchMatches[i].classList.remove("search-active");
        }
        if (_searchIndex >= 0 && _searchIndex < _searchMatches.length) {
            _searchMatches[_searchIndex].classList.add("search-active");
            _searchMatches[_searchIndex].scrollIntoView({ block: "nearest", behavior: "smooth" });
            if (el.transcriptSearchCount) {
                el.transcriptSearchCount.textContent = (_searchIndex + 1) + "/" + _searchMatches.length;
            }
        }
    }

    // ================================================================
    // Premiere Pro Theme Sync
    // ================================================================
    var _themeSyncRegistered = false;
    function initPremiereThemeSync() {
        if (!inPremiere || !cs) return;
        // CSInterface provides app skin info
        try {
            var skinInfo = cs.getHostEnvironment();
            if (skinInfo && skinInfo.appSkinInfo) {
                var bgColor = skinInfo.appSkinInfo.panelBackgroundColor;
                if (bgColor && bgColor.color) {
                    var r = bgColor.color.red || 0;
                    var g = bgColor.color.green || 0;
                    var b = bgColor.color.blue || 0;
                    var brightness = (r + g + b) / 3;
                    // If Premiere is using a light theme (brightness > 128),
                    // we could switch, but all our themes are dark, so just note it
                    if (brightness > 160) {
                        // Premiere is in a lighter mode - show a subtle note
                        logger("Premiere using light theme (brightness " + Math.round(brightness) + ")");
                    }
                }
            }
            // Register for theme change events (once only — prevent exponential listener leak)
            if (!_themeSyncRegistered) {
                _themeSyncRegistered = true;
                cs.addEventListener("com.adobe.csxs.events.ThemeColorChanged", function () {
                    initPremiereThemeSync();
                });
            }
        } catch (e) {
            // CSInterface theme API not available
        }
    }

    function logger(msg) {
        if (typeof console !== "undefined" && console.log) console.log("[OpenCut] " + msg);
    }

    // ================================================================
    // Waveform Preview (with per-file cache)
    // ================================================================
    var _waveformData = null;
    var _waveformCache = {}; // keyed by filepath
    var _WAVEFORM_CACHE_MAX = 10;

    function initWaveform() {
        if (!el.loadWaveformBtn) return;
        el.loadWaveformBtn.addEventListener("click", function () {
            if (!selectedPath) return;

            // Check cache first
            if (_waveformCache[selectedPath]) {
                _waveformData = _waveformCache[selectedPath];
                if (el.waveformContainer) el.waveformContainer.classList.remove("hidden");
                drawWaveform(_waveformData.peaks);
                updateThresholdLine();
                return;
            }

            el.loadWaveformBtn.textContent = "Loading...";
            el.loadWaveformBtn.disabled = true;
            var fetchPath = selectedPath; // capture for closure
            api("POST", "/audio/waveform", { file: fetchPath, samples: 500 }, function (err, data) {
                el.loadWaveformBtn.textContent = "Preview Waveform";
                el.loadWaveformBtn.disabled = !selectedPath;
                if (err || !data || !data.peaks) {
                    showToast("Failed to load waveform", "error");
                    return;
                }
                _waveformData = data;
                // Cache with FIFO eviction
                var keys = Object.keys(_waveformCache);
                if (keys.length >= _WAVEFORM_CACHE_MAX) {
                    delete _waveformCache[keys[0]];
                }
                _waveformCache[fetchPath] = data;
                if (el.waveformContainer) el.waveformContainer.classList.remove("hidden");
                drawWaveform(data.peaks);
                updateThresholdLine();
            });
        });
        // Drag threshold line
        if (el.waveformThreshold) {
            var dragging = false;
            el.waveformThreshold.addEventListener("mousedown", function (e) { dragging = true; e.preventDefault(); });
            document.addEventListener("mousemove", function (e) {
                if (!dragging || !el.waveformContainer) return;
                var rect = el.waveformCanvas.getBoundingClientRect();
                var y = Math.max(0, Math.min(e.clientY - rect.top, rect.height));
                var amplitude = 1 - (y / rect.height);
                // Convert to dB: 20 * log10(amplitude), range -60 to 0
                var db = amplitude > 0 ? Math.round(20 * Math.log10(amplitude)) : -60;
                db = Math.max(-60, Math.min(-10, db));
                var thresholdSlider = document.getElementById("threshold");
                if (thresholdSlider) {
                    thresholdSlider.value = db;
                    var valSpan = document.getElementById("thresholdVal");
                    if (valSpan) valSpan.textContent = db + " dB";
                }
                updateThresholdLine();
            });
            document.addEventListener("mouseup", function () { dragging = false; });
        }
    }

    function drawWaveform(peaks) {
        var canvas = el.waveformCanvas;
        if (!canvas) return;
        var ctx = canvas.getContext("2d");
        var w = canvas.width;
        var h = canvas.height;
        ctx.clearRect(0, 0, w, h);
        // Background
        ctx.fillStyle = "rgba(0, 0, 0, 0.3)";
        ctx.fillRect(0, 0, w, h);
        // Draw bars
        var barW = w / peaks.length;
        for (var i = 0; i < peaks.length; i++) {
            var val = peaks[i];
            var barH = val * h;
            // Color based on amplitude
            var hue = val > 0.5 ? 0 : val > 0.2 ? 60 : 180;
            ctx.fillStyle = "hsla(" + hue + ", 80%, 60%, 0.8)";
            ctx.fillRect(i * barW, h - barH, Math.max(1, barW - 0.5), barH);
        }
    }

    function updateThresholdLine() {
        if (!el.waveformThreshold || !el.waveformCanvas) return;
        var thresholdSlider = document.getElementById("threshold");
        var db = thresholdSlider ? parseInt(thresholdSlider.value) : -30;
        // Convert dB to amplitude: 10^(dB/20)
        var amplitude = Math.pow(10, db / 20);
        var h = el.waveformCanvas.height;
        var y = h - (amplitude * h);
        el.waveformThreshold.style.top = y + "px";
    }

    // ================================================================
    // Favorites / Pinned Operations
    // ================================================================
    var _favorites = [];
    var _favoriteOps = {
        "silence": { label: "Remove Silences", tab: "cut", sub: "silence", btn: "runSilenceBtn" },
        "fillers": { label: "Clean Fillers", tab: "cut", sub: "fillers", btn: "runFillersBtn" },
        "styled_captions": { label: "Styled Captions", tab: "captions", sub: "cap-styled", btn: "runStyledCaptionsBtn" },
        "transcribe": { label: "Transcribe", tab: "captions", sub: "cap-transcript", btn: "runTranscriptBtn" },
        "denoise": { label: "Denoise Audio", tab: "audio", sub: "aud-denoise", btn: "runDenoiseBtn" },
        "normalize": { label: "Normalize", tab: "audio", sub: "aud-normalize", btn: "runNormalizeBtn" },
        "separate": { label: "Stem Separate", tab: "audio", sub: "aud-separate", btn: "runSeparateBtn" },
        "stabilize": { label: "Stabilize Video", tab: "video", sub: "vid-effects", btn: "runVfxBtn" },
        "face_blur": { label: "Face Blur", tab: "video", sub: "vid-face", btn: "runFaceBlurBtn" },
        "export": { label: "Export Preset", tab: "export", sub: "exp-platform", btn: "runExportPresetBtn" },
    };

    function initFavorites() {
        // Load from backend
        api("GET", "/favorites", null, function (err, data) {
            if (!err && data && Array.isArray(data)) _favorites = data;
            renderFavorites();
        });
    }

    var _favDelegationAdded = false;
    function renderFavorites() {
        if (!el.favoritesItems || !el.favoritesBar) return;
        if (_favorites.length === 0) {
            el.favoritesItems.innerHTML = "";
            el.favoritesBar.classList.add("hidden");
            return;
        }
        el.favoritesBar.classList.remove("hidden");
        // Event delegation for favorite chips (F2 pattern)
        if (!_favDelegationAdded) {
            _favDelegationAdded = true;
            el.favoritesItems.addEventListener("click", function (e) {
                var chip = e.target.closest(".fav-chip");
                if (!chip) return;
                var favId = chip.dataset.fav;
                // Remove button clicked
                if (e.target.closest(".fav-chip-remove")) {
                    e.stopPropagation();
                    _favorites = _favorites.filter(function (f) { return f !== favId; });
                    saveFavorites();
                    renderFavorites();
                    showToast("Removed from favorites", "info");
                    return;
                }
                // Navigate on label click
                var op = _favoriteOps[favId];
                if (op) navigateToTab(op.tab, op.sub);
            });
        }
        var frag = document.createDocumentFragment();
        for (var i = 0; i < _favorites.length; i++) {
            var favId = _favorites[i];
            var op = _favoriteOps[favId];
            if (!op) continue;
            var chip = document.createElement("div");
            chip.className = "fav-chip";
            chip.dataset.fav = favId;
            chip.innerHTML = '<span>' + esc(op.label) + '</span><span class="fav-chip-remove">&times;</span>';
            frag.appendChild(chip);
        }
        el.favoritesItems.innerHTML = "";
        el.favoritesItems.appendChild(frag);
    }

    function navigateToTab(tab, sub) {
        // Click the main tab
        var navBtn = document.querySelector('.nav-tab[data-nav="' + tab + '"]');
        if (navBtn) navBtn.click();
        // Then click sub-tab
        if (sub) {
            setTimeout(function () {
                var subBtn = document.querySelector('.sub-tab[data-sub="' + sub + '"]');
                if (subBtn) subBtn.click();
            }, 50);
        }
    }

    function addFavorite(favId) {
        if (_favorites.indexOf(favId) !== -1) return;
        _favorites.push(favId);
        saveFavorites();
        renderFavorites();
        showToast("Added to favorites: " + (_favoriteOps[favId] || {}).label, "success");
    }

    function saveFavorites() {
        api("POST", "/favorites/save", { favorites: _favorites }, function () {});
    }

    // ================================================================
    // Side-by-Side Preview
    // ================================================================
    function initPreviewModal() {
        if (el.previewModalClose) {
            el.previewModalClose.addEventListener("click", function () {
                if (el.previewModal) el.previewModal.classList.add("hidden");
            });
        }
        if (el.previewRefreshBtn) {
            el.previewRefreshBtn.addEventListener("click", function () {
                loadPreviewFrame();
            });
        }
        if (el.previewVfxBtn) {
            el.previewVfxBtn.addEventListener("click", function () {
                if (!selectedPath) return;
                loadPreviewFrame();
            });
        }
    }

    function loadPreviewFrame() {
        if (!selectedPath) return;
        var ts = el.previewTimestamp ? el.previewTimestamp.value : "00:00:01";
        // Load original frame
        api("POST", "/video/preview-frame", { file: selectedPath, timestamp: ts }, function (err, data) {
            if (err || !data || !data.image) return;
            if (el.previewOriginal) el.previewOriginal.src = "data:image/jpeg;base64," + data.image;
            if (el.previewProcessed) el.previewProcessed.src = "data:image/jpeg;base64," + data.image;
            if (el.previewModal) el.previewModal.classList.remove("hidden");
        });
    }

    // ================================================================
    // Audio Preview Player
    // ================================================================
    function initAudioPreview() {
        if (el.audioPreviewClose) {
            el.audioPreviewClose.addEventListener("click", function () {
                if (el.audioPreview) el.audioPreview.classList.add("hidden");
                if (el.audioPreviewPlayer) {
                    el.audioPreviewPlayer.pause();
                    el.audioPreviewPlayer.src = "";
                }
            });
        }
    }

    function showAudioPreview(filePath) {
        if (!el.audioPreview || !el.audioPreviewPlayer) return;
        el.audioPreviewPlayer.src = BACKEND + "/file?path=" + encodeURIComponent(filePath);
        el.audioPreview.classList.remove("hidden");
        try { el.audioPreviewPlayer.play().catch(function() {}); } catch (e) {}
    }

    // ================================================================
    // Right-Click Context Menu
    // ================================================================
    function initContextMenu() {
        if (!el.contextMenu) return;
        // Show on clip select right-click
        var clipSelect = document.querySelector(".clip-select");
        if (clipSelect) {
            clipSelect.addEventListener("contextmenu", function (e) {
                if (!selectedPath) return;
                e.preventDefault();
                el.contextMenu.classList.remove("hidden");
                var menuW = el.contextMenu.offsetWidth || 160;
                var menuH = el.contextMenu.offsetHeight || 200;
                var left = Math.min(e.clientX, window.innerWidth - menuW - 4);
                var top = Math.min(e.clientY, window.innerHeight - menuH - 4);
                el.contextMenu.style.left = Math.max(0, left) + "px";
                el.contextMenu.style.top = Math.max(0, top) + "px";
            });
        }
        // Handle menu item clicks
        var items = el.contextMenu.querySelectorAll(".context-menu-item");
        for (var i = 0; i < items.length; i++) {
            items[i].addEventListener("click", function () {
                var action = this.dataset.action;
                el.contextMenu.classList.add("hidden");
                if (action === "favorite") {
                    // Determine current active operation
                    var activeTab = document.querySelector(".nav-tab.active");
                    var activeSub = document.querySelector(".sub-tab.active");
                    if (activeTab && activeSub) {
                        var favId = activeSub.dataset.sub;
                        // Map sub-tab to favorite ID
                        var subToFav = { "silence": "silence", "fillers": "fillers", "cap-styled": "styled_captions", "cap-transcript": "transcribe", "aud-denoise": "denoise", "aud-normalize": "normalize", "aud-separate": "separate", "vid-effects": "stabilize", "vid-face": "face_blur", "exp-platform": "export" };
                        if (subToFav[favId]) addFavorite(subToFav[favId]);
                    }
                } else {
                    var actionToNav = {
                        "silence": ["cut", "silence"],
                        "transcribe": ["captions", "cap-transcript"],
                        "denoise": ["audio", "aud-denoise"],
                        "normalize": ["audio", "aud-normalize"],
                        "stabilize": ["video", "vid-effects"],
                        "export": ["export", "exp-platform"]
                    };
                    var nav = actionToNav[action];
                    if (nav) navigateToTab(nav[0], nav[1]);
                }
            });
        }
        // Hide on click outside
        document.addEventListener("click", function (e) {
            if (!e.target.closest(".context-menu")) {
                el.contextMenu.classList.add("hidden");
            }
        });
    }

    // ================================================================
    // First-Run Wizard
    // ================================================================
    function initWizard() {
        if (!el.wizardOverlay) return;
        // Check if user has dismissed the wizard before
        try {
            var settings = JSON.parse(localStorage.getItem("opencut_settings") || "{}");
            if (settings.wizardDismissed) return;
        } catch (e) {}
        // Show wizard
        el.wizardOverlay.classList.remove("hidden");
        // Animate steps
        var steps = el.wizardOverlay.querySelectorAll(".wizard-step");
        for (var i = 1; i < steps.length; i++) {
            (function (idx) {
                setTimeout(function () { steps[idx].classList.add("active"); }, idx * 400);
            })(i);
        }
        if (el.wizardCloseBtn) {
            el.wizardCloseBtn.addEventListener("click", function () {
                el.wizardOverlay.classList.add("hidden");
                // Always dismiss permanently — wizard is one-time onboarding
                try {
                    var s = JSON.parse(localStorage.getItem("opencut_settings") || "{}");
                    s.wizardDismissed = true;
                    localStorage.setItem("opencut_settings", JSON.stringify(s));
                } catch (e) {}
            });
        }
    }

    // ================================================================
    // Output Browser
    // ================================================================
    var _outputBrowserOpen = false;

    function initOutputBrowser() {
        if (el.outputBrowserToggle) {
            el.outputBrowserToggle.addEventListener("click", function () {
                _outputBrowserOpen = !_outputBrowserOpen;
                if (el.outputBrowser) {
                    el.outputBrowser.classList.toggle("hidden", !_outputBrowserOpen);
                }
                if (_outputBrowserOpen) refreshOutputs();
            });
        }
        if (el.outputBrowserClose) {
            el.outputBrowserClose.addEventListener("click", function () {
                _outputBrowserOpen = false;
                if (el.outputBrowser) el.outputBrowser.classList.add("hidden");
            });
        }
        if (el.refreshOutputsBtn) {
            el.refreshOutputsBtn.addEventListener("click", refreshOutputs);
        }
    }

    function refreshOutputs() {
        api("GET", "/outputs/recent", null, function (err, data) {
            if (err || !data || !Array.isArray(data)) return;
            if (el.outputBrowserToggle) {
                el.outputBrowserToggle.textContent = "Outputs (" + data.length + ")";
            }
            if (!el.outputBrowserList) return;
            el.outputBrowserList.innerHTML = "";
            if (data.length === 0) {
                el.outputBrowserList.innerHTML = '<div class="hint" style="padding: 8px 12px;">No recent outputs.</div>';
                return;
            }
            for (var i = 0; i < data.length; i++) {
                var item = data[i];
                var div = document.createElement("div");
                div.className = "output-item";
                div.innerHTML = '<div class="output-item-info"><div class="output-item-name">' + esc(item.name) + '</div><div class="output-item-meta">' + safeFixed(item.size_mb, 1) + ' MB &mdash; ' + esc(item.type || "") + '</div></div><div class="output-item-actions"><button type="button" class="btn-sm" data-path="' + esc(item.path) + '">Import</button></div>';
                div.querySelector(".btn-sm").addEventListener("click", function () {
                    var p = this.dataset.path;
                    if (inPremiere && cs) {
                        PremiereBridge.autoImport(p, "output");
                        showToast("Imported: " + p.split(/[/\\]/).pop(), "success");
                    }
                });
                el.outputBrowserList.appendChild(div);
            }
        });
    }

    // ================================================================
    // Batch Multi-Select File Picker
    // ================================================================
    var _batchFiles = [];

    var _batchDelegationAdded = false;
    function initBatchPicker() {
        if (el.batchAddSelectedBtn) {
            el.batchAddSelectedBtn.addEventListener("click", function () {
                if (!selectedPath) { showToast("No clip selected", "error"); return; }
                if (_batchFiles.indexOf(selectedPath) !== -1) return;
                _batchFiles.push(selectedPath);
                renderBatchFiles();
            });
        }
        if (el.batchAddAllBtn) {
            el.batchAddAllBtn.addEventListener("click", function () {
                for (var i = 0; i < projectMedia.length; i++) {
                    var p = projectMedia[i].path || projectMedia[i];
                    if (_batchFiles.indexOf(p) === -1) _batchFiles.push(p);
                }
                renderBatchFiles();
            });
        }
        if (el.batchClearBtn) {
            el.batchClearBtn.addEventListener("click", function () {
                _batchFiles = [];
                renderBatchFiles();
            });
        }
        // Event delegation for batch file remove buttons (F2)
        if (el.batchFileList && !_batchDelegationAdded) {
            _batchDelegationAdded = true;
            el.batchFileList.addEventListener("click", function (e) {
                var removeBtn = e.target.closest(".batch-file-remove");
                if (removeBtn) {
                    var idx = parseInt(removeBtn.getAttribute("data-idx"), 10);
                    _batchFiles.splice(idx, 1);
                    renderBatchFiles();
                }
            });
        }
    }

    function renderBatchFiles() {
        if (!el.batchFileList) return;
        if (_batchFiles.length === 0) {
            el.batchFileList.innerHTML = '<div class="hint">No files added. Use "Add Selected" or drag files.</div>';
            return;
        }
        var frag = document.createDocumentFragment();
        for (var i = 0; i < _batchFiles.length; i++) {
            var item = document.createElement("div");
            item.className = "batch-file-item";
            var name = _batchFiles[i].split(/[/\\]/).pop();
            item.innerHTML = '<span>' + (i + 1) + '. ' + esc(name) + '</span><button type="button" class="batch-file-remove" data-idx="' + i + '">&times;</button>';
            frag.appendChild(item);
        }
        el.batchFileList.innerHTML = "";
        el.batchFileList.appendChild(frag);
    }

    // ================================================================
    // Dependency Health Dashboard
    // ================================================================
    function initDepDashboard() {
        if (el.refreshDepsBtn) {
            el.refreshDepsBtn.addEventListener("click", refreshDeps);
        }
    }

    function refreshDeps() {
        if (!el.depGrid) return;
        el.depGrid.innerHTML = '<div class="hint">Checking dependencies...</div>';
        api("GET", "/system/dependencies", null, function (err, data) {
            if (err || !data) {
                el.depGrid.innerHTML = '<div class="hint">Failed to check dependencies.</div>';
                return;
            }
            var keys = Object.keys(data);
            var frag = document.createDocumentFragment();
            for (var i = 0; i < keys.length; i++) {
                var name = keys[i];
                var info = data[name];
                var div = document.createElement("div");
                div.className = "dep-item";
                div.innerHTML = '<span class="dep-dot ' + (info.installed ? "installed" : "missing") + '"></span>' +
                    '<span class="dep-name">' + esc(name) + '</span>' +
                    '<span class="dep-version">' + esc(info.installed ? (info.version || "OK").toString().substring(0, 12) : "missing") + '</span>';
                frag.appendChild(div);
            }
            el.depGrid.innerHTML = "";
            el.depGrid.appendChild(frag);
        });
    }

    // ================================================================
    // Settings Import / Export
    // ================================================================
    function initSettingsIO() {
        if (el.exportSettingsBtn) {
            el.exportSettingsBtn.addEventListener("click", function () {
                api("GET", "/settings/export", null, function (err, data) {
                    if (err || !data) { showToast("Export failed", "error"); return; }
                    // Also include localStorage settings
                    try { data.localStorage = JSON.parse(localStorage.getItem("opencut_settings") || "{}"); } catch (e) {}
                    var blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
                    var url = URL.createObjectURL(blob);
                    var a = document.createElement("a");
                    a.href = url;
                    a.download = "opencut_settings_" + new Date().toISOString().slice(0, 10) + ".json";
                    a.click();
                    // Defer revocation so browser has time to start the download
                    setTimeout(function () { URL.revokeObjectURL(url); }, 5000);
                    showToast("Settings exported", "success");
                });
            });
        }
        if (el.importSettingsBtn && el.importSettingsFile) {
            el.importSettingsBtn.addEventListener("click", function () {
                el.importSettingsFile.click();
            });
            el.importSettingsFile.addEventListener("change", function () {
                var file = this.files[0];
                if (!file) return;
                var reader = new FileReader();
                reader.onload = function (e) {
                    try {
                        var data = JSON.parse(e.target.result);
                        api("POST", "/settings/import", data, function (err, result) {
                            if (err) { showToast("Import failed", "error"); return; }
                            if (data.localStorage) {
                                localStorage.setItem("opencut_settings", JSON.stringify(data.localStorage));
                                loadLocalSettings();
                            }
                            showToast("Settings imported: " + (result.imported || []).join(", "), "success");
                            if (typeof initPresets === "function") initPresets();
                        });
                    } catch (ex) {
                        showToast("Invalid settings file", "error");
                    }
                };
                reader.readAsText(file);
                this.value = "";
            });
        }

        // Log export / clear
        var exportLogsBtn = document.getElementById("exportLogsBtn");
        var clearLogsBtn = document.getElementById("clearLogsBtn");
        if (exportLogsBtn) {
            exportLogsBtn.addEventListener("click", function () {
                var a = document.createElement("a");
                a.href = BACKEND + "/logs/export";
                a.download = "opencut_crash.log";
                a.click();
            });
        }
        if (clearLogsBtn) {
            clearLogsBtn.addEventListener("click", function () {
                api("POST", "/logs/clear", {}, function (err, data) {
                    if (err) { showToast("Failed to clear logs", "error"); return; }
                    showToast("Crash log cleared", "success");
                });
            });
        }
    }

    // ================================================================
    // Custom Workflow Builder
    // ================================================================
    var _workflowSteps = [];
    var _workflowDelegationAdded = false;

    function initWorkflowBuilder() {
        if (el.workflowAddStepBtn) {
            el.workflowAddStepBtn.addEventListener("click", function () {
                var sel = el.workflowStepSelect;
                if (!sel) return;
                var selOpt = sel.selectedIndex >= 0 ? sel.options[sel.selectedIndex] : null;
                _workflowSteps.push({
                    endpoint: sel.value,
                    label: selOpt ? selOpt.textContent : sel.value
                });
                renderWorkflowSteps();
            });
        }
        // Event delegation for workflow step remove buttons (F2)
        if (el.workflowStepList && !_workflowDelegationAdded) {
            _workflowDelegationAdded = true;
            el.workflowStepList.addEventListener("click", function (e) {
                var removeBtn = e.target.closest(".workflow-step-remove");
                if (removeBtn) {
                    var idx = parseInt(removeBtn.getAttribute("data-idx"), 10);
                    _workflowSteps.splice(idx, 1);
                    renderWorkflowSteps();
                }
            });
        }
        if (el.saveCustomWorkflowBtn) {
            el.saveCustomWorkflowBtn.addEventListener("click", function () {
                var name = el.customWorkflowName ? el.customWorkflowName.value.trim() : "";
                if (!name) { showToast("Enter a workflow name", "error"); return; }
                if (_workflowSteps.length === 0) { showToast("Add at least one step", "error"); return; }
                api("POST", "/workflows/save", { name: name, steps: _workflowSteps }, function (err) {
                    if (err) { showToast("Save failed", "error"); return; }
                    showToast("Workflow saved: " + name, "success");
                    refreshSavedWorkflows();
                });
            });
        }
        if (el.loadCustomWorkflowBtn) {
            el.loadCustomWorkflowBtn.addEventListener("click", function () {
                var sel = el.savedWorkflowSelect;
                if (!sel || !sel.value) return;
                api("GET", "/workflows/list", null, function (err, data) {
                    if (err || !data) return;
                    for (var i = 0; i < data.length; i++) {
                        if (data[i].name === sel.value) {
                            _workflowSteps = data[i].steps || [];
                            if (el.customWorkflowName) el.customWorkflowName.value = data[i].name;
                            renderWorkflowSteps();
                            break;
                        }
                    }
                });
            });
        }
        if (el.deleteCustomWorkflowBtn) {
            el.deleteCustomWorkflowBtn.addEventListener("click", function () {
                var sel = el.savedWorkflowSelect;
                if (!sel || !sel.value) return;
                api("POST", "/workflows/delete", { name: sel.value }, function (err) {
                    if (!err) {
                        showToast("Workflow deleted", "success");
                        refreshSavedWorkflows();
                    }
                });
            });
        }
        if (el.runCustomWorkflowBtn) {
            el.runCustomWorkflowBtn.addEventListener("click", function () {
                if (_workflowSteps.length === 0 || !selectedPath) return;
                // Queue each step
                for (var i = 0; i < _workflowSteps.length; i++) {
                    addToQueue(_workflowSteps[i].endpoint, { filepath: selectedPath, output_dir: projectFolder });
                }
                showToast("Queued " + _workflowSteps.length + " workflow steps", "success");
            });
        }
        refreshSavedWorkflows();
    }

    function renderWorkflowSteps() {
        if (!el.workflowStepList) return;
        if (_workflowSteps.length === 0) {
            el.workflowStepList.innerHTML = '<div class="hint">Add steps to build a custom workflow.</div>';
            if (el.runCustomWorkflowBtn) el.runCustomWorkflowBtn.disabled = true;
            return;
        }
        if (el.runCustomWorkflowBtn) el.runCustomWorkflowBtn.disabled = false;
        var frag = document.createDocumentFragment();
        for (var i = 0; i < _workflowSteps.length; i++) {
            var item = document.createElement("div");
            item.className = "workflow-step-item";
            item.innerHTML = '<span class="workflow-step-num">' + (i + 1) + '</span><span>' + esc(_workflowSteps[i].label) + '</span><button type="button" class="workflow-step-remove" data-idx="' + i + '">&times;</button>';
            frag.appendChild(item);
        }
        el.workflowStepList.innerHTML = "";
        el.workflowStepList.appendChild(frag);
    }

    function refreshSavedWorkflows() {
        api("GET", "/workflows/list", null, function (err, data) {
            if (err || !data || !el.savedWorkflowSelect) return;
            el.savedWorkflowSelect.innerHTML = "";
            if (data.length === 0) {
                el.savedWorkflowSelect.innerHTML = '<option value="" disabled selected>No custom workflows</option>';
                return;
            }
            for (var i = 0; i < data.length; i++) {
                var opt = document.createElement("option");
                opt.value = data[i].name;
                opt.textContent = data[i].name + " (" + (data[i].steps || []).length + " steps)";
                el.savedWorkflowSelect.appendChild(opt);
            }
        });
    }

    // ================================================================
    // Collapsible Cards
    // ================================================================
    function initCollapsibleCards() {
        var headers = document.querySelectorAll("[data-collapsible]");
        for (var i = 0; i < headers.length; i++) {
            headers[i].addEventListener("click", function () {
                this.classList.toggle("collapsed");
                // Find the next sibling content (everything after header in the card)
                var card = this.closest(".card");
                if (!card) return;
                var children = card.children;
                var afterHeader = false;
                for (var j = 0; j < children.length; j++) {
                    if (children[j] === this) { afterHeader = true; continue; }
                    if (afterHeader) {
                        children[j].style.display = this.classList.contains("collapsed") ? "none" : "";
                    }
                }
            });
        }
    }

    // ================================================================
    // Job Time Estimates
    // ================================================================
    function fetchTimeEstimate(jobType) {
        if (!el.processingEstimate) return;
        // Get file duration from file info (fmtDur outputs M:SS or H:MM:SS)
        var fileDuration = 0;
        var metaEl = document.getElementById("fileMetaDisplay");
        if (metaEl) {
            var txt = metaEl.textContent || "";
            // Match M:SS or H:MM:SS format from fmtDur()
            var hmatch = txt.match(/(\d+):(\d+):(\d+)/);
            var mmatch = !hmatch && txt.match(/(\d+):(\d+)/);
            if (hmatch) fileDuration = parseInt(hmatch[1]) * 3600 + parseInt(hmatch[2]) * 60 + parseInt(hmatch[3]);
            else if (mmatch) fileDuration = parseInt(mmatch[1]) * 60 + parseInt(mmatch[2]);
        }
        api("POST", "/system/estimate-time", { type: jobType, file_duration: fileDuration }, function (err, data) {
            if (err || !data || !data.estimate_seconds) {
                el.processingEstimate.textContent = "";
                return;
            }
            var secs = Math.round(data.estimate_seconds);
            if (secs > 60) {
                el.processingEstimate.textContent = Math.floor(secs / 60) + "m " + (secs % 60) + "s est.";
            } else {
                el.processingEstimate.textContent = secs + "s est.";
            }
        });
    }

    // ================================================================
    // i18n / Localization (placeholder framework)
    // ================================================================
    var _currentLang = "en";

    function initI18n() {
        if (el.settingsLang) {
            el.settingsLang.addEventListener("change", function () {
                _currentLang = this.value;
                saveLocalSettings();
                if (_currentLang !== "en") {
                    showToast("Language support coming soon. UI will remain in English for now.", "info");
                }
            });
        }
    }

    // ================================================================
    // Enhanced Job History (with re-run and details)
    // ================================================================

    // ================================================================
    // v1.3.0 - Clip Preview Thumbnail
    // ================================================================
    function updateClipPreview() {
        if (!el.clipPreviewRow) return;
        if (!selectedPath) {
            el.clipPreviewRow.classList.add("hidden");
            return;
        }
        el.clipPreviewRow.classList.remove("hidden");
        if (el.clipThumb) el.clipThumb.innerHTML = '<div class="clip-thumb-loading"></div>';
        if (el.clipMetaRes) el.clipMetaRes.textContent = "";
        if (el.clipMetaDur) el.clipMetaDur.textContent = "";
        if (el.clipMetaSize) el.clipMetaSize.textContent = "";
        // Fetch thumbnail
        api("POST", "/video/preview-frame", { file: selectedPath, timestamp: "00:00:01", width: 160 }, function(err, data) {
            if (err || !data || !data.image) {
                if (el.clipThumb) el.clipThumb.innerHTML = '<div class="clip-thumb-none">No Preview</div>';
                return;
            }
            if (el.clipThumb) {
                var img = document.createElement("img");
                img.src = "data:image/jpeg;base64," + data.image;
                img.alt = "preview";
                el.clipThumb.innerHTML = "";
                el.clipThumb.appendChild(img);
            }
        });
        // Fetch metadata via lightweight probe (reuses /info endpoint)
        api("POST", "/info", { filepath: selectedPath }, function(err, data) {
            if (!err && data && data.duration) {
                if (el.clipMetaDur) el.clipMetaDur.textContent = fmtDur(data.duration);
            }
            if (!err && data && data.video) {
                if (el.clipMetaRes) el.clipMetaRes.textContent = data.video.width + "x" + data.video.height;
            }
            if (!err && data && data.file_size_mb) {
                if (el.clipMetaSize) el.clipMetaSize.textContent = safeFixed(data.file_size_mb, 1) + " MB";
            }
        });
    }

    // ================================================================
    // v1.3.0 - Recent Clips Dropdown
    // ================================================================
    var _recentClips = [];
    var MAX_RECENT = 10;

    function loadRecentClips() {
        try {
            _recentClips = JSON.parse(localStorage.getItem("opencut_recent_clips") || "[]");
        } catch(e) { _recentClips = []; }
    }

    function saveRecentClips() {
        try { localStorage.setItem("opencut_recent_clips", JSON.stringify(_recentClips)); } catch(e) {}
    }

    function addRecentClip(path) {
        if (!path) return;
        var idx = _recentClips.indexOf(path);
        if (idx !== -1) _recentClips.splice(idx, 1);
        _recentClips.unshift(path);
        if (_recentClips.length > MAX_RECENT) _recentClips = _recentClips.slice(0, MAX_RECENT);
        saveRecentClips();
    }

    function showRecentClips() {
        if (!el.recentClipsDropdown) return;
        loadRecentClips();
        if (_recentClips.length === 0) {
            el.recentClipsDropdown.innerHTML = '<div class="hint" style="padding:8px 12px;">No recent clips.</div>';
        } else {
            var html = "";
            for (var i = 0; i < _recentClips.length; i++) {
                var name = _recentClips[i].split(/[/\\]/).pop();
                html += '<div class="recent-clip-item" data-path="' + esc(_recentClips[i]) + '">' + esc(name) + '</div>';
            }
            el.recentClipsDropdown.innerHTML = html;
        }
        // Explicit show (not toggle) avoids race with outside-click dismiss handler
        el.recentClipsDropdown.classList.remove("hidden");
    }

    // ================================================================
    // v1.3.0 - Command Palette
    // ================================================================
    var _commandIndex = [
        {name: "Silence Removal", tab: "cut", sub: "silence", keywords: "silence remove cut clean"},
        {name: "Filler Words", tab: "cut", sub: "fillers", keywords: "filler um uh like words"},
        {name: "Trim Clip", tab: "cut", sub: "trim", keywords: "trim cut crop in out point"},
        {name: "Styled Captions", tab: "captions", sub: "cap-styled", keywords: "caption subtitle style burn"},
        {name: "Transcribe", tab: "captions", sub: "cap-transcript", keywords: "transcribe whisper speech text"},
        {name: "Translate", tab: "captions", sub: "cap-translate", keywords: "translate language"},
        {name: "Stem Separation", tab: "audio", sub: "aud-separate", keywords: "separate stems vocals drums bass demucs"},
        {name: "Denoise", tab: "audio", sub: "aud-denoise", keywords: "denoise noise reduce clean"},
        {name: "Normalize", tab: "audio", sub: "aud-normalize", keywords: "normalize loudness lufs volume"},
        {name: "Text to Speech", tab: "audio", sub: "aud-tts", keywords: "tts voice speech generate"},
        {name: "Music AI", tab: "audio", sub: "aud-musicai", keywords: "music generate ai musicgen"},
        {name: "Sound Effects", tab: "audio", sub: "aud-sfx", keywords: "sfx sound effect tone"},
        {name: "Audio Duck", tab: "audio", sub: "aud-duck", keywords: "duck ducking lower music dialogue"},
        {name: "Video Effects", tab: "video", sub: "vid-effects", keywords: "stabilize vignette grain letterbox"},
        {name: "Reframe", tab: "video", sub: "vid-reframe", keywords: "reframe resize phone tiktok shorts vertical portrait"},
        {name: "Merge Clips", tab: "video", sub: "vid-merge", keywords: "merge concatenate join combine clips"},
        {name: "Speed / Ramp", tab: "video", sub: "vid-speed", keywords: "speed slow fast ramp reverse"},
        {name: "Chroma Key", tab: "video", sub: "vid-chroma", keywords: "chroma green screen key"},
        {name: "Transitions", tab: "video", sub: "vid-transition", keywords: "transition fade wipe slide"},
        {name: "Upscale", tab: "video", sub: "vid-upscale", keywords: "upscale enhance resolution ai"},
        {name: "Color Correction", tab: "video", sub: "vid-color", keywords: "color correct grade exposure contrast"},
        {name: "LUTs", tab: "video", sub: "vid-lut", keywords: "lut color grade cinematic film look"},
        {name: "Face AI", tab: "video", sub: "vid-faceswap", keywords: "face swap enhance gfpgan"},
        {name: "Remove Object", tab: "video", sub: "vid-remove", keywords: "remove watermark object logo"},
        {name: "Titles", tab: "video", sub: "vid-titles", keywords: "title text overlay lower third"},
        {name: "Export Presets", tab: "export", sub: "exp-platform", keywords: "export platform youtube tiktok instagram"},
        {name: "Thumbnails", tab: "export", sub: "exp-thumbnail", keywords: "thumbnail extract frame"},
        {name: "Batch Processing", tab: "export", sub: "exp-batch", keywords: "batch process multiple files"},
        { name: "Repeat Detection",   tab: "captions", sub: "cap-repeat",     keywords: "repeat detect loop fumble duplicate take" },
        { name: "Chapter Generation", tab: "captions", sub: "cap-chapters",   keywords: "chapters youtube timestamps sections topics" },
        { name: "Footage Search",     tab: "nlp",      sub: "nlp-search",     keywords: "search footage clips index content find" },
        { name: "Color Match",        tab: "timeline", sub: "tl-colormatch",  keywords: "color match grade balance reference clip" },
        { name: "Multicam Switcher",  tab: "timeline", sub: "tl-multicam",    keywords: "multicam speaker podcast camera switch diarize" },
        { name: "Loudness Match",     tab: "audio",    sub: "aud-loudmatch",  keywords: "loudness lufs normalize match audio levels" },
        { name: "Auto Zoom",          tab: "timeline", sub: "tl-autozoom",    keywords: "auto zoom push in ken burns face zoom" },
        { name: "AI Command",         tab: "nlp",      sub: "nlp-command",    keywords: "nlp ai command natural language instruction" },
        { name: "Deliverables",       tab: "export",   sub: "exp-deliverables", keywords: "deliverables vfx adr music cue sheet asset list" },
    ];

    var _paletteSelectedIdx = 0;
    var _paletteResults = [];

    function openCommandPalette() {
        if (!el.commandPaletteOverlay || !el.commandPaletteInput || !el.commandPaletteResults) return;
        el.commandPaletteOverlay.classList.remove("hidden");
        el.commandPaletteInput.value = "";
        renderPaletteResults("");
        setTimeout(function() { if (el.commandPaletteInput) el.commandPaletteInput.focus(); }, 50);
    }

    function closeCommandPalette() {
        if (el.commandPaletteOverlay) el.commandPaletteOverlay.classList.add("hidden");
    }

    function renderPaletteResults(query) {
        _paletteResults = [];
        var q = query.toLowerCase().trim();
        for (var i = 0; i < _commandIndex.length; i++) {
            var item = _commandIndex[i];
            if (!q || item.name.toLowerCase().indexOf(q) !== -1 || item.keywords.indexOf(q) !== -1) {
                _paletteResults.push(item);
            }
        }
        _paletteSelectedIdx = 0;
        var html = "";
        for (var j = 0; j < _paletteResults.length; j++) {
            html += '<div class="command-palette-item' + (j === 0 ? ' selected' : '') + '" data-idx="' + j + '" data-tab="' + _paletteResults[j].tab + '" data-sub="' + _paletteResults[j].sub + '">' + esc(_paletteResults[j].name) + ' <span class="command-palette-tab">' + esc(_paletteResults[j].tab) + '</span></div>';
        }
        if (_paletteResults.length === 0) {
            html = '<div class="command-palette-empty">No matching operations</div>';
        }
        if (el.commandPaletteResults) el.commandPaletteResults.innerHTML = html;
    }

    function executePaletteItem(tab, sub) {
        closeCommandPalette();
        navigateToTab(tab, sub);
    }

    function paletteNavigate(dir) {
        if (!_paletteResults.length || !el.commandPaletteResults) return;
        var items = el.commandPaletteResults.querySelectorAll(".command-palette-item");
        if (items[_paletteSelectedIdx]) items[_paletteSelectedIdx].classList.remove("selected");
        _paletteSelectedIdx += dir;
        if (_paletteSelectedIdx < 0) _paletteSelectedIdx = _paletteResults.length - 1;
        if (_paletteSelectedIdx >= _paletteResults.length) _paletteSelectedIdx = 0;
        if (items[_paletteSelectedIdx]) {
            items[_paletteSelectedIdx].classList.add("selected");
            items[_paletteSelectedIdx].scrollIntoView({ block: "nearest" });
        }
    }

    function paletteExecuteSelected() {
        if (_paletteResults.length > 0 && _paletteResults[_paletteSelectedIdx]) {
            var item = _paletteResults[_paletteSelectedIdx];
            executePaletteItem(item.tab, item.sub);
        }
    }

    function initCommandPalette() {
        if (!el.commandPaletteOverlay || !el.commandPaletteInput || !el.commandPaletteResults) return;

        el.commandPaletteInput.addEventListener("input", function() {
            renderPaletteResults(this.value);
        });

        el.commandPaletteInput.addEventListener("keydown", function(e) {
            if (e.key === "ArrowDown") { e.preventDefault(); paletteNavigate(1); }
            else if (e.key === "ArrowUp") { e.preventDefault(); paletteNavigate(-1); }
            else if (e.key === "Enter") { e.preventDefault(); paletteExecuteSelected(); }
            else if (e.key === "Escape") { e.preventDefault(); closeCommandPalette(); }
        });

        el.commandPaletteOverlay.addEventListener("click", function(e) {
            if (e.target === el.commandPaletteOverlay) closeCommandPalette();
        });

        el.commandPaletteResults.addEventListener("click", function(e) {
            var item = e.target.closest(".command-palette-item");
            if (item) {
                executePaletteItem(item.getAttribute("data-tab"), item.getAttribute("data-sub"));
            }
        });

        // Ctrl+K to open
        document.addEventListener("keydown", function(e) {
            if (e.ctrlKey && e.key === "k") {
                var tag = e.target.tagName;
                if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || e.target.isContentEditable) return;
                e.preventDefault();
                openCommandPalette();
            }
        });
    }

    // ================================================================
    // v1.3.0 - Sub-Tab Filter (persistence infrastructure)
    // ================================================================
    function initSubTabFilter() {
        var hidden = {};
        try { hidden = JSON.parse(localStorage.getItem("opencut_hidden_tabs") || "{}"); } catch(e) {}
        var allSubs = document.querySelectorAll(".sub-tab");
        for (var i = 0; i < allSubs.length; i++) {
            var key = allSubs[i].dataset.sub;
            if (hidden[key]) allSubs[i].style.display = "none";
        }
    }

    // ================================================================
    // v1.3.0 - Audio Waveform Buttons (denoise/normalize)
    // ================================================================
    function addAudioWaveformButtons() {
        var btns = ["runDenoiseBtn", "runNormalizeBtn"];
        for (var i = 0; i < btns.length; i++) {
            var parent = document.getElementById(btns[i]);
            if (!parent) continue;
            parent = parent.parentNode;
            if (!parent || parent.querySelector(".waveform-audio-btn")) continue;
            var btn = document.createElement("button");
            btn.type = "button";
            btn.className = "btn-outline btn-sm waveform-audio-btn";
            btn.textContent = "Preview Waveform";
            btn.style.marginBottom = "6px";
            btn.addEventListener("click", function() {
                if (el.loadWaveformBtn) el.loadWaveformBtn.click();
            });
            parent.insertBefore(btn, parent.querySelector(".btn-primary"));
        }
    }

    // ================================================================
    // v1.3.0 - Trim Handler
    // ================================================================
    function parseTimeToSec(t) {
        var parts = (t || "0").split(":");
        var result;
        if (parts.length === 3) result = (+parts[0]) * 3600 + (+parts[1]) * 60 + (+parts[2]);
        else if (parts.length === 2) result = (+parts[0]) * 60 + (+parts[1]);
        else result = +parts[0];
        return isNaN(result) ? 0 : result;
    }

    function runTrim() {
        var startVal = el.trimStart ? el.trimStart.value.trim() || "00:00:00" : "00:00:00";
        var endVal = el.trimEnd ? el.trimEnd.value.trim() || "00:00:30" : "00:00:30";
        if (parseTimeToSec(endVal) <= parseTimeToSec(startVal)) {
            showAlert("End time must be after start time.");
            return;
        }
        var mode = el.trimMode ? el.trimMode.value : "reencode";
        var payload = {
            filepath: selectedPath,
            output_dir: projectFolder,
            start: startVal,
            end: endVal,
            quality: mode === "copy" ? "copy" : (el.trimQuality ? el.trimQuality.value : "medium")
        };
        startJob("/video/trim", payload);
    }

    // ================================================================
    // v1.3.0 - Merge Handler
    // ================================================================
    var _mergeFiles = [];

    function renderMergeFiles() {
        if (!el.mergeFileList) return;
        if (_mergeFiles.length === 0) {
            el.mergeFileList.innerHTML = '<div class="hint" style="padding:8px 12px;">No files added.</div>';
            if (el.runMergeBtn) el.runMergeBtn.disabled = true;
            return;
        }
        var html = "";
        for (var i = 0; i < _mergeFiles.length; i++) {
            var name = _mergeFiles[i].split(/[/\\]/).pop();
            html += '<div class="merge-file-item"><span class="merge-file-name">' + esc(name) + '</span><button type="button" class="btn-ghost btn-xs merge-file-remove" data-idx="' + i + '">&times;</button></div>';
        }
        el.mergeFileList.innerHTML = html;
        if (el.runMergeBtn) el.runMergeBtn.disabled = _mergeFiles.length < 2;
    }

    // Event delegation for merge file remove buttons (avoids listener accumulation)
    var _mergeDelegationAdded = false;
    function ensureMergeDelegation() {
        if (_mergeDelegationAdded || !el.mergeFileList) return;
        _mergeDelegationAdded = true;
        el.mergeFileList.addEventListener("click", function(e) {
            var btn = e.target.closest(".merge-file-remove");
            if (!btn) return;
            var idx = parseInt(btn.dataset.idx);
            if (idx >= 0 && idx < _mergeFiles.length) {
                _mergeFiles.splice(idx, 1);
                renderMergeFiles();
            }
        });
    }

    function runMerge() {
        if (_mergeFiles.length < 2) { showToast("Need at least 2 files to merge", "error"); return; }
        startJob("/video/merge", {
            files: _mergeFiles,
            output_dir: projectFolder,
            mode: el.mergeMode ? el.mergeMode.value : "concat",
            quality: el.mergeQuality ? el.mergeQuality.value : "medium"
        });
    }

    // ================================================================
    // v1.3.0 - Per-Operation Presets
    // ================================================================
    function saveOperationPreset(opName) {
        var settings = {};
        var activePanel = document.querySelector(".sub-panel:not(.hidden):not([style*='display: none'])");
        if (!activePanel) activePanel = document.querySelector(".sub-panel.active");
        if (!activePanel) return;
        var inputs = activePanel.querySelectorAll("input, select");
        for (var i = 0; i < inputs.length; i++) {
            if (!inputs[i].id) continue;
            if (inputs[i].type === "checkbox") {
                settings[inputs[i].id] = inputs[i].checked;
            } else {
                settings[inputs[i].id] = inputs[i].value;
            }
        }
        var all = {};
        try { all = JSON.parse(localStorage.getItem("opencut_op_presets") || "{}"); } catch(e) {}
        all[opName] = settings;
        try { localStorage.setItem("opencut_op_presets", JSON.stringify(all)); } catch(e) {}
        showToast("Preset saved for " + opName, "success");
    }

    function loadOperationPreset(opName) {
        var all = {};
        try { all = JSON.parse(localStorage.getItem("opencut_op_presets") || "{}"); } catch(e) {}
        var settings = all[opName];
        if (!settings) { showToast("No saved preset for " + opName, "info"); return; }
        for (var id in settings) {
            var el2 = document.getElementById(id);
            if (!el2) continue;
            if (el2.type === "checkbox") {
                el2.checked = settings[id];
            } else {
                el2.value = settings[id];
            }
            // Trigger appropriate event — "input" for sliders (display update), "change" for selects
            var evtName = (el2.type === "range") ? "input" : "change";
            var evt;
            try { evt = new Event(evtName, { bubbles: true }); }
            catch (err2) { evt = document.createEvent("Event"); evt.initEvent(evtName, true, true); }
            el2.dispatchEvent(evt);
        }
        showToast("Preset loaded for " + opName, "success");
    }

    // Health ping consolidated into checkHealth() above

    // ================================================================
    // v1.3.0 — New Feature Handlers
    // ================================================================

    // --- LLM Config helpers ---
    function getLLMConfig() {
        var provider = el.llmProvider ? el.llmProvider.value : "ollama";
        var config = { provider: provider };
        if (el.llmModel && el.llmModel.value) config.model = el.llmModel.value;
        if (el.llmApiKey && el.llmApiKey.value) config.api_key = el.llmApiKey.value;
        if (el.llmBaseUrl && el.llmBaseUrl.value) config.base_url = el.llmBaseUrl.value;
        return config;
    }

    function saveLLMSettings() {
        try {
            var cfg = getLLMConfig();
            localStorage.setItem("opencut_llm", JSON.stringify(cfg));
        } catch (e) {}
    }

    function loadLLMSettings() {
        try {
            var saved = localStorage.getItem("opencut_llm");
            if (!saved) return;
            var cfg = JSON.parse(saved);
            if (cfg.provider && el.llmProvider) el.llmProvider.value = cfg.provider;
            if (cfg.model && el.llmModel) el.llmModel.value = cfg.model;
            if (cfg.api_key && el.llmApiKey) el.llmApiKey.value = cfg.api_key;
            if (cfg.base_url && el.llmBaseUrl) el.llmBaseUrl.value = cfg.base_url;
            updateLLMProviderUI();
        } catch (e) {}
    }

    function updateLLMProviderUI() {
        if (!el.llmProvider) return;
        var provider = el.llmProvider.value;
        var needsKey = provider === "openai" || provider === "anthropic";
        if (el.llmApiKeyGroup) el.llmApiKeyGroup.classList.toggle("hidden", !needsKey);
        if (el.llmBaseUrl) {
            var defaults = { ollama: "http://localhost:11434", openai: "https://api.openai.com/v1", anthropic: "https://api.anthropic.com" };
            if (!el.llmBaseUrl.value || el.llmBaseUrl.value === defaults.ollama || el.llmBaseUrl.value === defaults.openai || el.llmBaseUrl.value === defaults.anthropic) {
                el.llmBaseUrl.placeholder = defaults[provider] || "";
            }
        }
        if (el.llmModel) {
            var modelDefaults = { ollama: "llama3.1", openai: "gpt-4o-mini", anthropic: "claude-sonnet-4-20250514" };
            if (!el.llmModel.value) el.llmModel.placeholder = modelDefaults[provider] || "";
        }
    }

    function testLLM() {
        var cfg = getLLMConfig();
        if (el.llmStatus) el.llmStatus.textContent = "Testing...";
        api("POST", "/llm/test", { prompt: "Say hello in one sentence.", provider: cfg.provider, model: cfg.model || "", api_key: cfg.api_key || "", base_url: cfg.base_url || "" }, function (err, resp) {
            if (err || !resp || !resp.success) {
                var msg = (resp && resp.error) ? resp.error : (err && typeof err === "object" && err.message) ? err.message : "Connection failed";
                if (el.llmStatus) el.llmStatus.textContent = "Failed: " + msg;
                return;
            }
            if (el.llmStatus) el.llmStatus.textContent = "Connected: " + resp.provider + "/" + resp.model;
            saveLLMSettings();
            showToast("LLM connected", "success");
        });
    }

    // --- Silence mode toggle ---
    function loadLlmSettings() {
        fetch(BACKEND + "/settings/llm", { headers: { "X-CSRF-Token": csrfToken } })
            .then(function(r) { return r.json(); })
            .then(function(s) {
                if (s.provider) {
                    var sel = document.getElementById("llmProvider2");
                    if (sel) sel.value = s.provider;
                }
                if (s.model) {
                    var m = document.getElementById("llmModel2");
                    if (m) m.value = s.model;
                }
                if (s.api_key && s.api_key !== "****") {
                    var k = document.getElementById("llmApiKey2");
                    if (k) k.value = s.api_key;
                }
                if (s.base_url) {
                    var u = document.getElementById("llmBaseUrl2");
                    if (u) u.value = s.base_url;
                }
                updateLlmProviderUI();
            })
            .catch(function() {});
    }

    function saveLlmSettings() {
        var provider = (document.getElementById("llmProvider2") || {}).value || "ollama";
        var model = (document.getElementById("llmModel2") || {}).value || "llama3";
        var apiKey = (document.getElementById("llmApiKey2") || {}).value || "";
        var baseUrl = (document.getElementById("llmBaseUrl2") || {}).value || "";
        fetch(BACKEND + "/settings/llm", {
            method: "POST",
            headers: { "Content-Type": "application/json", "X-CSRF-Token": csrfToken },
            body: JSON.stringify({ provider: provider, model: model, api_key: apiKey, base_url: baseUrl })
        }).then(function(r) { return r.json(); })
          .then(function() { showToast("LLM settings saved", "success"); })
          .catch(function() { showToast("Failed to save LLM settings", "error"); });
    }

    function updateLlmProviderUI() {
        var provider = (document.getElementById("llmProvider2") || {}).value || "ollama";
        var apiKeyRow = document.getElementById("llmApiKeyRow");
        var baseUrlRow = document.getElementById("llmBaseUrlRow");
        if (apiKeyRow) apiKeyRow.style.display = provider === "ollama" ? "none" : "";
        if (baseUrlRow) baseUrlRow.style.display = provider === "ollama" ? "" : "none";
    }

    function saveAudioZoomDefaults() {
        var lufs = parseFloat((document.getElementById("defaultLufs") || {}).value || -14);
        var zoom = parseFloat((document.getElementById("defaultZoom") || {}).value || 1.15);
        var easing = (document.getElementById("defaultZoomEasing") || {}).value || "ease_in_out";
        fetch(BACKEND + "/settings/loudness-target", {
            method: "POST",
            headers: { "Content-Type": "application/json", "X-CSRF-Token": csrfToken },
            body: JSON.stringify({ target_lufs: lufs })
        }).catch(function() {});
        fetch(BACKEND + "/settings/auto-zoom", {
            method: "POST",
            headers: { "Content-Type": "application/json", "X-CSRF-Token": csrfToken },
            body: JSON.stringify({ zoom_amount: zoom, easing: easing })
        }).then(function() { showToast("Defaults saved", "success"); })
          .catch(function() { showToast("Failed to save defaults", "error"); });
    }

    function toggleCard(cardId) {
        var card = document.getElementById(cardId);
        if (!card) return;
        var body = card.querySelector(".card-body");
        var toggle = card.querySelector(".card-toggle");
        if (body) body.style.display = body.style.display === "none" ? "" : "none";
        if (toggle) toggle.textContent = (body && body.style.display === "none") ? "▸" : "▾";
    }

    function updateSilenceModeUI() {
        if (!el.silenceMode) return;
        var isSpeedUp = el.silenceMode.value === "speedup";
        if (el.silenceSpeedGroup) el.silenceSpeedGroup.style.display = isSpeedUp ? "" : "none";
        // Hide preset/padding rows for speed-up mode
        if (el.silencePreset) { var fg1 = el.silencePreset.closest(".form-group"); if (fg1) fg1.style.display = isSpeedUp ? "none" : ""; }
        if (el.padBefore) { var fg2 = el.padBefore.closest(".form-group"); if (fg2) fg2.style.display = isSpeedUp ? "none" : ""; }
        if (el.padAfter) { var fg3 = el.padAfter.closest(".form-group"); if (fg3) fg3.style.display = isSpeedUp ? "none" : ""; }
    }

    // --- Face tracking smoothing toggle ---
    function updateFaceTrackingUI() {
        if (!el.reframeCropPos) return;
        var isFace = el.reframeCropPos.value === "face";
        if (el.reframeFaceSmoothing) el.reframeFaceSmoothing.style.display = isFace ? "" : "none";
    }

    // --- Auto-Edit ---
    function runAutoEdit() {
        startJob("/video/auto-edit", {
            filepath: selectedPath,
            output_dir: projectFolder,
            method: el.autoEditMethod ? el.autoEditMethod.value : "motion",
            threshold: el.autoEditThreshold ? parseFloat(el.autoEditThreshold.value) : 0.04,
            margin: el.autoEditMargin ? parseFloat(el.autoEditMargin.value) : 0.3,
            min_clip_length: el.autoEditMinClip ? parseFloat(el.autoEditMinClip.value) : 1.0,
        });
    }

    // --- Highlights ---
    function runHighlights() {
        var llm = getLLMConfig();
        startJob("/video/highlights", {
            filepath: selectedPath,
            output_dir: projectFolder,
            max_highlights: el.highlightMax ? parseInt(el.highlightMax.value) : 5,
            min_duration: el.highlightMinDur ? parseFloat(el.highlightMinDur.value) : 15,
            max_duration: el.highlightMaxDur ? parseFloat(el.highlightMaxDur.value) : 60,
            llm_provider: llm.provider,
            llm_model: llm.model,
            llm_api_key: llm.api_key,
            llm_base_url: llm.base_url,
        });
    }

    // --- Speech Enhance ---
    function runEnhance() {
        startJob("/audio/enhance", {
            filepath: selectedPath,
            output_dir: projectFolder,
            denoise: el.enhanceDenoise ? el.enhanceDenoise.checked : true,
            enhance: el.enhanceUpscale ? el.enhanceUpscale.checked : true,
        });
    }

    // --- Transcript Summarize ---
    function runSummarize() {
        var llm = getLLMConfig();
        if (el.summaryResult) el.summaryResult.classList.remove("hidden");
        if (el.summaryContent) el.summaryContent.textContent = "Summarizing...";
        startJob("/transcript/summarize", {
            filepath: selectedPath,
            style: "bullets",
            llm_provider: llm.provider,
            llm_model: llm.model || "",
            llm_api_key: llm.api_key || "",
            llm_base_url: llm.base_url || "",
        });
    }

    // --- Generate LUT from Reference ---
    function runGenerateLut() {
        var refPath = el.lutRefPath ? el.lutRefPath.value.trim() : "";
        if (!refPath) { showAlert("Select a reference image."); return; }
        var lutName = el.lutRefName ? el.lutRefName.value.trim() : "";
        if (!lutName) lutName = "custom_ref";
        startJob("/video/lut/generate-from-ref", {
            reference_path: refPath,
            lut_name: lutName,
            strength: el.lutRefStrength ? parseFloat(el.lutRefStrength.value) : 1.0,
        });
    }

    // --- Shorts Pipeline ---
    function runShorts() {
        var llm = getLLMConfig();
        var platform = el.shortsPlatform ? el.shortsPlatform.value : "tiktok";
        var dims = { tiktok: [1080, 1920], shorts: [1080, 1920], reels: [1080, 1920], square: [1080, 1080] };
        var d = dims[platform] || [1080, 1920];
        startJob("/video/shorts-pipeline", {
            filepath: selectedPath,
            output_dir: projectFolder,
            width: d[0],
            height: d[1],
            max_shorts: el.shortsMaxClips ? parseInt(el.shortsMaxClips.value) : 5,
            min_duration: el.shortsMinDur ? parseFloat(el.shortsMinDur.value) : 15,
            max_duration: el.shortsMaxDur ? parseFloat(el.shortsMaxDur.value) : 60,
            face_track: el.shortsFaceTrack ? el.shortsFaceTrack.checked : false,
            burn_captions: el.shortsCaptions ? el.shortsCaptions.checked : false,
            llm_provider: llm.provider,
            llm_model: llm.model || "",
            llm_api_key: llm.api_key || "",
            llm_base_url: llm.base_url || "",
        });
    }

    // --- Slider value display updaters ---
    function initNewSliderDisplays() {
        var sliders = [
            ["silenceSpeedFactor", "silenceSpeedVal", "x"],
            ["autoEditThreshold", "autoEditThresholdVal", ""],
            ["autoEditMargin", "autoEditMarginVal", "s"],
            ["autoEditMinClip", "autoEditMinClipVal", "s"],
            ["highlightMax", "highlightMaxVal", ""],
            ["faceSmoothing", "faceSmoothingVal", ""],
            ["lutRefStrength", "lutRefStrengthVal", ""],
            ["shortsMaxClips", "shortsMaxClipsVal", ""],
        ];
        for (var i = 0; i < sliders.length; i++) {
            (function (sliderId, displayId, unit) {
                var slider = document.getElementById(sliderId);
                var display = document.getElementById(displayId);
                if (slider && display) {
                    display.textContent = slider.value + unit;
                    slider.addEventListener("input", function () {
                        display.textContent = this.value + unit;
                    });
                }
            })(sliders[i][0], sliders[i][1], sliders[i][2]);
        }
    }

    // ================================================================
    // v1.5.0 — Timeline Tab Functions
    // ================================================================

    function applySequenceCuts(cuts) {
        if (!inPremiere) { showAlert("Premiere Pro connection required."); return; }
        var payload = JSON.stringify(cuts);
        cs.evalScript('ocApplySequenceCuts(' + JSON.stringify(payload) + ')', function (result) {
            try {
                var r = JSON.parse(result);
                showToast("Applied " + (r.applied || 0) + " cuts to sequence", "success");
                var statusEl = document.getElementById("tlWritebackStatus");
                if (statusEl) statusEl.textContent = "Applied " + (r.applied || 0) + " cuts to sequence.";
            } catch (e) { showAlert("Error applying cuts: " + (result || e.message)); }
        });
    }

    function runBeatMarkers() {
        startJob("/audio/beat-markers", {
            filepath: selectedPath,
            subdivisions: parseInt(document.getElementById("beatMarkerSubs").value || "1"),
        });
    }

    addJobDoneListener(function (job) {
        if (job.type !== "beat-markers" || job.status !== "complete" || !job.result) return;
        var r = job.result;
        beatMarkerTimes = r.beat_times || r.beats || [];
        var res = document.getElementById("beatMarkersResult");
        var sum = document.getElementById("beatMarkersSummary");
        if (res) res.classList.remove("hidden");
        if (sum) sum.textContent = beatMarkerTimes.length + " beat markers detected. BPM: " + safeFixed(r.bpm || 0, 1);
    });

    function addBeatMarkersToSequence() {
        if (!inPremiere) { showAlert("Premiere Pro connection required."); return; }
        if (!beatMarkerTimes || !beatMarkerTimes.length) { showAlert("No beat markers detected."); return; }
        var payload = JSON.stringify({ times: beatMarkerTimes, type: "Chapter" });
        cs.evalScript('ocAddSequenceMarkers(' + JSON.stringify(payload) + ')', function (result) {
            try {
                var r = JSON.parse(result);
                showToast("Added " + (r.added || beatMarkerTimes.length) + " markers", "success");
            } catch (e) { showAlert("Error adding markers: " + (result || e.message)); }
        });
    }

    function runMulticamCuts() {
        var trackMap = [];
        var rows = document.querySelectorAll(".multicam-track-row");
        for (var i = 0; i < rows.length; i++) {
            var trackInput = rows[i].querySelector(".multicam-track-input");
            trackMap.push(trackInput ? parseInt(trackInput.value) || i : i);
        }
        startJob("/video/multicam-cuts", {
            filepath: selectedPath,
            output_dir: projectFolder,
            num_speakers: parseInt(document.getElementById("multicamSpeakers").value || "2"),
            min_cut_duration: parseFloat(document.getElementById("multicamMinCut").value || "1.0"),
            track_map: trackMap,
        });
    }

    addJobDoneListener(function (job) {
        if (job.type !== "multicam-cuts" || job.status !== "complete" || !job.result) return;
        var r = job.result;
        multicamCutsData = r.cuts || r;
        var res = document.getElementById("multicamResult");
        var sum = document.getElementById("multicamSummary");
        if (res) res.classList.remove("hidden");
        if (sum) sum.textContent = (r.total_cuts || (r.cuts && r.cuts.length) || 0) + " cuts generated.";
    });

    function applyMulticamCuts() {
        if (!inPremiere) { showAlert("Premiere Pro connection required."); return; }
        if (!multicamCutsData) { showAlert("No multicam cuts available."); return; }
        var payload = JSON.stringify(multicamCutsData);
        cs.evalScript('ocApplySequenceCuts(' + JSON.stringify(payload) + ')', function (result) {
            try {
                var r = JSON.parse(result);
                showToast("Multicam cuts applied: " + (r.applied || 0), "success");
            } catch (e) { showAlert("Error: " + (result || e.message)); }
        });
    }

    function renderMulticamTrackMap() {
        var n = parseInt(document.getElementById("multicamSpeakers").value || "2");
        var container = document.getElementById("multicamTrackMap");
        if (!container) return;
        var html = "";
        for (var i = 0; i < n; i++) {
            html += '<div class="multicam-track-row" style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">'
                + '<span style="font-size:11px;color:var(--text-secondary);min-width:70px;">Speaker ' + (i + 1) + '</span>'
                + '<span style="font-size:11px;color:var(--text-muted);">\u2192 Track</span>'
                + '<input type="number" class="multicam-track-input" value="' + i + '" min="0" max="20" style="width:50px;">'
                + '</div>';
        }
        container.innerHTML = html;
    }

    function getSeqMarkers() {
        if (!inPremiere) { showAlert("Premiere Pro connection required."); return; }
        cs.evalScript('ocGetSequenceMarkers()', function (result) {
            try {
                var markers = JSON.parse(result);
                seqMarkersData = markers;
                var listEl = document.getElementById("markerExportList");
                var exportBtn = document.getElementById("exportMarkedClipsBtn");
                if (listEl) {
                    listEl.classList.remove("hidden");
                    if (!markers || !markers.length) {
                        listEl.innerHTML = '<div class="hint">No markers found in sequence.</div>';
                    } else {
                        var html = "";
                        for (var i = 0; i < markers.length; i++) {
                            var m = markers[i];
                            var dur = m.duration != null ? safeFixed(m.duration, 2) + "s" : "--";
                            html += '<div style="font-size:11px;padding:3px 0;border-bottom:1px solid var(--border);">'
                                + esc(m.name || ("Marker " + (i + 1))) + ' &mdash; ' + fmtDur(m.start || 0) + ' (' + dur + ')'
                                + '</div>';
                        }
                        listEl.innerHTML = html;
                    }
                }
                if (exportBtn) exportBtn.disabled = !(markers && markers.length);
            } catch (e) { showAlert("Error reading markers: " + (result || e.message)); }
        });
    }

    function exportMarkedClips() {
        if (!seqMarkersData || !seqMarkersData.length) { showAlert("Get sequence markers first."); return; }
        var outDir = (document.getElementById("markerExportDir") || {}).value || projectFolder;
        startJob("/timeline/export-from-markers", {
            filepath: selectedPath,
            output_dir: outDir,
            markers: seqMarkersData,
        });
    }

    addJobDoneListener(function (job) {
        if (job.type !== "export-from-markers" || job.status !== "complete" || !job.result) return;
        var r = job.result;
        var res = document.getElementById("markerExportResult");
        var sum = document.getElementById("markerExportSummary");
        if (res) res.classList.remove("hidden");
        if (sum) sum.textContent = "Exported " + (r.exported || 0) + " clips.";
    });

    function loadProjectItems() {
        if (!inPremiere) { showAlert("Premiere Pro connection required."); return; }
        cs.evalScript('getAllProjectMedia()', function (result) {
            try {
                var items = JSON.parse(result);
                renameItemsData = items || [];
                renderRenameItems();
                var applyBtn = document.getElementById("applyRenamePatternBtn");
                var renameBtn = document.getElementById("renameAllBtn");
                if (applyBtn) applyBtn.disabled = !renameItemsData.length;
                if (renameBtn) renameBtn.disabled = !renameItemsData.length;
            } catch (e) { showAlert("Error loading items: " + (result || e.message)); }
        });
    }

    function renderRenameItems() {
        var container = document.getElementById("renameItemsList");
        if (!container) return;
        if (!renameItemsData.length) {
            container.innerHTML = '<div class="hint">No items loaded.</div>';
            return;
        }
        var html = "";
        for (var i = 0; i < renameItemsData.length; i++) {
            var item = renameItemsData[i];
            html += '<div style="display:flex;align-items:center;gap:4px;margin-bottom:3px;">'
                + '<input type="text" class="text-input rename-name-input" data-idx="' + i + '" value="' + esc(item.name || "") + '" style="flex:1;font-size:11px;">'
                + '</div>';
        }
        container.innerHTML = html;
    }

    function applyRenamePattern() {
        var find = (document.getElementById("renameFindText") || {}).value || "";
        var replace = (document.getElementById("renameReplaceText") || {}).value || "";
        if (!find) { showAlert("Enter find text."); return; }
        var inputs = document.querySelectorAll(".rename-name-input");
        for (var i = 0; i < inputs.length; i++) {
            inputs[i].value = inputs[i].value.split(find).join(replace);
        }
    }

    function renameAll() {
        var inputs = document.querySelectorAll(".rename-name-input");
        var renames = [];
        for (var i = 0; i < inputs.length; i++) {
            var idx = parseInt(inputs[i].getAttribute("data-idx"));
            var orig = renameItemsData[idx];
            if (orig && inputs[i].value !== orig.name) {
                renames.push({ id: orig.id || orig.path, old_name: orig.name, new_name: inputs[i].value });
            }
        }
        if (!renames.length) { showAlert("No changes to apply."); return; }
        api("POST", "/timeline/batch-rename", { renames: renames }, function (err, data) {
            if (err || (data && data.error)) { showAlert("Validation failed: " + (data ? data.error : "Network error")); return; }
            if (!inPremiere) { showToast("Rename validated (no Premiere connection)", "info"); return; }
            var payload = JSON.stringify(renames);
            cs.evalScript('ocBatchRenameProjectItems(' + JSON.stringify(payload) + ')', function (result) {
                try {
                    var r = JSON.parse(result);
                    showToast("Renamed " + (r.renamed || renames.length) + " items", "success");
                } catch (e) { showAlert("Error: " + (result || e.message)); }
            });
        });
    }

    // ---- Smart Bins ----
    var smartBinRules = [];

    function addBinRule() {
        smartBinRules.push({ bin_name: "", rule_type: "contains", field: "name", value: "" });
        renderBinRules();
    }

    function removeBinRule(idx) {
        smartBinRules.splice(idx, 1);
        renderBinRules();
    }

    function renderBinRules() {
        var container = document.getElementById("smartBinRules");
        if (!container) return;
        if (!smartBinRules.length) {
            container.innerHTML = '<div class="hint">No rules yet. Click "+ Add Rule".</div>';
            return;
        }
        var html = "";
        for (var i = 0; i < smartBinRules.length; i++) {
            var r = smartBinRules[i];
            html += '<div class="smart-bin-rule" data-idx="' + i + '" style="display:flex;gap:4px;align-items:center;margin-bottom:6px;flex-wrap:wrap;">'
                + '<input type="text" class="text-input bin-name" data-idx="' + i + '" placeholder="Bin name" value="' + esc(r.bin_name) + '" style="width:80px;font-size:11px;">'
                + '<select class="bin-rule-type" data-idx="' + i + '" style="font-size:11px;">'
                + ['contains','starts_with','ends_with','type_is','duration_gt','duration_lt'].map(function(v) {
                    return '<option value="' + v + '"' + (r.rule_type === v ? ' selected' : '') + '>' + v + '</option>';
                }).join('')
                + '</select>'
                + '<select class="bin-field" data-idx="' + i + '" style="font-size:11px;">'
                + ['name','type','duration'].map(function(v) {
                    return '<option value="' + v + '"' + (r.field === v ? ' selected' : '') + '>' + v + '</option>';
                }).join('')
                + '</select>'
                + '<input type="text" class="text-input bin-value" data-idx="' + i + '" placeholder="Value" value="' + esc(r.value) + '" style="width:70px;font-size:11px;">'
                + '<button type="button" class="btn-ghost btn-xs bin-rule-remove" data-idx="' + i + '">&times;</button>'
                + '</div>';
        }
        container.innerHTML = html;
        // Attach change handlers
        container.querySelectorAll('.bin-name').forEach(function(el2) {
            el2.addEventListener('input', function() { smartBinRules[parseInt(this.dataset.idx)].bin_name = this.value; });
        });
        container.querySelectorAll('.bin-rule-type').forEach(function(el2) {
            el2.addEventListener('change', function() { smartBinRules[parseInt(this.dataset.idx)].rule_type = this.value; });
        });
        container.querySelectorAll('.bin-field').forEach(function(el2) {
            el2.addEventListener('change', function() { smartBinRules[parseInt(this.dataset.idx)].field = this.value; });
        });
        container.querySelectorAll('.bin-value').forEach(function(el2) {
            el2.addEventListener('input', function() { smartBinRules[parseInt(this.dataset.idx)].value = this.value; });
        });
        container.querySelectorAll('.bin-rule-remove').forEach(function(el2) {
            el2.addEventListener('click', function() { removeBinRule(parseInt(this.dataset.idx)); });
        });
    }

    function createSmartBins() {
        if (!smartBinRules.length) { showAlert("Add at least one rule."); return; }
        api("POST", "/timeline/smart-bins", { rules: smartBinRules }, function (err, data) {
            if (err || (data && data.error)) { showAlert("Validation failed: " + (data ? data.error : "Network error")); return; }
            if (!inPremiere) { showToast("Rules validated (no Premiere connection)", "info"); return; }
            var payload = JSON.stringify(smartBinRules);
            cs.evalScript('ocCreateSmartBins(' + JSON.stringify(payload) + ')', function (result) {
                try {
                    var r = JSON.parse(result);
                    showToast("Created " + (r.created || smartBinRules.length) + " bins", "success");
                } catch (e) { showAlert("Error: " + (result || e.message)); }
            });
        });
    }

    // ================================================================
    // v1.5.0 — Captions Tab New Features
    // ================================================================

    function runRepeatDetect() {
        startJob("/captions/repeat-detect", {
            filepath: selectedPath,
            model: document.getElementById("repeatModel").value,
            threshold: parseFloat(document.getElementById("repeatThreshold").value || "0.6"),
        });
    }

    addJobDoneListener(function (job) {
        if (job.type !== "repeat-detect" || job.status !== "complete" || !job.result) return;
        var r = job.result;
        repeatCutsData = r.cuts || r.ranges || [];
        lastTimelineCuts = repeatCutsData;
        var res = document.getElementById("repeatResults");
        var sum = document.getElementById("repeatSummary");
        var list = document.getElementById("repeatList");
        if (res) res.classList.remove("hidden");
        if (sum) sum.textContent = "Found " + repeatCutsData.length + " repeated takes.";
        if (list) {
            var html = "";
            for (var i = 0; i < repeatCutsData.length; i++) {
                var c = repeatCutsData[i];
                html += '<div style="font-size:11px;padding:3px 0;border-bottom:1px solid var(--border);">'
                    + fmtDur(c.start || 0) + " - " + fmtDur(c.end || 0)
                    + (c.text ? ' &mdash; <em>' + esc(c.text.substring(0, 60)) + '</em>' : '')
                    + '</div>';
            }
            list.innerHTML = html || '<div class="hint">No repeats found.</div>';
        }
        // Update writeback status
        var tlStatus = document.getElementById("tlWritebackStatus");
        if (tlStatus) tlStatus.textContent = repeatCutsData.length + " repeat cuts ready to apply.";
    });

    function applyRepeatCutsToTimeline() {
        if (!repeatCutsData || !repeatCutsData.length) { showAlert("No repeat cuts available."); return; }
        applySequenceCuts(repeatCutsData);
    }

    function runChapters() {
        var provider = document.getElementById("chaptersLlmProvider").value;
        var model = document.getElementById("chaptersLlmModel").value || "llama3";
        var apiKey = (document.getElementById("chaptersApiKey") || {}).value || "";
        var maxChapters = parseInt(document.getElementById("chaptersMax").value || "15");
        startJob("/captions/chapters", {
            filepath: selectedPath,
            llm_provider: provider,
            llm_model: model,
            llm_api_key: apiKey,
            max_chapters: maxChapters,
        });
    }

    addJobDoneListener(function (job) {
        if (job.type !== "chapters" || job.status !== "complete" || !job.result) return;
        var r = job.result;
        chaptersData = r.chapters || [];
        var res = document.getElementById("chaptersResult");
        var text = document.getElementById("chaptersText");
        if (res) res.classList.remove("hidden");
        if (text) {
            var block = r.description_block || "";
            if (!block && chaptersData.length) {
                block = chaptersData.map(function(c) {
                    return fmtDur(c.time || c.start || 0) + " " + (c.title || c.label || "");
                }).join("\n");
            }
            text.value = block;
        }
    });

    function copyChaptersDesc() {
        var text = document.getElementById("chaptersText");
        if (!text) return;
        if (navigator.clipboard) {
            navigator.clipboard.writeText(text.value).then(function() { showToast("Chapters copied", "success"); }).catch(function() { showToast("Copy failed", "warning"); });
        } else {
            text.select();
            document.execCommand("copy");
            showToast("Chapters copied", "success");
        }
    }

    function addChaptersAsMarkers() {
        if (!inPremiere) { showAlert("Premiere Pro connection required."); return; }
        if (!chaptersData || !chaptersData.length) { showAlert("No chapters available."); return; }
        var payload = JSON.stringify({ chapters: chaptersData, type: "chapter" });
        cs.evalScript('ocAddSequenceMarkers(' + JSON.stringify(payload) + ')', function (result) {
            try {
                var r = JSON.parse(result);
                showToast("Added " + (r.added || chaptersData.length) + " chapter markers", "success");
            } catch (e) { showAlert("Error: " + (result || e.message)); }
        });
    }

    function runSrtImport() {
        var path = (document.getElementById("srtImportPath") || {}).value || "";
        if (!path) { showAlert("Select an SRT file first."); return; }
        api("POST", "/timeline/srt-to-captions", { srt_path: path }, function (err, data) {
            if (err || (data && data.error)) { showAlert("Failed: " + (data ? data.error : "Network error")); return; }
            var segments = data.segments || [];
            if (!inPremiere) { showToast("SRT parsed (" + segments.length + " segments), no Premiere connection", "info"); return; }
            var payload = JSON.stringify(segments);
            cs.evalScript('ocAddNativeCaptionTrack(' + JSON.stringify(payload) + ')', function (result) {
                try {
                    var r = JSON.parse(result);
                    showToast("Imported " + (r.imported || segments.length) + " captions", "success");
                } catch (e) { showAlert("Error: " + (result || e.message)); }
            });
            var statusEl = document.getElementById("srtImportStatus");
            if (statusEl) { statusEl.textContent = "Imported " + segments.length + " caption segments."; statusEl.classList.remove("hidden"); }
        });
    }

    // ================================================================
    // v1.5.0 — Audio Tab: Loudness Match
    // ================================================================

    function runLoudMatch() {
        var paths = projectMedia.map(function(m) { return m.path || m; }).filter(Boolean);
        if (!paths.length) { showAlert("No project media found."); return; }
        var outDir = (document.getElementById("loudMatchOutputDir") || {}).value || projectFolder;
        startJob("/audio/loudness-match", {
            filepaths: paths,
            target_lufs: parseFloat(document.getElementById("loudMatchTarget").value || "-14"),
            output_dir: outDir,
        });
    }

    addJobDoneListener(function (job) {
        if (job.type !== "loudness-match" || job.status !== "complete" || !job.result) return;
        var r = job.result;
        var res = document.getElementById("loudMatchResults");
        var table = document.getElementById("loudMatchTable");
        if (res) res.classList.remove("hidden");
        if (table && r.clips) {
            var html = '<table style="width:100%;font-size:11px;border-collapse:collapse;">'
                + '<tr><th style="text-align:left;padding:2px 4px;">Clip</th><th>Original LUFS</th><th>Status</th></tr>';
            for (var i = 0; i < r.clips.length; i++) {
                var c = r.clips[i];
                var name = (c.path || c.name || "").split(/[/\\]/).pop();
                html += '<tr><td style="padding:2px 4px;">' + esc(name) + '</td>'
                    + '<td style="text-align:center;">' + safeFixed(c.original_lufs, 1) + '</td>'
                    + '<td style="text-align:center;color:' + (c.success ? 'var(--success)' : 'var(--error)') + ';">' + (c.success ? "OK" : "Failed") + '</td></tr>';
            }
            html += '</table>';
            table.innerHTML = html;
        }
    });

    // ================================================================
    // v1.5.0 — Export Tab: Deliverables
    // ================================================================

    function loadSeqInfo() {
        if (!inPremiere) { showAlert("Premiere Pro connection required."); return; }
        cs.evalScript('ocGetSequenceInfo()', function (result) {
            try {
                sequenceInfo = JSON.parse(result);
                var statusEl = document.getElementById("seqInfoStatus");
                if (statusEl) {
                    statusEl.textContent = "Loaded: " + (sequenceInfo.name || "Unknown") + " — " + (sequenceInfo.clip_count || 0) + " clips";
                    statusEl.classList.remove("hidden");
                }
                var btns = ["genVfxSheetBtn","genAdrListBtn","genMusicCueBtn","genAssetListBtn"];
                btns.forEach(function(id) { var b = document.getElementById(id); if (b) b.disabled = false; });
                showToast("Sequence info loaded", "success");
            } catch (e) { showAlert("Error loading sequence info: " + (result || e.message)); }
        });
    }

    function genDeliverableDoc(type) {
        if (!sequenceInfo) { showAlert("Load sequence info first."); return; }
        var outDir = (document.getElementById("deliverablesOutputDir") || {}).value || projectFolder;
        api("POST", "/deliverables/" + type, { sequence_data: sequenceInfo, output_dir: outDir }, function (err, data) {
            if (err || (data && data.error)) { showAlert("Generation failed: " + (data ? data.error : "Network error")); return; }
            var res = document.getElementById("deliverablesResult");
            var fp = document.getElementById("deliverablesFilePath");
            if (res) res.classList.remove("hidden");
            if (fp) fp.textContent = data.output_path || "File generated.";
            showToast(type + " generated", "success");
        });
    }

    // ================================================================
    // v1.5.0 — NLP Tab Functions
    // ================================================================

    function indexAllClips() {
        var paths = projectMedia.map(function(m) { return m.path || m; }).filter(Boolean);
        if (!paths.length) { showAlert("No project media found."); return; }
        var btn = document.getElementById("indexAllClipsBtn");
        if (btn) { btn.disabled = true; btn.textContent = "Indexing..."; }
        api("POST", "/search/index", { filepaths: paths }, function (err, data) {
            if (btn) { btn.disabled = false; btn.textContent = "Index All Project Clips"; }
            if (err || (data && data.error)) { showAlert("Indexing failed: " + (data ? data.error : "Network error")); return; }
            footageIndex = data;
            var statsEl = document.getElementById("searchIndexStats");
            if (statsEl) statsEl.textContent = "Index: " + (data.total_files || paths.length) + " files, " + (data.total_segments || 0) + " segments.";
            showToast("Indexing complete", "success");
        });
    }

    function runFootageSearch() {
        var query = (document.getElementById("footageSearchQuery") || {}).value || "";
        if (!query) { showAlert("Enter a search query."); return; }
        var maxResults = parseInt((document.getElementById("footageSearchMax") || {}).value || "10");
        api("POST", "/search/footage", { query: query, max_results: maxResults }, function (err, data) {
            var res = document.getElementById("footageSearchResults");
            if (!res) return;
            if (err || !data) { res.innerHTML = '<div class="hint">Search failed.</div>'; return; }
            var results = data.results || [];
            if (!results.length) { res.innerHTML = '<div class="hint">No results found.</div>'; return; }
            var html = "";
            for (var i = 0; i < results.length; i++) {
                var r = results[i];
                var name = (r.path || "").split(/[/\\]/).pop();
                var timeRange = r.start != null ? fmtDur(r.start) + " - " + fmtDur(r.end || r.start) : "";
                html += '<div class="footage-result-item" data-path="' + esc(r.path || "") + '" style="padding:6px;border-bottom:1px solid var(--border);cursor:pointer;" title="Click to select">'
                    + '<div style="font-size:11px;font-weight:600;">' + esc(name) + (timeRange ? ' &mdash; ' + timeRange : '') + '</div>'
                    + (r.text ? '<div style="font-size:10px;color:var(--text-muted);margin-top:2px;">' + esc(r.text.substring(0, 80)) + '</div>' : '')
                    + '</div>';
            }
            res.innerHTML = html;
            // Click to select clip
            var items = res.querySelectorAll(".footage-result-item");
            for (var j = 0; j < items.length; j++) {
                items[j].addEventListener("click", function() {
                    var p = this.getAttribute("data-path");
                    if (p) selectFile(p, p.split(/[/\\]/).pop());
                });
            }
        });
    }

    function runNlpCommand() {
        var text = (document.getElementById("nlpCommandText") || {}).value || "";
        if (!text) { showAlert("Enter a command."); return; }
        var provider = (document.getElementById("nlpLlmProvider") || {}).value || "ollama";
        var btn = document.getElementById("runNlpCommandBtn");
        if (btn) { btn.disabled = true; btn.textContent = "Processing..."; }
        api("POST", "/nlp/command", { command: text, filepath: selectedPath, llm_provider: provider }, function (err, data) {
            if (btn) { btn.disabled = false; btn.textContent = "Execute"; }
            var res = document.getElementById("nlpCommandResult");
            var routeEl = document.getElementById("nlpCommandRoute");
            var confEl = document.getElementById("nlpCommandConf");
            var outEl = document.getElementById("nlpCommandOutput");
            if (res) res.classList.remove("hidden");
            if (err || !data) {
                if (routeEl) routeEl.textContent = "Error: " + (err ? err.message : "Unknown");
                return;
            }
            if (routeEl) routeEl.textContent = "Route: " + (data.route || "unknown");
            if (confEl) confEl.textContent = "Confidence: " + safeFixed((data.confidence || 0) * 100, 0) + "%";
            if (outEl) outEl.textContent = data.result ? JSON.stringify(data.result, null, 2) : "";
            // Auto-execute matched route if high confidence
            if (data.route && data.confidence > 0.6 && data.params) {
                startJob(data.route, Object.assign({ filepath: selectedPath, output_dir: projectFolder }, data.params));
            }
        });
    }

    // ================================================================
    // v1.5.0 — Init Timeline/NLP features
    // ================================================================

    function initTimelineFeatures() {
        // Write-back
        var applyBtn = document.getElementById("applySeqCutsBtn");
        if (applyBtn) applyBtn.addEventListener("click", function() {
            if (!lastTimelineCuts) { showAlert("No cuts available. Run Silence Removal or Repeat Detection first."); return; }
            applySequenceCuts(lastTimelineCuts);
        });

        // Beat markers
        var beatBtn = document.getElementById("runBeatMarkersBtn");
        if (beatBtn) beatBtn.addEventListener("click", runBeatMarkers);
        var addBeatBtn = document.getElementById("addBeatMarkersBtn");
        if (addBeatBtn) addBeatBtn.addEventListener("click", addBeatMarkersToSequence);

        // Multicam
        var multicamBtn = document.getElementById("runMulticamBtn");
        if (multicamBtn) multicamBtn.addEventListener("click", runMulticamCuts);
        var applMcBtn = document.getElementById("applyMulticamCutsBtn");
        if (applMcBtn) applMcBtn.addEventListener("click", applyMulticamCuts);
        var speakersInput = document.getElementById("multicamSpeakers");
        if (speakersInput) {
            speakersInput.addEventListener("change", renderMulticamTrackMap);
            renderMulticamTrackMap();
        }

        // Export from markers
        var getMarkersBtn = document.getElementById("getSeqMarkersBtn");
        if (getMarkersBtn) getMarkersBtn.addEventListener("click", getSeqMarkers);
        var exportMarkedBtn = document.getElementById("exportMarkedClipsBtn");
        if (exportMarkedBtn) exportMarkedBtn.addEventListener("click", exportMarkedClips);

        // Batch rename
        var loadItemsBtn = document.getElementById("loadProjectItemsBtn");
        if (loadItemsBtn) loadItemsBtn.addEventListener("click", loadProjectItems);
        var patternBtn = document.getElementById("applyRenamePatternBtn");
        if (patternBtn) patternBtn.addEventListener("click", applyRenamePattern);
        var renameAllBtn = document.getElementById("renameAllBtn");
        if (renameAllBtn) renameAllBtn.addEventListener("click", renameAll);

        // Smart bins
        var addRuleBtn = document.getElementById("addBinRuleBtn");
        if (addRuleBtn) addRuleBtn.addEventListener("click", addBinRule);
        var createBinsBtn = document.getElementById("createSmartBinsBtn");
        if (createBinsBtn) createBinsBtn.addEventListener("click", createSmartBins);
        renderBinRules();

        // Slider: beat marker subdivisions (no display needed, select-only)
    }

    function initCaptionNewFeatures() {
        // Repeat detection
        var repeatBtn = document.getElementById("runRepeatDetectBtn");
        if (repeatBtn) repeatBtn.addEventListener("click", runRepeatDetect);
        var applyRepBtn = document.getElementById("applyRepeatCutsBtn");
        if (applyRepBtn) applyRepBtn.addEventListener("click", applyRepeatCutsToTimeline);
        var repThresh = document.getElementById("repeatThreshold");
        var repThreshVal = document.getElementById("repeatThresholdVal");
        if (repThresh && repThreshVal) {
            repThresh.addEventListener("input", function() { repThreshVal.textContent = safeFixed(parseFloat(this.value), 2); });
        }

        // Chapters
        var chaptersBtn = document.getElementById("runChaptersBtn");
        if (chaptersBtn) chaptersBtn.addEventListener("click", runChapters);
        var copyChapBtn = document.getElementById("copyChaptersDescBtn");
        if (copyChapBtn) copyChapBtn.addEventListener("click", copyChaptersDesc);
        var addChapMarkersBtn = document.getElementById("addChaptersMarkersBtn");
        if (addChapMarkersBtn) addChapMarkersBtn.addEventListener("click", addChaptersAsMarkers);
        var chapProvider = document.getElementById("chaptersLlmProvider");
        var chapApiKeyGroup = document.getElementById("chaptersApiKeyGroup");
        if (chapProvider && chapApiKeyGroup) {
            chapProvider.addEventListener("change", function() {
                chapApiKeyGroup.classList.toggle("hidden", this.value === "ollama");
            });
        }
        var chapMax = document.getElementById("chaptersMax");
        var chapMaxVal = document.getElementById("chaptersMaxVal");
        if (chapMax && chapMaxVal) {
            chapMax.addEventListener("input", function() { chapMaxVal.textContent = this.value; });
        }

        // SRT import
        var srtBtn = document.getElementById("runSrtImportBtn");
        if (srtBtn) srtBtn.addEventListener("click", runSrtImport);
    }

    function initAudioNewFeatures() {
        // Loudness match
        var loudBtn = document.getElementById("runLoudMatchBtn");
        if (loudBtn) loudBtn.addEventListener("click", runLoudMatch);
        var loudSlider = document.getElementById("loudMatchTarget");
        var loudVal = document.getElementById("loudMatchTargetVal");
        if (loudSlider && loudVal) {
            loudSlider.addEventListener("input", function() { loudVal.textContent = this.value + " LUFS"; });
        }
    }

    function initDeliverablesFeatures() {
        var loadSeqBtn = document.getElementById("loadSeqInfoBtn");
        if (loadSeqBtn) loadSeqBtn.addEventListener("click", loadSeqInfo);
        var vfxBtn = document.getElementById("genVfxSheetBtn");
        if (vfxBtn) vfxBtn.addEventListener("click", function() { genDeliverableDoc("vfx-sheet"); });
        var adrBtn = document.getElementById("genAdrListBtn");
        if (adrBtn) adrBtn.addEventListener("click", function() { genDeliverableDoc("adr-list"); });
        var musicBtn = document.getElementById("genMusicCueBtn");
        if (musicBtn) musicBtn.addEventListener("click", function() { genDeliverableDoc("music-cue-sheet"); });
        var assetBtn = document.getElementById("genAssetListBtn");
        if (assetBtn) assetBtn.addEventListener("click", function() { genDeliverableDoc("asset-list"); });
        var openFolderBtn = document.getElementById("openDeliverablesFolder");
        if (openFolderBtn) openFolderBtn.addEventListener("click", function() {
            var fp = document.getElementById("deliverablesFilePath");
            if (fp && fp.textContent && inPremiere) {
                cs.evalScript('openFolderInFinder(' + JSON.stringify(fp.textContent) + ')', function() {});
            }
        });
    }

    function initNlpFeatures() {
        var indexBtn = document.getElementById("indexAllClipsBtn");
        if (indexBtn) indexBtn.addEventListener("click", indexAllClips);
        var searchBtn = document.getElementById("runFootageSearchBtn");
        if (searchBtn) searchBtn.addEventListener("click", runFootageSearch);
        var footageQuery = document.getElementById("footageSearchQuery");
        if (footageQuery) {
            footageQuery.addEventListener("keydown", function(e) {
                if (e.key === "Enter") runFootageSearch();
            });
        }
        var nlpBtn = document.getElementById("runNlpCommandBtn");
        if (nlpBtn) nlpBtn.addEventListener("click", runNlpCommand);
    }

    // Hook silence result to update lastTimelineCuts
    addJobDoneListener(function (job) {
        if (job.type === "silence" && job.status === "complete" && job.result && job.result.cuts) {
            lastTimelineCuts = job.result.cuts;
            var tlStatus = document.getElementById("tlWritebackStatus");
            if (tlStatus) tlStatus.textContent = job.result.cuts.length + " silence cuts ready to apply.";
        }
    });

    // ================================================================
    // Init
    // ================================================================
    document.addEventListener("DOMContentLoaded", function () {
        initCSInterface();
        initDOM();
        setupNavTabs();
        checkSubTabOverflow();
        window.addEventListener("resize", checkSubTabOverflow);
        setupSliders();
        initCustomDropdowns(); // Initialize custom in-panel dropdowns

        // Event listeners - Clip selection
        el.refreshAllBtn.addEventListener("click", refreshAll);
        el.clipSelect.addEventListener("change", function () {
            var opt = this.selectedIndex >= 0 ? this.options[this.selectedIndex] : null;
            if (opt && opt.value) selectFile(opt.value, opt.getAttribute("data-name") || opt.value.split(/[/\\]/).pop());
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
        el.transcriptUndoBtn.addEventListener("click", undoTranscript);
        el.transcriptRedoBtn.addEventListener("click", redoTranscript);
        el.installWhisperBtn.addEventListener("click", installWhisper);
        el.captionStyle.addEventListener("change", updateStylePreview);

        // Audio tab buttons
        el.runSeparateBtn.addEventListener("click", runSeparate);
        el.installDemucsBtn.addEventListener("click", installDemucs);
        el.runDenoiseBtn.addEventListener("click", runDenoise);
        el.measureLoudnessBtn.addEventListener("click", measureLoudness);
        el.runNormalizeBtn.addEventListener("click", runNormalize);
        el.runBeatsBtn.addEventListener("click", runBeats);
        el.runEffectBtn.addEventListener("click", runEffect);

        // Video tab buttons
        el.runWatermarkBtn.addEventListener("click", runWatermark);
        el.installWatermarkBtn.addEventListener("click", installWatermark);
        el.runScenesBtn.addEventListener("click", runScenes);
        el.copyChaptersBtn.addEventListener("click", function () {
            var text = el.ytChaptersText.value;
            if (navigator.clipboard) {
                navigator.clipboard.writeText(text).then(function () { showAlert("Copied to clipboard!"); }).catch(function () { showAlert("Copy failed"); });
            } else {
                el.ytChaptersText.select();
                document.execCommand("copy");
                showAlert("Copied to clipboard!");
            }
        });

        // Video FX buttons
        el.runVfxBtn.addEventListener("click", runVfx);
        el.vfxSelect.addEventListener("change", showVfxParams);

        // Video AI buttons
        el.runVidAiBtn.addEventListener("click", runVidAi);
        el.vidAiTool.addEventListener("change", showVidAiParams);
        el.installVidAiBtn.addEventListener("click", installVidAi);

        // Audio Pro buttons
        el.runProFxBtn.addEventListener("click", runProFx);
        el.proFxCategory.addEventListener("change", updateProFxEffectList);
        el.proFxEffect.addEventListener("change", updateProFxParams);
        el.installPedalboardBtn.addEventListener("click", installPedalboard);
        el.runDeepFilterBtn.addEventListener("click", runDeepFilter);
        el.installDeepFilterBtn.addEventListener("click", installDeepFilter);

        // Face blur buttons
        el.runFaceBlurBtn.addEventListener("click", runFaceBlur);
        el.installMediapipeBtn.addEventListener("click", installMediapipe);

        // Style transfer button
        el.runStyleBtn.addEventListener("click", runStyleTransfer);

        // Caption translation buttons
        el.runTranslateBtn.addEventListener("click", runTranslate);
        el.installNllbBtn.addEventListener("click", installNllb);

        // Karaoke buttons
        el.runKaraokeBtn.addEventListener("click", runKaraoke);
        el.installWhisperxBtn.addEventListener("click", installWhisperx);

        // Export preset buttons
        el.runExportPresetBtn.addEventListener("click", runExportPreset);
        el.exportPresetCategory.addEventListener("change", updateExportPresetList);
        el.exportPresetSelect.addEventListener("change", updateExportPresetDesc);

        // Thumbnail button
        el.runThumbBtn.addEventListener("click", runThumbnails);

        // Batch button
        el.runBatchBtn.addEventListener("click", runBatch);
        el.runWorkflowBtn.addEventListener("click", runWorkflowPreset);

        // TTS buttons
        el.runTtsBtn.addEventListener("click", runTts);
        el.installEdgeTtsBtn.addEventListener("click", installEdgeTts);

        // SFX buttons
        el.runSfxBtn.addEventListener("click", runSfx);
        el.sfxType.addEventListener("change", showSfxParams);

        // Burn-in button
        el.runBurninBtn.addEventListener("click", runBurnin);

        // Speed buttons
        el.runSpeedBtn.addEventListener("click", runSpeed);
        el.speedMode.addEventListener("change", showSpeedParams);

        // LUT button
        el.runLutBtn.addEventListener("click", runLut);

        // Duck button
        el.runDuckBtn.addEventListener("click", runDuck);

        // Phase 6 buttons
        el.runChromaBtn.addEventListener("click", runChroma);
        el.chromaMode.addEventListener("change", showChromaParams);
        el.runTransBtn.addEventListener("click", runTransition);
        el.runParticlesBtn.addEventListener("click", runParticles);
        el.runTitleOverlayBtn.addEventListener("click", runTitleOverlay);
        el.runTitleCardBtn.addEventListener("click", runTitleCard);
        el.runReframeBtn.addEventListener("click", runReframe);
        el.reframePreset.addEventListener("change", updateReframeUI);
        el.reframeMode.addEventListener("change", updateReframeUI);
        el.reframeCustomW.addEventListener("input", updateReframeUI);
        el.reframeCustomH.addEventListener("input", updateReframeUI);
        updateReframeUI();
        el.runUpscaleBtn.addEventListener("click", runUpscale);
        el.runColorBtn.addEventListener("click", runColor);
        el.runRemoveBtn.addEventListener("click", runRemove);
        el.runFaceAiBtn.addEventListener("click", runFaceAi);
        el.faceAiMode.addEventListener("change", showFaceAiParams);
        el.runAnimCapBtn.addEventListener("click", runAnimCap);
        el.runMusicAiBtn.addEventListener("click", runMusicAi);

        // v1.3.0 - Trim
        if (el.runTrimBtn) el.runTrimBtn.addEventListener("click", runTrim);
        if (el.trimMode) el.trimMode.addEventListener("change", function() {
            if (el.trimQualityGroup) {
                el.trimQualityGroup.style.display = this.value === "copy" ? "none" : "";
            }
        });

        // v1.3.0 - Merge
        ensureMergeDelegation();
        if (el.mergeAddCurrentBtn) el.mergeAddCurrentBtn.addEventListener("click", function() {
            if (selectedPath && _mergeFiles.indexOf(selectedPath) === -1) {
                _mergeFiles.push(selectedPath);
                renderMergeFiles();
            }
        });
        if (el.mergeAddAllBtn) el.mergeAddAllBtn.addEventListener("click", function() {
            for (var i = 0; i < projectMedia.length; i++) {
                var path = projectMedia[i].path || projectMedia[i];
                if (_mergeFiles.indexOf(path) === -1) {
                    _mergeFiles.push(path);
                }
            }
            renderMergeFiles();
        });
        if (el.mergeClearBtn) el.mergeClearBtn.addEventListener("click", function() {
            _mergeFiles = [];
            renderMergeFiles();
        });
        if (el.runMergeBtn) el.runMergeBtn.addEventListener("click", runMerge);

        // v1.3.0 - Recent Clips
        if (el.recentClipsBtn) el.recentClipsBtn.addEventListener("click", showRecentClips);
        // Close recent clips dropdown on outside click
        document.addEventListener("click", function(e) {
            if (el.recentClipsDropdown && !el.recentClipsDropdown.classList.contains("hidden") &&
                !e.target.closest("#recentClipsBtn") && !e.target.closest("#recentClipsDropdown")) {
                el.recentClipsDropdown.classList.add("hidden");
            }
        });
        if (el.recentClipsDropdown) el.recentClipsDropdown.addEventListener("click", function(e) {
            var item = e.target.closest(".recent-clip-item");
            if (item) {
                var path = item.getAttribute("data-path");
                if (path) {
                    selectFile(path, path.split(/[/\\]/).pop());
                    el.recentClipsDropdown.classList.add("hidden");
                }
            }
        });

        // Export tab buttons
        el.runExpTranscriptBtn.addEventListener("click", runExpTranscript);

        // v1.3.0 — New feature event listeners
        if (el.silenceMode) el.silenceMode.addEventListener("change", updateSilenceModeUI);
        if (el.reframeCropPos) el.reframeCropPos.addEventListener("change", updateFaceTrackingUI);
        if (el.runAutoEditBtn) el.runAutoEditBtn.addEventListener("click", runAutoEdit);
        if (el.runHighlightsBtn) el.runHighlightsBtn.addEventListener("click", runHighlights);
        if (el.runEnhanceBtn) el.runEnhanceBtn.addEventListener("click", runEnhance);
        if (el.runShortsBtn) el.runShortsBtn.addEventListener("click", runShorts);
        if (el.summarizeTranscriptBtn) el.summarizeTranscriptBtn.addEventListener("click", runSummarize);
        if (el.generateLutBtn) el.generateLutBtn.addEventListener("click", runGenerateLut);
        if (el.testLLMBtn) el.testLLMBtn.addEventListener("click", testLLM);
        if (el.llmProvider) el.llmProvider.addEventListener("change", function () {
            updateLLMProviderUI();
            saveLLMSettings();
        });
        if (el.copySummaryBtn) el.copySummaryBtn.addEventListener("click", function () {
            var text = el.summaryContent ? el.summaryContent.textContent : "";
            if (navigator.clipboard) {
                navigator.clipboard.writeText(text).then(function () { showToast("Summary copied", "success"); }).catch(function () { showToast("Copy failed", "warning"); });
            } else {
                showToast("Copy not supported", "warning");
            }
        });

        // Settings tab buttons
        el.settingsInstallWhisperBtn.addEventListener("click", installWhisper);
        el.settingsReinstallWhisperBtn.addEventListener("click", reinstallWhisper);
        el.settingsClearCacheBtn.addEventListener("click", clearWhisperCache);
        el.whisperCpuMode.addEventListener("change", toggleCpuMode);
        el.restartBackendBtn.addEventListener("click", restartBackend);
        el.openLogsBtn.addEventListener("click", openLogs);
        
        // Settings persistence
        el.settingsAutoImport.addEventListener("change", saveLocalSettings);
        el.settingsAutoOpen.addEventListener("change", saveLocalSettings);
        el.settingsShowNotifications.addEventListener("change", saveLocalSettings);
        el.settingsOutputDir.addEventListener("change", saveLocalSettings);
        el.settingsDefaultModel.addEventListener("change", saveLocalSettings);
        el.settingsTheme.addEventListener("change", function () {
            applyTheme(this.value);
            saveLocalSettings();
        });
        
        // Load saved settings
        loadLocalSettings();

        // Progress / Results
        el.cancelBtn.addEventListener("click", cancelJob);
        el.processingCancel.addEventListener("click", cancelJob);
        el.newJobBtn.addEventListener("click", function () {
            el.resultsSection.classList.add("hidden");
            el.retryJobBtn.classList.add("hidden");
        });
        el.retryJobBtn.addEventListener("click", function () {
            el.resultsSection.classList.add("hidden");
            el.retryJobBtn.classList.add("hidden");
            if (lastJobEndpoint && lastJobPayload) {
                startJob(lastJobEndpoint, lastJobPayload);
            }
        });

        // Browse buttons for path inputs
        var browseBtns = document.querySelectorAll(".btn-browse");
        for (var i = 0; i < browseBtns.length; i++) {
            browseBtns[i].addEventListener("click", function () {
                browseForInput(this.getAttribute("data-target"));
            });
        }

        // Alert dismiss
        el.alertDismiss.addEventListener("click", function () {
            el.alertBanner.classList.add("hidden");
        });

        // Health check loop
        checkHealth();
        if (healthTimer) clearInterval(healthTimer);
        healthTimer = setInterval(checkHealth, HEALTH_MS);

        // Scan project media and populate recent files
        scanProjectMedia();
        populateRecentFiles();

        // Load style preview data
        loadStylePreview();

        // Load pedalboard effects list
        loadPedalboardEffects();

        // Init VFX param visibility
        showVfxParams();

        // Load export presets
        loadExportPresets();

        // New features
        initDropZone();
        initEnhancedDragDrop();
        initThemeToggle();
        initJobHistory();
        initKeyboardShortcuts();
        initPresets();
        initModelManagement();
        initGpuRecommendation();
        initQueue();
        initTranscriptSearch();
        initPremiereThemeSync();
        initWaveform();
        initFavorites();
        initPreviewModal();
        initAudioPreview();
        initContextMenu();
        initWizard();
        initOutputBrowser();
        initBatchPicker();
        initDepDashboard();
        initSettingsIO();
        initWorkflowBuilder();
        initCollapsibleCards();
        initI18n();

        // v1.3.0 inits
        loadRecentClips();
        initCommandPalette();
        initSubTabFilter();
        addAudioWaveformButtons();
        renderMergeFiles();
        initNewSliderDisplays();
        loadLLMSettings();
        loadLlmSettings();
        updateSilenceModeUI();
        updateFaceTrackingUI();

        // v1.5.0 inits
        initTimelineFeatures();
        initCaptionNewFeatures();
        initAudioNewFeatures();
        initDeliverablesFeatures();
        initNlpFeatures();

        // Pause CSS animations when panel is hidden (saves GPU/CPU in Premiere)
        document.addEventListener("visibilitychange", function () {
            var appEl = document.querySelector(".app");
            if (appEl) {
                if (document.hidden) appEl.classList.add("paused-animations");
                else appEl.classList.remove("paused-animations");
            }
        });

        // Cleanup SSE connections and timers on panel close/navigation
        window.addEventListener("beforeunload", function () {
            if (activeStream) {
                activeStream.close();
                activeStream = null;
            }
            if (healthTimer) {
                clearInterval(healthTimer);
                healthTimer = null;
            }
            if (pollTimer) {
                clearInterval(pollTimer);
                pollTimer = null;
            }
            if (elapsedTimer) {
                clearInterval(elapsedTimer);
                elapsedTimer = null;
            }
            if (batchPollTimer) {
                clearInterval(batchPollTimer);
                batchPollTimer = null;
            }
        });
    });

})();
