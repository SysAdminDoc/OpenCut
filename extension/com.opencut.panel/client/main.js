/* ============================================================
   OpenCut CEP Panel - Main Controller v0.6.5
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
    var lastJobEndpoint = "";  // for retry
    var lastJobPayload = null; // for retry

    // ---- Style Preview CSS Map (loaded from backend) ----
    var stylePreviewMap = {};

    // ============================================================
    // CUSTOM DROPDOWN SYSTEM - Inline Panel Dropdowns
    // ============================================================
    function initCustomDropdowns() {
        var selects = document.querySelectorAll('select:not(.no-custom)');
        for (var i = 0; i < selects.length; i++) {
            var select = selects[i];
            if (select.dataset.customized) continue;
            createCustomDropdown(select);
        }
        
        // Close dropdowns when clicking outside
        document.addEventListener('click', function(e) {
            if (!e.target.closest('.custom-dropdown')) {
                closeAllDropdowns();
            }
        });
        
        // Close on escape key
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                closeAllDropdowns();
            }
        });
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
        
        var selectedText = document.createElement('span');
        selectedText.className = 'custom-dropdown-text';
        
        var arrow = document.createElement('span');
        arrow.className = 'custom-dropdown-arrow';
        arrow.innerHTML = '<svg width="10" height="10" viewBox="0 0 16 16" fill="currentColor"><path d="M8 11L3 6h10l-5 5z"/></svg>';
        
        trigger.appendChild(selectedText);
        trigger.appendChild(arrow);
        
        var dropdown = document.createElement('div');
        dropdown.className = 'custom-dropdown-menu';
        
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
        trigger.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                toggleDropdown(e);
            } else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
                e.preventDefault();
                if (!wrapper.classList.contains('open')) {
                    toggleDropdown(e);
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
        
        // Store reference for updating
        select._customDropdown = {
            wrapper: wrapper,
            update: buildOptions,
            updateText: updateSelectedText
        };
    }
    
    function closeAllDropdowns() {
        var openDropdowns = document.querySelectorAll('.custom-dropdown.open');
        for (var i = 0; i < openDropdowns.length; i++) {
            openDropdowns[i].classList.remove('open');
        }
    }
    
    function updateCustomDropdown(selectId) {
        var select = document.getElementById(selectId);
        if (select && select._customDropdown) {
            select._customDropdown.update();
        }
    }

    // ---- DOM ----
    var el = {};
    function $(id) { return document.getElementById(id); }

    function initDOM() {
        // Header
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
        el.runSeparateBtn.disabled = !canRun;
        el.runDenoiseBtn.disabled = !canRun;
        el.measureLoudnessBtn.disabled = !canRun;
        el.runNormalizeBtn.disabled = !canRun;
        el.runBeatsBtn.disabled = !canRun;
        el.runEffectBtn.disabled = !canRun;

        // Video tab
        el.runWatermarkBtn.disabled = !canRun;
        el.runScenesBtn.disabled = !canRun;
        el.runVfxBtn.disabled = !canRun;
        el.runVidAiBtn.disabled = !canRun;

        // Audio pro tab
        el.runProFxBtn.disabled = !canRun;
        el.runDeepFilterBtn.disabled = !canRun;

        // Video face/style
        el.runFaceBlurBtn.disabled = !canRun;
        el.runStyleBtn.disabled = !canRun;

        // Captions translate/karaoke
        el.runTranslateBtn.disabled = !canRun;
        el.runKaraokeBtn.disabled = !canRun;

        // Export presets/thumbnails/batch
        el.runExportPresetBtn.disabled = !canRun;
        el.runThumbBtn.disabled = !canRun;
        el.runBatchBtn.disabled = !canRun;

        // Caption burn-in
        el.runBurninBtn.disabled = !canRun;

        // Speed / LUT / Duck
        el.runSpeedBtn.disabled = !canRun;
        el.runLutBtn.disabled = !canRun;
        el.runDuckBtn.disabled = !canRun;

        // Phase 6 buttons
        el.runChromaBtn.disabled = !canRun;
        el.runTransBtn.disabled = !canRun;
        el.runParticlesBtn.disabled = !canRun;
        el.runTitleOverlayBtn.disabled = !canRun;
        el.runUpscaleBtn.disabled = !canRun;
        el.runColorBtn.disabled = !canRun;
        el.runRemoveBtn.disabled = !canRun;
        el.runFaceAiBtn.disabled = !canRun;
        el.runAnimCapBtn.disabled = !canRun;

        // Export tab
        el.runExpTranscriptBtn.disabled = !canRun;

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
    // Job Execution & Tracking
    // ================================================================
    function startJob(endpoint, payload) {
        if (currentJob) {
            showAlert("A job is already running. Wait for it to finish or cancel it.");
            return;
        }
        if (!selectedPath && payload && !payload.filepath) {
            showAlert("Select a clip first.");
            return;
        }

        // Show persistent processing banner
        el.processingBanner.classList.remove("hidden");
        el.processingMsg.textContent = "Starting...";
        el.processingFill.style.width = "0%";
        el.processingElapsed.textContent = "0s";

        // Show inline progress section too
        el.progressSection.classList.remove("hidden");
        el.resultsSection.classList.add("hidden");
        el.progressBar.style.width = "0%";
        el.progressLabel.textContent = "Starting...";
        el.cancelBtn.classList.remove("hidden");

        // Lock the entire UI
        document.body.classList.add("job-active");

        // Track for retry
        lastJobEndpoint = endpoint;
        lastJobPayload = payload;

        jobStartTime = Date.now();
        elapsedTimer = setInterval(function () {
            var s = Math.floor((Date.now() - jobStartTime) / 1000);
            var timeStr = s < 60 ? s + "s" : Math.floor(s / 60) + "m " + (s % 60) + "s";
            el.progressElapsed.textContent = timeStr;
            el.processingElapsed.textContent = timeStr;
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
        var pct = (job.progress || 0) + "%";
        var msg = job.message || "Processing...";
        el.progressBar.style.width = pct;
        el.progressLabel.textContent = msg;
        // Sync to persistent banner
        el.processingFill.style.width = pct;
        el.processingMsg.textContent = msg;
    }

    function onJobDone(job) {
        currentJob = null;
        if (elapsedTimer) { clearInterval(elapsedTimer); elapsedTimer = null; }

        if (job.status === "error") {
            hideProgress();
            // Show error in results card for better visibility
            el.resultsSection.classList.remove("hidden");
            el.resultsTitle.textContent = "Error";
            el.resultsTitle.removeAttribute("style");
            el.resultsTitle.setAttribute("data-state", "error");
            el.resultsStats.innerHTML = job.error || job.message || "Unknown error";
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
                jsx('importXMLToProject("' + escPath(xmlPath) + '")', function (result) {
                    try {
                        var r = JSON.parse(result);
                        if (r.error) {
                            showAlert("Import error: " + r.error);
                        } else if (r.sequence_name) {
                            showAlert("Opened: " + r.sequence_name);
                        }
                    } catch (e) {}
                });
                lastXmlPath = xmlPath;
            }
            
            // Styled caption overlay video (.mov with alpha)
            var overlayPath = job.result.overlay_path;
            if (overlayPath) {
                jsx('importOverlayToProject("' + escPath(overlayPath) + '")', function (result) {
                    try {
                        var r = JSON.parse(result);
                        if (r.error) {
                            showAlert("Overlay import error: " + r.error);
                        } else if (r.message) {
                            showAlert(r.message);
                        }
                    } catch (e) {}
                });
                lastOverlayPath = overlayPath;
            }
            
            // Multiple output files (stem separation)
            var outputPaths = job.result.output_paths;
            if (outputPaths && outputPaths.length > 0) {
                var pathsJson = JSON.stringify(outputPaths);
                jsx('importFilesToProject(\'' + pathsJson.replace(/\\/g, "\\\\").replace(/'/g, "\\'") + '\', "OpenCut Stems")', function (result) {
                    try {
                        var r = JSON.parse(result);
                        if (r.error) {
                            showAlert("Stem import error: " + r.error);
                        } else if (r.message) {
                            showAlert(r.message);
                        }
                    } catch (e) {}
                });
            }
            
            // Single output file
            var outputPath = job.result.output_path;
            if (outputPath && !overlayPath && !xmlPath) {
                var ext = outputPath.toLowerCase().split(".").pop();
                // Caption files (SRT, VTT, ASS) - import to caption track
                if (ext === "srt" || ext === "vtt" || ext === "ass") {
                    jsx('importCaptions("' + escPath(outputPath) + '")', function (result) {
                        try {
                            var r = JSON.parse(result);
                            if (r.error) {
                                showAlert("Caption import error: " + r.error);
                            } else if (r.message) {
                                showAlert(r.message);
                            }
                        } catch (e) {}
                    });
                    lastCaptionPath = outputPath;
                }
                // Audio/video files - generic import to project
                else if (ext === "wav" || ext === "mp3" || ext === "flac" || ext === "aac" || ext === "ogg" ||
                         ext === "mp4" || ext === "mov" || ext === "avi" || ext === "mkv" || ext === "webm" || ext === "png" || ext === "jpg") {
                    jsx('importFileToProject("' + escPath(outputPath) + '", "OpenCut Output")', function (result) {
                        try {
                            var r = JSON.parse(result);
                            if (r.error) {
                                showAlert("Import error: " + r.error);
                            } else if (r.message) {
                                showAlert(r.message);
                            }
                        } catch (e) {}
                    });
                }
            }
            
            // SRT path from full pipeline (separate from output_path)
            var srtPath = job.result.srt_path;
            if (srtPath && srtPath !== outputPath) {
                jsx('importCaptions("' + escPath(srtPath) + '")', function (result) {
                    try {
                        var r = JSON.parse(result);
                        if (r.error) {
                            showAlert("Caption import error: " + r.error);
                        } else if (r.message) {
                            showAlert(r.message);
                        }
                    } catch (e) {}
                });
                lastCaptionPath = srtPath;
            }
        }
    }

    function hideProgress() {
        el.progressSection.classList.add("hidden");
        el.cancelBtn.classList.add("hidden");
        el.processingBanner.classList.add("hidden");
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
            stats += r.summary + "<br>";
        }
        if (r.segments !== undefined) {
            stats += r.segments + " segments";
        }
        if (r.filler_stats) {
            stats += " | " + r.filler_stats.removed_fillers + " fillers removed (" + r.filler_stats.total_filler_time.toFixed(1) + "s)";
        }
        if (r.caption_segments !== undefined) {
            stats += (stats ? " | " : "") + r.caption_segments + " captions, " + (r.words || 0) + " words";
        }
        if (r.style) {
            stats += " | Style: " + r.style;
        }
        // Audio results
        if (r.effect && !r.method) {
            stats += (stats ? "<br>" : "") + "Effect applied: " + r.effect;
        }
        if (r.method && r.strength !== undefined) {
            stats += (stats ? "<br>" : "") + "Denoise: " + r.method + " (" + (r.strength * 100).toFixed(0) + "% strength)";
        }
        if (r.preset && r.target_loudness !== undefined) {
            stats += (stats ? "<br>" : "") + "Normalized to " + r.target_loudness.toFixed(1) + " LUFS (" + r.preset + ")";
            if (r.input_loudness !== undefined) {
                stats += " | Was: " + r.input_loudness.toFixed(1) + " LUFS";
            }
        }
        if (r.bpm) {
            stats += (stats ? "<br>" : "") + "BPM: " + r.bpm.toFixed(0) + " | " + r.total_beats + " beats";
            if (r.confidence !== undefined) {
                stats += " | Confidence: " + (r.confidence * 100).toFixed(0) + "%";
            }
        }
        // Stem separation
        if (r.output_paths && r.output_paths.length > 0) {
            var stemNames = [];
            for (var i = 0; i < r.output_paths.length; i++) {
                var fname = r.output_paths[i].split(/[/\\]/).pop();
                stemNames.push(fname);
            }
            stats += (stats ? "<br>" : "") + r.output_paths.length + " stems: " + stemNames.join(", ");
        }
        // Scene detection
        if (r.total_scenes) {
            stats += (stats ? "<br>" : "") + "Scenes: " + r.total_scenes + " | Avg: " + r.avg_scene_length + "s";
        }

        el.resultsStats.innerHTML = stats || "Processing complete.";
        el.resultsPath.textContent = r.xml_path || r.output_path || r.overlay_path || (r.output_paths ? r.output_paths.length + " files exported" : "");
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
        api("POST", "/demucs/install", {}, function(err, data) {
            if (err || (data && data.error)) {
                el.separateHint.innerHTML = '<span style="color: var(--neon-red);">Installation failed: ' + (data ? data.error : 'Unknown error') + '</span>';
            } else {
                el.separateHint.classList.add("hidden");
                capabilities.separation = true;
                updateButtons();
                showAlert("Demucs installed successfully!");
            }
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
        api("POST", "/watermark/install", {}, function(err, data) {
            if (err || (data && data.error)) {
                el.watermarkHint.innerHTML = '<span style="color: var(--neon-red);">Installation failed: ' + (data ? data.error : 'Unknown error') + '</span>';
            } else {
                el.watermarkHint.classList.add("hidden");
                capabilities.watermark_removal = true;
                updateButtons();
                showAlert("Watermark remover installed successfully!");
            }
        });
    }
    
    function runScenes() {
        el.sceneResults.classList.add("hidden");
        startJob("/video/scenes", {
            filepath: selectedPath,
            threshold: parseFloat(el.sceneThreshold.value),
            min_scene_length: parseFloat(el.minSceneLen.value),
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
        startJob("/video/ai/install", { component: component });
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
        startJob("/audio/pro/install", { component: "pedalboard" });
    }

    function runDeepFilter() {
        startJob("/audio/pro/deepfilter", {
            filepath: selectedPath,
            output_dir: projectFolder,
        });
    }

    function installDeepFilter() {
        el.deepFilterHint.innerHTML = '<span style="color: var(--neon-cyan);">Installing DeepFilterNet...</span>';
        startJob("/audio/pro/install", { component: "deepfilter" });
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
        startJob("/video/face/install", {});
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

    function runTranslate() {
        // First transcribe, then translate
        if (!lastTranscriptSegments) {
            // Need to transcribe first - use the captions endpoint
            showAlert("Transcribing first, then translating...");
            api("POST", "/transcript", {
                filepath: selectedPath,
                model: el.translateModel.value,
            }, function (err, data) {
                // Wait for job completion - this is handled by startJob
            });
            // For now, just start the transcription job which will store segments
            startJob("/transcript", {
                filepath: selectedPath,
                model: el.translateModel.value,
            });
            return;
        }
        // We have segments, translate them
        startJob("/captions/translate", {
            filepath: selectedPath,
            segments: lastTranscriptSegments,
            source_lang: el.translateSourceLang.value,
            target_lang: el.translateTargetLang.value,
            format: el.translateFormat.value,
        });
    }

    function installNllb() {
        el.translateHint.innerHTML = '<span style="color: var(--neon-cyan);">Installing NLLB translation...</span>';
        startJob("/captions/enhanced/install", { component: "nllb" });
    }

    // --- KARAOKE CAPTIONS ---
    function runKaraoke() {
        startJob("/captions/whisperx", {
            filepath: selectedPath,
            model: el.karaokeModel.value,
            diarize: el.karaokeDiarize.checked,
        });
    }

    function installWhisperx() {
        el.karaokeHint.innerHTML = '<span style="color: var(--neon-cyan);">Installing WhisperX...</span>';
        startJob("/captions/enhanced/install", { component: "whisperx" });
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
        // Collect all clip paths from the clip selector
        var paths = [];
        if (el.clipSelect && el.clipSelect.options) {
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
            // Poll for status
            var pollInterval = setInterval(function () {
                api("GET", "/batch/" + batchId, null, function (e2, d2) {
                    if (e2 || !d2) return;
                    el.batchStatusText.textContent =
                        "Batch " + d2.status + ": " + d2.completed + "/" + d2.total +
                        " (" + d2.results.success + " ok, " + d2.results.failed + " failed)";
                    if (d2.status !== "running") {
                        clearInterval(pollInterval);
                        showAlert("Batch complete: " + d2.results.success + " succeeded");
                    }
                });
            }, 2000);
        });
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
        });
    }

    function installEdgeTts() {
        el.ttsHint.innerHTML = '<span style="color: var(--neon-cyan);">Installing Edge TTS...</span>';
        startJob("/audio/tts/install", { component: "edge_tts" });
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
            });
        } else {
            startJob("/audio/gen/sfx", {
                preset: el.sfxPreset.value,
                duration: parseFloat(el.sfxDuration.value),
                output_dir: projectFolder,
            });
        }
    }

    // --- CAPTION BURN-IN ---
    function runBurnin() {
        startJob("/transcript", {
            filepath: selectedPath,
            model: el.burninModel.value,
        });
        showAlert("Transcribing first, then burn-in will be available. Use the segments from the transcript to burn in captions.");
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
        startJob("/video/title/render", { text: t, output_dir: projectFolder,
            preset: el.titlePreset.value, duration: parseFloat(el.titleDur.value),
            font_size: parseInt(el.titleFontSize.value), subtitle: el.titleSubtext.value.trim() });
    }

    // --- PRO UPSCALE ---
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
        startJob("/transcript", {
            filepath: selectedPath,
            model: el.animCapModel.value,
            word_level: true,
        });
        showAlert("Transcribing with word-level timing first. Once complete, animated captions will render.");
    }

    // --- AI MUSIC GENERATION ---
    function runMusicAi() {
        var prompt = el.musicAiPrompt.value.trim();
        if (!prompt) { showAlert("Enter a music prompt."); return; }
        startJob("/audio/music-ai/generate", { prompt: prompt, output_dir: projectFolder,
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
    // Extended Job Result Handling
    // ================================================================
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
                el.gpuVram.textContent = data.available ? (data.vram_mb / 1024).toFixed(1) + " GB" : "--";
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
        startJob("/whisper/reinstall", { backend: "faster-whisper", cpu_mode: cpuMode });
    }

    function clearWhisperCache() {
        showAlert("Clearing Whisper cache...");
        api("POST", "/whisper/clear-cache", {}, function (err, data) {
            if (!err && data) {
                if (data.success) {
                    showAlert("Cache cleared! Cleared " + data.cleared.length + " location(s). Models will re-download on next use.");
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
        el.sceneThreshold.addEventListener("input", function () { el.sceneThresholdVal.textContent = parseFloat(this.value).toFixed(2); });
        el.minSceneLen.addEventListener("input", function () { el.minSceneLenVal.textContent = this.value + "s"; });

        // Video FX sliders
        el.vfxStabSmoothing.addEventListener("input", function () { el.vfxStabSmoothingVal.textContent = this.value; });
        el.vfxStabZoom.addEventListener("input", function () { el.vfxStabZoomVal.textContent = this.value + "%"; });
        el.vfxVignetteIntensity.addEventListener("input", function () { el.vfxVignetteIntensityVal.textContent = this.value; });
        el.vfxGrainIntensity.addEventListener("input", function () { el.vfxGrainIntensityVal.textContent = this.value; });
        el.vfxChromakeySim.addEventListener("input", function () { el.vfxChromakeySimVal.textContent = parseFloat(this.value).toFixed(2); });
        el.vfxChromakeyBlend.addEventListener("input", function () { el.vfxChromakeyBlendVal.textContent = parseFloat(this.value).toFixed(2); });
        el.vfxLutIntensity.addEventListener("input", function () { el.vfxLutIntensityVal.textContent = this.value; });

        // Video AI sliders
        el.vidAiDenoiseStrength.addEventListener("input", function () { el.vidAiDenoiseStrengthVal.textContent = this.value; });

        // Face blur slider
        el.faceBlurStrength.addEventListener("input", function () { el.faceBlurStrengthVal.textContent = this.value; });

        // Style transfer slider
        el.styleIntensity.addEventListener("input", function () { el.styleIntensityVal.textContent = this.value; });

        // Karaoke font size slider
        el.karaokeFontSize.addEventListener("input", function () { el.karaokeFontSizeVal.textContent = this.value; });

        // TTS rate slider
        el.ttsRate.addEventListener("input", function () {
            var v = parseInt(this.value);
            el.ttsRateVal.textContent = (v >= 0 ? "+" : "") + v + "%";
        });

        // SFX sliders
        el.toneFreq.addEventListener("input", function () { el.toneFreqVal.textContent = this.value; });
        el.sfxDuration.addEventListener("input", function () { el.sfxDurationVal.textContent = this.value; });

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
        el.transDur.addEventListener("input", function () { el.transDurVal.textContent = this.value; });
        el.particleDensity.addEventListener("input", function () { el.particleDensityVal.textContent = this.value; });
        el.titleDur.addEventListener("input", function () { el.titleDurVal.textContent = this.value; });
        el.titleFontSize.addEventListener("input", function () { el.titleFontSizeVal.textContent = this.value; });
        el.ccExposure.addEventListener("input", function () { el.ccExposureVal.textContent = this.value; });
        el.ccContrast.addEventListener("input", function () { el.ccContrastVal.textContent = this.value; });
        el.ccSaturation.addEventListener("input", function () { el.ccSaturationVal.textContent = this.value; });
        el.ccTemp.addEventListener("input", function () { el.ccTempVal.textContent = this.value; });
        el.ccShadows.addEventListener("input", function () { el.ccShadowsVal.textContent = this.value; });
        el.ccHighlights.addEventListener("input", function () { el.ccHighlightsVal.textContent = this.value; });
        el.animCapFontSize.addEventListener("input", function () { el.animCapFontSizeVal.textContent = this.value; });
        el.animCapWpl.addEventListener("input", function () { el.animCapWplVal.textContent = this.value; });
        el.musicAiDur.addEventListener("input", function () { el.musicAiDurVal.textContent = this.value; });
        el.musicAiTemp.addEventListener("input", function () { el.musicAiTempVal.textContent = this.value; });
    }

    // ================================================================
    // Refresh & Retry
    // ================================================================
    function refreshAll() {
        el.refreshAllBtn.classList.add("spinning");
        settingsLoaded = false;
        checkHealth();
        scanProjectMedia();
        loadStylePreview();
        setTimeout(function () {
            el.refreshAllBtn.classList.remove("spinning");
            showAlert("Refreshed");
        }, 1200);
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
    
    // Escape for use inside JSX string arguments (handles Windows paths)
    function escPath(s) {
        if (!s) return "";
        // Double backslashes for JavaScript string, then escape quotes
        return s.replace(/\\/g, "\\\\").replace(/"/g, '\\"');
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
        initCustomDropdowns(); // Initialize custom in-panel dropdowns

        // Event listeners - Clip selection
        el.refreshAllBtn.addEventListener("click", refreshAll);
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
                navigator.clipboard.writeText(text).then(function () { showAlert("Copied to clipboard!"); });
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
        el.runUpscaleBtn.addEventListener("click", runUpscale);
        el.runColorBtn.addEventListener("click", runColor);
        el.runRemoveBtn.addEventListener("click", runRemove);
        el.runFaceAiBtn.addEventListener("click", runFaceAi);
        el.faceAiMode.addEventListener("change", showFaceAiParams);
        el.runAnimCapBtn.addEventListener("click", runAnimCap);
        el.runMusicAiBtn.addEventListener("click", runMusicAi);

        // Export tab buttons
        el.runExpTranscriptBtn.addEventListener("click", runExpTranscript);

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

        // Load pedalboard effects list
        loadPedalboardEffects();

        // Init VFX param visibility
        showVfxParams();

        // Load export presets
        loadExportPresets();
    });

})();
