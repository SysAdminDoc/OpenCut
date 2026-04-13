/* ============================================================
   OpenCut CEP Panel - Main Controller v1.9.34
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
    var _updateCheckDone = false;
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
    var mediaScanTimer = null;
    var transcriptData = null; // stored transcript for editing/export
    var lastJobEndpoint = "";  // for retry
    var lastJobPayload = null; // for retry
    var jobLifecycleHandlers = {};
    var _utilityJobSeq = 0;
    var _waveformRequestSeq = 0;
    var _previewModalRequestSeq = 0;
    var _clipThumbRequestSeq = 0;

    // Hoist timers referenced by cleanupTimers() — strict mode would throw
    // ReferenceError if these were only declared inside their nested helpers.
    var _statusTimer = null;
    var _scanDebounceTimer = null;
    var _projectMediaRetryTimer = null;
    var editDebounceTimer = null;
    var _alertTimer = null;
    var _wsReconnectTimer = null;

    // ---- Centralized Timer Cleanup ----
    function cleanupTimers() {
        if (healthTimer) { clearInterval(healthTimer); healthTimer = null; }
        if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
        if (elapsedTimer) { clearInterval(elapsedTimer); elapsedTimer = null; }
        if (batchPollTimer) { clearInterval(batchPollTimer); batchPollTimer = null; }
        if (mediaScanTimer) { clearInterval(mediaScanTimer); mediaScanTimer = null; }
        if (_statusTimer) { clearInterval(_statusTimer); _statusTimer = null; }
        if (_scanDebounceTimer) { clearTimeout(_scanDebounceTimer); _scanDebounceTimer = null; }
        if (_projectMediaRetryTimer) { clearTimeout(_projectMediaRetryTimer); _projectMediaRetryTimer = null; }
        if (editDebounceTimer) { clearTimeout(editDebounceTimer); editDebounceTimer = null; }
        if (_alertTimer) { clearTimeout(_alertTimer); _alertTimer = null; }
        if (_wsReconnectTimer) { clearTimeout(_wsReconnectTimer); _wsReconnectTimer = null; }
    }

    // ---- Background poller restart hook ----
    //
    // cleanupTimers() nukes mediaScanTimer / _statusTimer on every disconnect.
    // Without this hook only the initial init path would restart them, so
    // after the first disconnect/reconnect cycle the media scan and system
    // status bar would stop polling forever. Called from checkHealth() on
    // every reconnect and from the DOMContentLoaded bootstrap.
    var MEDIA_POLL_MS = 20000; // 20 seconds
    function startBackgroundPollers() {
        if (!mediaScanTimer) {
            mediaScanTimer = setInterval(function () {
                if (!currentJob && (inPremiere || connected)) {
                    scanProjectMedia();
                }
            }, MEDIA_POLL_MS);
        }
        // initStatusBar() is idempotent — it guards against duplicate timers
        // via its own _statusTimer check, so it is safe to call on reconnect.
        if (typeof initStatusBar === "function" && !_statusTimer) {
            initStatusBar();
        }
    }

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

    // ---- Premiere Pro state cache (reduces evalScript round-trips) ----
    var _pproCache = {
        seq: null,
        seqTs: 0,
        clips: null,
        clipsTs: 0,
        bins: null,
        binsTs: 0,
        ttl: 8000
    };

    // ---- Keyboard Shortcut Registry (Phase 3.4) ----
    var DEFAULT_SHORTCUTS = {
        "silence-detect": { keys: "Ctrl+Shift+S", label: "Detect Silence" },
        "caption-generate": { keys: "Ctrl+Shift+C", label: "Generate Captions" },
        "audio-normalize": { keys: "Ctrl+Shift+N", label: "Normalize Audio" },
        "audio-denoise": { keys: "Ctrl+Shift+D", label: "Denoise Audio" },
        "export-video": { keys: "Ctrl+Shift+E", label: "Export Video" },
        "command-palette": { keys: "Ctrl+K", label: "Command Palette" },
        "cancel-job": { keys: "Escape", label: "Cancel Current Job" },
        "quick-workflow": { keys: "Ctrl+Shift+W", label: "Run Quick Workflow" }
    };
    var _shortcutRegistry = {};
    var WORKSPACE_STATE_KEY = "opencut_workspace_state";
    var _workspaceState = loadWorkspaceState();

    function loadShortcuts() {
        var saved = {};
        try {
            var raw = localStorage.getItem("opencut_shortcuts");
            if (raw) saved = JSON.parse(raw);
        } catch (e) {}
        _shortcutRegistry = {};
        var id;
        for (id in DEFAULT_SHORTCUTS) {
            if (DEFAULT_SHORTCUTS.hasOwnProperty(id)) {
                _shortcutRegistry[id] = {
                    keys: (saved[id] && saved[id].keys) ? saved[id].keys : DEFAULT_SHORTCUTS[id].keys,
                    label: DEFAULT_SHORTCUTS[id].label
                };
            }
        }
        return _shortcutRegistry;
    }

    function saveShortcuts() {
        var toSave = {};
        var id;
        for (id in _shortcutRegistry) {
            if (_shortcutRegistry.hasOwnProperty(id) && DEFAULT_SHORTCUTS[id] &&
                _shortcutRegistry[id].keys !== DEFAULT_SHORTCUTS[id].keys) {
                toSave[id] = { keys: _shortcutRegistry[id].keys };
            }
        }
        try { localStorage.setItem("opencut_shortcuts", JSON.stringify(toSave)); } catch (e) {}
    }

    function updateShortcut(actionId, newKeys) {
        if (_shortcutRegistry[actionId]) {
            _shortcutRegistry[actionId].keys = newKeys;
            saveShortcuts();
        }
    }

    function normalizeWorkspaceState(saved) {
        return {
            activeNav: saved && typeof saved.activeNav === "string" ? saved.activeNav : "cut",
            activeSubs: saved && saved.activeSubs && typeof saved.activeSubs === "object" ? saved.activeSubs : {},
            selectedPath: saved && typeof saved.selectedPath === "string" ? saved.selectedPath : "",
            selectedName: saved && typeof saved.selectedName === "string" ? saved.selectedName : ""
        };
    }

    function loadWorkspaceState() {
        try {
            var saved = localStorage.getItem(WORKSPACE_STATE_KEY);
            if (saved) return normalizeWorkspaceState(JSON.parse(saved));
        } catch (e) {}
        return normalizeWorkspaceState({});
    }

    function persistWorkspaceState() {
        try {
            localStorage.setItem(WORKSPACE_STATE_KEY, JSON.stringify(_workspaceState));
        } catch (e) {}
    }

    function rememberWorkspaceTab(tabName) {
        _workspaceState.activeNav = tabName || "cut";
        persistWorkspaceState();
    }

    function rememberWorkspaceSub(tabName, subName) {
        if (!_workspaceState.activeSubs || typeof _workspaceState.activeSubs !== "object") {
            _workspaceState.activeSubs = {};
        }
        if (tabName && subName) _workspaceState.activeSubs[tabName] = subName;
        persistWorkspaceState();
    }

    function rememberWorkspaceSelection(path, name) {
        _workspaceState.selectedPath = path || "";
        _workspaceState.selectedName = name || "";
        persistWorkspaceState();
    }

    function findSelectOptionByValue(select, value) {
        if (!select || !value) return null;
        for (var i = 0; i < select.options.length; i++) {
            if (select.options[i].value === value) return select.options[i];
        }
        return null;
    }

    function syncClipSelectValue(path) {
        if (!el.clipSelect) return null;
        var option = path ? findSelectOptionByValue(el.clipSelect, path) : null;
        el.clipSelect.value = option ? path : "";
        if (el.clipSelect._customDropdown) el.clipSelect._customDropdown.updateText();
        return option;
    }

    function getPanelTabName(panel) {
        if (!panel || !panel.id) return "";
        return panel.id.indexOf("panel-") === 0 ? panel.id.substring(6) : panel.id;
    }

    function getVisibleTabButtons(container, selector) {
        var buttons = container ? container.querySelectorAll(selector) : [];
        var visible = [];
        for (var i = 0; i < buttons.length; i++) {
            if (window.getComputedStyle(buttons[i]).display !== "none") visible.push(buttons[i]);
        }
        return visible;
    }

    function moveFocusAndActivate(buttons, currentButton, direction) {
        if (!buttons || !buttons.length || !currentButton) return;
        var currentIndex = buttons.indexOf(currentButton);
        if (currentIndex === -1) currentIndex = 0;
        var nextIndex = currentIndex + direction;
        if (nextIndex < 0) nextIndex = buttons.length - 1;
        if (nextIndex >= buttons.length) nextIndex = 0;
        buttons[nextIndex].focus();
        buttons[nextIndex].click();
    }

    function matchesShortcut(e, keysStr) {
        var parts = keysStr.split("+");
        var needCtrl = false, needShift = false, needAlt = false, needMeta = false;
        var keyPart = "";
        for (var i = 0; i < parts.length; i++) {
            var p = parts[i].trim().toLowerCase();
            if (p === "ctrl") needCtrl = true;
            else if (p === "shift") needShift = true;
            else if (p === "alt") needAlt = true;
            else if (p === "meta" || p === "cmd") needMeta = true;
            else keyPart = p;
        }
        if (e.ctrlKey !== needCtrl) return false;
        if (e.shiftKey !== needShift) return false;
        if (e.altKey !== needAlt) return false;
        if (e.metaKey !== needMeta) return false;
        var eventKey = e.key.toLowerCase();
        if (keyPart === "escape") return eventKey === "escape";
        return eventKey === keyPart;
    }

    var _shortcutActions = {
        "silence-detect": function () { if (el.runSilenceBtn && !el.runSilenceBtn.disabled) el.runSilenceBtn.click(); },
        "caption-generate": function () { if (el.runStyledCaptionsBtn && !el.runStyledCaptionsBtn.disabled) el.runStyledCaptionsBtn.click(); },
        "audio-normalize": function () { if (el.runNormalizeBtn && !el.runNormalizeBtn.disabled) el.runNormalizeBtn.click(); },
        "audio-denoise": function () { if (el.runDenoiseBtn && !el.runDenoiseBtn.disabled) el.runDenoiseBtn.click(); },
        "export-video": function () { if (el.runExportPresetBtn && !el.runExportPresetBtn.disabled) el.runExportPresetBtn.click(); },
        "command-palette": function () { openCommandPalette(); },
        "cancel-job": function () { if (currentJob) cancelJob(); },
        "quick-workflow": function () { if (el.runWorkflowBtn && !el.runWorkflowBtn.disabled) el.runWorkflowBtn.click(); }
    };

    // ---- Lazy Tab Rendering (Phase 5.1) ----
    var _tabRendered = {};

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
        if (!select.parentNode) return;
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
        hideRecentClipsDropdown(false);
        closeCommandPalette({ restoreFocus: false });
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
    function setExpanded(node, expanded) {
        if (node) node.setAttribute("aria-expanded", expanded ? "true" : "false");
    }

    function focusClipPicker(openDropdown) {
        if (!el.clipSelect) return;
        var trigger = null;
        if (el.clipSelect.parentNode) {
            trigger = el.clipSelect.parentNode.querySelector(".custom-dropdown-trigger");
        }
        if (trigger) {
            trigger.focus();
            if (openDropdown) trigger.click();
            return;
        }
        try { el.clipSelect.focus(); } catch (e) {}
    }

    function initDOM() {
        // Content header
        el.contentTitle = $("contentTitle");
        el.contentSubtitle = $("contentSubtitle");
        el.workspaceClipStatus = $("workspaceClipStatus");
        el.connDot = $("connDot");
        el.connStatus = $("connStatus");
        el.connLabel = $("connLabel");
        el.refreshAllBtn = $("refreshAllBtn");
    el.alertBanner = $("alertBanner");
    el.alertIcon = $("alertIcon");
    el.alertEyebrow = $("alertEyebrow");
    el.alertText = $("alertText");
    el.alertDismiss = $("alertDismiss");
    el.sessionContext = $("sessionContext");
    el.sessionContextBody = $("sessionContextBody");
    el.sessionContextDismiss = $("sessionContextDismiss");
    el.journalList = $("journalList");
    el.journalRefreshBtn = $("journalRefreshBtn");
    el.journalClearBtn = $("journalClearBtn");
    el.polishInterviewBtn = $("polishInterviewBtn");
    el.polishBatchBtn = $("polishBatchBtn");
    el.polishBatchCount = $("polishBatchCount");
    el.polishSteps = $("polishSteps");
    el.polishResult = $("polishResult");
    el.interviewPolishHint = $("interviewPolishHint");
    el.polishDetectRepeats = $("polishDetectRepeats");
    el.polishRemoveFillers = $("polishRemoveFillers");
    el.polishGenerateChapters = $("polishGenerateChapters");

        // Clip
        el.clipSelect = $("clipSelect");
        el.fileInfoBox = $("fileInfoBox");
        el.fileNameDisplay = $("fileNameDisplay");
        el.fileMetaDisplay = $("fileMetaDisplay");
        el.refreshClipsBtn = $("refreshClipsBtn");
        el.useSelectionBtn = $("useSelectionBtn");
        el.browseFileBtn = $("browseFileBtn");
        el.stageChooseMediaBtn = $("stageChooseMediaBtn");
        el.stageUseTimelineBtn = $("stageUseTimelineBtn");
        el.stageBrowseMediaBtn = $("stageBrowseMediaBtn");
        el.stageCommandPaletteBtn = $("stageCommandPaletteBtn");
        el.workspaceStageSource = $("workspaceStageSource");
        el.workspaceStageSuite = $("workspaceStageSuite");
        el.workspaceStageStatus = $("workspaceStageStatus");
        el.contextGuidanceBanner = $("contextGuidanceBanner");
        el.contextGuidanceText = $("contextGuidanceText");

        // Cut tab
        el.silencePreset = $("silencePreset");
        el.customSilenceSettings = $("customSilenceSettings");
        el.threshold = $("threshold");
        el.thresholdVal = $("thresholdVal");
        el.minDuration = $("minDuration");
        el.minDurationVal = $("minDurationVal");
        el.padBefore = $("padBefore");
        el.padAfter = $("padAfter");
        el.silenceDetectMethod = $("silenceDetectMethod");
        el.silenceMode = $("silenceMode");
        el.silenceSpeedGroup = $("silenceSpeedGroup");
        el.silenceSpeedFactor = $("silenceSpeedFactor");
        el.runSilenceBtn = $("runSilenceBtn");
        el.fillerModel = $("fillerModel");
        el.fillerBackend = $("fillerBackend");
        el.fillerChecks = $("fillerChecks");
        el.fillerCustom = $("fillerCustom");
        el.fillerSilence = $("fillerSilence");
        el.fillersHint = $("fillersHint");
        el.installCrisperWhisperBtn = $("installCrisperWhisperBtn");
        el.runFillersBtn = $("runFillersBtn");
        el.vadHint = $("vadHint");
        el.fullPreset = $("fullPreset");
        el.fullZoom = $("fullZoom");
        el.fullCaptions = $("fullCaptions");
        el.fullFillers = $("fullFillers");
        el.runFullBtn = $("runFullBtn");
        el.autoEditMethod = $("autoEditMethod");
        el.autoEditThreshold = $("autoEditThreshold");
        el.autoEditMargin = $("autoEditMargin");
        el.autoEditMinClip = $("autoEditMinClip");
        el.runAutoEditBtn = $("runAutoEditBtn");
        el.highlightMax = $("highlightMax");
        el.highlightMinDur = $("highlightMinDur");
        el.highlightMaxDur = $("highlightMaxDur");
        el.runHighlightsBtn = $("runHighlightsBtn");
        el.runEmotionHighlightsBtn = $("runEmotionHighlightsBtn");
        el.emotionHint = $("emotionHint");
        el.installEmotionBtn = $("installEmotionBtn");
        el.otioHint = $("otioHint");
        el.brollGenHint = $("brollGenHint");
        el.installBrollGenBtn = $("installBrollGenBtn");
        el.mmDiarizeHint = $("mmDiarizeHint");
        el.installMmDiarizeBtn = $("installMmDiarizeBtn");
        el.socialHint = $("socialHint");

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
    el.transcriptTimeline = $("transcriptTimeline");
    el.transcriptTimelineRuler = $("transcriptTimelineRuler");
    el.transcriptTimelineStatus = $("transcriptTimelineStatus");
    el.transcriptTimelineMeta = $("transcriptTimelineMeta");
    el.transcriptSegments = $("transcriptSegments");
        el.transcriptExportFormat = $("transcriptExportFormat");
        el.exportTranscriptBtn = $("exportTranscriptBtn");
        el.transcriptUndoBtn = $("transcriptUndoBtn");
        el.transcriptRedoBtn = $("transcriptRedoBtn");
        el.summarizeTranscriptBtn = $("summarizeTranscriptBtn");
        el.summaryResult = $("summaryResult");
        el.summaryContent = $("summaryContent");
        el.copySummaryBtn = $("copySummaryBtn");

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
        el.runDepthBtn = $("runDepthBtn");
        el.depthHint = $("depthHint");
        el.installDepthBtn = $("installDepthBtn");
        el.runBrollGenBtn = $("runBrollGenBtn");
        el.runMmDiarizeBtn = $("runMmDiarizeBtn");
        
        // Video tab - Scene Detection
        el.sceneThreshold = $("sceneThreshold");
        el.sceneThresholdVal = $("sceneThresholdVal");
        el.minSceneLen = $("minSceneLen");
        el.minSceneLenVal = $("minSceneLenVal");
        el.sceneMethod = $("sceneMethod");
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
        el.vidAiRembgBackend = $("vidAiRembgBackend");
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
        el.enhanceDenoise = $("enhanceDenoise");
        el.enhanceUpscale = $("enhanceUpscale");
        el.runEnhanceBtn = $("runEnhanceBtn");

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
        el.workflowPresetDesc = $("workflowPresetDesc");
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
        el.lutRefPath = $("lutRefPath");
        el.lutRefName = $("lutRefName");
        el.lutRefStrength = $("lutRefStrength");
        el.generateLutBtn = $("generateLutBtn");
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
        el.reframeFaceSmoothing = $("reframeFaceSmoothing");
        el.faceSmoothing = $("faceSmoothing");
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
        el.shortsPlatform = $("shortsPlatform");
        el.shortsMaxClips = $("shortsMaxClips");
        el.shortsMinDur = $("shortsMinDur");
        el.shortsMaxDur = $("shortsMaxDur");
        el.shortsFaceTrack = $("shortsFaceTrack");
        el.shortsCaptions = $("shortsCaptions");
        el.runShortsBtn = $("runShortsBtn");
        el.loadSeqInfoBtn = $("loadSeqInfoBtn");
        el.genVfxSheetBtn = $("genVfxSheetBtn");
        el.genAdrListBtn = $("genAdrListBtn");
        el.genMusicCueBtn = $("genMusicCueBtn");
        el.genAssetListBtn = $("genAssetListBtn");
        el.getSeqMarkersBtn = $("getSeqMarkersBtn");
        el.exportMarkedClipsBtn = $("exportMarkedClipsBtn");
        el.loadProjectItemsBtn = $("loadProjectItemsBtn");
        el.applyRenamePatternBtn = $("applyRenamePatternBtn");
        el.renameAllBtn = $("renameAllBtn");
        el.createSmartBinsBtn = $("createSmartBinsBtn");
        el.runSrtImportBtn = $("runSrtImportBtn");
        el.indexAllClipsBtn = $("indexAllClipsBtn");

        // Settings tab
        el.whisperStatusText = $("whisperStatusText");
        el.whisperDeviceText = $("whisperDeviceText");
        el.whisperCpuMode = $("whisperCpuMode");
        el.settingsDefaultModel = $("settingsDefaultModel");
        el.settingsInstallWhisperBtn = $("settingsInstallWhisperBtn");
        el.settingsReinstallWhisperBtn = $("settingsReinstallWhisperBtn");
        el.settingsClearCacheBtn = $("settingsClearCacheBtn");
        el.settingsAutoImport = $("settingsAutoImport");
        el.settingsAutoOpen = $("settingsAutoOpen");
        el.settingsShowNotifications = $("settingsShowNotifications");
        el.settingsOutputDir = $("settingsOutputDir");
        el.restartBackendBtn = $("restartBackendBtn");
        el.openLogsBtn = $("openLogsBtn");
        el.gpuName = $("gpuName");
        el.gpuVram = $("gpuVram");
        el.backendPort = $("backendPort");
        el.testLLMBtn = $("testLLMBtn");
        el.llmProvider = $("llmProvider");
        el.llmModel = $("llmModel");
        el.llmApiKeyGroup = $("llmApiKeyGroup");
        el.llmApiKey = $("llmApiKey");
        el.llmBaseUrl = $("llmBaseUrl");
        el.llmStatus = $("llmStatus");

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
        el.commandPaletteStatus = $("commandPaletteStatus");

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
        el.statusBar = $("statusBar");
        el.statusDot = $("statusDot");
        el.statusText = $("statusText");
        el.statusGpu = $("statusGpu");
        el.statusJobs = $("statusJobs");
        el.serverStatusBanner = $("serverStatusBanner");
        el.serverStatusMsg = $("serverStatusMsg");
        if (el.connStatus) el.connStatus.setAttribute("data-state", "offline");
        if (el.statusBar) el.statusBar.setAttribute("data-state", "offline");
    }

    function setConnectionBadge(state, label) {
        if (el.connLabel) el.connLabel.textContent = label;
        if (el.connStatus) {
            el.connStatus.setAttribute("data-state", state);
            el.connStatus.title = label;
        }
        if (el.connDot) {
            el.connDot.className = state === "online" ? "conn-dot on" : "conn-dot off";
            el.connDot.setAttribute("aria-label", state === "online" ? "Server connected" : "Server disconnected");
        }
        updateWorkspaceStageSession();
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
            cs.evalScript('autoImportResult("' + escPath(path) + '", "' + escPath(type || "output") + '")');
        },
        isProjectSaved: function (cb) {
            jsx("isProjectSaved()", cb);
        },
        // Journal inverse helpers (v1.9.28)
        removeSequenceMarkers: function (payload, cb) {
            var json = JSON.stringify(payload);
            jsx("ocRemoveSequenceMarkers('" + json.replace(/\\/g, "\\\\").replace(/'/g, "\\'") + "')", cb);
        },
        unrenameItems: function (payload, cb) {
            var json = JSON.stringify(payload);
            jsx("ocUnrenameItems('" + json.replace(/\\/g, "\\\\").replace(/'/g, "\\'") + "')", cb);
        },
        removeImportedSequence: function (payload, cb) {
            var json = JSON.stringify(payload);
            jsx("ocRemoveImportedSequence('" + json.replace(/\\/g, "\\\\").replace(/'/g, "\\'") + "')", cb);
        },
        removeImportedItem: function (payload, cb) {
            var json = JSON.stringify(payload);
            jsx("ocRemoveImportedItem('" + json.replace(/\\/g, "\\\\").replace(/'/g, "\\'") + "')", cb);
        },
        setPlayhead: function (seconds, cb) {
            jsx("ocSetSequencePlayhead(" + Number(seconds || 0) + ")", cb);
        }
    };

    // ================================================================
    // Operation Journal (v1.9.28) — frontend record + rollback
    // ================================================================
    // Every ExtendScript operation that mutates the Premiere project calls
    // journalRecord() after success. The Journal sub-tab under Settings
    // renders the history and dispatches inverse calls via PremiereBridge.
    function journalRecord(action, label, inversePayload, clipPath) {
        if (!connected) return;
        api("POST", "/journal/record", {
            action: action,
            label: label || "",
            clip_path: clipPath || "",
            inverse: inversePayload || {}
        }, function (err, entry) {
            if (err) {
                // Never surface journal-record failures to the user — the
                // forward operation already succeeded.
                try { console.warn("journal record failed:", err); } catch (_) {}
                return;
            }
            // Nudge the journal tab if it's currently visible.
            if (typeof renderJournalList === "function" && el.journalList &&
                el.journalList.offsetParent !== null) {
                renderJournalList();
            }
        });
    }

    // Helpers to extract the label/inverse from a forward-operation result.
    function _journalLabelForMarkers(count, clipName) {
        var n = count | 0;
        return n + " marker" + (n === 1 ? "" : "s") +
               (clipName ? " on '" + clipName + "'" : "");
    }
    function _journalLabelForRename(count) {
        var n = count | 0;
        return "Renamed " + n + " project item" + (n === 1 ? "" : "s");
    }

    // ================================================================
    // Backend Communication
    // ================================================================
    var _inflightRequests = {};
    function api(method, path, body, callback, timeout) {
        callback = typeof callback === "function" ? callback : function () {};
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
            // Treat non-2xx HTTP responses as errors
            if (!err && xhr.status >= 400) {
                err = new Error((data && data.error) ? data.error : "HTTP " + xhr.status);
                err.status = xhr.status;
            }
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

    function getButtonLabelNode(btn) {
        if (!btn || typeof btn.querySelector !== "function") return null;
        return btn.querySelector(".btn-label");
    }

    function getButtonText(btn) {
        if (!btn) return "";
        var labelNode = getButtonLabelNode(btn);
        return labelNode ? labelNode.textContent : btn.textContent;
    }

    function rememberButtonText(btn) {
        if (!btn) return "";
        var cached = btn.getAttribute("data-oc-default-text");
        if (cached !== null) return cached;
        var text = getButtonText(btn);
        btn.setAttribute("data-oc-default-text", text);
        return text;
    }

    function setButtonText(btn, text) {
        if (!btn) return;
        var labelNode = getButtonLabelNode(btn);
        if (labelNode) labelNode.textContent = text;
        else btn.textContent = text;
    }

    // Wrapper: api call with button spinner feedback
    function apiWithSpinner(btn, method, path, body, callback, timeout) {
        var origText = rememberButtonText(btn);
        btn.disabled = true;
        setButtonText(btn, "Working…");
        api(method, path, body, function (err, data) {
            btn.disabled = false;
            setButtonText(btn, origText);
            callback(err, data);
        }, timeout);
    }

    function formatInstallError(detail, fallback) {
        if (!detail) return fallback || "Unknown error";
        if (typeof detail === "string") return detail;

        var result = detail.result && typeof detail.result === "object" ? detail.result : null;
        var error = "";
        var suggestion = "";

        if (result) {
            error = result.error || result.message || "";
            suggestion = result.suggestion || "";
        }
        if (!error) error = detail.error || "";
        if (!suggestion) suggestion = detail.suggestion || "";
        if (!error && detail.status === "cancelled") error = "Cancelled";
        if (!error && detail.message) error = detail.message;
        if (!error && detail.code) error = detail.code;

        if (!error) return fallback || "Unknown error";
        return suggestion ? error + " — " + suggestion : error;
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
                setConnectionBadge("online", "Connected");
                if (data.csrf_token) csrfToken = data.csrf_token;
                if (data.capabilities) capabilities = data.capabilities;
                el.backendPort.textContent = BACKEND.replace("http://127.0.0.1:", "Port ");
                updateButtons();
                loadCapabilities();
                // Restart media scan + status bar pollers after a reconnect.
                // cleanupTimers() killed them on the preceding disconnect and
                // without this they would never come back until full reload.
                startBackgroundPollers();
                // Auto-connect WebSocket if available
                if (!_wsConnected && capabilities.websocket !== false) {
                    wsConnect();
                }
                // One-time checks after server connects
                if (!_updateCheckDone) {
                    _updateCheckDone = true;
                    api("GET", "/system/update-check", null, function (uerr, udata) {
                        if (!uerr && udata && udata.update_available) {
                            showToast("OpenCut v" + udata.latest_version + " available \u2014 visit GitHub to update", "info");
                        }
                    });
                    // Surface last-session history + interrupted jobs in a
                    // dedicated "Welcome back" card (replaces the older alert).
                    showSessionContext();
                }
                return;
            }
            if (connected && el.serverStatusBanner) {
                el.serverStatusBanner.classList.remove("hidden");
        if (el.serverStatusMsg) el.serverStatusMsg.textContent = "Server disconnected. Reconnecting…";
            }
            connected = false;
            // Clean up all active timers on disconnect
            cleanupTimers();
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
                            setConnectionBadge("online", "Connected" + (port !== BACKEND_BASE_PORT ? " (:" + port + ")" : ""));
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
            setConnectionBadge("offline", "Disconnected");
            // Clear capability cache so buttons get re-evaluated when server returns
            capabilitiesLoaded = false;
            capabilities = {};
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
    var _scanInProgress = false;
    var _scanDebounceTimer = null;
    var _projectMediaRetryTimer = null;
    var _projectMediaRetryCount = 0;
    var _projectMediaRetryNoticeShown = false;
    var PROJECT_MEDIA_RETRY_DELAYS = [800, 1600, 3200, 5000];

    function scanProjectMedia() {
        // Debounce: if multiple scan triggers fire in quick succession, only run once
        if (_scanDebounceTimer) clearTimeout(_scanDebounceTimer);
        _scanDebounceTimer = setTimeout(_doScanProjectMedia, 300);
    }

    function setProjectMediaPlaceholder(message) {
        if (!el.clipSelect || (projectMedia && projectMedia.length)) return;
        el.clipSelect.innerHTML = "";
        var placeholder = document.createElement("option");
        placeholder.value = "";
        placeholder.selected = true;
        placeholder.textContent = message;
        el.clipSelect.appendChild(placeholder);
        refreshClipDropdown();
    }

    function resetProjectMediaRetryState() {
        if (_projectMediaRetryTimer) {
            clearTimeout(_projectMediaRetryTimer);
            _projectMediaRetryTimer = null;
        }
        _projectMediaRetryCount = 0;
        _projectMediaRetryNoticeShown = false;
    }

    function scheduleProjectMediaRetry(reason) {
        if (!inPremiere) return false;
        if (_projectMediaRetryTimer) return true;
        if (_projectMediaRetryCount >= PROJECT_MEDIA_RETRY_DELAYS.length) return false;
        var delay = PROJECT_MEDIA_RETRY_DELAYS[_projectMediaRetryCount];
        _projectMediaRetryCount++;
        if (!projectMedia.length) {
        setProjectMediaPlaceholder("Scanning Premiere project media…");
        }
        if (!_projectMediaRetryNoticeShown && _projectMediaRetryCount > 1) {
        showToast("Refreshing Premiere project media…", "info");
            _projectMediaRetryNoticeShown = true;
        }
        console.warn("[OpenCut] Retrying project media scan in " + delay + "ms:", reason || "pending");
        _projectMediaRetryTimer = setTimeout(function () {
            _projectMediaRetryTimer = null;
            scanProjectMedia();
        }, delay);
        return true;
    }

    function isEvalScriptFailure(result) {
        return typeof result === "string" &&
            (result.indexOf("EvalScript error.") !== -1 || result.indexOf("EvalScript_ErrMessage") !== -1);
    }

    function applyProjectMediaScanData(data) {
        if (!data || typeof data !== "object") return false;
        if (data.error) {
            console.warn("[OpenCut] getProjectMedia error:", data.error);
            if (!scheduleProjectMediaRetry(data.error) && !projectMedia.length) {
                setProjectMediaPlaceholder("Couldn't load Premiere project media");
            }
            return false;
        }

        var media = Array.isArray(data.media) ? data.media : [];
        var nextProjectFolder = data.projectFolder || "";
        var folderChanged = nextProjectFolder !== (projectFolder || "");

        if (!media.length && inPremiere && Number(data.rootChildren || 0) > 0 && projectMedia.length && !folderChanged) {
            scheduleProjectMediaRetry("project items present but media list is temporarily empty");
            return false;
        }

        updateProjectMediaList(media, nextProjectFolder);

        if (media.length) {
            resetProjectMediaRetryState();
            return true;
        }

        if (inPremiere && Number(data.rootChildren || 0) > 0) {
            if (!scheduleProjectMediaRetry("project items present but media list is empty") && !projectMedia.length) {
                setProjectMediaPlaceholder("No importable project media found");
            }
            return false;
        }

        resetProjectMediaRetryState();
        return true;
    }

    function setHintContent(container, message) {
        if (!container) return;
        container.innerHTML = "";
        var hint = document.createElement("div");
        hint.className = "hint";
        hint.textContent = message;
        container.appendChild(hint);
    }

    var HINT_STATE_CLASSES = ["is-info", "is-error", "is-success", "is-warning"];

    function getHintCopyNode(hintEl) {
        if (!hintEl) return null;
        var copy = hintEl.querySelector(".hint-copy");
        if (copy) return copy;
        var firstEl = hintEl.firstElementChild;
        if (firstEl && firstEl.tagName !== "BUTTON" && !firstEl.classList.contains("btn-install") && !firstEl.classList.contains("btn-text")) {
            firstEl.classList.add("hint-copy");
            return firstEl;
        }
        copy = document.createElement("span");
        copy.className = "hint-copy";
        var child = hintEl.firstChild;
        while (child) {
            var next = child.nextSibling;
            if (child.nodeType === 3) hintEl.removeChild(child);
            child = next;
        }
        hintEl.insertBefore(copy, hintEl.firstChild);
        return copy;
    }

    function setHintState(hintEl, message, tone, actionBtn) {
        if (!hintEl) return;
        var copy = getHintCopyNode(hintEl);
        for (var i = 0; i < HINT_STATE_CLASSES.length; i++) {
            hintEl.classList.remove(HINT_STATE_CLASSES[i]);
        }
        hintEl.classList.add("is-" + (tone || "info"));
        hintEl.classList.remove("hidden");
        if (copy) copy.textContent = message || "";
        if (actionBtn) {
            var pending = tone === "info";
            actionBtn.classList.toggle("hidden", pending);
            actionBtn.disabled = pending;
        }
    }

    function hideHintState(hintEl, actionBtn) {
        if (!hintEl) return;
        hintEl.classList.add("hidden");
        for (var i = 0; i < HINT_STATE_CLASSES.length; i++) {
            hintEl.classList.remove(HINT_STATE_CLASSES[i]);
        }
        if (actionBtn) {
            actionBtn.classList.remove("hidden");
            actionBtn.disabled = false;
        }
    }

    function buildEmptyHintMarkup(title, copy, tone) {
        var classes = "hint hint-empty is-" + (tone || "info");
        var html = '<div class="' + classes + '">' +
            '<span class="hint-title">' + esc(title || "") + '</span>';
        if (copy) {
            html += '<span class="hint-copy">' + esc(copy) + '</span>';
        }
        html += "</div>";
        return html;
    }

    var _clipInfoCache = {};
    var _CLIP_INFO_CACHE_MAX = 12;

    function cacheClipInfo(path, data) {
        if (!path || !data) return;
        if (!_clipInfoCache[path]) {
            var keys = Object.keys(_clipInfoCache);
            if (keys.length >= _CLIP_INFO_CACHE_MAX) {
                delete _clipInfoCache[keys[0]];
            }
        }
        _clipInfoCache[path] = data;
    }

    function clearClipPreviewMeta() {
        if (el.clipMetaRes) el.clipMetaRes.textContent = "";
        if (el.clipMetaDur) el.clipMetaDur.textContent = "";
        if (el.clipMetaSize) el.clipMetaSize.textContent = "";
    }

    function applyClipPreviewMeta(path, data) {
        if (!data || path !== selectedPath) return;
        if (el.clipMetaDur) el.clipMetaDur.textContent = data.duration ? fmtDur(data.duration) : "";
        if (el.clipMetaRes) {
            el.clipMetaRes.textContent = data.video ? (data.video.width + "x" + data.video.height) : "";
        }
        if (el.clipMetaSize) {
            el.clipMetaSize.textContent = data.file_size_mb ? (safeFixed(data.file_size_mb, 1) + " MB") : "";
        }
    }

    function buildClipMetaText(path, data) {
        if (!data) return path || "";
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
        return meta || (path || "");
    }

    function clearSelectedFileState() {
        selectedPath = "";
        selectedName = "";
        transcriptData = null;
        lastTranscriptSegments = null;
        syncClipSelectValue("");
        rememberWorkspaceSelection("", "");
        if (el.fileInfoBox) el.fileInfoBox.classList.add("hidden");
        if (el.fileNameDisplay) el.fileNameDisplay.textContent = "";
        if (el.fileMetaDisplay) el.fileMetaDisplay.textContent = "";
        clearClipPreviewMeta();
        document.body.classList.remove("has-clip");
        updateWorkspaceClipStatus();
        updateButtons();
        updateClipPreview();
        if (el.recentClipsDropdown && !el.recentClipsDropdown.classList.contains("hidden")) {
            renderRecentClipsDropdown();
        }
    }

    function updateProjectMediaList(media, folder) {
        var files = Array.isArray(media) ? media : [];
        var desiredPath = selectedPath || _workspaceState.selectedPath || "";
        var desiredName = selectedName || _workspaceState.selectedName || "";
        projectMedia = files;
        projectFolder = folder || "";
        if (!el.clipSelect) return;
        // Batch DOM updates via DocumentFragment (one reflow instead of N)
        var frag = document.createDocumentFragment();
        var placeholder = document.createElement("option");
        placeholder.value = "";
        placeholder.selected = true;
        placeholder.textContent = files.length ? "-- Select a clip --" : "No project media found";
        frag.appendChild(placeholder);
        for (var i = 0; i < files.length; i++) {
            var clip = files[i] || {};
            var clipPath = clip.path || "";
            var clipName = clip.name || (clipPath ? clipPath.split(/[/\\]/).pop() : "Untitled clip");
            var option = document.createElement("option");
            option.value = clipPath;
            option.textContent = clipName;
            option.setAttribute("data-name", clipName);
            frag.appendChild(option);
        }
        el.clipSelect.innerHTML = "";
        el.clipSelect.appendChild(frag);
        populateRecentFiles();
        var restoredOption = syncClipSelectValue(desiredPath);
        if (restoredOption && restoredOption.value) {
            var restoredName = restoredOption.getAttribute("data-name") || desiredName || restoredOption.textContent || desiredPath.split(/[/\\]/).pop();
            if (desiredPath === selectedPath && restoredName === selectedName && el.fileInfoBox && !el.fileInfoBox.classList.contains("hidden")) {
                selectedPath = desiredPath;
                selectedName = restoredName;
                rememberWorkspaceSelection(selectedPath, selectedName);
                updateWorkspaceClipStatus();
                updateButtons();
                updateClipPreview();
                if (el.fileNameDisplay && !el.fileNameDisplay.textContent) el.fileNameDisplay.textContent = restoredName;
            } else {
                selectFile(desiredPath, restoredName);
            }
        } else {
            clearSelectedFileState();
        }
    }

    function _doScanProjectMedia() {
        _scanDebounceTimer = null;
        if (_scanInProgress) return;
        _scanInProgress = true;
        // If not running inside Premiere, try the backend's project media endpoint
        if (!inPremiere) {
            if (connected) {
                api("GET", "/project/media", null, function (err, data) {
                    _scanInProgress = false;
                    if (!err && data) {
                        applyProjectMediaScanData(data);
                    }
                });
            } else {
                _scanInProgress = false;
            }
            return;
        }
        // Warn once if project hasn't been saved
        if (!_projectSaveWarned) {
            PremiereBridge.isProjectSaved(function (res) {
                try {
                    var info = JSON.parse(res);
                    if (!info.saved) {
                        showToast("Tip: Save your project before processing", "info");
                    }
                    _projectSaveWarned = true;
                } catch (e) {}
            });
        }
        PremiereBridge.getProjectMedia(function (result) {
            _scanInProgress = false;
            if (!result || result === "null" || result === "undefined") {
                if (!scheduleProjectMediaRetry("empty ExtendScript response") && !projectMedia.length) {
                    setProjectMediaPlaceholder("Couldn't load Premiere project media");
                }
                return;
            }
            if (isEvalScriptFailure(result)) {
                console.error("scanProjectMedia ExtendScript error:", result);
                if (!scheduleProjectMediaRetry(result) && !projectMedia.length) {
                    setProjectMediaPlaceholder("Couldn't load Premiere project media");
                }
                return;
            }
            try {
                var data = JSON.parse(result);
                applyProjectMediaScanData(data);
            } catch (e) {
                console.error("scanProjectMedia parse error:", e, result);
                if (!scheduleProjectMediaRetry("parse error") && !projectMedia.length) {
                    setProjectMediaPlaceholder("Couldn't load Premiere project media");
                    showAlert("Couldn't read project media. Make sure a project is open in Premiere Pro.");
                }
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
        // Check if optgroup already exists
        var existing = el.clipSelect.querySelector('optgroup[label="Recent Files"]');
        if (existing) existing.parentNode.removeChild(existing);
        if (recent.length) {
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
        }
        refreshClipDropdown();
    }

    function refreshClipDropdown() {
        if (!el.clipSelect || !el.clipSelect.parentNode) return;
        // Disconnect old observer to prevent leak
        if (el.clipSelect._customDropdown && el.clipSelect._customDropdown.observer) {
            el.clipSelect._customDropdown.observer.disconnect();
        }
        var oldDd = el.clipSelect.parentNode.querySelector(".custom-dropdown");
        if (oldDd) {
            oldDd.parentNode.removeChild(oldDd);
        }
        // Always rebuild (handles both first-time creation and updates)
        delete el.clipSelect.dataset.customized;
        createCustomDropdown(el.clipSelect);
    }

    function selectFile(path, name) {
        if (!path) {
            clearSelectedFileState();
            return;
        }
        selectedPath = path;
        selectedName = name || path.split(/[/\\]/).pop();
        addRecentClip(path);
        addRecentFile(path, selectedName);
        populateRecentFiles();
        var selectedOption = syncClipSelectValue(path);
        if (selectedOption) {
            selectedName = selectedOption.getAttribute("data-name") || selectedOption.textContent || selectedName;
        }
        rememberWorkspaceSelection(path, selectedName);
        lastTranscriptSegments = loadCachedTranscript(path);
        transcriptData = null;
        updateWorkspaceClipStatus();
        el.fileInfoBox.classList.remove("hidden");
        el.fileNameDisplay.textContent = selectedName;
        el.fileMetaDisplay.innerHTML = '<span class="skeleton skeleton-wide"></span>';
        // CSS-driven button state: body.has-clip enables .requires-clip buttons
        if (path) document.body.classList.add("has-clip");
        else document.body.classList.remove("has-clip");
        updateButtons();
        updateClipPreview(path);
        if (el.recentClipsDropdown && !el.recentClipsDropdown.classList.contains("hidden")) {
            renderRecentClipsDropdown();
        }

        if (connected) {
            var infoPath = path;
            api("POST", "/info", { filepath: infoPath }, function (err, data) {
                if (infoPath !== selectedPath) return;
                if (!err && data && !data.error) {
                    cacheClipInfo(infoPath, data);
                    if (el.fileMetaDisplay) el.fileMetaDisplay.textContent = buildClipMetaText(infoPath, data);
                    applyClipPreviewMeta(infoPath, data);
                    analyzeClipContext(data, infoPath);
                } else {
                    if (el.fileMetaDisplay) el.fileMetaDisplay.textContent = infoPath;
                    clearClipPreviewMeta();
                }
            });
        }
    }

    // ================================================================
    // Tab Navigation
    // ================================================================
    var TAB_DESCRIPTIONS = {
        cut: "Trim dead space, fillers, and rough pacing with a tighter review flow.",
        captions: "Transcribe, refine, and style captions without leaving the panel.",
        audio: "Balance dialogue, stems, loudness, and polish from one focused surface.",
        video: "Repair, enhance, and shape the image with cleaner finishing controls.",
        export: "Package platform-ready deliverables with fewer passes and clearer status.",
        timeline: "Write back markers, bins, and sequence edits with confidence.",
        nlp: "Search footage and trigger edit actions with natural-language commands.",
        settings: "Tune models, defaults, and workspace behavior for the whole studio."
    };

    function updateContentHeader(tabName, titleText) {
        if (el.contentTitle) {
            el.contentTitle.textContent = titleText || tabName;
        }
        if (el.contentSubtitle) {
            el.contentSubtitle.textContent = TAB_DESCRIPTIONS[tabName] || "Focused tools for the current editing workflow.";
        }
        updateWorkspaceStageSession(titleText || tabName);
    }

    function updateWorkspaceStageSession(activeTitle) {
        if (el.workspaceStageSource) {
            el.workspaceStageSource.textContent = selectedName || "Awaiting media";
            el.workspaceStageSource.title = selectedPath || "Choose a clip or drop media to start";
        }
        if (el.workspaceStageSuite) {
            el.workspaceStageSuite.textContent = activeTitle || (el.contentTitle ? el.contentTitle.textContent : "Cut & Clean");
        }
        if (el.workspaceStageStatus) {
            if (!connected) {
                el.workspaceStageStatus.textContent = "Backend offline";
            } else if (selectedPath) {
                el.workspaceStageStatus.textContent = "Ready to process";
            } else {
                el.workspaceStageStatus.textContent = "Awaiting media";
            }
        }
    }

    function updateWorkspaceClipStatus() {
        if (!el.workspaceClipStatus) return;
        if (selectedName) {
            el.workspaceClipStatus.textContent = selectedName;
            el.workspaceClipStatus.title = selectedPath || selectedName;
            el.workspaceClipStatus.classList.add("is-active");
        } else {
            el.workspaceClipStatus.textContent = "No media selected";
            el.workspaceClipStatus.title = "Choose a clip or drop media to start";
            el.workspaceClipStatus.classList.remove("is-active");
        }
        updateWorkspaceStageSession();
    }

    function setPanelVisibility(panel, active) {
        if (!panel) return;
        panel.classList.toggle("active", !!active);
        panel.hidden = !active;
        panel.setAttribute("aria-hidden", active ? "false" : "true");
    }

    function activateSubTab(tabName, subName, options) {
        var panel = $("panel-" + tabName);
        if (!panel) return null;
        var container = panel.querySelector(".sub-tabs");
        if (!container) return null;
        var remember = !options || options.remember !== false;
        var scroll = !options || options.scroll !== false;
        var buttons = getVisibleTabButtons(container, ".sub-tab");
        if (!buttons.length) return null;

        var targetButton = null;
        var i;
        if (subName) {
            for (i = 0; i < buttons.length; i++) {
                if (buttons[i].getAttribute("data-sub") === subName) {
                    targetButton = buttons[i];
                    break;
                }
            }
        }
        if (!targetButton) {
            var activeButton = container.querySelector(".sub-tab.active");
            if (activeButton && buttons.indexOf(activeButton) !== -1) {
                targetButton = activeButton;
            }
        }
        if (!targetButton) targetButton = buttons[0];

        for (i = 0; i < buttons.length; i++) {
            var isActiveButton = buttons[i] === targetButton;
            buttons[i].classList.toggle("active", isActiveButton);
            buttons[i].setAttribute("aria-selected", isActiveButton ? "true" : "false");
            buttons[i].tabIndex = isActiveButton ? 0 : -1;
        }

        var targetPanelId = "sub-" + targetButton.getAttribute("data-sub");
        var panels = panel.querySelectorAll(".sub-panel");
        for (i = 0; i < panels.length; i++) {
            var isActivePanel = panels[i].id === targetPanelId;
            setPanelVisibility(panels[i], isActivePanel);
            if (isActivePanel) panels[i].setAttribute("aria-labelledby", targetButton.id);
        }

        if (remember) rememberWorkspaceSub(tabName, targetButton.getAttribute("data-sub"));
        if (scroll && targetButton.scrollIntoView) {
            targetButton.scrollIntoView({ block: "nearest", inline: "nearest" });
        }
        return targetButton;
    }

    function activateNavTab(tabName, options) {
        var navBtns = document.querySelectorAll(".nav-tab");
        if (!navBtns.length) return null;
        var remember = !options || options.remember !== false;
        var scroll = !options || options.scroll !== false;
        var targetButton = null;
        var i;
        for (i = 0; i < navBtns.length; i++) {
            if (navBtns[i].getAttribute("data-nav") === tabName) {
                targetButton = navBtns[i];
                break;
            }
        }
        if (!targetButton) targetButton = document.querySelector(".nav-tab.active") || navBtns[0];

        for (i = 0; i < navBtns.length; i++) {
            var isActiveNav = navBtns[i] === targetButton;
            navBtns[i].classList.toggle("active", isActiveNav);
            navBtns[i].setAttribute("aria-selected", isActiveNav ? "true" : "false");
            navBtns[i].tabIndex = isActiveNav ? 0 : -1;
            setPanelVisibility($("panel-" + navBtns[i].getAttribute("data-nav")), isActiveNav);
        }

        var activeTabName = targetButton.getAttribute("data-nav") || "cut";
        if (remember) rememberWorkspaceTab(activeTabName);
        updateContentHeader(activeTabName, targetButton.getAttribute("title") || activeTabName);
        _pproCache.seq = null;
        _pproCache.seqTs = 0;
        _pproCache.clips = null;
        _pproCache.clipsTs = 0;
        _pproCache.bins = null;
        _pproCache.binsTs = 0;
        checkSubTabOverflow();
        if (activeTabName === "settings") loadSettingsInfo();
        initTabOnFirstVisit(activeTabName);
        activateSubTab(activeTabName, (options && options.sub) || (_workspaceState.activeSubs || {})[activeTabName], {
            remember: remember,
            scroll: scroll
        });
        if (options && options.focus) targetButton.focus();
        return targetButton;
    }

    function setupNavTabs() {
        var navContainer = $("navTabs");
        if (navContainer) navContainer.setAttribute("aria-orientation", "vertical");

        var navBtns = document.querySelectorAll(".nav-tab");
        for (var i = 0; i < navBtns.length; i++) {
            var navName = navBtns[i].getAttribute("data-nav") || ("tab-" + i);
            navBtns[i].id = navBtns[i].id || ("nav-tab-" + navName);
            navBtns[i].setAttribute("aria-controls", "panel-" + navName);
            navBtns[i].tabIndex = navBtns[i].classList.contains("active") ? 0 : -1;
            var controlledPanel = $("panel-" + navName);
            if (controlledPanel) {
                controlledPanel.setAttribute("aria-labelledby", navBtns[i].id);
                controlledPanel.setAttribute("aria-hidden", controlledPanel.classList.contains("active") ? "false" : "true");
                controlledPanel.hidden = !controlledPanel.classList.contains("active");
            }

            navBtns[i].addEventListener("click", function () {
                activateNavTab(this.getAttribute("data-nav"));
            });
            navBtns[i].addEventListener("keydown", function (e) {
                var buttons = getVisibleTabButtons(this.parentElement, ".nav-tab");
                if (e.key === "ArrowDown" || e.key === "ArrowRight") {
                    e.preventDefault();
                    moveFocusAndActivate(buttons, this, 1);
                } else if (e.key === "ArrowUp" || e.key === "ArrowLeft") {
                    e.preventDefault();
                    moveFocusAndActivate(buttons, this, -1);
                } else if (e.key === "Home" && buttons.length) {
                    e.preventDefault();
                    buttons[0].focus();
                    buttons[0].click();
                } else if (e.key === "End" && buttons.length) {
                    e.preventDefault();
                    buttons[buttons.length - 1].focus();
                    buttons[buttons.length - 1].click();
                }
            });
        }

        var subTabContainers = document.querySelectorAll(".sub-tabs");
        for (var i = 0; i < subTabContainers.length; i++) {
            (function (container) {
                var btns = container.querySelectorAll(".sub-tab");
                var parentPanel = container.closest(".nav-panel");
                var parentTabName = getPanelTabName(parentPanel);
                container.setAttribute("aria-orientation", "horizontal");
                for (var j = 0; j < btns.length; j++) {
                    var subName = btns[j].getAttribute("data-sub") || (parentTabName + "-sub-" + j);
                    btns[j].id = btns[j].id || ("sub-tab-" + subName);
                    btns[j].setAttribute("role", "tab");
                    btns[j].setAttribute("aria-controls", "sub-" + subName);
                    btns[j].setAttribute("aria-selected", btns[j].classList.contains("active") ? "true" : "false");
                    btns[j].tabIndex = btns[j].classList.contains("active") ? 0 : -1;
                    var subPanel = $("sub-" + subName);
                    if (subPanel) {
                        subPanel.setAttribute("role", "tabpanel");
                        subPanel.setAttribute("aria-labelledby", btns[j].id);
                        subPanel.setAttribute("aria-hidden", subPanel.classList.contains("active") ? "false" : "true");
                        subPanel.hidden = !subPanel.classList.contains("active");
                    }

                    btns[j].addEventListener("click", function () {
                        activateSubTab(parentTabName, this.getAttribute("data-sub"));
                    });
                    btns[j].addEventListener("keydown", function (e) {
                        var buttons = getVisibleTabButtons(container, ".sub-tab");
                        if (e.key === "ArrowRight" || e.key === "ArrowDown") {
                            e.preventDefault();
                            moveFocusAndActivate(buttons, this, 1);
                        } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
                            e.preventDefault();
                            moveFocusAndActivate(buttons, this, -1);
                        } else if (e.key === "Home" && buttons.length) {
                            e.preventDefault();
                            buttons[0].focus();
                            buttons[0].click();
                        } else if (e.key === "End" && buttons.length) {
                            e.preventDefault();
                            buttons[buttons.length - 1].focus();
                            buttons[buttons.length - 1].click();
                        }
                    });
                }

                activateSubTab(parentTabName, (_workspaceState.activeSubs || {})[parentTabName], {
                    remember: false,
                    scroll: false
                });
            })(subTabContainers[i]);
        }

        var initiallyActiveNav = document.querySelector(".nav-tab.active");
        activateNavTab(_workspaceState.activeNav || (initiallyActiveNav ? initiallyActiveNav.getAttribute("data-nav") : "") || "cut", {
            remember: false,
            scroll: false
        });
        updateWorkspaceClipStatus();
    }

    // ================================================================
    // Lazy Tab Init (Phase 5.1) — Defer heavy init until tab first visited
    // ================================================================
    function initTabOnFirstVisit(tabName) {
        if (_tabRendered[tabName]) return;
        _tabRendered[tabName] = true;

        switch (tabName) {
            case "captions":
                initCaptionNewFeatures();
                break;
            case "audio":
                initAudioNewFeatures();
                break;
            case "timeline":
                initTimelineFeatures();
                break;
            case "nlp":
                initNlpFeatures();
                break;
            case "export":
                initDeliverablesFeatures();
                break;
            // cut, video, settings — no deferred init needed (core listeners bound in DOMContentLoaded)
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
        "runExportPresetBtn", "runThumbBtn", "runWorkflowBtn",
        "runBurninBtn",
        "runSpeedBtn", "runLutBtn", "runDuckBtn",
        "runChromaBtn", "runTransBtn", "runParticlesBtn",
        "runTitleOverlayBtn", "runReframeBtn", "runUpscaleBtn",
        "runColorBtn", "runRemoveBtn", "runFaceAiBtn", "runAnimCapBtn",
        "runExpTranscriptBtn", "loadWaveformBtn", "previewVfxBtn", "runTrimBtn",
        "runAutoEditBtn", "runHighlightsBtn", "runEmotionHighlightsBtn", "runEnhanceBtn", "runShortsBtn",
        "autoDetectWatermarkBtn", "runDepthBtn", "runBrollPlanBtn", "runBrollGenBtn", "runMmDiarizeBtn", "socialUploadBtn",
        "runRepeatDetectBtn", "runChaptersBtn", "runBeatMarkersBtn", "runMulticamBtn",
        "quickCleanInterview", "quickYouTube", "quickPodcast",
        "polishInterviewBtn",
        "quickAutoSubtitle", "quickTranslate",
        "quickStudioAudio", "quickDenoise",
        "quickAutoColor", "quickSocialReframe"
    ];

    function updateButtons() {
        var canRun = connected && selectedPath;
        var hasProjectMedia = Array.isArray(projectMedia) && projectMedia.length > 0;
        var hasSequenceInfo = !!(sequenceInfo && typeof sequenceInfo === "object" && !sequenceInfo.error);
        var hasSequenceMarkers = Array.isArray(seqMarkersData) && seqMarkersData.length > 0;
        var hasRenameItems = Array.isArray(renameItemsData) && renameItemsData.length > 0;
        var hasSmartBinRules = Array.isArray(smartBinRules) && smartBinRules.length > 0;

        // v1.9.30 (J): selection change toggles the "Apply to selection"
        // button visibility on history rows. Re-render if the tray is open.
        if (el.jobHistory && el.jobHistory.classList.contains("open")) {
            renderJobHistory();
        }

        // Interview polish hint — reflect the same state the button shows
        if (el.interviewPolishHint) {
            if (!connected) {
                el.interviewPolishHint.textContent = "Server disconnected.";
            } else if (!selectedPath) {
                el.interviewPolishHint.textContent = "Select a clip to run.";
            } else {
                el.interviewPolishHint.textContent = "Runs on '" + (selectedName || selectedPath) + "'.";
            }
        }

        // Batch-disable all clip-dependent buttons
        for (var i = 0; i < _clipButtons.length; i++) {
            var btn = el[_clipButtons[i]];
            if (btn) btn.disabled = !canRun;
        }

        // Merge has special logic
        if (el.runMergeBtn) el.runMergeBtn.disabled = !connected || _mergeFiles.length < 2;
        var batchCandidateCount = _batchFiles && _batchFiles.length ? _batchFiles.length : projectMedia.length;
        if (!batchCandidateCount && el.clipSelect && el.clipSelect.options) {
            for (var batchIdx = 0; batchIdx < el.clipSelect.options.length; batchIdx++) {
                if (el.clipSelect.options[batchIdx].value) batchCandidateCount++;
            }
        }
        if (el.runBatchBtn) el.runBatchBtn.disabled = !connected || batchCandidateCount < 2;
        if (el.runLoudMatchBtn) el.runLoudMatchBtn.disabled = !connected || !hasProjectMedia;
        if (el.runFootageSearchBtn) el.runFootageSearchBtn.disabled = !connected;
        if (el.runNlpCommandBtn) el.runNlpCommandBtn.disabled = !connected;
        if (el.generateLutBtn) el.generateLutBtn.disabled = !connected;
        if (el.testLLMBtn) el.testLLMBtn.disabled = !connected;
        if (el.settingsInstallWhisperBtn) el.settingsInstallWhisperBtn.disabled = !connected;
        if (el.settingsReinstallWhisperBtn) el.settingsReinstallWhisperBtn.disabled = !connected;
        if (el.loadSeqInfoBtn) el.loadSeqInfoBtn.disabled = !inPremiere;
        if (el.genVfxSheetBtn) el.genVfxSheetBtn.disabled = !connected || !inPremiere || !hasSequenceInfo;
        if (el.genAdrListBtn) el.genAdrListBtn.disabled = !connected || !inPremiere || !hasSequenceInfo;
        if (el.genMusicCueBtn) el.genMusicCueBtn.disabled = !connected || !inPremiere || !hasSequenceInfo;
        if (el.genAssetListBtn) el.genAssetListBtn.disabled = !connected || !inPremiere || !hasSequenceInfo;
        if (el.getSeqMarkersBtn) el.getSeqMarkersBtn.disabled = !inPremiere;
        if (el.exportMarkedClipsBtn) el.exportMarkedClipsBtn.disabled = !connected || !inPremiere || !selectedPath || !hasSequenceMarkers;
        if (el.loadProjectItemsBtn) el.loadProjectItemsBtn.disabled = !inPremiere;
        if (el.applyRenamePatternBtn) el.applyRenamePatternBtn.disabled = !hasRenameItems;
        if (el.renameAllBtn) el.renameAllBtn.disabled = !connected || !hasRenameItems;
        if (el.createSmartBinsBtn) el.createSmartBinsBtn.disabled = !connected || !hasSmartBinRules;
        if (el.runSrtImportBtn) el.runSrtImportBtn.disabled = !connected || !inPremiere;
        if (el.indexAllClipsBtn) el.indexAllClipsBtn.disabled = !connected || !hasProjectMedia;

        // Helper to safely toggle hint visibility and optionally disable a button
        function _setHint(hintEl, btnEl, showHint) {
            if (hintEl) hintEl.classList.toggle("hidden", !showHint);
            if (btnEl && showHint) btnEl.disabled = true;
            var actionBtn = hintEl ? hintEl.querySelector(".btn-install, .btn-text") : null;
            if (actionBtn && (!hintEl || !hintEl.classList.contains("is-info"))) {
                actionBtn.classList.remove("hidden");
                actionBtn.disabled = false;
            }
        }

        var whisperAvailable = capabilities.captions !== false;
        var crisperAvailable = capabilities.crisper_whisper !== false;
        var showFillersHint = false;
        var showCrisperInstallAction = false;
        var fillersHintMessage = "No filler detection backend is installed yet. Add Whisper from Settings or install CrisperWhisper here for verbatim filler detection.";

        if (el.fillerBackend) {
            var whisperOpt = el.fillerBackend.querySelector('option[value="whisper"]');
            var crisperOpt = el.fillerBackend.querySelector('option[value="crisper"]');
            var selectedFillerBackend = el.fillerBackend.value || "whisper";

            if (whisperOpt) whisperOpt.disabled = !whisperAvailable;
            if (crisperOpt) crisperOpt.disabled = !crisperAvailable;

            if (selectedFillerBackend === "whisper" && !whisperAvailable && crisperAvailable) {
                el.fillerBackend.value = "crisper";
            } else if (selectedFillerBackend === "crisper" && !crisperAvailable && whisperAvailable) {
                el.fillerBackend.value = "whisper";
            }

            if (el.fillerBackend._customDropdown) el.fillerBackend._customDropdown.updateText();
        }

        if (!whisperAvailable && !crisperAvailable) {
            showFillersHint = true;
            showCrisperInstallAction = true;
        }

        if (el.fillersHint) {
            var fillersHintCopy = getHintCopyNode(el.fillersHint);
            if (fillersHintCopy) fillersHintCopy.textContent = fillersHintMessage;
        }

        // Whisper hints
        _setHint(el.captionsHint, null, capabilities.captions === false);
        _setHint(el.fillersHint, el.runFillersBtn, showFillersHint);
        if (el.installCrisperWhisperBtn) {
            var hideCrisperAction = !showFillersHint || !showCrisperInstallAction || (el.fillersHint && el.fillersHint.classList.contains("is-info"));
            el.installCrisperWhisperBtn.classList.toggle("hidden", hideCrisperAction);
            el.installCrisperWhisperBtn.disabled = hideCrisperAction;
        }

        // Demucs hint
        _setHint(el.separateHint, el.runSeparateBtn, capabilities.separation === false);

        // Watermark removal hint
        _setHint(el.watermarkHint, el.runWatermarkBtn, capabilities.watermark_removal === false);

        // Pedalboard hint
        _setHint(el.proFxHint, el.runProFxBtn, capabilities.pedalboard === false);

        // DeepFilterNet hint
        _setHint(el.deepFilterHint, el.runDeepFilterBtn, capabilities.deepfilter === false);

        // Video AI hints (upscale/rembg need install, interp/denoise always work)
        var aiCaps = capabilities.video_ai || {};
        var tool = el.vidAiTool ? el.vidAiTool.value : "upscale";
        if (tool === "upscale" && aiCaps.upscale === false) {
            if (el.vidAiHint) el.vidAiHint.classList.remove("hidden");
            if (el.vidAiHintText) el.vidAiHintText.textContent = "Real-ESRGAN not installed.";
        } else if (tool === "rembg" && aiCaps.rembg === false) {
            if (el.vidAiHint) el.vidAiHint.classList.remove("hidden");
            if (el.vidAiHintText) el.vidAiHintText.textContent = "rembg not installed.";
        } else {
            if (el.vidAiHint) el.vidAiHint.classList.add("hidden");
        }

        // Face tools hint
        var faceCaps = capabilities.face_tools || {};
        _setHint(el.faceHint, null, faceCaps.mediapipe === false);

        // WhisperX / karaoke hint
        _setHint(el.karaokeHint, el.runKaraokeBtn, capabilities.whisperx === false);

        // NLLB translation hint
        _setHint(el.translateHint, null, capabilities.nllb === false);

        // Edge TTS hint
        _setHint(el.ttsHint, null, capabilities.edge_tts === false);

        // Silero VAD hint
        _setHint(el.vadHint, null, capabilities.silero_vad === false);

        // OTIO export hint
        _setHint(el.otioHint, null, capabilities.otio === false);
        var otioBtn = document.getElementById("exportOtioBtn");
        if (otioBtn) otioBtn.disabled = !canRun || capabilities.otio === false;

        // AI B-Roll generation hint
        _setHint(el.brollGenHint, el.runBrollGenBtn, capabilities.broll_generate === false);

        // Multimodal diarization hint
        _setHint(el.mmDiarizeHint, el.runMmDiarizeBtn, capabilities.multimodal_diarize === false);

        // Depth effects hint
        _setHint(el.depthHint, el.runDepthBtn, capabilities.depth_effects === false);

        // Emotion highlights hint
        _setHint(el.emotionHint, el.runEmotionHighlightsBtn, capabilities.deepface === false);

        // Social media posting hint
        _setHint(el.socialHint, null, capabilities.social_post === false);
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
                setConnectionBadge("online", "Connected (" + data.gpu_name + ")");
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
    var jobStepCurrent = 0;
    var jobStepTotal = 0;

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

    function normalizeJobOptions(options) {
        if (typeof options === "function") {
            return { onComplete: options };
        }
        return options || {};
    }

    function runStartJobErrorHook(options, detail) {
        if (options && typeof options.onStartError === "function") {
            try {
                options.onStartError(detail);
            } catch (hookErr) {
                console.error("startJob onStartError hook failed:", hookErr);
            }
        }
    }

    function settleJobLifecycle(job) {
        var jobId = job && (job.id || job.job_id);
        if (!jobId || !jobLifecycleHandlers[jobId]) return;
        var hooks = jobLifecycleHandlers[jobId];
        delete jobLifecycleHandlers[jobId];
        try {
            if (job.status === "complete" && typeof hooks.onComplete === "function") {
                hooks.onComplete(job.result || {}, job);
            } else if (job.status === "error" && typeof hooks.onError === "function") {
                hooks.onError(job);
            } else if (job.status === "cancelled" && typeof hooks.onCancel === "function") {
                hooks.onCancel(job);
            }
        } catch (hookErr) {
            console.error("startJob lifecycle hook failed:", hookErr);
        }
        if (typeof hooks.onFinally === "function") {
            try {
                hooks.onFinally(job);
            } catch (finalErr) {
                console.error("startJob onFinally hook failed:", finalErr);
            }
        }
    }

    function startUtilityJob(endpoint, payload, options) {
        var opts = options || {};
        var requestSeq = ++_utilityJobSeq;
        var isStale = typeof opts.isStale === "function" ? opts.isStale : null;

        function isAbandoned() {
            return !!(isStale && isStale(requestSeq));
        }

        function finish(job) {
            if (typeof opts.onFinally === "function") {
                try {
                    opts.onFinally(job, requestSeq);
                } catch (finalErr) {
                    console.error("utility job onFinally hook failed:", finalErr);
                }
            }
        }

        api("POST", endpoint, payload, function (err, data) {
            if (isAbandoned()) {
                finish(null);
                return;
            }
            if (err || !data || data.error || !data.job_id) {
                if (typeof opts.onError === "function") {
                    try {
                        opts.onError(data || err || { error: "Failed to start request" }, requestSeq);
                    } catch (hookErr) {
                        console.error("utility job onError hook failed:", hookErr);
                    }
                }
                finish(data || err || null);
                return;
            }

            var jobId = data.job_id;
            function poll() {
                if (isAbandoned()) {
                    finish(null);
                    return;
                }
                api("GET", "/status/" + jobId, null, function (statusErr, job) {
                    if (isAbandoned()) {
                        finish(job || null);
                        return;
                    }
                    if (statusErr || !job) {
                        setTimeout(poll, Math.max(250, POLL_MS));
                        return;
                    }
                    if (typeof opts.onProgress === "function" && job.status === "running") {
                        try {
                            opts.onProgress(job, requestSeq);
                        } catch (progressErr) {
                            console.error("utility job onProgress hook failed:", progressErr);
                        }
                    }
                    if (job.status === "complete") {
                        if (typeof opts.onComplete === "function") {
                            try {
                                opts.onComplete(job.result || {}, job, requestSeq);
                            } catch (hookErr) {
                                console.error("utility job onComplete hook failed:", hookErr);
                            }
                        }
                        finish(job);
                        return;
                    }
                    if (job.status === "error" || job.status === "cancelled") {
                        if (typeof opts.onError === "function") {
                            try {
                                opts.onError(job, requestSeq);
                            } catch (hookErr) {
                                console.error("utility job onError hook failed:", hookErr);
                            }
                        }
                        finish(job);
                        return;
                    }
                    setTimeout(poll, Math.max(250, POLL_MS));
                });
            }
            poll();
        });

        return requestSeq;
    }

    function startInlineInstallJob(config) {
        if (!config || !config.endpoint) return;

        var hintEl = config.hintEl || null;
        var actionBtn = config.actionBtn || null;
        var startMessage = config.startMessage || "Installing…";

        if (hintEl) {
            setHintState(hintEl, startMessage, "info", actionBtn);
        } else if (actionBtn) {
            actionBtn.disabled = true;
        }

        startUtilityJob(config.endpoint, config.payload || {}, {
            onProgress: function(job) {
                if (hintEl) {
                    setHintState(hintEl, (job && job.message) || startMessage, "info", actionBtn);
                }
            },
            onComplete: function(result, job) {
                if (hintEl) hideHintState(hintEl, actionBtn);
                if (typeof config.onSuccess === "function") config.onSuccess(result || {}, job || null);
            },
            onError: function(detail) {
                var errMsg = formatInstallError(detail, "Unknown error");
                if (hintEl) {
                    setHintState(hintEl, "Installation failed: " + errMsg, "error", actionBtn);
                } else if (actionBtn) {
                    actionBtn.disabled = false;
                }
                if (typeof config.onError === "function") config.onError(detail, errMsg);
            },
            onFinally: function(job) {
                if (!hintEl && actionBtn) actionBtn.disabled = false;
                if (typeof config.onFinally === "function") config.onFinally(job || null);
            }
        });
    }

    function startJob(endpoint, payload, options) {
        var opts = normalizeJobOptions(options);
        if (currentJob || jobStarting) {
            showAlert("Another task is in progress. You can cancel it from the processing bar above.");
            runStartJobErrorHook(opts, { reason: "busy" });
            return false;
        }
        if (!selectedPath && payload && !payload.filepath && !payload.no_input) {
            showAlert("Choose a clip from the Media section above to get started.");
            runStartJobErrorHook(opts, { reason: "missing-input" });
            return false;
        }

        // Lock immediately to prevent double-click race
        jobStarting = true;

        // Show persistent processing banner
        var stepPrefix = (jobStepTotal > 1) ? "Step " + jobStepCurrent + "/" + jobStepTotal + ": " : "";
        el.processingBanner.classList.remove("hidden");
        el.processingMsg.textContent = stepPrefix + "Starting…";
        el.processingFill.style.width = "0%";
        el.processingFill.setAttribute("aria-valuenow", "0");
        el.processingElapsed.textContent = "0s";

        // Show inline progress section too
        el.progressSection.classList.remove("hidden");
        el.resultsSection.classList.add("hidden");
        el.progressBar.style.width = "0%";
        el.progressBar.setAttribute("aria-valuenow", "0");
        el.progressLabel.textContent = stepPrefix + "Starting…";
        el.cancelBtn.classList.remove("hidden");

        // Lock the entire UI
        document.body.classList.add("job-active");

        // Track for retry
        lastJobEndpoint = endpoint;
        lastJobPayload = payload;

        // Show time estimate based on historical data
        fetchTimeEstimate(endpoint.replace(/^\//, "").replace(/\//g, "_"));

        jobStartTime = Date.now();
        if (elapsedTimer) { clearInterval(elapsedTimer); elapsedTimer = null; }
        elapsedTimer = setInterval(function () {
            var s = Math.floor((Date.now() - jobStartTime) / 1000);
            var timeStr = s < 60 ? s + "s" : Math.floor(s / 60) + "m " + (s % 60) + "s";
            el.progressElapsed.textContent = timeStr;
            el.processingElapsed.textContent = timeStr;
        }, 1000);

        try {
            api("POST", endpoint, payload, function (err, data) {
                if (err || !data || data.error) {
                    jobStarting = false;
                    showAlert(data ? data.error : "Failed to start job", data);
                    hideProgress();
                    runStartJobErrorHook(opts, { reason: "request-failed", err: err, data: data });
                    return;
                }
                // Set currentJob BEFORE clearing jobStarting to prevent
                // double-click race where a second job could start between
                // jobStarting=false and currentJob assignment.
                currentJob = data.job_id;
                jobStarting = false;
                if (opts.onComplete || opts.onError || opts.onCancel || opts.onFinally) {
                    jobLifecycleHandlers[data.job_id] = opts;
                }

                if (SSE_OK) {
                    trackJobSSE(data.job_id);
                } else {
                    trackJobPoll(data.job_id);
                }
            });
            return true;
        } catch (e) {
            jobStarting = false;
            hideProgress();
            showAlert("Failed to start job: " + e.message);
            runStartJobErrorHook(opts, { reason: "exception", error: e });
            return false;
        }
    }

    function trackJobSSE(jobId) {
        if (activeStream) { activeStream.close(); activeStream = null; }
        var es = new EventSource(BACKEND + "/stream/" + jobId);
        activeStream = es;

        es.onmessage = function (e) {
            // Guard against stale events firing after cancelJob()
            if (!currentJob || activeStream !== es) { try { es.close(); } catch (_) {} return; }
            try {
                var job = JSON.parse(e.data);
                updateProgress(job);
                if (job.status === "complete" || job.status === "error" || job.status === "cancelled") {
                    es.close();
                    activeStream = null;
                    onJobDone(job);
                }
            } catch (ex) {
                console.error("SSE JSON parse error:", ex);
            }
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
        if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
        pollTimer = setInterval(function () {
            api("GET", "/status/" + jobId, null, function (err, job) {
                // Guard against in-flight polls returning after cancelJob()
                if (!currentJob || currentJob !== jobId) return;
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
        var msg = job.message || "Processing…";
        if (jobStepTotal > 1) {
            msg = "Step " + jobStepCurrent + "/" + jobStepTotal + ": " + msg;
        }
        el.progressBar.style.width = pct;
        el.progressBar.setAttribute("aria-valuenow", String(job.progress || 0));
        el.progressLabel.textContent = msg;
        // Sync to persistent banner
        el.processingFill.style.width = pct;
        el.processingFill.setAttribute("aria-valuenow", String(job.progress || 0));
        el.processingMsg.textContent = msg;
    }

    // Structured error code -> actionable guidance map
    var ERROR_CODE_ACTIONS = {
        "GPU_OUT_OF_MEMORY": { tab: "settings", msg: "GPU ran out of memory. Try CPU mode in Settings." },
        "MISSING_DEPENDENCY": { tab: "settings", sub: "dependencies", msg: null },
        "FILE_NOT_FOUND": { msg: "File not found. Re-select your clip." },
        "RATE_LIMITED": { msg: null },
        "TOO_MANY_JOBS": { msg: "Too many jobs running. Wait or cancel one." },
        "INSTALL_FAILED": { tab: "settings", msg: null },
        "UNSUPPORTED_FORMAT": { msg: null },
        "FFMPEG_ERROR": { msg: null },
        "PERMISSION_DENIED": { msg: null },
        "OPERATION_TIMEOUT": { msg: null },
        "SERVER_BUSY": { msg: null }
    };

    function enhanceError(msg, errorData) {
        // 1. Check structured error code FIRST
        if (errorData && errorData.code && ERROR_CODE_ACTIONS[errorData.code]) {
            var action = ERROR_CODE_ACTIONS[errorData.code];
            var base = action.msg || errorData.error || msg;
            if (errorData.suggestion) {
                base = base + " \u2014 " + errorData.suggestion;
            }
            return base;
        }
        // 2. If the server returned a suggestion without a known code, use it directly
        if (errorData && errorData.suggestion) {
            return (errorData.error || msg) + " \u2014 " + errorData.suggestion;
        }
        if (!msg) return msg;
        // 3. Fallback: regex-based enhancement for legacy/unstructured errors
        if (/not installed|No module named/i.test(msg)) {
            return msg + " \u2014 You can install this from the Settings tab.";
        }
        if (/memory|CUDA out of memory|out of memory/i.test(msg)) {
            return msg + " \u2014 Try a smaller file, lower quality setting, or enable CPU mode in Settings.";
        }
        if (/Permission|Access denied|denied/i.test(msg)) {
            return msg + " \u2014 The file may be locked or read-only. Check your file permissions.";
        }
        if (/No such file|not found/i.test(msg)) {
            return msg + " \u2014 The file may have been moved or deleted.";
        }
        if (/timed? ?out|timeout/i.test(msg)) {
            return msg + " \u2014 The operation took too long. Try a shorter clip or simpler settings.";
        }
        if (/connection|ECONNREFUSED|network/i.test(msg)) {
            return msg + " \u2014 Make sure the OpenCut server is running.";
        }
        return msg;
    }

    function getErrorCodeAction(errorData) {
        if (!errorData || !errorData.code) return null;
        return ERROR_CODE_ACTIONS[errorData.code] || null;
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
            el.resultsStats.textContent = enhanceError(job.error || job.message || "Unknown error", job);
            el.resultsPath.textContent = "";
            // Show retry button if we have a last job to retry
            if (lastJobEndpoint) {
                el.retryJobBtn.classList.remove("hidden");
            }
            // Also show alert banner with action link for code-aware errors
            if (job.code) {
                showErrorWithAction(job);
            }
            settleJobLifecycle(job);
            return;
        }

        if (job.status === "cancelled") {
            hideProgress();
            settleJobLifecycle(job);
            return;
        }

        // Success
        hideProgress();
        el.retryJobBtn.classList.add("hidden");
        showResults(job);
        settleJobLifecycle(job);

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
                            journalRecord(
                                "import_sequence",
                                _sessionCtxOpText(job) + " → '" + r.sequence_name + "'",
                                { name: r.sequence_name },
                                selectedPath
                            );
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

            // Re-scan project media after auto-import so new files appear in the clip list
            setTimeout(function () { scanProjectMedia(); }, 1500);
        }
    }

    function hideProgress() {
        el.progressSection.classList.add("hidden");
        el.cancelBtn.classList.add("hidden");
        setButtonText(el.cancelBtn, rememberButtonText(el.cancelBtn));
        el.cancelBtn.disabled = false;
        el.processingBanner.classList.add("hidden");
        setButtonText(el.processingCancel, rememberButtonText(el.processingCancel));
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
        if (r.indexed !== undefined && r.total !== undefined) {
            stats += (stats ? "<br>" : "") + Number(r.indexed) + " of " + Number(r.total) + " files indexed";
            if (r.errors && r.errors.length) {
                stats += " | " + Number(r.errors.length) + " error" + (r.errors.length === 1 ? "" : "s");
            }
        }

        el.resultsStats.innerHTML = stats || "Processing complete.";
        el.resultsPath.textContent = r.xml_path || r.output_path || r.overlay_path || (r.output_paths ? r.output_paths.length + " files exported" : "");
    }

    function cancelJob() {
        if (currentJob) {
            var cancellingJob = currentJob;
            rememberButtonText(el.processingCancel);
            setButtonText(el.processingCancel, "Cancelling…");
            el.processingCancel.disabled = true;
            rememberButtonText(el.cancelBtn);
            setButtonText(el.cancelBtn, "Cancelling…");
            el.cancelBtn.disabled = true;
            // Close SSE/poll FIRST to prevent "complete" events from firing
            if (pollTimer) {
                clearInterval(pollTimer);
                pollTimer = null;
            }
            if (activeStream) {
                activeStream.close();
                activeStream = null;
            }
            // Now safe to null currentJob — no more event handlers can fire
            currentJob = null;
            hideProgress();
            // Fire cancel to backend (best-effort, UI already updated)
            api("POST", "/cancel/" + cancellingJob, {}, function (err) {
                if (err) {
                    showToast("Couldn't cancel — server not responding", "error");
                }
            });
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
        var detectMethod = el.silenceDetectMethod ? el.silenceDetectMethod.value : "auto";
        var payload = { filepath: selectedPath, output_dir: projectFolder, method: detectMethod };
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
        var fillerBackend = el.fillerBackend ? el.fillerBackend.value : "whisper";

        startJob("/fillers", {
            filepath: selectedPath,
            output_dir: projectFolder,
            model: el.fillerModel.value,
            remove_fillers: removeKeys,
            custom_words: custom,
            remove_silence: el.fillerSilence.checked,
            filler_backend: fillerBackend,
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
            word_highlight: el.captionWordHighlight ? el.captionWordHighlight.checked : false,
            auto_emoji: el.captionAutoEmoji ? el.captionAutoEmoji.checked : false,
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
                showAlert("Exported to: " + String(data.output_path).split(/[/\\]/).pop());
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
            showAlert("Choose at least one stem type to extract (Vocals, Drums, etc.)");
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
        setHintState(el.separateHint, "Installing Demucs… This may take a few minutes.", "info", el.installDemucsBtn);
        apiWithSpinner(el.installDemucsBtn, "POST", "/demucs/install", {}, function(err, data) {
            if (err || (data && data.error)) {
                var errMsg = data ? (data.suggestion ? data.error + " \u2014 " + data.suggestion : data.error) : 'Unknown error';
                setHintState(el.separateHint, "Installation failed: " + errMsg, "error", el.installDemucsBtn);
            } else {
                hideHintState(el.separateHint, el.installDemucsBtn);
                capabilities.separation = true;
                updateButtons();
                showAlert("Demucs installed successfully!");
            }
        }, 300000);
    }

    function measureLoudness() {
        el.loudnessMeter.classList.remove("hidden");
        el.meterLUFS.textContent = "Measuring…";
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
    
    function installDepth() {
        startInlineInstallJob({
            endpoint: "/video/depth/install",
            hintEl: el.depthHint,
            actionBtn: el.installDepthBtn,
            startMessage: "Installing Depth Anything V2… This may take several minutes.",
            onSuccess: function() {
                capabilities.depth_effects = true;
                updateButtons();
                showAlert("Depth Anything V2 installed successfully!");
            }
        });
    }

    function installEmotion() {
        startInlineInstallJob({
            endpoint: "/video/emotion/install",
            hintEl: el.emotionHint,
            actionBtn: el.installEmotionBtn,
            startMessage: "Installing emotion analysis… This may take a few minutes.",
            onSuccess: function() {
                capabilities.deepface = true;
                updateButtons();
                showAlert("Emotion analysis installed successfully!");
            }
        });
    }

    function installCrisperWhisper() {
        startInlineInstallJob({
            endpoint: "/audio/crisper-whisper/install",
            hintEl: el.fillersHint,
            actionBtn: el.installCrisperWhisperBtn,
            startMessage: "Installing CrisperWhisper… This may take a few minutes.",
            onSuccess: function() {
                capabilities.crisper_whisper = true;
                if (el.fillerBackend) {
                    el.fillerBackend.value = "crisper";
                    if (el.fillerBackend._customDropdown) el.fillerBackend._customDropdown.updateText();
                }
                updateButtons();
                showAlert("CrisperWhisper installed successfully!");
            }
        });
    }

    function installBrollGenerate() {
        startInlineInstallJob({
            endpoint: "/video/broll-generate/install",
            hintEl: el.brollGenHint,
            actionBtn: el.installBrollGenBtn,
            startMessage: "Installing AI B-roll generation dependencies… This may take several minutes.",
            onSuccess: function() {
                capabilities.broll_generate = true;
                updateButtons();
                showAlert("AI B-roll generation installed successfully!");
            }
        });
    }

    function installMultimodalDiarize() {
        startInlineInstallJob({
            endpoint: "/video/multimodal-diarize/install",
            hintEl: el.mmDiarizeHint,
            actionBtn: el.installMmDiarizeBtn,
            startMessage: "Installing multimodal diarization dependencies… This may take several minutes.",
            onSuccess: function() {
                capabilities.multimodal_diarize = true;
                updateButtons();
                showAlert("Multimodal diarization installed successfully!");
            }
        });
    }

    function installWatermark() {
        setHintState(el.watermarkHint, "Installing watermark remover… This may take several minutes.", "info", el.installWatermarkBtn);
        apiWithSpinner(el.installWatermarkBtn, "POST", "/watermark/install", {}, function(err, data) {
            if (err || (data && data.error)) {
                var errMsg = data ? (data.suggestion ? data.error + " \u2014 " + data.suggestion : data.error) : 'Unknown error';
                setHintState(el.watermarkHint, "Installation failed: " + errMsg, "error", el.installWatermarkBtn);
            } else {
                hideHintState(el.watermarkHint, el.installWatermarkBtn);
                capabilities.watermark_removal = true;
                updateButtons();
                showAlert("Watermark remover installed successfully!");
            }
        }, 300000);
    }
    
    function autoDetectWatermark() {
        if (!selectedPath) return;
        var btn = document.getElementById("autoDetectWatermarkBtn");
        var resEl = document.getElementById("wmDetectResult");
        var originalBtnText = rememberButtonText(btn);
        if (btn) { btn.disabled = true; setButtonText(btn, "Detecting…"); }
        api("POST", "/video/auto-detect-watermark", { filepath: selectedPath, prompt: (el.wmPrompt ? el.wmPrompt.value.trim() : "watermark") || "watermark" }, function (err, data) {
            if (btn) { btn.disabled = false; setButtonText(btn, originalBtnText); }
            if (err || !data) {
                setHintState(resEl, "Detection failed: " + ((err && err.message) || "Unknown error"), "error");
                return;
            }
            if (data.error) {
                setHintState(resEl, data.error + (data.suggestion ? " — " + data.suggestion : ""), "error");
                return;
            }
            if (data.x !== undefined) {
                // Fill in the detected region
                if (el.removeX) el.removeX.value = data.x;
                if (el.removeY) el.removeY.value = data.y;
                if (el.removeW) el.removeW.value = data.width;
                if (el.removeH) el.removeH.value = data.height;
                setHintState(resEl, "Detected at (" + data.x + ", " + data.y + ") — " + data.width + "×" + data.height + " px (" + (data.method || "auto") + ", " + safeFixed((data.confidence || 0) * 100, 0) + "% confidence)", "success");
                showToast("Watermark detected — region auto-filled", "success");
            } else {
                setHintState(resEl, "No watermark detected. Try adjusting the prompt.", "warning");
            }
        });
    }

    function runEmotionHighlights() {
        startJob("/video/emotion-highlights", {
            filepath: selectedPath,
            sample_interval: 1.0,
            min_intensity: 0.6,
            min_duration: 2.0,
        });
    }

    function runDepthEffect() {
        var effect = (document.getElementById("depthEffect") || {}).value || "bokeh";
        var modelSize = (document.getElementById("depthModelSize") || {}).value || "small";
        var endpoint = "/video/depth/" + effect;
        var payload = { filepath: selectedPath, output_dir: projectFolder, model_size: modelSize };
        if (effect === "bokeh") {
            payload.focus_point = parseFloat((document.getElementById("depthFocusPoint") || {}).value || "0.5");
            payload.blur_strength = parseInt((document.getElementById("depthBlurStrength") || {}).value || "25");
        } else if (effect === "parallax") {
            payload.zoom_amount = parseFloat((document.getElementById("depthZoomAmount") || {}).value || "1.15");
        }
        startJob(endpoint, payload);
    }

    function showDepthParams() {
        var effect = (document.getElementById("depthEffect") || {}).value || "bokeh";
        var bokehParams = document.getElementById("depthBokehParams");
        var parallaxParams = document.getElementById("depthParallaxParams");
        if (bokehParams) bokehParams.classList.toggle("hidden", effect !== "bokeh");
        if (parallaxParams) parallaxParams.classList.toggle("hidden", effect !== "parallax");
    }

    function runBrollPlan() {
        startJob("/video/broll-plan", { filepath: selectedPath, min_gap: 1.0, max_results: 15 });
    }

    function runBrollGenerate() {
        var prompt = (document.getElementById("brollGenPrompt") || {}).value || "";
        if (!prompt.trim()) { showAlert("Enter a description for the B-roll clip."); return; }
        var backend = (document.getElementById("brollGenBackend") || {}).value || "auto";
        var seedEl = document.getElementById("brollGenSeed");
        var seed = seedEl && seedEl.value ? parseInt(seedEl.value) : null;
        var payload = { prompt: prompt.trim(), backend: backend, output_dir: projectFolder, no_input: true };
        if (seed !== null && !isNaN(seed)) payload.seed = seed;
        startJob("/video/broll-generate", payload);
    }

    function runMultimodalDiarize() {
        var numSpeakers = (document.getElementById("mmDiarizeNumSpeakers") || {}).value || "";
        var sampleFps = parseFloat((document.getElementById("mmDiarizeSampleFps") || {}).value || "2");
        var confidence = parseFloat((document.getElementById("mmDiarizeConfidence") || {}).value || "0.5");
        var payload = { filepath: selectedPath, sample_fps: sampleFps, min_face_confidence: confidence };
        if (numSpeakers) payload.num_speakers = parseInt(numSpeakers);
        startJob("/video/multimodal-diarize", payload);
    }

    // Called from job completion handler to update multimodal diarize results
    function _onMmDiarizeComplete(result) {
        var resEl = document.getElementById("mmDiarizeResults");
        if (resEl) resEl.classList.remove("hidden");
        var r = result || {};
        var spkEl = document.getElementById("mmDiarizeSpeakers");
        var faceEl = document.getElementById("mmDiarizeFaces");
        var mapEl = document.getElementById("mmDiarizeMappings");
        if (spkEl) spkEl.textContent = r.num_speakers || 0;
        if (faceEl) faceEl.textContent = r.num_faces || 0;
        if (mapEl) mapEl.textContent = (r.mappings || []).length;
    }

    function socialConnect() {
        var platform = (document.getElementById("socialPlatform") || {}).value || "youtube";
        api("POST", "/social/auth-url", { platform: platform }, function(err, r) {
            if (err) { showAlert("OAuth error: " + err.message); return; }
            if (r && r.auth_url) {
                // Validate auth URL before passing to shell — prevent command injection
                var authUrl = String(r.auth_url);
                if (!/^https?:\/\//i.test(authUrl)) {
                    showAlert("Invalid authorization URL received from server.");
                    return;
                }
                // Open OAuth URL in user's browser
                if (typeof cep_node !== "undefined" && cep_node.require) {
                    cep_node.require("child_process").execFile("cmd", ["/c", "start", "", authUrl]);
                } else {
                    window.open(authUrl, "_blank");
                }
            showToast("Opening " + platform + " authorization page…", "info");
            } else {
                showAlert("OAuth not configured for " + platform + ". Set API credentials in environment variables.");
            }
        });
    }

    function socialUpload() {
        var platform = (document.getElementById("socialPlatform") || {}).value || "youtube";
        var title = (document.getElementById("socialTitle") || {}).value || "";
        var description = (document.getElementById("socialDescription") || {}).value || "";
        var privacy = (document.getElementById("socialPrivacy") || {}).value || "private";
        var payload = {
            filepath: selectedPath,
            platform: platform,
            title: title,
            description: description,
            privacy: privacy,
        };
        startJob("/social/upload", payload, function(result) {
            var resEl = document.getElementById("socialResult");
            if (resEl) resEl.classList.remove("hidden");
            var urlEl = document.getElementById("socialResultUrl");
            if (urlEl && result && result.url) {
                urlEl.href = result.url;
                urlEl.textContent = "View on " + platform;
            }
        });
    }

    function loadSocialPlatforms() {
        api("GET", "/social/platforms", null, function(err, r) {
            if (err) return;
            if (r && r.platforms) {
                var platform = (document.getElementById("socialPlatform") || {}).value || "";
                for (var i = 0; i < r.platforms.length; i++) {
                    if (r.platforms[i].platform === platform && r.platforms[i].connected) {
                        var badge = document.getElementById("socialConnectedBadge");
                        if (badge) { badge.classList.remove("hidden"); badge.textContent = "Connected as " + (r.platforms[i].username || ""); }
                        return;
                    }
                }
            }
        });
    }

    // ---- WebSocket Client ----
    var _ws = null;
    var _wsReconnectTimer = null;
    var _wsConnected = false;

    function wsConnect() {
        if (_ws && (_ws.readyState === WebSocket.OPEN || _ws.readyState === WebSocket.CONNECTING)) {
            showToast("WebSocket already connected", "info");
            return;
        }
        var port = 5680;
        var url = "ws://127.0.0.1:" + port;
        try {
            _ws = new WebSocket(url);
        } catch (e) {
            showToast("WebSocket connection failed", "warning");
            return;
        }

        _ws.onopen = function () {
            _wsConnected = true;
            _ws.send(JSON.stringify({ type: "identify", client_type: "cep", id: "cep-1" }));
            _ws.send(JSON.stringify({ type: "command", action: "subscribe", params: { events: ["progress", "job_complete", "job_error", "timeline"] }, id: "sub-1" }));
            _updateWsStatus();
            showToast("WebSocket connected", "success");
        };

        _ws.onmessage = function (evt) {
            try {
                var msg = JSON.parse(evt.data);
                _handleWsMessage(msg);
            } catch (e) { /* ignore parse errors */ }
        };

        _ws.onclose = function () {
            _wsConnected = false;
            _ws = null;
            _updateWsStatus();
            // Auto-reconnect after 5s
            if (!_wsReconnectTimer) {
                _wsReconnectTimer = setTimeout(function () {
                    _wsReconnectTimer = null;
                    wsConnect();
                }, 5000);
            }
        };

        _ws.onerror = function () {
            _wsConnected = false;
        };
    }

    function wsDisconnect() {
        if (_wsReconnectTimer) { clearTimeout(_wsReconnectTimer); _wsReconnectTimer = null; }
        if (_ws) { _ws.close(); _ws = null; }
        _wsConnected = false;
        _updateWsStatus();
    }

    function _handleWsMessage(msg) {
        if (msg.type === "progress" && msg.job_id) {
            // Update progress bar in real-time (no polling needed)
            var pct = msg.percent || 0;
            var message = msg.message || "";
            if (el.processingFill) el.processingFill.style.width = pct + "%";
            if (el.processingMsg && message) el.processingMsg.textContent = message;
        } else if (msg.type === "event" && msg.event === "job_complete") {
            // Job completed via WS — trigger poll to pick up result
            if (currentJob) pollJob();
        } else if (msg.type === "event" && msg.event === "job_error") {
            if (currentJob) pollJob();
        }
    }

    function _updateWsStatus() {
        var statusEl = document.getElementById("wsStatusText");
        var countEl = document.getElementById("wsClientCount");
        if (_wsConnected) {
            if (statusEl) statusEl.textContent = "Connected";
            if (statusEl) statusEl.style.color = "var(--accent)";
        } else {
            if (statusEl) statusEl.textContent = "Disconnected";
            if (statusEl) statusEl.style.color = "";
        }
        // Also fetch server-side status
        api("GET", "/ws/status", null, function (err, r) {
            if (err) return;
            if (r) {
                if (countEl) countEl.textContent = r.clients || 0;
                if (!r.running && statusEl && !_wsConnected) statusEl.textContent = "Server stopped";
            }
        });
    }

    function wsStartBridge() {
        api("POST", "/ws/start", {}, function (err, r) {
            if (err) { showAlert("WS start error: " + err.message); return; }
            if (r && r.success) {
                showToast("WebSocket bridge started", "success");
                setTimeout(function () { wsConnect(); }, 500);
            } else {
                showAlert(r && r.error ? r.error : "Failed to start WebSocket bridge");
            }
        });
    }

    function wsStopBridge() {
        wsDisconnect();
        api("POST", "/ws/stop", {}, function (err, r) {
            if (err) return;
            if (r && r.success) {
                showToast("WebSocket bridge stopped", "success");
                _updateWsStatus();
            }
        });
    }

    // ---- Engine Registry UI ----
    function loadEngineRegistry() {
        var grid = document.getElementById("engineRegistryGrid");
        if (!grid) return;
            grid.innerHTML = '<div class="hint">Loading engines…</div>';

        api("GET", "/engines", null, function (err, r) {
            if (err || !r || !r.engines) {
                grid.innerHTML = '<div class="hint">Could not load engine data.</div>';
                return;
            }
            var html = "";
            var domains = Object.keys(r.engines).sort();
            for (var i = 0; i < domains.length; i++) {
                var domain = domains[i];
                var info = r.engines[domain];
                var engines = info.engines || [];
                var active = info.active || "";
                var preferred = info.preferred || "";

                html += '<div class="engine-domain">';
                html += '<label class="param-label">' + esc(domain.replace(/_/g, " ")) + '</label>';
                html += '<select class="engine-select" data-domain="' + esc(domain) + '">';
                html += '<option value="">Auto (highest priority)</option>';
                for (var j = 0; j < engines.length; j++) {
                    var eng = engines[j];
                    var selected = (preferred === eng.name) ? " selected" : "";
                    var avail = eng.available ? "" : " (unavailable)";
                    var label = esc(eng.display_name) + " — " + esc(eng.quality) + "/" + esc(eng.speed) + avail;
                    if (eng.name === active) label += " *";
                    html += '<option value="' + esc(eng.name) + '"' + selected + '>' + label + '</option>';
                }
                html += '</select>';
                html += '</div>';
            }
            grid.innerHTML = html;

            // Bind change events for preference setting
            var selects = grid.querySelectorAll(".engine-select");
            for (var k = 0; k < selects.length; k++) {
                selects[k].addEventListener("change", function () {
                    var dom = this.getAttribute("data-domain");
                    var eng = this.value;
                    if (eng) {
                        api("POST", "/engines/preference", { domain: dom, engine: eng }, function (perr, r) {
                            if (perr) { showAlert("Error: " + perr.message); return; }
                            if (r && r.success) showToast("Engine preference saved", "success");
                            else showAlert(r && r.error ? r.error : "Failed to save preference");
                        });
                    }
                });
            }
        });
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
            var rembgBackend = el.vidAiRembgBackend ? el.vidAiRembgBackend.value : "rembg";
            startJob("/video/ai/rembg", {
                filepath: selectedPath,
                output_dir: projectFolder,
                backend: rembgBackend,
                model: el.vidAiRembgModel ? el.vidAiRembgModel.value : "birefnet-general",
                bg_color: el.vidAiRembgBg ? el.vidAiRembgBg.value : "",
                alpha_only: el.vidAiRembgAlpha ? el.vidAiRembgAlpha.checked : false,
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
        setHintState(el.vidAiHint, "Installing… This may take several minutes.", "info", el.installVidAiBtn);
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
        setHintState(el.proFxHint, "Installing Pedalboard…", "info", el.installPedalboardBtn);
        startJob("/audio/pro/install", { component: "pedalboard", no_input: true });
    }

    function runDeepFilter() {
        startJob("/audio/pro/deepfilter", {
            filepath: selectedPath,
            output_dir: projectFolder,
        });
    }

    function installDeepFilter() {
        setHintState(el.deepFilterHint, "Installing DeepFilterNet…", "info", el.installDeepFilterBtn);
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
        setHintState(el.faceHint, "Installing MediaPipe…", "info", el.installMediapipeBtn);
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
        showAlert("Step 1/2: Transcribing first, then translating…");
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
        setHintState(el.translateHint, "Installing NLLB translation…", "info", el.installNllbBtn);
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
        setHintState(el.karaokeHint, "Installing WhisperX…", "info", el.installWhisperxBtn);
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
        el.batchStatusText.textContent = "Starting batch: " + paths.length + " files…";

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
        el.batchStatusText.textContent = "Batch running: 0/" + data.total + " complete…";
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
    var _workflowPresets = []; // Loaded from backend

    function loadWorkflowPresets() {
        api("GET", "/workflow/presets", null, function (err, data) {
            if (err || !data) return;
            _workflowPresets = [];
            var sel = el.workflowPreset;
            if (!sel) return;
            sel.innerHTML = "";
            // Built-in presets
            var builtins = data.builtins || [];
            var customs = data.custom || [];
            var globalIdx = 0;
            if (builtins.length) {
                var optg = document.createElement("optgroup");
                optg.label = "Built-in";
                for (var i = 0; i < builtins.length; i++) {
                    var opt = document.createElement("option");
                    opt.value = "idx:" + globalIdx;
                    opt.textContent = builtins[i].name + " (" + (builtins[i].steps || []).length + " steps)";
                    optg.appendChild(opt);
                    _workflowPresets.push(builtins[i]);
                    globalIdx++;
                }
                sel.appendChild(optg);
            }
            if (customs.length) {
                var optg2 = document.createElement("optgroup");
                optg2.label = "Custom";
                for (var j = 0; j < customs.length; j++) {
                    var opt2 = document.createElement("option");
                    opt2.value = "idx:" + globalIdx;
                    opt2.textContent = customs[j].name + " (" + (customs[j].steps || []).length + " steps)";
                    optg2.appendChild(opt2);
                    _workflowPresets.push(customs[j]);
                    globalIdx++;
                }
                sel.appendChild(optg2);
            }
            if (!builtins.length && !customs.length) {
                sel.innerHTML = '<option value="" disabled selected>No presets available</option>';
            }
            // Show description on change
            sel.addEventListener("change", function () {
                var idx = _getWorkflowPresetIndex(sel.value);
                var desc = el.workflowPresetDesc;
                if (desc && idx >= 0 && _workflowPresets[idx]) {
                    desc.textContent = _workflowPresets[idx].description || "";
                }
            });
        });
    }

    function _getWorkflowPresetIndex(val) {
        // Value format: "idx:N" where N is the index into _workflowPresets
        if (!val) return -1;
        var parts = val.split(":");
        if (parts.length !== 2) return -1;
        var idx = parseInt(parts[1], 10);
        return isNaN(idx) ? -1 : idx;
    }

    function runWorkflowPreset() {
        var sel = el.workflowPreset;
        if (!sel || !sel.value || !selectedPath) {
            showAlert("Select a preset and a clip first.");
            return;
        }
        var idx = _getWorkflowPresetIndex(sel.value);
        var preset = _workflowPresets[idx];
        if (!preset || !preset.steps || !preset.steps.length) {
            showAlert("Invalid workflow preset.");
            return;
        }
        // Use server-side workflow runner for reliable chained execution
        startJob("/workflow/run", {
            filepath: selectedPath,
            workflow: { steps: preset.steps },
            output_dir: projectFolder,
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
            no_input: true,
        });
    }

    function installEdgeTts() {
        setHintState(el.ttsHint, "Installing Edge TTS…", "info", el.installEdgeTtsBtn);
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
        showAlert("Step 1/2: Transcribing first, then burning in captions…");
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
        showAlert("Step 1/2: Transcribing with word-level timing first…");
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
        showAlert("Step 2/2: Burning in captions…");
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
        showAlert("Step 2/2: Rendering animated captions…");
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
        showAlert("Step 2/2: Translating captions…");
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
    var _activeTranscriptSegmentIdx = -1;

    function getTranscriptSegments() {
        return transcriptData && transcriptData.segments ? transcriptData.segments : [];
    }

    // v1.9.34 (H): jump the Premiere playhead to a transcript segment's
    // start time. Quiet no-op outside Premiere so dev-mode use stays clean.
    function _jumpPlayheadToSegment(idx) {
        if (!inPremiere) return;
        var segs = getTranscriptSegments();
        if (!segs || idx < 0 || idx >= segs.length) return;
        var t = Number(segs[idx].start || 0);
        if (!isFinite(t) || t < 0) return;
        PremiereBridge.setPlayhead(t, function (result) {
            // Only warn on errors if the first click fails; subsequent
            // ones stay quiet to avoid spamming toasts on every tap.
            if (!_playheadSyncWarned && result && result.indexOf("error") !== -1) {
                try {
                    var r = JSON.parse(result);
                    if (r && r.error) {
                        _playheadSyncWarned = true;
                        showToast("Playhead sync unavailable: " + r.error, "warn");
                    }
                } catch (e) {}
            }
        });
    }
    var _playheadSyncWarned = false;

    function getTranscriptTotalDuration(data) {
        var segments = data && data.segments ? data.segments : [];
        var maxEnd = 0;
        for (var i = 0; i < segments.length; i++) {
            maxEnd = Math.max(maxEnd, Number(segments[i].end || segments[i].start || 0));
        }
        return maxEnd;
    }

    function renderTranscriptTimelineMeta(data) {
        if (!el.transcriptTimelineMeta) return;
        var segments = data && data.segments ? data.segments : [];
        if (!segments.length) {
            el.transcriptTimelineMeta.innerHTML =
                '<div class="transcript-timeline-stat is-empty">' +
                '<span class="transcript-timeline-stat-label">Editor</span>' +
                '<span class="transcript-timeline-stat-value">Awaiting transcript segments</span>' +
                '</div>';
            return;
        }

        var totalDuration = getTranscriptTotalDuration(data);
        var wordCount = Number(data && data.word_count || 0);
        var longest = 0;
        for (var i = 0; i < segments.length; i++) {
            longest = Math.max(longest, Math.max(0, Number(segments[i].end || 0) - Number(segments[i].start || 0)));
        }
        var avgDuration = segments.length ? (totalDuration / segments.length) : 0;
        el.transcriptTimelineMeta.innerHTML =
            '<div class="transcript-timeline-stat">' +
            '<span class="transcript-timeline-stat-label">Segments</span>' +
            '<span class="transcript-timeline-stat-value">' + segments.length + '</span>' +
            '</div>' +
            '<div class="transcript-timeline-stat">' +
            '<span class="transcript-timeline-stat-label">Runtime</span>' +
            '<span class="transcript-timeline-stat-value">' + fmtDur(totalDuration) + '</span>' +
            '</div>' +
            '<div class="transcript-timeline-stat">' +
            '<span class="transcript-timeline-stat-label">Pace</span>' +
            '<span class="transcript-timeline-stat-value">' + safeFixed(avgDuration, 1) + 's avg</span>' +
            '</div>' +
            '<div class="transcript-timeline-stat">' +
            '<span class="transcript-timeline-stat-label">Longest</span>' +
            '<span class="transcript-timeline-stat-value">' + safeFixed(longest, 1) + 's</span>' +
            '</div>';
    }

    function updateTranscriptTimelineStatus(idx) {
        if (!el.transcriptTimelineStatus) return;
        var segments = getTranscriptSegments();
        if (idx == null || idx < 0 || idx >= segments.length) {
            el.transcriptTimelineStatus.textContent = "Select a segment to focus the edit.";
            return;
        }
        var seg = segments[idx];
        var duration = Math.max(0, Number(seg.end || 0) - Number(seg.start || 0));
        el.transcriptTimelineStatus.textContent = "Segment " + (idx + 1) + " · " + fmtDur(seg.start || 0) + " to " + fmtDur(seg.end || 0) + " · " + safeFixed(duration, 1) + "s";
    }

    function setTranscriptTimelineEmptyState(message) {
        if (!el.transcriptTimeline) return;
        el.transcriptTimeline.removeAttribute("aria-activedescendant");
        el.transcriptTimeline.innerHTML =
            '<div class="transcript-timeline-empty">' + esc(message || "Transcript segments will appear here.") + '</div>';
    }

    function updateTranscriptTimelinePlayhead(idx) {
        if (!el.transcriptTimeline) return;
        var playhead = el.transcriptTimeline.querySelector(".transcript-timeline-playhead");
        if (!playhead) return;
        var segments = getTranscriptSegments();
        var totalDuration = getTranscriptTotalDuration({ segments: segments });
        if (idx == null || idx < 0 || idx >= segments.length || totalDuration <= 0) {
            playhead.classList.remove("is-visible");
            playhead.style.left = "0%";
            return;
        }
        var seg = segments[idx];
        var start = Math.max(0, Number(seg.start || 0));
        var left = Math.max(0, Math.min(100, (start / totalDuration) * 100));
        playhead.style.left = left.toFixed(3) + "%";
        playhead.classList.add("is-visible");
    }

    function renderTranscriptTimeline(data) {
        if (!el.transcriptTimeline || !el.transcriptTimelineRuler) return;
        var segments = data && data.segments ? data.segments : [];
        var totalDuration = getTranscriptTotalDuration(data);
        renderTranscriptTimelineMeta(data);
        if (!segments.length || totalDuration <= 0) {
            el.transcriptTimelineRuler.innerHTML = '<span>0:00</span><span>0:00</span><span>0:00</span>';
            setTranscriptTimelineEmptyState("Transcript segments will appear here once the clip is ready.");
            updateTranscriptTimelineStatus(-1);
            return;
        }

        el.transcriptTimelineRuler.innerHTML =
            '<span>0:00</span>' +
            '<span>' + fmtDur(totalDuration / 2) + '</span>' +
            '<span>' + fmtDur(totalDuration) + '</span>';

        var html = "";
        for (var i = 0; i < segments.length; i++) {
            var seg = segments[i];
            var start = Math.max(0, Number(seg.start || 0));
            var end = Math.max(start, Number(seg.end || seg.start || 0));
            var duration = Math.max(0.12, end - start);
            var left = Math.min(96, (start / totalDuration) * 100);
            var width = Math.max(2.8, (duration / totalDuration) * 100);
            if (left + width > 100) width = Math.max(2.8, 100 - left);
            var preview = String(seg.text || "").replace(/\s+/g, " ").trim();
            if (preview.length > 72) preview = preview.substring(0, 72) + "…";
            html += '<button type="button" class="transcript-timeline-seg" data-idx="' + i + '" ' +
                'id="transcriptTimelineSeg' + i + '" role="option" tabindex="-1" ' +
                'style="left:' + left.toFixed(3) + '%;width:' + width.toFixed(3) + '%;" ' +
                'aria-selected="false" ' +
                'aria-label="Segment ' + (i + 1) + ', ' + fmtDur(start) + ' to ' + fmtDur(end) + (preview ? ', ' + esc(preview) : '') + '" ' +
                'title="' + esc(preview || ('Segment ' + (i + 1))) + '">' +
                '<span class="transcript-timeline-chip">' + (i + 1) + '</span>' +
                '</button>';
        }
        el.transcriptTimeline.innerHTML = '<span class="transcript-timeline-playhead" aria-hidden="true"></span>' + html;
        updateTranscriptTimelinePlayhead(_activeTranscriptSegmentIdx);
    }

    function focusTranscriptSegment(idx, options) {
        if (!el.transcriptSegments) return;
        var rows = el.transcriptSegments.querySelectorAll(".transcript-seg");
        var clips = el.transcriptTimeline ? el.transcriptTimeline.querySelectorAll(".transcript-timeline-seg") : [];
        var activeClip = null;
        for (var i = 0; i < rows.length; i++) {
            rows[i].classList.toggle("is-active", i === idx);
        }
        for (var j = 0; j < clips.length; j++) {
            var isActive = j === idx;
            clips[j].classList.toggle("is-active", isActive);
            clips[j].setAttribute("aria-selected", isActive ? "true" : "false");
            clips[j].tabIndex = isActive ? 0 : -1;
            if (isActive) activeClip = clips[j];
        }
        if (idx == null || idx < 0 || idx >= rows.length) {
            _activeTranscriptSegmentIdx = -1;
            if (el.transcriptTimeline) el.transcriptTimeline.removeAttribute("aria-activedescendant");
            updateTranscriptTimelineStatus(-1);
            updateTranscriptTimelinePlayhead(-1);
            return;
        }
        _activeTranscriptSegmentIdx = idx;
        if (el.transcriptTimeline && activeClip) {
            el.transcriptTimeline.setAttribute("aria-activedescendant", activeClip.id);
        }
        updateTranscriptTimelineStatus(idx);
        updateTranscriptTimelinePlayhead(idx);

        var row = rows[idx];
        if (row && (!options || options.scroll !== false) && row.scrollIntoView) {
            row.scrollIntoView({ block: "nearest", behavior: options && options.instant ? "auto" : "smooth" });
        }
        if (activeClip && (!options || options.scrollTimeline !== false) && activeClip.scrollIntoView) {
            activeClip.scrollIntoView({ block: "nearest", inline: "nearest", behavior: options && options.instant ? "auto" : "smooth" });
        }
        if (options && options.focusControl === "timeline" && activeClip && activeClip.focus) {
            activeClip.focus();
        }
        if (options && (options.focusInput || options.focusControl === "input") && row) {
            var ta = row.querySelector(".transcript-seg-text");
            if (ta) ta.focus();
        }
    }

    function refreshTranscriptSearch() {
        if (!el.transcriptSearchInput) return;
        doTranscriptSearch((el.transcriptSearchInput.value || "").trim());
    }

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
        refreshTranscriptSearch();
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
        var runtime = getTranscriptTotalDuration(data);
        var lang = (data.language || "en").toUpperCase();
        el.transcriptInfo.textContent = wordCount + " words · " + segCount + " segments · " + fmtDur(runtime) + " runtime · " + lang;
        if (!data.segments || !data.segments.length) {
            el.transcriptSegments.innerHTML = '<div class="transcript-empty-state">Transcribe a clip to start shaping dialogue, timing, and cut decisions.</div>';
            renderTranscriptTimeline(data);
            transcriptHistory = [];
            transcriptHistoryIdx = -1;
            updateUndoRedoButtons();
            focusTranscriptSegment(-1, { scroll: false, scrollTimeline: false, instant: true });
            refreshTranscriptSearch();
            return;
        }

        var html = "";
        for (var i = 0; i < data.segments.length; i++) {
            var seg = data.segments[i];
            var start = Number(seg.start || 0);
            var end = Number(seg.end || seg.start || 0);
            var timeStr = fmtDur(start) + " - " + fmtDur(end);
            var duration = Math.max(0, end - start);
            html += '<article class="transcript-seg" data-idx="' + i + '">'
                + '<div class="transcript-seg-header">'
                + '<div class="transcript-seg-primary">'
                + '<span class="transcript-seg-index">' + (i + 1) + '</span>'
                + '<span class="transcript-seg-time">' + timeStr + '</span>'
                + '</div>'
                + '<span class="transcript-seg-duration">' + safeFixed(duration, 1) + 's</span>'
                + '</div>'
                + '<textarea class="transcript-seg-text" data-idx="' + i + '" rows="1" aria-label="Transcript segment ' + (i + 1) + '">' + esc(seg.text) + '</textarea>'
                + '</article>';
        }
        el.transcriptSegments.innerHTML = html;
        renderTranscriptTimeline(data);

        // Auto-resize textareas
        var textareas = el.transcriptSegments.querySelectorAll(".transcript-seg-text");
        for (var i = 0; i < textareas.length; i++) {
            autoResize(textareas[i]);
        }

        // Reset history and take initial snapshot
        transcriptHistory = [];
        transcriptHistoryIdx = -1;
        snapshotTranscript();
        focusTranscriptSegment(Math.min(Math.max(_activeTranscriptSegmentIdx, 0), segCount - 1), { scroll: false, scrollTimeline: false, instant: true });
        refreshTranscriptSearch();
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
        el.transcriptSegments.addEventListener("click", function (e) {
            var row = e.target.closest(".transcript-seg");
            if (!row) return;
            var idx = parseInt(row.getAttribute("data-idx"), 10);
            if (!(idx >= 0)) return;
            focusTranscriptSegment(idx, { scroll: false, scrollTimeline: true });
            // v1.9.34 (H): jump the Premiere playhead to this segment's
            // start time. Silently no-op when outside Premiere.
            _jumpPlayheadToSegment(idx);
            if (!e.target.classList.contains("transcript-seg-text")) {
                var ta = row.querySelector(".transcript-seg-text");
                if (ta) ta.focus();
            }
        });
        el.transcriptSegments.addEventListener("focusin", function (e) {
            var ta = e.target;
            if (!ta || !ta.classList.contains("transcript-seg-text")) return;
            var idx = parseInt(ta.getAttribute("data-idx"), 10);
            if (idx >= 0) focusTranscriptSegment(idx, { scroll: true, scrollTimeline: true });
        });
        if (el.transcriptTimeline) {
            el.transcriptTimeline.addEventListener("click", function (e) {
                var btn = e.target.closest(".transcript-timeline-seg");
                if (!btn) return;
                var idx = parseInt(btn.getAttribute("data-idx"), 10);
                if (idx >= 0) {
                    focusTranscriptSegment(idx, { focusInput: true });
                    _jumpPlayheadToSegment(idx);
                }
            });
            el.transcriptTimeline.addEventListener("keydown", function (e) {
                var btn = e.target.closest(".transcript-timeline-seg");
                if (!btn) return;
                var idx = parseInt(btn.getAttribute("data-idx"), 10);
                if (!(idx >= 0)) return;
                if (e.key === "ArrowRight" || e.key === "ArrowDown") {
                    e.preventDefault();
                    focusTranscriptSegment(Math.min(idx + 1, getTranscriptSegments().length - 1), { focusControl: "timeline" });
                } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
                    e.preventDefault();
                    focusTranscriptSegment(Math.max(idx - 1, 0), { focusControl: "timeline" });
                } else if (e.key === "Home") {
                    e.preventDefault();
                    focusTranscriptSegment(0, { focusControl: "timeline" });
                } else if (e.key === "End") {
                    e.preventDefault();
                    focusTranscriptSegment(getTranscriptSegments().length - 1, { focusControl: "timeline" });
                } else if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    focusTranscriptSegment(idx, { focusInput: true });
                }
            });
        }
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
        showAlert("Installing faster-whisper… This may take a minute.");
        startJob("/install-whisper", { backend: "faster-whisper", no_input: true });
    }

    function reinstallWhisper() {
        var cpuMode = el.whisperCpuMode.checked;
        showAlert("Reinstalling Whisper" + (cpuMode ? " in CPU mode" : "") + "… Please wait.");
        startJob("/whisper/reinstall", { backend: "faster-whisper", cpu_mode: cpuMode, no_input: true });
    }

    function clearWhisperCache() {
        showAlert("Clearing Whisper cache…");
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
                    showAlert("CPU mode enabled. Transcription may be slower but more stable.");
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
        showAlert("Restarting backend…");
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
            autoImport: el.settingsAutoImport ? el.settingsAutoImport.checked : true,
            autoOpen: el.settingsAutoOpen ? el.settingsAutoOpen.checked : false,
            showNotifications: el.settingsShowNotifications ? el.settingsShowNotifications.checked : true,
            outputDir: el.settingsOutputDir ? el.settingsOutputDir.value : "",
            defaultModel: el.settingsDefaultModel ? el.settingsDefaultModel.value : "medium",
            lang: el.settingsLang ? el.settingsLang.value : "en"
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
                if (settings.autoImport !== undefined && el.settingsAutoImport) el.settingsAutoImport.checked = settings.autoImport;
                if (settings.autoOpen !== undefined && el.settingsAutoOpen) el.settingsAutoOpen.checked = settings.autoOpen;
                if (settings.showNotifications !== undefined && el.settingsShowNotifications) el.settingsShowNotifications.checked = settings.showNotifications;
                if (settings.outputDir && el.settingsOutputDir) el.settingsOutputDir.value = settings.outputDir;
                if (settings.defaultModel && el.settingsDefaultModel) el.settingsDefaultModel.value = settings.defaultModel;
            }
        } catch (e) {
            // localStorage may not be available
        }
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
        // Safe slider binding: skips if slider or value element is missing from DOM
        function _bindSlider(sliderId, valId, fmt) {
            var slider = el[sliderId], valEl = el[valId];
            if (!slider || !valEl) return;
            slider.addEventListener("input", function () { valEl.textContent = fmt(this.value); });
        }

        // Silence sliders
        _bindSlider("threshold", "thresholdVal", function (v) { return v + " dB"; });
        _bindSlider("minDuration", "minDurationVal", function (v) { return v + "s"; });
        if (el.silencePreset) {
            el.silencePreset.addEventListener("change", function () {
                if (el.customSilenceSettings) el.customSilenceSettings.classList.toggle("hidden", this.value !== "");
            });
        }

        // Audio sliders
        _bindSlider("denoiseStrength", "denoiseStrengthVal", function (v) { return v; });
        _bindSlider("beatSensitivity", "beatSensitivityVal", function (v) { return v; });

        // Video sliders
        _bindSlider("wmMaxBbox", "wmMaxBboxVal", function (v) { return v + "%"; });
        _bindSlider("wmFrameSkip", "wmFrameSkipVal", function (v) { return v; });
        _bindSlider("sceneThreshold", "sceneThresholdVal", function (v) { return safeFixed(parseFloat(v), 2); });
        _bindSlider("minSceneLen", "minSceneLenVal", function (v) { return v + "s"; });

        // Video FX sliders
        _bindSlider("vfxStabSmoothing", "vfxStabSmoothingVal", function (v) { return v; });
        _bindSlider("vfxStabZoom", "vfxStabZoomVal", function (v) { return v + "%"; });
        _bindSlider("vfxVignetteIntensity", "vfxVignetteIntensityVal", function (v) { return v; });
        _bindSlider("vfxGrainIntensity", "vfxGrainIntensityVal", function (v) { return v; });
        _bindSlider("vfxChromakeySim", "vfxChromakeySimVal", function (v) { return safeFixed(parseFloat(v), 2); });
        _bindSlider("vfxChromakeyBlend", "vfxChromakeyBlendVal", function (v) { return safeFixed(parseFloat(v), 2); });
        _bindSlider("vfxLutIntensity", "vfxLutIntensityVal", function (v) { return v; });

        // Video AI sliders
        _bindSlider("vidAiDenoiseStrength", "vidAiDenoiseStrengthVal", function (v) { return v; });

        // Face blur slider
        _bindSlider("faceBlurStrength", "faceBlurStrengthVal", function (v) { return v; });

        // Style transfer slider
        _bindSlider("styleIntensity", "styleIntensityVal", function (v) { return v; });

        // Karaoke font size slider
        _bindSlider("karaokeFontSize", "karaokeFontSizeVal", function (v) { return v + "px"; });

        // TTS rate slider
        if (el.ttsRate && el.ttsRateVal) {
            el.ttsRate.addEventListener("input", function () {
                var v = parseInt(this.value);
                el.ttsRateVal.textContent = (v >= 0 ? "+" : "") + v + "%";
            });
        }

        // SFX sliders
        _bindSlider("toneFreq", "toneFreqVal", function (v) { return v + " Hz"; });
        _bindSlider("sfxDuration", "sfxDurationVal", function (v) { return v + "s"; });

        // Speed multiplier slider
        _bindSlider("speedMultiplier", "speedMultiplierVal", function (v) { return v + "x"; });

        // LUT intensity slider
        _bindSlider("lutIntensity", "lutIntensityVal", function (v) { return v; });

        // Duck sliders
        _bindSlider("duckMusicVol", "duckMusicVolVal", function (v) { return v; });
        _bindSlider("duckAmount", "duckAmountVal", function (v) { return v; });

        // Phase 6 sliders
        _bindSlider("chromaTol", "chromaTolVal", function (v) { return v; });
        _bindSlider("pipScale", "pipScaleVal", function (v) { return v; });
        _bindSlider("blendOpacity", "blendOpacityVal", function (v) { return v; });
        _bindSlider("transDur", "transDurVal", function (v) { return v + "s"; });
        _bindSlider("particleDensity", "particleDensityVal", function (v) { return v; });
        _bindSlider("titleDur", "titleDurVal", function (v) { return v + "s"; });
        _bindSlider("titleFontSize", "titleFontSizeVal", function (v) { return v + "px"; });
        _bindSlider("ccExposure", "ccExposureVal", function (v) { return v; });
        _bindSlider("ccContrast", "ccContrastVal", function (v) { return v; });
        _bindSlider("ccSaturation", "ccSaturationVal", function (v) { return v; });
        _bindSlider("ccTemp", "ccTempVal", function (v) { return v; });
        _bindSlider("ccShadows", "ccShadowsVal", function (v) { return v; });
        _bindSlider("ccHighlights", "ccHighlightsVal", function (v) { return v; });
        _bindSlider("animCapFontSize", "animCapFontSizeVal", function (v) { return v + "px"; });
        _bindSlider("animCapWpl", "animCapWplVal", function (v) { return v; });
        _bindSlider("musicAiDur", "musicAiDurVal", function (v) { return v + "s"; });
        _bindSlider("musicAiTemp", "musicAiTempVal", function (v) { return v; });
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

    var NOTIFICATION_TONE_CLASSES = ["is-info", "is-success", "is-warning", "is-error"];

    function applyNotificationTone(node, tone) {
        if (!node) return;
        for (var i = 0; i < NOTIFICATION_TONE_CLASSES.length; i++) {
            node.classList.remove(NOTIFICATION_TONE_CLASSES[i]);
        }
        node.classList.add("is-" + (tone || "info"));
    }

    function inferNotificationTone(message, errorData, explicitType) {
        if (explicitType && /^(success|error|warning|info)$/.test(explicitType)) {
            return explicitType;
        }
        var lower = String(message || "").toLowerCase();
        if (errorData && (errorData.error || errorData.message || errorData.code)) {
            return "error";
        }
        if (/(^error\b|failed|failure|couldn't|could not|invalid|unable|not configured|unexpected|import error|oauth error)/.test(lower)) {
            return "error";
        }
        if (/(success|saved|loaded|opened|exported|installed successfully|complete|completed|deleted|enabled|disabled|cleared|copied|refreshed|queue cleared|succeeded)/.test(lower)) {
            return "success";
        }
        if (/(select|choose|enter|make sure|no clip|no clips|no cuts|no markers|no items|no project media|required|another task is in progress|connection required)/.test(lower)) {
            return "warning";
        }
        return "info";
    }

    function getNotificationHeading(tone, message) {
        var lower = String(message || "").toLowerCase();
        if (tone === "success") {
            return /(saved|loaded|opened|exported|copied|ready)/.test(lower) ? "Ready" : "Done";
        }
        if (tone === "warning") {
            return /(select|choose|enter|required)/.test(lower) ? "Action needed" : "Heads up";
        }
        if (tone === "error") {
            return "Needs attention";
        }
        if (/(step \d+\/\d+|installing|reinstalling|restarting|loading|checking|processing|transcribing|translating|burning|indexing)/.test(lower)) {
            return "In progress";
        }
        return "Status update";
    }

    function getNotificationIconSvg(tone) {
        switch (tone) {
            case "success":
                return '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.85" stroke-linecap="round" stroke-linejoin="round"><circle cx="10" cy="10" r="7.25"/><path d="M6.8 10.2l2.1 2.15 4.3-4.45"/></svg>';
            case "warning":
                return '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M10 3.4 16.2 14a1.15 1.15 0 0 1-.99 1.72H4.79A1.15 1.15 0 0 1 3.8 14L10 3.4Z"/><path d="M10 7.35v3.9"/><circle cx="10" cy="13.45" r="0.9" fill="currentColor" stroke="none"/></svg>';
            case "error":
                return '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="10" cy="10" r="7.25"/><path d="m7.35 7.35 5.3 5.3"/><path d="m12.65 7.35-5.3 5.3"/></svg>';
            default:
                return '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="10" cy="10" r="7.25"/><path d="M10 6.4v4.45"/><circle cx="10" cy="13.85" r="0.9" fill="currentColor" stroke="none"/></svg>';
        }
    }

    function showAlert(msg, errorData) {
        var display = enhanceError(msg, errorData);
        var tone = inferNotificationTone(display, errorData);
        var heading = getNotificationHeading(tone, display);
        applyNotificationTone(el.alertBanner, tone);
        if (el.alertIcon) {
            el.alertIcon.innerHTML = getNotificationIconSvg(tone);
        }
        if (el.alertEyebrow) {
            el.alertEyebrow.textContent = heading;
        }
        el.alertText.textContent = display;
        // Remove any previous action link
        var oldLink = el.alertBanner.querySelector(".alert-action-link");
        if (oldLink) oldLink.parentNode.removeChild(oldLink);
        // If errorData has an error code with a tab action, add a clickable nav link
        var action = getErrorCodeAction(errorData);
        if (action && action.tab) {
            var link = document.createElement("button");
            link.type = "button";
            link.className = "alert-action-link";
            link.textContent = "Open " + action.tab.charAt(0).toUpperCase() + action.tab.slice(1);
            link.addEventListener("click", function () {
                navigateToTab(action.tab, action.sub || null);
                el.alertBanner.classList.add("hidden");
            });
            el.alertText.parentNode.appendChild(link);
        }
        el.alertBanner.classList.remove("hidden");
        if (_alertTimer) clearTimeout(_alertTimer);
        _alertTimer = setTimeout(function () { el.alertBanner.classList.add("hidden"); }, 15000);
    }

    function showErrorWithAction(errorData) {
        var msg = errorData.error || errorData.message || "Unknown error";
        showAlert(msg, errorData);
    }

    // ================================================================
    // Session Context — "Where you left off" (v1.9.27)
    // ================================================================
    // Dismissal is persisted per-session; a new connect will re-show
    // only when new completed/interrupted jobs exist.
    var _sessionCtxLoaded = false;
    var _sessionCtxDismissedAt = 0;

    function _sessionCtxRelativeTime(unixSec) {
        if (!unixSec) return "";
        var delta = Date.now() / 1000 - unixSec;
        if (delta < 45) return "just now";
        if (delta < 90) return "1 min ago";
        if (delta < 3600) return Math.round(delta / 60) + " min ago";
        if (delta < 7200) return "1 hr ago";
        if (delta < 86400) return Math.round(delta / 3600) + " hr ago";
        if (delta < 172800) return "yesterday";
        return Math.round(delta / 86400) + " days ago";
    }

    function _sessionCtxClipName(job) {
        var path = job.filepath || "";
        if (!path) return "No clip";
        var parts = path.split(/[\\/]/);
        return parts[parts.length - 1] || path;
    }

    function _sessionCtxOpText(job) {
        var t = (job.type || "unknown").replace(/[_-]/g, " ");
        return t.charAt(0).toUpperCase() + t.slice(1);
    }

    function _sessionCtxResultPath(job) {
        var r = job.result;
        if (!r || typeof r !== "object") return "";
        return r.output_path || r.xml_path || r.overlay_path || r.srt_path ||
               (Array.isArray(r.output_paths) && r.output_paths[0]) || "";
    }

    function showSessionContext() {
        if (_sessionCtxLoaded) return;
        _sessionCtxLoaded = true;
        if (!el.sessionContext || !el.sessionContextBody) return;

        el.sessionContext.classList.remove("hidden");
        el.sessionContextBody.innerHTML =
            '<div class="session-context-loading">Loading recent work…</div>';

        var historyPending = true, interruptedPending = true;
        var historyData = [], interruptedData = [];

        function _maybeRender() {
            if (historyPending || interruptedPending) return;
            renderSessionContext(historyData, interruptedData);
        }

        api("GET", "/jobs/history?limit=5&status=complete", null, function (err, data) {
            historyData = (!err && Array.isArray(data)) ? data : [];
            historyPending = false;
            _maybeRender();
        });
        api("GET", "/jobs/interrupted", null, function (err, data) {
            interruptedData = (!err && Array.isArray(data)) ? data : [];
            interruptedPending = false;
            _maybeRender();
        });
    }

    function renderSessionContext(history, interrupted) {
        if (!el.sessionContextBody) return;

        // Nothing to show — hide the card silently. First run of the app
        // on a clean machine ends up here.
        if ((!history || !history.length) && (!interrupted || !interrupted.length)) {
            el.sessionContext.classList.add("hidden");
            return;
        }

        el.sessionContextBody.innerHTML = "";
        var frag = document.createDocumentFragment();

        // Interrupted banner (at most one even if multiple; users just
        // need to know it happened, details are in the history tab).
        if (interrupted && interrupted.length) {
            var warn = document.createElement("div");
            warn.className = "session-context-interrupted";
            warn.setAttribute("role", "status");
            var msg = document.createElement("span");
            msg.className = "session-context-interrupted-msg";
            msg.textContent = interrupted.length +
                " job" + (interrupted.length === 1 ? "" : "s") +
                " interrupted when the server restarted.";
            var openHistory = document.createElement("button");
            openHistory.type = "button";
            openHistory.className = "session-context-action session-context-interrupted-btn";
            openHistory.textContent = "View history";
            openHistory.addEventListener("click", function () {
                dismissSessionContext();
                if (el.jobHistory && !el.jobHistory.classList.contains("open")) {
                    el.jobHistory.classList.add("open");
                }
                if (el.jobHistoryToggle) {
                    setExpanded(el.jobHistoryToggle, true);
                    el.jobHistoryToggle.scrollIntoView({ behavior: "smooth", block: "center" });
                }
            });
            warn.appendChild(msg);
            warn.appendChild(openHistory);
            frag.appendChild(warn);
        }

        if (history && history.length) {
            var list = document.createElement("ul");
            list.className = "session-context-list";
            for (var i = 0; i < history.length; i++) {
                list.appendChild(_buildSessionRow(history[i]));
            }
            frag.appendChild(list);
        }

        el.sessionContextBody.appendChild(frag);
    }

    function _buildSessionRow(job) {
        var row = document.createElement("li");
        row.className = "session-context-item";

        var copy = document.createElement("div");
        copy.className = "session-context-item-copy";

        var title = document.createElement("div");
        title.className = "session-context-item-title";
        var dot = document.createElement("span");
        dot.className = "session-context-item-status";
        dot.setAttribute("data-status", job.status || "complete");
        dot.setAttribute("aria-hidden", "true");
        title.appendChild(dot);
        var label = document.createElement("span");
        label.textContent = _sessionCtxOpText(job) + " · " + _sessionCtxClipName(job);
        title.appendChild(label);

        var meta = document.createElement("div");
        meta.className = "session-context-item-meta";
        meta.textContent = _sessionCtxRelativeTime(job.created);

        copy.appendChild(title);
        copy.appendChild(meta);
        row.appendChild(copy);

        var actions = document.createElement("div");
        actions.className = "session-context-item-actions";

        var outputPath = _sessionCtxResultPath(job);
        if (outputPath) {
            var openBtn = _sessionCtxActionBtn("Open", "Open output file", function () {
                _sessionCtxOpenPath(outputPath, "open");
            });
            actions.appendChild(openBtn);

            var revealBtn = _sessionCtxActionBtn("Reveal", "Reveal in file manager", function () {
                _sessionCtxOpenPath(outputPath, "reveal");
            });
            actions.appendChild(revealBtn);
        }

        if (job.endpoint && job.payload) {
            var rerunBtn = _sessionCtxActionBtn("Re-run", "Run the same job again", function () {
                _sessionCtxRerun(job);
            });
            actions.appendChild(rerunBtn);
            // v1.9.30 (J): if the current selection differs from the original
            // clip, offer a one-click "apply same params to selection".
            var originalPath = (job.payload && job.payload.filepath) || "";
            if (originalPath && selectedPath && selectedPath !== originalPath) {
                var applyBtn = _sessionCtxActionBtn(
                    "Apply to selection",
                    "Re-run on the currently selected clip",
                    function () { _sessionCtxApplyToSelection(job); }
                );
                actions.appendChild(applyBtn);
            }
        }

        row.appendChild(actions);
        return row;
    }

    function _sessionCtxActionBtn(label, title, onClick) {
        var b = document.createElement("button");
        b.type = "button";
        b.className = "session-context-action";
        b.textContent = label;
        b.title = title;
        b.addEventListener("click", onClick);
        return b;
    }

    function _sessionCtxOpenPath(path, mode) {
        api("POST", "/system/open-path", { path: path, mode: mode }, function (err, data) {
            if (err) {
                showToast("Couldn't " + mode + ": " + (err.error || err.message || err), "error");
            } else if (data && data.ok) {
                showToast(mode === "reveal" ? "Revealed in file manager" : "Opened", "success");
            }
        });
    }

    function _sessionCtxRerun(job) {
        if (!job.endpoint || !job.payload) {
            showToast("Nothing to re-run — this job wasn't recorded with parameters", "warn");
            return;
        }
        dismissSessionContext();
        showToast("Re-running " + _sessionCtxOpText(job) + "…", "info");
        api("POST", job.endpoint, job.payload, function (err, data) {
            if (err) {
                showAlert("Re-run failed: " + (err.error || err.message || err));
                return;
            }
            if (data && data.job_id) {
                currentJob = data.job_id;
                if (SSE_OK) { trackJobSSE(data.job_id); } else { trackJobPoll(data.job_id); }
            }
        });
    }

    function _sessionCtxApplyToSelection(job) {
        if (!selectedPath) { showAlert("Select a clip first."); return; }
        if (!job.endpoint || !job.payload) {
            showToast("Job params aren't recorded — can't replay.", "warn");
            return;
        }
        var payload = JSON.parse(JSON.stringify(job.payload));
        payload.filepath = selectedPath;
        dismissSessionContext();
        showToast("Applying " + _sessionCtxOpText(job) + " to " + (selectedName || "selection") + "…", "info");
        startJob(job.endpoint, payload);
    }

    function dismissSessionContext() {
        if (!el.sessionContext) return;
        el.sessionContext.classList.add("hidden");
        _sessionCtxDismissedAt = Date.now();
    }

    // ================================================================
    // Operation Journal UI (v1.9.28)
    // ================================================================
    function _journalActionLabel(action) {
        return ({
            add_markers:       "Add markers",
            batch_rename:      "Batch rename",
            import_sequence:   "Import sequence",
            import_overlay:    "Import overlay",
            import_captions:   "Import captions",
            create_smart_bins: "Create bins"
        })[action] || action;
    }

    function renderJournalList() {
        if (!el.journalList) return;
        el.journalList.innerHTML = '<div class="journal-empty">Loading recent operations…</div>';
        api("GET", "/journal/list?limit=30", null, function (err, data) {
            if (err) {
                el.journalList.innerHTML =
                    '<div class="journal-empty journal-error">Couldn\'t load journal: ' +
                    esc(err.error || err.message || "Unknown error") + '</div>';
                return;
            }
            if (!Array.isArray(data) || !data.length) {
                el.journalList.innerHTML =
                    '<div class="journal-empty">No operations recorded yet. Run an action that writes to your Premiere project and it will appear here.</div>';
                return;
            }
            el.journalList.innerHTML = "";
            var frag = document.createDocumentFragment();
            for (var i = 0; i < data.length; i++) {
                frag.appendChild(_buildJournalRow(data[i]));
            }
            el.journalList.appendChild(frag);
        });
    }

    function _buildJournalRow(entry) {
        var row = document.createElement("div");
        row.className = "journal-row" + (entry.reverted ? " journal-row-reverted" : "");
        row.setAttribute("data-id", entry.id);

        var copy = document.createElement("div");
        copy.className = "journal-row-copy";

        var title = document.createElement("div");
        title.className = "journal-row-title";
        title.textContent = _journalActionLabel(entry.action);

        var meta = document.createElement("div");
        meta.className = "journal-row-meta";
        var parts = [_sessionCtxRelativeTime(entry.created_at)];
        if (entry.label) parts.push(entry.label);
        meta.textContent = parts.join(" · ");

        copy.appendChild(title);
        copy.appendChild(meta);
        row.appendChild(copy);

        var actions = document.createElement("div");
        actions.className = "journal-row-actions";

        if (entry.reverted) {
            var pill = document.createElement("span");
            pill.className = "journal-pill journal-pill-reverted";
            pill.textContent = "Reverted";
            actions.appendChild(pill);
        } else if (!entry.revertible) {
            var pill2 = document.createElement("span");
            pill2.className = "journal-pill journal-pill-info";
            pill2.textContent = "No auto-revert";
            pill2.title = "This action type has no ExtendScript inverse. It's recorded for context only.";
            actions.appendChild(pill2);
        } else {
            var revertBtn = document.createElement("button");
            revertBtn.type = "button";
            revertBtn.className = "btn btn-secondary btn-sm journal-revert-btn";
            revertBtn.textContent = "Revert";
            revertBtn.addEventListener("click", function () {
                _journalRevert(entry, revertBtn);
            });
            actions.appendChild(revertBtn);
        }

        row.appendChild(actions);
        return row;
    }

    function _journalRevert(entry, btn) {
        if (!inPremiere) { showAlert("Premiere Pro connection required to revert."); return; }
        if (!entry.revertible) { return; }

        btn.disabled = true;
        btn.textContent = "Reverting…";

        var dispatch = {
            add_markers:     function (p, cb) { PremiereBridge.removeSequenceMarkers(p, cb); },
            batch_rename:    function (p, cb) { PremiereBridge.unrenameItems(p, cb); },
            import_sequence: function (p, cb) { PremiereBridge.removeImportedSequence(p, cb); },
            import_overlay:  function (p, cb) { PremiereBridge.removeImportedItem(p, cb); }
        }[entry.action];

        if (!dispatch) {
            btn.disabled = false;
            btn.textContent = "Revert";
            showAlert("This action can't be reverted automatically.");
            return;
        }

        dispatch(entry.inverse || {}, function (result) {
            var r;
            try { r = JSON.parse(result || "{}"); } catch (e) { r = { error: result || "Parse error" }; }
            if (r.error) {
                btn.disabled = false;
                btn.textContent = "Revert";
                showAlert("Revert failed: " + r.error);
                return;
            }
            // Mark server-side so the UI reflects the new state.
            api("POST", "/journal/mark-reverted/" + entry.id, {}, function (err) {
                if (err) {
                    showToast("Reverted in Premiere but couldn't update the journal", "warn");
                } else {
                    showToast("Reverted: " + _journalActionLabel(entry.action), "success");
                }
                renderJournalList();
            });
        });
    }

    // ================================================================
    // Interview Polish pipeline (v1.9.29)
    // ================================================================
    var _polishActive = false;
    var _polishStepDefs = [
        { key: "silence",    label: "Detect speech segments" },
        { key: "transcribe", label: "Transcribe audio" },
        { key: "repeats",    label: "Find repeated takes" },
        { key: "fillers",    label: "Remove filler words" },
        { key: "diarize",    label: "Identify speakers" },
        { key: "chapters",   label: "Generate chapters" }
    ];

    function _polishStepRow(def, status, detail) {
        var li = document.createElement("li");
        li.className = "polish-step polish-step-" + status;
        li.setAttribute("data-key", def.key);

        var icon = document.createElement("span");
        icon.className = "polish-step-icon";
        icon.setAttribute("aria-hidden", "true");
        var iconChar = {
            pending: "\u2022",
            running: "\u2022",
            ok:      "\u2713",
            fail:    "\u00D7",
            skipped: "\u2013"
        }[status] || "\u2022";
        icon.textContent = iconChar;

        var label = document.createElement("span");
        label.className = "polish-step-label";
        label.textContent = def.label;

        var meta = document.createElement("span");
        meta.className = "polish-step-meta";
        meta.textContent = detail || "";

        li.appendChild(icon);
        li.appendChild(label);
        li.appendChild(meta);
        return li;
    }

    function renderPolishSteps(stepMap) {
        if (!el.polishSteps) return;
        el.polishSteps.classList.remove("hidden");
        el.polishSteps.innerHTML = "";
        var frag = document.createDocumentFragment();
        for (var i = 0; i < _polishStepDefs.length; i++) {
            var def = _polishStepDefs[i];
            var entry = stepMap[def.key];
            var status = "pending";
            var detail = "";
            if (entry) {
                status = entry.ok ? "ok" : (entry.reason ? "skipped" : "fail");
                if (entry.ok) {
                    if (entry.removed_fillers != null) detail = entry.removed_fillers + " fillers";
                    else if (entry.removed_ranges != null) detail = entry.removed_ranges + " repeats";
                    else if (entry.kept_segments != null) detail = entry.kept_segments + " segments";
                    else if (entry.word_count) detail = entry.word_count + " words";
                    else if (entry.count != null) detail = entry.count + " chapters";
                } else {
                    detail = entry.reason || "Failed";
                }
            }
            frag.appendChild(_polishStepRow(def, status, detail));
        }
        el.polishSteps.appendChild(frag);
    }

    function _polishStepsFromResult(result) {
        var map = {};
        var steps = (result && result.steps) || [];
        for (var i = 0; i < steps.length; i++) {
            map[steps[i].key] = steps[i];
        }
        return map;
    }

    function _renderPolishRunning() {
        if (!el.polishSteps) return;
        el.polishSteps.classList.remove("hidden");
        el.polishSteps.innerHTML = "";
        var frag = document.createDocumentFragment();
        for (var i = 0; i < _polishStepDefs.length; i++) {
            frag.appendChild(_polishStepRow(_polishStepDefs[i], "pending", ""));
        }
        el.polishSteps.appendChild(frag);
    }

    function _renderPolishResult(job) {
        if (!el.polishResult) return;
        el.polishResult.classList.remove("hidden");
        el.polishResult.innerHTML = "";

        var r = job.result || {};
        var header = document.createElement("div");
        header.className = "polish-result-title";
        var ratio = r.compression_ratio ? Math.round(r.compression_ratio * 100) : 0;
        header.textContent = "Compressed to " + ratio + "% of original" +
            (r.speech_duration ? " (" + fmtDur(r.speech_duration) + ")" : "");
        el.polishResult.appendChild(header);

        var actions = document.createElement("div");
        actions.className = "polish-result-actions";

        if (r.xml_path && inPremiere) {
            var importBtn = document.createElement("button");
            importBtn.type = "button";
            importBtn.className = "btn btn-primary btn-sm";
            importBtn.textContent = "Import to Premiere";
            importBtn.addEventListener("click", function () {
                PremiereBridge.importXML(r.xml_path, function (result) {
                    try {
                        var jr = JSON.parse(result);
                        if (jr.error) { showAlert("Import error: " + jr.error); return; }
                        showToast("Imported '" + (jr.sequence_name || r.sequence_name) + "'", "success");
                        if (jr.sequence_name) {
                            journalRecord("import_sequence",
                                "Interview Polish → '" + jr.sequence_name + "'",
                                { name: jr.sequence_name }, selectedPath);
                        }
                    } catch (e) { showAlert("Import failed: " + (result || e.message)); }
                });
            });
            actions.appendChild(importBtn);
        }

        if (r.srt_path) {
            var srtBtn = document.createElement("button");
            srtBtn.type = "button";
            srtBtn.className = "btn btn-secondary btn-sm";
            srtBtn.textContent = "Open SRT";
            srtBtn.addEventListener("click", function () {
                api("POST", "/system/open-path", { path: r.srt_path, mode: "open" }, function () {});
            });
            actions.appendChild(srtBtn);
        }
        if (r.chapters_path) {
            var chapBtn = document.createElement("button");
            chapBtn.type = "button";
            chapBtn.className = "btn btn-secondary btn-sm";
            chapBtn.textContent = "Open Chapters";
            chapBtn.addEventListener("click", function () {
                api("POST", "/system/open-path", { path: r.chapters_path, mode: "open" }, function () {});
            });
            actions.appendChild(chapBtn);
        }
        el.polishResult.appendChild(actions);
    }

    function runInterviewPolish() {
        if (_polishActive) return;
        if (!selectedPath) { showAlert("Select a clip first."); return; }

        // v1.9.33 (G): preflight — catches missing Whisper / no disk space
        // / bad path before the user waits minutes for failure.
        preflightPipeline("interview-polish", selectedPath, projectFolder, function (go) {
            if (go) _runInterviewPolishInner();
        });
    }

    function _runInterviewPolishInner() {
        _polishActive = true;
        el.polishInterviewBtn.disabled = true;
        el.polishInterviewBtn.textContent = "Polishing…";
        if (el.polishResult) el.polishResult.classList.add("hidden");
        _renderPolishRunning();

        var payload = {
            filepath: selectedPath,
            output_dir: projectFolder,
            detect_repeats: el.polishDetectRepeats && el.polishDetectRepeats.checked,
            remove_fillers: el.polishRemoveFillers && el.polishRemoveFillers.checked,
            generate_chapters: el.polishGenerateChapters && el.polishGenerateChapters.checked,
            diarize: false  // diarization is advisory; the step reports how to enable it
        };

        startJob("/interview-polish", payload, {
            onComplete: function (job) {
                renderPolishSteps(_polishStepsFromResult(job.result));
                _renderPolishResult(job);
            },
            onError: function (job) {
                renderPolishSteps(_polishStepsFromResult(job.result || {}));
            },
            onFinally: function () {
                _polishActive = false;
                el.polishInterviewBtn.disabled = !selectedPath;
                el.polishInterviewBtn.textContent = "Polish this interview";
            }
        });
    }

    // ================================================================
    // Preflight modal (v1.9.33, feature G) — reusable for any pipeline
    // ================================================================
    function preflightPipeline(pipeline, filepath, outputDir, cb) {
        api("POST", "/preflight/" + pipeline,
            { filepath: filepath || "", output_dir: outputDir || "" },
            function (err, report) {
                if (err || !report) {
                    // Preflight itself failed — be forgiving; don't block
                    // the user from running their pipeline.
                    cb(true);
                    return;
                }
                _showPreflightModal(report, cb);
            });
    }

    function _showPreflightModal(report, cb) {
        // Fast path: no blocking + no warnings -> skip the modal, just go.
        if (report.pass && (!report.warnings || !report.warnings.length)) {
            cb(true);
            return;
        }
        // Build modal
        var overlay = document.createElement("div");
        overlay.className = "preflight-overlay";
        overlay.setAttribute("role", "dialog");
        overlay.setAttribute("aria-modal", "true");
        overlay.setAttribute("aria-label", "Preflight check for " +
            (report.pipeline_label || report.pipeline));

        var box = document.createElement("div");
        box.className = "preflight-modal";

        var header = document.createElement("div");
        header.className = "preflight-header";
        var h = document.createElement("div");
        h.className = "preflight-title";
        h.textContent = "Preflight: " + (report.pipeline_label || report.pipeline);
        header.appendChild(h);
        var sub = document.createElement("div");
        sub.className = "preflight-sub";
        sub.textContent = report.pass
            ? "Ready to run. Some optional checks won't be available."
            : "Fix the items below before running.";
        header.appendChild(sub);
        box.appendChild(header);

        var body = document.createElement("div");
        body.className = "preflight-body";

        function addSection(title, items, tone) {
            if (!items || !items.length) return;
            var sect = document.createElement("div");
            sect.className = "preflight-section preflight-section-" + tone;
            var st = document.createElement("div");
            st.className = "preflight-section-title";
            st.textContent = title;
            sect.appendChild(st);
            for (var i = 0; i < items.length; i++) {
                var it = items[i];
                var row = document.createElement("div");
                row.className = "preflight-row";
                var label = document.createElement("div");
                label.className = "preflight-row-label";
                label.textContent = it.label || "";
                var detail = document.createElement("div");
                detail.className = "preflight-row-detail";
                detail.textContent = it.detail || it.fix || "";
                row.appendChild(label);
                row.appendChild(detail);
                sect.appendChild(row);
            }
            body.appendChild(sect);
        }

        if (report.file && !report.file.ok) {
            addSection("Input file", [{
                label: "File",
                detail: (report.file.detail || "not found")
            }], "blocking");
        }
        addSection("Fix before running", report.blocking, "blocking");
        addSection("Heads-up", report.warnings, "warning");

        box.appendChild(body);

        var foot = document.createElement("div");
        foot.className = "preflight-foot";
        var cancel = document.createElement("button");
        cancel.type = "button";
        cancel.className = "btn btn-ghost";
        cancel.textContent = "Cancel";
        cancel.addEventListener("click", function () {
            document.body.removeChild(overlay);
            cb(false);
        });
        var run = document.createElement("button");
        run.type = "button";
        run.className = "btn " + (report.pass ? "btn-primary" : "btn-ghost");
        run.textContent = report.pass ? "Run anyway" : "Run anyway (may fail)";
        run.addEventListener("click", function () {
            document.body.removeChild(overlay);
            cb(true);
        });
        if (!report.pass) {
            // Still allow override for power users, but emphasise cancel.
            run.className = "btn btn-secondary";
        }
        foot.appendChild(cancel);
        foot.appendChild(run);
        box.appendChild(foot);

        overlay.appendChild(box);
        overlay.addEventListener("click", function (e) {
            if (e.target === overlay) { document.body.removeChild(overlay); cb(false); }
        });
        document.body.appendChild(overlay);
        run.focus();
    }

    function initInterviewPolish() {
        if (!el.polishInterviewBtn) return;
        el.polishInterviewBtn.addEventListener("click", runInterviewPolish);
        if (el.polishBatchBtn) {
            el.polishBatchBtn.addEventListener("click", runInterviewPolishBatch);
        }
        // Enable/disable whenever a clip selection changes — piggy-back on
        // the existing updateButtons pass below.
        updateButtons();
    }

    // ================================================================
    // Interview Polish — batch mode (v1.9.32, feature F)
    // ================================================================
    // Serial (not parallel) to avoid OOM on 8 concurrent Whisper runs.
    // One parent job_id per file; they show up normally in job history.
    var _polishBatchQueue = null;
    var _polishBatchResults = null;

    function updatePolishBatchButton() {
        if (!el.polishBatchBtn || !el.polishBatchCount) return;
        var n = (typeof _batchFiles !== "undefined" && _batchFiles) ? _batchFiles.length : 0;
        if (n < 2) {
            el.polishBatchBtn.classList.add("hidden");
        } else {
            el.polishBatchBtn.classList.remove("hidden");
            el.polishBatchCount.textContent = n;
            el.polishBatchBtn.disabled = _polishActive || !connected;
        }
    }

    function runInterviewPolishBatch() {
        if (_polishActive) return;
        if (!_batchFiles || _batchFiles.length < 2) {
            showAlert("Add at least 2 files to the batch picker first.");
            return;
        }
        _polishBatchQueue = _batchFiles.slice();
        _polishBatchResults = [];
        _polishStartNextBatch();
    }

    function _polishStartNextBatch() {
        if (!_polishBatchQueue || !_polishBatchQueue.length) {
            _polishBatchFinish();
            return;
        }
        var total = _batchFiles.length;
        var idx = total - _polishBatchQueue.length;
        var filepath = _polishBatchQueue.shift();

        _polishActive = true;
        el.polishInterviewBtn.disabled = true;
        el.polishInterviewBtn.textContent = "Batch " + (idx + 1) + "/" + total + "…";
        if (el.polishBatchBtn) el.polishBatchBtn.disabled = true;
        if (el.polishResult) el.polishResult.classList.add("hidden");
        _renderPolishRunning();
        showToast("Polishing " + (idx + 1) + " of " + total + ": " +
                  filepath.split(/[\\/]/).pop(), "info");

        var payload = {
            filepath: filepath,
            output_dir: projectFolder,
            detect_repeats: el.polishDetectRepeats && el.polishDetectRepeats.checked,
            remove_fillers: el.polishRemoveFillers && el.polishRemoveFillers.checked,
            generate_chapters: el.polishGenerateChapters && el.polishGenerateChapters.checked,
            diarize: false
        };
        startJob("/interview-polish", payload, {
            onComplete: function (job) {
                _polishBatchResults.push({ filepath: filepath, ok: true, result: job.result });
                renderPolishSteps(_polishStepsFromResult(job.result));
            },
            onError: function (job) {
                _polishBatchResults.push({ filepath: filepath, ok: false,
                    error: (job && job.error) || "Unknown error" });
            },
            onFinally: function () {
                _polishActive = false;
                // Move on to the next file — or finish.
                setTimeout(_polishStartNextBatch, 200);
            }
        });
    }

    function _polishBatchFinish() {
        _polishActive = false;
        var ok = 0, failed = 0;
        for (var i = 0; i < _polishBatchResults.length; i++) {
            if (_polishBatchResults[i].ok) ok++; else failed++;
        }
        el.polishInterviewBtn.disabled = !selectedPath;
        el.polishInterviewBtn.textContent = "Polish this interview";
        if (el.polishBatchBtn) el.polishBatchBtn.disabled = false;
        showAlert("Batch polish done: " + ok + " succeeded" +
                  (failed ? ", " + failed + " failed" : "") + ".");
        _polishBatchQueue = null;
        _polishBatchResults = null;
    }

    function initJournal() {
        if (!el.journalList) return;
        if (el.journalRefreshBtn) {
            el.journalRefreshBtn.addEventListener("click", renderJournalList);
        }
        if (el.journalClearBtn) {
            el.journalClearBtn.addEventListener("click", function () {
                if (!confirm("Clear all journal entries? This does not undo anything in Premiere.")) return;
                api("POST", "/journal/clear", {}, function (err) {
                    if (err) { showAlert("Could not clear: " + (err.error || err)); return; }
                    showToast("Journal cleared", "success");
                    renderJournalList();
                });
            });
        }
        // Lazy render the first time the user actually opens Settings.
        var settingsTab = document.querySelector('[data-nav="settings"]');
        if (settingsTab) {
            settingsTab.addEventListener("click", function () {
                setTimeout(renderJournalList, 50);
            });
        }
    }

    function esc(s) {
        if (!s) return "";
        return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
    }

    function safeFixed(v, digits) { var n = Number(v); return isFinite(n) ? n.toFixed(digits) : "0"; }
    
    // Escape for use inside JSX string arguments (handles Windows paths)
    function escPath(s) {
        if (!s) return "";
        // Escape for safe embedding inside a JS/ExtendScript double-quoted string:
        // backslashes, quotes, and control characters that could break the string
        return s.replace(/\\/g, "\\\\").replace(/"/g, '\\"')
                .replace(/\n/g, "\\n").replace(/\r/g, "\\r").replace(/\t/g, "\\t");
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
    // Cut Review Panel (Phase 3.3 — Preview Before Commit)
    // ================================================================

    var _cutReviewApplyCallback = null;

    /**
     * Format seconds as MM:SS.s (e.g. "01:23.4")
     */
    function formatTimecode(seconds) {
        if (!seconds && seconds !== 0) return "00:00.0";
        var totalTenths = Math.round(Math.abs(Number(seconds) || 0) * 10);
        var totalSec = Math.floor(totalTenths / 10);
        var tenths = totalTenths % 10;
        var m = Math.floor(totalSec / 60);
        var sec = totalSec % 60;
        return (m < 10 ? "0" : "") + m + ":" + (sec < 10 ? "0" : "") + sec + "." + tenths;
    }

    /**
     * Update the summary text showing how many cuts are selected.
     */
    function updateCutReviewSummary() {
        var panel = document.getElementById("cutReviewList");
        var summary = document.getElementById("cutReviewSummary");
        if (!panel || !summary) return;
        var boxes = panel.querySelectorAll("input[type='checkbox']");
        var checked = 0;
        var totalDuration = 0;
        for (var i = 0; i < boxes.length; i++) {
            if (boxes[i].checked) {
                checked++;
                totalDuration += parseFloat(boxes[i].getAttribute("data-duration") || 0);
            }
        }
        summary.textContent = checked + " of " + boxes.length + " cuts selected · " + safeFixed(totalDuration, 1) + "s marked for write-back";
    }

    /**
     * Show the cut review panel with a list of cuts.
     * @param {Array} cuts - Array of {start, end, label, ...}
     * @param {Function} onApply - Called with the filtered array of selected cuts
     */
    function showCutReview(cuts, onApply) {
        var panel = document.getElementById("cutReviewPanel");
        var list = document.getElementById("cutReviewList");
        if (!panel || !list) return;
        if (!cuts || !cuts.length) {
            showAlert("No cuts detected in this clip.");
            return;
        }

        _cutReviewApplyCallback = onApply;

        var totalDuration = 0;
        for (var t = 0; t < cuts.length; t++) {
            totalDuration = Math.max(totalDuration, Number(cuts[t].end || cuts[t].start || 0));
        }
        var html = "";
        for (var i = 0; i < cuts.length; i++) {
            var c = cuts[i];
            var startSec = Number(c.start || 0);
            var endSec = Number(c.end || 0);
            var dur = Math.max(0, endSec - startSec);
            var label = c.label || c.text || c.reason || ("Cut " + (i + 1));
            var left = totalDuration > 0 ? Math.max(0, Math.min(100, (startSec / totalDuration) * 100)) : 0;
            var width = totalDuration > 0 ? Math.max(3, (dur / totalDuration) * 100) : 100;
            if (left + width > 100) width = Math.max(3, 100 - left);
            var checkboxLabel = "Include cut " + (i + 1) + ", " + formatTimecode(startSec) + " to " + formatTimecode(endSec);
            html += '<label class="cut-review-row" data-idx="' + i + '">'
                + '<input type="checkbox" checked data-index="' + i + '" data-duration="' + dur.toFixed(2) + '" aria-label="' + esc(checkboxLabel) + '">'
                + '<div class="cut-review-main">'
                + '<div class="cut-review-top">'
                + '<div class="cut-review-meta">'
                + '<span class="cut-review-index">' + (i + 1) + '</span>'
                + '<span class="cut-review-time">' + formatTimecode(startSec) + '</span>'
                + '<span class="cut-review-time">' + formatTimecode(endSec) + '</span>'
                + '<span class="cut-review-duration">' + safeFixed(dur, 1) + 's</span>'
                + '</div>'
                + '<span class="cut-review-label">' + esc(String(label).substring(0, 120)) + '</span>'
                + '</div>'
                + '<div class="cut-review-track"><span class="cut-review-track-fill" style="left:' + left.toFixed(3) + '%;width:' + width.toFixed(3) + '%;"></span></div>'
                + '</div>'
                + '</label>';
        }
        list.innerHTML = html;

        // Attach change listeners to checkboxes
        var boxes = list.querySelectorAll("input[type='checkbox']");
        for (var j = 0; j < boxes.length; j++) {
            boxes[j].addEventListener("change", updateCutReviewSummary);
        }

        panel.classList.remove("hidden");
        updateCutReviewSummary();

        // Store the raw cuts array for retrieval on apply
        panel._cutsData = cuts;
    }

    /**
     * Hide the cut review panel.
     */
    function hideCutReview() {
        var panel = document.getElementById("cutReviewPanel");
        if (panel) panel.classList.add("hidden");
        _cutReviewApplyCallback = null;
    }

    /**
     * Initialize cut review panel event listeners.
     */
    function initCutReviewPanel() {
        var closeBtn = document.getElementById("cutReviewClose");
        if (closeBtn) closeBtn.addEventListener("click", hideCutReview);

        var selectAllBtn = document.getElementById("cutReviewSelectAll");
        if (selectAllBtn) selectAllBtn.addEventListener("click", function () {
            var boxes = document.getElementById("cutReviewList").querySelectorAll("input[type='checkbox']");
            for (var i = 0; i < boxes.length; i++) boxes[i].checked = true;
            updateCutReviewSummary();
        });

        var deselectAllBtn = document.getElementById("cutReviewDeselectAll");
        if (deselectAllBtn) deselectAllBtn.addEventListener("click", function () {
            var boxes = document.getElementById("cutReviewList").querySelectorAll("input[type='checkbox']");
            for (var i = 0; i < boxes.length; i++) boxes[i].checked = false;
            updateCutReviewSummary();
        });

        var applyBtn = document.getElementById("cutReviewApply");
        if (applyBtn) applyBtn.addEventListener("click", function () {
            var panel = document.getElementById("cutReviewPanel");
            var list = document.getElementById("cutReviewList");
            if (!panel || !list) return;

            var cuts = panel._cutsData || [];
            var boxes = list.querySelectorAll("input[type='checkbox']");
            var selected = [];
            for (var i = 0; i < boxes.length; i++) {
                if (boxes[i].checked) {
                    var idx = parseInt(boxes[i].getAttribute("data-index"));
                    if (cuts[idx]) selected.push(cuts[idx]);
                }
            }

            if (!selected.length) {
                showAlert("No cuts selected. Select at least one cut to apply.");
                return;
            }

            if (typeof _cutReviewApplyCallback === "function") {
                _cutReviewApplyCallback(selected);
            }
            hideCutReview();
        });
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
        // Note: toast is intentionally NOT shown here — onJobDone/showResults
        // already displays results and showAlert. Adding a toast here would
        // duplicate notifications on every job completion.
    }

    function setToggleButtonCount(button, label, count) {
        if (!button) return;
        var labelEl = button.querySelector(".toggle-label");
        var countEl = button.querySelector(".toggle-count");
        if (labelEl && countEl) {
            labelEl.textContent = label;
            countEl.textContent = "(" + count + ")";
        } else {
            button.textContent = label + " (" + count + ")";
        }
        button.setAttribute("aria-label", label + " (" + count + ")");
    }

    function renderJobHistory() {
        if (!el.jobHistory || !el.jobHistoryToggle) return;
        setToggleButtonCount(el.jobHistoryToggle, "History", jobHistoryList.length);
        var html = "";
        for (var i = 0; i < jobHistoryList.length; i++) {
            var h = jobHistoryList[i];
            var statusClass = h.status === "complete" ? "complete" : (h.status === "cancelled" ? "cancelled" : "error");
            // Show "Apply to selection" only when there's a different clip selected
            // from the one this job originally targeted (v1.9.30: feature J).
            var payloadPath = h.payload && h.payload.filepath ? h.payload.filepath : "";
            var showApply = h.endpoint && payloadPath && selectedPath && selectedPath !== payloadPath;
            html += '<div class="job-history-item" data-idx="' + i + '">' +
                '<div class="job-history-main">' +
                '<span class="job-history-status ' + statusClass + '"></span>' +
                '<span class="job-history-type">' + esc(h.type) + '</span>' +
                '</div>' +
                '<div class="job-history-meta">' +
                '<span class="job-history-time">' + esc(h.time) + '</span>' +
                (h.endpoint ? '<button type="button" class="btn-sm job-history-rerun" data-idx="' + i + '" title="Re-run on the original clip with the same params">Re-run</button>' : '') +
                (showApply ? '<button type="button" class="btn-sm job-history-apply" data-idx="' + i + '" title="Run this job on the currently selected clip with the same params">Apply to selection</button>' : '') +
                '</div></div>';
        }
        el.jobHistory.innerHTML = html;
    }

    // Event delegation for job history re-run buttons (avoids listener accumulation)
    var _jobHistoryDelegationAdded = false;
    function ensureJobHistoryDelegation() {
        if (_jobHistoryDelegationAdded || !el.jobHistory) return;
        _jobHistoryDelegationAdded = true;
        el.jobHistory.addEventListener("click", function (e) {
            var rerunBtn = e.target.closest(".job-history-rerun");
            var applyBtn = e.target.closest(".job-history-apply");
            var btn = rerunBtn || applyBtn;
            if (!btn) return;
            e.stopPropagation();
            var idx = parseInt(btn.getAttribute("data-idx"));
            var entry = jobHistoryList[idx];
            if (!entry || !entry.endpoint || !entry.payload) return;
            if (applyBtn) {
                if (!selectedPath) { showAlert("Select a clip first."); return; }
                // Clone so we don't mutate the stored payload
                var payload = JSON.parse(JSON.stringify(entry.payload));
                payload.filepath = selectedPath;
                showToast("Applying '" + entry.type + "' to " + (selectedName || "selection") + "…", "info");
                startJob(entry.endpoint, payload);
            } else {
                startJob(entry.endpoint, entry.payload);
            }
        });
    }

    function initJobHistory() {
        if (!el.jobHistoryToggle || !el.jobHistory) return;
        ensureJobHistoryDelegation();
        el.jobHistoryToggle.addEventListener("click", function () {
            var isOpen = el.jobHistory.classList.toggle("open");
            setExpanded(el.jobHistoryToggle, isOpen);
        });

        // Add listener to record finished jobs
        addJobDoneListener(function (job) {
            addJobHistory(job);
        });

        // Load persistent job history from backend (fills history across restarts)
        api("GET", "/jobs/history?limit=20", null, function (err, data) {
            if (err || !data || !data.length) return;
            for (var i = data.length - 1; i >= 0; i--) {
                var j = data[i];
                // Avoid duplicating entries already in the client-side list
                var alreadyHas = false;
                for (var k = 0; k < jobHistoryList.length; k++) {
                    if (jobHistoryList[k].type === j.type && jobHistoryList[k].status === j.status) {
                        alreadyHas = true; break;
                    }
                }
                if (!alreadyHas) {
                    var created = j.created ? new Date(j.created * 1000).toLocaleTimeString() : "";
                    jobHistoryList.push({
                        type: j.type || "unknown",
                        status: j.status || "complete",
                        message: j.message || "",
                        time: created,
                        endpoint: j.endpoint || "",
                        payload: j.payload || null
                    });
                }
            }
            if (jobHistoryList.length > MAX_JOB_HISTORY) jobHistoryList.length = MAX_JOB_HISTORY;
            renderJobHistory();
        });
    }

    // ================================================================
    // Escape to Cancel + Keyboard Shortcuts
    // ================================================================
    function initKeyboardShortcuts() {
        loadShortcuts();

        document.addEventListener("keydown", function (e) {
            // Cancel-job shortcut works even in inputs (Escape) — but not if already consumed by a dropdown
            if (!e.defaultPrevented && matchesShortcut(e, _shortcutRegistry["cancel-job"].keys) && currentJob) {
                _shortcutActions["cancel-job"]();
                return;
            }

            // Don't handle other shortcuts when typing in inputs
            var tag = e.target.tagName;
            if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || e.target.isContentEditable) return;

            // Check registry shortcuts (skip cancel-job, already handled above)
            var id;
            for (id in _shortcutRegistry) {
                if (id === "cancel-job") continue;
                if (_shortcutRegistry.hasOwnProperty(id) && matchesShortcut(e, _shortcutRegistry[id].keys)) {
                    e.preventDefault();
                    if (_shortcutActions[id]) _shortcutActions[id]();
                    return;
                }
            }

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

            // Tab switching: 1-8 for main tabs (skip if focus is on interactive elements)
            if (e.key >= "1" && e.key <= "8" && !e.ctrlKey && !e.altKey && !e.metaKey && !e.shiftKey
                && e.target === document.body) {
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
                    html = '<option value="" disabled selected>Select preset…</option>';
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
    // Preset Export/Import as .opencut-preset file
    // ================================================================

    function exportPresetFile() {
        if (!el.presetSelect || !el.presetSelect.value) {
            showToast("Select a preset to export first", "error");
            return;
        }
        var presetName = el.presetSelect.value;
        api("GET", "/presets", null, function (err, data) {
            if (err || !data || !data[presetName]) {
                showToast("Could not load preset for export", "error");
                return;
            }
            var exportData = {
                opencut_preset: true,
                version: "1.7.2",
                name: presetName,
                settings: data[presetName].settings,
                exported: new Date().toISOString()
            };
            var blob = new Blob([JSON.stringify(exportData, null, 2)], { type: "application/json" });
            var url = URL.createObjectURL(blob);
            var a = document.createElement("a");
            a.href = url;
            a.download = presetName.replace(/[^a-zA-Z0-9_-]/g, "_") + ".opencut-preset";
            a.click();
            setTimeout(function () { URL.revokeObjectURL(url); }, 5000);
            showToast("Preset exported: " + presetName, "success");
        });
    }

    function importPresetFile() {
        var input = document.getElementById("importPresetFileInput");
        if (!input) {
            input = document.createElement("input");
            input.type = "file";
            input.id = "importPresetFileInput";
            input.accept = ".opencut-preset,.json";
            input.style.display = "none";
            document.body.appendChild(input);
            input.addEventListener("change", function () {
                var file = this.files[0];
                if (!file) return;
                var reader = new FileReader();
                reader.onload = function (e) {
                    try {
                        var data = JSON.parse(e.target.result);
                        if (!data.opencut_preset || !data.name || !data.settings) {
                            showToast("Invalid preset file: missing required fields", "error");
                            return;
                        }
                        if (typeof data.settings !== "object") {
                            showToast("Invalid preset file: settings must be an object", "error");
                            return;
                        }
                        api("POST", "/presets/save", { name: data.name, settings: data.settings }, function (err, result) {
                            if (!err && result && result.success) {
                                showToast("Preset imported: " + data.name, "success");
                                refreshPresetList();
                            } else {
                                showToast("Failed to import preset", "error");
                            }
                        });
                    } catch (ex) {
                        showToast("Invalid preset file format", "error");
                    }
                };
                reader.readAsText(file);
                this.value = "";
            });
        }
        input.click();
    }

    // ================================================================
    // Project Templates
    // ================================================================
    var _projectTemplates = [];

    function initProjectTemplates() {
        var select = document.getElementById("templateSelect");
        var applyBtn = document.getElementById("applyTemplateBtn");
        var saveBtn = document.getElementById("saveCustomTemplateBtn");
        var nameInput = document.getElementById("templateCustomName");
        if (!select) return;

        loadTemplateList();

        if (applyBtn) {
            applyBtn.addEventListener("click", function () {
                var id = select.value;
                if (!id) { showToast("Select a template first", "error"); return; }
                api("POST", "/templates/apply", { id: id }, function (err, data) {
                    if (err || !data || !data.success) {
                        showToast("Failed to apply template", "error");
                        return;
                    }
                    var tpl = data.template;
                    // Apply audio settings
                    if (tpl.audio) {
                        var lufs = document.getElementById("defaultLufs");
                        if (lufs && tpl.audio.loudness_target !== undefined) {
                            lufs.value = tpl.audio.loudness_target;
                            try { lufs.dispatchEvent(new Event("input", { bubbles: true })); } catch (ev) {}
                        }
                        if (tpl.audio.normalize !== undefined && el.settingsAutoImport) {
                            // Normalize flag stored as preference
                        }
                    }
                    // Apply export settings
                    if (tpl.export) {
                        var _set = function (id, val) {
                            var elem = document.getElementById(id);
                            if (elem && val !== undefined) {
                                elem.value = val;
                                try { elem.dispatchEvent(new Event("change", { bubbles: true })); } catch (ev) {}
                            }
                        };
                        _set("exportFormat", tpl.export.format);
                        _set("exportCodec", tpl.export.codec);
                        _set("exportResolution", tpl.export.resolution);
                        _set("exportBitrate", tpl.export.bitrate);
                        _set("exportFps", tpl.export.fps);
                        _set("exportAudioBitrate", tpl.export.audio_bitrate);
                    }
                    // Apply caption settings
                    if (tpl.captions) {
                        if (tpl.captions.style) {
                            var styleEl = document.getElementById("captionStyle");
                            if (styleEl) {
                                styleEl.value = tpl.captions.style;
                                try { styleEl.dispatchEvent(new Event("change", { bubbles: true })); } catch (ev) {}
                            }
                        }
                        if (tpl.captions.font_size) {
                            var fsEl = document.getElementById("captionFontSize");
                            if (fsEl) {
                                fsEl.value = tpl.captions.font_size;
                                try { fsEl.dispatchEvent(new Event("input", { bubbles: true })); } catch (ev) {}
                            }
                        }
                    }
                    showToast("Template applied: " + tpl.name, "success");
                });
            });
        }

        if (saveBtn) {
            saveBtn.addEventListener("click", function () {
                var name = nameInput ? nameInput.value.trim() : "";
                if (!name) { showToast("Enter a template name", "error"); return; }
                var templateData = {
                    name: name,
                    description: "Custom template",
                    export: {},
                    audio: {},
                    captions: {},
                    aspect: "16:9"
                };
                // Capture audio settings
                var lufs = document.getElementById("defaultLufs");
                if (lufs) templateData.audio.loudness_target = parseFloat(lufs.value) || -14;
                templateData.audio.normalize = true;
                // Capture export settings from current panel values
                var _get = function (id) { var e = document.getElementById(id); return e ? e.value : ""; };
                var fmt = _get("exportFormat"); if (fmt) templateData.export.format = fmt;
                var codec = _get("exportCodec"); if (codec) templateData.export.codec = codec;
                var res = _get("exportResolution"); if (res) templateData.export.resolution = res;
                var br = _get("exportBitrate"); if (br) templateData.export.bitrate = br;
                var fps = _get("exportFps"); if (fps) templateData.export.fps = fps;
                // Capture caption style
                var cstyle = _get("captionStyle"); if (cstyle) templateData.captions.style = cstyle;
                var cfont = _get("captionFontSize"); if (cfont) templateData.captions.font_size = parseInt(cfont, 10) || 24;
                api("POST", "/templates/save", templateData, function (err, data) {
                    if (!err && data && data.success) {
                        showToast("Template saved: " + name, "success");
                        if (nameInput) nameInput.value = "";
                        loadTemplateList();
                    } else {
                        showToast("Failed to save template", "error");
                    }
                });
            });
        }

        select.addEventListener("change", updateTemplateDescription);
    }

    function loadTemplateList() {
        var select = document.getElementById("templateSelect");
        if (!select) return;
        api("GET", "/templates/list", null, function (err, data) {
            if (err || !data) return;
            _projectTemplates = (data.builtin || []).concat(data.user || []);
                var html = '<option value="" disabled selected>Select a template…</option>';
            if (data.builtin && data.builtin.length) {
                html += '<optgroup label="Built-in">';
                for (var i = 0; i < data.builtin.length; i++) {
                    html += '<option value="' + esc(data.builtin[i].id) + '">' + esc(data.builtin[i].name) + '</option>';
                }
                html += '</optgroup>';
            }
            if (data.user && data.user.length) {
                html += '<optgroup label="Custom">';
                for (var j = 0; j < data.user.length; j++) {
                    html += '<option value="' + esc(data.user[j].id) + '">' + esc(data.user[j].name) + '</option>';
                }
                html += '</optgroup>';
            }
            select.innerHTML = html;
            if (select._customDropdown) select._customDropdown.update();
        });
    }

    function updateTemplateDescription() {
        var select = document.getElementById("templateSelect");
        var descEl = document.getElementById("templateDesc");
        if (!select || !descEl) return;
        var id = select.value;
        var tpl = null;
        for (var i = 0; i < _projectTemplates.length; i++) {
            if (_projectTemplates[i].id === id) { tpl = _projectTemplates[i]; break; }
        }
        descEl.textContent = tpl ? tpl.description : "";
    }

    // ================================================================
    // Model Management
    // ================================================================
    var _modelListDelegationAdded = false;

    function ensureModelListDelegation() {
        if (_modelListDelegationAdded || !el.modelList) return;
        _modelListDelegationAdded = true;
        el.modelList.addEventListener("click", function (e) {
            var btn = e.target.closest(".model-item-delete");
            if (!btn) return;
            var path = btn.dataset.path || "";
            if (!path) {
                showAlert("Couldn't determine which model to delete.");
                return;
            }
            btn.disabled = true;
                btn.textContent = "Deleting…";
            api("POST", "/models/delete", { path: path }, function (err, data) {
                if (!err && data && data.success) {
                    showToast("Model deleted", "success");
                    refreshModelList();
                } else {
                    btn.disabled = false;
                    btn.textContent = "Delete";
                    showAlert("Failed to delete model.");
                }
            });
        });
    }

    function initModelManagement() {
        if (!el.refreshModelsBtn) return;
        ensureModelListDelegation();
        el.refreshModelsBtn.addEventListener("click", refreshModelList);
    }

    function refreshModelList() {
        if (!el.modelList) return;
        ensureModelListDelegation();
        setHintContent(el.modelList, "Scanning…");
        api("GET", "/models/list", null, function (err, data) {
            if (err || !data) {
                setHintContent(el.modelList, "Failed to load models.");
                return;
            }
            if (!data.models || data.models.length === 0) {
                setHintContent(el.modelList, "No models found.");
                if (el.modelsTotalSize) el.modelsTotalSize.textContent = "0 MB";
                return;
            }
            var frag = document.createDocumentFragment();
            for (var i = 0; i < data.models.length; i++) {
                var m = data.models[i];
                var sizeStr = m.size_mb >= 1024 ? safeFixed(m.size_mb / 1024, 1) + " GB" : safeFixed(m.size_mb, 0) + " MB";
                var item = document.createElement("div");
                item.className = "model-item";
                var info = document.createElement("div");
                info.className = "model-item-info";
                var name = document.createElement("span");
                name.className = "model-item-name";
                name.textContent = m.name || "Unknown model";
                var meta = document.createElement("span");
                meta.className = "model-item-meta";
                meta.textContent = sizeStr + " - " + (m.source || "Unknown source");
                var deleteBtn = document.createElement("button");
                deleteBtn.type = "button";
                deleteBtn.className = "model-item-delete";
                deleteBtn.textContent = "Delete";
                deleteBtn.title = "Delete model";
                deleteBtn.setAttribute("aria-label", "Delete model " + (m.name || "Unknown model"));
                deleteBtn.dataset.path = m.path || "";
                deleteBtn.disabled = !m.path;
                info.appendChild(name);
                info.appendChild(meta);
                item.appendChild(info);
                item.appendChild(deleteBtn);
                frag.appendChild(item);
            }
            el.modelList.innerHTML = "";
            el.modelList.appendChild(frag);
            if (el.modelsTotalSize) {
                var totalStr = data.total_mb >= 1024 ? safeFixed(data.total_mb / 1024, 1) + " GB" : safeFixed(data.total_mb, 0) + " MB";
                el.modelsTotalSize.textContent = totalStr;
            }
        }, 30000);
    }

    // ================================================================
    // GPU Recommendation
    // ================================================================
    function initGpuRecommendation() {
        if (!el.getGpuRecBtn) return;
        el.getGpuRecBtn.addEventListener("click", function () {
            var originalBtnText = rememberButtonText(el.getGpuRecBtn);
            setButtonText(el.getGpuRecBtn, "Checking…");
            el.getGpuRecBtn.disabled = true;
            api("GET", "/system/gpu-recommend", null, function (err, data) {
                setButtonText(el.getGpuRecBtn, originalBtnText);
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
    function _reflowToasts() {
        var live = document.querySelectorAll(".toast-notification");
        var offset = 24;
        for (var ri = 0; ri < live.length; ri++) {
            live[ri].style.bottom = offset + "px";
            offset += live[ri].offsetHeight + 12;
        }
    }

    function showToast(message, type) {
        // Only show if notifications enabled
        if (el.settingsShowNotifications && !el.settingsShowNotifications.checked) return;
        // Cap concurrent toasts — remove oldest if at limit
        var existing = document.querySelectorAll(".toast-notification");
        if (existing.length >= MAX_TOASTS) {
            for (var ti = 0; ti <= existing.length - MAX_TOASTS; ti++) {
                if (existing[ti].parentNode) existing[ti].parentNode.removeChild(existing[ti]);
            }
            _reflowToasts();
        }
        var toast = document.createElement("div");
        var tone = inferNotificationTone(message, null, type);
        toast.className = "toast-notification " + tone + " is-" + tone;
        toast.innerHTML = '<span class="toast-icon" aria-hidden="true">' + getNotificationIconSvg(tone) + '</span>' +
            '<span class="toast-copy">' +
            '<span class="toast-title">' + esc(getNotificationHeading(tone, message)) + '</span>' +
            '<span class="toast-message">' + esc(message) + '</span>' +
            '</span>';
        toast.setAttribute("role", tone === "error" ? "alert" : "status");
        toast.setAttribute("aria-live", tone === "error" ? "assertive" : "polite");
        toast.setAttribute("aria-atomic", "true");
        document.body.appendChild(toast);
        _reflowToasts();
        setTimeout(function () {
            toast.classList.add("fade-out");
            setTimeout(function () {
                if (toast.parentNode) toast.parentNode.removeChild(toast);
                _reflowToasts();
            }, 300);
        }, tone === "error" ? 4200 : 3200);
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
        el.transcriptSearchInput.addEventListener("keydown", function (e) {
            if (e.key === "Enter" && _searchMatches.length) {
                e.preventDefault();
                _searchIndex = e.shiftKey
                    ? (_searchIndex - 1 + _searchMatches.length) % _searchMatches.length
                    : (_searchIndex + 1) % _searchMatches.length;
                highlightSearchMatch();
            }
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
        var segments = el.transcriptSegments ? el.transcriptSegments.querySelectorAll(".transcript-seg") : [];
        var clips = el.transcriptTimeline ? el.transcriptTimeline.querySelectorAll(".transcript-timeline-seg") : [];
        for (var i = 0; i < segments.length; i++) {
            segments[i].classList.remove("search-highlight", "search-active");
        }
        for (var j = 0; j < clips.length; j++) {
            clips[j].classList.remove("search-highlight", "search-active");
        }

        if (!query) {
            if (el.transcriptSearchCount) el.transcriptSearchCount.textContent = "";
            return;
        }

        var lower = query.toLowerCase();
        for (var k = 0; k < segments.length; k++) {
            var textarea = segments[k].querySelector(".transcript-seg-text");
            var text = textarea ? textarea.value : (segments[k].textContent || "");
            if (text.toLowerCase().indexOf(lower) !== -1) {
                _searchMatches.push(k);
                segments[k].classList.add("search-highlight");
                if (clips[k]) clips[k].classList.add("search-highlight");
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
        var segments = el.transcriptSegments ? el.transcriptSegments.querySelectorAll(".transcript-seg") : [];
        var clips = el.transcriptTimeline ? el.transcriptTimeline.querySelectorAll(".transcript-timeline-seg") : [];
        for (var i = 0; i < _searchMatches.length; i++) {
            var idx = _searchMatches[i];
            if (segments[idx]) segments[idx].classList.remove("search-active");
            if (clips[idx]) clips[idx].classList.remove("search-active");
        }
        if (_searchIndex >= 0 && _searchIndex < _searchMatches.length) {
            var activeIdx = _searchMatches[_searchIndex];
            if (segments[activeIdx]) segments[activeIdx].classList.add("search-active");
            if (clips[activeIdx]) clips[activeIdx].classList.add("search-active");
            focusTranscriptSegment(activeIdx, { scroll: true, scrollTimeline: true });
            if (el.transcriptSearchCount) {
                el.transcriptSearchCount.textContent = (_searchIndex + 1) + "/" + _searchMatches.length;
            }
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

            var originalWaveformBtnText = rememberButtonText(el.loadWaveformBtn);
            setButtonText(el.loadWaveformBtn, "Loading…");
            el.loadWaveformBtn.disabled = true;
            var fetchPath = selectedPath; // capture for closure
            var requestSeq = ++_waveformRequestSeq;
            startUtilityJob("/audio/waveform", { file: fetchPath, samples: 500 }, {
                isStale: function () {
                    return requestSeq !== _waveformRequestSeq || fetchPath !== selectedPath;
                },
                onComplete: function (data) {
                    // Cache even if the user switched clips while it was loading.
                    var keys = Object.keys(_waveformCache);
                    if (keys.length >= _WAVEFORM_CACHE_MAX && !_waveformCache[fetchPath]) {
                        delete _waveformCache[keys[0]];
                    }
                    _waveformCache[fetchPath] = data;
                    if (requestSeq !== _waveformRequestSeq || fetchPath !== selectedPath) return;
                    if (!data || !data.peaks) {
                        showToast("Failed to load waveform", "error");
                        return;
                    }
                    _waveformData = data;
                    if (el.waveformContainer) el.waveformContainer.classList.remove("hidden");
                    drawWaveform(data.peaks);
                    updateThresholdLine();
                },
                onError: function (job) {
                    if (requestSeq !== _waveformRequestSeq || fetchPath !== selectedPath) return;
                    showToast((job && (job.error || job.message)) ? "Failed to load waveform: " + (job.error || job.message) : "Failed to load waveform", "error");
                },
                onFinally: function () {
                    if (requestSeq !== _waveformRequestSeq) return;
                    setButtonText(el.loadWaveformBtn, originalWaveformBtnText);
                    el.loadWaveformBtn.disabled = !selectedPath;
                }
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
                var removeBtn = e.target.closest(".fav-chip-remove");
                if (removeBtn) {
                    var removeId = removeBtn.dataset.fav;
                    _favorites = _favorites.filter(function (f) { return f !== removeId; });
                    saveFavorites();
                    renderFavorites();
                    showToast("Removed from favorites", "info");
                    return;
                }
                var chip = e.target.closest(".fav-chip");
                if (!chip) return;
                var favId = chip.dataset.fav;
                var op = _favoriteOps[favId];
                if (op) navigateToTab(op.tab, op.sub);
            });
        }
        var frag = document.createDocumentFragment();
        var visibleCount = 0;
        for (var i = 0; i < _favorites.length; i++) {
            var favId = _favorites[i];
            var op = _favoriteOps[favId];
            if (!op) continue;
            var group = document.createElement("div");
            group.className = "favorite-chip-group";
            var chip = document.createElement("button");
            chip.type = "button";
            chip.className = "fav-chip";
            chip.dataset.fav = favId;
            chip.textContent = op.label;
            chip.title = op.label;
            chip.setAttribute("aria-label", "Open favorite " + op.label);
            var removeBtn = document.createElement("button");
            removeBtn.type = "button";
            removeBtn.className = "fav-chip-remove";
            removeBtn.dataset.fav = favId;
            removeBtn.textContent = "x";
            removeBtn.title = "Remove favorite";
            removeBtn.setAttribute("aria-label", "Remove favorite " + op.label);
            group.appendChild(chip);
            group.appendChild(removeBtn);
            frag.appendChild(group);
            visibleCount++;
        }
        if (!visibleCount) {
            el.favoritesItems.innerHTML = "";
            el.favoritesBar.classList.add("hidden");
            return;
        }
        el.favoritesItems.innerHTML = "";
        el.favoritesItems.appendChild(frag);
    }

    function navigateToTab(tab, sub) {
        var navBtn = activateNavTab(tab, { sub: sub, focus: false });
        if (!navBtn) return;
        if (sub) {
            var panel = $("panel-" + tab);
            var subBtn = panel ? panel.querySelector('.sub-tab[data-sub="' + sub + '"]') : null;
            if (subBtn) subBtn.focus();
        } else {
            navBtn.focus();
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
        api("POST", "/favorites/save", { favorites: _favorites }, function (err) {
            if (err) console.warn("API call failed:", err);
        });
    }

    // ================================================================
    // Side-by-Side Preview
    // ================================================================
    function setPreviewModalMode(mode) {
        var single = mode !== "compare";
        var title = document.getElementById("previewModalTitle");
        var originalLabel = document.getElementById("previewOriginalLabel");
        var processedLabel = document.getElementById("previewProcessedLabel");
        var processedPane = document.getElementById("previewProcessedPane");
        var divider = document.getElementById("previewDivider");
        if (el.previewModal) {
            el.previewModal.classList.toggle("is-single", single);
        }
        if (title) title.textContent = single ? "Frame Preview" : "Before / After Preview";
        if (originalLabel) originalLabel.textContent = single ? "Selected Frame" : "Original";
        if (processedLabel) processedLabel.textContent = "Processed Preview";
        if (processedPane) processedPane.setAttribute("aria-hidden", single ? "true" : "false");
        if (divider) divider.setAttribute("aria-hidden", single ? "true" : "false");
    }

    function closePreviewModal() {
        if (el.previewModal) el.previewModal.classList.add("hidden");
        setPreviewModalMode("single");
        if (el.previewOriginal) el.previewOriginal.removeAttribute("src");
        if (el.previewProcessed) el.previewProcessed.removeAttribute("src");
    }

    function initPreviewModal() {
        if (el.previewModalClose) {
            el.previewModalClose.addEventListener("click", closePreviewModal);
        }
        if (el.previewModal) {
            el.previewModal.addEventListener("click", function (e) {
                if (e.target === el.previewModal) closePreviewModal();
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
        var previewPath = selectedPath;
        var requestSeq = ++_previewModalRequestSeq;
        setPreviewModalMode("single");
        if (el.previewRefreshBtn) {
            el.previewRefreshBtn.disabled = true;
            var refreshLabel = el.previewRefreshBtn.querySelector(".btn-label");
            if (refreshLabel) refreshLabel.textContent = "Loading…";
        }
        // Load original frame
        startUtilityJob("/video/preview-frame", { filepath: previewPath, timestamp: ts }, {
            isStale: function () {
                return requestSeq !== _previewModalRequestSeq || previewPath !== selectedPath;
            },
            onComplete: function (data) {
                if (requestSeq !== _previewModalRequestSeq || previewPath !== selectedPath) return;
                if (!data || !data.image) {
                    showToast("Preview frame unavailable", "error");
                    return;
                }
                if (el.previewOriginal) el.previewOriginal.src = "data:image/jpeg;base64," + data.image;
                if (el.previewProcessed) el.previewProcessed.removeAttribute("src");
                if (el.previewModal) el.previewModal.classList.remove("hidden");
                if (el.previewModalClose) el.previewModalClose.focus();
            },
            onError: function (job) {
                if (requestSeq !== _previewModalRequestSeq || previewPath !== selectedPath) return;
                showToast((job && (job.error || job.message)) ? "Preview failed: " + (job.error || job.message) : "Preview failed", "error");
            },
            onFinally: function () {
                if (requestSeq !== _previewModalRequestSeq) return;
                if (el.previewRefreshBtn) {
                    el.previewRefreshBtn.disabled = false;
                    var refreshLabel = el.previewRefreshBtn.querySelector(".btn-label");
                    if (refreshLabel) refreshLabel.textContent = "Refresh Frame";
                }
            }
        });
    }

    // ================================================================
    // Audio Preview Player
    // ================================================================
    function closeAudioPreview() {
        if (el.audioPreview) el.audioPreview.classList.add("hidden");
        if (el.audioPreviewPlayer) {
            el.audioPreviewPlayer.pause();
            el.audioPreviewPlayer.src = "";
        }
    }

    function initAudioPreview() {
        if (el.audioPreviewClose) {
            el.audioPreviewClose.addEventListener("click", closeAudioPreview);
        }
    }

    function showAudioPreview(filePath) {
        if (!el.audioPreview || !el.audioPreviewPlayer) return;
        el.audioPreviewPlayer.src = BACKEND + "/file?path=" + encodeURIComponent(filePath);
        el.audioPreview.classList.remove("hidden");
        if (el.audioPreviewClose) el.audioPreviewClose.focus();
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
                var firstItem = el.contextMenu.querySelector(".context-menu-item");
                if (firstItem) firstItem.focus();
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
        document.addEventListener("keydown", function (e) {
            if (e.key === "Escape") {
                el.contextMenu.classList.add("hidden");
            }
        });
    }

    // ================================================================
    // First-Run Wizard
    // ================================================================
    var _globalEscapeHandlersAdded = false;

    function closeWizard() {
        if (!el.wizardOverlay) return;
        el.wizardOverlay.classList.add("hidden");
        try {
            var s = JSON.parse(localStorage.getItem(LOCAL_SETTINGS_KEY) || "{}");
            s.wizardDismissed = !!(el.wizardDontShow && el.wizardDontShow.checked);
            localStorage.setItem(LOCAL_SETTINGS_KEY, JSON.stringify(s));
        } catch (e) {}
        if (el.stageChooseMediaBtn) el.stageChooseMediaBtn.focus();
    }

    function initWizard() {
        if (!el.wizardOverlay) return;
        if (!_globalEscapeHandlersAdded) {
            _globalEscapeHandlersAdded = true;
            document.addEventListener("keydown", function (e) {
                if (e.key === "Escape" && el.wizardOverlay && !el.wizardOverlay.classList.contains("hidden")) {
                    closeWizard();
                }
                if (e.key === "Escape" && el.previewModal && !el.previewModal.classList.contains("hidden")) {
                    closePreviewModal();
                }
                if (e.key === "Escape" && el.audioPreview && !el.audioPreview.classList.contains("hidden")) {
                    closeAudioPreview();
                }
                if (e.key === "Escape" && el.recentClipsDropdown && !el.recentClipsDropdown.classList.contains("hidden")) {
                    hideRecentClipsDropdown(false);
                }
                if (e.key === "Escape" && el.outputBrowser && !el.outputBrowser.classList.contains("hidden")) {
                    _outputBrowserOpen = false;
                    el.outputBrowser.classList.add("hidden");
                    setExpanded(el.outputBrowserToggle, false);
                }
            });
        }
        // Check if user has dismissed the wizard before
        try {
            var settings = JSON.parse(localStorage.getItem(LOCAL_SETTINGS_KEY) || "{}");
            if (settings.wizardDismissed) return;
        } catch (e) {}
        // Show wizard
        if (el.wizardDontShow) el.wizardDontShow.checked = false;
        el.wizardOverlay.classList.remove("hidden");
        if (el.wizardCloseBtn) el.wizardCloseBtn.focus();
        // Animate steps
        var steps = el.wizardOverlay.querySelectorAll(".wizard-step");
        for (var i = 1; i < steps.length; i++) {
            (function (idx) {
                setTimeout(function () { steps[idx].classList.add("active"); }, idx * 400);
            })(i);
        }
        if (el.wizardCloseBtn) {
            el.wizardCloseBtn.addEventListener("click", closeWizard);
        }
        el.wizardOverlay.addEventListener("click", function (e) {
            if (e.target === el.wizardOverlay) closeWizard();
        });
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
                setExpanded(el.outputBrowserToggle, _outputBrowserOpen);
                if (_outputBrowserOpen) refreshOutputs();
            });
        }
        if (el.outputBrowserClose) {
            el.outputBrowserClose.addEventListener("click", function () {
                _outputBrowserOpen = false;
                if (el.outputBrowser) el.outputBrowser.classList.add("hidden");
                setExpanded(el.outputBrowserToggle, false);
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
                setToggleButtonCount(el.outputBrowserToggle, "Outputs", data.length);
            }
            if (!el.outputBrowserList) return;
            el.outputBrowserList.textContent = "";
            if (data.length === 0) {
                var emptyHint = document.createElement("div");
                emptyHint.className = "hint";
                emptyHint.style.padding = "8px 12px";
                emptyHint.textContent = "No recent outputs.";
                el.outputBrowserList.appendChild(emptyHint);
                return;
            }
            for (var i = 0; i < data.length; i++) {
                var item = data[i] || {};
                var row = document.createElement("div");
                row.className = "output-item";

                var info = document.createElement("div");
                info.className = "output-item-info";

                var nameEl = document.createElement("div");
                nameEl.className = "output-item-name";
                nameEl.textContent = item.name || (item.path ? item.path.split(/[/\\]/).pop() : "Untitled output");

                var metaEl = document.createElement("div");
                metaEl.className = "output-item-meta";
                metaEl.textContent = safeFixed(item.size_mb, 1) + " MB" + (item.type ? " - " + item.type : "");

                info.appendChild(nameEl);
                info.appendChild(metaEl);

                var actions = document.createElement("div");
                actions.className = "output-item-actions";

                var importBtn = document.createElement("button");
                importBtn.type = "button";
                importBtn.className = "btn-sm";
                importBtn.textContent = "Import";
                importBtn.dataset.path = item.path || "";
                importBtn.disabled = !item.path;
                importBtn.setAttribute("aria-label", "Import " + nameEl.textContent);
                importBtn.addEventListener("click", function () {
                    var p = this.dataset.path || "";
                    if (!p) {
                        showToast("This output is missing a file path.", "error");
                        return;
                    }
                    if (inPremiere && cs) {
                        PremiereBridge.autoImport(p, "output");
                        showToast("Imported: " + p.split(/[/\\]/).pop(), "success");
                    }
                });

                actions.appendChild(importBtn);
                row.appendChild(info);
                row.appendChild(actions);
                el.outputBrowserList.appendChild(row);
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
                if (!selectedPath) { showToast("Select a clip first", "warning"); return; }
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
        // Keep the Cut-tab "Polish batch" button in sync with picker state.
        if (typeof updatePolishBatchButton === "function") updatePolishBatchButton();
        if (!el.batchFileList) return;
        if (_batchFiles.length === 0) {
            el.batchFileList.innerHTML = buildEmptyHintMarkup("No files added", 'Use "Add Selected" or drag files.');
            return;
        }
        var frag = document.createDocumentFragment();
        for (var i = 0; i < _batchFiles.length; i++) {
            var item = document.createElement("div");
            item.className = "batch-file-item";
            var name = _batchFiles[i].split(/[/\\]/).pop();
            item.innerHTML = '<div class="batch-file-main">' +
                '<span class="batch-file-index">' + (i + 1) + '</span>' +
                '<span class="batch-file-name">' + esc(name) + '</span>' +
                '</div>' +
                '<button type="button" class="batch-file-remove" data-idx="' + i + '">Remove</button>';
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
        el.depGrid.innerHTML = '<div class="hint">Checking dependencies…</div>';
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
                    if (err || !data) { showToast("Couldn't export settings", "error"); return; }
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
                            if (err) { showToast("Couldn't import settings", "error"); return; }
                            if (data.localStorage) {
                                localStorage.setItem("opencut_settings", JSON.stringify(data.localStorage));
                                loadLocalSettings();
                            }
                            showToast("Settings imported: " + (result.imported || []).join(", "), "success");
                            if (typeof initPresets === "function") initPresets();
                        });
                    } catch (ex) {
                        showToast("This file doesn't contain valid OpenCut settings", "error");
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
                    if (err) { showToast("Couldn't clear the log file", "error"); return; }
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
                api("POST", "/workflow/save", { name: name, steps: _workflowSteps }, function (err, data) {
                    if (err || (data && data.error)) { showToast(data ? data.error : "Save failed", "error"); return; }
                    showToast("Workflow saved: " + name, "success");
                    refreshSavedWorkflows();
                    loadWorkflowPresets();
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
                api("DELETE", "/workflow/delete", { name: sel.value }, function (err, data) {
                    if (!err && !(data && data.error)) {
                        showToast("Workflow deleted", "success");
                        refreshSavedWorkflows();
                        loadWorkflowPresets();
                    }
                });
            });
        }
        if (el.runCustomWorkflowBtn) {
            el.runCustomWorkflowBtn.addEventListener("click", function () {
                if (_workflowSteps.length === 0 || !selectedPath) return;
                // Use server-side workflow runner for reliable chained execution
                startJob("/workflow/run", {
                    filepath: selectedPath,
                    workflow: { steps: _workflowSteps },
                    output_dir: projectFolder,
                });
            });
        }
        refreshSavedWorkflows();
    }

    function renderWorkflowSteps() {
        if (!el.workflowStepList) return;
        if (_workflowSteps.length === 0) {
            el.workflowStepList.innerHTML = buildEmptyHintMarkup("Workflow is empty", "Add steps to build a custom workflow.");
            if (el.runCustomWorkflowBtn) el.runCustomWorkflowBtn.disabled = true;
            return;
        }
        if (el.runCustomWorkflowBtn) el.runCustomWorkflowBtn.disabled = false;
        var frag = document.createDocumentFragment();
        for (var i = 0; i < _workflowSteps.length; i++) {
            var item = document.createElement("div");
            item.className = "workflow-step-item";
            item.innerHTML = '<div class="workflow-step-main">' +
                '<span class="workflow-step-num">' + (i + 1) + '</span>' +
                '<span class="workflow-step-label">' + esc(_workflowSteps[i].label) + '</span>' +
                '</div>' +
                '<button type="button" class="workflow-step-remove" data-idx="' + i + '">Remove</button>';
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
    // Status Bar — Health Monitoring (Phase 4.3)
    // ================================================================
    // NOTE: _statusTimer is declared once at module scope (line ~49) so
    // cleanupTimers() can clear it. Do NOT redeclare it here.
    var _STATUS_POLL_MS = 5000;

    var _statusBarRetries = 0;
    function initStatusBar() {
        // Don't start polling until the initial health check has connected
        if (!connected) {
            _statusBarRetries++;
            if (_statusBarRetries < 60) { // Cap at 60 retries (1 minute)
                setTimeout(initStatusBar, 1000);
            }
            return;
        }
        // Idempotent: skip if a poller is already running (re-entry from
        // startBackgroundPollers() on reconnect, or from a second init call).
        if (_statusTimer) return;
        _statusBarRetries = 0;
        pollSystemStatus();
        _statusTimer = setInterval(pollSystemStatus, _STATUS_POLL_MS);
    }

    function pollSystemStatus() {
        api("GET", "/system/status", null, function (err, data) {
            var dot = el.statusDot;
            var text = el.statusText;
            var gpu = el.statusGpu;
            var jobsEl = el.statusJobs;
            if (!dot || !text || !gpu || !jobsEl) return;

            if (err || !data || !data.connected) {
                // Disconnected
                dot.className = "status-dot";
                if (el.statusBar) el.statusBar.setAttribute("data-state", "offline");
                text.textContent = "Disconnected";
                gpu.textContent = "GPU: --";
                jobsEl.textContent = "Jobs: --";
                return;
            }

            // Determine dot state
            if (data.gpu && data.gpu.available) {
                dot.className = "status-dot connected";
                if (el.statusBar) el.statusBar.setAttribute("data-state", "online");
            } else {
                dot.className = "status-dot degraded";
                if (el.statusBar) el.statusBar.setAttribute("data-state", "degraded");
            }

            // Uptime text
            var up = data.uptime_seconds || 0;
            var upStr;
            if (up >= 3600) {
                upStr = Math.floor(up / 3600) + "h " + Math.floor((up % 3600) / 60) + "m";
            } else if (up >= 60) {
                upStr = Math.floor(up / 60) + "m";
            } else {
                upStr = up + "s";
            }
            text.textContent = "Up " + upStr;

            // CPU/RAM if available
            if (data.cpu_percent > 0 || data.ram_used_mb > 0) {
                text.textContent += " \u00B7 CPU " + Math.round(data.cpu_percent) + "%";
                if (data.ram_total_mb > 0) {
                    text.textContent += " \u00B7 RAM " + Math.round(data.ram_used_mb / 1024 * 10) / 10 + "/" + Math.round(data.ram_total_mb / 1024 * 10) / 10 + "GB";
                }
            }

            // GPU
            if (data.gpu && data.gpu.available) {
                var gpuLabel = data.gpu.name || "GPU";
                // Shorten long GPU names
                gpuLabel = gpuLabel.replace("NVIDIA ", "").replace("GeForce ", "");
                if (data.gpu.vram_total_mb > 0) {
                    gpu.textContent = gpuLabel + " " + Math.round(data.gpu.vram_used_mb / 1024 * 10) / 10 + "/" + Math.round(data.gpu.vram_total_mb / 1024 * 10) / 10 + "GB";
                } else {
                    gpu.textContent = gpuLabel;
                }
            } else {
                gpu.textContent = "GPU: N/A";
            }

            // Jobs
            var j = data.jobs || {};
            var parts = [];
            if (j.running) parts.push(j.running + " running");
            if (j.queued) parts.push(j.queued + " queued");
            if (!parts.length && j.completed_today) parts.push(j.completed_today + " done today");
            jobsEl.textContent = parts.length ? "Jobs: " + parts.join(", ") : "Jobs: 0";
        }, 4000);
    }

    // ================================================================
    // i18n / Localization Framework
    // ================================================================
    var _currentLang = "en";
    var _i18n = {};

    function t(key, fallback) {
        return _i18n[key] || fallback || key;
    }

    function applyI18nToDOM() {
        var els = document.querySelectorAll("[data-i18n]");
        for (var i = 0; i < els.length; i++) {
            var k = els[i].getAttribute("data-i18n");
            if (!k) continue;
            var translated = t(k);
            var labelTarget = els[i].querySelector(".btn-label, .i18n-text");
            if (labelTarget) labelTarget.textContent = translated;
            else els[i].textContent = translated;
        }
    }

    function loadLocale(lang) {
        // Always load English first as base, then overlay the target locale
        function _loadJson(locale, cb) {
            var xhr = new XMLHttpRequest();
            xhr.open("GET", "locales/" + locale + ".json", true);
            xhr.onload = function () {
                if (xhr.status === 200) {
                    try { cb(JSON.parse(xhr.responseText)); } catch (e) { cb(null); }
                } else { cb(null); }
            };
            xhr.onerror = function () { cb(null); };
            xhr.send();
        }
        _loadJson("en", function (enData) {
            _i18n = enData || {};
            if (lang === "en") {
                _currentLang = "en";
                applyI18nToDOM();
                return;
            }
            _loadJson(lang, function (localeData) {
                if (localeData) {
                    // Merge locale over English base so missing keys fall back to English
                    for (var k in localeData) {
                        if (localeData.hasOwnProperty(k)) _i18n[k] = localeData[k];
                    }
                    _currentLang = lang;
                } else {
                    _currentLang = "en";
                    showToast("Language '" + lang + "' not available yet, using English", "info");
                }
                applyI18nToDOM();
            });
        });
    }

    function initI18n() {
        // Load saved language preference
        var savedLang = "en";
        try {
            var saved = localStorage.getItem(LOCAL_SETTINGS_KEY);
            if (saved) {
                var settings = JSON.parse(saved);
                if (settings.lang) savedLang = settings.lang;
            }
        } catch (e) {}

        // Load the locale file
        loadLocale(savedLang);

        if (el.settingsLang) {
            if (savedLang !== "en") el.settingsLang.value = savedLang;
            el.settingsLang.addEventListener("change", function () {
                _currentLang = this.value;
                saveLocalSettings();
                loadLocale(_currentLang);
            });
        }
    }

    // ================================================================
    // Enhanced Job History (with re-run and details)
    // ================================================================

    // ================================================================
    // v1.3.0 - Clip Preview Thumbnail
    // ================================================================
    function updateClipPreview(pathOverride) {
        if (!el.clipPreviewRow) return;
        var clipPath = pathOverride || selectedPath;
        if (!clipPath) {
            el.clipPreviewRow.classList.add("hidden");
            if (el.clipThumb) el.clipThumb.innerHTML = "";
            clearClipPreviewMeta();
            return;
        }
        el.clipPreviewRow.classList.remove("hidden");
        if (el.clipThumb) el.clipThumb.innerHTML = '<div class="clip-thumb-loading"></div>';
        clearClipPreviewMeta();
        if (_clipInfoCache[clipPath]) {
            applyClipPreviewMeta(clipPath, _clipInfoCache[clipPath]);
        }
        // Fetch thumbnail
        var previewPath = clipPath;
        var requestSeq = ++_clipThumbRequestSeq;
        startUtilityJob("/video/preview-frame", { filepath: previewPath, timestamp: "00:00:01", width: 160 }, {
            isStale: function () {
                return requestSeq !== _clipThumbRequestSeq || previewPath !== selectedPath;
            },
            onComplete: function(data) {
                if (requestSeq !== _clipThumbRequestSeq || previewPath !== selectedPath) return;
                if (!data || !data.image) {
                    if (el.clipThumb) el.clipThumb.innerHTML = '<div class="clip-thumb-none">No Preview</div>';
                    return;
                }
                if (el.clipThumb) {
                    var img = document.createElement("img");
                    img.src = "data:image/jpeg;base64," + data.image;
                    img.alt = (selectedName || previewPath.split(/[/\\]/).pop() || "Clip") + " preview frame";
                    el.clipThumb.innerHTML = "";
                    el.clipThumb.appendChild(img);
                }
            },
            onError: function() {
                if (requestSeq !== _clipThumbRequestSeq || previewPath !== selectedPath) return;
                if (el.clipThumb) el.clipThumb.innerHTML = '<div class="clip-thumb-none">No Preview</div>';
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

    function getRecentClipItems() {
        if (!el.recentClipsDropdown) return [];
        return Array.prototype.slice.call(el.recentClipsDropdown.querySelectorAll(".recent-clip-item"));
    }

    function getRecentClipButtons() {
        var buttons = getRecentClipItems();
        if (!el.recentClipsDropdown) return buttons;
        var clearBtn = el.recentClipsDropdown.querySelector(".recent-clips-clear");
        if (clearBtn) buttons.push(clearBtn);
        return buttons;
    }

    function focusRecentClipItem(position) {
        var items = getRecentClipItems();
        if (items.length) {
            items[position === "last" ? items.length - 1 : 0].focus();
            return;
        }
        var buttons = getRecentClipButtons();
        if (buttons.length) buttons[0].focus();
    }

    function hideRecentClipsDropdown(returnFocus) {
        if (!el.recentClipsDropdown) return;
        el.recentClipsDropdown.classList.add("hidden");
        setExpanded(el.recentClipsBtn, false);
        if (returnFocus && el.recentClipsBtn) el.recentClipsBtn.focus();
    }

    function clearRecentClipsHistory() {
        _recentClips = [];
        saveRecentClips();
        renderRecentClipsDropdown();
        showToast("Cleared recent clips", "info");
    }

    function renderRecentClipsDropdown() {
        if (!el.recentClipsDropdown) return;
        loadRecentClips();
        if (_recentClips.length === 0) {
            setHintContent(el.recentClipsDropdown, "No recent clips yet.");
            return;
        }

        el.recentClipsDropdown.innerHTML = "";
        var header = document.createElement("div");
        header.className = "recent-clips-header";

        var copy = document.createElement("div");
        copy.className = "recent-clips-copy";

        var title = document.createElement("div");
        title.className = "recent-clips-title";
        title.textContent = "Recent clips";
        copy.appendChild(title);

        var subtitle = document.createElement("div");
        subtitle.className = "recent-clips-subtitle";
        subtitle.textContent = "Jump back into a source without rescanning.";
        copy.appendChild(subtitle);
        header.appendChild(copy);

        var clearBtn = document.createElement("button");
        clearBtn.type = "button";
        clearBtn.className = "recent-clips-clear";
        clearBtn.setAttribute("data-action", "clear");
        clearBtn.setAttribute("aria-label", "Clear recent clip history");
        clearBtn.textContent = "Clear";
        header.appendChild(clearBtn);
        el.recentClipsDropdown.appendChild(header);

        var list = document.createElement("div");
        list.className = "recent-clips-list";

        for (var i = 0; i < _recentClips.length; i++) {
            var path = _recentClips[i];
            var name = path.split(/[/\\]/).pop();
            var item = document.createElement("button");
            item.type = "button";
            item.className = "recent-clip-item";
            item.setAttribute("data-path", path);
            item.setAttribute("data-name", name);
            item.title = path;
            item.setAttribute("aria-label", "Open recent clip " + name);
            if (path === selectedPath) {
                item.classList.add("is-current");
                item.setAttribute("aria-current", "true");
            }

            var nameNode = document.createElement("span");
            nameNode.className = "recent-clip-name";
            nameNode.textContent = name;
            item.appendChild(nameNode);

            var pathNode = document.createElement("span");
            pathNode.className = "recent-clip-path";
            pathNode.textContent = path;
            item.appendChild(pathNode);

            list.appendChild(item);
        }
        el.recentClipsDropdown.appendChild(list);
    }

    function showRecentClips(options) {
        if (!el.recentClipsDropdown) return;
        closeAllDropdowns();
        renderRecentClipsDropdown();
        el.recentClipsDropdown.classList.remove("hidden");
        setExpanded(el.recentClipsBtn, true);
        if (options && options.focus) focusRecentClipItem(options.focus === "last" ? "last" : "first");
    }

    function toggleRecentClips(options) {
        if (!el.recentClipsDropdown) return;
        if (el.recentClipsDropdown.classList.contains("hidden")) showRecentClips(options);
        else hideRecentClipsDropdown(options && options.returnFocus);
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
        {name: "Voice Generation", tab: "audio", sub: "aud-tts", keywords: "tts voice speech generate voiceover narration"},
        {name: "Music AI", tab: "audio", sub: "aud-musicai", keywords: "music generate ai musicgen"},
        {name: "Sound Effects", tab: "audio", sub: "aud-sfx", keywords: "sfx sound effect tone"},
        {name: "Audio Ducking", tab: "audio", sub: "aud-duck", keywords: "duck ducking lower music dialogue"},
        {name: "Video Effects", tab: "video", sub: "vid-effects", keywords: "stabilize vignette grain letterbox"},
        {name: "Reframe", tab: "video", sub: "vid-reframe", keywords: "reframe resize phone tiktok shorts vertical portrait"},
        {name: "Merge Clips", tab: "video", sub: "vid-merge", keywords: "merge concatenate join combine clips"},
        {name: "Speed / Ramp", tab: "video", sub: "vid-speed", keywords: "speed slow fast ramp reverse"},
        {name: "Compositing & Keying", tab: "video", sub: "vid-chroma", keywords: "chroma green screen key pip blend composite"},
        {name: "Transitions", tab: "video", sub: "vid-transition", keywords: "transition fade wipe slide"},
        {name: "Upscale", tab: "video", sub: "vid-upscale", keywords: "upscale enhance resolution ai"},
        {name: "Color Correction", tab: "video", sub: "vid-color", keywords: "color correct grade exposure contrast"},
        {name: "LUTs", tab: "video", sub: "vid-lut", keywords: "lut color grade cinematic film look"},
        {name: "Face AI", tab: "video", sub: "vid-faceswap", keywords: "face swap enhance gfpgan"},
        {name: "De-Logo / Remove Object", tab: "video", sub: "vid-remove", keywords: "remove watermark object logo delogo inpaint"},
        {name: "Titles", tab: "video", sub: "vid-titles", keywords: "title text overlay lower third"},
        {name: "Export Presets", tab: "export", sub: "exp-platform", keywords: "export platform youtube tiktok instagram"},
        {name: "Thumbnails", tab: "export", sub: "exp-thumbnail", keywords: "thumbnail extract frame"},
        {name: "Batch Processing", tab: "export", sub: "exp-batch", keywords: "batch process multiple files"},
        { name: "Repeat Detection",   tab: "captions", sub: "cap-repeat",     keywords: "repeat detect loop fumble duplicate take" },
        { name: "Chapter Generation", tab: "captions", sub: "cap-chapters",   keywords: "chapters youtube timestamps sections topics" },
        { name: "Footage Search",     tab: "nlp",      sub: "nlp-search",     keywords: "search footage clips index content find" },
        { name: "Color Match",        tab: "video",    sub: "vid-color",      keywords: "color match grade balance reference clip" },
        { name: "Multicam Switcher",  tab: "timeline", sub: "tl-multicam",    keywords: "multicam speaker podcast camera switch diarize" },
        { name: "Loudness Match",     tab: "audio",    sub: "aud-loudmatch",  keywords: "loudness lufs normalize match audio levels" },
        { name: "Auto Zoom",          tab: "video",    sub: "vid-effects",    keywords: "auto zoom push in ken burns face zoom" },
        { name: "AI Command",         tab: "nlp",      sub: "nlp-command",    keywords: "nlp ai command natural language instruction" },
        { name: "Deliverables",       tab: "export",   sub: "exp-deliverables", keywords: "deliverables vfx adr music cue sheet asset list" },
        { name: "Auto Shorts",        tab: "export",   sub: "exp-shorts",       keywords: "shorts tiktok reels auto highlight clip vertical" },
        { name: "Workflow Presets",   tab: "export",   sub: "exp-workflow",     keywords: "workflow preset pipeline chain steps auto" },
        { name: "Project Templates",  tab: "settings", sub: "set-prefs",       keywords: "template project youtube podcast broadcast cinema preset" },
        { name: "Keyboard Shortcuts", tab: "settings", sub: "set-prefs",       keywords: "keyboard shortcut hotkey keybind" },
        { name: "Job History",        tab: "settings", sub: "set-system",      keywords: "job history log past completed failed" },
    ];

    var PALETTE_HISTORY_KEY = "opencut_palette_history";
    var MAX_PALETTE_HISTORY = 8;
    var PALETTE_DESCRIPTION_MAP = {
        "Silence Removal::cut::silence": "Strip dead air and tighten spoken takes.",
        "Styled Captions::captions::cap-styled": "Create branded subtitles with faster review passes.",
        "Denoise::audio::aud-denoise": "Clean dialogue and reduce room noise before finishing.",
        "Reframe::video::vid-reframe": "Recompose shots for vertical, square, and alternate aspect ratios.",
        "Export Presets::export::exp-platform": "Package platform-ready outputs without rebuilding settings.",
        "Footage Search::nlp::nlp-search": "Find usable shots from transcripts and semantic matches.",
        "AI Command::nlp::nlp-command": "Jump to tools and edit actions from natural-language prompts."
    };

    var _paletteSelectedIdx = -1;
    var _paletteResults = [];
    var _paletteReturnFocusEl = null;

    function normalizePaletteText(value) {
        return (value || "").toLowerCase().replace(/\s+/g, " ").trim();
    }

    function formatPaletteLabel(value) {
        return (value || "").replace(/[-_]+/g, " ").replace(/\b[a-z]/g, function (letter) {
            return letter.toUpperCase();
        });
    }

    function getPaletteItemKey(item) {
        if (!item) return "";
        return [item.name || "", item.tab || "", item.sub || ""].join("::");
    }

    function loadPaletteHistory() {
        try {
            var saved = localStorage.getItem(PALETTE_HISTORY_KEY);
            var parsed = saved ? JSON.parse(saved) : [];
            return Array.isArray(parsed) ? parsed : [];
        } catch (e) {
            return [];
        }
    }

    function savePaletteHistory(history) {
        try {
            localStorage.setItem(PALETTE_HISTORY_KEY, JSON.stringify((history || []).slice(0, MAX_PALETTE_HISTORY)));
        } catch (e) {
            // localStorage may not be available in CEP
        }
    }

    function getPaletteItemByKey(key) {
        for (var i = 0; i < _commandIndex.length; i++) {
            if (getPaletteItemKey(_commandIndex[i]) === key) return _commandIndex[i];
        }
        return null;
    }

    function rememberPaletteItem(item) {
        var key = getPaletteItemKey(item);
        if (!key) return;
        var history = loadPaletteHistory().filter(function (entry) { return entry !== key; });
        history.unshift(key);
        savePaletteHistory(history);
    }

    function getPaletteTabLabel(tab) {
        var navBtn = document.querySelector('.nav-tab[data-nav="' + (tab || "") + '"]');
        if (!navBtn) return formatPaletteLabel(tab);
        return (navBtn.getAttribute("title") || navBtn.textContent || formatPaletteLabel(tab)).replace(/\s+/g, " ").trim();
    }

    function getPaletteSubLabel(sub) {
        var subBtn = document.querySelector('.sub-tab[data-sub="' + (sub || "") + '"]');
        if (!subBtn) return formatPaletteLabel(sub);
        return (subBtn.textContent || formatPaletteLabel(sub)).replace(/\s+/g, " ").trim();
    }

    function getActivePaletteTabName() {
        var activeNav = document.querySelector(".nav-tab.active");
        return activeNav ? (activeNav.getAttribute("data-nav") || "") : "";
    }

    function getPaletteFavoriteId(item) {
        if (!item) return "";
        var normalizedName = normalizePaletteText(item.name);
        var fallback = "";
        for (var favId in _favoriteOps) {
            if (!_favoriteOps.hasOwnProperty(favId)) continue;
            var op = _favoriteOps[favId];
            if (op.tab !== item.tab || op.sub !== item.sub) continue;
            if (normalizePaletteText(op.label) === normalizedName) return favId;
            if (!fallback) fallback = favId;
        }
        return fallback;
    }

    function getPaletteItemForFavorite(favId) {
        var op = _favoriteOps[favId];
        if (!op) return null;
        var preferredLabel = normalizePaletteText(op.label);
        var fallback = null;
        for (var i = 0; i < _commandIndex.length; i++) {
            var item = _commandIndex[i];
            if (item.tab !== op.tab || item.sub !== op.sub) continue;
            if (normalizePaletteText(item.name) === preferredLabel) return item;
            if (!fallback) fallback = item;
        }
        return fallback;
    }

    function getPaletteItemDescription(item) {
        var itemKey = getPaletteItemKey(item);
        if (PALETTE_DESCRIPTION_MAP[itemKey]) return PALETTE_DESCRIPTION_MAP[itemKey];
        if (!item) return "Open tools across the editing workflow.";
        switch (item.tab) {
        case "cut":
            return "Tighten pacing, trims, and spoken edits from one focused cut workflow.";
        case "captions":
            return "Transcribe, translate, and shape subtitle deliverables without leaving the panel.";
        case "audio":
            return "Polish dialogue, stems, loudness, and generated sound from one audio surface.";
        case "video":
            return "Repair, reframe, and finish image work with cleaner visual controls.";
        case "export":
            return "Build deliverables, thumbnails, and repeatable output presets faster.";
        case "timeline":
            return "Write sequence edits and timeline metadata back into Premiere with more control.";
        case "nlp":
            return "Use search and language-driven tools to find footage or trigger edit actions.";
        case "settings":
            return "Adjust workspace defaults, templates, and system-level behavior.";
        default:
            return "Open this tool and jump directly to the matching workspace.";
        }
    }

    function scorePaletteItem(item, query) {
        var q = normalizePaletteText(query);
        if (!q) return 0;

        var name = normalizePaletteText(item.name);
        var keywords = normalizePaletteText(item.keywords);
        var tabLabel = normalizePaletteText(getPaletteTabLabel(item.tab));
        var subLabel = normalizePaletteText(getPaletteSubLabel(item.sub));
        var score = 0;
        var matchedTokens = 0;
        var tokens = q.split(" ");
        var favoriteId = getPaletteFavoriteId(item);

        if (name === q) score += 220;
        else if (name.indexOf(q) === 0) score += 140;
        else if (name.indexOf(q) !== -1) score += 96;

        if (keywords.indexOf(q) !== -1) score += 56;
        if (tabLabel.indexOf(q) !== -1) score += 24;
        if (subLabel.indexOf(q) !== -1) score += 28;

        for (var i = 0; i < tokens.length; i++) {
            var token = tokens[i];
            if (!token) continue;
            if (name.indexOf(token) !== -1) {
                score += 32;
                matchedTokens++;
            } else if (keywords.indexOf(token) !== -1) {
                score += 18;
                matchedTokens++;
            } else if (tabLabel.indexOf(token) !== -1 || subLabel.indexOf(token) !== -1) {
                score += 10;
                matchedTokens++;
            }
        }

        if (!score && !matchedTokens) return 0;
        if (matchedTokens > 1) score += matchedTokens * 8;
        if (favoriteId && _favorites.indexOf(favoriteId) !== -1) score += 16;
        if (item.tab === getActivePaletteTabName()) score += 12;
        return score;
    }

    function createPaletteEntry(item, extras) {
        extras = extras || {};
        var key = getPaletteItemKey(item);
        var favoriteId = getPaletteFavoriteId(item);
        var tabLabel = getPaletteTabLabel(item.tab);
        var subLabel = getPaletteSubLabel(item.sub);
        return {
            item: item,
            key: key,
            description: getPaletteItemDescription(item),
            tabLabel: tabLabel,
            subLabel: subLabel,
            location: subLabel ? (tabLabel + " / " + subLabel) : tabLabel,
            favoriteId: favoriteId,
            isFavorite: favoriteId ? _favorites.indexOf(favoriteId) !== -1 : false,
            isRecent: !!extras.isRecent,
            isCurrent: !!extras.isCurrent,
            score: extras.score || 0
        };
    }

    function buildPaletteEntries(items, resolver, seen) {
        var entries = [];
        for (var i = 0; i < items.length; i++) {
            var item = items[i];
            if (!item) continue;
            var key = getPaletteItemKey(item);
            if (seen && seen[key]) continue;
            if (seen) seen[key] = true;
            entries.push(createPaletteEntry(item, resolver ? resolver(item, key, i) : null));
        }
        return entries;
    }

    function addPaletteSection(sections, label, items, resolver, seen) {
        var entries = buildPaletteEntries(items, resolver, seen);
        if (entries.length) sections.push({ label: label, entries: entries });
    }

    function buildPaletteSections(query) {
        var q = normalizePaletteText(query);
        var sections = [];
        var activeTab = getActivePaletteTabName();
        var historyKeys = loadPaletteHistory();
        var historyLookup = {};
        for (var i = 0; i < historyKeys.length; i++) historyLookup[historyKeys[i]] = true;

        if (!q) {
            var seen = {};
            var recentItems = [];
            for (i = 0; i < historyKeys.length; i++) {
                var historyItem = getPaletteItemByKey(historyKeys[i]);
                if (historyItem) recentItems.push(historyItem);
            }

            var favoriteItems = [];
            for (i = 0; i < _favorites.length; i++) {
                favoriteItems.push(getPaletteItemForFavorite(_favorites[i]));
            }

            var currentItems = [];
            for (i = 0; i < _commandIndex.length; i++) {
                if (_commandIndex[i].tab === activeTab) currentItems.push(_commandIndex[i]);
            }

            var browseItems = _commandIndex.slice(0);
            browseItems.sort(function (a, b) {
                var tabCompare = getPaletteTabLabel(a.tab).localeCompare(getPaletteTabLabel(b.tab));
                if (tabCompare !== 0) return tabCompare;
                return a.name.localeCompare(b.name);
            });

            addPaletteSection(sections, "Recent", recentItems, function (item) {
                return {
                    isRecent: true,
                    isCurrent: item.tab === activeTab
                };
            }, seen);

            addPaletteSection(sections, "Favorites", favoriteItems, function (item, key) {
                return {
                    isRecent: !!historyLookup[key],
                    isCurrent: item.tab === activeTab
                };
            }, seen);

            addPaletteSection(sections, activeTab ? "Current Workspace" : "Suggested Tools", currentItems, function (item, key) {
                return {
                    isRecent: !!historyLookup[key],
                    isCurrent: true
                };
            }, seen);

            addPaletteSection(sections, "Browse All", browseItems, function (item, key) {
                return {
                    isRecent: !!historyLookup[key],
                    isCurrent: item.tab === activeTab
                };
            }, seen);
            return sections;
        }

        var matches = [];
        for (i = 0; i < _commandIndex.length; i++) {
            var commandItem = _commandIndex[i];
            var score = scorePaletteItem(commandItem, q);
            if (!score) continue;
            matches.push(createPaletteEntry(commandItem, {
                score: score,
                isRecent: !!historyLookup[getPaletteItemKey(commandItem)],
                isCurrent: commandItem.tab === activeTab
            }));
        }

        matches.sort(function (a, b) {
            if (b.score !== a.score) return b.score - a.score;
            if (a.isFavorite !== b.isFavorite) return a.isFavorite ? -1 : 1;
            if (a.isCurrent !== b.isCurrent) return a.isCurrent ? -1 : 1;
            return a.item.name.localeCompare(b.item.name);
        });

        if (matches.length) sections.push({ label: "Matching Tools", entries: matches });
        return sections;
    }

    function createPaletteBadge(label, extraClass) {
        var badge = document.createElement("span");
        badge.className = "command-palette-badge" + (extraClass ? (" " + extraClass) : "");
        badge.textContent = label;
        return badge;
    }

    function renderPaletteItem(entry, idx) {
        var itemNode = document.createElement("button");
        itemNode.type = "button";
        itemNode.className = "command-palette-item";
        itemNode.id = "commandPaletteOption" + idx;
        itemNode.setAttribute("role", "option");
        itemNode.setAttribute("aria-selected", "false");
        itemNode.setAttribute("data-idx", String(idx));
        itemNode.setAttribute("data-key", entry.key);
        itemNode.setAttribute("data-tab", entry.item.tab);
        itemNode.setAttribute("data-sub", entry.item.sub);
        itemNode.setAttribute("aria-label", entry.item.name + ". " + entry.description + ". " + entry.location + ".");

        var main = document.createElement("div");
        main.className = "command-palette-item-main";

        var top = document.createElement("div");
        top.className = "command-palette-item-top";

        var name = document.createElement("span");
        name.className = "command-palette-name";
        name.textContent = entry.item.name;
        top.appendChild(name);

        var badges = document.createElement("span");
        badges.className = "command-palette-badges";
        if (entry.isFavorite) badges.appendChild(createPaletteBadge("Pinned", "is-favorite"));
        if (entry.isRecent) badges.appendChild(createPaletteBadge("Recent", "is-recent"));
        if (entry.isCurrent) badges.appendChild(createPaletteBadge("Current", "is-current"));
        top.appendChild(badges);

        var desc = document.createElement("div");
        desc.className = "command-palette-desc";
        desc.textContent = entry.description;

        var meta = document.createElement("div");
        meta.className = "command-palette-meta";

        var location = document.createElement("span");
        location.className = "command-palette-location";
        location.textContent = entry.location;
        meta.appendChild(location);

        var chevron = document.createElement("span");
        chevron.className = "command-palette-chevron";
        chevron.setAttribute("aria-hidden", "true");
        chevron.textContent = "Open";
        meta.appendChild(chevron);

        main.appendChild(top);
        main.appendChild(desc);
        main.appendChild(meta);
        itemNode.appendChild(main);
        return itemNode;
    }

    function updatePaletteStatus(query) {
        if (!el.commandPaletteStatus) return;
        if (!query) {
            var activeTab = getActivePaletteTabName();
            var activeLabel = activeTab ? getPaletteTabLabel(activeTab) : "All Tools";
            el.commandPaletteStatus.textContent = _paletteResults.length ?
                (activeLabel + " tools plus recent actions and pinned shortcuts.") :
                "Run tools or pin favorites to make the launcher smarter.";
            return;
        }

        if (_paletteResults.length === 0) {
            el.commandPaletteStatus.textContent = 'No matches for "' + query + '"';
            return;
        }

        el.commandPaletteStatus.textContent = _paletteResults.length + (_paletteResults.length === 1 ? ' match for "' : ' matches for "') + query + '"';
    }

    function renderPaletteEmptyState(query) {
        if (!el.commandPaletteResults) return;
        var empty = document.createElement("div");
        empty.className = "command-palette-empty";

        var title = document.createElement("div");
        title.className = "command-palette-empty-title";
        title.textContent = query ? "No Matching Tools" : "Launcher Ready";
        empty.appendChild(title);

        var copy = document.createElement("div");
        copy.className = "command-palette-empty-copy";
        copy.textContent = query ?
            'Try a broader term like "audio", "captions", or "export".' :
            "Recent actions and pinned favorites will appear here once you start using the workspace.";
        empty.appendChild(copy);

        el.commandPaletteResults.appendChild(empty);
        updatePaletteStatus(query);
        setPaletteSelectedIndex(-1, false);
    }

    function setPaletteSelectedIndex(index, shouldScroll) {
        if (!el.commandPaletteResults) return;
        var items = el.commandPaletteResults.querySelectorAll(".command-palette-item");
        if (!items.length || index < 0 || !_paletteResults.length) {
            _paletteSelectedIdx = -1;
            if (el.commandPaletteInput) el.commandPaletteInput.removeAttribute("aria-activedescendant");
            return;
        }

        if (index >= _paletteResults.length) index = 0;
        if (index < 0) index = _paletteResults.length - 1;
        _paletteSelectedIdx = index;

        for (var i = 0; i < items.length; i++) {
            var isSelected = i === index;
            items[i].classList.toggle("selected", isSelected);
            items[i].setAttribute("aria-selected", isSelected ? "true" : "false");
        }

        if (!items[index]) return;
        if (el.commandPaletteInput) el.commandPaletteInput.setAttribute("aria-activedescendant", items[index].id);
        if (shouldScroll !== false && items[index].scrollIntoView) {
            items[index].scrollIntoView({ block: "nearest" });
        }
    }

    function renderPaletteResults(query) {
        if (!el.commandPaletteResults) return;
        var q = (query || "").replace(/\s+/g, " ").trim();
        var sections = buildPaletteSections(q);
        _paletteResults = [];
        el.commandPaletteResults.innerHTML = "";
        el.commandPaletteResults.scrollTop = 0;

        if (!sections.length) {
            renderPaletteEmptyState(q);
            return;
        }

        var frag = document.createDocumentFragment();
        for (var i = 0; i < sections.length; i++) {
            var sectionNode = document.createElement("section");
            sectionNode.className = "command-palette-section";

            var label = document.createElement("div");
            label.className = "command-palette-section-label";
            label.textContent = sections[i].label;
            sectionNode.appendChild(label);

            var group = document.createElement("div");
            group.className = "command-palette-section-items";
            group.setAttribute("role", "group");
            group.setAttribute("aria-label", sections[i].label);

            for (var j = 0; j < sections[i].entries.length; j++) {
                var idx = _paletteResults.length;
                _paletteResults.push(sections[i].entries[j]);
                group.appendChild(renderPaletteItem(sections[i].entries[j], idx));
            }

            sectionNode.appendChild(group);
            frag.appendChild(sectionNode);
        }

        el.commandPaletteResults.appendChild(frag);
        updatePaletteStatus(q);
        setPaletteSelectedIndex(_paletteResults.length ? 0 : -1, false);
    }

    function openCommandPalette() {
        if (!el.commandPaletteOverlay || !el.commandPaletteInput || !el.commandPaletteResults) return;
        var previousFocus = document.activeElement && document.activeElement !== document.body ? document.activeElement : null;
        closeAllDropdowns();
        _paletteReturnFocusEl = previousFocus;
        el.commandPaletteOverlay.classList.remove("hidden");
        setExpanded(el.commandPaletteInput, true);
        el.commandPaletteInput.value = "";
        el.commandPaletteInput.removeAttribute("aria-activedescendant");
        renderPaletteResults("");
        setTimeout(function () {
            if (el.commandPaletteInput) el.commandPaletteInput.focus();
        }, 50);
    }

    function closeCommandPalette(options) {
        var restoreFocus = !options || options.restoreFocus !== false;
        if (el.commandPaletteOverlay) el.commandPaletteOverlay.classList.add("hidden");
        if (el.commandPaletteInput) {
            setExpanded(el.commandPaletteInput, false);
            el.commandPaletteInput.removeAttribute("aria-activedescendant");
        }
        _paletteSelectedIdx = -1;
        if (restoreFocus && _paletteReturnFocusEl && typeof _paletteReturnFocusEl.focus === "function") {
            try { _paletteReturnFocusEl.focus(); } catch (e) {}
        }
        _paletteReturnFocusEl = null;
    }

    function executePaletteItem(itemOrTab, sub) {
        var item = itemOrTab && typeof itemOrTab === "object" ? itemOrTab : null;
        if (!item) {
            for (var i = 0; i < _commandIndex.length; i++) {
                if (_commandIndex[i].tab === itemOrTab && _commandIndex[i].sub === sub) {
                    item = _commandIndex[i];
                    break;
                }
            }
        }
        if (!item) return;
        rememberPaletteItem(item);
        closeCommandPalette();
        navigateToTab(item.tab, item.sub);
    }

    function paletteNavigate(dir) {
        if (!_paletteResults.length) return;
        if (_paletteSelectedIdx < 0) {
            setPaletteSelectedIndex(dir > 0 ? 0 : (_paletteResults.length - 1), true);
            return;
        }
        setPaletteSelectedIndex(_paletteSelectedIdx + dir, true);
    }

    function paletteExecuteSelected() {
        if (_paletteSelectedIdx < 0 || !_paletteResults[_paletteSelectedIdx]) return;
        executePaletteItem(_paletteResults[_paletteSelectedIdx].item);
    }

    function initCommandPalette() {
        if (!el.commandPaletteOverlay || !el.commandPaletteInput || !el.commandPaletteResults) return;

        el.commandPaletteInput.addEventListener("input", function () {
            renderPaletteResults(this.value);
        });

        el.commandPaletteInput.addEventListener("keydown", function (e) {
            if (e.key === "ArrowDown") {
                e.preventDefault();
                paletteNavigate(1);
            } else if (e.key === "ArrowUp") {
                e.preventDefault();
                paletteNavigate(-1);
            } else if (e.key === "Home" && _paletteResults.length) {
                e.preventDefault();
                setPaletteSelectedIndex(0, true);
            } else if (e.key === "End" && _paletteResults.length) {
                e.preventDefault();
                setPaletteSelectedIndex(_paletteResults.length - 1, true);
            } else if (e.key === "Enter") {
                e.preventDefault();
                paletteExecuteSelected();
            } else if (e.key === "Escape") {
                e.preventDefault();
                closeCommandPalette();
            }
        });

        el.commandPaletteOverlay.addEventListener("click", function (e) {
            if (e.target === el.commandPaletteOverlay) closeCommandPalette();
        });

        el.commandPaletteResults.addEventListener("mousemove", function (e) {
            var item = e.target.closest(".command-palette-item");
            if (!item) return;
            var idx = parseInt(item.getAttribute("data-idx"), 10);
            if (!isNaN(idx) && idx !== _paletteSelectedIdx) {
                setPaletteSelectedIndex(idx, false);
            }
        });

        el.commandPaletteResults.addEventListener("click", function (e) {
            var itemNode = e.target.closest(".command-palette-item");
            if (!itemNode) return;
            var idx = parseInt(itemNode.getAttribute("data-idx"), 10);
            if (isNaN(idx) || !_paletteResults[idx]) return;
            executePaletteItem(_paletteResults[idx].item);
        });

        // Note: Ctrl+K is now handled by the keyboard shortcut registry in initKeyboardShortcuts
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
            el.mergeFileList.innerHTML = buildEmptyHintMarkup("Nothing queued", "Add two or more files to merge in sequence.");
            if (el.runMergeBtn) el.runMergeBtn.disabled = !connected || _mergeFiles.length < 2;
            return;
        }
        var html = "";
        for (var i = 0; i < _mergeFiles.length; i++) {
            var name = _mergeFiles[i].split(/[/\\]/).pop();
            html += '<div class="merge-file-item">' +
                '<div class="merge-file-main">' +
                '<span class="merge-file-index">' + (i + 1) + '</span>' +
                '<span class="merge-file-name">' + esc(name) + '</span>' +
                '</div>' +
                '<button type="button" class="merge-file-remove" data-idx="' + i + '">Remove</button>' +
                '</div>';
        }
        el.mergeFileList.innerHTML = html;
        if (el.runMergeBtn) el.runMergeBtn.disabled = !connected || _mergeFiles.length < 2;
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
        if (_mergeFiles.length < 2) { showToast("Add at least 2 clips to merge", "warning"); return; }
        startJob("/video/merge", {
            files: _mergeFiles,
            output_dir: projectFolder,
            mode: el.mergeMode ? el.mergeMode.value : "concat",
            quality: el.mergeQuality ? el.mergeQuality.value : "medium",
            no_input: true
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

    function applyLLMConfig(cfg, fillBlankOnly) {
        if (!cfg) return;
        var assign = function (input, value) {
            if (!input || value == null || value === "") return;
            if (fillBlankOnly && input.value) return;
            input.value = value;
        };
        assign(el.llmProvider, cfg.provider);
        assign(el.llmModel, cfg.model);
        assign(el.llmApiKey, cfg.api_key);
        assign(el.llmBaseUrl, cfg.base_url);
        updateLLMProviderUI();
    }

    function saveLLMSettings(options) {
        var cfg = getLLMConfig();
        try {
            localStorage.setItem("opencut_llm", JSON.stringify(cfg));
        } catch (e) {}
        if (options && options.localOnly) return;
        api("POST", "/settings/llm", cfg, function (err) {
            if (err) {
                if (!options || !options.silent) showToast("Failed to save LLM settings", "error");
                return;
            }
            if (options && options.toastSuccess) showToast("LLM settings saved", "success");
        });
    }

    function loadLLMSettings() {
        var hasLocalSettings = false;
        try {
            var saved = localStorage.getItem("opencut_llm");
            if (saved) {
                hasLocalSettings = true;
                applyLLMConfig(JSON.parse(saved), false);
            }
        } catch (e) {}
        api("GET", "/settings/llm", null, function (err, s) {
            if (err || !s) return;
            applyLLMConfig({
                provider: s.provider || "",
                model: s.model || "",
                api_key: s.api_key && s.api_key !== "****" ? s.api_key : "",
                base_url: s.base_url || ""
            }, hasLocalSettings);
        });
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
        if (el.llmStatus) el.llmStatus.textContent = "Testing…";
        api("POST", "/llm/test", { prompt: "Say hello in one sentence.", provider: cfg.provider, model: cfg.model || "", api_key: cfg.api_key || "", base_url: cfg.base_url || "" }, function (err, resp) {
            if (err || !resp || !resp.success) {
                var msg = (resp && resp.error) ? resp.error : (err && typeof err === "object" && err.message) ? err.message : "Couldn't reach the LLM provider";
                if (el.llmStatus) el.llmStatus.textContent = "Failed: " + msg;
                return;
            }
            if (el.llmStatus) el.llmStatus.textContent = "Connected: " + resp.provider + "/" + resp.model;
            saveLLMSettings({ silent: true });
            showToast("LLM connected", "success");
        });
    }

    // --- Silence mode toggle ---
    function loadLlmSettings() {
        loadLLMSettings();
    }

    function saveLlmSettings() {
        saveLLMSettings({ toastSuccess: true });
    }

    function saveAudioZoomDefaults() {
        var lufs = parseFloat((document.getElementById("defaultLufs") || {}).value || -14);
        var zoom = parseFloat((document.getElementById("defaultZoom") || {}).value || 1.15);
        var easing = (document.getElementById("defaultZoomEasing") || {}).value || "ease_in_out";
        api("POST", "/settings/loudness-target", { target_lufs: lufs }, function (err) {
            if (err) console.warn("API call failed:", err);
        });
        api("POST", "/settings/auto-zoom", { zoom_amount: zoom, easing: easing }, function (err) {
            if (err) showToast("Failed to save defaults", "error");
            else showToast("Defaults saved", "success");
        });
    }

    function updateSilenceModeUI() {
        if (!el.silenceMode) return;
        var isSpeedUp = el.silenceMode.value === "speedup";
        if (el.silenceSpeedGroup) el.silenceSpeedGroup.classList.toggle("hidden", !isSpeedUp);
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
        if (el.summaryContent) el.summaryContent.textContent = "Summarizing…";
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
            no_input: true,
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
        cs.evalScript("ocApplySequenceCuts('" + payload.replace(/\\/g, "\\\\").replace(/'/g, "\\'") + "')", function (result) {
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
        var markers = beatMarkerTimes.map(function(t) { return { time: t, name: "Beat", type: "Chapter" }; });
        var payload = JSON.stringify(markers);
        cs.evalScript("ocAddSequenceMarkers('" + payload.replace(/\\/g, "\\\\").replace(/'/g, "\\'") + "')", function (result) {
            try {
                var r = JSON.parse(result);
                showToast("Added " + (r.added || beatMarkerTimes.length) + " markers", "success");
                if (!r.error) {
                    // Journal the op for one-click rollback. Fingerprint each
                    // marker by its {time, comment} pair so the inverse can
                    // find+delete exactly these rows.
                    var fingerprints = [];
                    for (var k = 0; k < markers.length; k++) {
                        fingerprints.push({
                            time: markers[k].time,
                            comment: markers[k].name || "Beat"
                        });
                    }
                    journalRecord(
                        "add_markers",
                        _journalLabelForMarkers(markers.length, selectedName),
                        { markers: fingerprints },
                        selectedPath
                    );
                }
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
        cs.evalScript("ocApplySequenceCuts('" + payload.replace(/\\/g, "\\\\").replace(/'/g, "\\'") + "')", function (result) {
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
            html += '<div class="multicam-track-row">'
                + '<span class="multicam-track-label">Speaker ' + (i + 1) + '</span>'
                + '<span class="multicam-track-arrow">\u2192 Track</span>'
                + '<input type="number" class="multicam-track-input" value="' + i + '" min="0" max="20">'
                + '</div>';
        }
        container.innerHTML = html;
    }

    function getSeqMarkers() {
        if (!inPremiere) { showAlert("Premiere Pro connection required."); return; }
        cs.evalScript('ocGetSequenceMarkers()', function (result) {
            var listEl = document.getElementById("markerExportList");
            try {
                var markers = JSON.parse(result);
                if (!Array.isArray(markers)) {
                    throw new Error((markers && markers.error) || "Unexpected Premiere response");
                }
                seqMarkersData = markers;
                if (listEl) {
                    listEl.classList.remove("hidden");
                    if (!markers || !markers.length) {
                        listEl.innerHTML = '<div class="hint">No markers found in sequence.</div>';
                    } else {
                        var html = "";
                        for (var i = 0; i < markers.length; i++) {
                            var m = markers[i];
                            var dur = m.duration != null ? safeFixed(m.duration, 2) + "s" : "--";
                            html += '<div class="marker-export-item">'
                                + '<div class="marker-export-main">'
                                + '<span class="marker-export-name">' + esc(m.name || ("Marker " + (i + 1))) + '</span>'
                                + '<span class="marker-export-time">' + fmtDur(m.start || 0) + '</span>'
                                + '</div>'
                                + '<span class="marker-export-duration">' + dur + '</span>'
                                + '</div>';
                        }
                        listEl.innerHTML = html;
                    }
                }
                updateButtons();
            } catch (e) {
                seqMarkersData = null;
                if (listEl) {
                    listEl.classList.remove("hidden");
                    listEl.innerHTML = '<div class="hint">Could not read sequence markers.</div>';
                }
                updateButtons();
                showAlert("Error reading markers: " + (result || e.message));
            }
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
        if (sum) sum.textContent = "Exported " + (r.count || r.exported || 0) + " clips.";
    });

    function loadProjectItems() {
        if (!inPremiere) { showAlert("Premiere Pro connection required."); return; }
        // Return cached project media if still fresh
        if (Array.isArray(_pproCache.clips) && (Date.now() - _pproCache.clipsTs < _pproCache.ttl)) {
            renameItemsData = _pproCache.clips;
            renderRenameItems();
            updateButtons();
            return;
        }
        cs.evalScript('getAllProjectMedia()', function (result) {
            try {
                var items = JSON.parse(result);
                if (!Array.isArray(items)) {
                    showAlert("Error loading items: " + ((items && items.error) || "Unexpected Premiere response"));
                    return;
                }
                renameItemsData = items;
                _pproCache.clips = renameItemsData;
                _pproCache.clipsTs = Date.now();
                renderRenameItems();
                updateButtons();
            } catch (e) {
                renameItemsData = [];
                updateButtons();
                showAlert("Error loading items: " + (result || e.message));
            }
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
            html += '<div class="rename-item">'
                + '<span class="rename-item-index">' + (i + 1) + '</span>'
                + '<input type="text" class="text-input rename-name-input" data-idx="' + i + '" value="' + esc(item.name || "") + '">'
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
                renames.push({ nodeId: orig.nodeId || orig.id || orig.path, newName: inputs[i].value });
            }
        }
        if (!renames.length) { showAlert("No changes to apply."); return; }
        api("POST", "/timeline/batch-rename", { renames: renames }, function (err, data) {
            if (err || (data && data.error)) { showAlert("Validation failed: " + (data ? data.error : "Network error")); return; }
            if (!inPremiere) { showToast("Rename validated (no Premiere connection)", "info"); return; }
            var payload = JSON.stringify(renames);
            cs.evalScript("ocBatchRenameProjectItems('" + payload.replace(/\\/g, "\\\\").replace(/'/g, "\\'") + "')", function (result) {
                try {
                    var r = JSON.parse(result);
                    showToast("Renamed " + (r.renamed || renames.length) + " items", "success");
                    if (!r.error) {
                        // Journal each (nodeId -> oldName) pair so Unrename
                        // can restore the previous names.
                        var reverseList = [];
                        for (var k = 0; k < renames.length; k++) {
                            var idx = -1;
                            for (var j = 0; j < renameItemsData.length; j++) {
                                if ((renameItemsData[j].nodeId || renameItemsData[j].id ||
                                     renameItemsData[j].path) === renames[k].nodeId) {
                                    idx = j; break;
                                }
                            }
                            if (idx >= 0) {
                                reverseList.push({
                                    nodeId: renames[k].nodeId,
                                    oldName: renameItemsData[idx].name,
                                    currentName: renames[k].newName
                                });
                            }
                        }
                        journalRecord(
                            "batch_rename",
                            _journalLabelForRename(reverseList.length),
                            { renames: reverseList }
                        );
                    }
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
            updateButtons();
            return;
        }
        var html = "";
        for (var i = 0; i < smartBinRules.length; i++) {
            var r = smartBinRules[i];
            html += '<div class="smart-bin-rule" data-idx="' + i + '">'
                + '<div class="smart-bin-rule-main">'
                + '<input type="text" class="text-input bin-name" data-idx="' + i + '" placeholder="Bin name" value="' + esc(r.bin_name) + '">'
                + '<button type="button" class="bin-rule-remove" data-idx="' + i + '">Remove</button>'
                + '</div>'
                + '<div class="smart-bin-rule-fields">'
                + '<select class="bin-rule-type" data-idx="' + i + '">'
                + ['contains','starts_with','ends_with','type_is','duration_gt','duration_lt'].map(function(v) {
                    return '<option value="' + v + '"' + (r.rule_type === v ? ' selected' : '') + '>' + v + '</option>';
                }).join('')
                + '</select>'
                + '<select class="bin-field" data-idx="' + i + '">'
                + ['name','type','duration'].map(function(v) {
                    return '<option value="' + v + '"' + (r.field === v ? ' selected' : '') + '>' + v + '</option>';
                }).join('')
                + '</select>'
                + '<input type="text" class="text-input bin-value" data-idx="' + i + '" placeholder="Value" value="' + esc(r.value) + '">'
                + '</div>'
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
        updateButtons();
    }

    function createSmartBins() {
        if (!smartBinRules.length) { showAlert("Add at least one rule."); return; }
        api("POST", "/timeline/smart-bins", { rules: smartBinRules }, function (err, data) {
            if (err || (data && data.error)) { showAlert("Validation failed: " + (data ? data.error : "Network error")); return; }
            if (!inPremiere) { showToast("Rules validated (no Premiere connection)", "info"); return; }
            var jsxRules = smartBinRules.map(function(r) { return { binName: r.bin_name, rule: r.rule_type, field: r.field, value: r.value }; });
            var payload = JSON.stringify(jsxRules);
            cs.evalScript("ocCreateSmartBins('" + payload.replace(/\\/g, "\\\\").replace(/'/g, "\\'") + "')", function (result) {
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
        repeatCutsData = r.repeats || r.cuts || r.ranges || [];
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
                html += '<div class="analysis-item">'
                    + '<div class="analysis-item-main">'
                    + '<span class="analysis-item-title">' + fmtDur(c.start || 0) + " - " + fmtDur(c.end || 0) + '</span>'
                    + (c.text ? '<span class="analysis-item-copy">' + esc(c.text.substring(0, 60)) + '</span>' : '')
                    + '</div>'
                    + '<span class="analysis-item-badge">Repeat</span>'
                    + '</div>';
            }
            list.innerHTML = html || '<div class="hint">No repeats found.</div>';
        }
        // Update writeback status
        var tlStatus = document.getElementById("tlWritebackStatus");
        if (tlStatus) tlStatus.textContent = repeatCutsData.length + " repeat cuts ready to review.";
        // Phase 3.3: Show cut review panel
        showCutReview(repeatCutsData, function (selectedCuts) {
            lastTimelineCuts = selectedCuts;
            applySequenceCuts(selectedCuts);
        });
    });

    function applyRepeatCutsToTimeline() {
        if (!repeatCutsData || !repeatCutsData.length) { showAlert("No repeat cuts available."); return; }
        // Phase 3.3: Show review panel instead of direct apply
        showCutReview(repeatCutsData, function (selectedCuts) {
            applySequenceCuts(selectedCuts);
        });
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
            api_key: apiKey,
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
        var markers = chaptersData.map(function(c) { return { time: c.seconds || c.start || c.time || 0, name: c.title || c.label || "Chapter", type: "Chapter" }; });
        var payload = JSON.stringify(markers);
        cs.evalScript("ocAddSequenceMarkers('" + payload.replace(/\\/g, "\\\\").replace(/'/g, "\\'") + "')", function (result) {
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
            cs.evalScript("ocAddNativeCaptionTrack('" + payload.replace(/\\/g, "\\\\").replace(/'/g, "\\'") + "')", function (result) {
                try {
                    var r = JSON.parse(result);
                    showToast("Imported " + (r.imported || segments.length) + " captions", "success");
                } catch (e) { showAlert("Error: " + (result || e.message)); }
            });
            var statusEl = document.getElementById("srtImportStatus");
            setHintState(statusEl, "Imported " + segments.length + " caption segments.", "success");
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
            files: paths,
            target_lufs: parseFloat(document.getElementById("loudMatchTarget").value || "-14"),
            output_dir: outDir,
            no_input: true,
        });
    }

    addJobDoneListener(function (job) {
        if (job.type !== "loudness-match" || job.status !== "complete" || !job.result) return;
        var r = job.result;
        var res = document.getElementById("loudMatchResults");
        var table = document.getElementById("loudMatchTable");
        if (res) res.classList.remove("hidden");
        var outputs = r.outputs || r.clips || [];
        if (table && outputs.length) {
            var html = '<div class="report-table">';
            html += '<div class="report-table-row report-table-head">'
                + '<span>Clip</span><span>Original LUFS</span><span>Status</span></div>';
            for (var i = 0; i < outputs.length; i++) {
                var c = outputs[i];
                var name = (c.input || c.path || c.name || "").split(/[/\\]/).pop();
                html += '<div class="report-table-row">'
                    + '<span class="report-table-cell report-table-primary">' + esc(name) + '</span>'
                    + '<span class="report-table-cell">' + safeFixed(c.original_lufs, 1) + '</span>'
                    + '<span class="report-table-cell"><span class="report-status ' + ((c.job_ok || c.success) ? 'is-success' : 'is-error') + '">' + ((c.job_ok || c.success) ? "OK" : "Failed") + '</span></span>'
                    + '</div>';
            }
            html += '</div>';
            table.innerHTML = html;
        } else if (table) {
            table.innerHTML = '<div class="hint">No loudness results available.</div>';
        }
    });

    // ================================================================
    // v1.5.0 — Export Tab: Deliverables
    // ================================================================

    function loadSeqInfo() {
        if (!inPremiere) { showAlert("Premiere Pro connection required."); return; }
        // Return cached sequence info if still fresh
        if (_pproCache.seq && typeof _pproCache.seq === "object" && !_pproCache.seq.error && (Date.now() - _pproCache.seqTs < _pproCache.ttl)) {
            var cached = _pproCache.seq;
            sequenceInfo = cached;
            var statusEl = document.getElementById("seqInfoStatus");
            if (statusEl) {
                statusEl.textContent = "Loaded: " + (cached.name || "Unknown") + " — " + (cached.clip_count || 0) + " clips (cached)";
                statusEl.classList.remove("hidden");
            }
            updateButtons();
            return;
        }
        cs.evalScript('ocGetSequenceInfo()', function (result) {
            var statusEl = document.getElementById("seqInfoStatus");
            try {
                sequenceInfo = JSON.parse(result);
                if (!sequenceInfo || typeof sequenceInfo !== "object" || sequenceInfo.error) {
                    throw new Error((sequenceInfo && sequenceInfo.error) || "Unexpected Premiere response");
                }
                _pproCache.seq = sequenceInfo;
                _pproCache.seqTs = Date.now();
                if (statusEl) {
                    statusEl.textContent = "Loaded: " + (sequenceInfo.name || "Unknown") + " — " + (sequenceInfo.clip_count || 0) + " clips";
                    statusEl.classList.remove("hidden");
                }
                updateButtons();
                showToast("Sequence info loaded", "success");
            } catch (e) {
                sequenceInfo = null;
                _pproCache.seq = null;
                _pproCache.seqTs = 0;
                if (statusEl) {
                    statusEl.textContent = "Couldn't load the active sequence.";
                    statusEl.classList.remove("hidden");
                }
                updateButtons();
                showAlert("Error loading sequence info: " + (result || e.message));
            }
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
            if (fp) fp.textContent = data.output || data.output_path || "File generated.";
            showToast(type + " generated", "success");
        });
    }

    // ================================================================
    // v1.5.0 — NLP Tab Functions
    // ================================================================

    function renderSearchIndexStats(stats) {
        var statsEl = document.getElementById("searchIndexStats");
        footageIndex = stats || {};
        if (!statsEl) return;
        var totalFiles = Number((stats && stats.total_files) || 0);
        var totalSegments = Number((stats && stats.total_segments) || 0);
        if (!totalFiles) {
            statsEl.textContent = "Search index is empty. Index project clips to enable footage search.";
            return;
        }
        statsEl.textContent = "Index: " + totalFiles + " file" + (totalFiles === 1 ? "" : "s") + ", " + totalSegments + " segment" + (totalSegments === 1 ? "" : "s") + ".";
    }

    function refreshSearchIndexStatus() {
        api("GET", "/timeline/index-status", null, function (err, data) {
            if (err || !data) return;
            renderSearchIndexStats(data);
        });
    }

    function indexAllClips() {
        var paths = projectMedia.map(function(m) { return m.path || m; }).filter(Boolean);
        if (!paths.length) { showAlert("No project media found."); return; }
        var btn = document.getElementById("indexAllClipsBtn");
        if (currentJob || jobStarting) {
            showAlert("Another task is in progress. You can cancel it from the processing bar above.");
            return;
        }
        var originalBtnText = rememberButtonText(btn);
        if (btn) { btn.disabled = true; setButtonText(btn, "Indexing…"); }
        startJob("/search/index", { files: paths, no_input: true }, {
            onComplete: function (result) {
                refreshSearchIndexStatus();
                var indexed = Number((result && result.indexed) || 0);
                var total = Number((result && result.total) || paths.length);
                var errors = (result && result.errors) || [];
                showToast("Indexed " + indexed + " of " + total + " project clips" + (errors.length ? " with " + errors.length + " issue" + (errors.length === 1 ? "" : "s") : "") + ".", errors.length ? "warning" : "success");
            },
            onFinally: function () {
                if (btn) {
                    btn.disabled = false;
                    setButtonText(btn, originalBtnText);
                }
            },
            onStartError: function () {
                if (btn) {
                    btn.disabled = false;
                    setButtonText(btn, originalBtnText);
                }
            }
        });
    }

    function runFootageSearch() {
        var query = (document.getElementById("footageSearchQuery") || {}).value || "";
        if (!query) { showAlert("Enter a search query."); return; }
        var maxResults = parseInt((document.getElementById("footageSearchMax") || {}).value || "10");
        api("POST", "/search/footage", { query: query, top_k: maxResults }, function (err, data) {
            var res = document.getElementById("footageSearchResults");
            if (!res) return;
            if (err || !data) { setHintContent(res, "Search failed."); return; }
            var results = data.results || [];
            if (!results.length) { setHintContent(res, "No results found."); return; }
            var frag = document.createDocumentFragment();
            for (var i = 0; i < results.length; i++) {
                var r = results[i];
                var path = r.path || "";
                var name = path ? path.split(/[/\\]/).pop() : "Clip unavailable";
                var timeRange = r.start != null ? fmtDur(r.start) + " - " + fmtDur(r.end || r.start) : "";
                var item = document.createElement("button");
                item.type = "button";
                item.className = "footage-result-item";
                item.dataset.path = path;
                item.disabled = !path;
                if (!path) item.classList.add("is-disabled");
                item.title = path ? "Select clip" : "Clip path unavailable";
                item.setAttribute("aria-label", path ? "Select clip " + name : "Unavailable clip result");
                var title = document.createElement("div");
                title.className = "footage-result-title";
                title.textContent = name;
                if (timeRange) {
                    title.appendChild(document.createTextNode(" - "));
                    var range = document.createElement("span");
                    range.className = "footage-result-range";
                    range.textContent = timeRange;
                    title.appendChild(range);
                }
                item.appendChild(title);
                if (r.text) {
                    var snippet = document.createElement("div");
                    snippet.className = "footage-result-snippet";
        snippet.textContent = r.text.substring(0, 80) + (r.text.length > 80 ? "…" : "");
                    item.appendChild(snippet);
                }
                if (timeRange || typeof r.score === "number") {
                    var meta = document.createElement("div");
                    meta.className = "footage-result-meta";
                    if (timeRange) {
                        var metaRange = document.createElement("span");
                        metaRange.className = "footage-result-range";
                        metaRange.textContent = timeRange;
                        meta.appendChild(metaRange);
                    }
                    if (typeof r.score === "number") {
                        var score = document.createElement("span");
                        score.className = "footage-result-score";
                        score.textContent = "Score " + safeFixed(r.score, 2);
                        meta.appendChild(score);
                    }
                    item.appendChild(meta);
                }
                frag.appendChild(item);
            }
            res.innerHTML = "";
            res.appendChild(frag);
        });
    }

    // Event delegation for footage search results (avoids listener per-item)
    var _footageDelegationAdded = false;
    function ensureFootageDelegation() {
        if (_footageDelegationAdded) return;
        var res = document.getElementById("footageSearchResults");
        if (!res) return;
        _footageDelegationAdded = true;
        res.addEventListener("click", function (e) {
            var item = e.target.closest(".footage-result-item");
            if (!item) return;
            var p = item.dataset.path || "";
            if (p) selectFile(p, p.split(/[/\\]/).pop());
        });
    }

    function runNlpCommand() {
        var text = (document.getElementById("nlpCommandText") || {}).value || "";
        if (!text) { showAlert("Enter a command."); return; }
        var provider = (document.getElementById("nlpLlmProvider") || {}).value || "ollama";
        var btn = document.getElementById("runNlpCommandBtn");
        var originalBtnText = rememberButtonText(btn);
        // Capture state at command time so async callback uses the correct clip
        var snapPath = selectedPath;
        var snapFolder = projectFolder;
        if (btn) { btn.disabled = true; setButtonText(btn, "Processing…"); }
        api("POST", "/nlp/command", { command: text, filepath: snapPath, llm_provider: provider }, function (err, data) {
            if (btn) { btn.disabled = false; setButtonText(btn, originalBtnText); }
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
            // Auto-execute matched route if high confidence — uses snapshot from command time
            if (data.route && data.confidence > 0.6 && data.params) {
                startJob(data.route, Object.assign({ filepath: snapPath, output_dir: snapFolder }, data.params));
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
            // Phase 3.3: Show review panel instead of direct apply
            showCutReview(lastTimelineCuts, function (selectedCuts) {
                lastTimelineCuts = selectedCuts;
                applySequenceCuts(selectedCuts);
            });
        });

        // OTIO export
        var otioBtn = document.getElementById("exportOtioBtn");
        if (otioBtn) otioBtn.addEventListener("click", function () {
            if (!selectedPath) { showAlert("Select a clip first."); return; }
            var mode = (document.getElementById("otioExportMode") || {}).value || "cuts";
            var payload = { filepath: selectedPath, output_dir: projectFolder, mode: mode };
            if (mode === "cuts") {
                if (!lastTimelineCuts || !lastTimelineCuts.length) {
                    showAlert("No cuts available. Run Silence Removal or Repeat Detection first.");
                    return;
                }
                payload.cuts = lastTimelineCuts;
            } else if (mode === "markers") {
                if (beatMarkerTimes && beatMarkerTimes.length) {
                    payload.markers = beatMarkerTimes.map(function(t) { return { time: t, name: "Beat" }; });
                } else if (chaptersData && chaptersData.length) {
                    payload.markers = chaptersData.map(function(c) { return { time: c.seconds || c.start || c.time || 0, name: c.title || "Chapter" }; });
                } else {
                    showAlert("No markers available. Run Beat Detection or Chapter Generation first.");
                    return;
                }
            }
            otioBtn.disabled = true;
            var originalOtioText = rememberButtonText(otioBtn);
            setButtonText(otioBtn, "Exporting…");
            api("POST", "/timeline/export-otio", payload, function (err, data) {
                otioBtn.disabled = false;
                setButtonText(otioBtn, originalOtioText);
                if (err || !data || data.error) {
                    var otioRes = document.getElementById("otioResult");
                    setHintState(otioRes, "Error: " + ((data && data.error) || (err && err.message) || "Unknown"), "error");
                    return;
                }
                showToast("OTIO exported: " + (data.output_path || "").split(/[/\\]/).pop(), "success");
                var otioRes = document.getElementById("otioResult");
                setHintState(otioRes, "Saved: " + (data.output_path || ""), "success");
            });
        });
        // OTIO install button
        var installOtioBtn = document.getElementById("installOtioBtn");
        if (installOtioBtn) installOtioBtn.addEventListener("click", function () {
            startInlineInstallJob({
                endpoint: "/timeline/otio/install",
                hintEl: el.otioHint || document.getElementById("otioHint"),
                actionBtn: installOtioBtn,
                startMessage: "Installing OpenTimelineIO… This may take a minute.",
                onSuccess: function() {
                    capabilities.otio = true;
                    updateButtons();
                    showToast("OpenTimelineIO installed successfully", "success");
                }
            });
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
                cs.evalScript('openFolderInFinder("' + escPath(fp.textContent) + '")', function() {});
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
        ensureFootageDelegation();
        refreshSearchIndexStatus();
    }

    // Hook silence result to update lastTimelineCuts and show review panel
    addJobDoneListener(function (job) {
        if (job.type === "silence" && job.status === "complete" && job.result && job.result.cuts) {
            lastTimelineCuts = job.result.cuts;
            var tlStatus = document.getElementById("tlWritebackStatus");
            if (tlStatus) tlStatus.textContent = job.result.cuts.length + " silence cuts ready to review.";
            // Phase 3.3: Show cut review panel instead of auto-applying
            showCutReview(job.result.cuts, function (selectedCuts) {
                lastTimelineCuts = selectedCuts;
                applySequenceCuts(selectedCuts);
            });
        }
    });

    // Phase 3.3: Hook filler detection — show review panel when cuts data is available
    addJobDoneListener(function (job) {
        if (job.type === "fillers" && job.status === "complete" && job.result && job.result.cuts) {
            lastTimelineCuts = job.result.cuts;
            showCutReview(job.result.cuts, function (selectedCuts) {
                lastTimelineCuts = selectedCuts;
                applySequenceCuts(selectedCuts);
            });
        }
    });

    // Phase 3.3: Hook auto-edit — show review panel when cuts data is available
    addJobDoneListener(function (job) {
        if (job.type === "auto-edit" && job.status === "complete" && job.result && job.result.cuts) {
            lastTimelineCuts = job.result.cuts;
            showCutReview(job.result.cuts, function (selectedCuts) {
                lastTimelineCuts = selectedCuts;
                applySequenceCuts(selectedCuts);
            });
        }
    });

    // Phase 3.3: Hook highlights — show review panel when cuts/ranges are available
    addJobDoneListener(function (job) {
        if (job.type === "highlights" && job.status === "complete" && job.result) {
            var cuts = job.result.cuts || job.result.highlights || job.result.ranges;
            if (cuts && cuts.length) {
                lastTimelineCuts = cuts;
                showCutReview(cuts, function (selectedCuts) {
                    lastTimelineCuts = selectedCuts;
                    applySequenceCuts(selectedCuts);
                });
            }
        }
    });

    // Workflow completion handler — show step-by-step summary
    addJobDoneListener(function (job) {
        if (job.type !== "workflow") return;
        if (job.status === "complete" && job.result) {
            var r = job.result;
            var msg = "Workflow complete: " + (r.steps_completed || 0) + " steps processed.";
            if (r.output) msg += " Output: " + String(r.output).split("/").pop().split("\\").pop();
            showToast(msg, "success");
        }
    });

    // Multimodal diarization results
    addJobDoneListener(function (job) {
        if (job.type !== "multimodal-diarize" || job.status !== "complete" || !job.result) return;
        _onMmDiarizeComplete(job.result);
    });

    // B-roll generation results — auto-import generated clip
    addJobDoneListener(function (job) {
        if (job.type !== "broll-generate" || job.status !== "complete" || !job.result) return;
        var path = job.result.output_path;
        if (path) {
            showToast("B-roll generated: " + path.split("/").pop().split("\\").pop(), "success");
        }
    });

    // Social upload results
    addJobDoneListener(function (job) {
        if (job.type !== "social-upload" || job.status !== "complete" || !job.result) return;
        var r = job.result;
        var resEl = document.getElementById("socialResult");
        if (resEl) resEl.classList.remove("hidden");
        var urlEl = document.getElementById("socialResultUrl");
        if (urlEl && r.url) {
            urlEl.href = r.url;
            urlEl.textContent = "View on " + r.platform;
        }
        showToast("Uploaded to " + r.platform + "!", "success");
    });

    // ================================================================
    // Context Awareness (Phase 3.2)
    // ================================================================
    var _lastContextResult = null;

    function analyzeClipContext(infoData, pathSnapshot) {
        if (!connected) return;
        var payload = {
            has_audio: !!infoData.audio,
            has_video: !!infoData.video,
            duration: infoData.duration || 0,
            width: infoData.video ? infoData.video.width : 0,
            height: infoData.video ? infoData.video.height : 0,
            frame_rate: infoData.video ? infoData.video.fps : 0,
            num_audio_channels: infoData.audio ? (infoData.audio.channels || 2) : 0
        };
        api("POST", "/context/analyze", payload, function (err, data) {
            if (err || !data || data.error) return;
            if (pathSnapshot && pathSnapshot !== selectedPath) return;
            _lastContextResult = data;
            showContextGuidance(data.guidance, data.tab_scores);
            highlightSuggestedTabs(data.features);
        });
    }

    function showContextGuidance(guidance, tabScores) {
        var banner = el.contextGuidanceBanner;
        var textEl = el.contextGuidanceText;
        if (!banner || !textEl) return;
        if (!guidance) {
            banner.classList.add("hidden");
            return;
        }
        textEl.textContent = guidance;
        banner.classList.remove("hidden");
    }

    // Stores original sub-tab order per container so we can restore it
    var _originalTabOrders = {};

    function highlightSuggestedTabs(features) {
        // Remove previous highlights
        var allSubs = document.querySelectorAll(".sub-tab.context-suggested");
        for (var i = 0; i < allSubs.length; i++) {
            allSubs[i].classList.remove("context-suggested");
        }
        if (!features) return;

        // Highlight sub-tabs for top-scoring features (score >= 65)
        for (var i = 0; i < features.length && i < 10; i++) {
            var f = features[i];
            if (f.score < 65) break;
            var subTab = document.querySelector('.sub-tab[data-sub="' + f.id + '"]');
            if (subTab) subTab.classList.add("context-suggested");
        }

        // Reorder sub-tabs within each panel based on relevance scores
        _reorderSubTabs(features);
    }

    function _reorderSubTabs(features) {
        var tabPanelMap = {
            cut: "#panel-cut .sub-tabs",
            captions: "#panel-captions .sub-tabs",
            audio: "#panel-audio .sub-tabs",
            video: "#panel-video .sub-tabs"
        };

        // Group features by tab, keyed by feature id for quick lookup
        var scoreMap = {};
        for (var i = 0; i < features.length; i++) {
            scoreMap[features[i].id] = features[i].score;
        }

        // Group features by tab field
        var tabGroups = {};
        for (var i = 0; i < features.length; i++) {
            var tab = features[i].tab;
            if (!tabGroups[tab]) tabGroups[tab] = [];
            tabGroups[tab].push(features[i]);
        }

        // Process each tab panel
        for (var tabKey in tabPanelMap) {
            if (!tabPanelMap.hasOwnProperty(tabKey)) continue;
            var container = document.querySelector(tabPanelMap[tabKey]);
            if (!container) continue;
            var buttons = container.querySelectorAll(".sub-tab");
            if (!buttons.length) continue;

            // Save original order on first call (keyed by selector)
            var selectorKey = tabPanelMap[tabKey];
            if (!_originalTabOrders[selectorKey]) {
                _originalTabOrders[selectorKey] = [];
                for (var j = 0; j < buttons.length; j++) {
                    _originalTabOrders[selectorKey].push(buttons[j]);
                }
            }

            // Build an array of {element, score, originalIndex} for sorting
            var items = [];
            var origOrder = _originalTabOrders[selectorKey];
            for (var j = 0; j < buttons.length; j++) {
                var btn = buttons[j];
                var subId = btn.getAttribute("data-sub");
                var score = (subId && scoreMap[subId] !== undefined) ? scoreMap[subId] : -1;
                // Find original index for stable fallback sort
                var origIdx = 0;
                for (var k = 0; k < origOrder.length; k++) {
                    if (origOrder[k] === btn) { origIdx = k; break; }
                }
                items.push({ el: btn, score: score, origIdx: origIdx });
            }

            // Sort: scored items first (descending by score), then unscored in original order
            items.sort(function (a, b) {
                if (a.score >= 0 && b.score >= 0) return b.score - a.score || a.origIdx - b.origIdx;
                if (a.score >= 0) return -1;
                if (b.score >= 0) return 1;
                return a.origIdx - b.origIdx;
            });

            // Reappend in new order (DOM reorder)
            for (var j = 0; j < items.length; j++) {
                container.appendChild(items[j].el);
            }
        }
    }

    function resetTabOrder() {
        // Remove all context-suggested highlights
        var allSubs = document.querySelectorAll(".sub-tab.context-suggested");
        for (var i = 0; i < allSubs.length; i++) {
            allSubs[i].classList.remove("context-suggested");
        }
        // Restore original tab order for each saved container
        for (var selectorKey in _originalTabOrders) {
            if (!_originalTabOrders.hasOwnProperty(selectorKey)) continue;
            var container = document.querySelector(selectorKey);
            if (!container) continue;
            var origOrder = _originalTabOrders[selectorKey];
            for (var i = 0; i < origOrder.length; i++) {
                container.appendChild(origOrder[i]);
            }
        }
    }

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
        initCutReviewPanel(); // Phase 3.3: Cut review panel

        // Safe event listener binding — skips silently if element is null.
        // Prevents a single missing HTML element from crashing the entire panel on load.
        function _on(elemId, event, handler) {
            var elem = typeof elemId === "string" ? el[elemId] : elemId;
            if (elem) elem.addEventListener(event, handler);
        }

        // Event listeners - Clip selection
        _on("contextGuidanceDismiss", "click", function () {
            if (el.contextGuidanceBanner) el.contextGuidanceBanner.classList.add("hidden");
        });
        _on("refreshAllBtn", "click", refreshAll);
        _on("clipSelect", "change", function () {
            var opt = this.selectedIndex >= 0 ? this.options[this.selectedIndex] : null;
            if (opt && opt.value) selectFile(opt.value, opt.getAttribute("data-name") || opt.value.split(/[/\\]/).pop());
        });
        _on("refreshClipsBtn", "click", scanProjectMedia);
        _on("useSelectionBtn", "click", useTimelineSelection);
        _on("browseFileBtn", "click", browseForFile);
        _on("stageChooseMediaBtn", "click", function () { focusClipPicker(true); });
        _on("stageUseTimelineBtn", "click", function () {
            if (el.useSelectionBtn) el.useSelectionBtn.click();
        });
        _on("stageBrowseMediaBtn", "click", function () {
            if (el.browseFileBtn) el.browseFileBtn.click();
        });
        _on("stageCommandPaletteBtn", "click", openCommandPalette);

        // Cut tab buttons
        _on("runSilenceBtn", "click", runSilence);
        _on("runFillersBtn", "click", runFillers);
        _on("runFullBtn", "click", runFull);
        _on("fillerBackend", "change", updateButtons);

        // Captions tab buttons
        _on("runStyledCaptionsBtn", "click", runStyledCaptions);
        _on("runSubtitleBtn", "click", runSubtitle);
        _on("runTranscriptBtn", "click", runTranscript);
        _on("exportTranscriptBtn", "click", exportEditedTranscript);
        _on("transcriptUndoBtn", "click", undoTranscript);
        _on("transcriptRedoBtn", "click", redoTranscript);
        _on("installWhisperBtn", "click", installWhisper);
        _on("captionStyle", "change", updateStylePreview);

        // Audio tab buttons
        _on("runSeparateBtn", "click", runSeparate);
        _on("installDemucsBtn", "click", installDemucs);
        _on("runDenoiseBtn", "click", runDenoise);
        _on("measureLoudnessBtn", "click", measureLoudness);
        _on("runNormalizeBtn", "click", runNormalize);
        _on("runBeatsBtn", "click", runBeats);
        _on("runEffectBtn", "click", runEffect);

        // Video tab buttons
        _on("runWatermarkBtn", "click", runWatermark);
        _on("autoDetectWatermarkBtn", "click", autoDetectWatermark);
        _on("installWatermarkBtn", "click", installWatermark);
        _on("runScenesBtn", "click", runScenes);
        _on("copyChaptersBtn", "click", function () {
            var text = el.ytChaptersText ? el.ytChaptersText.value : "";
            if (navigator.clipboard) {
                navigator.clipboard.writeText(text).then(function () { showAlert("Copied to clipboard!"); }).catch(function () { showAlert("Copy failed"); });
            } else if (el.ytChaptersText) {
                el.ytChaptersText.select();
                document.execCommand("copy");
                showAlert("Copied to clipboard!");
            }
        });

        // Video FX buttons
        _on("runVfxBtn", "click", runVfx);
        _on("vfxSelect", "change", showVfxParams);

        // Video AI buttons
        _on("runVidAiBtn", "click", runVidAi);
        _on("vidAiTool", "change", showVidAiParams);
        _on("installVidAiBtn", "click", installVidAi);

        // Audio Pro buttons
        _on("runProFxBtn", "click", runProFx);
        _on("proFxCategory", "change", updateProFxEffectList);
        _on("proFxEffect", "change", updateProFxParams);
        _on("installPedalboardBtn", "click", installPedalboard);
        _on("runDeepFilterBtn", "click", runDeepFilter);
        _on("installDeepFilterBtn", "click", installDeepFilter);

        // Face blur buttons
        _on("runFaceBlurBtn", "click", runFaceBlur);
        _on("installMediapipeBtn", "click", installMediapipe);

        // Style transfer button
        _on("runStyleBtn", "click", runStyleTransfer);

        // Caption translation buttons
        _on("runTranslateBtn", "click", runTranslate);
        _on("installNllbBtn", "click", installNllb);

        // Karaoke buttons
        _on("runKaraokeBtn", "click", runKaraoke);
        _on("installWhisperxBtn", "click", installWhisperx);

        // Export preset buttons
        _on("runExportPresetBtn", "click", runExportPreset);
        _on("exportPresetCategory", "change", updateExportPresetList);
        _on("exportPresetSelect", "change", updateExportPresetDesc);

        // Thumbnail button
        _on("runThumbBtn", "click", runThumbnails);

        // Batch button
        _on("runBatchBtn", "click", runBatch);
        _on("runWorkflowBtn", "click", runWorkflowPreset);

        // TTS buttons
        _on("runTtsBtn", "click", runTts);
        _on("installEdgeTtsBtn", "click", installEdgeTts);
        _on("installCrisperWhisperBtn", "click", installCrisperWhisper);
        _on("installDepthBtn", "click", installDepth);
        _on("installEmotionBtn", "click", installEmotion);
        _on("installBrollGenBtn", "click", installBrollGenerate);
        _on("installMmDiarizeBtn", "click", installMultimodalDiarize);

        // SFX buttons
        _on("runSfxBtn", "click", runSfx);
        _on("sfxType", "change", showSfxParams);

        // Burn-in button
        _on("runBurninBtn", "click", runBurnin);

        // Speed buttons
        _on("runSpeedBtn", "click", runSpeed);
        _on("speedMode", "change", showSpeedParams);

        // LUT button
        _on("runLutBtn", "click", runLut);

        // Duck button
        _on("runDuckBtn", "click", runDuck);

        // Phase 6 buttons
        _on("runChromaBtn", "click", runChroma);
        _on("chromaMode", "change", showChromaParams);
        _on("runTransBtn", "click", runTransition);
        _on("runParticlesBtn", "click", runParticles);
        _on("runTitleOverlayBtn", "click", runTitleOverlay);
        _on("runTitleCardBtn", "click", runTitleCard);
        _on("runReframeBtn", "click", runReframe);
        _on("reframePreset", "change", updateReframeUI);
        _on("reframeMode", "change", updateReframeUI);
        _on("reframeCustomW", "input", updateReframeUI);
        _on("reframeCustomH", "input", updateReframeUI);
        updateReframeUI();
        _on("runUpscaleBtn", "click", runUpscale);
        _on("runColorBtn", "click", runColor);
        _on("runRemoveBtn", "click", runRemove);
        _on("runFaceAiBtn", "click", runFaceAi);
        _on("faceAiMode", "change", showFaceAiParams);
        _on("runAnimCapBtn", "click", runAnimCap);
        _on("runMusicAiBtn", "click", runMusicAi);

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
        if (el.recentClipsBtn) el.recentClipsBtn.addEventListener("click", function () {
            toggleRecentClips({ returnFocus: false });
        });
        if (el.recentClipsBtn) el.recentClipsBtn.addEventListener("keydown", function (e) {
            if (e.key === "ArrowDown") {
                e.preventDefault();
                showRecentClips({ focus: true });
            } else if (e.key === "ArrowUp") {
                e.preventDefault();
                showRecentClips({ focus: "last" });
            } else if (e.key === "Escape" && el.recentClipsDropdown && !el.recentClipsDropdown.classList.contains("hidden")) {
                e.preventDefault();
                hideRecentClipsDropdown(true);
            }
        });
        // Close recent clips dropdown on outside click
        document.addEventListener("click", function(e) {
            if (el.recentClipsDropdown && !el.recentClipsDropdown.classList.contains("hidden") &&
                !e.target.closest("#recentClipsBtn") && !e.target.closest("#recentClipsDropdown")) {
                hideRecentClipsDropdown(false);
            }
        });
        if (el.recentClipsDropdown) el.recentClipsDropdown.addEventListener("click", function(e) {
            var clearBtn = e.target.closest(".recent-clips-clear");
            if (clearBtn) {
                clearRecentClipsHistory();
                if (!getRecentClipItems().length && el.recentClipsBtn) el.recentClipsBtn.focus();
                return;
            }
            var item = e.target.closest(".recent-clip-item");
            if (item) {
                var path = item.getAttribute("data-path");
                if (path) {
                    selectFile(path, item.getAttribute("data-name") || path.split(/[/\\]/).pop());
                    hideRecentClipsDropdown(true);
                }
            }
        });
        if (el.recentClipsDropdown) el.recentClipsDropdown.addEventListener("keydown", function (e) {
            var buttons = getRecentClipButtons();
            if (!buttons.length) return;
            var currentIndex = buttons.indexOf(document.activeElement);
            if (currentIndex === -1) currentIndex = 0;
            if (e.key === "Escape") {
                e.preventDefault();
                hideRecentClipsDropdown(true);
            } else if (e.key === "ArrowDown") {
                e.preventDefault();
                buttons[(currentIndex + 1 + buttons.length) % buttons.length].focus();
            } else if (e.key === "ArrowUp") {
                e.preventDefault();
                buttons[(currentIndex - 1 + buttons.length) % buttons.length].focus();
            } else if (e.key === "Home") {
                e.preventDefault();
                buttons[0].focus();
            } else if (e.key === "End") {
                e.preventDefault();
                buttons[buttons.length - 1].focus();
            }
        });

        // Export tab buttons
        if (el.runExpTranscriptBtn) el.runExpTranscriptBtn.addEventListener("click", runExpTranscript);

        // v1.3.0 — New feature event listeners
        if (el.silenceMode) el.silenceMode.addEventListener("change", updateSilenceModeUI);
        if (el.reframeCropPos) el.reframeCropPos.addEventListener("change", updateFaceTrackingUI);
        if (el.runAutoEditBtn) el.runAutoEditBtn.addEventListener("click", runAutoEdit);
        if (el.runHighlightsBtn) el.runHighlightsBtn.addEventListener("click", runHighlights);
        if (document.getElementById("runEmotionHighlightsBtn")) document.getElementById("runEmotionHighlightsBtn").addEventListener("click", runEmotionHighlights);

        // Depth effects
        var depthBtn = document.getElementById("runDepthBtn");
        if (depthBtn) depthBtn.addEventListener("click", runDepthEffect);
        var depthSelect = document.getElementById("depthEffect");
        if (depthSelect) depthSelect.addEventListener("change", showDepthParams);

        // B-roll analysis
        var brollBtn = document.getElementById("runBrollPlanBtn");
        if (brollBtn) brollBtn.addEventListener("click", runBrollPlan);

        // AI B-roll generation
        _on("runBrollGenBtn", "click", runBrollGenerate);

        // Multimodal diarization
        _on("runMmDiarizeBtn", "click", runMultimodalDiarize);
        _on("mmDiarizeSampleFps", "input", function() {
            var valEl = document.getElementById("mmDiarizeSampleFpsVal");
            if (valEl) valEl.textContent = safeFixed(parseFloat(this.value), 1);
        });
        _on("mmDiarizeConfidence", "input", function() {
            var valEl = document.getElementById("mmDiarizeConfidenceVal");
            if (valEl) valEl.textContent = safeFixed(parseFloat(this.value), 2);
        });

        // Social media posting
        _on("socialConnectBtn", "click", socialConnect);
        _on("socialUploadBtn", "click", socialUpload);
        _on("socialPlatform", "change", loadSocialPlatforms);

        // WebSocket bridge controls
        var wsStartBtn = document.getElementById("wsStartBtn");
        var wsStopBtn = document.getElementById("wsStopBtn");
        var wsConnectBtn = document.getElementById("wsConnectBtn");
        if (wsStartBtn) wsStartBtn.addEventListener("click", wsStartBridge);
        if (wsStopBtn) wsStopBtn.addEventListener("click", wsStopBridge);
        if (wsConnectBtn) wsConnectBtn.addEventListener("click", wsConnect);

        // Engine registry
        var refreshEnginesBtn = document.getElementById("refreshEnginesBtn");
        if (refreshEnginesBtn) refreshEnginesBtn.addEventListener("click", loadEngineRegistry);

        if (el.runEnhanceBtn) el.runEnhanceBtn.addEventListener("click", runEnhance);
        if (el.runShortsBtn) el.runShortsBtn.addEventListener("click", runShorts);
        if (el.summarizeTranscriptBtn) el.summarizeTranscriptBtn.addEventListener("click", runSummarize);
        if (el.generateLutBtn) el.generateLutBtn.addEventListener("click", runGenerateLut);
        if (el.testLLMBtn) el.testLLMBtn.addEventListener("click", testLLM);
        if (el.llmProvider) el.llmProvider.addEventListener("change", function () {
            updateLLMProviderUI();
            saveLLMSettings({ silent: true });
        });
        if (el.llmModel) el.llmModel.addEventListener("change", function () { saveLLMSettings({ silent: true }); });
        if (el.llmApiKey) el.llmApiKey.addEventListener("change", function () { saveLLMSettings({ silent: true }); });
        if (el.llmBaseUrl) el.llmBaseUrl.addEventListener("change", function () { saveLLMSettings({ silent: true }); });
        if (el.copySummaryBtn) el.copySummaryBtn.addEventListener("click", function () {
            var text = el.summaryContent ? el.summaryContent.textContent : "";
            if (navigator.clipboard) {
                navigator.clipboard.writeText(text).then(function () { showToast("Summary copied", "success"); }).catch(function () { showToast("Copy failed", "warning"); });
            } else {
                showToast("Copy not supported", "warning");
            }
        });

        // Settings tab buttons
        if (el.settingsInstallWhisperBtn) el.settingsInstallWhisperBtn.addEventListener("click", installWhisper);
        if (el.settingsReinstallWhisperBtn) el.settingsReinstallWhisperBtn.addEventListener("click", reinstallWhisper);
        if (el.settingsClearCacheBtn) el.settingsClearCacheBtn.addEventListener("click", clearWhisperCache);
        if (el.whisperCpuMode) el.whisperCpuMode.addEventListener("change", toggleCpuMode);
        if (el.restartBackendBtn) el.restartBackendBtn.addEventListener("click", restartBackend);
        if (el.openLogsBtn) el.openLogsBtn.addEventListener("click", openLogs);
        
        // Settings persistence
        if (el.settingsAutoImport) el.settingsAutoImport.addEventListener("change", saveLocalSettings);
        if (el.settingsAutoOpen) el.settingsAutoOpen.addEventListener("change", saveLocalSettings);
        if (el.settingsShowNotifications) el.settingsShowNotifications.addEventListener("change", saveLocalSettings);
        if (el.settingsOutputDir) el.settingsOutputDir.addEventListener("change", saveLocalSettings);
        if (el.settingsDefaultModel) el.settingsDefaultModel.addEventListener("change", saveLocalSettings);
        
        // Audio / Zoom defaults card
        var saveAudioZoomBtn = $("saveAudioZoomDefaultsBtn");
        if (saveAudioZoomBtn) saveAudioZoomBtn.addEventListener("click", saveAudioZoomDefaults);
        var defaultLufsSlider = $("defaultLufs");
        var defaultLufsValEl = $("defaultLufsVal");
        if (defaultLufsSlider && defaultLufsValEl) {
            defaultLufsSlider.addEventListener("input", function () {
                defaultLufsValEl.textContent = this.value + " LUFS";
            });
        }
        var defaultZoomSlider = $("defaultZoom");
        var defaultZoomValEl = $("defaultZoomVal");
        if (defaultZoomSlider && defaultZoomValEl) {
            defaultZoomSlider.addEventListener("input", function () {
                defaultZoomValEl.innerHTML = safeFixed(this.value, 2) + "&times;";
            });
        }

        // Load saved settings
        loadLocalSettings();

        // Progress / Results
        if (el.cancelBtn) el.cancelBtn.addEventListener("click", cancelJob);
        if (el.processingCancel) el.processingCancel.addEventListener("click", cancelJob);
        if (el.newJobBtn) el.newJobBtn.addEventListener("click", function () {
            if (el.resultsSection) el.resultsSection.classList.add("hidden");
            if (el.retryJobBtn) el.retryJobBtn.classList.add("hidden");
        });
        if (el.retryJobBtn) el.retryJobBtn.addEventListener("click", function () {
            if (el.resultsSection) el.resultsSection.classList.add("hidden");
            if (el.retryJobBtn) el.retryJobBtn.classList.add("hidden");
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
        if (el.sessionContextDismiss) {
            el.sessionContextDismiss.addEventListener("click", dismissSessionContext);
        }
        if (el.alertDismiss) {
            el.alertDismiss.addEventListener("click", function () {
                el.alertBanner.classList.add("hidden");
            });
        }

        // Health check loop
        checkHealth();
        if (healthTimer) clearInterval(healthTimer);
        healthTimer = setInterval(checkHealth, HEALTH_MS);

        // Scan project media and populate recent files
        scanProjectMedia();
        populateRecentFiles();

        // Periodic soft re-scan: picks up media imported outside OpenCut
        // (e.g. user dragging files into Premiere, or Media Browser imports).
        // Shared helper so the reconnect path can restart this after
        // cleanupTimers() kills it on disconnect.
        startBackgroundPollers();

        // Re-scan when panel regains focus or becomes visible
        document.addEventListener("visibilitychange", function () {
            if (!document.hidden && (inPremiere || connected)) {
                scanProjectMedia();
            }
        });
        window.addEventListener("focus", function () {
            if (!currentJob && (inPremiere || connected)) {
                scanProjectMedia();
            }
        });

        // Load style preview data
        loadStylePreview();

        // Load pedalboard effects list
        loadPedalboardEffects();

        // Init VFX param visibility
        showVfxParams();

        // Load export presets
        loadExportPresets();

        // New features — each wrapped in try/catch so a single feature failure
        // doesn't prevent the rest of the panel from initialising
        var _featureInits = [
            initDropZone, initEnhancedDragDrop, initJobHistory,
            initKeyboardShortcuts, initPresets, initModelManagement, initGpuRecommendation,
            initQueue, initTranscriptSearch, initWaveform,
            initFavorites, initPreviewModal, initAudioPreview, initContextMenu,
            initWizard, initOutputBrowser, initBatchPicker, initDepDashboard,
            initSettingsIO, initWorkflowBuilder, loadWorkflowPresets,
            initCollapsibleCards, initI18n, initProjectTemplates,
            initJournal, initInterviewPolish
        ];
        for (var fi = 0; fi < _featureInits.length; fi++) {
            try { _featureInits[fi](); }
            catch (initErr) { console.error("[OpenCut] Feature init failed:", _featureInits[fi].name || fi, initErr); }
        }

        // Preset file export/import buttons
        var exportPresetFileBtn = document.getElementById("exportPresetFileBtn");
        var importPresetFileBtn = document.getElementById("importPresetFileBtn");
        if (exportPresetFileBtn) exportPresetFileBtn.addEventListener("click", exportPresetFile);
        if (importPresetFileBtn) importPresetFileBtn.addEventListener("click", importPresetFile);

        // v1.3.0 inits
        loadRecentClips();
        initCommandPalette();
        initSubTabFilter();
        addAudioWaveformButtons();
        renderMergeFiles();
        initNewSliderDisplays();
        loadLLMSettings();
        updateSilenceModeUI();
        updateFaceTrackingUI();

        // v1.7.2 inits — status bar
        initStatusBar();

        // Quick action buttons (one-click workflows on Cut tab)
        function _quickWorkflow(workflowName) {
            if (!selectedPath) { showAlert("Select a clip first."); return; }
            // Find the workflow by name from loaded presets
            for (var qi = 0; qi < _workflowPresets.length; qi++) {
                if (_workflowPresets[qi].name === workflowName) {
                    startJob("/workflow/run", {
                        filepath: selectedPath,
                        workflow: { steps: _workflowPresets[qi].steps },
                        output_dir: projectFolder,
                    });
                    return;
                }
            }
            showAlert("Workflow '" + workflowName + "' not found. Presets may still be loading.");
        }
        var qClean = document.getElementById("quickCleanInterview");
        var qYT = document.getElementById("quickYouTube");
        var qPod = document.getElementById("quickPodcast");
        if (qClean) qClean.addEventListener("click", function () { _quickWorkflow("Clean Interview"); });
        if (qYT) qYT.addEventListener("click", function () { _quickWorkflow("YouTube Upload"); });
        if (qPod) qPod.addEventListener("click", function () { _quickWorkflow("Podcast Polish"); });

        // Captions quick actions
        var qSub = document.getElementById("quickAutoSubtitle");
        var qTrans = document.getElementById("quickTranslate");
        if (qSub) qSub.addEventListener("click", function () {
            if (!selectedPath) { showAlert("Select a clip first."); return; }
            startJob("/transcript", { filepath: selectedPath, model: "base", export_format: "srt", output_dir: projectFolder });
        });
        if (qTrans) qTrans.addEventListener("click", function () {
            if (!selectedPath) { showAlert("Select a clip first."); return; }
            startJob("/transcript", { filepath: selectedPath, model: "base", output_dir: projectFolder });
        });

        // Audio quick actions
        var qStudio = document.getElementById("quickStudioAudio");
        var qDen = document.getElementById("quickDenoise");
        if (qStudio) qStudio.addEventListener("click", function () { _quickWorkflow("Studio Audio"); });
        if (qDen) qDen.addEventListener("click", function () {
            if (!selectedPath) { showAlert("Select a clip first."); return; }
            startJob("/audio/denoise", { filepath: selectedPath, output_dir: projectFolder });
        });

        // Video quick actions
        var qColor = document.getElementById("quickAutoColor");
        var qReframe = document.getElementById("quickSocialReframe");
        if (qColor) qColor.addEventListener("click", function () {
            if (!selectedPath) { showAlert("Select a clip first."); return; }
            startJob("/video/color/correct", { filepath: selectedPath, auto: true, output_dir: projectFolder });
        });
        if (qReframe) qReframe.addEventListener("click", function () {
            if (!selectedPath) { showAlert("Select a clip first."); return; }
            startJob("/video/reframe", { filepath: selectedPath, aspect: "9:16", method: "face", output_dir: projectFolder });
        });

        // v1.5.0 inits — deferred via lazy tab rendering (Phase 5.1)
        // These are now called by initTabOnFirstVisit() when user first visits each tab:
        // initTimelineFeatures(), initCaptionNewFeatures(), initAudioNewFeatures(),
        // initDeliverablesFeatures(), initNlpFeatures()
        // Mark the default visible tab (cut) as rendered
        _tabRendered["cut"] = true;

        // Pause CSS animations when panel is hidden (saves GPU/CPU in Premiere)
        document.addEventListener("visibilitychange", function () {
            var appEl = document.querySelector(".app");
            if (appEl) {
                if (document.hidden) appEl.classList.add("paused-animations");
                else appEl.classList.remove("paused-animations");
            }
        });

        // Auto-connect WebSocket after first successful health check
        addJobDoneListener(function () {}); // no-op, WS auto-connect handled below
        var _wsAutoConnected = false;
        var _origOnHealth = null;

        // Cleanup SSE/WS connections and timers on panel close/navigation
        window.addEventListener("beforeunload", function () {
            wsDisconnect();
            if (activeStream) {
                activeStream.close();
                activeStream = null;
            }
            cleanupTimers();
        });
    });

})();
