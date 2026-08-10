/* OpenCut Studio Workbench v2
 *
 * Adds the page-specific editorial canvases used by both the CEP and UXP
 * surfaces. The canvases are deliberately static until a real project is
 * loaded; their primary actions proxy to the existing production controls so
 * the established host and backend contracts remain authoritative.
 */
(function initStudioWorkbenchV2() {
  "use strict";

  if (document.documentElement.dataset.studioWorkbench === "v2") return;
  document.documentElement.dataset.studioWorkbench = "v2";

  var isCep = Boolean(document.querySelector(".sidebar"));
  var surface = isCep ? "cep" : "uxp";

  var pageCopy = {
    cut: ["EDITING WORKSPACE", "Cut & Clean", "Turn raw footage into a confident first pass."],
    captions: ["TEXT & ACCESSIBILITY", "Captions", "Transcribe, shape, and deliver readable dialogue."],
    audio: ["SOUND WORKSPACE", "Audio", "Clean dialogue, balance levels, and shape the mix."],
    video: ["PICTURE WORKSPACE", "Video", "Refine framing, color, motion, and picture quality."],
    timeline: ["SEQUENCE OPERATIONS", "Timeline", "Inspect structure, repair timing, and deliver clean sequence changes."],
    search: ["FOOTAGE INTELLIGENCE", "Search", "Find the exact moment across transcripts, scenes, and projects."],
    export: ["OUTPUT WORKSPACE", "Export", "Package the current sequence for every destination."],
    deliverables: ["PRODUCTION REPORTING", "Deliverables", "Turn the current sequence into production-ready documents."],
    agent: ["ASSISTED EDITING", "Agent", "Plan complex edits, inspect every step, and stay in control."],
    settings: ["SYSTEM CONTROL", "Settings", "Configure the local engine, editing defaults, and integrations."],
  };

  function escapeAttr(value) {
    return String(value).replace(/&/g, "&amp;").replace(/"/g, "&quot;");
  }

  function action(label, targets, primary, icon) {
    return '<button type="button" class="studio-action' + (primary ? " is-primary" : "") +
      '" data-studio-targets="' + escapeAttr(targets || "") + '">' +
      (icon ? '<span class="studio-action-icon" aria-hidden="true">' + icon + "</span>" : "") +
      '<span>' + label + "</span></button>";
  }

  function visualIcon(name) {
    var icons = {
      play: '<svg viewBox="0 0 16 16"><path d="M5 3.2 12 8l-7 4.8z"/></svg>',
      search: '<svg viewBox="0 0 16 16"><circle cx="7" cy="7" r="4.5"/><path d="m10.5 10.5 3 3"/></svg>',
      spark: '<svg viewBox="0 0 16 16"><path d="M8 1.5 9.5 6 14 7.5 9.5 9 8 14 6.5 9 2 7.5 6.5 6z"/></svg>',
      check: '<svg viewBox="0 0 16 16"><path d="m3 8.5 3 3 7-7"/></svg>',
      folder: '<svg viewBox="0 0 16 16"><path d="M1.5 4h5l1.2 1.5h6.8v7h-13z"/></svg>',
      sliders: '<svg viewBox="0 0 16 16"><path d="M3 2v12M8 2v12M13 2v12M1.5 5h3M6.5 10h3M11.5 7h3"/></svg>',
    };
    return icons[name] || icons.spark;
  }

  function pageHeading(page) {
    var copy = pageCopy[page];
    return '<header class="studio-page-heading">' +
      '<span class="studio-eyebrow">' + copy[0] + "</span>" +
      '<h2 id="studio-title-' + page + "-" + surface + '">' + copy[1] + "</h2>" +
      "<p>" + copy[2] + "</p>" +
      "</header>";
  }

  function toggle(label, detail, enabled) {
    return '<div class="studio-toggle-row">' +
      '<div><strong>' + label + "</strong><span>" + detail + "</span></div>" +
      '<span class="studio-toggle' + (enabled ? " is-on" : "") + '" aria-hidden="true"><i></i></span>' +
      "</div>";
  }

  function waveform(tone) {
    return '<div class="studio-wave studio-wave--' + (tone || "blue") + '" aria-hidden="true">' +
      '<span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span>' +
      "</div>";
  }

  function transport() {
    return '<div class="studio-transport" aria-hidden="true"><span>◀◀</span><span>◀</span>' +
      '<span class="studio-play">' + visualIcon("play") + "</span><span>▶</span><span>▶▶</span>" +
      '<i></i><span>−</span><b></b><span>＋</span></div>';
  }

  function quickActions(items) {
    return '<div class="studio-quick-actions">' + items.map(function (item) {
      return '<div class="studio-quick-item"><span class="studio-quick-icon" aria-hidden="true">' +
        visualIcon(item[2] || "spark") + '</span><div><strong>' + item[0] + "</strong><span>" + item[1] + "</span></div></div>";
    }).join("") + "</div>";
  }

  function cutPage() {
    return '<div class="studio-grid studio-grid--editor">' +
      '<section class="studio-surface studio-timeline-surface" aria-label="Sequence preview">' +
      '<div class="studio-surface-bar"><span class="studio-source-state">No clip selected</span>' +
      action("Use active sequence", "stageUseTimelineBtn,useTimelineBtn,workspaceGuideAction", false, visualIcon("folder")) + "</div>" +
      '<div class="studio-timebar"><strong>00:01:18:12</strong><span>00:00:00</span><span>00:00:30</span><span>00:01:00</span><span>00:01:30</span><span>00:02:00</span></div>' +
      '<div class="studio-timeline" aria-hidden="true"><i class="studio-playhead"></i>' +
      '<div class="studio-track-label">V2</div><div class="studio-track"><div class="studio-clip clip-short">A001_C001.mov</div></div>' +
      '<div class="studio-track-label">V1</div><div class="studio-track"><div class="studio-clip clip-long">A001_C002.mov</div></div>' +
      '<div class="studio-track-label">A1</div><div class="studio-track studio-audio-track"><div class="studio-audio-range keep"></div><div class="studio-audio-range silence one"></div><div class="studio-audio-range keep two"></div>' + waveform("mint") + "</div></div>" +
      '<div class="studio-legend"><span class="keep">Keep</span><span class="silence">Silence</span><span class="review">Review</span></div>' + transport() + "</section>" +
      '<aside class="studio-surface studio-inspector"><h3>First-pass recipe</h3>' +
      toggle("Remove silence", "Detect and remove silent sections.", true) +
      toggle("Detect repeated takes", "Find and mark repeated segments.", true) +
      toggle("Tighten pauses", "Reduce long gaps and filler words.", true) +
      '<div class="studio-control"><div><strong>Strength</strong><span>Natural</span></div><div class="studio-slider"><i></i><b></b></div></div>' +
      '<div class="studio-estimate"><strong>Estimated result</strong><b>—</b><span>Awaiting sequence</span></div>' +
      action("Analyze sequence", "quickCleanInterview,runSilenceBtn,cutRunBtn", true, visualIcon("spark")) + "</aside></div>" +
      quickActions([["Clean interview", "Silence trim + filler cut", "spark"], ["Podcast polish", "Dialogue-first cleanup", "sliders"], ["Social cut", "Shorten for platforms", "play"]]);
  }

  function captionsPage() {
    var rows = [
      ["00:00:12:18", "HOST", "So, to start, can you tell me about your role and the team you work with?", "96%"],
      ["00:00:32:09", "GUEST", "I think the biggest challenge was aligning everyone on the vision.", "88%"],
      ["00:01:03:07", "HOST", "There were a lot of moving parts and tight deadlines.", "94%"],
      ["00:01:47:11", "GUEST", "We had to make trade-offs, but we kept the user at the center.", "92%"],
    ];
    return '<div class="studio-grid studio-grid--editor">' +
      '<section class="studio-surface studio-transcript"><div class="studio-toolbar"><span>English (US)</span><span>Balanced</span><span>Timecode mode</span>' +
      action("Transcribe sequence", "quickAutoSubtitle,transcribeBtn,captionRunBtn", true, visualIcon("sliders")) + "</div>" +
      '<div class="studio-transcript-wave">' + waveform("slate") + "</div>" +
      '<div class="studio-filterbar"><span>⌕&nbsp; Search in transcript</span><span>All speakers</span><span>All confidence</span><small>5 captions</small></div>' +
      '<div class="studio-transcript-list">' + rows.map(function (row, index) {
        return '<div class="studio-transcript-row' + (index === 1 ? " is-selected" : "") + '"><time>' + row[0] + '</time><span class="studio-speaker ' + row[1].toLowerCase() + '">' + row[1] + "</span><p>" + (index === 1 ? row[2].replace("aligning", "<mark>aligning</mark>") : row[2]) + "</p><b>" + row[3] + "</b></div>";
      }).join("") + "</div>" +
      '<div class="studio-bottom-actions"><span>Import SRT</span><span>Export captions</span><span>Review low confidence</span></div></section>' +
      '<aside class="studio-surface studio-inspector"><h3>Caption style</h3><label class="studio-label">Preset</label><div class="studio-select">Editorial clean <b>⌄</b></div>' +
      '<div class="studio-caption-preview"><span>The biggest challenge<br>was aligning everyone.</span></div>' +
      '<div class="studio-two-fields"><div><label>Font</label><span>Inter</span></div><div><label>Size</label><span>36</span></div></div>' +
      '<div class="studio-control"><div><strong>Line length</strong><span>42 / 84</span></div><div class="studio-slider"><i></i><b></b></div></div>' +
      toggle("Highlight active words", "Word-level emphasis", true) + toggle("Remove filler words", "Clean the transcript", true) +
      action("Apply captions", "applyCaptionsBtn,addCaptionTrackBtn,captionExportBtn", true, visualIcon("check")) + "</aside></div>";
  }

  function audioPage() {
    function track(name, code, tone, gain) {
      return '<div class="studio-mix-track"><div class="studio-mix-label"><strong>' + name + "</strong><span>" + code + '</span></div><div class="studio-meter"><i></i><i></i><i></i><i></i><i></i></div><div class="studio-mix-wave">' + waveform(tone) + '<b class="studio-gain-line"></b></div><span class="studio-gain">' + gain + "</span></div>";
    }
    return '<div class="studio-grid studio-grid--editor"><section class="studio-surface studio-mixer">' +
      '<div class="studio-loudness"><div><span>Integrated</span><strong>−18.4 <small>LUFS</small></strong></div><div><span>Target</span><strong>−14 <small>LUFS</small></strong></div><div><span>True peak</span><strong class="warning">−1.2 <small>dB</small></strong></div><div class="studio-loudness-chart"><span>Loudness history</span><i></i></div></div>' +
      '<div class="studio-timebar"><strong>00:01:18:12</strong><span>00:00:00</span><span>00:00:30</span><span>00:01:00</span><span>00:01:30</span></div>' +
      '<div class="studio-mix-tracks" aria-hidden="true">' + track("Dialogue", "D1", "blue", "+1.8 dB") + track("Music", "M1", "violet", "−3.2 dB") + track("Room tone", "R1", "mint", "−8.6 dB") + "</div>" + transport() + "</section>" +
      '<aside class="studio-surface studio-inspector studio-chain"><h3>Dialogue chain</h3>' +
      toggle("1  Voice isolate", "Enhance voice, reduce bleed.", true) + toggle("2  De-reverb", "Reduce room reflections.", true) +
      '<div class="studio-chain-selected"><div><strong>3&nbsp;&nbsp; EQ & clarity</strong><span>Improve tone and intelligibility.</span></div><div class="studio-eq" aria-hidden="true"><i></i><i></i><i></i><i></i><b></b></div><div class="studio-control"><div><strong>Strength</strong><span>42%</span></div><div class="studio-slider"><i></i><b></b></div></div></div>' +
      toggle("4  Loudness match", "Match target loudness.", true) + action("Process sequence", "quickStudioAudio,runDenoiseBtn,audioRunBtn", true, visualIcon("sliders")) + "</aside></div>" +
      quickActions([["Podcast polish", "Dialogue-first cleanup", "sliders"], ["Remove noise", "Reduce background noise", "spark"], ["Separate stems", "Extract individual stems", "folder"], ["Match loudness", "Align to target loudness", "sliders"]]);
  }

  function videoPage() {
    return '<div class="studio-grid studio-grid--editor"><section class="studio-surface studio-monitor">' +
      '<div class="studio-surface-bar"><span>Program&nbsp;&nbsp;⌄</span><span>Fit&nbsp;&nbsp;⌄</span><strong>00:01:18:12</strong></div>' +
      '<div class="studio-program-frame" aria-hidden="true"><div class="studio-safe-area"></div><div class="studio-subject"><i></i><b></b><span></span></div><div class="studio-before-after"></div><span class="studio-frame-label">Action Safe<br>Title Safe</span></div>' + transport() +
      '<div class="studio-bottom-actions"><span>Auto reframe</span><span>Stabilize</span><span>Match color</span><span>Upscale</span></div></section>' +
      '<aside class="studio-surface studio-inspector"><h3>Picture recipe</h3><div class="studio-tabs"><b>Framing</b><span>Color</span><span>Enhance</span></div>' +
      '<label class="studio-label">Aspect ratio</label><div class="studio-aspects"><b>16:9</b><span>9:16</span><span>1:1</span></div>' +
      '<label class="studio-label">Subject tracking</label><div class="studio-tracking-preview" aria-hidden="true"><div></div><span>● Locked</span></div>' +
      '<div class="studio-control"><div><strong>Follow strength</strong><span>64%</span></div><div class="studio-slider"><i></i><b></b></div></div>' +
      '<div class="studio-control"><div><strong>Headroom</strong><span>12%</span></div><div class="studio-slider is-short"><i></i><b></b></div></div>' +
      toggle("Smooth camera path", "Reduce abrupt movements", true) + toggle("Protect titles", "Avoid cropping text and graphics", true) +
      '<div class="studio-output-row"><strong>Output</strong><span>1080 × 1920</span></div>' + action("Analyze framing", "quickAutoColor,autoReframeBtn,videoRunBtn", true, visualIcon("spark")) + "</aside></div>";
  }

  function timelinePage() {
    function clip(name, wide) { return '<div class="studio-sequence-clip' + (wide ? " is-wide" : "") + '"><span>' + name + "</span><i></i></div>"; }
    return '<div class="studio-grid studio-grid--editor"><section><div class="studio-stats"><div><span>Duration</span><strong>04:32</strong></div><div><strong>24</strong><span>clips</span></div><div><strong>8</strong><span>markers</span></div><div><strong>2</strong><span>gaps</span></div></div>' +
      '<div class="studio-surface studio-sequence"><div class="studio-surface-bar"><strong>00:02:14:08</strong><span>Overview&nbsp;&nbsp; ▬▬▬</span></div><div class="studio-timebar"><span>00:00:00</span><span>00:01:00</span><span>00:02:00</span><span>00:03:00</span><span>00:04:00</span></div>' +
      '<div class="studio-sequence-grid" aria-hidden="true"><i class="studio-playhead"></i><b>V2</b><div>' + clip("A001_C001.mov") + clip("A002_C003.mov") + clip("A003_C005.mov", true) + '</div><b>V1</b><div>' + clip("A001_C002.mov", true) + clip("A002_C004.mov") + clip("A003_C007.mov") + '</div><b>A1</b><div class="studio-audio-sequence">' + waveform("mint") + '<em>Gap 12f</em>' + waveform("mint") + '</div><b>A2</b><div class="studio-audio-sequence">' + waveform("slate") + waveform("mint") + "</div></div>" + transport() + "</div>" +
      quickActions([["Export OTIO", "Interchange timeline", "folder"], ["Create smart bins", "Organize source media", "folder"], ["Batch rename", "Normalize clip names", "spark"], ["Markers to clips", "Build selects", "play"]]) + "</section>" +
      '<aside class="studio-surface studio-inspector studio-check"><h3>Sequence check</h3><div class="studio-check-row warning"><span>▲ Gaps</span><b>2</b></div><div class="studio-check-row warning"><span>▲ Flash frames</span><b>1</b></div><div class="studio-check-row success"><span>● Duplicate clips</span><b>0</b></div><div class="studio-check-row success"><span>● Offline media</span><b>0</b></div>' +
      '<div class="studio-issue"><strong>Gap at 00:02:14:08</strong><p>12 frames between A001_C002.wav and A002_C004.wav on A1.</p><div><span>A001_C002.wav</span><b>12f</b><span>A002_C004.wav</span></div></div>' + action("Reveal in timeline", "stageUseTimelineBtn,timelineRevealBtn", false, visualIcon("search")) + action("Repair selected", "timelineRepairBtn,timelineRunBtn", true, visualIcon("spark")) + "</aside></div>";
  }

  function searchPage() {
    var results = [
      ["A001_C002.mov", "…the biggest challenge we face is getting alignment across teams.", "95%", "00:00:12"],
      ["A001_C005.mov", "…the biggest challenge for our customers is integrating with legacy systems.", "89%", "00:00:09"],
      ["A001_C003.mov", "…so the biggest challenge right now is scale.", "86%", "00:00:10"],
      ["A001_C006.mov", "…one of the biggest challenges is proving ROI early in the process.", "78%", "00:00:11"],
    ];
    return '<div class="studio-searchbar"><span class="studio-search-icon" aria-hidden="true">' + visualIcon("search") + '</span><strong>customer explains the biggest challenge</strong><span>This project⌄</span><span>Dialogue⌄</span><span>Any speaker⌄</span>' + action("Search footage", "searchFootageBtn,searchBtn,footageSearchBtn", true, "") + "</div>" +
      '<div class="studio-search-layout"><aside class="studio-surface studio-facets"><small>24 clips indexed · Updated 2 min ago</small><h3>Speakers</h3><label>☑ Any speaker <b>24</b></label><label>□ Sarah Chen <b>9</b></label><label>□ Marcus Reed <b>7</b></label><h3>Media type</h3><label>□ Interviews <b>18</b></label><label>□ B-roll <b>4</b></label><h3>Date</h3><label>◉ All time</label><label>○ Last 7 days</label><h3>Confidence</h3><div class="studio-slider"><i></i><b></b></div></aside>' +
      '<section class="studio-surface studio-results" aria-label="Search results"><div class="studio-results-head"><span>Match</span><span>Confidence</span><span>Duration</span></div>' + results.map(function (row, index) { return '<div class="studio-result' + (index === 0 ? " is-selected" : "") + '"><div class="studio-result-thumb" aria-hidden="true">' + visualIcon("play") + '</div><div><strong>' + row[0] + "</strong><p>" + row[1].replace(/challenge(s)?/g, "<mark>challenge$1</mark>") + "</p></div><b>" + row[2] + "</b><time>" + row[3] + "</time></div>"; }).join("") + "</section>" +
      '<aside class="studio-surface studio-search-preview"><h3>A001_C002.mov</h3><div class="studio-preview-frame"><div class="studio-subject"><i></i><b></b></div></div><div class="studio-preview-transport">▶&nbsp;&nbsp; 00:00:28:14 <i></i> 00:00:40:12</div><h4>Transcript context</h4><p>We have made a lot of progress this year, but there’s still work to do.</p><p class="is-selected">… the biggest <mark>challenge</mark> we face is getting alignment across teams.</p><div class="studio-source-meta"><strong>Source</strong><span>A001_C002.mov</span><small>In / Out&nbsp;&nbsp; 00:00:28:14 — 00:00:40:12</small></div>' + action("Open in source", "searchOpenSourceBtn", false, "") + action("Insert into sequence", "searchInsertBtn,insertSearchResultBtn", true, "") + "</aside></div>";
  }

  function exportPage() {
    var settings = [["Source", "Sequence 01 · 04:32"], ["Format", "H.264"], ["Frame", "3840 × 2160 · 23.976"], ["Quality", "High · VBR 2-pass"], ["Audio", "AAC · 48 kHz · −14 LUFS"], ["Destination", "C:/Exports/OpenCut/Sequence 01/YouTube 4K"]];
    var rows = settings.map(function (row) {
      return '<div><strong>' + row[0] + '</strong><span>' + row[1] + '<\/span><b>⌄<\/b><\/div>';
    }).join("");
    return `<div class="studio-grid studio-grid--export"><section><div class="studio-presets"><b>YouTube 4K</b><span>Vertical social</span><span>Review copy</span><span>Audio master</span></div><div class="studio-surface studio-export-settings">${rows}</div><div class="studio-surface studio-export-meta"><div><strong>Estimated size</strong><i></i><span>1.8 GB</span></div><div><strong>Hardware encoder</strong><span class="success">● NVIDIA NVENC available</span></div></div></section><aside class="studio-surface studio-inspector studio-export-queue"><h3>Export queue</h3><div class="studio-queue-item"><strong>Sequence 01 — YouTube 4K</strong><span>Ready</span></div><ul><li>Range: Entire sequence</li><li>Captions: Burn in</li><li>Color: Rec.709</li><li>Review: Passed</li></ul>${action("Add to queue", "addQueueBtn,queueExportBtn", true, "")}${action("Export now", "exportBtn,runExportBtn", false, "")}</aside></div><div class="studio-surface studio-recent"><h3>Recent outputs</h3><div class="studio-table-row head"><span>Status</span><span>File name</span><span>Duration</span><span>Size</span></div><div class="studio-table-row"><span class="success">✓ Success</span><span>Sequence 01_YouTube 4K.mp4</span><span>04:32</span><span>1.82 GB</span></div><div class="studio-table-row"><span class="success">✓ Success</span><span>Sequence 01_Review Copy.mp4</span><span>04:32</span><span>512 MB</span></div></div>`;
  }

  function deliverablesPage() {
    var docs = [["VFX sheet", "Shot notes, vendor handoff, and status tracking.", "12 rows"], ["ADR list", "Dialogue pickup notes for editorial and mix.", "18 rows"], ["Music cue sheet", "Music usage context and licensing handoff.", "9 rows"], ["Asset list", "Master inventory of media, references, and outputs.", "156 rows"]];
    return '<div class="studio-grid studio-grid--deliverables"><section><div class="studio-section-line"><strong>Documents to generate</strong><span>4 of 4 enabled</span></div><div class="studio-surface studio-doc-list">' + docs.map(function (doc, index) { return '<div class="studio-doc-row' + (index === 0 ? " is-selected" : "") + '"><span class="studio-doc-icon">▦</span><div><strong>' + doc[0] + "</strong><span>" + doc[1] + "</span></div><small>" + doc[2] + '</small><span class="studio-toggle is-on"><i></i></span></div>' + (index === 0 ? '<div class="studio-field-map"><div><b>Include</b><b>Field</b><b>Source</b></div><div><span>☑</span><span>Shot</span><span>Timeline › Clip › Name</span></div><div><span>☑</span><span>Timecode in</span><span>Timeline › Clip › In</span></div><div><span>☑</span><span>Duration</span><span>Timeline › Clip › Duration</span></div><div><span>☑</span><span>Status</span><span>Metadata › VFX Status</span></div></div>' : ""); }).join("") + '</div><div class="studio-path-row"><span>CSV⌄</span><strong>C:/Documents/OpenCut/Deliverables</strong><span>Browse</span></div><div class="studio-surface studio-recent"><h3>Recent documents</h3><div class="studio-table-row head"><span>Name</span><span>Type</span><span>Generated</span><span>Status</span></div><div class="studio-table-row"><span>Interview_Edit_VFX_Sheet.csv</span><span>VFX sheet</span><span>Today 10:42</span><span class="success">Ready</span></div><div class="studio-table-row"><span>Interview_Edit_ADR_List.csv</span><span>ADR list</span><span>Today 10:42</span><span class="success">Ready</span></div></div></section>' +
      '<aside class="studio-surface studio-inspector studio-doc-preview"><h3>Document preview</h3><small>VFX sheet preview</small><div class="studio-mini-table"><div><b>Shot</b><b>Timecode In</b><b>Duration</b><b>Status</b></div><div><span>A001_C001</span><span>00:00:04:12</span><span>00:00:03:06</span><span>Approved</span></div><div><span>A002_C002</span><span>00:00:12:03</span><span>00:00:03:07</span><span>In progress</span></div><div><span>A003_C003</span><span>00:00:18:22</span><span>00:00:03:19</span><span>Approved</span></div></div><p class="success">✓ Timecode validated</p><p class="success">✓ No missing media</p><div class="studio-inspector-spacer"></div>' + action("Preview document", "previewDeliverablesBtn", false, "") + action("Generate deliverables", "generateDeliverablesBtn,generateVfxBtn", true, visualIcon("spark")) + "</aside></div>";
  }

  function agentPage() {
    var steps = [["Search customer-story moments", "Evidence: 22 candidate moments found across 4 clips.", "Ready"], ["Select strongest 75 seconds", "Evidence: Top 2 moments score 0.92 and 0.88.", "Needs review"], ["Remove pauses and repeats", "Evidence: 11 pauses and 3 repeats identified.", "Ready"], ["Reframe to 9:16", "Evidence: Auto reframe with face tracking enabled.", "Ready"], ["Add branded captions", "Evidence: Captions style “Social – Brand” will be applied.", "Ready"]];
    var plan = steps.map(function (step, index) {
      var stateClass = step[2] === "Ready" ? "success" : "warning";
      return `<div class="studio-plan-step${index === 1 ? " is-selected" : ""}"><b>${index + 1}</b><div><strong>${step[0]}</strong><span>${step[1]}</span></div><em class="${stateClass}">${step[2]}</em><i>${index === 1 ? "✓" : "□"}</i></div>`;
    }).join("");
    return `<div class="studio-grid studio-grid--editor"><section><div class="studio-surface studio-agent-composer"><p>Make a tight 60-second social cut from the strongest customer story</p><div><span>Scope&nbsp;&nbsp; Current sequence⌄</span><span>Mode&nbsp;&nbsp; Plan first⌄</span>${action("Build plan", "agentBuildPlanBtn,ocAgentSendBtn", true, "")}</div></div><div class="studio-section-line"><strong>Assistant plan</strong><span></span></div><div class="studio-plan">${plan}</div></section><aside class="studio-surface studio-inspector studio-step-review"><h3>Step review</h3><div class="studio-review-status"><strong>Step 2 of 5</strong><span class="warning">● Needs review</span></div><h4>Source</h4><div class="studio-review-source"><div class="studio-result-thumb"></div><span>A001_C002.mov<br><small>01:02:13:08 – 01:03:28:03</small></span></div><h4>Proposed range</h4><p class="accent">01:02:13:08 – 01:03:28:03 (00:01:14:19)</p><h4>Rationale</h4><p>Highest-scoring, continuous customer story moment with clear problem, solution, and outcome.</p><h4>Inputs / Outputs</h4><p>All interview clips → selected subclip and review metadata.</p><div class="studio-safety">ⓘ No timeline changes until approved.</div>${action("Preview step", "agentPreviewStepBtn", false, "")}${action("Approve & run", "agentApproveBtn,ocAgentApproveBtn", true, "")}<div class="studio-activity"><strong>Activity & evidence</strong><span>⌕ Searched customer-story moments</span><span>♢ Scored and ranked candidates</span><span>↳ Proposed best 75-second range</span></div></aside></div>`;
  }

  function settingsPage() {
    var rows = [["Theme", "Choose the application theme.", "Dark"], ["Density", "Adjust the interface density.", "Compact"], ["Start page", "Choose the page that opens at launch.", "Cut"], ["Default language", "Set the default language for OpenCut.", "English (US)"], ["Timecode", "Choose the default timecode display.", "Sequence"], ["Autosave plans", "Automatically save timeline and caption plans.", "On"]];
    return '<div class="studio-settings-layout"><nav class="studio-settings-nav" aria-label="Settings categories"><b>General</b><span>Local engine</span><span>AI models</span><span>Premiere</span><span>Storage</span><span>Integrations</span><span>Privacy</span><span>Diagnostics</span><span>About</span></nav><section class="studio-settings-detail"><h3>General</h3><h4>Appearance & behavior</h4>' + rows.map(function (row) { return '<div class="studio-setting-row"><strong>' + row[0] + "</strong><span>" + row[1] + "</span><b>" + row[2] + "⌄</b></div>"; }).join("") + '<h4>Editing defaults</h4><div class="studio-setting-row"><strong>Default transition</strong><span>Transition for new edits.</span><b>Cross Dissolve⌄</b></div><div class="studio-setting-row"><strong>Default caption style</strong><span>Style for new captions.</span><b>OpenCut Standard⌄</b></div><div class="studio-setting-row"><strong>Version</strong><span>Current installed version.</span><b>OpenCut 1.48.0</b></div><div class="studio-settings-actions">' + action("Check for updates", "checkUpdatesBtn,uxpCheckUpdatesBtn", true, "") + action("Export settings", "exportSettingsBtn,uxpExportSettingsBtn", false, "") + '</div></section><aside class="studio-surface studio-inspector studio-system-status"><h3>System status</h3><div><span>Local engine</span><b class="warning">Offline</b></div><div><span>Premiere</span><b class="success">Connected</b></div><div><span>GPU</span><b class="success">NVIDIA RTX detected</b></div><div><span>Storage</span><b>184 GB available</b></div>' + action("Run diagnostics", "refreshAllBtn,uxpDiagnosticsRefreshBtn,refreshBtn", false, "") + "</aside></div>";
  }

  var renderPage = {
    cut: cutPage,
    captions: captionsPage,
    audio: audioPage,
    video: videoPage,
    timeline: timelinePage,
    search: searchPage,
    export: exportPage,
    deliverables: deliverablesPage,
    agent: agentPage,
    settings: settingsPage,
  };

  function mountPage(page, panel) {
    if (!panel || panel.querySelector(":scope > .studio-workbench")) return;
    var content = renderPage[page]();
    panel.insertAdjacentHTML("afterbegin", '<section class="studio-workbench studio-page-' + page + '" aria-labelledby="studio-title-' + page + "-" + surface + '">' + pageHeading(page) + content + '<div class="studio-controls-divider"><span>Advanced controls & automation</span></div></section>');
  }

  if (isCep) {
    ["cut", "captions", "audio", "video", "timeline", "export", "settings"].forEach(function (page) {
      mountPage(page, document.getElementById("panel-" + page));
    });
    mountPage("search", document.getElementById("panel-nlp"));
  } else {
    ["cut", "captions", "audio", "video", "timeline", "search", "deliverables", "agent", "settings"].forEach(function (page) {
      mountPage(page, document.getElementById("tab-" + page));
    });
  }

  function pageName(raw) {
    var page = raw === "nlp" ? "search" : raw;
    return pageCopy[page] ? pageCopy[page][1] : "Cut & Clean";
  }

  function activePage() {
    var node = document.querySelector(isCep ? ".nav-tab.active" : ".oc-tab.active");
    return node ? pageName(node.getAttribute(isCep ? "data-nav" : "data-tab")) : "Cut & Clean";
  }

  function enhanceShell() {
    if (isCep) {
      var header = document.querySelector(".content-header");
      var actions = document.querySelector(".content-actions");
      var refresh = document.getElementById("refreshAllBtn");
      var brandMeta = document.querySelector(".brand-meta");
      if (brandMeta) brandMeta.textContent = "● Local";
      if (header && !header.querySelector(".studio-breadcrumb")) {
        header.insertAdjacentHTML("afterbegin", '<div class="studio-breadcrumb">Workspace <span>/</span> <strong data-studio-current-page>' + activePage() + "</strong></div>");
      }
      if (actions && !actions.querySelector(".studio-sequence-context")) {
        actions.insertAdjacentHTML("afterbegin", '<div class="studio-sequence-context"><span aria-hidden="true">▦</span> Sequence 01 <b>⌄</b></div>');
      }
      if (refresh && !refresh.querySelector(".studio-connect-label")) {
        refresh.insertAdjacentHTML("beforeend", '<span class="studio-connect-label">Connect engine</span>');
        refresh.setAttribute("aria-label", "Connect engine");
        refresh.title = "Connect engine";
      }
      var footerLabel = document.querySelector("#jobHistoryToggle .toggle-label");
      if (footerLabel) footerLabel.textContent = "Jobs";
    } else {
      var uxpHeader = document.querySelector(".oc-header");
      var uxpRight = document.querySelector(".oc-header-right");
      if (uxpHeader && !uxpHeader.querySelector(".studio-breadcrumb")) {
        var left = uxpHeader.querySelector(".oc-header-left");
        if (left) left.insertAdjacentHTML("afterend", '<div class="studio-breadcrumb">Workspace <span>/</span> <strong data-studio-current-page>' + activePage() + "</strong></div>");
      }
      if (uxpRight && !uxpRight.querySelector(".studio-sequence-context")) {
        uxpRight.insertAdjacentHTML("afterbegin", '<div class="studio-sequence-context"><span aria-hidden="true">▦</span> Sequence 01 <b>⌄</b></div>');
        uxpRight.insertAdjacentHTML("beforeend", '<button type="button" class="studio-header-connect" data-studio-targets="refreshBtn">Connect engine</button>');
      }
      var app = document.getElementById("app");
      if (app && !document.querySelector(".studio-jobbar")) {
        app.insertAdjacentHTML("beforeend", '<footer class="studio-jobbar" aria-label="Jobs"><strong>☷&nbsp;&nbsp; Jobs</strong><span>No active renders</span><span class="studio-jobbar-output">●&nbsp;&nbsp; Output ready when engine connects</span></footer>');
      }
    }
  }

  function syncBreadcrumb() {
    var current = activePage();
    document.querySelectorAll("[data-studio-current-page]").forEach(function (node) {
      node.textContent = current;
    });
  }

  enhanceShell();
  document.addEventListener("click", function (event) {
    var proxy = event.target.closest("[data-studio-targets]");
    if (proxy && proxy.classList.contains("studio-action") || proxy && proxy.classList.contains("studio-header-connect")) {
      var targetIds = (proxy.getAttribute("data-studio-targets") || "").split(",");
      var target = targetIds.map(function (id) { return document.getElementById(id.trim()); }).find(Boolean);
      if (target && target !== proxy) {
        target.click();
      } else {
        var panel = proxy.closest(isCep ? ".nav-panel" : ".oc-tab-panel");
        var fallback = panel && panel.querySelector(":scope > .studio-workbench + *");
        if (fallback) fallback.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    }
    if (event.target.closest(isCep ? ".nav-tab" : ".oc-tab")) {
      window.setTimeout(syncBreadcrumb, 0);
    }
  });
})();
