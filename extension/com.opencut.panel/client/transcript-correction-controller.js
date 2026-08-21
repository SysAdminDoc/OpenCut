(function (global) {
    "use strict";

    function createTranscriptCorrectionController(options) {
        options = options || {};
        var pending = null;

        function element(name) {
            var elements = options.elements ? options.elements() : {};
            return elements[name];
        }

        function setStatus(message, state) {
            var line = element("transcriptCorrectionStatus");
            if (!line) return;
            line.textContent = message || "";
            if (state) line.setAttribute("data-state", state);
        }

        function update() {
            var segments = options.getSegments();
            var find = element("transcriptFindInput");
            var hasFind = !!(find && find.value.trim());
            var canPreview = options.isConnected() && segments.length > 0 && hasFind;
            var preview = element("transcriptPreviewReplaceBtn");
            var apply = element("transcriptApplyReplaceBtn");
            if (preview) preview.disabled = !canPreview;
            if (apply) apply.disabled = !(canPreview && pending);
        }

        function payload() {
            var find = element("transcriptFindInput");
            var replace = element("transcriptReplaceInput");
            var caseSensitive = element("transcriptCaseSensitive");
            var wholeWord = element("transcriptWholeWord");
            var glossary = element("transcriptSaveGlossary");
            return {
                project_path: options.projectPath(),
                segments: JSON.parse(JSON.stringify(options.getSegments())),
                find: find ? find.value : "",
                replace: replace ? replace.value : "",
                case_sensitive: !!(caseSensitive && caseSensitive.checked),
                whole_word: !(wholeWord && !wholeWord.checked),
                save_to_glossary: !!(glossary && glossary.checked)
            };
        }

        function render(data) {
            var line = element("transcriptCorrectionPreview");
            if (!line) return;
            var summary = data && data.summary ? data.summary : {};
            var changes = data && data.changes ? data.changes : [];
            var copy = options.t(
                "transcript.bulk_preview_summary",
                "{replacements} replacement(s) across {segments} segment(s)."
            )
                .replace("{replacements}", summary.total_replacements || 0)
                .replace("{segments}", summary.changed_segments || 0);
            for (var i = 0; i < Math.min(changes.length, 3); i++) {
                copy += " " + String(changes[i].before || "") + " → " + String(changes[i].after || "");
            }
            line.textContent = copy;
            line.classList.remove("hidden");
        }

        function previewCorrection() {
            var request = payload();
            if (!request.find || !request.segments.length) return;
            var button = element("transcriptPreviewReplaceBtn");
            if (button) button.disabled = true;
            setStatus(options.t("transcript.bulk_preview_working", "Building correction preview…"), "working");
            options.api("POST", "/transcript-edit/corrections/preview", request, function (err, data) {
                if (err || !data || data.error) {
                    pending = null;
                    setStatus(options.t("transcript.bulk_preview_failed", "Correction preview failed: {error}")
                        .replace("{error}", (data && data.error) || (err && err.message) || options.t("toast.unknown_error", "no details reported")), "error");
                    update();
                    return;
                }
                pending = { payload: request, preview: data };
                render(data);
                setStatus(options.t("transcript.bulk_preview_ready", "Preview ready. Review the affected segments, then apply or edit the rule."), "success");
                update();
            });
        }

        function applyResult(segments) {
            var data = options.getTranscriptData();
            if (!data || !Array.isArray(segments)) return;
            options.snapshot();
            data.segments = segments;
            data.full_text = segments.map(function (segment) { return segment.text || ""; }).join(" ");
            options.setLastSegments(segments);
            options.cache(segments);
            var container = element("transcriptSegments");
            if (container) {
                var textareas = container.querySelectorAll(".transcript-seg-text");
                for (var i = 0; i < textareas.length && i < segments.length; i++) {
                    textareas[i].value = segments[i].text || "";
                    options.autoResize(textareas[i]);
                }
            }
            options.snapshot();
            options.refreshSearch();
        }

        function applyCorrection() {
            if (!pending) return;
            var request = pending.payload;
            request.confirm_token = pending.preview.confirm_token;
            setStatus(options.t("transcript.bulk_apply_working", "Applying transcript correction…"), "working");
            var button = element("transcriptApplyReplaceBtn");
            if (button) button.disabled = true;
            options.api("POST", "/transcript-edit/corrections/apply", request, function (err, data) {
                if (err || !data || data.error || !data.applied) {
                    setStatus(options.t("transcript.bulk_apply_failed", "Correction was not applied: {error}")
                        .replace("{error}", (data && data.error) || (err && err.message) || options.t("toast.unknown_error", "no details reported")), "error");
                    update();
                    return;
                }
                applyResult(data.segments || []);
                pending = null;
                var line = element("transcriptCorrectionPreview");
                if (line) line.classList.add("hidden");
                setStatus(options.t("transcript.bulk_applied", "Correction applied. Use Undo in the transcript header to restore the previous wording."), "success");
                update();
            });
        }

        return {
            preview: previewCorrection,
            apply: applyCorrection,
            reset: function () { pending = null; update(); },
            update: update
        };
    }

    global.OpenCutTranscriptCorrectionController = {
        create: createTranscriptCorrectionController
    };
}(typeof window !== "undefined" ? window : this));
