/* OpenCut CEP terminal-result controller. Classic script + CommonJS for tests. */
(function (root, factory) {
    "use strict";
    var api = factory();
    if (typeof module === "object" && module.exports) module.exports = api;
    if (root) root.OpenCutResultsController = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
    "use strict";

    function createResultsController(options) {
        options = options || {};
        var documentRef = options.documentRef
            || (typeof document !== "undefined" ? document : null);
        var elements = options.elements || {};
        var translate = typeof options.translate === "function"
            ? options.translate
            : function (key, fallback) { return fallback || key; };
        var escapeHtml = typeof options.escapeHtml === "function"
            ? options.escapeHtml
            : function (value) { return String(value == null ? "" : value); };
        var safeFixed = typeof options.safeFixed === "function"
            ? options.safeFixed
            : function (value, digits) {
                var number = Number(value);
                return isFinite(number) ? number.toFixed(digits || 0) : "0";
            };
        var onBoundaryReview = typeof options.onBoundaryReview === "function"
            ? options.onBoundaryReview
            : function () {};
        var onAnnounce = typeof options.onAnnounce === "function"
            ? options.onAnnounce
            : function () {};
        var disposed = false;

        function getElement(id) {
            var node = elements[id];
            if (node) return node;
            return documentRef && typeof documentRef.getElementById === "function"
                ? documentRef.getElementById(id)
                : null;
        }

        function t(key, fallback) {
            return translate(key, fallback);
        }

        function showSuccess(job, sourcePayload) {
            if (disposed) return false;
            var result = job && job.result ? job.result : {};
            var resultsSection = getElement("resultsSection");
            var resultsTitle = getElement("resultsTitle");
            var resultsStats = getElement("resultsStats");
            var resultsPath = getElement("resultsPath");
            if (!resultsSection || !resultsTitle || !resultsStats || !resultsPath) return false;

            resultsSection.classList.remove("hidden");
            resultsTitle.textContent = t("progress.finished", "Finished");
            resultsTitle.removeAttribute("style");
            resultsTitle.setAttribute("data-state", "success");

            var stats = "";
            var r = result;

            if (r.boundary_review && r.boundary_review.required) {
                onBoundaryReview(r.boundary_review, sourcePayload);
            }

            if (r.summary) {
                stats += escapeHtml(r.summary) + "<br>";
            }
            if (r.segments !== undefined) {
                var segmentCount = Number(r.segments);
                stats += t("progress.result_segments", "{count} segment{plural}")
                    .replace("{count}", segmentCount)
                    .replace("{plural}", segmentCount === 1 ? "" : "s");
            }
            if (r.filler_stats) {
                var fillerCount = Number(r.filler_stats.removed_fillers);
                stats += " | " + t("progress.result_fillers_removed", "{count} filler{plural} removed ({seconds}s)")
                    .replace("{count}", fillerCount)
                    .replace("{plural}", fillerCount === 1 ? "" : "s")
                    .replace("{seconds}", safeFixed(r.filler_stats.total_filler_time, 1));
            }
            if (r.boundary_review && r.boundary_review.required) {
                stats += (stats ? "<br>" : "") + t(
                    "cut.boundary_review_summary",
                    "{count} boundary or alignment result needs review before OpenCut changes the timeline."
                ).replace("{count}", Number(r.boundary_review.review_hits || 0));
            }
            if (r.asr_provenance) {
                var provenance = r.asr_provenance;
                var revision = String(provenance.model_revision || "unknown");
                if (revision.length > 12) revision = revision.slice(0, 12);
                stats += (stats ? "<br>" : "") + t("progress.result_asr_provenance", "ASR: {engine} · {model} @ {revision} · {alignment} · {language}")
                    .replace("{engine}", escapeHtml(provenance.engine || "unknown"))
                    .replace("{model}", escapeHtml(provenance.model_id || "unknown"))
                    .replace("{revision}", escapeHtml(revision))
                    .replace("{alignment}", escapeHtml(provenance.alignment_mode || "none"))
                    .replace("{language}", escapeHtml(provenance.language_decision || "unknown"));
            }
            if (r.caption_segments !== undefined) {
                var captionCount = Number(r.caption_segments);
                var wordCount = Number(r.words || 0);
                stats += (stats ? " | " : "") + t("progress.result_captions_words", "{captions} caption{caption_plural}, {words} word{word_plural}")
                    .replace("{captions}", captionCount)
                    .replace("{caption_plural}", captionCount === 1 ? "" : "s")
                    .replace("{words}", wordCount)
                    .replace("{word_plural}", wordCount === 1 ? "" : "s");
            }
            if (r.style) {
                stats += " | " + t("progress.result_style", "Style: {style}")
                    .replace("{style}", escapeHtml(r.style));
            }
            if (r.effect && !r.method) {
                stats += (stats ? "<br>" : "") + t("progress.result_effect_applied", "Effect applied: {effect}")
                    .replace("{effect}", escapeHtml(r.effect));
            }
            if (r.method && r.strength !== undefined) {
                stats += (stats ? "<br>" : "") + t("progress.result_denoise", "Denoise: {method} ({strength}% strength)")
                    .replace("{method}", escapeHtml(r.method))
                    .replace("{strength}", safeFixed(r.strength * 100, 0));
            }
            if (r.preset && r.target_loudness !== undefined) {
                stats += (stats ? "<br>" : "") + t("progress.result_normalized_to", "Normalized to {target} LUFS ({preset})")
                    .replace("{target}", safeFixed(r.target_loudness, 1))
                    .replace("{preset}", escapeHtml(r.preset));
                if (r.input_loudness !== undefined) {
                    stats += " | " + t("progress.result_loudness_was", "Was: {lufs} LUFS")
                        .replace("{lufs}", safeFixed(r.input_loudness, 1));
                }
            }
            if (r.bpm) {
                var beatCount = r.total_beats != null ? Number(r.total_beats) : 0;
                stats += (stats ? "<br>" : "") + t("progress.result_bpm_beats", "BPM: {bpm} | {beats} beat{plural}")
                    .replace("{bpm}", safeFixed(r.bpm, 0))
                    .replace("{beats}", beatCount)
                    .replace("{plural}", beatCount === 1 ? "" : "s");
                if (r.confidence !== undefined) {
                    stats += " | " + t("progress.result_confidence", "Confidence: {percent}%")
                        .replace("{percent}", safeFixed(r.confidence * 100, 0));
                }
            }
            if (r.output_paths && r.output_paths.length > 0) {
                var stemNames = [];
                for (var i = 0; i < r.output_paths.length; i++) {
                    var fname = r.output_paths[i].split(/[/\\]/).pop();
                    stemNames.push(escapeHtml(fname));
                }
                stats += (stats ? "<br>" : "") + t("progress.result_stems", "{count} stem{plural}: {names}")
                    .replace("{count}", r.output_paths.length)
                    .replace("{plural}", r.output_paths.length === 1 ? "" : "s")
                    .replace("{names}", stemNames.join(", "));
            }
            if (r.magic_clips_bundle) {
                var bundleOutputs = Number(r.magic_clips_bundle.output_count || 0);
                var bundleCandidates = Number(r.magic_clips_bundle.candidate_count || 0);
                stats += (stats ? "<br>" : "") + "Magic Clips bundle: " +
                    bundleOutputs + " output" + (bundleOutputs === 1 ? "" : "s") +
                    " across " + bundleCandidates + " candidate" + (bundleCandidates === 1 ? "" : "s");
            }
            if (r.total_scenes) {
                stats += (stats ? "<br>" : "") + t("progress.result_scenes", "Scenes: {count} | Avg: {seconds}s")
                    .replace("{count}", Number(r.total_scenes))
                    .replace("{seconds}", safeFixed(r.avg_scene_length, 1));
            }
            if (r.indexed !== undefined && r.total !== undefined) {
                stats += (stats ? "<br>" : "") + t("progress.result_files_indexed", "{indexed} of {total} files indexed")
                    .replace("{indexed}", Number(r.indexed))
                    .replace("{total}", Number(r.total));
                if (r.errors && r.errors.length) {
                    stats += " | " + t("progress.result_errors", "{count} error{plural}")
                        .replace("{count}", Number(r.errors.length))
                        .replace("{plural}", r.errors.length === 1 ? "" : "s");
                }
            }

            var resultPath = r.magic_clips_bundle_manifest || r.xml_path || r.output_path || r.overlay_path || (r.output_paths
                ? t("progress.result_files_exported", "{count} file{plural} exported")
                    .replace("{count}", r.output_paths.length)
                    .replace("{plural}", r.output_paths.length === 1 ? "" : "s")
                : "");
            resultsStats.innerHTML = stats || t("progress.success_summary", "The run finished successfully.");
            resultsPath.textContent = resultPath;
            resultsPath.title = resultPath || "";
            onAnnounce("polite", t("progress.announce_finished", "Run finished. {summary}")
                .replace("{summary}", resultsStats.textContent || ""));
            return true;
        }

        function showFailure(job, message, canRetry) {
            if (disposed) return false;
            var resultsSection = getElement("resultsSection");
            var resultsTitle = getElement("resultsTitle");
            var resultsStats = getElement("resultsStats");
            var resultsPath = getElement("resultsPath");
            var retryJob = getElement("retryJobBtn");
            if (!resultsSection || !resultsTitle || !resultsStats || !resultsPath) return false;

            resultsSection.classList.remove("hidden");
            resultsTitle.textContent = t("progress.run_failed", "Run failed");
            resultsTitle.removeAttribute("style");
            resultsTitle.setAttribute("data-state", "error");
            resultsStats.textContent = message || t("progress.unknown_error", "no details reported");
            resultsPath.textContent = "";
            resultsPath.title = "";
            if (canRetry && retryJob) retryJob.classList.remove("hidden");
            return true;
        }

        function dispose() {
            disposed = true;
        }

        return {
            dispose: dispose,
            showFailure: showFailure,
            showSuccess: showSuccess
        };
    }

    return { createResultsController: createResultsController };
});
