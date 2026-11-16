/* OpenCut visual (CLIP) footage search. Classic script + CommonJS for tests.
 *
 * Owns the Search by switch on the Footage Search card, the result list and
 * the feedback for /search/semantic and its index job. main.js keeps both job
 * routes and hands them in as startSearch and startIndex, because the panel
 * route scanners only read main.js. Clicks on a result reach main.js through
 * its existing .footage-result-item delegation, the same as transcript hits.
 */
(function (root, factory) {
    "use strict";
    var api = factory();
    if (typeof module === "object" && module.exports) module.exports = api;
    if (root) root.OpenCutVisualSearchController = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
    "use strict";

    var STATUS_ID = "searchStatus";
    var MAX_RESULTS = 50;

    function createVisualSearchController(deps) {
        var doc = deps.documentRef;
        var fmt = deps.format;
        var t = deps.t;
        var fill = deps.replaceTemplateValue;
        var setStatusLine = deps.setStatusLine;
        var renderEmpty = deps.renderSearchResultsEmpty;

        function unknownError(job) {
            return (job && (job.error || job.message)) || t("progress.unknown_error", "no details reported");
        }

        /* "visual" searches frames with CLIP, "transcript" (the default) searches spoken words. */
        function getMode() {
            var modeEl = doc.getElementById("footageSearchMode");
            return modeEl && modeEl.value === "visual" ? "visual" : "transcript";
        }

        function applyMode() {
            var visual = getMode() === "visual";
            var connected = deps.isConnected();
            var indexBtn = doc.getElementById("buildVisualIndexBtn");
            if (indexBtn) {
                indexBtn.classList.toggle("hidden", !visual);
                indexBtn.title = t("search.visual_index_title", "Index the key frames of every video in the project so visual searches start faster.");
            }
            if (visual) {
                renderEmpty(
                    t("search.visual_empty_title", "Search by what's on screen"),
                    t("search.visual_empty_body", "Describe a shot and OpenCut looks through the key frames of every video in the project. The first search indexes them, which takes a while, so you can build the visual index ahead of time."),
                    "info"
                );
                setStatusLine(
                    STATUS_ID,
                    connected
                        ? t("search.visual_mode_status", "Visual search reads the pictures, not the words. Describe what's on screen.")
                        : t("search.index_reconnect_status", "Reconnect the backend to search the footage library."),
                    connected ? "idle" : "warning"
                );
                return;
            }
            renderEmpty(
                t("search.library_title", "Search the footage library"),
                t("search.library_body", "Index project clips, then use descriptive queries to find the right sound bite or shot."),
                "info"
            );
            if (!connected) {
                setStatusLine(STATUS_ID, t("search.index_reconnect_status", "Reconnect the backend to search the footage library."), "warning");
            } else if (deps.getIndexedFileCount()) {
                setStatusLine(STATUS_ID, t("search.index_ready_status", "Library ready. Search with descriptive phrases and click a result to load it into the workspace."), "ready");
            } else {
                setStatusLine(STATUS_ID, t("search.index_empty_status", "Index project clips, then search with descriptive phrases to surface the right moment faster."), "idle");
            }
        }

        /* Adds project_dir only for a saved local project, so the portable index has a home. */
        function withProjectDir(payload) {
            var projectDir = deps.getProjectDir();
            if (projectDir) payload.project_dir = projectDir;
            return payload;
        }

        /* Shared preflight for search and indexing. Returns the clip list, or
         * null after telling the user why nothing can run. */
        function clipPathsOrWarn(reconnectMessage) {
            if (!deps.isConnected()) {
                setStatusLine(STATUS_ID, reconnectMessage, "warning");
                return null;
            }
            var clipPaths = fmt.collectVisualSearchClipPaths(deps.getProjectMedia());
            if (!clipPaths.length) {
                var noClips = t("search.visual_no_clips", "There are no video clips in the project yet. Import footage into Premiere, then try again.");
                setStatusLine(STATUS_ID, noClips, "warning");
                deps.showAlert(noClips);
                return null;
            }
            if (deps.isBusy()) {
                setStatusLine(STATUS_ID, t("search.task_in_progress_status", "Another task is already in progress. Cancel it from the processing bar before re-indexing."), "warning");
                deps.showAlert(t("search.task_in_progress_alert", "Another task is in progress. You can cancel it from the processing bar above."));
                return null;
            }
            return clipPaths;
        }

        function appendDiv(parent, className, text) {
            var node = doc.createElement("div");
            node.className = className;
            if (text !== undefined) node.textContent = text;
            parent.appendChild(node);
            return node;
        }

        function buildResultItem(card) {
            var label = card.name || t("search.clip_unavailable", "Clip unavailable");
            var scoreText = card.percent === null
                ? ""
                : fill(t("search.match_percent", "{percent}% match"), "{percent}", card.percent);
            var item = doc.createElement("button");
            item.type = "button";
            item.className = "footage-result-item";
            item.dataset.path = card.path;
            item.dataset.name = label;
            item.disabled = !card.path;
            if (!card.path) item.classList.add("is-disabled");
            if (card.path && deps.isSelectedPath(card.path)) item.classList.add("is-selected");
            item.title = card.path
                ? t("search.select_clip_title", "Select clip")
                : t("search.clip_path_unavailable_title", "Clip path unavailable");
            var ariaLabel;
            if (!card.path) {
                ariaLabel = t("search.unavailable_clip_aria", "Unavailable clip result");
            } else if (scoreText) {
                ariaLabel = fill(
                    fill(fill(t("search.visual_result_aria", "{label} at {time}, {score}"), "{label}", label), "{time}", card.timestamp),
                    "{score}", scoreText
                );
            } else {
                ariaLabel = fill(t("search.select_clip_aria", "Select clip {name}"), "{name}", label);
            }
            item.setAttribute("aria-label", ariaLabel);

            var body = item;
            if (card.thumbnail) {
                item.classList.add("has-thumb");
                var thumb = doc.createElement("img");
                thumb.className = "footage-result-thumb";
                thumb.alt = "";
                thumb.src = card.thumbnail;
                item.appendChild(thumb);
                body = appendDiv(item, "footage-result-body");
            }
            appendDiv(body, "footage-result-title", label);
            var meta = appendDiv(body, "footage-result-meta");
            var range = doc.createElement("span");
            range.className = "footage-result-range";
            range.textContent = card.timestamp;
            meta.appendChild(range);
            if (scoreText) {
                var score = doc.createElement("span");
                score.className = "footage-result-score";
                score.textContent = scoreText;
                meta.appendChild(score);
            }
            return item;
        }

        function renderResults(result) {
            var res = doc.getElementById("footageSearchResults");
            if (!res) return;
            var hits = result && Array.isArray(result.results) ? result.results : [];
            if (!hits.length) {
                renderEmpty(
                    t("search.no_matches_title", "No matches yet"),
                    t("search.visual_no_matches_body", "Nothing on screen matched closely enough. Describe the shot another way, or name what's in the frame."),
                    "warning"
                );
                setStatusLine(STATUS_ID, t("search.visual_no_matches_status", "No clips matched that description."), "warning");
                return;
            }
            var frag = doc.createDocumentFragment();
            for (var i = 0; i < hits.length; i++) {
                frag.appendChild(buildResultItem(fmt.buildVisualSearchCard(hits[i], i)));
            }
            res.textContent = "";
            res.appendChild(frag);
            setStatusLine(
                STATUS_ID,
                t("search.results_ready_status", "{count} match{plural} ready. Click a result to load it into the workspace.")
                    .replace("{count}", hits.length)
                    .replace("{plural}", hits.length === 1 ? "" : "es"),
                "ready"
            );
        }

        /* Disables a button with busy text and returns the function that puts it back. */
        function busyButton(id, busyText) {
            var btn = doc.getElementById(id);
            var originalText = deps.rememberButtonText(btn);
            if (btn) {
                btn.disabled = true;
                deps.setButtonText(btn, busyText);
            }
            return function () {
                if (!btn) return;
                btn.disabled = !deps.isConnected();
                deps.setButtonText(btn, originalText);
            };
        }

        function search(query) {
            var clipPaths = clipPathsOrWarn(
                t("search.index_reconnect_status", "Reconnect the backend to search the footage library.")
            );
            if (!clipPaths) return;
            var maxEl = doc.getElementById("footageSearchMax");
            var maxResults = parseInt((maxEl && maxEl.value) || "10", 10);
            if (!isFinite(maxResults) || maxResults < 1) maxResults = 10;
            if (maxResults > MAX_RESULTS) maxResults = MAX_RESULTS;
            var restoreButton = busyButton("runFootageSearchBtn", t("search.searching_button", "Searching…"));
            setStatusLine(STATUS_ID, t("search.visual_searching", "Searching clips by what's on screen…"), "working");
            renderEmpty(
                t("search.searching_title", "Searching footage"),
                fill(
                    t("search.visual_searching_body", "Comparing your description with the key frames of {clips}. A clip that isn't indexed yet gets indexed first, which takes a while."),
                    "{clips}", deps.formatSearchProjectClips(clipPaths.length)
                ),
                "info"
            );
            deps.startSearch(withProjectDir({
                query: query,
                clip_paths: clipPaths,
                max_results: maxResults,
                no_input: true
            }), {
                onComplete: function (result) {
                    renderResults(result);
                },
                onError: function (job) {
                    var reason = unknownError(job);
                    renderEmpty(
                        t("search.unavailable_title", "Search unavailable"),
                        fill(t("search.visual_failed_body", "The visual search didn't finish. {error}"), "{error}", reason),
                        "error"
                    );
                    setStatusLine(STATUS_ID, t("search.visual_failed_status", "Visual search failed. Check the error in the results card, then try again."), "error", reason);
                },
                onCancel: function () {
                    applyMode();
                },
                onFinally: restoreButton,
                onStartError: function () {
                    restoreButton();
                    renderEmpty(
                        t("search.unavailable_title", "Search unavailable"),
                        t("search.unavailable_body", "The footage search request failed. Check the backend connection and try again."),
                        "error"
                    );
                    setStatusLine(STATUS_ID, t("search.failed_status", "Search failed. Check the backend connection and try again."), "error");
                }
            });
        }

        function buildIndex() {
            var clipPaths = clipPathsOrWarn(
                t("search.reconnect_before_visual_index", "Reconnect the backend before building the visual index.")
            );
            if (!clipPaths) return;
            var restoreButton = busyButton("buildVisualIndexBtn", t("search.indexing_button", "Indexing…"));
            setStatusLine(
                STATUS_ID,
                fill(
                    clipPaths.length === 1
                        ? t("search.visual_indexing_status_one", "Indexing the key frames of {count} video. The first run takes a while.")
                        : t("search.visual_indexing_status_many", "Indexing the key frames of {count} videos. The first run takes a while."),
                    "{count}", clipPaths.length
                ),
                "working"
            );
            deps.startIndex(withProjectDir({ clip_paths: clipPaths, no_input: true }), {
                onComplete: function (result) {
                    var count = Number((result && result.clip_count) || clipPaths.length);
                    var message = fill(
                        count === 1
                            ? t("search.visual_index_ready_one", "Visual index ready. {count} video can be searched by what's on screen.")
                            : t("search.visual_index_ready_many", "Visual index ready. {count} videos can be searched by what's on screen."),
                        "{count}", count
                    );
                    setStatusLine(STATUS_ID, message, "success");
                    deps.showToast(message, "success");
                },
                onError: function (job) {
                    var reason = unknownError(job);
                    setStatusLine(
                        STATUS_ID,
                        fill(t("search.visual_index_failed", "Couldn't build the visual index. {error}"), "{error}", reason),
                        "error",
                        reason
                    );
                },
                onCancel: function () {
                    setStatusLine(STATUS_ID, t("search.visual_mode_status", "Visual search reads the pictures, not the words. Describe what's on screen."), "idle");
                },
                onFinally: restoreButton,
                onStartError: function () {
                    restoreButton();
                    setStatusLine(STATUS_ID, t("search.indexing_failed_status", "Couldn't start indexing right now. Check the backend connection and try again."), "error");
                }
            });
        }

        return {
            getMode: getMode,
            applyMode: applyMode,
            renderResults: renderResults,
            search: search,
            buildIndex: buildIndex
        };
    }

    return { createVisualSearchController: createVisualSearchController };
});
