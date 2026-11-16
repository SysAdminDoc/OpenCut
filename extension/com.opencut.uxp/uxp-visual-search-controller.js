/**
 * Visual (CLIP) mode of the Footage Search card: the Search by switch, the
 * result list and the feedback for /search/semantic and its index job.
 *
 * main.js keeps both job routes and hands them in as `startSearch` and
 * `startIndex`, so the panel route scanners still find the literals there.
 * Everything else lives here so a test can drive it with a fake document.
 */
import { buildVisualSearchCard, collectVisualSearchClipPaths } from "./uxp-utils.js";

const STATUS_ID = "searchStatus";
const SEARCH_BUTTON_ID = "runFootageSearchBtn";
const INDEX_BUTTON_ID = "runVisualIndexBtn";
const MAX_RESULTS = 100;

export function createVisualSearchController({
  documentRef,
  translate: t,
  formatTranslate: formatI18n,
  formatCountTranslate: formatCountI18n,
  ui,
  setTextAndTitle,
  resetSearchResults,
  syncSearchPanelState,
  isBackendConnected,
  getSearchResultScoreLabel,
  onLoadResult,
  getInputs,
  startSearch,
  startIndex,
}) {
  /** "visual" searches frames with CLIP, "transcript" (the default) searches spoken words. */
  function getMode() {
    return documentRef.getElementById("searchMode")?.value === "visual" ? "visual" : "transcript";
  }

  function applyMode() {
    if (getMode() === "visual") {
      resetSearchResults(
        t("uxp.search.runtime.visual_empty_kicker", "Search by what's on screen"),
        t("uxp.search.runtime.visual_empty_hint", "Describe a shot and OpenCut looks through the key frames of every video in the project. The first search indexes them, which takes a while, so you can build the visual index ahead of time.")
      );
      setTextAndTitle(
        STATUS_ID,
        t("uxp.search.runtime.visual_mode_status", "Visual search reads the pictures, not the words. Describe what's on screen.")
      );
    } else {
      resetSearchResults(
        t("uxp.search.search_the_library", "Search the library"),
        t("uxp.search.empty_search_hint", "Index a folder, then search with descriptive phrases to pull the right shot back into the workspace.")
      );
      setTextAndTitle(
        STATUS_ID,
        t("uxp.search.choose_query_status", "Choose a descriptive query, then load the best match back into the workspace.")
      );
    }
    syncSearchPanelState();
  }

  /**
   * Shows the index button in visual mode and, there, owns the search button.
   * Returns true when visual mode is on so the transcript rules stay out.
   */
  function syncButtons(backendOnline, query) {
    const visual = getMode() === "visual";
    const searchBtn = documentRef.getElementById(SEARCH_BUTTON_ID);
    if (visual && searchBtn && !searchBtn.classList.contains("loading")) {
      // Visual search builds its own CLIP index on the fly, so it doesn't
      // wait for the transcript library above.
      searchBtn.disabled = !backendOnline || !query;
      searchBtn.title = !backendOnline
        ? t("uxp.search.runtime.reconnect_before_searching", "Reconnect the backend before searching the library.")
        : (query
            ? t("uxp.search.runtime.visual_search_title", "Search the project's clips by what's on screen and load the best shot into the workspace.")
            : t("uxp.search.runtime.visual_enter_query_title", "Describe what's on screen before searching the project's clips."));
    }
    const indexBtn = documentRef.getElementById(INDEX_BUTTON_ID);
    if (indexBtn) {
      indexBtn.classList.toggle("oc-hidden", !visual);
      if (!indexBtn.classList.contains("loading")) {
        indexBtn.disabled = !backendOnline;
        indexBtn.title = backendOnline
          ? t("uxp.search.runtime.visual_index_title", "Index the key frames of every video in the project so visual searches start faster.")
          : t("uxp.search.runtime.reconnect_before_visual_index", "Reconnect the backend before building the visual index.");
      }
    }
    return visual;
  }

  /** Adds project_dir only when the project is saved, so the portable index has a home. */
  function withProjectDir(payload, projectDir) {
    return projectDir ? { ...payload, project_dir: projectDir } : payload;
  }

  /** Video clips and the saved project's folder, or null after telling the user there are none. */
  async function readInputs() {
    const { clips, projectDir } = await getInputs();
    const clipPaths = collectVisualSearchClipPaths(clips);
    if (clipPaths.length) return { clipPaths, projectDir: projectDir || "" };
    const message = t("uxp.search.runtime.visual_no_clips", "There are no video clips in the project yet. Import footage into Premiere, then try again.");
    setTextAndTitle(STATUS_ID, message);
    ui.showToast(message, "warning");
    syncSearchPanelState();
    return null;
  }

  function appendSpan(parent, className, text) {
    const span = documentRef.createElement("span");
    span.className = className;
    span.textContent = text;
    parent.appendChild(span);
    return span;
  }

  function resultLabel(card) {
    return card.name || formatI18n("uxp.search.runtime.result_number", "Result {index}", { index: card.index + 1 });
  }

  function buildResultElement(card, list) {
    const label = resultLabel(card);
    const scoreLabel = card.percent === null
      ? getSearchResultScoreLabel({}, card.index)
      : formatI18n("uxp.search.runtime.match_percent", "{percent}% match", { percent: card.percent });
    const el = documentRef.createElement("button");
    el.type = "button";
    el.className = `oc-result-list-item${card.index === 0 ? " is-featured" : ""}`;
    el.setAttribute("aria-pressed", "false");
    el.setAttribute(
      "aria-label",
      formatI18n("uxp.search.runtime.visual_result_aria", "{label} at {time}, {score}", { label, time: card.timestamp, score: scoreLabel })
    );
    el.title = card.path || label;
    if (card.thumbnail) {
      const img = documentRef.createElement("img");
      img.className = "oc-result-list-item-thumb";
      img.src = card.thumbnail;
      img.alt = "";
      el.appendChild(img);
    }
    const copy = appendSpan(el, "oc-result-list-item-copy", "");
    const topline = appendSpan(copy, "oc-result-list-item-topline", "");
    appendSpan(topline, "oc-result-list-item-title", label);
    appendSpan(topline, "oc-result-list-item-badge", t("uxp.search.runtime.kind_visual", "Visual"));
    appendSpan(topline, "oc-result-list-item-badge", card.timestamp);
    appendSpan(copy, "oc-result-list-item-path", card.path || t("uxp.search.runtime.path_unavailable", "Path unavailable for this result"));
    appendSpan(el, "oc-result-list-item-meta", scoreLabel);
    el.addEventListener("click", () => {
      if (!card.path) {
        ui.showToast(t("uxp.search.runtime.match_missing_path_toast", "This result does not include a loadable file path yet."), "warning");
        return;
      }
      onLoadResult(list, el, card.path, label);
    });
    return el;
  }

  function renderResults(result, query) {
    const hits = Array.isArray(result?.results) ? result.results : [];
    const cards = hits.map((item, index) => buildVisualSearchCard(item, index));
    if (!cards.length) {
      resetSearchResults(
        t("uxp.search.runtime.no_matches_yet", "No matches yet"),
        t("uxp.search.runtime.visual_no_matches_detail", "Nothing on screen matched closely enough. Describe the shot another way, or name what's in the frame.")
      );
      setTextAndTitle(STATUS_ID, t("uxp.search.runtime.visual_no_matches_status", "No clips matched that description."), query);
      ui.setStatus(t("uxp.search.runtime.search_no_matches_status", "Search returned no matches."));
      syncSearchPanelState();
      return;
    }
    const list = documentRef.getElementById("searchResultList");
    if (list) {
      list.textContent = "";
      cards.forEach((card) => list.appendChild(buildResultElement(card, list)));
    }
    setTextAndTitle(
      STATUS_ID,
      formatCountI18n(
        cards.length,
        "uxp.search.runtime.matches_ready_one",
        "{count} match is ready. Start with {label}, or compare results before loading a shot back into the workspace.",
        "uxp.search.runtime.matches_ready_many",
        "{count} matches are ready. Start with {label}, or compare results before loading a shot back into the workspace.",
        { label: resultLabel(cards[0]) }
      ),
      query
    );
    ui.setStatus(
      formatCountI18n(
        cards.length,
        "uxp.search.runtime.search_ready_one",
        "Search ready. {count} result.",
        "uxp.search.runtime.search_ready_many",
        "Search ready. {count} results."
      )
    );
    syncSearchPanelState();
  }

  function onProgress(pct, msg) {
    ui.setProgress(pct);
    if (msg) ui.setProcessingMsg(msg);
  }

  function finishJob(buttonId) {
    ui.hideProcessing();
    ui.setButtonLoading(buttonId, false);
  }

  async function search(query, limit) {
    if (!query) {
      setTextAndTitle(
        STATUS_ID,
        t("uxp.search.runtime.visual_enter_query_title", "Describe what's on screen before searching the project's clips.")
      );
      ui.showToast(t("uxp.search.runtime.enter_search_query", "Please enter a search query."), "warning");
      syncSearchPanelState();
      return;
    }
    const inputs = await readInputs();
    if (!inputs) return;
    const maxResults = Number.isFinite(limit) && limit > 0 ? Math.min(limit, MAX_RESULTS) : 10;

    ui.setButtonLoading(SEARCH_BUTTON_ID, true);
    ui.showProcessing(t("uxp.search.runtime.visual_searching", "Searching clips by what's on screen…"));
    setTextAndTitle(
      STATUS_ID,
      formatI18n("uxp.search.runtime.searching_for_query", "Searching for \"{query}\"…", { query })
    );

    await startSearch(
      withProjectDir({ query, clip_paths: inputs.clipPaths, max_results: maxResults }, inputs.projectDir),
      onProgress,
      (result) => {
        finishJob(SEARCH_BUTTON_ID);
        renderResults(result, query);
      },
      (err) => {
        finishJob(SEARCH_BUTTON_ID);
        const error = String(err ?? "");
        setTextAndTitle(
          STATUS_ID,
          formatI18n("uxp.search.runtime.search_failed_detail", "Could not search the library. {error}", { error }),
          error
        );
        ui.showToast(formatI18n("uxp.search.runtime.search_error", "Search error: {error}", { error }), "error");
        ui.setStatus(t("uxp.search.runtime.search_failed_status", "Search failed. Review the query or reconnect the backend."), "error");
        syncSearchPanelState();
      }
    );
  }

  async function buildIndex() {
    if (!isBackendConnected()) {
      const message = t("uxp.search.runtime.reconnect_before_visual_index", "Reconnect the backend before building the visual index.");
      setTextAndTitle(STATUS_ID, message);
      ui.showToast(message, "warning");
      syncSearchPanelState();
      return;
    }
    const inputs = await readInputs();
    if (!inputs) return;
    const { clipPaths, projectDir } = inputs;

    ui.setButtonLoading(INDEX_BUTTON_ID, true);
    ui.showProcessing(t("uxp.search.runtime.visual_indexing", "Building the visual index…"));
    setTextAndTitle(
      STATUS_ID,
      formatCountI18n(
        clipPaths.length,
        "uxp.search.runtime.visual_indexing_status_one",
        "Indexing the key frames of {count} video. The first run takes a while.",
        "uxp.search.runtime.visual_indexing_status_many",
        "Indexing the key frames of {count} videos. The first run takes a while."
      )
    );

    await startIndex(
      withProjectDir({ clip_paths: clipPaths }, projectDir),
      onProgress,
      (result) => {
        finishJob(INDEX_BUTTON_ID);
        const count = Number(result?.clip_count ?? clipPaths.length) || 0;
        const message = formatCountI18n(
          count,
          "uxp.search.runtime.visual_index_ready_one",
          "Visual index ready. {count} video can be searched by what's on screen.",
          "uxp.search.runtime.visual_index_ready_many",
          "Visual index ready. {count} videos can be searched by what's on screen."
        );
        setTextAndTitle(STATUS_ID, message);
        ui.showToast(message, "success");
        ui.setStatus(message, "success");
        syncSearchPanelState();
      },
      (err) => {
        finishJob(INDEX_BUTTON_ID);
        const error = String(err ?? "");
        const message = formatI18n("uxp.search.runtime.visual_index_failed", "Couldn't build the visual index. {error}", { error });
        setTextAndTitle(STATUS_ID, message, error);
        ui.showToast(message, "error");
        ui.setStatus(message, "error");
        syncSearchPanelState();
      }
    );
  }

  return { getMode, applyMode, syncButtons, renderResults, search, buildIndex };
}
