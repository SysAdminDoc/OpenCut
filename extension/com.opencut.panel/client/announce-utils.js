/* OpenCut CEP live-region + focus-recovery helpers. Classic script + CommonJS for tests.
 *
 * Terminal job results previously only unhid the results card, which is a
 * silent DOM mutation: assistive technology has no reason to read it, so a
 * screen-reader user could be left with no completion signal and no route to
 * the recovery action.
 *
 * Two separate regions are used rather than one whose politeness is swapped.
 * Assistive technology reads `aria-live` when the region is first seen, so
 * flipping polite/assertive on a live node is unreliable across screen
 * readers; owning one node per politeness keeps the contract predictable.
 */
(function (root, factory) {
    "use strict";
    var api = factory();
    if (typeof module === "object" && module.exports) module.exports = api;
    if (root) root.OpenCutAnnounce = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
    "use strict";

    /* Write a message so repeats re-announce.
     *
     * Setting identical text on a live region is a no-op for most screen
     * readers, so running the same job twice would announce only the first
     * result. Clearing first makes the second write a real change. */
    function setLiveRegionMessage(node, message) {
        if (!node) return "";
        var text = message == null ? "" : String(message);
        node.textContent = "";
        node.textContent = text;
        return text;
    }

    /* Route a terminal result to the region matching its urgency.
     *
     * Success is polite: it must not interrupt whatever the user is reading.
     * Failure is assertive because it carries a recovery action the user
     * needs before continuing. Only one region ever holds text, so a result
     * is never announced twice. */
    function announceResult(regions, tone, message) {
        var safe = regions || {};
        var polite = safe.polite || null;
        var assertive = safe.assertive || null;
        var urgent = tone === "error";
        setLiveRegionMessage(polite, urgent ? "" : message);
        setLiveRegionMessage(assertive, urgent ? message : "");
        return urgent ? "assertive" : "polite";
    }

    function clearAnnouncements(regions) {
        var safe = regions || {};
        setLiveRegionMessage(safe.polite || null, "");
        setLiveRegionMessage(safe.assertive || null, "");
    }

    function isElementRenderable(node) {
        if (!node || node.nodeType !== 1) return false;
        if (node.hasAttribute && node.hasAttribute("hidden")) return false;
        if (node.getAttribute && node.getAttribute("aria-hidden") === "true") return false;
        // offsetParent is null for display:none subtrees. Fixed-position
        // elements also report null, so treat a measurable box as visible.
        if ("offsetParent" in node && node.offsetParent === null) {
            var rect = typeof node.getBoundingClientRect === "function"
                ? node.getBoundingClientRect()
                : null;
            if (!rect || (!rect.width && !rect.height)) return false;
        }
        return true;
    }

    /* Decide whether finishing a job stranded the user's focus.
     *
     * Moving focus on every result would be its own accessibility defect —
     * it yanks the user out of wherever they were. Focus is only rescued
     * when it has nowhere left to go: the run control that had focus is now
     * disabled or hidden, or focus already fell back to the document body. */
    function focusWasStranded(activeElement, doc) {
        var body = doc && doc.body ? doc.body : null;
        if (!activeElement) return true;
        if (body && activeElement === body) return true;
        if (doc && activeElement === doc.documentElement) return true;
        if (activeElement.disabled) return true;
        if (!isElementRenderable(activeElement)) return true;
        return false;
    }

    /* Give the results card focus without leaving a permanent tab stop. */
    function focusResultsRegion(node) {
        if (!node || typeof node.focus !== "function") return false;
        if (node.getAttribute && node.getAttribute("tabindex") === null) {
            node.setAttribute("tabindex", "-1");
        }
        try {
            node.focus({ preventScroll: false });
        } catch (err) {
            node.focus();
        }
        return true;
    }

    return {
        setLiveRegionMessage: setLiveRegionMessage,
        announceResult: announceResult,
        clearAnnouncements: clearAnnouncements,
        isElementRenderable: isElementRenderable,
        focusWasStranded: focusWasStranded,
        focusResultsRegion: focusResultsRegion
    };
});
