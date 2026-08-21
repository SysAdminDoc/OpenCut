/* OpenCut CEP collapsible card headers. Classic script + CommonJS for tests. */
(function (root, factory) {
    "use strict";
    var api = factory();
    if (typeof module === "object" && module.exports) module.exports = api;
    if (root) root.OpenCutCollapsibleCards = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
    "use strict";

    // A CEP card has no single body element: everything after the header is
    // content, so collapsing means hiding the header's later siblings.
    function bodyElements(header) {
        var card = typeof header.closest === "function" ? header.closest(".card") : null;
        if (!card) return [];
        var bodies = [];
        var afterHeader = false;
        for (var i = 0; i < card.children.length; i++) {
            if (card.children[i] === header) { afterHeader = true; continue; }
            if (afterHeader) bodies.push(card.children[i]);
        }
        return bodies;
    }

    function setCollapsed(header, collapsed) {
        if (!header) return false;
        header.classList.toggle("collapsed", collapsed);
        header.setAttribute("aria-expanded", collapsed ? "false" : "true");
        var bodies = bodyElements(header);
        for (var i = 0; i < bodies.length; i++) {
            bodies[i].style.display = collapsed ? "none" : "";
        }
        return collapsed;
    }

    function toggleCollapsibleCard(header) {
        if (!header) return false;
        return setCollapsed(header, !header.classList.contains("collapsed"));
    }

    function bindCollapsibleCard(header) {
        if (!header || header.getAttribute("data-collapsible-bound") === "true") return false;
        header.setAttribute("role", "button");
        header.setAttribute("tabindex", "0");
        header.setAttribute("aria-expanded", header.classList.contains("collapsed") ? "false" : "true");
        header.addEventListener("click", function () {
            toggleCollapsibleCard(header);
        });
        header.addEventListener("keydown", function (event) {
            // "Spacebar" is the legacy key name CEP's embedded Chromium reports.
            if (event.key !== "Enter" && event.key !== " " && event.key !== "Spacebar") return;
            event.preventDefault();
            toggleCollapsibleCard(header);
        });
        header.setAttribute("data-collapsible-bound", "true");
        return true;
    }

    function initCollapsibleCards(scope) {
        var root = scope || (typeof document !== "undefined" ? document : null);
        if (!root || typeof root.querySelectorAll !== "function") return 0;
        var headers = root.querySelectorAll("[data-collapsible]");
        var bound = 0;
        for (var i = 0; i < headers.length; i++) {
            if (bindCollapsibleCard(headers[i])) bound++;
        }
        return bound;
    }

    return {
        bindCollapsibleCard: bindCollapsibleCard,
        initCollapsibleCards: initCollapsibleCards,
        toggleCollapsibleCard: toggleCollapsibleCard
    };
});
