/* OpenCut studio shell refinements shared by the CEP and UXP surfaces. */
(function initStudioShell() {
  "use strict";

  if (document.documentElement.dataset.studioShell === "v2") return;
  document.documentElement.dataset.studioShell = "v2";

  var isCep = Boolean(document.querySelector(".sidebar"));
  var refreshButton = document.getElementById(isCep ? "refreshAllBtn" : "refreshBtn");
  var workspaceTitle = document.getElementById("workspaceOverviewTitle");
  var workspaceKicker = document.querySelector(".oc-workspace-kicker");

  function ensureVisibleRefreshAction() {
    if (!refreshButton) return;

    if (isCep) {
      if (!refreshButton.querySelector(".studio-connect-label")) {
        refreshButton.insertAdjacentHTML(
          "beforeend",
          '<span class="studio-connect-label" aria-hidden="true"></span>',
        );
      }
      syncVisibleRefreshLabel();
      return;
    }

    var actions = document.querySelector(".oc-header-right");
    if (!actions || actions.querySelector(".studio-header-connect")) return;
    var header = document.querySelector(".oc-header");
    var left = document.querySelector(".oc-header-left");
    if (header && left && !header.querySelector(".studio-header-context")) {
      left.insertAdjacentHTML(
        "afterend",
        '<div class="studio-header-context" aria-hidden="true">' +
          '<span data-studio-workspace-kicker></span>' +
          '<strong data-studio-workspace-title></strong>' +
        "</div>",
      );
    }
    actions.insertAdjacentHTML(
      "beforeend",
      '<button type="button" class="studio-header-connect" data-studio-refresh>' +
        '<span class="studio-connect-label" aria-hidden="true"></span>' +
      "</button>",
    );
    syncWorkspaceContext();
    syncVisibleRefreshLabel();
  }

  function refreshLabel() {
    if (!refreshButton) return "Refresh connection";
    return refreshButton.getAttribute("aria-label") ||
      refreshButton.getAttribute("title") ||
      "Refresh connection";
  }

  function syncVisibleRefreshLabel() {
    var label = refreshLabel();
    var visibleLabel = label.replace(/\s+and capabilities$/i, "");
    document.querySelectorAll(".studio-connect-label").forEach(function (node) {
      node.textContent = visibleLabel;
    });
    document.querySelectorAll("[data-studio-refresh]").forEach(function (node) {
      node.setAttribute("aria-label", label);
      node.setAttribute("title", label);
    });
  }

  function syncWorkspaceContext() {
    document.querySelectorAll("[data-studio-workspace-kicker]").forEach(function (node) {
      node.textContent = workspaceKicker ? workspaceKicker.textContent : "";
    });
    document.querySelectorAll("[data-studio-workspace-title]").forEach(function (node) {
      node.textContent = workspaceTitle ? workspaceTitle.textContent : "";
    });
  }

  ensureVisibleRefreshAction();

  document.addEventListener("click", function (event) {
    var proxy = event.target.closest("[data-studio-refresh]");
    if (proxy && refreshButton && proxy !== refreshButton) refreshButton.click();
  });

  if (refreshButton && typeof MutationObserver === "function") {
    new MutationObserver(syncVisibleRefreshLabel).observe(refreshButton, {
      attributes: true,
      attributeFilter: ["aria-label", "title"],
    });
  }

  if (typeof MutationObserver === "function") {
    new MutationObserver(function () {
      syncVisibleRefreshLabel();
      syncWorkspaceContext();
    }).observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["lang"],
    });
    [workspaceTitle, workspaceKicker].filter(Boolean).forEach(function (node) {
      new MutationObserver(syncWorkspaceContext).observe(node, {
        childList: true,
        subtree: true,
      });
    });
  }
})();
