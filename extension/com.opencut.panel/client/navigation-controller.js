/* OpenCut CEP navigation wiring controller. Classic script + CommonJS for tests. */
(function (root, factory) {
    "use strict";
    var api = factory();
    if (typeof module === "object" && module.exports) module.exports = api;
    if (root) root.OpenCutNavigationController = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
    "use strict";

    function createNavigationController(options) {
        options = options || {};
        var documentRef = options.documentRef
            || (typeof document !== "undefined" ? document : null);
        var windowRef = options.windowRef
            || (typeof window !== "undefined" ? window : null);
        var getElement = typeof options.getElement === "function"
            ? options.getElement
            : function (id) { return documentRef && documentRef.getElementById ? documentRef.getElementById(id) : null; };
        var getVisibleTabButtons = typeof options.getVisibleTabButtons === "function"
            ? options.getVisibleTabButtons
            : function (container, selector) { return container ? Array.prototype.slice.call(container.querySelectorAll(selector)) : []; };
        var moveFocusAndActivate = typeof options.moveFocusAndActivate === "function"
            ? options.moveFocusAndActivate
            : function () {};
        var getPanelTabName = typeof options.getPanelTabName === "function" ? options.getPanelTabName : function () { return ""; };
        var ensureSubTabShell = typeof options.ensureSubTabShell === "function" ? options.ensureSubTabShell : function () {};
        var updateSubTabOverflowState = typeof options.updateSubTabOverflowState === "function" ? options.updateSubTabOverflowState : function () {};
        var activateNavTab = typeof options.activateNavTab === "function" ? options.activateNavTab : function () {};
        var activateSubTab = typeof options.activateSubTab === "function" ? options.activateSubTab : function () {};
        var getInitialNav = typeof options.getInitialNav === "function" ? options.getInitialNav : function () { return "cut"; };
        var getInitialSub = typeof options.getInitialSub === "function" ? options.getInitialSub : function () { return ""; };
        var updateWorkspaceClipStatus = typeof options.updateWorkspaceClipStatus === "function" ? options.updateWorkspaceClipStatus : function () {};
        var onResize = typeof options.onResize === "function" ? options.onResize : function () {};
        var cleanupCallbacks = [];
        var bound = false;
        var disposed = false;

        function get(id) {
            return getElement(id);
        }

        function listen(target, type, handler, listenerOptions) {
            if (!target || typeof target.addEventListener !== "function") return;
            target.addEventListener(type, handler, listenerOptions);
            cleanupCallbacks.push(function () {
                if (target && typeof target.removeEventListener === "function") {
                    target.removeEventListener(type, handler, listenerOptions);
                }
            });
        }

        function bind() {
            if (disposed || bound || !documentRef) return false;
            bound = true;
            var navContainer = get("navTabs");
            if (navContainer) navContainer.setAttribute("aria-orientation", "vertical");

            var navButtons = documentRef.querySelectorAll(".nav-tab");
            for (var i = 0; i < navButtons.length; i++) {
                var navName = navButtons[i].getAttribute("data-nav") || ("tab-" + i);
                navButtons[i].id = navButtons[i].id || ("nav-tab-" + navName);
                navButtons[i].setAttribute("aria-controls", "panel-" + navName);
                navButtons[i].tabIndex = navButtons[i].classList.contains("active") ? 0 : -1;
                var controlledPanel = get("panel-" + navName);
                if (controlledPanel) {
                    controlledPanel.setAttribute("aria-labelledby", navButtons[i].id);
                    controlledPanel.setAttribute("aria-hidden", controlledPanel.classList.contains("active") ? "false" : "true");
                    controlledPanel.hidden = !controlledPanel.classList.contains("active");
                }

                (function (button) {
                    listen(button, "click", function () {
                        activateNavTab(button.getAttribute("data-nav"));
                    });
                    listen(button, "keydown", function (event) {
                        var buttons = getVisibleTabButtons(button.parentElement, ".nav-tab");
                        if (event.key === "ArrowDown" || event.key === "ArrowRight") {
                            event.preventDefault();
                            moveFocusAndActivate(buttons, button, 1);
                        } else if (event.key === "ArrowUp" || event.key === "ArrowLeft") {
                            event.preventDefault();
                            moveFocusAndActivate(buttons, button, -1);
                        } else if (event.key === "Home" && buttons.length) {
                            event.preventDefault();
                            buttons[0].focus();
                            buttons[0].click();
                        } else if (event.key === "End" && buttons.length) {
                            event.preventDefault();
                            buttons[buttons.length - 1].focus();
                            buttons[buttons.length - 1].click();
                        }
                    });
                })(navButtons[i]);
            }

            var subTabContainers = documentRef.querySelectorAll(".sub-tabs");
            for (var j = 0; j < subTabContainers.length; j++) {
                (function (container) {
                    var buttons = container.querySelectorAll(".sub-tab");
                    var parentPanel = container.closest(".nav-panel");
                    var parentTabName = getPanelTabName(parentPanel);
                    ensureSubTabShell(container);
                    container.setAttribute("aria-orientation", "horizontal");
                    listen(container, "scroll", function () {
                        updateSubTabOverflowState(container);
                    }, { passive: true });
                    listen(container, "wheel", function (event) {
                        var hasOverflow = container.scrollWidth > container.clientWidth + 1;
                        if (!hasOverflow || Math.abs(event.deltaY) <= Math.abs(event.deltaX)) return;
                        container.scrollLeft += event.deltaY;
                        event.preventDefault();
                    }, { passive: false });
                    for (var k = 0; k < buttons.length; k++) {
                        var subName = buttons[k].getAttribute("data-sub") || (parentTabName + "-sub-" + k);
                        buttons[k].id = buttons[k].id || ("sub-tab-" + subName);
                        buttons[k].setAttribute("role", "tab");
                        buttons[k].setAttribute("aria-controls", "sub-" + subName);
                        buttons[k].setAttribute("aria-selected", buttons[k].classList.contains("active") ? "true" : "false");
                        buttons[k].tabIndex = buttons[k].classList.contains("active") ? 0 : -1;
                        var subPanel = get("sub-" + subName);
                        if (subPanel) {
                            subPanel.setAttribute("role", "tabpanel");
                            subPanel.setAttribute("aria-labelledby", buttons[k].id);
                            subPanel.setAttribute("aria-hidden", subPanel.classList.contains("active") ? "false" : "true");
                            subPanel.hidden = !subPanel.classList.contains("active");
                        }

                        (function (button) {
                            listen(button, "click", function () {
                                activateSubTab(parentTabName, button.getAttribute("data-sub"));
                            });
                            listen(button, "keydown", function (event) {
                                var visibleButtons = getVisibleTabButtons(container, ".sub-tab");
                                if (event.key === "ArrowRight" || event.key === "ArrowDown") {
                                    event.preventDefault();
                                    moveFocusAndActivate(visibleButtons, button, 1);
                                } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
                                    event.preventDefault();
                                    moveFocusAndActivate(visibleButtons, button, -1);
                                } else if (event.key === "Home" && visibleButtons.length) {
                                    event.preventDefault();
                                    visibleButtons[0].focus();
                                    visibleButtons[0].click();
                                } else if (event.key === "End" && visibleButtons.length) {
                                    event.preventDefault();
                                    visibleButtons[visibleButtons.length - 1].focus();
                                    visibleButtons[visibleButtons.length - 1].click();
                                }
                            });
                        })(buttons[k]);
                    }
                    activateSubTab(parentTabName, getInitialSub(parentTabName), { remember: false, scroll: false });
                    updateSubTabOverflowState(container);
                })(subTabContainers[j]);
            }

            activateNavTab(getInitialNav(), { remember: false, scroll: false });
            updateWorkspaceClipStatus();
            listen(windowRef, "resize", onResize);
            return true;
        }

        function dispose() {
            if (disposed) return;
            disposed = true;
            for (var i = cleanupCallbacks.length - 1; i >= 0; i--) cleanupCallbacks[i]();
            cleanupCallbacks = [];
            bound = false;
        }

        return { bind: bind, dispose: dispose };
    }

    return { createNavigationController: createNavigationController };
});
