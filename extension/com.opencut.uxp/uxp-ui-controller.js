function fallbackEscapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

export function createUxpUiController({
  documentRef = globalThis.document,
  windowRef = globalThis.window || globalThis,
  requestAnimationFrameFn = globalThis.requestAnimationFrame,
  setIntervalFn = globalThis.setInterval,
  clearIntervalFn = globalThis.clearInterval,
  setTimeoutFn = globalThis.setTimeout,
  clearTimeoutFn = globalThis.clearTimeout,
  ResizeObserverCtor = globalThis.ResizeObserver,
  translate = (_key, fallback) => fallback,
  onInvalidatePProCache = () => {},
  onWorkspaceTabChange = () => {},
  isBackendConnected = () => false,
  getWorkspaceTitle = (tabId) => tabId || "",
  onQuickActionStateChange = () => {},
  escapeHtmlValue = fallbackEscapeHtml,
} = {}) {
  const requestFrame = requestAnimationFrameFn || ((callback) => callback());
  const setIntervalImpl = setIntervalFn || globalThis.setInterval;
  const clearIntervalImpl = clearIntervalFn || globalThis.clearInterval;
  const setTimeoutImpl = setTimeoutFn || globalThis.setTimeout;
  const clearTimeoutImpl = clearTimeoutFn || globalThis.clearTimeout;
  const cleanupCallbacks = [];
  const timeoutHandles = new Set();
  let elapsedTimer = null;
  let elapsedSec = 0;
  let navigationBound = false;
  let disposed = false;

  function getElement(id) {
    return documentRef?.getElementById?.(id) || null;
  }

  function listen(target, type, listener, options) {
    if (!target?.addEventListener) return;
    target.addEventListener(type, listener, options);
    cleanupCallbacks.push(() => target.removeEventListener?.(type, listener, options));
  }

  function schedule(callback, delay) {
    const handle = setTimeoutImpl(() => {
      timeoutHandles.delete(handle);
      callback();
    }, delay);
    timeoutHandles.add(handle);
    return handle;
  }

  function syncTabOverflowControls() {
    const nav = getElement("tabNav");
    const shell = getElement("tabNavShell");
    const previous = getElement("tabScrollPrev");
    const next = getElement("tabScrollNext");
    if (!nav || !shell || !previous || !next) return;
    const overflowing = nav.scrollWidth > nav.clientWidth + 2;
    shell.dataset.overflow = overflowing ? "true" : "false";
    previous.hidden = !overflowing;
    next.hidden = !overflowing;
    if (!overflowing) {
      nav.scrollLeft = 0;
      previous.disabled = true;
      next.disabled = true;
      return;
    }
    // The buttons move the ACTIVE tab (activateRelativeTab), not the scroll
    // position, so derive disabled state from the active tab index.
    const tabs = Array.from(nav.querySelectorAll?.(".oc-tab") || []);
    const activeIndex = tabs.findIndex((tab) => tab.classList.contains("active"));
    previous.disabled = activeIndex <= 0;
    next.disabled = activeIndex < 0 || activeIndex >= tabs.length - 1;
  }

  function revealTabButton(button) {
    const nav = getElement("tabNav");
    if (!nav || !button) return;
    const alignButton = () => {
      const navBounds = nav.getBoundingClientRect();
      const tabBounds = button.getBoundingClientRect();
      if (tabBounds.left < navBounds.left) {
        nav.scrollLeft -= navBounds.left - tabBounds.left + 6;
      } else if (tabBounds.right > navBounds.right) {
        nav.scrollLeft += tabBounds.right - navBounds.right + 6;
      }
      syncTabOverflowControls();
    };
    alignButton();
    requestFrame(alignButton);
  }

  function activateRelativeTab(delta) {
    const tabs = Array.from(documentRef?.querySelectorAll?.(".oc-tab") || []);
    const activeIndex = tabs.findIndex((tab) => tab.classList.contains("active"));
    if (activeIndex < 0) return;
    const target = tabs[Math.max(0, Math.min(tabs.length - 1, activeIndex + delta))];
    if (!target || target === tabs[activeIndex]) return;
    switchTab(target.dataset.tab);
    target.focus?.();
  }

  function bindNavigation() {
    if (navigationBound || disposed) return;
    navigationBound = true;
    const tabs = Array.from(documentRef?.querySelectorAll?.(".oc-tab") || []);
    tabs.forEach((button, index) => {
      listen(button, "click", () => switchTab(button.dataset.tab));
      listen(button, "keydown", (event) => {
        let target = null;
        if (event.key === "ArrowRight" || event.key === "ArrowDown") {
          target = tabs[(index + 1) % tabs.length];
        } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
          target = tabs[(index - 1 + tabs.length) % tabs.length];
        } else if (event.key === "Home") {
          target = tabs[0];
        } else if (event.key === "End") {
          target = tabs[tabs.length - 1];
        }
        if (target) {
          event.preventDefault();
          target.focus?.();
          switchTab(target.dataset.tab);
        }
      });
    });

    const tabNav = getElement("tabNav");
    listen(tabNav, "scroll", syncTabOverflowControls, { passive: true });
    listen(getElement("tabScrollPrev"), "click", () => activateRelativeTab(-1));
    listen(getElement("tabScrollNext"), "click", () => activateRelativeTab(1));
    listen(windowRef, "resize", syncTabOverflowControls);
    if (typeof ResizeObserverCtor === "function" && tabNav) {
      const tabResizeObserver = new ResizeObserverCtor(syncTabOverflowControls);
      tabResizeObserver.observe(tabNav);
      cleanupCallbacks.push(() => tabResizeObserver.disconnect?.());
    }
    requestFrame(syncTabOverflowControls);
  }

  function switchTab(tabId) {
    onInvalidatePProCache();
    let activeButton = null;
    documentRef?.querySelectorAll?.(".oc-tab").forEach((button) => {
      const active = button.dataset.tab === tabId;
      button.classList.toggle("active", active);
      button.setAttribute("aria-selected", active ? "true" : "false");
      button.tabIndex = active ? 0 : -1;
      if (active) activeButton = button;
    });
    documentRef?.querySelectorAll?.(".oc-tab-panel").forEach((panel) => {
      const active = panel.id === "tab-" + tabId;
      panel.classList.toggle("active", active);
      panel.hidden = !active;
      panel.setAttribute("aria-hidden", active ? "false" : "true");
    });
    const main = getElement("mainContent");
    if (main) main.scrollTop = 0;
    revealTabButton(activeButton);
    onWorkspaceTabChange(tabId);
    if (isBackendConnected()) {
      setStatus(
        translate("uxp.status.workspace", "{workspace} workspace")
          .replace("{workspace}", getWorkspaceTitle(tabId)),
      );
    } else {
      setStatus(
        translate("uxp.status.backend_offline", "OpenCut backend offline. Start the local service to run jobs."),
        "error",
      );
    }
  }

  function showProcessing(msg = translate("processing.processing", "Processing…")) {
    const banner = getElement("processingBanner");
    if (banner) banner.classList.remove("hidden");
    getElement("mainContent")?.setAttribute("aria-busy", "true");
    setProcessingMsg(msg);
    setProgress(0);
    startElapsedTimer();
    onQuickActionStateChange();
  }

  function hideProcessing() {
    const banner = getElement("processingBanner");
    if (banner) banner.classList.add("hidden");
    getElement("mainContent")?.setAttribute("aria-busy", "false");
    stopElapsedTimer();
    onQuickActionStateChange();
  }

  function setProcessingMsg(msg) {
    const element = getElement("processingMsg");
    if (element) element.textContent = msg;
  }

  function setProgress(pct) {
    const fill = getElement("progressFill");
    if (!fill) return;
    const clamped = Math.min(100, Math.max(0, pct));
    fill.style.width = clamped + "%";
    fill.setAttribute("aria-valuenow", String(Math.round(clamped)));
  }

  function startElapsedTimer() {
    stopElapsedTimer();
    elapsedSec = 0;
    updateElapsedDisplay();
    elapsedTimer = setIntervalImpl(() => {
      elapsedSec++;
      updateElapsedDisplay();
    }, 1000);
  }

  function stopElapsedTimer() {
    if (elapsedTimer) {
      clearIntervalImpl(elapsedTimer);
      elapsedTimer = null;
    }
    elapsedSec = 0;
    updateElapsedDisplay();
  }

  function updateElapsedDisplay() {
    const element = getElement("processingElapsed");
    if (!element) return;
    const minutes = Math.floor(elapsedSec / 60);
    const seconds = elapsedSec % 60;
    element.textContent = minutes > 0 ? minutes + "m " + seconds + "s" : seconds + "s";
  }

  function inferStatusTone(msg) {
    const text = String(msg || "").toLowerCase();
    if (!text) return "neutral";
    if (/(error|failed|offline|unavailable|timed out|timeout|could not|stopped)/.test(text)) return "error";
    if (/(connecting|running|processing|loading|refreshing|starting|detecting|indexing|scanning)/.test(text)) return "working";
    if (/(online|connected|saved|ready|done|complete|loaded|updated|synced)/.test(text)) return "success";
    return "neutral";
  }

  function setStatus(msg, tone) {
    const element = getElement("statusText");
    if (element) element.textContent = msg || "";
    const bar = getElement("statusBar");
    if (bar) {
      bar.dataset.state = tone || inferStatusTone(msg);
      bar.title = msg || "";
    }
  }

  function setStatusRight(msg) {
    const element = getElement("statusRight");
    if (!element) return;
    element.textContent = msg || "";
    element.classList.toggle("is-empty", !msg);
  }

  function setConnection(state) {
    const dot = getElement("connDot");
    const label = getElement("connLabel");
    const status = getElement("connectionStatus");
    const statusBar = getElement("statusBar");
    if (!dot || !label) return;
    dot.className = "oc-conn-dot " + state;
    if (status) status.dataset.state = state;
    if (statusBar) statusBar.dataset.connection = state;
    const labels = {
      connected: translate("conn.online", "Online"),
      connecting: translate("conn.connecting", "Connecting…"),
      disconnected: translate("conn.offline", "Offline"),
    };
    label.textContent = labels[state] ?? state;
    onWorkspaceTabChange();
  }

  function getToastHeading(type, message) {
    const lower = String(message || "").toLowerCase();
    if (type === "success") {
      return /(saved|loaded|opened|exported|copied|ready)/.test(lower)
        ? translate("uxp.toast.heading_ready", "Ready")
        : translate("uxp.toast.heading_done", "Done");
    }
    if (type === "warning") {
      return /(select|choose|enter|required)/.test(lower)
        ? translate("uxp.toast.heading_action_needed", "Action needed")
        : translate("uxp.toast.heading_heads_up", "Heads up");
    }
    if (type === "error") return translate("uxp.toast.heading_needs_attention", "Needs attention");
    if (/(step \d+\/\d+|installing|reinstalling|restarting|loading|checking|processing|transcribing|translating|burning|indexing)/.test(lower)) {
      return translate("uxp.toast.heading_in_progress", "In progress");
    }
    return translate("uxp.toast.heading_status_update", "Status update");
  }

  function getToastDuration(type, explicitDuration) {
    if (typeof explicitDuration === "number") return explicitDuration;
    if (type === "error") return 0;
    if (type === "warning") return 5600;
    return 4000;
  }

  function showToast(message, type = "info", duration) {
    const area = getElement("toastArea");
    if (!area) return;
    const maxVisibleToasts = 4;
    while (area.children.length >= maxVisibleToasts) area.firstElementChild?.remove();

    const payload = message && typeof message === "object" ? message : { message };
    const tone = payload.type || type || "info";
    const text = String(payload.message ?? payload.text ?? "").trim() || String(message ?? "").trim();
    const title = String(payload.title ?? getToastHeading(tone, text)).trim() || getToastHeading(tone, text);
    const detail = String(payload.detail ?? "").trim();
    const icons = {
      success: '<svg width="13" height="13" viewBox="0 0 16 16" fill="currentColor"><path d="M13.854 3.646a.5.5 0 010 .708l-7 7a.5.5 0 01-.708 0l-3.5-3.5a.5.5 0 11.708-.708L6.5 10.293l6.646-6.647a.5.5 0 01.708 0z"/></svg>',
      error: '<svg width="13" height="13" viewBox="0 0 16 16" fill="currentColor"><path d="M4.646 4.646a.5.5 0 000 .708L7.293 8l-2.647 2.646a.5.5 0 00.708-.708L8 8.707l2.646 2.647a.5.5 0 00-.708-.708L8 7.293 5.354 4.646a.5.5 0 00-.708 0z"/></svg>',
      warning: '<svg width="13" height="13" viewBox="0 0 16 16" fill="currentColor"><path d="M8 1a7 7 0 100 14A7 7 0 008 1zm0 3a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 018 4zm0 8a1 1 0 110-2 1 1 0 010 2z"/></svg>',
      info: '<svg width="13" height="13" viewBox="0 0 16 16" fill="currentColor"><path d="M8 1a7 7 0 100 14A7 7 0 008 1zm.93 6.588l-2.29.287-.082.38.45.083c.294.07.352.176.288.469l-.738 3.468c-.194.897.105 1.319.808 1.319.545 0 1.178-.252 1.465-.598l.088-.416c-.2.176-.492.246-.686.246-.275 0-.375-.193-.304-.533L8.93 7.588z"/></svg>',
    };
    const toast = documentRef.createElement("div");
    toast.className = "oc-toast " + tone;
    toast.dataset.state = tone;
    toast.setAttribute("role", tone === "error" ? "alert" : "status");
    toast.setAttribute("aria-live", tone === "error" ? "assertive" : "polite");
    toast.innerHTML = [
      '<span class="oc-toast-icon" aria-hidden="true">', icons[tone] ?? icons.info, "</span>",
      '<span class="oc-toast-content"><span class="oc-toast-title">', escapeHtmlValue(title),
      '</span><span class="oc-toast-msg">', escapeHtmlValue(text || title), "</span>",
      detail ? '<span class="oc-toast-detail">' + escapeHtmlValue(detail) + "</span>" : "",
      '</span><button type="button" class="oc-toast-dismiss" aria-label="',
      escapeHtmlValue(translate("uxp.toast.dismiss", "Dismiss notification")),
      '">&times;</button>',
    ].join("");
    area.appendChild(toast);

    const dismiss = () => {
      if (toast.dataset.closing === "true") return;
      toast.dataset.closing = "true";
      toast.classList.add("fade-out");
      schedule(() => toast.remove(), 320);
    };
    const dismissButton = toast.querySelector?.(".oc-toast-dismiss");
    listen(dismissButton, "click", dismiss);
    const explicitDuration = payload.duration ?? (arguments.length >= 3 ? duration : undefined);
    const resolvedDuration = getToastDuration(tone, explicitDuration);
    if (resolvedDuration > 0) schedule(dismiss, resolvedDuration);
  }

  function bindSlider(sliderId, valueId, formatter) {
    const slider = getElement(sliderId);
    const valueElement = getElement(valueId);
    if (!slider || !valueElement) return;
    const update = () => { valueElement.textContent = formatter(parseFloat(slider.value)); };
    listen(slider, "input", update);
    update();
  }

  function initCollapsibles() {
    documentRef?.querySelectorAll?.(".oc-card-header.collapsible").forEach((header) => {
      if (header.dataset.collapsibleBound === "true") return;
      const targetId = header.dataset.target;
      const initialBody = targetId ? getElement(targetId) : null;
      header.setAttribute("role", "button");
      header.tabIndex = 0;
      if (targetId) header.setAttribute("aria-controls", targetId);
      header.setAttribute("aria-expanded", initialBody?.classList.contains("collapsed") ? "false" : "true");
      const toggle = () => {
        const body = getElement(header.dataset.target);
        if (!body) return;
        const collapsed = body.classList.toggle("collapsed");
        header.classList.toggle("collapsed", collapsed);
        header.setAttribute("aria-expanded", collapsed ? "false" : "true");
      };
      listen(header, "click", toggle);
      listen(header, "keydown", (event) => {
        if (event.key !== "Enter" && event.key !== " ") return;
        event.preventDefault();
        toggle();
      });
      header.dataset.collapsibleBound = "true";
    });
  }

  function setButtonLoading(btnId, loading) {
    const button = getElement(btnId);
    if (!button) return;
    button.classList.toggle("loading", loading);
    const locked = button.dataset.backendLocked === "true" || button.dataset.jobLocked === "true";
    button.disabled = loading || locked;
    button.setAttribute("aria-disabled", button.disabled ? "true" : "false");
  }

  function clearButtonLoadingStates() {
    documentRef?.querySelectorAll?.("button.loading").forEach((button) => {
      button.classList.remove("loading");
      const locked = button.dataset.backendLocked === "true" || button.dataset.jobLocked === "true";
      button.disabled = locked;
      button.setAttribute("aria-disabled", button.disabled ? "true" : "false");
    });
  }

  function escapeHtml(value) {
    return escapeHtmlValue(value);
  }

  function dispose() {
    if (disposed) return;
    disposed = true;
    stopElapsedTimer();
    timeoutHandles.forEach((handle) => clearTimeoutImpl(handle));
    timeoutHandles.clear();
    cleanupCallbacks.splice(0).reverse().forEach((cleanup) => cleanup());
    navigationBound = false;
  }

  return {
    bindNavigation,
    syncTabOverflowControls,
    activateRelativeTab,
    switchTab,
    showProcessing,
    hideProcessing,
    setProcessingMsg,
    setProgress,
    setStatus,
    setStatusRight,
    setConnection,
    showToast,
    bindSlider,
    initCollapsibles,
    setButtonLoading,
    clearButtonLoadingStates,
    escapeHtml,
    dispose,
  };
}
