import { createUxpGpuSelectionController } from "./uxp-gpu-selection-controller.js";

export function createUxpSettingsController({
  documentRef = globalThis.document,
  windowRef = globalThis.window || globalThis,
  requestAnimationFrameFn = globalThis.requestAnimationFrame,
  client,
  translate = (_key, fallback) => fallback,
  formatTranslate = (_key, fallback) => fallback,
  showToast = () => {},
  getLocalFileSystem = async () => null,
  isBackendConnected: isBackendConnectedFn = () => false,
  openExternalUrl = async () => false,
  onWorkspaceAction = () => {},
} = {}) {
  const document = documentRef;
  const window = windowRef;
  const requestAnimationFrame = requestAnimationFrameFn || ((callback) => callback());
  const BackendClient = client;
  const UIController = { showToast };
  const getUxpLocalFileSystem = getLocalFileSystem;
  const openHttpsExternalUrl = openExternalUrl;
  const isBackendConnected = isBackendConnectedFn;
  const handleWorkspaceAction = onWorkspaceAction;
  const t = translate;
  const formatI18n = formatTranslate;
  const GpuSelectionController = createUxpGpuSelectionController({
    documentRef: document,
    client: BackendClient,
    translate: t,
    formatTranslate: formatI18n,
    showToast,
  });
  const cleanupCallbacks = [];
  let settingsNavigationBound = false;
  let settingsIOBound = false;
  let supportIOBound = false;
  let onboardingBound = false;
  let disposed = false;

  function listen(target, type, listener, options) {
    if (disposed || !target?.addEventListener) return;
    target.addEventListener(type, listener, options);
    cleanupCallbacks.push(() => target.removeEventListener?.(type, listener, options));
  }
  function initSettingsNavigation() {
    if (settingsNavigationBound || disposed) return;
    const nav = document.querySelector(".oc-settings-nav");
    if (!nav) return;
  
    const buttons = Array.from(nav.querySelectorAll(".oc-settings-nav-item"));
    const panes = Array.from(document.querySelectorAll("#tab-settings [data-settings-pane]"));
    if (!buttons.length || !panes.length) return;
    settingsNavigationBound = true;
  
    const activate = (section, { focus = false } = {}) => {
      const nextButton = buttons.find((button) => button.dataset.settingsSection === section) || buttons[0];
      const activeSection = nextButton.dataset.settingsSection;
  
      buttons.forEach((button) => {
        const selected = button === nextButton;
        button.classList.toggle("active", selected);
        button.setAttribute("aria-selected", selected ? "true" : "false");
        button.tabIndex = selected ? 0 : -1;
      });
  
      panes.forEach((pane) => {
        const selected = pane.dataset.settingsPane === activeSection;
        pane.classList.toggle("active", selected);
        pane.hidden = !selected;
      });
  
      if (typeof nextButton.scrollIntoView === "function") {
        nextButton.scrollIntoView({ block: "nearest", inline: "nearest" });
      }
  
      if (focus) nextButton.focus();
    };
  
    listen(nav, "click", (event) => {
      const button = event.target.closest(".oc-settings-nav-item");
      if (button) activate(button.dataset.settingsSection);
    });
  
    listen(nav, "keydown", (event) => {
      const currentIndex = buttons.indexOf(document.activeElement);
      if (currentIndex < 0) return;
  
      let nextIndex = currentIndex;
      if (event.key === "ArrowDown" || event.key === "ArrowRight") nextIndex = (currentIndex + 1) % buttons.length;
      else if (event.key === "ArrowUp" || event.key === "ArrowLeft") nextIndex = (currentIndex - 1 + buttons.length) % buttons.length;
      else if (event.key === "Home") nextIndex = 0;
      else if (event.key === "End") nextIndex = buttons.length - 1;
      else return;
  
      event.preventDefault();
      activate(buttons[nextIndex].dataset.settingsSection, { focus: true });
    });
  
    activate(buttons.find((button) => button.classList.contains("active"))?.dataset.settingsSection || "workspace");
  }
  
  function describeSettingsImport(result) {
    const imported = Array.isArray(result?.imported) ? result.imported : [];
    let skippedCount = 0;
    let reasons = [];
    for (const section of ["presets", "favorites", "workflows"]) {
      const details = result?.[section];
      if (!details?.skipped) continue;
      skippedCount += Number(details.skipped) || 0;
      if (Array.isArray(details.reasons)) reasons = reasons.concat(details.reasons);
    }
    const items = imported.join(", ") || t("uxp.settings.settings_none_imported", "none");
    if (skippedCount > 0) {
      let reasonText = reasons.slice(0, 3).join("; ");
      if (reasons.length > 3) reasonText += "…";
      return {
        message: formatI18n(
          "uxp.settings.settings_import_skipped",
          "Settings imported: {items}. Skipped {count} item(s){reasons}",
          { items, count: skippedCount, reasons: reasonText ? `: ${reasonText}` : "" },
        ),
        type: "warning",
      };
    }
    return {
      message: formatI18n("uxp.settings.settings_imported", "Settings imported: {items}.", { items }),
      type: "success",
    };
  }
  
  async function initSettingsIO() {
    if (settingsIOBound || disposed) return;
    GpuSelectionController.bind();
    const exportButton = document.getElementById("uxpExportSettingsBtn");
    const importButton = document.getElementById("uxpImportSettingsBtn");
    const status = document.getElementById("uxpSettingsIOStatus");
    if (!exportButton && !importButton) return;
    settingsIOBound = true;
  
    const setBusy = (busy) => {
      for (const button of [exportButton, importButton]) {
        if (button) button.disabled = busy;
      }
    };
    const responseData = (response) => response?.data ?? response ?? {};
    const responseError = (response) => response?.error || responseData(response)?.error || t("common.unknown", "unknown");
  
    listen(exportButton, "click", async () => {
      setBusy(true);
      try {
        const response = await BackendClient.get("/settings/export");
        if (!response?.ok) throw new Error(responseError(response));
        const bundle = { ...responseData(response) };
        try {
          bundle.localStorage = JSON.parse(window.localStorage?.getItem("opencut_settings") || "{}");
        } catch (_) { /* local panel state is optional */ }
        const localFileSystem = await getUxpLocalFileSystem();
        if (!localFileSystem) throw new Error(t("uxp.settings.settings_file_api_unavailable", "UXP file storage is unavailable."));
        const date = new Date().toISOString().slice(0, 10);
        const folder = await localFileSystem.getFolder();
        if (!folder) return;
        const file = await folder.createFile(`opencut_settings_${date}.json`, { overwrite: true });
        await file.write(JSON.stringify(bundle, null, 2));
        if (status) status.textContent = t("uxp.settings.settings_exported", "Settings exported.");
        UIController.showToast(t("uxp.settings.settings_exported", "Settings exported."), "success");
      } catch (error) {
        if (status) status.textContent = formatI18n("uxp.settings.settings_export_failed", "Settings export failed: {error}", { error: error?.message || error });
        UIController.showToast(formatI18n("uxp.settings.settings_export_failed", "Settings export failed: {error}", { error: error?.message || error }), "error");
      } finally {
        setBusy(false);
      }
    });
  
    listen(importButton, "click", async () => {
      setBusy(true);
      try {
        const localFileSystem = await getUxpLocalFileSystem();
        if (!localFileSystem) throw new Error(t("uxp.settings.settings_file_api_unavailable", "UXP file storage is unavailable."));
        const file = await localFileSystem.getFileForOpening({ allowMultiple: false, types: ["json"] });
        if (!file) return;
        const raw = await file.read();
        const bundle = JSON.parse(typeof raw === "string" ? raw : String(raw));
        if (!bundle || typeof bundle !== "object" || Array.isArray(bundle)) {
          throw new Error(t("uxp.settings.settings_import_invalid", "This file does not contain valid OpenCut settings."));
        }
        const response = await BackendClient.post("/settings/import", bundle);
        if (!response?.ok) throw new Error(responseError(response));
        if (bundle.localStorage) {
          try {
            window.localStorage?.setItem("opencut_settings", JSON.stringify(bundle.localStorage));
          } catch (_) {
            UIController.showToast(t("uxp.settings.settings_import_local_failed", "Settings imported, but local panel preferences could not be saved."), "warning");
          }
        }
        const summary = describeSettingsImport(responseData(response));
        if (status) status.textContent = summary.message;
        UIController.showToast(summary.message, summary.type);
      } catch (error) {
        const message = error?.message || String(error);
        if (status) status.textContent = message;
        UIController.showToast(message, "error");
      } finally {
        setBusy(false);
      }
    });
  }
  
  async function initSupportIO() {
    if (supportIOBound || disposed) return;
    const exportButton = document.getElementById("uxpExportSupportBundleBtn");
    const issueButton = document.getElementById("uxpOpenIssueReportBtn");
    const restartButton = document.getElementById("uxpRestartOnboardingBtn");
    const status = document.getElementById("uxpSupportStatus");
    if (!exportButton && !issueButton && !restartButton) return;
    supportIOBound = true;
  
    const setBusy = (busy) => {
      for (const button of [exportButton, issueButton, restartButton]) {
        if (button) button.disabled = busy;
      }
    };
    const setStatus = (message, state = "") => {
      if (!status) return;
      status.textContent = message;
      if (state) status.dataset.state = state;
      else delete status.dataset.state;
    };
    const responseData = (response) => response?.data ?? response ?? {};
    const responseError = (response, fallback) => (
      response?.error || responseData(response)?.error || fallback
    );
  
    const requestSupportBundle = async () => {
      const response = await BackendClient.post("/system/support-bundle", {
        title: "OpenCut UXP support bundle",
        description: "UXP panel support bundle requested from Settings.",
        include_crash: true,
        include_logs: true,
        log_tail_lines: 200,
      });
      if (!response?.ok) {
        throw new Error(responseError(
          response,
          t("uxp.settings.support_bundle_unavailable", "The local support bundle could not be created."),
        ));
      }
      const data = responseData(response);
      if (data.redacted !== true || data.kind !== "support_bundle") {
        throw new Error(t(
          "uxp.settings.support_bundle_invalid",
          "The backend returned an invalid support bundle. Nothing was exported.",
        ));
      }
      return data;
    };
  
    listen(exportButton, "click", async () => {
      setBusy(true);
      try {
        const bundle = await requestSupportBundle();
        const localFileSystem = await getUxpLocalFileSystem();
        if (!localFileSystem) {
          throw new Error(t(
            "uxp.settings.support_file_api_unavailable",
            "UXP file storage is unavailable. Copy the report from the issue workflow instead.",
          ));
        }
        const folder = await localFileSystem.getFolder();
        if (!folder) return;
        const date = new Date().toISOString().slice(0, 10);
        const filename = `opencut_support_bundle_${date}.json`;
        const file = await folder.createFile(filename, { overwrite: true });
        await file.write(JSON.stringify({
          schema_version: "opencut.support_bundle.v1",
          kind: bundle.kind,
          redacted: true,
          title: bundle.title,
          body: bundle.body,
          size_bytes: bundle.size_bytes,
        }, null, 2));
        const message = formatI18n(
          "uxp.settings.support_bundle_exported",
          "Redacted support bundle exported as {filename}.",
          { filename },
        );
        setStatus(message, "success");
        UIController.showToast(message, "success");
      } catch (error) {
        const message = formatI18n(
          "uxp.settings.support_bundle_failed",
          "Support bundle export failed: {error}",
          { error: error?.message || error },
        );
        setStatus(message, "error");
        UIController.showToast(message, "error");
      } finally {
        setBusy(false);
      }
    });
  
    listen(issueButton, "click", async () => {
      setBusy(true);
      try {
        const response = await BackendClient.post("/system/issue-report/bundle", {
          title: "OpenCut UXP issue report",
          description: "Issue report opened from the UXP Settings panel.",
          include_crash: true,
          include_logs: true,
          log_tail_lines: 200,
        });
        if (!response?.ok) {
          throw new Error(responseError(
            response,
            t(
              "uxp.settings.issue_report_network_required",
              "Opening an issue report requires network access. Export a local support bundle instead.",
            ),
          ));
        }
        const data = responseData(response);
        if (!data.url) {
          throw new Error(t(
            "uxp.settings.issue_report_failed",
            "The issue report URL was not returned. Export a local support bundle instead.",
          ));
        }
        const opened = await openHttpsExternalUrl(
          data.url,
          t("uxp.settings.issue_report_opening", "Opening a reviewed OpenCut issue report"),
        );
        if (opened) {
          const message = t("uxp.settings.issue_report_opened", "Issue report opened for review.");
          setStatus(message, "success");
          UIController.showToast(message, "success");
        }
      } catch (error) {
        const message = formatI18n(
          "uxp.settings.issue_report_failed",
          "Issue report could not be opened: {error}",
          { error: error?.message || error },
        );
        setStatus(message, "error");
        UIController.showToast(message, "error");
      } finally {
        setBusy(false);
      }
    });
  
    listen(restartButton, "click", async () => {
      setBusy(true);
      try {
        await restartUxpOnboarding();
        setStatus(t("uxp.settings.onboarding_restarted", "Getting Started is ready to review."), "success");
      } catch (error) {
        const message = formatI18n(
          "uxp.settings.onboarding_restart_failed",
          "Getting Started could not be opened: {error}",
          { error: error?.message || error },
        );
        setStatus(message, "error");
        UIController.showToast(message, "error");
      } finally {
        setBusy(false);
      }
    });
  }
  
  const UXP_ONBOARDING_STEPS = Object.freeze([
    {
      titleKey: "uxp.onboarding.step_welcome_title",
      bodyKey: "uxp.onboarding.step_welcome_body",
    },
    {
      titleKey: "uxp.onboarding.step_media_title",
      bodyKey: "uxp.onboarding.step_media_body",
      action: "choose-clip",
      actionKey: "uxp.onboarding.action_choose_media",
    },
    {
      titleKey: "uxp.onboarding.step_cut_title",
      bodyKey: "uxp.onboarding.step_cut_body",
      action: "switch-cut",
      actionKey: "uxp.onboarding.action_open_cut",
    },
    {
      titleKey: "uxp.onboarding.step_review_title",
      bodyKey: "uxp.onboarding.step_review_body",
      action: "open-timeline",
      actionKey: "uxp.onboarding.action_open_timeline",
    },
  ]);
  
  let uxpOnboardingIndex = 0;
  let uxpOnboardingReturnFocus = null;
  
  function readUxpOnboardingLocalState() {
    try {
      const raw = window.localStorage?.getItem("opencut_uxp_onboarding");
      const parsed = raw ? JSON.parse(raw) : {};
      return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
    } catch (_) {
      return {};
    }
  }
  
  function writeUxpOnboardingLocalState(patch) {
    try {
      window.localStorage?.setItem(
        "opencut_uxp_onboarding",
        JSON.stringify({ ...readUxpOnboardingLocalState(), ...patch }),
      );
    } catch (_) {
      // Local state is only a fallback; the backend remains authoritative.
    }
  }
  
  async function persistUxpOnboarding(patch) {
    writeUxpOnboardingLocalState(patch);
    if (!isBackendConnected()) return false;
    const response = await BackendClient.post("/settings/onboarding", patch);
    if (!response?.ok) {
      console.warn("[OpenCut UXP] Could not persist onboarding state", response?.error);
    }
    return !!response?.ok;
  }
  
  function renderUxpOnboarding() {
    const step = UXP_ONBOARDING_STEPS[uxpOnboardingIndex] || UXP_ONBOARDING_STEPS[0];
    const title = document.getElementById("uxpOnboardingTitle");
    const body = document.getElementById("uxpOnboardingBody");
    const stepLabel = document.getElementById("uxpOnboardingStep");
    const action = document.getElementById("uxpOnboardingActionBtn");
    const back = document.getElementById("uxpOnboardingBackBtn");
    const next = document.getElementById("uxpOnboardingNextBtn");
  
    if (title) title.textContent = t(step.titleKey, "Getting started");
    if (body) body.textContent = t(step.bodyKey, "OpenCut is ready to help you move from media to a first edit.");
    if (stepLabel) {
      stepLabel.textContent = formatI18n(
        "uxp.onboarding.step_count",
        "Step {current} of {total}",
        { current: uxpOnboardingIndex + 1, total: UXP_ONBOARDING_STEPS.length },
      );
    }
    if (action) {
      action.hidden = !step.action;
      action.dataset.action = step.action || "";
      action.textContent = step.action
        ? t(step.actionKey, "Open next step")
        : "";
    }
    if (back) {
      back.disabled = uxpOnboardingIndex === 0;
      back.setAttribute("aria-disabled", back.disabled ? "true" : "false");
    }
    if (next) {
      next.textContent = uxpOnboardingIndex === UXP_ONBOARDING_STEPS.length - 1
        ? t("uxp.onboarding.finish", "Finish")
        : t("uxp.onboarding.next", "Next");
    }
  }
  
  function closeUxpOnboarding() {
    const overlay = document.getElementById("uxpOnboardingOverlay");
    if (!overlay) return;
    overlay.hidden = true;
    overlay.setAttribute("aria-hidden", "true");
    document.body.classList.remove("oc-onboarding-open");
    if (uxpOnboardingReturnFocus && typeof uxpOnboardingReturnFocus.focus === "function") {
      uxpOnboardingReturnFocus.focus();
    }
    uxpOnboardingReturnFocus = null;
  }
  
  function showUxpOnboarding(step = 0, returnFocus = document.activeElement) {
    if (!isBackendConnected()) {
      UIController.showToast(
        t("uxp.onboarding.unavailable", "Connect the OpenCut backend before opening Getting Started."),
        "warning",
      );
      return false;
    }
    const overlay = document.getElementById("uxpOnboardingOverlay");
    if (!overlay) return false;
    uxpOnboardingIndex = Math.max(0, Math.min(UXP_ONBOARDING_STEPS.length - 1, Number(step) || 0));
    uxpOnboardingReturnFocus = returnFocus && returnFocus !== document.body ? returnFocus : null;
    renderUxpOnboarding();
    overlay.hidden = false;
    overlay.setAttribute("aria-hidden", "false");
    document.body.classList.add("oc-onboarding-open");
    requestAnimationFrame(() => {
      document.getElementById("uxpOnboardingActionBtn")?.hidden
        ? document.getElementById("uxpOnboardingNextBtn")?.focus()
        : document.getElementById("uxpOnboardingActionBtn")?.focus();
    });
    return true;
  }
  
  async function loadUxpOnboarding() {
    if (!isBackendConnected()) return;
    await GpuSelectionController.load();
    const response = await BackendClient.get("/settings/onboarding");
    if (!response?.ok) return;
    const state = response.data || {};
    if (state.seen === true) {
      writeUxpOnboardingLocalState({ seen: true, step: state.step || 0 });
      return;
    }
    const localState = readUxpOnboardingLocalState();
    showUxpOnboarding(
      Number.isFinite(Number(state.step)) ? Number(state.step) : Number(localState.step) || 0,
    );
  }
  
  async function restartUxpOnboarding() {
    if (!isBackendConnected()) {
      throw new Error(t(
        "uxp.onboarding.unavailable",
        "Connect the OpenCut backend before opening Getting Started.",
      ));
    }
    const response = await BackendClient.post("/settings/onboarding", { seen: false, step: 0 });
    if (!response?.ok) {
      throw new Error(response?.error || t("common.unknown", "unknown"));
    }
    writeUxpOnboardingLocalState({ seen: false, step: 0 });
    showUxpOnboarding(0, document.getElementById("uxpRestartOnboardingBtn"));
  }
  
  function initUxpOnboardingEvents() {
    if (onboardingBound || disposed) return;
    const overlay = document.getElementById("uxpOnboardingOverlay");
    if (!overlay) return;
    onboardingBound = true;
    listen(document.getElementById("uxpOnboardingActionBtn"), "click", async (event) => {
      const action = event.currentTarget?.dataset?.action || "";
      await persistUxpOnboarding({ seen: false, step: uxpOnboardingIndex });
      closeUxpOnboarding();
      if (action) handleWorkspaceAction(action);
    });
    listen(document.getElementById("uxpOnboardingBackBtn"), "click", async () => {
      if (uxpOnboardingIndex <= 0) return;
      uxpOnboardingIndex -= 1;
      await persistUxpOnboarding({ seen: false, step: uxpOnboardingIndex });
      renderUxpOnboarding();
    });
    listen(document.getElementById("uxpOnboardingSkipBtn"), "click", async () => {
      await persistUxpOnboarding({ seen: true, step: uxpOnboardingIndex });
      closeUxpOnboarding();
    });
    listen(document.getElementById("uxpOnboardingNextBtn"), "click", async () => {
      if (uxpOnboardingIndex >= UXP_ONBOARDING_STEPS.length - 1) {
        await persistUxpOnboarding({ seen: true, step: uxpOnboardingIndex });
        closeUxpOnboarding();
        UIController.showToast(t("uxp.onboarding.ready", "You are ready to make your first edit."), "success");
        return;
      }
      uxpOnboardingIndex += 1;
      await persistUxpOnboarding({ seen: false, step: uxpOnboardingIndex });
      renderUxpOnboarding();
    });
    listen(overlay, "click", (event) => {
      if (event.target === overlay) closeUxpOnboarding();
    });
    listen(document, "keydown", (event) => {
      if (overlay.hidden || event.key !== "Escape") return;
      event.preventDefault();
      closeUxpOnboarding();
    });
  }
  
  function dispose() {
    if (disposed) return;
    disposed = true;
    GpuSelectionController.dispose();
    cleanupCallbacks.splice(0).reverse().forEach((cleanup) => cleanup());
  }

  return {
    initSettingsNavigation,
    initSettingsIO,
    initSupportIO,
    initOnboardingEvents: initUxpOnboardingEvents,
    loadOnboarding: loadUxpOnboarding,
    restartOnboarding: restartUxpOnboarding,
    showOnboarding: showUxpOnboarding,
    dispose,
  };
}
