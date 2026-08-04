export function createUxpUpdateController({
  documentRef = globalThis.document,
  windowRef = globalThis.window || globalThis,
  storageRef = null,
  client,
  translate = (_key, fallback) => fallback,
  formatTranslate = (_key, fallback) => fallback,
  normalizeReleaseUrl = (value) => value,
  openExternalUrl = async () => false,
  setStatus = () => {},
  showToast = () => {},
  currentVersion = "",
} = {}) {
  let latestUpdate = null;
  let checkDone = false;
  let checkFailed = false;
  let checkInFlight = null;
  let disposed = false;
  const cleanupCallbacks = [];
  const dismissedVersionKey = "opencut_update_dismissed_version";

  function getElement(id) {
    return documentRef?.getElementById?.(id) || null;
  }

  function getStorage() {
    if (storageRef) return storageRef;
    try {
      return windowRef?.localStorage || null;
    } catch (_) {
      return null;
    }
  }

  function listen(target, type, listener) {
    if (!target?.addEventListener) return;
    target.addEventListener(type, listener);
    cleanupCallbacks.push(() => target.removeEventListener?.(type, listener));
  }

  function getDismissedVersion() {
    try {
      return getStorage()?.getItem(dismissedVersionKey) || "";
    } catch (_) {
      return "";
    }
  }

  function setDismissedVersion(version) {
    try {
      getStorage()?.setItem(dismissedVersionKey, String(version || ""));
    } catch (_) {
      // UXP storage can be unavailable during host startup.
    }
  }

  function setUpdateHidden(id, hidden) {
    const node = getElement(id);
    if (node) node.hidden = Boolean(hidden);
  }

  function formatPublishedAt(value) {
    if (!value) return "";
    try {
      const date = new Date(value);
      if (!Number.isNaN(date.getTime())) return date.toLocaleDateString();
    } catch (_) {
      // Fall through to the server value if the host cannot parse the date.
    }
    return String(value);
  }

  function renderNotice(result, checking = false) {
    if (disposed) return;
    const card = getElement("uxpUpdateNoticeCard");
    const status = getElement("uxpUpdateStatusText");
    const summary = getElement("uxpUpdateSummary");
    const current = getElement("uxpUpdateCurrentVersion");
    const available = getElement("uxpUpdateAvailableVersion");
    const releaseName = getElement("uxpUpdateReleaseName");
    const notes = getElement("uxpUpdateReleaseNotes");
    const notesDetails = getElement("uxpUpdateNotesDetails");
    const retry = getElement("uxpUpdateRetryBtn");
    if (!card || !summary) return;

    const currentVersionText = String(result?.current_version || current?.textContent || currentVersion || "—");
    if (current) current.textContent = currentVersionText;
    if (retry) retry.disabled = Boolean(checking);

    if (checking) {
      card.dataset.state = "checking";
      if (status) {
        status.dataset.state = "working";
        status.textContent = translate("uxp.settings.update_checking", "Checking…");
      }
      summary.textContent = translate(
        "uxp.settings.update_checking_summary",
        "Checking GitHub for the latest OpenCut release.",
      );
      setUpdateHidden("uxpUpdateReleaseDetails", true);
      return;
    }

    if (!result || result.error || !result.latest_version) {
      card.dataset.state = "error";
      if (status) {
        status.dataset.state = "error";
        status.textContent = translate("uxp.settings.update_unavailable_status", "Unavailable");
      }
      summary.textContent = translate(
        "uxp.settings.update_check_failed",
        "Couldn't check for updates. Use Check again to retry.",
      );
      if (available) available.textContent = "—";
      setUpdateHidden("uxpUpdateReleaseDetails", true);
      return;
    }

    const latestVersion = String(result.latest_version);
    if (available) available.textContent = latestVersion;
    if (!result.update_available) {
      card.dataset.state = "current";
      if (status) {
        status.dataset.state = "success";
        status.textContent = translate("uxp.settings.update_current_status", "Up to date");
      }
      summary.textContent = formatTranslate(
        "uxp.settings.update_up_to_date",
        "You're up to date on v{version}.",
        { version: currentVersionText },
      );
      setUpdateHidden("uxpUpdateReleaseDetails", true);
      return;
    }

    if (getDismissedVersion() === latestVersion) {
      card.dataset.state = "dismissed";
      if (status) {
        status.dataset.state = "neutral";
        status.textContent = translate("uxp.settings.update_dismissed_status", "Dismissed");
      }
      summary.textContent = formatTranslate(
        "uxp.settings.update_dismissed",
        "Update v{version} is dismissed for this panel. A newer release will appear here.",
        { version: latestVersion },
      );
      setUpdateHidden("uxpUpdateReleaseDetails", true);
      return;
    }

    card.dataset.state = "available";
    if (status) {
      status.dataset.state = "warning";
      status.textContent = translate("uxp.settings.update_available_status", "Update available");
    }
    let availableSummary = formatTranslate(
      "uxp.settings.update_available_summary",
      "OpenCut v{version} is available. Review the release notes before opening GitHub.",
      { version: latestVersion },
    );
    const published = formatPublishedAt(result.published_at);
    if (published) {
      availableSummary += " " + formatTranslate(
        "uxp.settings.update_published_at",
        "Published {date}.",
        { date: published },
      );
    }
    summary.textContent = availableSummary;
    if (releaseName) releaseName.textContent = result.release_name || "OpenCut " + latestVersion;
    if (notes) notes.textContent = result.release_notes || translate(
      "uxp.settings.update_no_release_notes",
      "No release notes were published.",
    );
    if (notesDetails) notesDetails.hidden = !result.release_notes;
    setUpdateHidden("uxpUpdateReleaseDetails", false);
  }

  async function openRelease() {
    const releaseUrl = normalizeReleaseUrl(latestUpdate?.release_url);
    if (!releaseUrl) {
      showToast(translate("uxp.settings.update_invalid_release", "The release link could not be verified."), "error");
      return false;
    }
    const opened = await openExternalUrl(
      releaseUrl,
      translate("uxp.settings.update_opening_release", "Opening the verified release page in your browser"),
    );
    if (opened) showToast(translate("uxp.settings.update_opened", "Release page opened."), "success");
    return opened;
  }

  function dismissNotice() {
    if (!latestUpdate?.latest_version) return;
    const version = String(latestUpdate.latest_version);
    setDismissedVersion(version);
    renderNotice(latestUpdate);
    showToast(formatTranslate(
      "uxp.settings.update_dismissed_toast",
      "Update v{version} dismissed until a newer release is available.",
      { version },
    ), "info");
  }

  async function checkForUpdates({ force = false } = {}) {
    if (disposed) return false;
    if (!force && (checkDone || checkFailed)) return checkDone;
    if (checkInFlight) return checkInFlight;

    checkInFlight = (async () => {
      renderNotice(null, true);
      setStatus(translate("uxp.status.update_checking", "Checking for updates..."), "working");
      let result;
      try {
        result = await client?.get?.("/system/update-check");
      } catch (error) {
        result = { ok: false, error: error?.message || "offline", data: null };
      }
      if (disposed) return false;

      const data = result?.data;
      if (!result?.ok || !data || data.error || !data.latest_version) {
        checkDone = false;
        checkFailed = true;
        const message = translate(
          "uxp.status.update_check_failed",
          "Couldn't check for updates. Use Refresh to try again.",
        );
        setStatus(message, "error");
        showToast({
          title: translate("uxp.status.update_check_failed_title", "Couldn't check for updates"),
          message,
          type: "error",
          duration: 0,
        });
        renderNotice(data || { error: "offline" });
        return false;
      }

      checkDone = true;
      checkFailed = false;
      latestUpdate = data;
      renderNotice(data);
      setStatus(translate("uxp.status.backend_connected", "OpenCut backend connected."), "success");
      if (data.update_available && getDismissedVersion() !== String(data.latest_version)) {
        showToast(
          formatTranslate(
            "uxp.status.update_available",
            "OpenCut v{version} available - visit GitHub to update",
            { version: data.latest_version },
          ),
          "info",
          6000,
        );
      }
      return true;
    })().finally(() => {
      checkInFlight = null;
    });

    return checkInFlight;
  }

  function bind({ onRefresh = async () => {} } = {}) {
    listen(getElement("refreshBtn"), "click", () => onRefresh());
    listen(getElement("uxpUpdateRetryBtn"), "click", () => checkForUpdates({ force: true }));
    listen(getElement("uxpUpdateOpenBtn"), "click", () => openRelease());
    listen(getElement("uxpUpdateDismissBtn"), "click", () => dismissNotice());
  }

  function dispose() {
    if (disposed) return;
    disposed = true;
    cleanupCallbacks.splice(0).reverse().forEach((cleanup) => cleanup());
  }

  return {
    bind,
    checkForUpdates,
    renderNotice,
    openRelease,
    dismissNotice,
    getLatestUpdate: () => latestUpdate,
    dispose,
  };
}
