/** DOM-safe component helpers used by both the controller and focused tests. */
export function escapeHtml(value) {
  // Always escape via the full replacement chain. The previous DOM-based
  // branch (div.textContent -> innerHTML) left quotes unescaped, which is
  // unsafe when callers interpolate the result into quoted HTML attributes
  // (e.g. loadJournalRecoveryUxp's data-transaction-id="...").
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

export function safeDomIdSegment(value) {
  return String(value ?? "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "-")
    .replace(/^-+|-+$/g, "") || "item";
}

export function setButtonBusy(button, busy, busyLabel = "Working…") {
  if (!button) return "";
  const label = button.querySelector?.(".btn-label");
  const target = label || button;
  if (!button.dataset.defaultLabel) button.dataset.defaultLabel = target.textContent || "";
  target.textContent = busy ? busyLabel : button.dataset.defaultLabel;
  button.classList.toggle("loading", Boolean(busy));
  button.disabled = Boolean(busy);
  button.setAttribute("aria-busy", busy ? "true" : "false");
  return target.textContent;
}

export function setElementHidden(element, hidden) {
  if (!element) return;
  element.classList.toggle("oc-hidden", Boolean(hidden));
  element.setAttribute("aria-hidden", hidden ? "true" : "false");
}
