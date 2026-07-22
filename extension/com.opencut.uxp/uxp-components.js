/** DOM-safe component helpers used by both the controller and focused tests. */
export function escapeHtml(value) {
  const div = typeof document !== "undefined" ? document.createElement("div") : null;
  if (!div) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
  div.textContent = String(value ?? "");
  return div.innerHTML;
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
