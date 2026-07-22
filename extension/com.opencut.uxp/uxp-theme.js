const THEME_CLASSES = Object.freeze([
  "theme-light",
  "theme-dark",
  "theme-darkest",
]);

export function normalizePremiereTheme(value) {
  const candidate = String(value?.theme ?? value?.detail ?? value ?? "")
    .trim()
    .toLowerCase();
  if (candidate.includes("light")) return "light";
  if (candidate.includes("darkest")) return "darkest";
  if (candidate.includes("dark")) return "dark";
  return "darkest";
}

export function applyPremiereTheme(value, documentRef = globalThis.document) {
  const root = documentRef?.documentElement;
  const theme = normalizePremiereTheme(value);
  if (!root?.classList) return theme;
  root.classList.remove(...THEME_CLASSES);
  root.classList.add(`theme-${theme}`);
  if (root.dataset) root.dataset.premiereTheme = theme;
  return theme;
}

export function createPremiereThemeSync({
  documentRef = globalThis.document,
  logger = globalThis.console,
} = {}) {
  let currentTheme = "darkest";
  let updateListener = null;

  const apply = (value) => {
    currentTheme = applyPremiereTheme(value, documentRef);
    return currentTheme;
  };

  const dispose = () => {
    const event = documentRef?.theme?.onUpdated;
    if (updateListener && typeof event?.removeListener === "function") {
      try {
        event.removeListener(updateListener);
      } catch (error) {
        logger?.warn?.("[OpenCut UXP] Could not remove host theme listener:", error);
      }
    }
    updateListener = null;
  };

  const start = () => {
    if (updateListener) return dispose;
    const themeApi = documentRef?.theme;
    try {
      apply(themeApi?.getCurrent?.() ?? "darkest");
    } catch (error) {
      logger?.warn?.("[OpenCut UXP] Could not read the host theme; using darkest.", error);
      apply("darkest");
    }

    if (typeof themeApi?.onUpdated?.addListener === "function") {
      updateListener = (theme) => apply(theme);
      themeApi.onUpdated.addListener(updateListener);
    }
    return dispose;
  };

  return {
    start,
    dispose,
    apply,
    get currentTheme() {
      return currentTheme;
    },
  };
}

export { THEME_CLASSES as PREMIERE_THEME_CLASSES };
