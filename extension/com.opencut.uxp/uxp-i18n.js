import {
  getLocaleCandidates,
  interpolateI18n,
  normalizeLocaleTag,
  translate,
} from "./uxp-utils.js";

/** Runtime i18n boundary shared by UXP bootstrap and feature modules. */
export function createI18nRuntime({
  defaultLocale = "en",
  localeDir = "locales",
  fetchJson,
  documentRef = typeof document !== "undefined" ? document : null,
  navigatorRef = typeof navigator !== "undefined" ? navigator : null,
  locationSearch = () => (
    typeof window !== "undefined" && window.location ? window.location.search : ""
  ),
  warn = (...args) => console.warn(...args),
} = {}) {
  if (typeof fetchJson !== "function") {
    throw new TypeError("createI18nRuntime requires fetchJson");
  }

  let currentLang = defaultLocale;
  let messages = {};

  const t = (key, fallback) => translate(messages, key, fallback);
  const format = (key, fallback, values = {}) => interpolateI18n(t(key, fallback), values);
  const formatCount = (count, oneKey, oneFallback, manyKey, manyFallback, values = {}) => (
    format(count === 1 ? oneKey : manyKey, count === 1 ? oneFallback : manyFallback, {
      count,
      ...values,
    })
  );

  function applyToDom(root = documentRef) {
    const scope = root || documentRef;
    if (!scope?.querySelectorAll) return;
    scope.querySelectorAll("[data-i18n]").forEach((node) => {
      const key = node.getAttribute("data-i18n");
      if (!key) return;
      const labelTarget = node.querySelector(".btn-label, .i18n-text");
      if (!node.hasAttribute("data-i18n-fallback")) {
        node.setAttribute("data-i18n-fallback", labelTarget ? labelTarget.textContent : node.textContent);
      }
      const translated = t(key, node.getAttribute("data-i18n-fallback") || "");
      if (labelTarget) labelTarget.textContent = translated;
      else node.textContent = translated;
    });

    const mappings = [
      ["data-i18n-title", "title"],
      ["data-i18n-label", "label"],
      ["data-i18n-alt", "alt"],
      ["data-i18n-placeholder", "placeholder"],
      ["data-i18n-aria-label", "aria-label"],
    ];
    mappings.forEach(([dataAttr, targetAttr]) => {
      scope.querySelectorAll(`[${dataAttr}]`).forEach((node) => {
        const key = node.getAttribute(dataAttr);
        if (!key) return;
        const fallbackAttr = `${dataAttr}-fallback`;
        if (!node.hasAttribute(fallbackAttr)) {
          node.setAttribute(fallbackAttr, node.getAttribute(targetAttr) || "");
        }
        node.setAttribute(targetAttr, t(key, node.getAttribute(fallbackAttr) || ""));
      });
    });
  }

  function getOverride() {
    try {
      const override = new URLSearchParams(locationSearch()).get("lang");
      return override ? normalizeLocaleTag(override, defaultLocale) : "";
    } catch (_) {
      return "";
    }
  }

  function getPreferredLocale() {
    const override = getOverride();
    if (override) return override;
    const languages = Array.isArray(navigatorRef?.languages) && navigatorRef.languages.length
      ? navigatorRef.languages
      : [navigatorRef?.language];
    return normalizeLocaleTag(languages.find(Boolean) || defaultLocale, defaultLocale);
  }

  async function load(lang = getPreferredLocale()) {
    const requested = normalizeLocaleTag(lang, defaultLocale);
    const base = await fetchJson(`${localeDir}/${defaultLocale}.json`) || {};
    let activeLang = defaultLocale;
    let active = {};
    for (const candidate of getLocaleCandidates(requested, defaultLocale)) {
      if (candidate === defaultLocale) {
        active = base;
        break;
      }
      const locale = await fetchJson(`${localeDir}/${candidate}.json`);
      if (locale) {
        active = locale;
        activeLang = candidate;
        break;
      }
    }
    messages = { ...base, ...active };
    currentLang = Object.keys(messages).length ? activeLang : defaultLocale;
    if (!Object.keys(messages).length) {
      warn("[OpenCut UXP] Locale files unavailable; using inline English fallbacks.");
    }
    if (documentRef?.documentElement) documentRef.documentElement.lang = currentLang;
    applyToDom();
    return currentLang;
  }

  return {
    t,
    format,
    formatCount,
    applyToDom,
    load,
    getPreferredLocale,
    get currentLang() { return currentLang; },
    get messages() { return { ...messages }; },
  };
}
