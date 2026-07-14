import { expect, test } from "@playwright/test";

const THEMES = ["dark", "light", "auto"];
const SURFACES = {
  cep: {
    url: "/extension/com.opencut.panel/client/index.html",
    tabSelector: ".nav-tab",
    activeTabSelector: ".nav-tab.active",
    activePanelSelector: ".nav-panel.active",
    tabAttribute: "data-nav",
    widths: [480, 900, 1200],
  },
  uxp: {
    url: "/extension/com.opencut.uxp/index.html",
    tabSelector: ".oc-tab",
    activeTabSelector: ".oc-tab.active",
    activePanelSelector: ".oc-tab-panel.active",
    tabAttribute: "data-tab",
    widths: [480, 520, 1200],
  },
};

function hostEnvironment() {
  return JSON.stringify({
    appId: "PPRO",
    appName: "Premiere Pro",
    appVersion: "26.3.0",
    appLocale: "en_US",
    appUILocale: "en_US",
    appSkinInfo: {
      baseFontFamily: "Arial",
      baseFontSize: 12,
      panelBackgroundColor: {
        color: { red: 24, green: 25, blue: 28, alpha: 255 },
      },
    },
  });
}

async function preparePage(page, surface, theme) {
  const pageErrors = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.emulateMedia({ colorScheme: theme === "auto" ? "light" : theme });
  await page.addInitScript(
    ({ surfaceName, selectedTheme, environment }) => {
      localStorage.clear();
      localStorage.setItem("opencut_debug", "0");
      if (surfaceName === "cep") {
        localStorage.setItem(
          "opencut_settings",
          JSON.stringify({
            theme: selectedTheme,
            wizardDismissed: true,
          }),
        );
        const callbacks = new Map();
        window.__adobe_cep__ = new Proxy(
          {
            getHostEnvironment: () => environment,
            getHostCapabilities: () =>
              JSON.stringify({ EXTENDED_PANEL_MENU: true }),
            getSystemPath: () => "C:/OpenCut/fixture",
            getExtensionId: () => "com.opencut.panel",
            getScaleFactor: () => 1,
            getMonitorScaleFactor: () => 1,
            getCurrentApiVersion: () =>
              JSON.stringify({ major: 13, minor: 0, micro: 0 }),
            evalScript: (script, callback) => {
              const result =
                /oc(GetProjectBins|GetSequenceMarkers|GetProjectClips)/.test(
                  script,
                )
                  ? "[]"
                  : "{}";
              if (typeof callback === "function")
                queueMicrotask(() => callback(result));
              return result;
            },
            addEventListener: (type, listener) => callbacks.set(type, listener),
            removeEventListener: (type) => callbacks.delete(type),
            invokeSync: () => "",
            invokeAsync: (_name, _payload, callback) => {
              if (callback) callback("");
            },
          },
          {
            get(target, property) {
              if (property in target) return target[property];
              return () => "";
            },
          },
        );
      }
      window.WebSocket = class RenderedWebSocket {
        static OPEN = 1;
        static CLOSED = 3;
        constructor() {
          this.readyState = RenderedWebSocket.CLOSED;
        }
        addEventListener() {}
        removeEventListener() {}
        close() {
          this.readyState = RenderedWebSocket.CLOSED;
        }
        send() {}
      };
      window.EventSource = class RenderedEventSource {
        constructor() {
          this.readyState = 2;
        }
        addEventListener() {}
        close() {}
      };
    },
    {
      surfaceName: surface,
      selectedTheme: theme,
      environment: hostEnvironment(),
    },
  );
  await page.route("http://127.0.0.1:*/**", async (route) => {
    const url = new URL(route.request().url());
    if (url.port === "41737") return route.continue();
    return route.fulfill({
      status: 503,
      contentType: "application/json",
      body: JSON.stringify({
        error: "Rendered fixture: backend offline",
        code: "BACKEND_OFFLINE",
        suggestion: "Start the local OpenCut backend.",
      }),
    });
  });
  return pageErrors;
}

async function openSurface(page, surfaceName, theme, width) {
  const surface = SURFACES[surfaceName];
  await page.setViewportSize({ width, height: 900 });
  const pageErrors = await preparePage(page, surfaceName, theme);
  await page.goto(surface.url, { waitUntil: "domcontentloaded" });
  await page.addStyleTag({
    content: `
    *, *::before, *::after {
      animation: none !important;
      transition: none !important;
      caret-color: transparent !important;
    }
  `,
  });
  await expect(page.locator(surface.tabSelector).first()).toBeVisible();
  await page.waitForTimeout(150);
  return { surface, pageErrors };
}

async function visibleControlsWithoutNames(page) {
  return page
    .locator(
      "button, input:not([type='hidden']), select, textarea, a[href], [role='button'], [role='tab'], [role='menuitem']",
    )
    .evaluateAll((elements) => {
      const visible = (element) => {
        const style = getComputedStyle(element);
        const rect = element.getBoundingClientRect();
        return (
          style.display !== "none" &&
          style.visibility !== "hidden" &&
          rect.width > 0 &&
          rect.height > 0 &&
          !element.closest("[aria-hidden='true']")
        );
      };
      const labelText = (element) => {
        const labelledBy = (element.getAttribute("aria-labelledby") || "")
          .split(/\s+/)
          .filter(Boolean)
          .map((id) => document.getElementById(id)?.textContent || "")
          .join(" ");
        const explicit = element.id
          ? document.querySelector(`label[for="${CSS.escape(element.id)}"]`)
              ?.textContent || ""
          : "";
        const wrapping = element.closest("label")?.textContent || "";
        return [
          element.getAttribute("aria-label"),
          labelledBy,
          explicit,
          wrapping,
          element.getAttribute("title"),
          element.getAttribute("alt"),
          element.textContent,
          element.getAttribute("placeholder"),
          element.getAttribute("value"),
        ]
          .filter(Boolean)
          .join(" ")
          .replace(/\s+/g, " ")
          .trim();
      };
      return elements
        .filter((element) => visible(element) && !labelText(element))
        .map((element) => ({
          tag: element.tagName.toLowerCase(),
          id: element.id,
          role: element.getAttribute("role"),
          className: element.className,
        }));
    });
}

async function assertNoPageOverflow(page) {
  const geometry = await page.evaluate(() => ({
    viewport: window.innerWidth,
    document: document.documentElement.scrollWidth,
    body: document.body.scrollWidth,
    app: (document.querySelector("#app, .app") || document.body).scrollWidth,
  }));
  expect(geometry.document, JSON.stringify(geometry)).toBeLessThanOrEqual(
    geometry.viewport + 1,
  );
  expect(geometry.body, JSON.stringify(geometry)).toBeLessThanOrEqual(
    geometry.viewport + 1,
  );
  expect(geometry.app, JSON.stringify(geometry)).toBeLessThanOrEqual(
    geometry.viewport + 1,
  );
}

for (const [surfaceName, surface] of Object.entries(SURFACES)) {
  for (const theme of THEMES) {
    for (const width of surface.widths) {
      test(`${surfaceName} renders every tab at ${width}px in ${theme}`, async ({
        page,
      }) => {
        const { pageErrors } = await openSurface(
          page,
          surfaceName,
          theme,
          width,
        );
        const tabs = page.locator(surface.tabSelector);
        const count = await tabs.count();
        expect(count).toBe(surfaceName === "cep" ? 8 : 9);

        for (let index = 0; index < count; index += 1) {
          const tab = tabs.nth(index);
          const tabName = await tab.getAttribute(surface.tabAttribute);
          await tab.click();
          await expect(tab).toHaveAttribute("aria-selected", "true");
          await expect(tab).toHaveAttribute("tabindex", "0");
          const activePanel = page.locator(surface.activePanelSelector);
          await expect(activePanel, tabName || `tab-${index}`).toBeVisible();
          const content = await activePanel.evaluate(
            (panel) => panel.textContent?.replace(/\s+/g, " ").trim() || "",
          );
          expect(
            content.length,
            `${surfaceName}/${tabName} active panel content`,
          ).toBeGreaterThan(20);
          if (surfaceName === "cep") {
            await expect(
              page.locator("#contentTitle, .content-header h1").first(),
            ).toBeVisible();
          } else {
            await expect(
              activePanel
                .locator("h1, h2, .oc-section-title, .oc-workspace-title")
                .first(),
            ).toBeVisible();
          }
          await assertNoPageOverflow(page);
          expect(
            await visibleControlsWithoutNames(page),
            `unnamed controls in ${surfaceName}/${tabName}`,
          ).toEqual([]);
        }

        expect(pageErrors).toEqual([]);
        await expect(page).toHaveScreenshot(
          `${surfaceName}-${theme}-${width}.png`,
          {
            fullPage: false,
          },
        );
      });
    }
  }
}

test("CEP keyboard tabs, focus trap, Escape, and destructive confirmation", async ({
  page,
}) => {
  const { surface, pageErrors } = await openSurface(page, "cep", "dark", 900);
  const tabs = page.locator(surface.tabSelector);
  await tabs.first().focus();
  await page.keyboard.press("ArrowDown");
  await expect(tabs.nth(1)).toBeFocused();
  await expect(tabs.nth(1)).toHaveAttribute("aria-selected", "true");
  await page.keyboard.press("End");
  await expect(tabs.last()).toBeFocused();
  await page.keyboard.press("Home");
  await expect(tabs.first()).toBeFocused();

  const launcher = page.locator("#stageCommandPaletteBtn");
  await launcher.focus();
  await launcher.click();
  const palette = page.locator("#commandPaletteOverlay");
  await expect(palette).toBeVisible();
  await expect(page.locator("#commandPaletteInput")).toBeFocused();
  await page.keyboard.press("Shift+Tab");
  await expect
    .poll(() =>
      page.evaluate(
        () =>
          document.activeElement?.closest("#commandPaletteOverlay")?.id || "",
      ),
    )
    .toBe("commandPaletteOverlay");
  await page.keyboard.press("Escape");
  await expect(palette).toBeHidden();
  await expect(launcher).toBeFocused();

  await page.locator(".nav-tab[data-nav='settings']").click();
  const clearJournal = page.locator("#journalClearBtn");
  await clearJournal.scrollIntoViewIfNeeded();
  await clearJournal.click();
  const confirmation = page.locator(".panel-dialog-overlay[role='dialog']");
  await expect(confirmation).toBeVisible();
  await expect(confirmation).toContainText("Clear operation journal?");
  await page.keyboard.press("Escape");
  await expect(confirmation).toHaveCount(0);
  await expect(clearJournal).toBeFocused();
  expect(pageErrors).toEqual([]);
});

test("UXP keyboard tabs retain focus and selection", async ({ page }) => {
  const { surface, pageErrors } = await openSurface(page, "uxp", "dark", 520);
  const tabs = page.locator(surface.tabSelector);
  await tabs.first().focus();
  await page.keyboard.press("ArrowRight");
  await expect(tabs.nth(1)).toBeFocused();
  await expect(tabs.nth(1)).toHaveAttribute("aria-selected", "true");
  await page.keyboard.press("End");
  await expect(tabs.last()).toBeFocused();
  await page.keyboard.press("Home");
  await expect(tabs.first()).toBeFocused();
  expect(pageErrors).toEqual([]);
});

test("offline, empty, loading, error, permission, and confirmation states stay semantic", async ({
  page,
}) => {
  const { pageErrors } = await openSurface(page, "uxp", "dark", 520);
  const state = await page.evaluate(() => {
    const panel = document.querySelector(".oc-tab-panel.active");
    const fixture = document.createElement("section");
    fixture.id = "renderedStateFixture";
    fixture.innerHTML = `
      <div role="status" aria-label="Loading media" aria-busy="true">Loading media...</div>
      <div role="status" aria-label="No media selected">No media selected.</div>
      <div role="alert">The local backend is offline.</div>
      <div role="alert">Permission denied. Choose a readable folder.</div>
      <div role="dialog" aria-modal="true" aria-labelledby="renderedConfirmTitle">
        <h2 id="renderedConfirmTitle">Delete generated proxy?</h2>
        <button type="button">Cancel</button><button type="button">Delete proxy</button>
      </div>`;
    panel.prepend(fixture);
    return {
      offline: document.querySelector("#connLabel")?.textContent?.trim(),
      roles: Array.from(fixture.querySelectorAll("[role]")).map((node) =>
        node.getAttribute("role"),
      ),
      busy: fixture.querySelector("[aria-busy]")?.getAttribute("aria-busy"),
      dialogName: fixture
        .querySelector("[role='dialog']")
        ?.getAttribute("aria-labelledby"),
    };
  });
  expect(state.offline).toMatch(/offline|disconnected/i);
  expect(state.roles).toEqual(["status", "status", "alert", "alert", "dialog"]);
  expect(state.busy).toBe("true");
  expect(state.dialogName).toBe("renderedConfirmTitle");
  expect(await visibleControlsWithoutNames(page)).toEqual([]);
  expect(pageErrors).toEqual([]);
});
