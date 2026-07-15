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

async function preparePage(page, surface, theme, backendFixtures = {}) {
  const pageErrors = [];
  const capturedRequests = [];
  const locale = backendFixtures.locale || "en-US";
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.emulateMedia({ colorScheme: theme === "auto" ? "light" : theme });
  await page.addInitScript(
    ({ surfaceName, selectedTheme, environment, localeTag }) => {
      localStorage.clear();
      localStorage.setItem("opencut_debug", "0");
      Object.defineProperty(navigator, "language", {
        configurable: true,
        value: localeTag,
      });
      Object.defineProperty(navigator, "languages", {
        configurable: true,
        value: [localeTag],
      });
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
      localeTag: locale,
    },
  );
  await page.route("http://127.0.0.1:*/**", async (route) => {
    const url = new URL(route.request().url());
    if (url.port === "41737") return route.continue();
    if (url.pathname === "/plugins/trust" && backendFixtures.pluginTrust) {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(backendFixtures.pluginTrust),
      });
    }
    if (
      url.pathname === "/plugins/marketplace/install" &&
      route.request().method() === "POST"
    ) {
      capturedRequests.push(route.request().postDataJSON());
      return route.fulfill({
        status: 202,
        contentType: "application/json",
        body: JSON.stringify({ job_id: "plugin-install-fixture" }),
      });
    }
    if (
      url.pathname === "/plugins/workers/restart" &&
      route.request().method() === "POST"
    ) {
      capturedRequests.push({ worker_restart: route.request().postDataJSON() });
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ ok: true, worker: { state: "running" } }),
      });
    }
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
  return { pageErrors, capturedRequests };
}

async function openSurface(
  page,
  surfaceName,
  theme,
  width,
  backendFixtures = {},
) {
  const surface = SURFACES[surfaceName];
  await page.setViewportSize({
    width,
    height: backendFixtures.height || 900,
  });
  const { pageErrors, capturedRequests } = await preparePage(
    page,
    surfaceName,
    theme,
    backendFixtures,
  );
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
  return { surface, pageErrors, capturedRequests };
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
                .locator("h1:visible, h2:visible, .oc-section-title:visible, .oc-workspace-title:visible")
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
  const endGeometry = await page.evaluate(() => {
    const nav = document.getElementById("tabNav")?.getBoundingClientRect();
    const selected = document
      .querySelector(".oc-tab[aria-selected='true']")
      ?.getBoundingClientRect();
    return nav && selected
      ? { navLeft: nav.left, navRight: nav.right, tabLeft: selected.left, tabRight: selected.right }
      : null;
  });
  expect(endGeometry).not.toBeNull();
  expect(endGeometry.tabLeft).toBeGreaterThanOrEqual(endGeometry.navLeft - 1);
  expect(endGeometry.tabRight).toBeLessThanOrEqual(endGeometry.navRight + 1);
  await page.keyboard.press("Home");
  await expect(tabs.first()).toBeFocused();
  expect(pageErrors).toEqual([]);
});

for (const width of [480, 520]) {
  for (const locale of ["en-US", "es-ES"]) {
    test(`UXP constrained shell keeps orientation and action visible at ${width}px in ${locale}`, async ({
      page,
    }) => {
      const { surface, pageErrors } = await openSurface(
        page,
        "uxp",
        "dark",
        width,
        { height: 800, locale },
      );
      await expect(page.locator("#tabScrollPrev")).toBeHidden();
      await expect(page.locator("#tabScrollNext")).toBeHidden();
      const compactNav = await page.evaluate(() => {
        const nav = document.getElementById("tabNav");
        const tabs = Array.from(document.querySelectorAll(".oc-tab"));
        return {
          fitsWithoutScroll: !!nav && nav.scrollWidth <= nav.clientWidth + 1,
          maxTabWidth: Math.max(...tabs.map((tab) => tab.getBoundingClientRect().width)),
          labelsVisuallyClipped: tabs.every((tab) => {
            const label = tab.querySelector("span");
            return label && getComputedStyle(label).clipPath === "inset(50%)";
          }),
        };
      });
      expect(compactNav.fitsWithoutScroll).toBe(true);
      expect(compactNav.maxTabWidth).toBeLessThanOrEqual(40.5);
      expect(compactNav.labelsVisuallyClipped).toBe(true);

      const tabs = page.locator(surface.tabSelector);
      for (let index = 0; index < (await tabs.count()); index += 1) {
        const tab = tabs.nth(index);
        await tab.click();
        await expect(page.locator("#workspaceOverviewTitle")).toBeVisible();
        await expect(page.locator("#workspaceGuide")).toBeVisible();
        await expect(page.locator("#workspaceGuideAction")).toBeVisible();
        const geometry = await page.evaluate(() => {
          const nav = document.getElementById("tabNav")?.getBoundingClientRect();
          const selected = document
            .querySelector(".oc-tab[aria-selected='true']")
            ?.getBoundingClientRect();
          const title = document.getElementById("workspaceOverviewTitle")?.getBoundingClientRect();
          const state = document.getElementById("workspaceGuide")?.getBoundingClientRect();
          const action = document.getElementById("workspaceGuideAction")?.getBoundingClientRect();
          return {
            mainScrollTop: document.getElementById("mainContent")?.scrollTop || 0,
            nav,
            selected,
            title,
            state,
            action,
          };
        });
        expect(geometry.mainScrollTop).toBe(0);
        expect(geometry.selected.left).toBeGreaterThanOrEqual(geometry.nav.left - 1);
        expect(geometry.selected.right).toBeLessThanOrEqual(geometry.nav.right + 1);
        for (const region of [geometry.title, geometry.state, geometry.action]) {
          expect(region.top).toBeGreaterThanOrEqual(0);
          expect(region.bottom).toBeLessThanOrEqual(800);
        }
      }

      expect(pageErrors).toEqual([]);
    });
  }
}

test("UXP wide shell keeps overflow controls hidden and expands offline details", async ({
  page,
}) => {
  const { pageErrors } = await openSurface(page, "uxp", "dark", 1200, {
    height: 800,
  });
  await expect(page.locator("#tabScrollPrev")).toBeHidden();
  await expect(page.locator("#tabScrollNext")).toBeHidden();
  await expect(page.locator("#connectionStatus")).toBeVisible();
  await page.locator("#connectionStatus").click();
  await expect(page.locator("#statusBar")).toBeVisible();
  await expect(page.locator("#statusText")).toContainText(/backend offline/i);
  await expect(page.locator("#refreshBtn")).toBeVisible();
  const menuGeometry = await page.evaluate(() => {
    const detail = document.getElementById("statusBar");
    const select = document.getElementById("silenceMode");
    const detailStyle = getComputedStyle(detail);
    const selectStyle = getComputedStyle(select);
    return {
      detailRadius: detailStyle.borderRadius,
      detailFontSize: Number.parseFloat(getComputedStyle(document.getElementById("statusText")).fontSize),
      selectRadius: selectStyle.borderRadius,
      selectHeight: select.getBoundingClientRect().height,
      selectAppearance: selectStyle.appearance,
      selectArrow: selectStyle.backgroundImage,
    };
  });
  expect(menuGeometry.detailRadius).toBe("9px");
  expect(menuGeometry.detailFontSize).toBeGreaterThanOrEqual(12);
  expect(menuGeometry.selectRadius).toBe("7px");
  expect(menuGeometry.selectHeight).toBeGreaterThanOrEqual(40);
  expect(menuGeometry.selectAppearance).toBe("none");
  expect(menuGeometry.selectArrow).toContain("data:image/svg+xml");
  expect(pageErrors).toEqual([]);
});

test("wide command-center shells expose editorial rails and settings grids", async ({
  page,
}) => {
  const cep = await openSurface(page, "cep", "dark", 1200, { height: 800 });
  await page.locator(".nav-tab[data-nav='settings']").click();
  const cepGeometry = await page.evaluate(() => {
    const sidebar = document.querySelector(".sidebar")?.getBoundingClientRect();
    const cards = Array.from(document.querySelectorAll("#panel-settings.active > .card"))
      .slice(0, 2)
      .map((card) => card.getBoundingClientRect());
    return {
      sidebarWidth: sidebar?.width || 0,
      cardColumns: cards.length === 2 ? Math.abs(cards[0].left - cards[1].left) : 0,
      bodyFontSize: Number.parseFloat(getComputedStyle(document.body).fontSize),
      brandMetaDisplay: getComputedStyle(document.querySelector(".brand-meta")).display,
      kickerDisplay: getComputedStyle(document.querySelector(".content-kicker-row")).display,
      cardShadow: getComputedStyle(document.querySelector("#panel-settings.active > .card")).boxShadow,
    };
  });
  expect(cepGeometry.sidebarWidth).toBeGreaterThanOrEqual(160);
  expect(cepGeometry.cardColumns).toBeGreaterThan(200);
  expect(cepGeometry.bodyFontSize).toBeGreaterThanOrEqual(13);
  expect(cepGeometry.brandMetaDisplay).toBe("none");
  expect(cepGeometry.kickerDisplay).toBe("none");
  expect(cepGeometry.cardShadow).toBe("none");
  expect(cep.pageErrors).toEqual([]);

  const uxp = await openSurface(page, "uxp", "dark", 1200, { height: 800 });
  await page.locator(".oc-tab[data-tab='settings']").click();
  const uxpGeometry = await page.evaluate(() => {
    const rail = document.getElementById("tabNavShell")?.getBoundingClientRect();
    const tabs = getComputedStyle(document.getElementById("tabNav"));
    const header = document.querySelector(".oc-header")?.getBoundingClientRect();
    const commandBar = document.querySelector(".oc-workspace-actions")?.getBoundingClientRect();
    const groups = Array.from(document.querySelectorAll("#tab-settings.active > .oc-settings-group"))
      .slice(0, 2)
      .map((group) => group.getBoundingClientRect());
    return {
      railWidth: rail?.width || 0,
      tabDirection: tabs.flexDirection,
      groupColumns: groups.length === 2 ? Math.abs(groups[0].left - groups[1].left) : 0,
      bodyFontSize: Number.parseFloat(getComputedStyle(document.body).fontSize),
      groupShadow: getComputedStyle(document.querySelector("#tab-settings.active > .oc-settings-group")).boxShadow,
      commandBarInHeader: !!header && !!commandBar
        && commandBar.top >= header.top
        && commandBar.bottom <= header.bottom,
      guideDisplay: getComputedStyle(document.getElementById("workspaceGuide")).display,
      groupTitles: Array.from(document.querySelectorAll("#tab-settings.active > .oc-settings-group > .oc-section-title"))
        .slice(0, 4)
        .map((title) => title.textContent?.trim()),
    };
  });
  expect(uxpGeometry.railWidth).toBeGreaterThanOrEqual(160);
  expect(uxpGeometry.tabDirection).toBe("column");
  expect(uxpGeometry.groupColumns).toBeGreaterThan(200);
  expect(uxpGeometry.bodyFontSize).toBeGreaterThanOrEqual(13);
  expect(uxpGeometry.groupShadow).toBe("none");
  expect(uxpGeometry.commandBarInHeader).toBe(true);
  expect(uxpGeometry.guideDisplay).toBe("none");
  expect(uxpGeometry.groupTitles).toEqual([
    "Workspace",
    "Engine Routing",
    "Live Updates",
    "Diagnostics",
  ]);
  await expect(page.locator("#workspaceChooseClipBtn")).toBeEnabled();
  await expect(page.locator("#settingsWorkspaceBackendValue")).toHaveText("Offline");
  await expect(page.locator("#settingsDiagnosticsBackendValue")).toHaveText("Offline");
  await expect(page.locator("#settingsDiagnosticsEndpointValue")).toContainText("127.0.0.1");
  await expect(page.locator("#settingsDiagnosticsLastCheckValue")).toHaveText("Just now");
  await page.locator("#settingsDiagnosticsDetailsBtn").click();
  await expect(page.locator("#connectionDetails")).toHaveAttribute("open", "");
  await page.locator(".oc-tab[data-tab='cut']").click();
  await page.locator("#clipPathCut").fill("C:/media/interview.mov");
  await page.locator(".oc-tab[data-tab='settings']").click();
  await expect(page.locator("#settingsWorkspaceSourceValue")).toHaveText("interview.mov");
  await page.locator("#settingsWorkspaceSearchBtn").click();
  await expect(page.locator(".oc-tab[data-tab='search']")).toHaveAttribute("aria-selected", "true");
  expect(uxp.pageErrors).toEqual([]);
});

test("CEP tool submenus keep every label readable behind explicit overflow controls", async ({
  page,
}) => {
  const { pageErrors } = await openSurface(page, "cep", "dark", 1200, {
    height: 900,
  });
  await page.locator(".nav-tab[data-nav='video']").click();
  const shell = page.locator("#panel-video .sub-tabs-shell");
  const tablist = page.locator("#videoSubTabs");
  const previous = shell.locator(".sub-tabs-scroll--previous");
  const next = shell.locator(".sub-tabs-scroll--next");
  await expect(shell).toBeVisible();
  await expect(previous).toBeVisible();
  await expect(previous).toBeDisabled();
  await expect(next).toBeVisible();
  await expect(next).toBeEnabled();

  const initial = await tablist.evaluate((node) => ({
    clientWidth: node.clientWidth,
    scrollWidth: node.scrollWidth,
    scrollLeft: node.scrollLeft,
    tabTops: Array.from(node.querySelectorAll(".sub-tab")).map(
      (tab) => tab.getBoundingClientRect().top,
    ),
    shellRadius: getComputedStyle(node.parentElement).borderRadius,
  }));
  expect(initial.scrollWidth).toBeGreaterThan(initial.clientWidth);
  expect(initial.scrollLeft).toBe(0);
  expect(Math.max(...initial.tabTops) - Math.min(...initial.tabTops)).toBeLessThan(1);
  expect(initial.shellRadius).toBe("8px");
  await expect(shell).toHaveScreenshot("cep-video-submenu-dark-1200.png");

  await next.click();
  await expect.poll(() => tablist.evaluate((node) => node.scrollLeft)).toBeGreaterThan(0);
  await expect(previous).toBeEnabled();
  await page.locator("#videoSubTabs .sub-tab").last().click();
  const activeGeometry = await page.evaluate(() => {
    const list = document.getElementById("videoSubTabs")?.getBoundingClientRect();
    const active = document.querySelector("#videoSubTabs .sub-tab.active")?.getBoundingClientRect();
    return { list, active };
  });
  expect(activeGeometry.active.left).toBeGreaterThanOrEqual(activeGeometry.list.left - 1);
  expect(activeGeometry.active.right).toBeLessThanOrEqual(activeGeometry.list.right + 1);
  expect(pageErrors).toEqual([]);
});

test("CEP listboxes and clip context menu share focus-safe menu behavior", async ({
  page,
}) => {
  const { pageErrors } = await openSurface(page, "cep", "dark", 900, {
    height: 900,
  });
  await page.locator(".nav-tab[data-nav='cut']").click();
  const fieldTrigger = page
    .locator(".nav-panel.active .sub-panel.active .custom-dropdown-trigger")
    .first();
  await fieldTrigger.scrollIntoViewIfNeeded();
  await fieldTrigger.click();
  const listboxId = await fieldTrigger.getAttribute("aria-controls");
  const listbox = page.locator(`#${listboxId}`);
  await expect(fieldTrigger).toHaveAttribute("aria-expanded", "true");
  await expect(listbox).toBeVisible();
  const dropdownGeometry = await listbox.evaluate((node) => ({
    radius: getComputedStyle(node).borderRadius,
    itemHeight: node.querySelector(".custom-dropdown-item")?.getBoundingClientRect().height || 0,
    selectedState: node.querySelector(".custom-dropdown-item.selected")?.getAttribute("aria-selected"),
  }));
  expect(dropdownGeometry.radius).toBe("8px");
  expect(dropdownGeometry.itemHeight).toBeGreaterThanOrEqual(36);
  expect(dropdownGeometry.selectedState).toBe("true");
  const dropdownBounds = await page.evaluate((id) => {
    const menu = document.getElementById(id)?.getBoundingClientRect();
    const footer = document.querySelector(".content-footer")?.getBoundingClientRect();
    return { menuBottom: menu?.bottom || 0, footerTop: footer?.top || window.innerHeight };
  }, listboxId);
  expect(dropdownBounds.menuBottom).toBeLessThanOrEqual(dropdownBounds.footerTop + 1);
  await expect(listbox).toHaveScreenshot("cep-dropdown-dark-900.png");
  await page.keyboard.press("Escape");
  await expect(listbox).toBeHidden();
  await expect(fieldTrigger).toBeFocused();

  await page.evaluate(() => {
    const select = document.getElementById("clipSelect");
    const option = document.createElement("option");
    option.value = "C:/media/interview.mov";
    option.textContent = "interview.mov";
    option.selected = true;
    select.appendChild(option);
    select.value = option.value;
    select._customDropdown.update();
  });
  const clipTrigger = page.locator(".custom-dropdown[data-for='clipSelect'] .custom-dropdown-trigger");
  await clipTrigger.click();
  await page.locator("#clipSelect-listbox .custom-dropdown-item").last().click();
  await expect(page.locator("body")).toHaveClass(/has-clip/);
  await expect(clipTrigger).toHaveAttribute("data-context-menu-target", "clip-actions");
  await clipTrigger.dispatchEvent("contextmenu", { clientX: 240, clientY: 220 });
  const contextMenu = page.locator("#contextMenu");
  await expect(contextMenu).toBeVisible();
  await expect(contextMenu.locator(".context-menu-item").first()).toBeFocused();
  await expect(contextMenu).toHaveScreenshot("cep-context-menu-dark-900.png");
  await page.keyboard.press("End");
  await expect(contextMenu.locator(".context-menu-item").last()).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(contextMenu).toBeHidden();
  await expect(clipTrigger).toBeFocused();

  const recentClipsButton = page.locator("#recentClipsBtn");
  await recentClipsButton.click();
  const recentClipsMenu = page.locator("#recentClipsDropdown");
  await expect(recentClipsMenu).toBeVisible();
  await expect(recentClipsButton).toHaveAttribute("aria-expanded", "true");
  await expect(recentClipsMenu.locator(".recent-clip-item")).toHaveCount(1);
  await expect(recentClipsMenu).toHaveScreenshot("cep-recent-clips-dark-900.png");
  await page.keyboard.press("Escape");
  await expect(recentClipsMenu).toBeHidden();
  await expect(recentClipsButton).toBeFocused();

  await page.mouse.move(890, 890);
  await page.keyboard.press("Control+K");
  const palette = page.locator(".command-palette");
  await expect(palette).toBeVisible();
  await expect(palette.locator(".command-palette-item.selected")).toHaveCount(1);
  const paletteDescriptions = await palette.locator(".command-palette-desc").allTextContents();
  expect(paletteDescriptions.length).toBeGreaterThan(0);
  expect(paletteDescriptions.every((description) => !description.trim().startsWith("function"))).toBe(true);
  await expect(palette).toHaveScreenshot("cep-command-palette-dark-900.png");
  await page.keyboard.press("Escape");
  await expect(palette).toBeHidden();
  expect(pageErrors).toEqual([]);
});

test("offline, empty, loading, error, permission, and confirmation states stay semantic", async ({
  page,
}) => {
  const { pageErrors } = await openSurface(page, "uxp", "dark", 520);
  await expect(page.locator("#connLabel")).toHaveText(/offline|disconnected/i);
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

const PLUGIN_TRUST_FIXTURE = {
  plugins: [],
  summary: {
    loaded: 0,
    failed: 0,
    lock_missing: 0,
    unsigned: 0,
    quarantined: 0,
    marketplace: 1,
  },
  quarantine: { entries: [] },
  marketplace: {
    plugins: [
      {
        plugin_id: "signed-captions",
        name: "Signed Captions",
        version: "2.1.0",
        description: "Caption workflow fixture",
        installed: false,
        authenticated: true,
        artifact_sha256: "a".repeat(64),
        publisher_id: "publisher.example",
        publisher_fingerprint: "b".repeat(64),
        capabilities: ["http.routes", "host.network"],
      },
    ],
  },
  actions: {
    marketplace: {
      registry_route: "/plugins/registry",
      install_route: "/plugins/marketplace/install",
    },
  },
};

const PLUGIN_WORKER_TRUST_FIXTURE = {
  plugins: [
    {
      name: "isolated-captions",
      version: "1.0.0",
      description: "Isolated caption helper",
      load_status: "loaded",
      trust: { source: "locked", errors: [], warnings: [] },
      capability_badges: [{ id: "http.routes", label: "HTTP routes", kind: "network" }],
      runtime: "supervised_process",
      worker: {
        state: "stopped",
        crash_count: 1,
        last_error: "request_timeout",
        security_boundary: "availability isolation; not an OS sandbox",
      },
    },
  ],
  summary: { loaded: 1, failed: 0, lock_missing: 0, unsigned: 0, quarantined: 0, marketplace: 0 },
  quarantine: { entries: [] },
  marketplace: { plugins: [] },
  actions: {
    restart_worker: { route: "/plugins/workers/restart", method: "POST" },
  },
};

for (const surfaceName of ["cep", "uxp"]) {
  test(`${surfaceName} requires explicit publisher and capability approval`, async ({
    page,
  }) => {
    const width = surfaceName === "cep" ? 900 : 520;
    const { surface, pageErrors, capturedRequests } = await openSurface(
      page,
      surfaceName,
      "dark",
      width,
      { pluginTrust: PLUGIN_TRUST_FIXTURE },
    );
    await page
      .locator(`${surface.tabSelector}[${surface.tabAttribute}='settings']`)
      .click();
    const checkbox = page.locator(
      surfaceName === "cep"
        ? ".plugin-install-approval-checkbox"
        : ".oc-plugin-install-approval-checkbox",
    );
    const button = page.locator(
      surfaceName === "cep"
        ? ".plugin-install-btn"
        : ".oc-plugin-install-btn",
    );
    await expect(checkbox).toBeVisible();
    await expect(button).toBeDisabled();
    await expect(checkbox.locator("xpath=.."))
      .toContainText("http.routes, host.network");
    await expect(checkbox.locator("xpath=../.."))
      .toContainText("publisher.example");
    await expect(checkbox.locator("xpath=../.."))
      .toContainText("b".repeat(64));
    await checkbox.check();
    await expect(button).toBeEnabled();
    await button.click();
    await expect.poll(() => capturedRequests.length).toBe(1);
    expect(capturedRequests[0]).toEqual({
      plugin_id: "signed-captions",
      approved_capabilities: ["http.routes", "host.network"],
      approve_publisher_fingerprint: "b".repeat(64),
    });
    await assertNoPageOverflow(page);
    expect(await visibleControlsWithoutNames(page)).toEqual([]);
    expect(pageErrors).toEqual([]);
  });
}

const CEP_CONTEXTUAL_CONTRAST_PAIRS = [
  ["text-danger-control", "bg-card"],
  ["text-danger-control-hover", "bg-card"],
  ["text-accent-control", "bg-card"],
  ["text-accent-control-hover", "bg-card"],
  ["text-warning-control-hover", "bg-card"],
  ["text-action-hover", "bg-card"],
  ["text-on-danger-surface", "danger-surface"],
];

for (const theme of ["dark", "light"]) {
  test(`cep contextual control tokens meet WCAG AA in ${theme}`, async ({ page }) => {
    const { pageErrors } = await openSurface(page, "cep", theme, 900);
    const findings = await page.evaluate((pairs) => {
      const linearize = (channel) => {
        const normalized = channel / 255;
        return normalized <= 0.04045
          ? normalized / 12.92
          : ((normalized + 0.055) / 1.055) ** 2.4;
      };
      const luminance = ([red, green, blue]) => (
        0.2126 * linearize(red) +
        0.7152 * linearize(green) +
        0.0722 * linearize(blue)
      );
      const parseRgb = (value) => (
        (value.match(/[\d.]+/g) || []).slice(0, 3).map(Number)
      );
      return pairs.map(([foreground, background]) => {
        const probe = document.createElement("span");
        probe.style.color = `var(--${foreground})`;
        probe.style.backgroundColor = `var(--${background})`;
        document.body.appendChild(probe);
        const style = getComputedStyle(probe);
        const foregroundRgb = parseRgb(style.color);
        const backgroundRgb = parseRgb(style.backgroundColor);
        probe.remove();
        const foregroundLum = luminance(foregroundRgb);
        const backgroundLum = luminance(backgroundRgb);
        const ratio = (Math.max(foregroundLum, backgroundLum) + 0.05) /
          (Math.min(foregroundLum, backgroundLum) + 0.05);
        return { foreground, background, ratio };
      });
    }, CEP_CONTEXTUAL_CONTRAST_PAIRS);
    for (const finding of findings) {
      expect(finding.ratio, `${finding.foreground} on ${finding.background}`).toBeGreaterThanOrEqual(4.5);
    }
    expect(pageErrors).toEqual([]);
  });
}

for (const surfaceName of ["cep", "uxp"]) {
  test(`${surfaceName} shows isolated worker health and restart control`, async ({
    page,
  }) => {
    const width = surfaceName === "cep" ? 900 : 520;
    const { surface, pageErrors, capturedRequests } = await openSurface(
      page,
      surfaceName,
      "dark",
      width,
      { pluginTrust: PLUGIN_WORKER_TRUST_FIXTURE },
    );
    await page
      .locator(`${surface.tabSelector}[${surface.tabAttribute}='settings']`)
      .click();
    const row = page.locator(
      surfaceName === "cep" ? ".plugin-trust-row" : ".oc-plugin-trust-row",
    ).first();
    const button = page.getByRole("button", { name: "Restart worker" });
    await expect(row).toContainText("Worker: stopped");
    await expect(row).toContainText("not an OS security sandbox");
    await expect(button).toBeVisible();
    await button.click();
    await expect.poll(() => capturedRequests.length).toBe(1);
    expect(capturedRequests[0]).toEqual({
      worker_restart: { name: "isolated-captions" },
    });
    expect(pageErrors).toEqual([]);
  });
}
