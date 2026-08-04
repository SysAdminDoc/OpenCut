import { expect, test } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { readFile } from "node:fs/promises";

const THEMES = ["dark", "light", "auto"];
const WCAG_TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"];

// Every exception must be a narrowly scoped selector with a rule id and a
// reviewable reason. Keep this registry empty unless a production fixture has
// a documented, non-actionable host limitation.
const WCAG_SUPPRESSIONS = [];

function formatWcagViolations(violations) {
  return violations.map(({ id, impact, help, nodes }) => ({
    id,
    impact,
    help,
    targets: nodes.map((node) => node.target),
  }));
}

async function assertWcagCompliance(page, stateName) {
  let builder = new AxeBuilder({ page }).withTags(WCAG_TAGS);
  for (const suppression of WCAG_SUPPRESSIONS) {
    builder = builder.exclude(suppression.selector);
  }
  const results = await builder.analyze();
  expect(
    formatWcagViolations(results.violations),
    `${stateName} WCAG 2.2 AA violations`,
  ).toEqual([]);
}
const BREAKPOINT_BOUNDARIES = {
  // These are the layout transitions used by the production command-center
  // styles. UXP's 820px max-width rule hands off to the 821px min-width rule,
  // so the 821 boundary exercises 820/821 explicitly.
  cep: [620, 700, 980],
  uxp: [620, 821, 1020, 1050],
};
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

// Panel greys Premiere reports for its skins. "auto" must resolve from the
// host skin, so the fixture has to carry a real one rather than a fixed dark.
const HOST_SKIN_GREY = { light: 180, dark: 50, darkest: 24 };

function hostEnvironment(skin = "darkest") {
  const grey = HOST_SKIN_GREY[skin] ?? HOST_SKIN_GREY.darkest;
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
        color: { red: grey, green: grey, blue: grey, alpha: 255 },
      },
    },
  });
}

async function preparePage(page, surface, theme, backendFixtures = {}) {
  const pageErrors = [];
  const capturedRequests = [];
  const destructiveTokens = new Map();
  const destructivePreviewCounts = new Map();
  let queueTokenExpired = false;
  const liveBridgeState = {
    running: true,
    stopOutcomes: [...(backendFixtures.liveBridge?.stopOutcomes || [])],
  };
  const onboardingState = {
    seen: true,
    step: 0,
    updated_at: 0,
    ...(backendFixtures.onboardingState || {}),
  };
  const locale = backendFixtures.locale || "en-US";
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.emulateMedia({ colorScheme: theme === "auto" ? "light" : theme });
  await page.addInitScript(
    ({ surfaceName, selectedTheme, environment, localeTag, forceEventSourceError, hostTheme, liveBridge }) => {
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
          }),
        );
        const callbacks = new Map();
        let currentEnvironment = environment;
        window.__opencutCepThemeHarness = {
          setEnvironment(next) {
            currentEnvironment = next;
          },
          emit() {
            const listener = callbacks.get(
              "com.adobe.csxs.events.ThemeColorChanged",
            );
            if (listener) listener({ type: "com.adobe.csxs.events.ThemeColorChanged" });
          },
          listenerCount: () =>
            callbacks.has("com.adobe.csxs.events.ThemeColorChanged") ? 1 : 0,
        };
        window.__adobe_cep__ = new Proxy(
          {
            getHostEnvironment: () => currentEnvironment,
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
      } else {
        const themeListeners = new Set();
        let currentHostTheme = hostTheme;
        Object.defineProperty(document, "theme", {
          configurable: true,
          value: {
            getCurrent: () => currentHostTheme,
            onUpdated: {
              addListener: (listener) => themeListeners.add(listener),
              removeListener: (listener) => themeListeners.delete(listener),
            },
          },
        });
        window.__opencutThemeHarness = {
          emit(theme) {
            currentHostTheme = theme;
            themeListeners.forEach((listener) => listener(theme));
          },
          listenerCount: () => themeListeners.size,
        };
      }
      window.WebSocket = class RenderedWebSocket {
        static CONNECTING = 0;
        static OPEN = 1;
        static CLOSED = 3;
        constructor() {
          this.readyState = liveBridge
            ? RenderedWebSocket.OPEN
            : RenderedWebSocket.CLOSED;
          if (liveBridge) {
            setTimeout(() => this.onopen?.(), 0);
          }
        }
        addEventListener() {}
        removeEventListener() {}
        close() {
          this.readyState = RenderedWebSocket.CLOSED;
          queueMicrotask(() => this.onclose?.());
        }
        send() {}
      };
      window.EventSource = class RenderedEventSource {
        constructor() {
          this.readyState = 2;
          if (forceEventSourceError) {
            setTimeout(() => this.onerror?.(new Event("error")), 0);
          }
        }
        addEventListener() {}
        close() {}
      };
    },
    {
      surfaceName: surface,
      selectedTheme: theme,
      environment: hostEnvironment(
        theme === "light" ? "light" : theme === "dark" ? "dark" : "darkest",
      ),
      localeTag: locale,
      forceEventSourceError: Boolean(backendFixtures.boundaryReview),
      hostTheme: theme === "light" ? "light" : theme === "dark" ? "dark" : "darkest",
      liveBridge: Boolean(backendFixtures.liveBridge),
    },
  );
  await page.route("http://127.0.0.1:*/**", async (route) => {
    const url = new URL(route.request().url());
    if (url.port === "41737") return route.continue();
    if (url.pathname === "/settings/onboarding") {
      const method = route.request().method();
      capturedRequests.push({
        onboarding: method,
        body: method === "POST" ? route.request().postDataJSON() || {} : null,
      });
      if (backendFixtures.onboardingUnavailable) {
        return route.fulfill({
          status: 503,
          contentType: "application/json",
          body: JSON.stringify({
            error: "Rendered fixture: onboarding state unavailable",
            code: "BACKEND_OFFLINE",
          }),
        });
      }
      if (method === "POST") {
        const body = route.request().postDataJSON() || {};
        if (typeof body.seen === "boolean") onboardingState.seen = body.seen;
        if (Number.isFinite(Number(body.step))) onboardingState.step = Number(body.step);
      }
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(onboardingState),
      });
    }
    if (backendFixtures.boundaryReview) {
      const method = route.request().method();
      if (url.pathname === "/health" && method === "GET") {
        return route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({ status: "ok", csrf_token: "fixture-token" }),
        });
      }
      if (url.pathname === "/fillers" && method === "POST") {
        const body = route.request().postDataJSON() || {};
        capturedRequests.push({ fillers: body });
        return route.fulfill({
          status: 202,
          contentType: "application/json",
          body: JSON.stringify({ job_id: "boundary-fixture" }),
        });
      }
      if (url.pathname === "/status/boundary-fixture" && method === "GET") {
        return route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            status: "complete",
            progress: 100,
            result: {
              preview_only: true,
              mutation_blocked: true,
              filler_stats: { removed_fillers: 0, total_filler_time: 0.2 },
              boundary_review: {
                required: true,
                review_hits: 1,
                items: [
                  {
                    text: "um",
                    start: 1,
                    end: 1.2,
                    boundary_confidence: null,
                    audition: {
                      filepath: "C:/media/interview.mov",
                      start: 0.25,
                      duration: 1.7,
                      filter: "raw",
                    },
                  },
                ],
              },
              asr_provenance: {
                engine: "faster-whisper",
                model_id: "Systran/faster-whisper-base",
                model_revision: "ebe41a7c92b6db74b05b378aa96b6dcf5251e2c4",
                alignment_mode: "decoder-token-timestamps",
                language_decision: "auto-detected:en",
              },
            },
          }),
        });
      }
      if (url.pathname === "/preview/audio" && method === "POST") {
        capturedRequests.push({
          boundaryAudition: route.request().postDataJSON() || {},
        });
        return route.fulfill({
          status: 200,
          contentType: "audio/wav",
          body: "RIFF....WAVEfmt ",
        });
      }
    }
    if (backendFixtures.workflowPreflight) {
      const method = route.request().method();
      if (url.pathname === "/health" && method === "GET") {
        return route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({ status: "ok", csrf_token: "fixture-token" }),
        });
      }
      if (url.pathname === "/workflow/presets" && method === "GET") {
        return route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            builtins: [{
              name: "Reviewable Cloud Fixture",
              builtin: true,
              description: "A fixture workflow that requires approval.",
              steps: [{ endpoint: "/audio/tts/generate", params: { url: "https://example.test/voice" } }],
            }],
            custom: [],
          }),
        });
      }
      if (url.pathname === "/workflows/list" && method === "GET") {
        return route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify([]),
        });
      }
      if (url.pathname === "/workflow/compile" && method === "POST") {
        const body = route.request().postDataJSON() || {};
        capturedRequests.push({ workflowCompile: body });
        return route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            success: true,
            requires_approval: true,
            plan: {
              schema_version: 1,
              plan_id: "workflow-plan-fixture",
              definition_id: "workflow-definition-fixture",
              source: { filepath: body.filepath, fingerprint: {}, media: {} },
              steps: [{
                index: 0,
                endpoint: "/audio/tts/generate",
                label: "Generating TTS",
                params: { url: "https://example.test/voice" },
                side_effect: "cloud",
                idempotent: false,
              }],
              preflight: {
                status: "ready",
                blocked_reasons: [],
                approval_reasons: ["/audio/tts/generate: external network or remote service"],
                checks: [],
              },
              approval: {
                required: true,
                approved: false,
                plan_id: "workflow-plan-fixture",
                token: "",
              },
              resume: { enabled: true, strategy: "idempotent-artifact-checksum", completed_steps: 0 },
            },
          }),
        });
      }
      if (url.pathname === "/workflow/approve" && method === "POST") {
        const body = route.request().postDataJSON() || {};
        capturedRequests.push({ workflowApprove: body });
        const plan = body.plan || {};
        plan.approval = {
          required: true,
          approved: true,
          plan_id: plan.plan_id,
          token: "workflow-approval-fixture",
        };
        return route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({ success: true, plan, approval: plan.approval }),
        });
      }
      if (url.pathname === "/workflow/run" && method === "POST") {
        capturedRequests.push({ workflowRun: route.request().postDataJSON() || {} });
        return route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({ job_id: "workflow-fixture" }),
        });
      }
      if (url.pathname === "/status/workflow-fixture" && method === "GET") {
        return route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            status: "complete",
            progress: 100,
            result: { success: true, steps_completed: 1 },
          }),
        });
      }
    }
    if (backendFixtures.liveBridge) {
      const method = route.request().method();
      if (url.pathname === "/health" && method === "GET") {
        return route.fulfill({
          status: 200,
          contentType: "application/json",
          headers: { "X-OpenCut-Token": "fixture-token" },
          body: JSON.stringify({
            status: "ok",
            csrf_token: "fixture-token",
            capabilities: { websocket: true },
          }),
        });
      }
      if (url.pathname === "/ws/status" && method === "GET") {
        return route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            running: liveBridgeState.running,
            clients: liveBridgeState.running ? 1 : 0,
            port: 5680,
          }),
        });
      }
      if (url.pathname === "/ws/stop" && method === "POST") {
        capturedRequests.push({ liveBridgeStop: true });
        await new Promise((resolve) =>
          setTimeout(resolve, backendFixtures.liveBridge.stopDelayMs || 0),
        );
        const outcome = liveBridgeState.stopOutcomes.shift() || "success";
        if (outcome === "fail") {
          return route.fulfill({
            status: 503,
            contentType: "application/json",
            body: JSON.stringify({
              error: "Rendered fixture: bridge process did not stop",
              code: "BRIDGE_STOP_FAILED",
            }),
          });
        }
        liveBridgeState.running = false;
        return route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({ success: true }),
        });
      }
    }
    if (
      url.pathname === "/queue/list" &&
      Array.isArray(backendFixtures.queueEntries)
    ) {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(backendFixtures.queueEntries),
      });
    }
    if (
      url.pathname.startsWith("/queue/replay/") &&
      route.request().method() === "POST" &&
      Array.isArray(backendFixtures.queueEntries)
    ) {
      const queueId = decodeURIComponent(url.pathname.slice("/queue/replay/".length));
      capturedRequests.push({ queueReplay: queueId });
      const entry = backendFixtures.queueEntries.find((item) => item.id === queueId);
      if (entry) entry.status = "queued";
      return route.fulfill({
        status: entry ? 200 : 404,
        contentType: "application/json",
        body: JSON.stringify(
          entry
            ? { queue_id: queueId, status: "queued" }
            : { error: "Queue entry not found" },
        ),
      });
    }
    if (backendFixtures.destructiveProtocol) {
      const method = route.request().method();
      const listFixtures = {
        "/presets": {
          "Editorial Clean": { settings: { denoise: true }, saved: 1 },
        },
        "/models/list": {
          models: [
            {
              name: "whisper-fixture.bin",
              path: "C:/OpenCut/models/whisper-fixture.bin",
              size_mb: 24,
              source: "whisper",
            },
          ],
          total_mb: 24,
        },
        "/queue/list": [
          { id: "queued-fixture", endpoint: "/silence", status: "queued" },
        ],
        "/workflows/list": [
          {
            name: "Fixture Workflow",
            description: "Rendered destructive protocol fixture",
            steps: [{ endpoint: "/silence", label: "Silence" }],
          },
        ],
      };
      if (method === "GET" && Object.hasOwn(listFixtures, url.pathname)) {
        return route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify(listFixtures[url.pathname]),
        });
      }
      const destructivePaths = new Set([
        "/presets/delete",
        "/models/delete",
        "/queue/clear",
        "/workflow/delete",
        "/logs/clear",
      ]);
      if (destructivePaths.has(url.pathname) && method !== "GET") {
        const body = route.request().postDataJSON() || {};
        capturedRequests.push({
          destructive: true,
          path: url.pathname,
          method,
          body,
        });
        const definitions = {
          "/presets/delete": {
            operation: "user_data.preset.delete",
            records: [{ key: body.name, kind: "preset", bytes: 32 }],
            targets: [],
            reversible: true,
          },
          "/models/delete": {
            operation: "models.delete",
            records: [],
            targets: [{ path: body.path, bytes: 25165824 }],
            reversible: false,
          },
          "/queue/clear": {
            operation: "queue.clear",
            records: [
              { id: "queued-fixture", endpoint: "/silence", status: "queued" },
            ],
            targets: [],
            reversible: false,
          },
          "/workflow/delete": {
            operation: "user_data.workflow.delete",
            records: [{ key: body.name, kind: "workflow", bytes: 48 }],
            targets: [],
            reversible: true,
          },
          "/logs/clear": {
            operation: "logs.clear",
            records: [],
            targets: [
              { name: "crash.log", path: "C:/OpenCut/crash.log", bytes: 5 },
              { name: "opencut.log", path: "C:/OpenCut/opencut.log", bytes: 10 },
            ],
            reversible: false,
          },
        };
        if (body.dry_run) {
          const previewCount =
            (destructivePreviewCounts.get(url.pathname) || 0) + 1;
          destructivePreviewCounts.set(url.pathname, previewCount);
          const token = `${url.pathname}-token-${previewCount}`;
          destructiveTokens.set(url.pathname, token);
          const plan = {
            ...definitions[url.pathname],
            metadata: { route: url.pathname },
            confirm_token: token,
          };
          let payload;
          if (url.pathname === "/queue/clear") {
            payload = { success: true, dry_run: true, removed: 0, plan };
          } else if (url.pathname === "/logs/clear") {
            payload = {
              success: true,
              dry_run: true,
              plan,
              total_bytes: 15,
              cleared: [],
            };
          } else {
            payload = {
                  success: true,
                  dry_run: true,
                  destructive_plan: plan,
                  confirm_token: token,
                };
          }
          return route.fulfill({
            status: 200,
            contentType: "application/json",
            body: JSON.stringify(payload),
          });
        }
        if (
          body.confirm_token !== destructiveTokens.get(url.pathname) ||
          (url.pathname === "/queue/clear" && !queueTokenExpired)
        ) {
          if (url.pathname === "/queue/clear") queueTokenExpired = true;
          return route.fulfill({
            status: 409,
            contentType: "application/json",
            body: JSON.stringify({
              error: "confirm_token required",
              code: "DESTRUCTIVE_CONFIRMATION_REQUIRED",
              suggestion: "Refresh and review the plan.",
            }),
          });
        }
        const successPayloads = {
          "/presets/delete": { success: true, deleted: body.name },
          "/models/delete": { success: true, deleted: [body.path] },
          "/queue/clear": { success: true, removed: 1 },
          "/workflow/delete": { success: true, deleted: body.name },
          "/logs/clear": {
            success: true,
            cleared: ["crash.log", "opencut.log"],
            total_bytes: 15,
          },
        };
        return route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify(successPayloads[url.pathname]),
        });
      }
    }
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

async function openForcedColorSurface(page, ...args) {
  const result = await openSurface(page, ...args);
  // preparePage emulates the requested colour scheme after the test context
  // is created, so set forced-colors again after the document is loaded.
  await page.emulateMedia({ forcedColors: "active" });
  return result;
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
        // Placeholder and current value are not reliable accessible names
        // for text controls. Keep this oracle aligned with the platform name
        // computation: explicit ARIA, associated labels, title fallback, and
        // content-bearing controls only.
        const content = /^(BUTTON|A)$/.test(element.tagName)
          || ["button", "tab", "menuitem"].includes(element.getAttribute("role"))
          ? element.textContent
          : "";
        return [
          element.getAttribute("aria-label"),
          labelledBy,
          explicit,
          wrapping,
          element.getAttribute("title"),
          element.getAttribute("alt"),
          content,
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

async function assertProductionBoundaryContract(page, surfaceName, surface) {
  const emptyStateSelector = surfaceName === "cep"
    ? "#deliverablesSeqPill"
    : "#captionsSessionPill";
  await expect(page.locator("#connLabel")).toContainText(/offline|disconnected/i);
  await expect(page.locator(emptyStateSelector)).toHaveAttribute(
    "data-state",
    /^(empty|error)$/,
  );
  await expect(page.locator(surface.activePanelSelector)).toHaveCount(1);

  const statusNodes = await page.locator("[role='status']").count();
  expect(statusNodes).toBeGreaterThan(0);

  const firstTab = page.locator(surface.tabSelector).first();
  await firstTab.focus();
  await expect(firstTab).toBeFocused();
  await page.keyboard.press("Tab");
  const focusState = await page.evaluate(() => {
    const node = document.activeElement;
    if (!node) return { visible: false, insideDocument: false };
    const style = getComputedStyle(node);
    const rect = node.getBoundingClientRect();
    return {
      visible: style.display !== "none"
        && style.visibility !== "hidden"
        && rect.width > 0
        && rect.height > 0,
      insideDocument: document.documentElement.contains(node),
    };
  });
  expect(focusState).toEqual({ visible: true, insideDocument: true });
  expect(await visibleControlsWithoutNames(page)).toEqual([]);
  await assertNoPageOverflow(page);
}

test("panel styles use the compact radius scale without pill geometry", async () => {
  const styleUrls = [
    new URL("../../client/style.css", import.meta.url),
    new URL("../../client/command-center-layout.css", import.meta.url),
    new URL("../../client/command-center.css", import.meta.url),
    new URL("../../../com.opencut.uxp/style.css", import.meta.url),
    new URL("../../../com.opencut.uxp/command-center-layout.css", import.meta.url),
    new URL("../../../com.opencut.uxp/command-center.css", import.meta.url),
  ];
  const allowed = new Set([0, 4, 6, 8, 10, 12]);

  for (const styleUrl of styleUrls) {
    const source = await readFile(styleUrl, "utf8");
    expect(source, styleUrl.pathname).not.toMatch(
      /(?:border-radius\s*:\s*(?:999|9999)px|rounded-full|capsule\s*\()/i,
    );
    const declarations = source.matchAll(/border-radius\s*:\s*([^;]+);/gi);
    for (const declaration of declarations) {
      if (declaration[1].includes("%")) continue;
      for (const token of declaration[1].matchAll(/(\d+(?:\.\d+)?)px/g)) {
        expect(
          allowed.has(Number(token[1])),
          `${styleUrl.pathname}: ${declaration[0]}`,
        ).toBe(true);
      }
    }
  }
});

async function assertPremiumControlContract(page, surfaceName) {
  const violations = await page.evaluate(() => {
    const allowedRadii = new Set([0, 4, 6, 8, 10, 12]);
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
    return Array.from(
      document.querySelectorAll(
        "button, input:not([type='hidden']), select, textarea, [role='tab'], [role='menuitem']",
      ),
    )
      .filter(visible)
      .flatMap((element) => {
        const style = getComputedStyle(element);
        const rect = element.getBoundingClientRect();
        const radii = style.borderRadius.match(/[\d.]+/g)?.map(Number) || [0];
        const hasText = Boolean(
          (element.textContent || element.getAttribute("value") || "").trim(),
        );
        const issues = [];
        if (
          radii.some((radius) => !allowedRadii.has(radius)) &&
          !(rect.width <= 24 && rect.height <= 24)
        ) {
          issues.push(`radius=${style.borderRadius}`);
        }
        const fontSize = Number.parseFloat(style.fontSize);
        if (hasText && fontSize > 0 && fontSize < 12) {
          issues.push(`font=${style.fontSize}`);
        }
        if (
          element.tagName !== "INPUT" ||
          !["checkbox", "radio", "range"].includes(element.type)
        ) {
          if (rect.height < 28) issues.push(`height=${rect.height}`);
        }
        return issues.length
          ? [{
              tag: element.tagName.toLowerCase(),
              id: element.id,
              className: element.className,
              issues,
            }]
          : [];
      });
  });
  expect(violations, `${surfaceName} control contract`).toEqual([]);
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
            if (tabName === "captions") {
              const captionDisplaySelects = page.locator(
                "#captionDisplaySettingsCard select",
              );
              await expect(captionDisplaySelects).toHaveCount(7);
              for (let optionIndex = 0; optionIndex < 7; optionIndex += 1) {
                await expect(captionDisplaySelects.nth(optionIndex)).toBeDisabled();
                await expect(captionDisplaySelects.nth(optionIndex)).toHaveValue("");
              }
              await expect(page.locator("#capDispFont option:checked")).toHaveText(
                "Unavailable",
              );
              await expect(page.locator("#capDispPreviewBtn")).toBeDisabled();
            }
          } else {
            await expect(
              activePanel
                .locator("h1:visible, h2:visible, .oc-section-title:visible, .oc-workspace-title:visible")
                .first(),
            ).toBeVisible();
          }
          await assertNoPageOverflow(page);
          await assertPremiumControlContract(page, surfaceName);
          expect(
            await visibleControlsWithoutNames(page),
            `unnamed controls in ${surfaceName}/${tabName}`,
          ).toEqual([]);
          await assertWcagCompliance(
            page,
            `${surfaceName}/${theme}/${width}/${tabName || index}`,
          );
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

for (const [surfaceName, surface] of Object.entries(SURFACES)) {
  test(`${surfaceName} premium workspace hierarchy stays consistent across every page`, async ({
    page,
  }) => {
    const { pageErrors } = await openSurface(page, surfaceName, "dark", 1200, {
      height: 800,
    });
    const tabs = page.locator(surface.tabSelector);
    const count = await tabs.count();

    for (let index = 0; index < count; index += 1) {
      const tab = tabs.nth(index);
      const tabName = await tab.getAttribute(surface.tabAttribute);
      await tab.click();
      await expect(page.locator(surface.activePanelSelector)).toBeVisible();
      await expect(page).toHaveScreenshot(
        `${surfaceName}-page-${tabName || index}-dark-1200.png`,
        { fullPage: false },
      );
    }

    expect(pageErrors).toEqual([]);
  });
}

test("UXP follows live Premiere theme updates with legible tokens and cleanup", async ({
  page,
}) => {
  const { pageErrors } = await openSurface(page, "uxp", "dark", 520);

  const snapshot = async () => page.evaluate(() => {
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
    const contrast = (foreground, background) => {
      const foregroundLum = luminance(parseRgb(foreground));
      const backgroundLum = luminance(parseRgb(background));
      return (Math.max(foregroundLum, backgroundLum) + 0.05) /
        (Math.min(foregroundLum, backgroundLum) + 0.05);
    };
    const probe = document.createElement("span");
    probe.style.color = "var(--cc-text)";
    probe.style.backgroundColor = "var(--cc-bg)";
    document.body.appendChild(probe);
    const probeStyle = getComputedStyle(probe);
    const text = probeStyle.color;
    const background = probeStyle.backgroundColor;
    probe.remove();

    const sidebar = getComputedStyle(document.querySelector(".oc-header")).backgroundColor;
    const icon = getComputedStyle(document.querySelector(".oc-logo-mark path")).fill;
    const connection = getComputedStyle(document.getElementById("connectionStatus"));
    const secondary = getComputedStyle(document.querySelector("#tab-settings .oc-hint")).color;
    const surfaceValue = getComputedStyle(document.querySelector("#tab-settings .oc-settings-group")).backgroundColor;
    const surfaceParts = surfaceValue.match(/[\d.]+/g) || [];
    const surface = surfaceParts.length > 3 && Number(surfaceParts[3]) === 0
      ? background
      : surfaceValue;
    return {
      className: document.documentElement.className,
      theme: document.documentElement.dataset.premiereTheme,
      background,
      text,
      icon,
      connectionBackground: connection.backgroundColor,
      textContrast: contrast(text, background),
      iconContrast: contrast(icon, sidebar),
      secondaryContrast: contrast(secondary, surface),
      listenerCount: window.__opencutThemeHarness.listenerCount(),
    };
  });

  const dark = await snapshot();
  expect(dark.className).toContain("theme-dark");
  expect(dark.theme).toBe("dark");
  expect(dark.listenerCount).toBe(1);
  expect(dark.textContrast).toBeGreaterThanOrEqual(4.5);
  expect(dark.iconContrast).toBeGreaterThanOrEqual(3);
  expect(dark.secondaryContrast).toBeGreaterThanOrEqual(4.5);

  await page.evaluate(() => window.__opencutThemeHarness.emit("light"));
  await expect(page.locator("html")).toHaveClass(/theme-light/);
  const light = await snapshot();
  expect(light.theme).toBe("light");
  expect(light.background).not.toBe(dark.background);
  expect(light.text).not.toBe(dark.text);
  expect(light.textContrast).toBeGreaterThanOrEqual(4.5);
  expect(light.iconContrast).toBeGreaterThanOrEqual(3);
  expect(light.secondaryContrast).toBeGreaterThanOrEqual(4.5);
  expect(light.connectionBackground).toBe("rgba(0, 0, 0, 0)");

  await page.evaluate(() => window.__opencutThemeHarness.emit("darkest"));
  await expect(page.locator("html")).toHaveClass(/theme-darkest/);
  const darkest = await snapshot();
  expect(darkest.theme).toBe("darkest");
  expect(darkest.background).not.toBe(dark.background);
  expect(darkest.background).not.toBe(light.background);
  expect(darkest.textContrast).toBeGreaterThanOrEqual(4.5);
  expect(darkest.iconContrast).toBeGreaterThanOrEqual(3);
  expect(darkest.secondaryContrast).toBeGreaterThanOrEqual(4.5);

  await page.evaluate(() => window.dispatchEvent(new Event("beforeunload")));
  await expect.poll(
    () => page.evaluate(() => window.__opencutThemeHarness.listenerCount()),
  ).toBe(0);
  expect(pageErrors).toEqual([]);
});

test("CEP first run uses one accessible server-backed onboarding dialog", async ({
  page,
}) => {
  const { capturedRequests, pageErrors } = await openSurface(
    page,
    "cep",
    "dark",
    480,
    { onboardingState: { seen: false, step: 0 } },
  );
  const dialog = page.locator("#wizardOverlay");
  await expect(dialog).toBeVisible();
  await expect(dialog).toHaveAccessibleName("Welcome to OpenCut");
  await expect(page.locator("[role='dialog']:visible")).toHaveCount(1);
  await expect.poll(
    () => page.evaluate(() => document.activeElement?.id),
  ).toBe("ocOnboardingTitle");
  await expect(page).toHaveScreenshot("cep-onboarding-first-run-480.png");

  const actions = dialog.locator(".oc-onboarding-actions button");
  await expect(actions).toHaveCount(2);
  await actions.last().focus();
  await page.keyboard.press("Tab");
  await expect(actions.first()).toBeFocused();

  await page.keyboard.press("Escape");
  await expect(dialog).toBeHidden();
  await expect(page.locator("#stageChooseMediaBtn")).toBeFocused();
  await expect.poll(() => capturedRequests.some(
    (request) => request.onboarding === "POST" && request.body?.seen === true,
  )).toBe(true);

  await page.locator(".nav-tab[data-nav='settings']").click();
  const restart = page.locator("#ocWaveHRestartTour");
  await restart.click();
  await expect(dialog).toBeVisible();
  await expect(dialog).toHaveAccessibleName("Welcome to OpenCut");
  await expect.poll(() => capturedRequests.some(
    (request) => request.onboarding === "POST" &&
      request.body?.seen === false && request.body?.step === 0,
  )).toBe(true);
  await dialog.getByRole("button", { name: "Skip" }).click();
  await expect(dialog).toBeHidden();
  await expect(restart).toBeFocused();
  expect(pageErrors).toEqual([]);
});

test("CEP onboarding exposes explicit backend-offline recovery", async ({ page }) => {
  const { capturedRequests, pageErrors } = await openSurface(
    page,
    "cep",
    "dark",
    480,
    { onboardingUnavailable: true },
  );
  const dialog = page.locator("#wizardOverlay");
  // The startup probe still runs, but it must stay silent while the backend
  // is unreachable — the focus-trapped recovery card is reserved for
  // explicit restart-tour actions.
  await expect.poll(() => capturedRequests.filter(
    (request) => request.onboarding === "GET",
  ).length).toBeGreaterThanOrEqual(1);
  await expect(dialog).toBeHidden();

  await page.evaluate(() => window.OpenCutWaveH.restartOnboarding());
  await expect(dialog).toBeVisible();
  await expect(dialog).toHaveAccessibleName("Tour unavailable");
  await expect(dialog).toContainText("local backend");
  await expect(page.locator("[role='dialog']:visible")).toHaveCount(1);
  await expect(page).toHaveScreenshot("cep-onboarding-offline-480.png");

  const retry = dialog.getByRole("button", { name: "Retry" });
  await retry.click();
  await expect(dialog).toHaveAccessibleName("Tour unavailable");
  await expect.poll(() => capturedRequests.filter(
    (request) => request.onboarding === "POST",
  ).length).toBeGreaterThanOrEqual(2);

  await dialog.getByRole("button", { name: "Continue without tour" }).click();
  await expect(dialog).toBeHidden();
  await expect(page.locator("#stageChooseMediaBtn")).toBeFocused();
  expect(pageErrors).toEqual([]);
});

test("UXP first run guides a connected user from media to review", async ({ page }) => {
  const { capturedRequests, pageErrors } = await openSurface(
    page,
    "uxp",
    "dark",
    520,
    { boundaryReview: true, onboardingState: { seen: false, step: 0 } },
  );
  const overlay = page.locator("#uxpOnboardingOverlay");
  await expect(overlay).toBeVisible();
  await expect(overlay).toHaveAttribute("aria-hidden", "false");
  await expect(page.locator("#uxpOnboardingTitle")).toHaveText("Start with a connected workspace");
  await page.locator("#uxpOnboardingNextBtn").click();
  await expect(page.locator("#uxpOnboardingTitle")).toHaveText("Choose your source media");
  await expect(page.locator("#uxpOnboardingActionBtn")).toHaveText("Choose Media");
  await page.locator("#uxpOnboardingActionBtn").click();
  await expect(overlay).toBeHidden();
  await expect.poll(() => capturedRequests.some(
    (request) => request.onboarding === "POST" && request.body?.seen === false,
  )).toBe(true);

  await page.locator(".oc-tab[data-tab='settings']").click();
  await page.locator("#uxpRestartOnboardingBtn").click();
  await expect(overlay).toBeVisible();
  await expect.poll(() => capturedRequests.some(
    (request) => request.onboarding === "POST" &&
      request.body?.seen === false && request.body?.step === 0,
  )).toBe(true);
  await page.locator("#uxpOnboardingSkipBtn").click();
  await expect(overlay).toBeHidden();
  await expect.poll(() => capturedRequests.some(
    (request) => request.onboarding === "POST" && request.body?.seen === true,
  )).toBe(true);
  expect(pageErrors).toEqual([]);
});

test("CEP workflow runs compile before approval and dispatch", async ({ page }) => {
  const { capturedRequests, pageErrors } = await openSurface(
    page,
    "cep",
    "dark",
    900,
    { workflowPreflight: true },
  );

  await page.evaluate(() => {
    const select = document.getElementById("clipSelect");
    const option = document.createElement("option");
    option.value = "C:/media/interview.mov";
    option.textContent = "interview.mov";
    option.setAttribute("data-name", "interview.mov");
    select.appendChild(option);
    select.value = option.value;
    select._customDropdown.update();
  });
  const clipTrigger = page.locator(
    ".custom-dropdown[data-for='clipSelect'] .custom-dropdown-trigger",
  );
  await clipTrigger.click();
  await page.locator("#clipSelect-listbox .custom-dropdown-item").last().click();

  await page.locator(".nav-tab[data-nav='export']").click();
  await page.locator("#exportSubTabs .sub-tab[data-sub='exp-batch']").click();
  const preset = page.locator("#workflowPreset");
  await expect(preset.locator("option")).toHaveCount(1);
  await page.locator(".custom-dropdown[data-for='workflowPreset'] .custom-dropdown-trigger").click();
  await page.locator("#workflowPreset-listbox .custom-dropdown-item[data-value='idx:0']").click();
  const run = page.locator("#runWorkflowBtn");
  await expect(run).toBeEnabled();
  await run.click();

  await expect.poll(() => capturedRequests.filter((item) => item.workflowCompile).length).toBe(1);
  const dialog = page.locator(".panel-dialog-overlay");
  await expect(dialog).toContainText("Review workflow side effects");
  await expect(dialog).toContainText("external network or remote service");
  expect(capturedRequests.some((item) => item.workflowRun)).toBe(false);

  await dialog.getByRole("button", { name: "Approve and run" }).click();
  await expect.poll(() => capturedRequests.filter((item) => item.workflowApprove).length).toBe(1);
  await expect.poll(() => capturedRequests.filter((item) => item.workflowRun).length).toBe(1);
  expect(capturedRequests.find((item) => item.workflowRun).workflowRun.plan.approval.approved).toBe(true);
  expect(pageErrors).toEqual([]);
});

test("CEP exposes interrupted queue recovery without overflowing", async ({ page }) => {
  const queueEntries = [
    {
      id: "recover-fixture",
      endpoint: "/silence",
      payload: { filepath: "C:/fixture.mp4" },
      status: "interrupted",
      added: 1,
    },
  ];
  const { capturedRequests, pageErrors } = await openSurface(
    page,
    "cep",
    "dark",
    480,
    { queueEntries },
  );

  await expect(page.locator("#jobQueueBar")).toBeVisible();
  await expect(page.locator("#queueStatusText")).toHaveText(
    "Queue: 0 active, 1 interrupted",
  );
  await expect(page.locator("#recoverQueueBtn")).toBeVisible();
  await assertNoPageOverflow(page);

  await page.locator("#recoverQueueBtn").click();
  await expect.poll(() => capturedRequests).toContainEqual({
    queueReplay: "recover-fixture",
  });
  await expect(page.locator("#recoverQueueBtn")).toBeHidden();
  await expect(page.locator("#queueStatusText")).toHaveText("Queue: 1 job");
  expect(pageErrors).toEqual([]);
});

test("CEP auditions uncertain ASR boundaries before timeline mutation", async ({
  page,
}) => {
  const { capturedRequests, pageErrors } = await openSurface(
    page,
    "cep",
    "dark",
    480,
    { boundaryReview: true },
  );

  await page.evaluate(() => {
    const select = document.getElementById("clipSelect");
    const option = document.createElement("option");
    option.value = "C:/media/interview.mov";
    option.textContent = "interview.mov";
    option.setAttribute("data-name", "interview.mov");
    select.appendChild(option);
    select.value = option.value;
    select._customDropdown.update();
  });
  const clipTrigger = page.locator(
    ".custom-dropdown[data-for='clipSelect'] .custom-dropdown-trigger",
  );
  await clipTrigger.click();
  await page.locator("#clipSelect-listbox .custom-dropdown-item").last().click();
  await page.locator(".nav-tab[data-nav='cut']").click();
  await page.locator("#cutSubTabs [data-sub='fillers']").click();
  await expect(page.locator("#runFillersBtn")).toBeEnabled();
  await page.locator("#runFillersBtn").click();

  const review = page.locator("#fillerBoundaryReview");
  await expect(review).toBeVisible({ timeout: 5000 });
  await expect(review).toContainText("um · 1.00–1.20s · boundary unavailable");
  await expect(page.locator("#resultsStats")).toContainText(
    "Systran/faster-whisper-base",
  );
  await expect(page.locator("#resultsStats")).toContainText(
    "decoder-token-timestamps",
  );
  await assertNoPageOverflow(page);

  await review.getByRole("button", { name: "Audition" }).click();
  await expect.poll(() => capturedRequests).toContainEqual({
    boundaryAudition: {
      filepath: "C:/media/interview.mov",
      start: 0.25,
      duration: 1.7,
      filter: "raw",
    },
  });
  await expect(page.locator("#fillerBoundaryPlayer")).toBeVisible();

  await page.locator("#applyFillerBoundariesBtn").click();
  await expect
    .poll(
      () =>
        capturedRequests.filter(
          (request) => request.fillers?.accept_low_confidence_boundaries,
        ).length,
    )
    .toBe(1);
  expect(
    capturedRequests.find(
      (request) => request.fillers?.accept_low_confidence_boundaries,
    ).fillers.filepath,
  ).toBe("C:/media/interview.mov");
  expect(pageErrors).toEqual([]);
});

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
        const tabName = await tab.getAttribute(surface.tabAttribute);
        await tab.click();
        await expect(page.locator("#workspaceOverviewTitle")).toBeVisible();
        if (tabName === "settings") {
          await expect(page.locator("#workspaceGuide")).toBeHidden();
        } else {
          await expect(page.locator("#workspaceGuide")).toBeVisible();
          await expect(page.locator("#workspaceGuideAction")).toBeVisible();
        }
        const geometry = await page.evaluate(() => {
          const nav = document.getElementById("tabNav")?.getBoundingClientRect();
          const selected = document
            .querySelector(".oc-tab[aria-selected='true']")
            ?.getBoundingClientRect();
          const title = document.getElementById("workspaceOverviewTitle")?.getBoundingClientRect();
          const state = document.getElementById("workspaceGuide")?.getBoundingClientRect();
          const action = document.getElementById("workspaceGuideAction")?.getBoundingClientRect();
          const settingsGroup = document.querySelector(
            "#tab-settings.active > .oc-settings-group:not([hidden])",
          )?.getBoundingClientRect();
          return {
            mainScrollTop: document.getElementById("mainContent")?.scrollTop || 0,
            nav,
            selected,
            title,
            state,
            action,
            settingsGroup,
          };
        });
        expect(geometry.mainScrollTop).toBe(0);
        expect(geometry.selected.left).toBeGreaterThanOrEqual(geometry.nav.left - 1);
        expect(geometry.selected.right).toBeLessThanOrEqual(geometry.nav.right + 1);
        for (const region of [geometry.title, geometry.state, geometry.action].filter(Boolean)) {
          expect(region.top).toBeGreaterThanOrEqual(0);
          expect(region.bottom).toBeLessThanOrEqual(800);
        }
        if (tabName === "settings") {
          expect(geometry.settingsGroup.width).toBeGreaterThanOrEqual(width - 32);
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
  expect(menuGeometry.detailRadius).toBe("8px");
  expect(menuGeometry.detailFontSize).toBeGreaterThanOrEqual(12);
  expect(menuGeometry.selectRadius).toBe("6px");
  expect(menuGeometry.selectHeight).toBeGreaterThanOrEqual(36);
  expect(menuGeometry.selectAppearance).toBe("none");
  expect(menuGeometry.selectArrow).toContain("data:image/svg+xml");
  expect(pageErrors).toEqual([]);
});

test("wide command-center shells expose editorial rails and settings grids", async ({
  page,
}) => {
  const cep = await openSurface(page, "cep", "dark", 1200, { height: 800 });
  const cepActionGrammar = await page.evaluate(() => {
    const quickActions = document.querySelector(".quick-actions")?.getBoundingClientRect();
    const stageActions = Array.from(document.querySelectorAll(".workspace-stage-actions .stage-action"))
      .map((action) => action.getBoundingClientRect());
    const label = document.querySelector("#panel-cut label");
    const labelStyle = label ? getComputedStyle(label) : null;
    const labelMarker = label ? getComputedStyle(label, "::before") : null;
    return {
      quickActionHeight: quickActions?.height || 0,
      stageActionRowSpread: stageActions.length
        ? Math.max(...stageActions.map((action) => action.top))
          - Math.min(...stageActions.map((action) => action.top))
        : 0,
      labelLetterSpacing: labelStyle?.letterSpacing || "",
      labelMarkerDisplay: labelMarker?.display || "",
    };
  });
  expect(cepActionGrammar.quickActionHeight).toBeLessThanOrEqual(54);
  expect(cepActionGrammar.stageActionRowSpread).toBeLessThan(2);
  expect(["normal", "0px"]).toContain(cepActionGrammar.labelLetterSpacing);
  expect(cepActionGrammar.labelMarkerDisplay).toBe("none");
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
      cardRadius: getComputedStyle(document.querySelector("#panel-settings.active > .card")).borderRadius,
      statusRadius: getComputedStyle(document.getElementById("statusBar")).borderRadius,
      utilityRadius: getComputedStyle(document.querySelector("#ocWaveHTryDemo")).borderRadius,
      utilityBackground: getComputedStyle(document.querySelector("#ocWaveHTryDemo")).backgroundColor,
      journalClearRadius: getComputedStyle(document.getElementById("journalClearBtn")).borderRadius,
    };
  });
  expect(cepGeometry.sidebarWidth).toBeGreaterThanOrEqual(160);
  expect(cepGeometry.cardColumns).toBeGreaterThan(200);
  expect(cepGeometry.bodyFontSize).toBeGreaterThanOrEqual(14);
  expect(cepGeometry.brandMetaDisplay).toBe("none");
  expect(cepGeometry.kickerDisplay).toBe("none");
  expect(cepGeometry.cardShadow).toBe("none");
  expect(cepGeometry.cardRadius).toBe("0px");
  expect(cepGeometry.statusRadius).toBe("0px");
  expect(cepGeometry.utilityRadius).toBe("0px");
  expect(cepGeometry.utilityBackground).toBe("rgba(0, 0, 0, 0)");
  expect(cepGeometry.journalClearRadius).toBe("0px");
  expect(cep.pageErrors).toEqual([]);

  const uxp = await openSurface(page, "uxp", "dark", 1200, { height: 800 });
  await page.locator(".oc-tab[data-tab='settings']").click();
  const uxpGeometry = await page.evaluate(() => {
    const rail = document.getElementById("tabNavShell")?.getBoundingClientRect();
    const tabs = getComputedStyle(document.getElementById("tabNav"));
    const header = document.querySelector(".oc-header")?.getBoundingClientRect();
    const overview = document.querySelector(".oc-workspace-overview")?.getBoundingClientRect();
    const commandBar = document.querySelector(".oc-workspace-actions")?.getBoundingClientRect();
    const settingsNav = document.querySelector("#tab-settings .oc-settings-nav")?.getBoundingClientRect();
    const visibleGroup = document.querySelector("#tab-settings.active > .oc-settings-group:not([hidden])");
    const group = visibleGroup?.getBoundingClientRect();
    const meta = getComputedStyle(document.querySelector(".oc-workspace-meta"));
    const metaItem = getComputedStyle(document.querySelector(".oc-workspace-meta-item"));
    const statusPill = getComputedStyle(document.querySelector("#tab-settings .oc-status-pill"));
    return {
      railWidth: rail?.width || 0,
      tabDirection: tabs.flexDirection,
      headerHeight: header?.height || 0,
      overviewHeight: overview?.height || 0,
      connectionRadius: getComputedStyle(document.querySelector(".oc-connection")).borderRadius,
      connectionBackground: getComputedStyle(document.querySelector(".oc-connection")).backgroundColor,
      settingsColumns: settingsNav && group ? group.left - settingsNav.left : 0,
      settingsNavItems: document.querySelectorAll("#tab-settings .oc-settings-nav-item").length,
      visibleGroups: document.querySelectorAll("#tab-settings.active > .oc-settings-group:not([hidden])").length,
      bodyFontSize: Number.parseFloat(getComputedStyle(document.body).fontSize),
      groupShadow: getComputedStyle(visibleGroup).boxShadow,
      groupRadius: getComputedStyle(visibleGroup).borderRadius,
      statusPillBorder: statusPill.borderTopWidth,
      statusPillRadius: statusPill.borderRadius,
      statusPillBackground: statusPill.backgroundColor,
      metaBackground: meta.backgroundColor,
      metaItemBackground: metaItem.backgroundColor,
      metaItemBorderTop: metaItem.borderTopWidth,
      commandBarInHeader: !!header && !!commandBar
        && commandBar.top >= header.top
        && commandBar.bottom <= header.bottom,
      guideDisplay: getComputedStyle(document.getElementById("workspaceGuide")).display,
      groupTitles: Array.from(document.querySelectorAll("#tab-settings.active > .oc-settings-group:not([hidden]) > .oc-section-title"))
        .map((title) => title.textContent?.trim()),
    };
  });
  expect(uxpGeometry.railWidth).toBeGreaterThanOrEqual(148);
  expect(uxpGeometry.railWidth).toBeLessThanOrEqual(160);
  expect(uxpGeometry.tabDirection).toBe("column");
  expect(uxpGeometry.headerHeight).toBeLessThanOrEqual(48);
  expect(uxpGeometry.overviewHeight).toBeLessThanOrEqual(90);
  expect(uxpGeometry.connectionRadius).toBe("0px");
  expect(uxpGeometry.connectionBackground).toBe("rgba(0, 0, 0, 0)");
  expect(uxpGeometry.settingsColumns).toBeGreaterThan(170);
  expect(uxpGeometry.settingsNavItems).toBe(9);
  expect(uxpGeometry.visibleGroups).toBe(1);
  expect(uxpGeometry.bodyFontSize).toBeGreaterThanOrEqual(14);
  expect(uxpGeometry.groupShadow).toBe("none");
  expect(uxpGeometry.groupRadius).toBe("0px");
  expect(uxpGeometry.statusPillBorder).toBe("0px");
  expect(uxpGeometry.statusPillRadius).toBe("0px");
  expect(uxpGeometry.statusPillBackground).toBe("rgba(0, 0, 0, 0)");
  expect(uxpGeometry.metaBackground).toBe("rgba(0, 0, 0, 0)");
  expect(uxpGeometry.metaItemBackground).toBe("rgba(0, 0, 0, 0)");
  expect(uxpGeometry.metaItemBorderTop).toBe("0px");
  expect(uxpGeometry.commandBarInHeader).toBe(true);
  expect(uxpGeometry.guideDisplay).toBe("none");
  expect(uxpGeometry.groupTitles).toEqual(["Workspace"]);
  await expect(page.locator("#workspaceChooseClipBtn")).toBeEnabled();
  await expect(page.locator("#settingsWorkspaceBackendValue")).toHaveText("Offline");
  await page.locator("#settingsNavDiagnostics").click();
  await expect(page.locator("#settingsDiagnosticsBackendValue")).toHaveText("Offline");
  await expect(page.locator("#settingsDiagnosticsEndpointValue")).toContainText("127.0.0.1");
  await expect(page.locator("#settingsDiagnosticsLastCheckValue")).toHaveText("Just now");
  await page.locator("#settingsDiagnosticsDetailsBtn").click();
  await expect(page.locator("#connectionDetails")).toHaveAttribute("open", "");
  await page.locator(".oc-tab[data-tab='captions']").click();
  await expect(page.locator("#captionsPlanModel")).toHaveCount(0);
  const controlGrammar = await page.evaluate(() => {
    const summaryItems = Array.from(document.querySelectorAll(".oc-inline-summary-grid--captions .oc-inline-stat"));
    const summaryTops = summaryItems.map((item) => item.getBoundingClientRect().top);
    const status = getComputedStyle(document.getElementById("captionsStatusLine"));
    return {
      summaryCount: summaryItems.length,
      summaryRowSpread: summaryTops.length
        ? Math.max(...summaryTops) - Math.min(...summaryTops)
        : 0,
      statusBackground: status.backgroundColor,
      statusRadius: status.borderRadius,
    };
  });
  expect(controlGrammar.summaryCount).toBe(3);
  expect(controlGrammar.summaryRowSpread).toBeLessThan(1);
  expect(controlGrammar.statusBackground).toBe("rgba(0, 0, 0, 0)");
  expect(controlGrammar.statusRadius).toBe("0px");
  await page.locator(".oc-tab[data-tab='cut']").click();
  const fieldGrammar = await page.locator("#clipPathCut").evaluate((field) => {
    const input = getComputedStyle(field);
    const select = getComputedStyle(document.getElementById("silenceMode"));
    return {
      inputRadius: input.borderRadius,
      inputTop: input.borderTopWidth,
      inputBottom: input.borderBottomWidth,
      selectRadius: select.borderRadius,
      selectTop: select.borderTopWidth,
      selectBottom: select.borderBottomWidth,
    };
  });
  expect(fieldGrammar).toEqual({
    inputRadius: "6px",
    inputTop: "1px",
    inputBottom: "1px",
    selectRadius: "6px",
    selectTop: "1px",
    selectBottom: "1px",
  });
  await page.locator("#clipPathCut").fill("C:/media/interview.mov");
  await page.locator(".oc-tab[data-tab='settings']").click();
  await page.locator("#settingsNavWorkspace").click();
  await expect(page.locator("#settingsWorkspaceSourceValue")).toHaveText("interview.mov");
  await page.locator("#settingsWorkspaceSearchBtn").click();
  await expect(page.locator(".oc-tab[data-tab='search']")).toHaveAttribute("aria-selected", "true");
  expect(uxp.pageErrors).toEqual([]);
});

test("UXP hierarchy keeps tertiary actions, suggestions, and notices visually open", async ({
  page,
}) => {
  const { pageErrors } = await openSurface(page, "uxp", "dark", 1200, {
    height: 800,
  });

  await page.locator(".oc-tab[data-tab='settings']").click();
  const settingsGrammar = await page.evaluate(() => {
    const action = getComputedStyle(document.getElementById("settingsWorkspaceSearchBtn"));
    return {
      footerCount: document.querySelectorAll(".oc-nav-footer").length,
      actionBorder: action.borderTopWidth,
      actionRadius: action.borderRadius,
      actionBackground: action.backgroundColor,
    };
  });
  expect(settingsGrammar).toEqual({
    footerCount: 0,
    actionBorder: "0px",
    actionRadius: "0px",
    actionBackground: "rgba(0, 0, 0, 0)",
  });

  await page.locator(".oc-tab[data-tab='search']").click();
  const suggestionGrammar = await page.locator(".oc-chip-group .oc-chip").first().evaluate((chip) => {
    const style = getComputedStyle(chip);
    return {
      borderTop: style.borderTopWidth,
      radius: style.borderRadius,
      background: style.backgroundColor,
    };
  });
  expect(suggestionGrammar).toEqual({
    borderTop: "0px",
    radius: "0px",
    background: "rgba(0, 0, 0, 0)",
  });

  await page.locator(".oc-tab[data-tab='timeline']").click();
  const noticeGrammar = await page.locator(".oc-uxp-notice").evaluate((notice) => {
    const style = getComputedStyle(notice);
    return {
      borderTop: style.borderTopWidth,
      borderLeft: style.borderLeftWidth,
      radius: style.borderRadius,
      background: style.backgroundColor,
    };
  });
  expect(noticeGrammar).toEqual({
    borderTop: "0px",
    borderLeft: "2px",
    radius: "0px",
    background: "rgba(0, 0, 0, 0)",
  });
  expect(pageErrors).toEqual([]);
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
  expect(initial.shellRadius).toBe("0px");
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
  expect(dropdownGeometry.itemHeight).toBeGreaterThanOrEqual(34);
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

  await page.evaluate(() => {
    const probe = document.createElement("input");
    probe.id = "placeholderOnlyNameProbe";
    probe.type = "text";
    probe.placeholder = "Placeholder only";
    probe.value = "Current value only";
    document.getElementById("renderedStateFixture").append(probe);
  });
  expect(await visibleControlsWithoutNames(page)).toEqual(
    expect.arrayContaining([
      expect.objectContaining({ id: "placeholderOnlyNameProbe" }),
    ]),
  );
  await page.locator("#placeholderOnlyNameProbe").evaluate((probe) => probe.remove());
  expect(pageErrors).toEqual([]);
});

test("production states stay semantic at every real breakpoint boundary", async ({
  page,
}) => {
  test.setTimeout(180_000);
  for (const surfaceName of ["cep", "uxp"]) {
    for (const theme of ["dark", "light"]) {
      for (const boundary of BREAKPOINT_BOUNDARIES[surfaceName]) {
        for (const width of [boundary - 1, boundary, boundary + 1]) {
          const { surface, pageErrors } = await openSurface(
            page,
            surfaceName,
            theme,
            width,
            { height: 800 },
          );
          await assertProductionBoundaryContract(page, surfaceName, surface);
          expect(pageErrors).toEqual([]);
        }
      }
    }
  }
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

test("CEP destructive controls preview signed plans before confirmation", async ({
  page,
}) => {
  const { pageErrors, capturedRequests } = await openSurface(
    page,
    "cep",
    "dark",
    900,
    { destructiveProtocol: true },
  );
  const dialog = page.locator(".panel-dialog-overlay");

  await page.locator("#navTabSettings").click();
  const clearLogs = page.locator("#clearLogsBtn");
  await clearLogs.click();
  await expect(dialog).toContainText("Clear diagnostic logs?");
  await expect(dialog).toContainText("crash.log");
  await expect(dialog).toContainText("opencut.log");
  await expect(dialog).toContainText("5 bytes");
  await expect(dialog).toContainText("10 bytes");
  await expect(dialog).toContainText("permanent and cannot be undone");
  await expect(dialog.getByRole("button", { name: "Cancel" })).toBeFocused();
  await dialog.getByRole("button", { name: "Cancel" }).click();
  await expect(dialog).toHaveCount(0);
  await expect(
    capturedRequests.filter((request) => request.path === "/logs/clear"),
  ).toHaveLength(1);
  await clearLogs.click();
  await dialog.getByRole("button", { name: "Clear Logs" }).click();
  await expect(dialog).toHaveCount(0);
  await expect(page.locator(".toast-notification[role='status']")).toContainText(
    "Cleared 2 diagnostic log files",
  );

  await expect(page.locator("#presetSelect option[value='Editorial Clean']")).toHaveCount(1);
  await page.locator("#presetSelect").evaluate((select) => {
    select.value = "Editorial Clean";
    select.dispatchEvent(new Event("change", { bubbles: true }));
  });
  const presetDelete = page.locator("#deletePresetBtn");
  await presetDelete.click();
  await expect(dialog).toContainText("Delete preset?");
  await expect(dialog).toContainText("Affected items: 1");
  await expect(dialog).toContainText("can be restored");
  const cancelButton = dialog.getByRole("button", { name: "Cancel" });
  const presetConfirm = dialog.getByRole("button", { name: "Delete Preset" });
  await expect(cancelButton).toBeFocused();
  await page.keyboard.press("Shift+Tab");
  await expect(presetConfirm).toBeFocused();
  await page.keyboard.press("Tab");
  await expect(cancelButton).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(dialog).toHaveCount(0);
  await expect(presetDelete).toBeFocused();
  await expect
    .poll(
      () =>
        capturedRequests.filter(
          (request) => request.path === "/presets/delete",
        ).length,
    )
    .toBe(1);
  expect(
    capturedRequests.find((request) => request.path === "/presets/delete")
      .body,
  ).toEqual({ name: "Editorial Clean", dry_run: true });

  await presetDelete.click();
  await dialog.getByRole("button", { name: "Delete Preset" }).click();
  await expect(dialog).toHaveCount(0);
  await expect
    .poll(
      () =>
        capturedRequests.filter(
          (request) =>
            request.path === "/presets/delete" && request.body.confirm_token,
        ).length,
    )
    .toBe(1);

  await page.locator("#refreshModelsBtn").click();
  const modelDelete = page.locator(".model-item-delete");
  await expect(modelDelete).toBeVisible();
  await modelDelete.click();
  await expect(dialog).toContainText("Delete model?");
  await expect(dialog).toContainText("25165824 bytes");
  await expect(dialog).toContainText("permanent and cannot be undone");
  await expect(dialog.getByRole("button", { name: "Cancel" })).toBeFocused();
  await dialog.getByRole("button", { name: "Delete Model" }).click();
  await expect(dialog).toHaveCount(0);

  const clearQueue = page.locator("#clearQueueBtn");
  await expect(clearQueue).toBeVisible();
  await clearQueue.click();
  await expect(dialog).toContainText("Clear queued jobs?");
  await expect(dialog).toContainText("queued-fixture");
  await dialog.getByRole("button", { name: "Clear Queue" }).click();
  await expect(dialog).toContainText(/previous plan changed or expired/i);
  await expect(dialog.getByRole("button", { name: "Cancel" })).toBeFocused();
  await dialog.getByRole("button", { name: "Clear Queue" }).click();
  await expect(dialog).toHaveCount(0);

  await page.locator("#navTabExport").click();
  await page.locator("button[data-sub='exp-batch']").click();
  await expect(
    page.locator("#savedWorkflowSelect option[value='Fixture Workflow']"),
  ).toHaveCount(1);
  await page.locator("#savedWorkflowSelect").evaluate((select) => {
    select.value = "Fixture Workflow";
    select.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await page.locator("#deleteCustomWorkflowBtn").click();
  await expect(dialog).toContainText("Delete workflow?");
  await expect(dialog).toContainText("Fixture Workflow");
  await expect(dialog).toContainText("can be restored");
  await dialog.getByRole("button", { name: "Delete Workflow" }).click();
  await expect(dialog).toHaveCount(0);

  const destructiveRequests = capturedRequests.filter(
    (request) => request.destructive,
  );
  for (const path of [
    "/presets/delete",
    "/models/delete",
    "/queue/clear",
    "/workflow/delete",
    "/logs/clear",
  ]) {
    const requests = destructiveRequests.filter(
      (request) => request.path === path,
    );
    expect(requests[0].body.dry_run).toBe(true);
    expect(requests.some((request) => request.body.confirm_token)).toBe(true);
    expect(
      requests.findIndex((request) => request.body.confirm_token),
    ).toBeGreaterThan(0);
  }
  expect(
    destructiveRequests.filter((request) => request.path === "/queue/clear"),
  ).toHaveLength(4);
  expect(pageErrors).toEqual([]);
});

for (const surfaceName of ["cep", "uxp"]) {
  test(`${surfaceName} live bridge stop waits for confirmation and recovers`, async ({
    page,
  }) => {
    const width = surfaceName === "cep" ? 900 : 520;
    const { surface, pageErrors, capturedRequests } = await openSurface(
      page,
      surfaceName,
      "dark",
      width,
      {
        liveBridge: {
          stopDelayMs: 150,
          stopOutcomes: ["fail", "success"],
        },
      },
    );
    await page
      .locator(`${surface.tabSelector}[${surface.tabAttribute}='settings']`)
      .click();
    if (surfaceName === "uxp") {
      await page.locator("#settingsNavLiveUpdates").click();
    }

    const status = page.locator(
      surfaceName === "cep" ? "#wsStatusText" : "#uxpWsStatus",
    );
    const hint = page.locator(
      surfaceName === "cep" ? "#wsHint" : "#settingsBridgeStatus",
    );
    const connect = page.locator(
      surfaceName === "cep" ? "#wsConnectBtn" : "#uxpWsConnectBtn",
    );
    const stop = page.locator(
      surfaceName === "cep" ? "#wsStopBtn" : "#uxpWsStopBtn",
    );
    const errorToast = page
      .locator(
        surfaceName === "cep"
          ? ".toast-notification[role='alert']"
          : ".oc-toast[role='alert']",
      )
      .filter({ hasText: /still connected/i });
    const successToast = page
      .locator(
        surfaceName === "cep"
          ? ".toast-notification[role='status']"
          : ".oc-toast[role='status']",
      )
      .filter({ hasText: /bridge stopped/i });

    // The fixture bridge is already running, so the panel connects on load.
    await expect(status).toContainText(/connected/i);
    await expect(connect).toBeDisabled();
    await expect(stop).toBeEnabled();

    await stop.click();
    await expect(stop).toBeDisabled();
    await expect(status).toContainText(/connected/i);
    await expect(hint).toContainText(/stopping/i);
    await expect(hint).toContainText(/still connected/i);
    await expect(errorToast).toContainText(/still connected/i);
    await expect(stop).toBeEnabled();
    await expect(status).toContainText(/connected/i);

    await stop.click();
    await expect(stop).toBeDisabled();
    await expect(status).toContainText(/connected/i);
    await expect(successToast).toContainText(/bridge stopped/i);
    await expect(status).toContainText(/bridge stopped/i);
    await expect
      .poll(
        () =>
          capturedRequests.filter((request) => request.liveBridgeStop).length,
      )
      .toBe(2);
    expect(pageErrors).toEqual([]);
  });
}

test("CEP Auto theme tracks the Premiere host skin, not the OS", async ({
  page,
}) => {
  // The OS is emulated light while the host skin is dark. Auto must follow
  // the host — that mismatch is the whole defect.
  const { pageErrors } = await openSurface(page, "cep", "auto", 900);
  const root = page.locator("html");

  await expect(root).toHaveAttribute("data-theme-source", "host");
  await expect(root).toHaveAttribute("data-premiere-theme", "darkest");
  expect(await root.evaluate((n) => n.classList.contains("theme-light"))).toBe(false);

  const listeners = await page.evaluate(() =>
    window.__opencutCepThemeHarness.listenerCount(),
  );
  expect(listeners).toBe(1);

  // Switching the skin inside Premiere must repaint without a reload.
  await page.evaluate(() => {
    const env = JSON.parse(window.__adobe_cep__.getHostEnvironment());
    env.appSkinInfo.panelBackgroundColor.color = {
      red: 180,
      green: 180,
      blue: 180,
      alpha: 255,
    };
    window.__opencutCepThemeHarness.setEnvironment(JSON.stringify(env));
    window.__opencutCepThemeHarness.emit();
  });
  await expect(root).toHaveAttribute("data-premiere-theme", "light");
  expect(await root.evaluate((n) => n.classList.contains("theme-light"))).toBe(true);

  // An explicit choice outranks the host and survives a further skin change.
  // The native select is replaced by a custom dropdown, so drive the same
  // change event the dropdown dispatches rather than the hidden control.
  await page.evaluate(() => {
    const select = document.getElementById("settingsTheme");
    select.value = "dark";
    select.dispatchEvent(new Event("change", { bubbles: true }));
  });
  await expect(root).toHaveAttribute("data-theme-source", "user");
  expect(await root.evaluate((n) => n.classList.contains("theme-light"))).toBe(false);

  await page.evaluate(() => window.__opencutCepThemeHarness.emit());
  await expect(root).toHaveAttribute("data-theme-source", "user");
  expect(await root.evaluate((n) => n.classList.contains("theme-light"))).toBe(false);

  expect(pageErrors).toEqual([]);
});

test("CEP terminal job results announce through live regions without trapping focus", async ({
  page,
}) => {
  const { pageErrors } = await openSurface(page, "cep", "dark", 900);

  const polite = page.locator("#resultsAnnouncePolite");
  const assertive = page.locator("#resultsAnnounceAssertive");

  await expect(polite).toHaveAttribute("role", "status");
  await expect(polite).toHaveAttribute("aria-live", "polite");
  await expect(assertive).toHaveAttribute("role", "alert");
  await expect(assertive).toHaveAttribute("aria-live", "assertive");

  // Present to assistive technology but not to the sighted layout. A
  // display:none region is skipped by screen readers entirely.
  const geometry = await polite.evaluate((node) => {
    const style = getComputedStyle(node);
    const rect = node.getBoundingClientRect();
    return { display: style.display, visibility: style.visibility, width: rect.width };
  });
  expect(geometry.display).not.toBe("none");
  expect(geometry.visibility).not.toBe("hidden");
  expect(geometry.width).toBeLessThanOrEqual(2);

  // Drive the production announce module against the production markup.
  const run = (tone, message) =>
    page.evaluate(
      ([t, m]) =>
        window.OpenCutAnnounce.announceResult(
          {
            polite: document.getElementById("resultsAnnouncePolite"),
            assertive: document.getElementById("resultsAnnounceAssertive"),
          },
          t,
          m,
        ),
      [tone, message],
    );

  await run("polite", "Run finished. 3 segments");
  await expect(polite).toHaveText("Run finished. 3 segments");
  await expect(assertive).toHaveText("");

  // A later failure must not leave the stale success text behind.
  await run("error", "Run failed: disk full Use the Retry button.");
  await expect(assertive).toContainText("Run failed: disk full");
  await expect(polite).toHaveText("");

  // Focus is only rescued when it was stranded; a usable control keeps it.
  const stranded = await page.evaluate(() => {
    const doc = document;
    // A control the user can actually still reach right now.
    const usable = Array.from(doc.querySelectorAll("button")).find(
      (node) => !node.disabled && node.getBoundingClientRect().width > 0,
    );
    // The Retry button lives inside the results card, which is hidden until
    // a run fails — focus parked there has genuinely nowhere to go.
    const inHiddenCard = doc.getElementById("newJobBtn");
    return {
      onBody: window.OpenCutAnnounce.focusWasStranded(doc.body, doc),
      onUsable: window.OpenCutAnnounce.focusWasStranded(usable, doc),
      onHidden: window.OpenCutAnnounce.focusWasStranded(inHiddenCard, doc),
      usableId: usable ? usable.id || usable.className : null,
    };
  });
  expect(stranded.usableId).toBeTruthy();
  expect(stranded.onBody).toBe(true);
  expect(stranded.onUsable).toBe(false);
  expect(stranded.onHidden).toBe(true);

  // Rescuing focus must not leave a permanent tab stop behind.
  const tabindex = await page.locator("#resultsSection").getAttribute("tabindex");
  expect(tabindex).toBe("-1");

  await page.evaluate(() =>
    window.OpenCutAnnounce.clearAnnouncements({
      polite: document.getElementById("resultsAnnouncePolite"),
      assertive: document.getElementById("resultsAnnounceAssertive"),
    }),
  );
  await expect(polite).toHaveText("");
  await expect(assertive).toHaveText("");
  expect(pageErrors).toEqual([]);
});

test("UXP timeline reports the rendered CEP fallback honestly", async ({
  page,
}) => {
  const { pageErrors } = await openSurface(page, "uxp", "dark", 520);
  await page.locator("#tabBtnTimeline").click();

  for (const selector of ["#timelineRenamePill", "#timelineSmartBinsPill"]) {
    await expect(page.locator(selector)).toHaveText("CEP fallback");
    await expect(page.locator(selector)).toHaveAttribute("data-state", "warning");
    await expect(page.locator(selector)).toHaveAttribute("title", /CEP panel/i);
  }
  await expect(page.locator("#runBatchRenameBtn")).toBeDisabled();
  await expect(page.locator("#runSmartBinsBtn")).toBeDisabled();
  expect(pageErrors).toEqual([]);
});

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
    if (surfaceName === "uxp") await page.locator("#settingsNavPlugins").click();
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
    if (surfaceName === "uxp") await page.locator("#settingsNavPlugins").click();
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

// ---------------------------------------------------------------------------
// Windows High Contrast / forced-colors
//
// In this mode the OS replaces every author colour, so any state the panels
// expressed only as a background tint — active tab, disabled control, focus
// ring, status severity — collapses into the same surface. These tests drive
// the panels with `forcedColors: "active"` and assert each distinction still
// resolves to a *different* computed value, which is the only thing that
// proves the rules took effect rather than merely existing in the stylesheet.
// ---------------------------------------------------------------------------
for (const surfaceName of Object.keys(SURFACES)) {
  test.describe(`${surfaceName} forced-colors`, () => {
    test.use({ forcedColors: "active" });

    test(`${surfaceName} keeps the active tab distinguishable`, async ({ page }) => {
      const width = surfaceName === "cep" ? 900 : 520;
      const { surface, pageErrors } = await openForcedColorSurface(
        page,
        surfaceName,
        "dark",
        width,
      );
      const active = page.locator(surface.activeTabSelector).first();
      await expect(active).toBeVisible();

      const [activeStyle, inactiveStyle] = await Promise.all([
        active.evaluate((el) => {
          const s = getComputedStyle(el);
          return { bg: s.backgroundColor, color: s.color, border: s.borderTopColor };
        }),
        page
          .locator(`${surface.tabSelector}:not(${surface.activeTabSelector})`)
          .first()
          .evaluate((el) => {
            const s = getComputedStyle(el);
            return { bg: s.backgroundColor, color: s.color, border: s.borderTopColor };
          }),
      ]);

      // Selection must differ by something the OS palette actually renders.
      const differs =
        activeStyle.bg !== inactiveStyle.bg ||
        activeStyle.color !== inactiveStyle.color ||
        activeStyle.border !== inactiveStyle.border;
      expect(differs, JSON.stringify({ activeStyle, inactiveStyle })).toBe(true);
      expect(pageErrors).toEqual([]);
    });

    test(`${surfaceName} shows a focus indicator that is not a tint`, async ({ page }) => {
      const width = surfaceName === "cep" ? 900 : 520;
      const { surface } = await openForcedColorSurface(page, surfaceName, "dark", width);
      const tab = page.locator(surface.tabSelector).first();
      await tab.focus();
      const outline = await tab.evaluate((el) => {
        const s = getComputedStyle(el);
        return { width: s.outlineWidth, style: s.outlineStyle };
      });
      expect(outline.style).not.toBe("none");
      expect(parseFloat(outline.width)).toBeGreaterThan(0);
    });

    test(`${surfaceName} keeps toast and progress boundaries visible`, async ({ page }) => {
      const width = surfaceName === "cep" ? 900 : 520;
      await openForcedColorSurface(page, surfaceName, "dark", width);
      const progressSelector = surfaceName === "cep" ? ".processing-track" : ".oc-progress-track";
      const toastClass = surfaceName === "cep" ? "toast-notification" : "oc-toast";

      await page.locator("#processingBanner").evaluate((element) => {
        element.classList.remove("hidden");
      });
      await page.evaluate((className) => {
        const toast = document.createElement("div");
        toast.className = className;
        toast.textContent = "Forced-colors boundary probe";
        (document.querySelector("#toastArea") || document.body).append(toast);
      }, toastClass);

      for (const selector of [progressSelector, `.${toastClass}`]) {
        const element = page.locator(selector).last();
        await expect(element).toBeVisible();
        const boundary = await element.evaluate((node) => {
          const style = getComputedStyle(node);
          const rect = node.getBoundingClientRect();
          return {
            width: style.borderTopWidth,
            style: style.borderTopStyle,
            visibleWidth: rect.width,
            visibleHeight: rect.height,
          };
        });
        expect(boundary.style, JSON.stringify({ selector, boundary })).toBe("solid");
        expect(parseFloat(boundary.width), JSON.stringify({ selector, boundary })).toBeGreaterThan(0);
        expect(boundary.visibleWidth, JSON.stringify({ selector, boundary })).toBeGreaterThan(0);
        expect(boundary.visibleHeight, JSON.stringify({ selector, boundary })).toBeGreaterThan(0);
      }
    });

    test(`${surfaceName} distinguishes disabled controls without a tint`, async ({ page }) => {
      const width = surfaceName === "cep" ? 900 : 520;
      await openForcedColorSurface(page, surfaceName, "dark", width);
      // Compare like with like: two controls of the same class, one disabled.
      // Comparing across classes would prove nothing about the disabled cue.
      const klass = surfaceName === "cep" ? "quick-action-btn" : "oc-btn";
      const disabled = page.locator(`button.${klass}:disabled`).first();
      const enabled = page.locator(`button.${klass}:not(:disabled)`).first();
      test.skip(
        (await disabled.count()) === 0 || (await enabled.count()) === 0,
        `initial view has no enabled/disabled .${klass} pair`,
      );
      const read = (locator) =>
        locator.evaluate((el) => {
          const s = getComputedStyle(el);
          return { color: s.color, border: s.borderTopColor };
        });
      const disabledStyle = await read(disabled);
      const enabledStyle = await read(enabled);
      // The tinted background that carried "disabled" is replaced by the OS,
      // so text or border has to carry it instead.
      expect(
        disabledStyle.color !== enabledStyle.color ||
          disabledStyle.border !== enabledStyle.border,
        JSON.stringify({ disabledStyle, enabledStyle }),
      ).toBe(true);
    });

    test(`${surfaceName} keeps the shell navigable`, async ({ page }) => {
      const width = surfaceName === "cep" ? 900 : 520;
      const { surface, pageErrors } = await openForcedColorSurface(
        page,
        surfaceName,
        "dark",
        width,
      );
      const tabs = page.locator(surface.tabSelector);
      const total = await tabs.count();
      expect(total).toBeGreaterThan(1);
      await tabs.nth(1).click();
      await expect(page.locator(surface.activePanelSelector)).toBeVisible();
      expect(pageErrors).toEqual([]);
    });
  });
}
