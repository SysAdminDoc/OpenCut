import { defineConfig, devices } from "@playwright/test";

// Each test builds its own page and backend mocks, so the tests spread across
// workers. A worker is a Node process plus a Chromium, about 1 GB with the axe
// scans, which keeps the default low enough for a memory-capped release gate.
// OPENCUT_RENDERED_WORKERS sets another count.
const DEFAULT_WORKERS = 2;
const requestedWorkers = process.env.OPENCUT_RENDERED_WORKERS?.trim();
if (requestedWorkers && !/^[1-9]\d*$/.test(requestedWorkers)) {
  throw new Error(`OPENCUT_RENDERED_WORKERS must be a whole number of 1 or more, not "${requestedWorkers}"`);
}

export default defineConfig({
  testDir: "./tests/rendered",
  fullyParallel: true,
  workers: requestedWorkers ? Number(requestedWorkers) : DEFAULT_WORKERS,
  timeout: 90_000,
  expect: {
    timeout: 10_000,
    toHaveScreenshot: {
      animations: "disabled",
      maxDiffPixelRatio: 0.01,
    },
  },
  snapshotPathTemplate: "{testDir}/__screenshots__/{projectName}/{arg}{ext}",
  reporter: process.env.CI ? [["line"]] : [["list"]],
  use: {
    ...devices["Desktop Chrome"],
    baseURL: "http://127.0.0.1:41737",
    headless: true,
    locale: "en-US",
    timezoneId: "UTC",
    reducedMotion: "reduce",
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: { browserName: "chromium" },
    },
  ],
  webServer: {
    command: "python -m http.server 41737 --bind 127.0.0.1 --directory ../..",
    url: "http://127.0.0.1:41737/extension/com.opencut.panel/client/index.html",
    reuseExistingServer: false,
    timeout: 30_000,
    stdout: "ignore",
    stderr: "pipe",
  },
});
