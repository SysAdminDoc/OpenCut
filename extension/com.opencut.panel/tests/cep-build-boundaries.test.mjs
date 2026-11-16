import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

const panelRoot = new URL("../", import.meta.url);
const repoRoot = new URL("../../../", import.meta.url);

function readPanel(relativePath) {
  return readFileSync(new URL(relativePath, panelRoot), "utf8");
}

function readRepo(relativePath) {
  return readFileSync(new URL(relativePath, repoRoot), "utf8");
}

const CLASSIC_SCRIPT_FILES = [
  "CSInterface.js",
  "panel-utils.js",
  "feature-state.js",
  "format-utils.js",
  "job-meta-utils.js",
  "classify-utils.js",
  "data-shape-utils.js",
  "string-utils.js",
  "lookup-utils.js",
  "i18n-utils.js",
  "panel-state.js",
  "backend-client.js",
  "job-runtime.js",
  "job-lifecycle.js",
  "host-write-verification.js",
  "component-utils.js",
  "announce-utils.js",
  "cep-theme.js",
  "timeline-utils.js",
  "onboarding-state.js",
  "update-controller.js",
  "results-controller.js",
  "settings-diagnostics-controller.js",
  "navigation-controller.js",
  "collapsible-cards.js",
  "disabled-reasons.js",
  "bootstrap.js",
  "main.js",
];

describe("CEP production build", () => {
  it("launches the bundled runtime from the generated index", () => {
    const manifest = readPanel("CSXS/manifest.xml");
    const index = readPanel("client/dist/index.html");

    expect(manifest).toContain("<MainPath>./client/dist/index.html</MainPath>");
    expect(index.match(/<script src="opencut-panel\.js"><\/script>/g)).toHaveLength(1);
    expect(index).toContain('href="./index.css"');
    for (const fileName of CLASSIC_SCRIPT_FILES) {
      expect(index).not.toContain(`<script src="${fileName}"`);
    }
  });

  it("bundles every classic script the source index loads", () => {
    // A classic script missing from vite.config.mjs stays a separate tag after
    // the bundle and is never copied into dist. ws-bridge-port.js shipped that
    // way in v1.56.0, and the built panel stopped at its first reference.
    const listBlock = readPanel("vite.config.mjs").match(/const classicScriptFiles = \[([\s\S]*?)\];/);
    expect(listBlock).not.toBeNull();
    const bundled = [...listBlock[1].matchAll(/"([^"]+\.js)"/g)].map((match) => match[1]);
    const classicSources = (html) => [...html.matchAll(/<script\b([^>]*)>/g)]
      .filter((match) => !/\btype\s*=\s*"module"/.test(match[1]))
      .map((match) => (match[1].match(/\bsrc\s*=\s*"([^"]+)"/) || [])[1])
      .filter(Boolean);

    const sourceScripts = classicSources(readPanel("client/index.html"));
    expect(sourceScripts.length).toBeGreaterThan(10);
    for (const src of sourceScripts) expect(bundled, src).toContain(src);
    expect(classicSources(readPanel("client/dist/index.html"))).toEqual(["opencut-panel.js"]);
  });

  it("keeps the built artifact distinct from every classic source file", () => {
    const bundle = readPanel("client/dist/opencut-panel.js");
    const sources = CLASSIC_SCRIPT_FILES.map((fileName) => readPanel(`client/${fileName}`));
    const sourceBundle = sources.join("\n;\n");

    expect(bundle).not.toBe(sourceBundle);
    for (const source of sources) expect(bundle).not.toBe(source);
    expect(bundle.length).toBeLessThan(sourceBundle.length);
    expect(readPanel("client/dist/locales/en.json")).toBe(readPanel("client/locales/en.json"));
  });

  it("keeps installer staging and verification tied to the bundle", () => {
    const installerBuilder = readRepo("installer/InstallerBuilder.ps1");
    const innoScript = readRepo("OpenCut.iss");
    const verifier = readPanel("scripts/verify-build.mjs");

    expect(installerBuilder).toContain("client\\dist\\opencut-panel.js");
    expect(installerBuilder).toContain("robocopy $extensionDir $extDest /E");
    expect(innoScript).toContain('Source: "extension\\com.opencut.panel\\*"');
    expect(verifier).toContain("byte-identical to a source script");
  });
});

describe("panel controller budgets", () => {
  const budgets = [
    // F303 added the panel bootstrap wiring; the reader itself lives in
    // client/panel-bootstrap-token.js, so only the wiring counts here.
    // F367 wired a poll failure budget into both CEP poll loops (the state
    // machine itself lives in client/job-runtime.js). F377 then deleted the
    // dead bindings behind the panel's 25 lint warnings, which paid for that
    // wiring outright, so the original ceiling still holds.
    // F389 spends one line of that ceiling on `onBootstrapRefused`, the retry
    // that lets a panel which loaded before the backend recover its CSRF
    // token. The eager-capture fix it ships alongside is line-neutral:
    // `evalScript: jsx` replaced a three-line ternary.
    // F415 raises the ceiling by ten for the live-updates bridge port. The
    // panel dialled a hardcoded 5680 while /ws/start walks 5680-5689 for a
    // free port and skips the HTTP port, so it opened a socket to nothing
    // whenever the first choice was taken. The ten lines are `_wsBridgePort`,
    // `rememberWsBridgePort` and its two call sites, and three comments.
    // A review then found that recording the port was not enough: `wsConnect`
    // read `_wsBridgePort` synchronously, so the very first connect after a
    // panel load still used 5680 because nothing had answered yet. Resolving
    // the port before dialling cost twelve lines, and moving the tracker into
    // client/ws-bridge-port.js gave five of them back, which is the point of
    // this budget: logic belongs in a module a test can call.
    // F466 raises the ceiling by thirty-six for add-on model weights. It
    // first landed a hundred lines in main.js and nothing ran this test. The
    // decisions (model missing, download blocked, hint text, the weights-only
    // install) now live in client/capability-status.js. What stays is its
    // fifteen-line factory wiring, the CrisperWhisper filler-backend choice
    // between Whisper and a model download (thirteen), and the multimodal,
    // depth and emotion hint call sites (eight).
    // F447 adds thirty-one for the out-of-process capability probe. Reading
    // the probe (which failures are new, the GPU name, whether Whisper is
    // still being checked) lives in client/capability-status.js too. main.js
    // keeps the toast and GPU badge in the health poll (four), the Whisper
    // card's checking state (twenty-six, the same shape as its installed and
    // missing states) and one more line for the failed-check tooltip.
    // Two more let the health poll refresh that card when the probe reports
    // after Settings opened. Deciding when to refresh lives in
    // client/capability-status.js, so main.js only records what it showed
    // and asks. Ticking the CPU mode checkbox inside that redraw, instead of
    // in a separate Settings hook the poll skipped, gave two back.
    ["client/main.js", 18900],
    ["client/update-controller.js", 300],
    ["client/results-controller.js", 300],
    ["client/settings-diagnostics-controller.js", 110],
    ["client/navigation-controller.js", 220],
    ["../com.opencut.uxp/main.js", 10175],
    ["../com.opencut.uxp/uxp-ui-controller.js", 500],
    ["../com.opencut.uxp/uxp-update-controller.js", 330],
    ["../com.opencut.uxp/uxp-settings-controller.js", 620],
  ];

  it("keeps controller size and ownership churn inside explicit budgets", () => {
    for (const [relativePath, maxLines] of budgets) {
      const source = readPanel(relativePath);
      expect(source.split(/\r?\n/).length, relativePath).toBeLessThanOrEqual(maxLines);
    }

    const cepMain = readPanel("client/main.js");
    const uxpMain = readPanel("../com.opencut.uxp/main.js");
    for (const owner of [
      "OpenCutUpdateController.createUpdateController",
      "OpenCutResultsController.createResultsController",
      "OpenCutSettingsDiagnosticsController.createSettingsDiagnosticsController",
      "OpenCutNavigationController.createNavigationController",
    ]) {
      expect(cepMain).toContain(owner);
    }
    for (const owner of [
      'import { createUxpUiController } from "./uxp-ui-controller.js";',
      'import { createUxpUpdateController } from "./uxp-update-controller.js";',
      'import { createUxpSettingsController } from "./uxp-settings-controller.js";',
    ]) {
      expect(uxpMain).toContain(owner);
    }
    for (const duplicate of [
      "function checkForUpdateNotice(force)",
      "function showResults(job)",
      "function loadSettingsInfo()",
      "function setupNavTabs()",
    ]) {
      expect(cepMain).not.toContain(duplicate);
    }
    for (const teardown of [
      "UpdateController.dispose();",
      "ResultsController.dispose();",
      "SettingsDiagnosticsController.dispose();",
      "NavigationController.dispose();",
    ]) {
      expect(cepMain).toContain(teardown);
    }
    for (const teardown of [
      "UIController.dispose();",
      "UxpUpdateController.dispose();",
      "UxpSettingsController.dispose();",
    ]) {
      expect(uxpMain).toContain(teardown);
    }
  });
});
