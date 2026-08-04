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
  "component-utils.js",
  "announce-utils.js",
  "cep-theme.js",
  "timeline-utils.js",
  "onboarding-state.js",
  "update-controller.js",
  "results-controller.js",
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
