#!/usr/bin/env node
// Lightweight build-output smoke check for CI.
//
// The CEP manifest loads the Vite output at `client/dist/index.html`. This
// verifier keeps the source tree intact and checks that the release output is
// a real, minified bundle rather than a copied source file.
//
// Strict mode (default): every required source file exists and validates a
// present dist/ tree. Build-output mode (--require-build): also requires the
// dist/ tree to exist.

import { existsSync, readFileSync, statSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const panelRoot = resolve(__dirname, "..");
const requireBuild = process.argv.includes("--require-build");
const BUNDLE_FILE = "opencut-panel.js";
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
  "transcript-correction-controller.js",
  "gpu-selection-controller.js",
  "bootstrap.js",
  "main.js",
];
const STATIC_FILES = ["client/locales/en.json"];

const SOURCE_FILES = [
  "client/index.html",
  "client/main.js",
  "client/panel-utils.js",
  "client/style.css",
  ...CLASSIC_SCRIPT_FILES
    .filter((fileName) => fileName !== "main.js")
    .map((fileName) => `client/${fileName}`),
  ...STATIC_FILES,
  "CSXS/manifest.xml",
  "host/index.jsx",
];

const failures = [];

for (const rel of SOURCE_FILES) {
  const abs = resolve(panelRoot, rel);
  if (!existsSync(abs)) {
    failures.push(`missing source file: ${rel}`);
    continue;
  }
  const stat = statSync(abs);
  if (!stat.isFile() || stat.size === 0) {
    failures.push(`empty or non-file: ${rel}`);
  }
}

const distIndex = resolve(panelRoot, "client/dist/index.html");
if (existsSync(distIndex)) {
  const html = readFileSync(distIndex, "utf8");
  const bundleTag = `<script src="${BUNDLE_FILE}"></script>`;
  const bundleTagCount = html.split(bundleTag).length - 1;
  if (bundleTagCount !== 1) {
    failures.push(`client/dist/index.html must reference ${BUNDLE_FILE} exactly once`);
  }
  for (const fileName of CLASSIC_SCRIPT_FILES) {
    if (html.includes(`<script src="${fileName}"`)) {
      failures.push(`client/dist/index.html still references source script ${fileName}`);
    }
  }
  if (html.includes("OPENCUT_CEP_BUNDLE:")) {
    failures.push("client/dist/index.html contains an unresolved CEP bundle marker");
  }
  if (html.includes('href="/') || html.includes('src="/')) {
    failures.push("client/dist/index.html contains an absolute asset URL");
  }

  const bundlePath = resolve(panelRoot, `client/dist/${BUNDLE_FILE}`);
  if (!existsSync(bundlePath)) {
    failures.push(`missing build artifact: client/dist/${BUNDLE_FILE}`);
  } else {
    const bundle = readFileSync(bundlePath, "utf8");
    if (bundle.trim().length === 0) {
      failures.push(`empty build artifact: client/dist/${BUNDLE_FILE}`);
    }
    const classicSourcePaths = CLASSIC_SCRIPT_FILES.map(
      (fileName) => resolve(panelRoot, `client/${fileName}`),
    );
    if (classicSourcePaths.every((sourcePath) => existsSync(sourcePath))) {
      const sourceBundle = classicSourcePaths
        .map((sourcePath) => readFileSync(sourcePath, "utf8"))
        .join("\n;\n");
      if (bundle === sourceBundle || classicSourcePaths.some(
        (sourcePath) => bundle === readFileSync(sourcePath, "utf8"),
      )) {
        failures.push(`client/dist/${BUNDLE_FILE} is byte-identical to a source script`);
      }
      if (bundle.length >= sourceBundle.length) {
        failures.push(`client/dist/${BUNDLE_FILE} is not smaller than the classic source bundle`);
      }
    }
  }

  for (const rel of STATIC_FILES) {
    const sourcePath = resolve(panelRoot, rel);
    const distPath = resolve(panelRoot, `client/dist/${rel.slice("client/".length)}`);
    if (!existsSync(distPath)) {
      failures.push(`missing static build asset: client/dist/${rel.slice("client/".length)}`);
    } else if (existsSync(sourcePath) && readFileSync(distPath, "utf8") !== readFileSync(sourcePath, "utf8")) {
      failures.push(`static build asset differs from source: ${rel}`);
    }
  }
} else if (requireBuild) {
  failures.push("client/dist/index.html missing (run `npm run build` before --require-build)");
}

if (failures.length > 0) {
  console.error("[verify-build] FAILED:");
  for (const f of failures) console.error(`  - ${f}`);
  process.exit(1);
}

console.log("[verify-build] OK — panel source tree intact" + (existsSync(distIndex) ? " + bundled dist output healthy" : ""));
