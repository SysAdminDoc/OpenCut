#!/usr/bin/env node
// Lightweight build-output smoke check for CI.
//
// The CEP manifest loads `client/index.html` directly (not the Vite output),
// so this verifier only needs to confirm the source tree is intact and that
// the optional `client/dist/` output is in a healthy state when present.
//
// Strict mode (default): every required source file exists. Build-output
// mode (--require-build): also assert `client/dist/index.html` exists and
// references the classic CEP scripts.

import { existsSync, readFileSync, statSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const panelRoot = resolve(__dirname, "..");
const requireBuild = process.argv.includes("--require-build");

const SOURCE_FILES = [
  "client/index.html",
  "client/main.js",
  "client/panel-utils.js",
  "client/style.css",
  "client/CSInterface.js",
  "client/feature-state.js",
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
  for (const tag of ["CSInterface.js", "panel-utils.js", "feature-state.js", "main.js"]) {
    if (!html.includes(`<script src="${tag}"`)) {
      failures.push(`client/dist/index.html lost classic script reference for ${tag}`);
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

console.log("[verify-build] OK — panel source tree intact" + (existsSync(distIndex) ? " + dist output healthy" : ""));
