#!/usr/bin/env node
// F131 — assert every resolved esbuild instance is >=0.25.0.
//
// F095 pinned esbuild via package.json `overrides`. This script
// verifies the pin actually takes effect inside the resolved tree
// so a future hoisting / re-resolution can't silently downgrade us
// back into the GHSA-67mh-4wv8-2f99 vulnerable range.
//
// Output:
//   exit 0 — no esbuild is resolved, or every resolved esbuild >=0.25.0
//   exit 1 — at least one resolved esbuild <0.25.0 (or unparseable)
//
// Flags:
//   --json  Emit a machine-readable summary on stdout
//
// Used by both `npm run audit:esbuild` and the Python release_smoke
// `esbuild-pin` step.

import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const panelRoot = resolve(__dirname, "..");

const JSON_MODE = process.argv.includes("--json");
const MIN_MAJOR = 0;
const MIN_MINOR = 25;
const MIN_PATCH = 0;

function runLs() {
  const isWin = process.platform === "win32";
  const quotedPanelRoot = panelRoot.replace(/'/g, "''");
  const result = isWin
    ? spawnSync(
        "powershell.exe",
        [
          "-NoProfile",
          "-ExecutionPolicy",
          "Bypass",
          "-Command",
          `Set-Location -LiteralPath '${quotedPanelRoot}'; npm ls esbuild --all --json`,
        ],
        { encoding: "utf8", maxBuffer: 32 * 1024 * 1024 },
      )
    : spawnSync("npm", ["ls", "esbuild", "--all", "--json"], {
        cwd: panelRoot,
        encoding: "utf8",
        maxBuffer: 32 * 1024 * 1024,
      });
  if (result.error) {
    throw new Error(`npm ls failed to spawn: ${result.error.message}`);
  }
  // `npm ls` exits non-zero when "extraneous" or missing dependencies are
  // reported, but the JSON body is still valid. Don't gate on returncode.
  try {
    return JSON.parse(result.stdout || "{}");
  } catch (err) {
    throw new Error(
      `npm ls produced unparseable JSON (exit ${result.status}): ${err.message}\n` +
        `--- stdout ---\n${(result.stdout || "").slice(0, 4000)}\n` +
        `--- stderr ---\n${(result.stderr || "").slice(0, 2000)}`,
    );
  }
}

function compareVersion(version) {
  // Strip leading whitespace, "v" prefixes, and pre-release suffixes.
  const cleaned = String(version || "").trim().replace(/^v/, "").split("-")[0];
  const parts = cleaned.split(".").map((p) => parseInt(p, 10));
  while (parts.length < 3) parts.push(0);
  const [maj, min, patch] = parts;
  if (Number.isNaN(maj) || Number.isNaN(min) || Number.isNaN(patch)) {
    return { satisfied: false, parsed: null };
  }
  if (maj > MIN_MAJOR) return { satisfied: true, parsed: { maj, min, patch } };
  if (maj < MIN_MAJOR) return { satisfied: false, parsed: { maj, min, patch } };
  if (min > MIN_MINOR) return { satisfied: true, parsed: { maj, min, patch } };
  if (min < MIN_MINOR) return { satisfied: false, parsed: { maj, min, patch } };
  if (patch >= MIN_PATCH) return { satisfied: true, parsed: { maj, min, patch } };
  return { satisfied: false, parsed: { maj, min, patch } };
}

function collect(node, location, accum) {
  if (!node || typeof node !== "object") return;
  if (node.name === "esbuild" || (location && location.endsWith("/esbuild"))) {
    const version = node.version;
    accum.push({ location: location || "(root)", version, name: node.name || "esbuild" });
  }
  const deps = node.dependencies || {};
  for (const [name, child] of Object.entries(deps)) {
    const nextLocation = `${location || ""}/${name}`;
    if (name === "esbuild") {
      accum.push({
        location: nextLocation,
        version: child && child.version,
        name,
      });
    }
    collect(child, nextLocation, accum);
  }
}

function main() {
  let payload;
  try {
    payload = runLs();
  } catch (err) {
    const report = {
      status: "error",
      message: err.message,
      min_version: `${MIN_MAJOR}.${MIN_MINOR}.${MIN_PATCH}`,
    };
    if (JSON_MODE) {
      console.log(JSON.stringify(report, null, 2));
    } else {
      console.error(`[esbuild-pin] ${err.message}`);
    }
    process.exit(1);
  }

  const instances = [];
  collect(payload, "", instances);

  // De-duplicate by location so nested traversal doesn't double-count.
  const seen = new Map();
  for (const entry of instances) {
    const key = `${entry.location}@${entry.version || "?"}`;
    if (!seen.has(key)) seen.set(key, entry);
  }
  const unique = Array.from(seen.values());

  const breached = [];
  const fine = [];
  for (const entry of unique) {
    if (!entry.version) {
      breached.push({ ...entry, reason: "missing version" });
      continue;
    }
    const cmp = compareVersion(entry.version);
    if (cmp.satisfied) fine.push(entry);
    else breached.push({ ...entry, reason: "below pin" });
  }

  const report = {
    status: breached.length === 0 ? "ok" : "fail",
    min_version: `${MIN_MAJOR}.${MIN_MINOR}.${MIN_PATCH}`,
    total_instances: unique.length,
    satisfied: fine.length,
    violations: breached,
  };

  if (JSON_MODE) {
    console.log(JSON.stringify(report, null, 2));
    process.exit(report.status === "ok" ? 0 : 1);
  }

  if (unique.length === 0) {
    console.log(
      "[esbuild-pin] OK — no resolved esbuild instances; " +
        "package.json override remains as a future transitive guard.",
    );
    process.exit(0);
  }

  if (breached.length > 0) {
    console.error("[esbuild-pin] FAIL: some esbuild instances violate the F095 pin:");
    for (const b of breached) {
      console.error(`  - ${b.location} = ${b.version || "?"} (${b.reason})`);
    }
    console.error(
      "\nFix: ensure package.json `overrides.esbuild` is still set " +
        "(>=0.25) and re-run `npm install` in extension/com.opencut.panel.",
    );
    process.exit(1);
  }

  console.log(
    `[esbuild-pin] OK — ${unique.length} esbuild instance(s) all >= ` +
      `${MIN_MAJOR}.${MIN_MINOR}.${MIN_PATCH}.`,
  );
  for (const f of fine) {
    console.log(`  - ${f.location} = ${f.version}`);
  }
}

main();
