#!/usr/bin/env node
// Verify the panel's npm advisory surface matches the documented allow-list.
//
// The CEP panel only runs `vite build` to produce the production bundle. The
// vite/esbuild dev-server CVEs do not apply to that workflow, but they still
// appear in `npm audit`. This script:
//
//   1. Forces a fresh `npm audit --json`.
//   2. Asserts every reported finding is on the documented allow-list with
//      a justification.
//   3. Exits non-zero on any unexpected advisory (high/critical, or anything
//      we haven't explicitly waived).
//
// Allow-list entries point at docs/NODE_ADVISORIES.md so the rationale stays
// versioned alongside the project.

import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const panelRoot = resolve(__dirname, "..");

const ALLOWED = new Map([
  [
    "GHSA-4w7w-66w2-5vf9",
    {
      package: "vite",
      severity: "moderate",
      reason:
        "Path traversal in optimized-deps .map handling. Only reachable via `vite dev`/`vite preview`; this project only ships `vite build` output. Tracked in docs/NODE_ADVISORIES.md.",
    },
  ],
]);

function runAudit() {
  const isWin = process.platform === "win32";
  // On Windows, `npm` resolves to `npm.cmd` which can only be launched via the
  // shell. We hardcode the argv (no user input) so shell expansion is safe.
  const result = isWin
    ? spawnSync("cmd.exe", ["/d", "/s", "/c", "npm audit --json"], {
        cwd: panelRoot,
        encoding: "utf8",
        maxBuffer: 64 * 1024 * 1024,
        windowsVerbatimArguments: true,
      })
    : spawnSync("npm", ["audit", "--json"], {
        cwd: panelRoot,
        encoding: "utf8",
        maxBuffer: 64 * 1024 * 1024,
      });
  if (result.error) {
    throw new Error(`npm audit failed to spawn: ${result.error.message}`);
  }
  try {
    return JSON.parse(result.stdout || "{}");
  } catch (err) {
    throw new Error(
      `npm audit produced unparseable JSON (exit ${result.status}): ${err.message}\n` +
        `--- stdout ---\n${result.stdout?.slice(0, 4000)}\n` +
        `--- stderr ---\n${result.stderr?.slice(0, 2000)}`,
    );
  }
}

function gatherAdvisoryIds(via) {
  if (!Array.isArray(via)) return [];
  const ids = [];
  for (const entry of via) {
    if (entry && typeof entry === "object" && entry.url) {
      const match = String(entry.url).match(/GHSA-[a-z0-9]{4}-[a-z0-9]{4}-[a-z0-9]{4}/i);
      if (match) ids.push(match[0]);
    }
  }
  return ids;
}

function main() {
  const report = runAudit();
  const vulns = report.vulnerabilities ?? {};
  const findings = [];
  const allowed = [];

  for (const [pkg, payload] of Object.entries(vulns)) {
    const ids = gatherAdvisoryIds(payload.via);
    const ok = ids.length > 0 && ids.every((id) => ALLOWED.has(id));
    if (ok) {
      allowed.push({ pkg, severity: payload.severity, ids });
    } else {
      findings.push({ pkg, severity: payload.severity, ids });
    }
  }

  if (findings.length > 0) {
    console.error("\n[advisory-check] Unwaived advisories detected:");
    for (const f of findings) {
      console.error(`  - ${f.pkg} (${f.severity}): ${f.ids.join(", ") || "<no GHSA id>"}`);
    }
    console.error(
      "\nAdd a justified entry to scripts/check-advisories.mjs ALLOWED + docs/NODE_ADVISORIES.md before re-running.",
    );
    process.exit(1);
  }

  if (allowed.length > 0) {
    console.log("[advisory-check] All advisories on documented allow-list:");
    for (const a of allowed) {
      console.log(`  - ${a.pkg} (${a.severity}): ${a.ids.join(", ")}`);
    }
  } else {
    console.log("[advisory-check] No advisories reported by npm audit.");
  }
}

main();
