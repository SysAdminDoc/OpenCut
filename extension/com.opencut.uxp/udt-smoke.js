const UDT_HARNESS_PATH = "uxp-udt-harness.json";

function parseOptions(options) {
  if (options == null) return {};
  if (typeof options === "string") {
    try {
      return JSON.parse(options);
    } catch (error) {
      return { parseError: error.message };
    }
  }
  return options;
}

function normalizeFilter(values) {
  if (!values) return null;
  const list = Array.isArray(values) ? values : [values];
  return new Set(list.map(value => String(value)));
}

async function loadManifest() {
  const response = await fetch(`${UDT_HARNESS_PATH}?ts=${Date.now()}`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Unable to load ${UDT_HARNESS_PATH}: HTTP ${response.status}`);
  }
  return response.json();
}

function summarize(results) {
  const summary = { passed: 0, failed: 0, blocked: 0, skipped: 0 };
  for (const result of results) {
    if (Object.prototype.hasOwnProperty.call(summary, result.status)) {
      summary[result.status] += 1;
    }
  }
  return summary;
}

function reasonText(result) {
  return String(result?.reason ?? result?.error ?? result?.message ?? "");
}

function classifyResult(scenario, result) {
  if (result?.ok === true || result?.success === true) return "passed";
  const reason = reasonText(result);
  if (scenario.acceptable_blockers?.some(blocker => reason.includes(blocker))) {
    return "blocked";
  }
  if (result?.requiresUdt === true) return "blocked";
  return "failed";
}

async function run(options = {}) {
  const parsed = parseOptions(options);
  if (parsed.parseError) throw new Error(`Invalid UDT harness options: ${parsed.parseError}`);

  const includeMutating = parsed.includeMutating === true;
  const actionFilter = normalizeFilter(parsed.actions);
  const idFilter = normalizeFilter(parsed.ids);
  const manifest = parsed.manifest ?? await loadManifest();
  const scenarios = Array.isArray(manifest.scenarios) ? manifest.scenarios : [];
  const results = [];

  for (const scenario of scenarios) {
    if (actionFilter && !actionFilter.has(String(scenario.action))) continue;
    if (idFilter && !idFilter.has(String(scenario.id))) continue;
    if (scenario.safe_by_default !== true && !includeMutating) {
      results.push({
        id: scenario.id,
        action: scenario.action,
        status: "skipped",
        reason: "Skipped by default; rerun with includeMutating: true inside a disposable UDT project.",
      });
      continue;
    }
    const started = Date.now();
    try {
      const payload = typeof structuredClone === "function"
        ? structuredClone(scenario.payload ?? {})
        : JSON.parse(JSON.stringify(scenario.payload ?? {}));
      const result = await window.OpenCutUXPHost.executeHostAction(scenario.action, payload);
      results.push({
        id: scenario.id,
        action: scenario.action,
        status: classifyResult(scenario, result),
        durationMs: Date.now() - started,
        result,
      });
    } catch (error) {
      results.push({
        id: scenario.id,
        action: scenario.action,
        status: "failed",
        durationMs: Date.now() - started,
        error: error.message,
      });
    }
  }

  const summary = summarize(results);
  return {
    ok: summary.failed === 0,
    harnessVersion: manifest.harness_version,
    scenarioCount: scenarios.length,
    includeMutating,
    summary,
    results,
  };
}

if (typeof window !== "undefined") {
  window.OpenCutUXPUdtHarness = Object.freeze({
    loadManifest,
    run,
    runReadOnly: () => run(),
    runAll: () => run({ includeMutating: true }),
  });
}
