import { createRequire } from "node:module";

import { describe, expect, it } from "vitest";

const require = createRequire(import.meta.url);
const jr = require("../client/job-runtime.js");

// F367: a dead or restarted backend used to leave both CEP poll loops
// retrying forever — progress frozen, elapsed timer climbing, the
// exclusive-job lock held, and no error ever shown to the user.

describe("createPollFailureBudget", () => {
  it("retries transient failures until the budget runs out", () => {
    const budget = jr.createPollFailureBudget(3);
    expect(budget.failed(new Error("network"))).toBe("retry");
    expect(budget.failed(new Error("network"))).toBe("retry");
    expect(budget.failed(new Error("network"))).toBe("unreachable");
  });

  it("gives up immediately on 404 because the job no longer exists", () => {
    const budget = jr.createPollFailureBudget(3);
    const gone = Object.assign(new Error("Job not found"), { status: 404 });
    // The job registry is in-memory, so a backend restart orphans the id.
    // Retrying cannot bring it back.
    expect(budget.failed(gone)).toBe("missing");
    expect(budget.failures()).toBe(0);
  });

  it("resets the run of failures after any successful poll", () => {
    const budget = jr.createPollFailureBudget(3);
    budget.failed(new Error("blip"));
    budget.failed(new Error("blip"));
    budget.succeeded();
    expect(budget.failures()).toBe(0);
    // A long job that blips twice, recovers, then blips twice again must not
    // be killed off by the earlier failures.
    expect(budget.failed(new Error("blip"))).toBe("retry");
    expect(budget.failed(new Error("blip"))).toBe("retry");
  });

  it("defaults to the shared failure limit and rejects nonsense limits", () => {
    expect(jr.DEFAULT_MAX_POLL_FAILURES).toBe(3);
    for (const bad of [0, -1, undefined, null, "abc", NaN]) {
      const budget = jr.createPollFailureBudget(bad);
      let verdict = "retry";
      for (let i = 0; i < jr.DEFAULT_MAX_POLL_FAILURES; i += 1) {
        verdict = budget.failed(new Error("network"));
      }
      expect(verdict, String(bad)).toBe("unreachable");
    }
  });

  it("treats a non-404 HTTP error as transient", () => {
    const budget = jr.createPollFailureBudget(3);
    const server = Object.assign(new Error("HTTP 500"), { status: 500 });
    expect(budget.failed(server)).toBe("retry");
  });
});

describe("pollFailureJob", () => {
  it("settles as a terminal error so the panel unlocks", () => {
    for (const verdict of ["missing", "unreachable"]) {
      const job = jr.pollFailureJob("job-1", verdict, null);
      expect(job.status, verdict).toBe("error");
      expect(jr.isTerminalStatus(job.status), verdict).toBe(true);
      expect(String(job.error).length, verdict).toBeGreaterThan(0);
    }
  });

  it("carries the job id so per-job lifecycle hooks still fire", () => {
    // job-lifecycle settle() drops any record without an id, which would skip
    // the caller's onError/onFinally and leave it waiting forever — the
    // interview-polish button stays disabled, the batch chain never advances.
    const registry = require("../client/job-lifecycle.js").createJobLifecycleRegistry();
    const calls = [];
    registry.register("job-7", {
      onError: (job) => calls.push(["onError", job.status]),
      onFinally: () => calls.push(["onFinally"]),
    });

    registry.settle(jr.pollFailureJob("job-7", "unreachable", null));

    expect(calls).toEqual([["onError", "error"], ["onFinally"]]);
  });

  it("tells the user the run is gone versus the backend is unreachable", () => {
    expect(jr.pollFailureJob("job-1", "missing", null).error).toMatch(/no longer on the backend/i);
    expect(jr.pollFailureJob("job-1", "unreachable", null).error).toMatch(
      /couldn't reach the local backend/i,
    );
  });

  it("routes both messages through the panel translator", () => {
    const seen = [];
    const translate = (key, fallback) => {
      seen.push(key);
      return `translated:${fallback}`;
    };
    expect(jr.pollFailureJob("job-1", "missing", translate).error).toBe(
      "translated:That run is no longer on the backend. It may have restarted. Start the job again.",
    );
    jr.pollFailureJob("job-1", "unreachable", translate);
    expect(seen).toEqual(["error.job_missing", "error.backend_unreachable"]);
  });
});
