/* OpenCut CEP exclusive-job lifecycle boundary. Classic script + CommonJS for tests. */
(function (root, factory) {
    "use strict";
    var api = factory();
    if (typeof module === "object" && module.exports) module.exports = api;
    if (root) root.OpenCutJobRuntime = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
    "use strict";

    var TERMINAL_STATUSES = { complete: true, error: true, cancelled: true };
    var DEFAULT_MAX_POLL_FAILURES = 3;

    function isTerminalStatus(status) {
        return !!TERMINAL_STATUSES[String(status || "").toLowerCase()];
    }

    /**
     * Decide when a run of failing /status polls means the job is gone.
     *
     * Without a budget a dead or restarted backend leaves the panel polling
     * forever: progress frozen, elapsed timer climbing, the exclusive-job lock
     * held, and no error ever shown. Mirrors the UXP panel's maxStatusFailures.
     *
     * failed() returns "retry", or a terminal verdict:
     *   "missing"     — the backend answered 404, so the job no longer exists
     *                   (the registry is in-memory; a restart orphans the id).
     *                   Retrying cannot bring it back, so give up at once.
     *   "unreachable" — the budget of consecutive failures ran out.
     */
    function createPollFailureBudget(maxFailures) {
        var limit = Number(maxFailures) > 0 ? Number(maxFailures) : DEFAULT_MAX_POLL_FAILURES;
        var failures = 0;
        return {
            succeeded: function () { failures = 0; },
            failures: function () { return failures; },
            failed: function (err) {
                if (err && err.status === 404) return "missing";
                failures += 1;
                return failures >= limit ? "unreachable" : "retry";
            }
        };
    }

    /**
     * Build the terminal job record a spent poll budget settles with, so the
     * progress banner clears and the exclusive-job lock releases exactly as a
     * server-reported failure would.
     */
    function pollFailureJob(jobId, verdict, translate) {
        var t = typeof translate === "function" ? translate : function (_key, fallback) { return fallback; };
        return {
            // The id matters: job-lifecycle settle() drops any record without
            // one, which would silently skip the per-job onError/onFinally
            // hooks and leave their callers stuck waiting.
            id: jobId,
            status: "error",
            error: verdict === "missing"
                ? t("error.job_missing", "That run is no longer on the backend. It may have restarted. Start the job again.")
                : t("error.backend_unreachable", "OpenCut couldn't reach the local backend. Restart it from Settings, then try again.")
        };
    }

    function createJobRuntime() {
        var starting = false;
        var jobId = null;

        return {
            beginStart: function () {
                if (starting || jobId) return false;
                starting = true;
                return true;
            },
            failStart: function () { starting = false; },
            activate: function (nextJobId) {
                if (!nextJobId) throw new TypeError("activate requires a job id");
                jobId = nextJobId;
                starting = false;
                return jobId;
            },
            finish: function (job) {
                if (job && !isTerminalStatus(job.status)) return false;
                jobId = null;
                starting = false;
                return true;
            },
            cancel: function () {
                var cancelled = jobId;
                jobId = null;
                starting = false;
                return cancelled;
            },
            current: function () { return jobId; },
            isStarting: function () { return starting; },
            isBusy: function () { return starting || !!jobId; },
            isIdle: function () { return !starting && !jobId; },
            isCurrent: function (candidate) { return !!jobId && jobId === candidate; }
        };
    }

    return {
        TERMINAL_STATUSES: TERMINAL_STATUSES,
        DEFAULT_MAX_POLL_FAILURES: DEFAULT_MAX_POLL_FAILURES,
        isTerminalStatus: isTerminalStatus,
        createJobRuntime: createJobRuntime,
        createPollFailureBudget: createPollFailureBudget,
        pollFailureJob: pollFailureJob
    };
});
