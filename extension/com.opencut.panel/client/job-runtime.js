/* OpenCut CEP exclusive-job lifecycle boundary. Classic script + CommonJS for tests. */
(function (root, factory) {
    "use strict";
    var api = factory();
    if (typeof module === "object" && module.exports) module.exports = api;
    if (root) root.OpenCutJobRuntime = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
    "use strict";

    var TERMINAL_STATUSES = { complete: true, error: true, cancelled: true };

    function isTerminalStatus(status) {
        return !!TERMINAL_STATUSES[String(status || "").toLowerCase()];
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
        isTerminalStatus: isTerminalStatus,
        createJobRuntime: createJobRuntime
    };
});
