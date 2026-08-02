/* OpenCut CEP job lifecycle hook registry. Classic script + CommonJS for tests. */
(function (root, factory) {
    "use strict";
    var api = factory();
    if (typeof module === "object" && module.exports) module.exports = api;
    if (root) root.OpenCutJobLifecycle = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
    "use strict";

    function createJobLifecycleRegistry() {
        var handlers = {};

        function register(jobId, hooks) {
            if (!jobId || !hooks) return;
            handlers[jobId] = hooks;
        }

        function settle(job) {
            var jobId = job && (job.id || job.job_id);
            if (!jobId || !handlers[jobId]) return;
            var hooks = handlers[jobId];
            delete handlers[jobId];
            try {
                if (job.status === "complete" && typeof hooks.onComplete === "function") {
                    hooks.onComplete(job.result || {}, job);
                } else if (job.status === "error" && typeof hooks.onError === "function") {
                    hooks.onError(job);
                } else if (job.status === "cancelled" && typeof hooks.onCancel === "function") {
                    hooks.onCancel(job);
                }
            } catch (hookErr) {
                console.error("startJob lifecycle hook failed:", hookErr);
            }
            if (typeof hooks.onFinally === "function") {
                try {
                    hooks.onFinally(job);
                } catch (finalErr) {
                    console.error("startJob onFinally hook failed:", finalErr);
                }
            }
        }

        return {
            register: register,
            settle: settle,
            pendingCount: function () { return Object.keys(handlers).length; }
        };
    }

    return { createJobLifecycleRegistry: createJobLifecycleRegistry };
});
