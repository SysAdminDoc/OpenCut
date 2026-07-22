/* OpenCut CEP onboarding state machine. Classic script + CommonJS for tests. */
(function (root, factory) {
    "use strict";
    var api = factory();
    if (typeof module === "object" && module.exports) module.exports = api;
    if (root) root.OpenCutOnboardingState = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
    "use strict";

    var STATUS = Object.freeze({
        IDLE: "idle",
        LOADING: "loading",
        ACTIVE: "active",
        SAVING: "saving",
        UNAVAILABLE: "unavailable",
        DISMISSED: "dismissed"
    });

    function clampStep(value, stepCount) {
        var parsed = parseInt(value, 10);
        if (!isFinite(parsed)) parsed = 0;
        return Math.max(0, Math.min(parsed, stepCount - 1));
    }

    function createOnboardingState(options) {
        options = options || {};
        var stepCount = Math.max(1, parseInt(options.stepCount, 10) || 1);
        var onChange = typeof options.onChange === "function"
            ? options.onChange
            : function () {};
        var state = {
            status: STATUS.IDLE,
            step: 0,
            seen: false,
            error: ""
        };

        function snapshot() {
            return Object.freeze({
                status: state.status,
                step: state.step,
                seen: state.seen,
                error: state.error,
                stepCount: stepCount
            });
        }

        function publish() {
            var current = snapshot();
            onChange(current);
            return current;
        }

        function transition(event, payload) {
            payload = payload || {};
            switch (event) {
            case "load":
                state.status = STATUS.LOADING;
                state.error = "";
                break;
            case "loaded":
                state.seen = !!payload.seen;
                state.step = clampStep(payload.step, stepCount);
                state.status = state.seen ? STATUS.DISMISSED : STATUS.ACTIVE;
                state.error = "";
                break;
            case "failed":
                state.status = STATUS.UNAVAILABLE;
                state.error = String(payload.error || "unavailable");
                break;
            case "back":
                state.status = STATUS.ACTIVE;
                state.step = clampStep(state.step - 1, stepCount);
                state.error = "";
                break;
            case "next":
                state.status = STATUS.ACTIVE;
                state.step = clampStep(state.step + 1, stepCount);
                state.error = "";
                break;
            case "step":
                state.status = STATUS.ACTIVE;
                state.step = clampStep(payload.step, stepCount);
                state.error = "";
                break;
            case "saving":
                state.status = STATUS.SAVING;
                state.error = "";
                break;
            case "completed":
                state.status = STATUS.DISMISSED;
                state.seen = true;
                state.error = "";
                break;
            case "restart":
                state.status = STATUS.ACTIVE;
                state.step = 0;
                state.seen = false;
                state.error = "";
                break;
            case "close-local":
                state.status = STATUS.DISMISSED;
                state.error = "";
                break;
            default:
                throw new Error("Unknown onboarding transition: " + event);
            }
            return publish();
        }

        return {
            transition: transition,
            snapshot: snapshot
        };
    }

    return {
        STATUS: STATUS,
        clampStep: clampStep,
        createOnboardingState: createOnboardingState
    };
});
