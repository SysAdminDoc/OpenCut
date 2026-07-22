const DEFAULT_STATE = Object.freeze({
  backendUrl: "http://127.0.0.1:5679",
  csrfToken: null,
  activeJobId: null,
  jobStartInFlight: false,
  activeSse: null,
  healthBackoffMs: 8000,
});

/**
 * Small mutable runtime store shared by the UXP transport and job layers.
 * Feature-specific view state stays with its owning feature module.
 */
export function createUxpState(initial = {}) {
  const state = { ...DEFAULT_STATE, ...initial };

  state.hasActiveJob = () => Boolean(state.activeJobId || state.jobStartInFlight);
  state.markJobStarting = () => {
    if (state.hasActiveJob()) return false;
    state.jobStartInFlight = true;
    return true;
  };
  state.trackJob = (jobId) => {
    state.activeJobId = jobId || null;
    state.jobStartInFlight = false;
    return state.activeJobId;
  };
  state.clearJob = (jobId = null) => {
    if (!jobId || state.activeJobId === jobId) state.activeJobId = null;
    state.jobStartInFlight = false;
    return !state.hasActiveJob();
  };
  state.replaceSse = (next = null) => {
    const previous = state.activeSse;
    state.activeSse = next;
    return previous;
  };
  state.resetTransport = ({ backendUrl, healthBackoffMs } = {}) => {
    state.backendUrl = backendUrl || DEFAULT_STATE.backendUrl;
    state.csrfToken = null;
    state.healthBackoffMs = healthBackoffMs ?? DEFAULT_STATE.healthBackoffMs;
    state.clearJob();
    state.activeSse = null;
  };

  return state;
}

export { DEFAULT_STATE as UXP_DEFAULT_STATE };
