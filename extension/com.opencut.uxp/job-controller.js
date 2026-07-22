const TERMINAL_SUCCESS = new Set(["done", "complete", "success"]);
const TERMINAL_ERROR = new Set(["error", "failed", "cancelled", "interrupted"]);

/** Create the async-job transport independently from DOM and feature handlers. */
export function createJobController({
  client,
  state,
  getBackendUrl = () => state?.backendUrl || "",
  EventSourceCtor = null,
  setTimeoutFn = setTimeout,
  pollIntervalMs = 1200,
  maxPollAttempts = 3000,
  maxStatusFailures = 3,
  translate = (_key, fallback) => fallback,
  setLocked = () => {},
  onStateChange = () => {},
} = {}) {
  if (!client || !state) throw new TypeError("createJobController requires client and state");

  const completionHooks = [];

  function notifyState() {
    onStateChange(state);
  }

  function clear(jobId = null) {
    state.clearJob(jobId);
    setLocked(false);
    notifyState();
  }

  function fireCompletionHooks() {
    for (const hook of completionHooks) {
      try { hook(); } catch (_) {}
    }
  }

  function finishSuccess(jobId, job, onComplete) {
    clear(jobId);
    onComplete(job?.result ?? job);
    fireCompletionHooks();
  }

  function finishError(jobId, job, onError) {
    clear(jobId);
    onError(job?.error ?? job?.message ?? "Job failed");
    fireCompletionHooks();
  }

  async function start(endpoint, body, onProgress, onComplete, onError) {
    if (!state.markJobStarting()) {
      onError(translate("uxp.runtime.job_already_running", "Another OpenCut job is already running."));
      return;
    }
    setLocked(true);
    notifyState();

    let result;
    try {
      result = await client.post(endpoint, body);
    } catch (err) {
      clear();
      onError(err?.message ?? "Failed to start job");
      return;
    }
    if (!result.ok) {
      clear();
      onError(result.error ?? "Failed to start job");
      return;
    }

    const jobId = result.data?.job_id ?? result.data?.id ?? null;
    if (!jobId) {
      clear();
      onProgress(100, "Done");
      onComplete(result.data);
      return;
    }

    state.trackJob(jobId);
    notifyState();
    if (EventSourceCtor) trackSse(jobId, onProgress, onComplete, onError);
    else pollJob(jobId, onProgress, onComplete, onError);
  }

  function closeSse() {
    const stream = state.replaceSse(null);
    if (stream) stream.close();
  }

  function trackSse(jobId, onProgress, onComplete, onError) {
    closeSse();
    const stream = new EventSourceCtor(`${getBackendUrl()}/stream/${jobId}`);
    state.replaceSse(stream);
    stream.onmessage = (event) => {
      try {
        const job = JSON.parse(event.data);
        const status = job.status ?? "running";
        onProgress(
          typeof job.progress === "number" ? job.progress : 0,
          job.message ?? job.msg ?? translate("processing.processing", "Processing..."),
        );
        if (TERMINAL_SUCCESS.has(status)) {
          closeSse();
          finishSuccess(jobId, job, onComplete);
        } else if (TERMINAL_ERROR.has(status)) {
          closeSse();
          finishError(jobId, job, onError);
        }
      } catch (_) {}
    };
    stream.onerror = () => {
      if (state.activeSse !== stream) return;
      closeSse();
      pollJob(jobId, onProgress, onComplete, onError);
    };
  }

  function schedulePoll(jobId, onProgress, onComplete, onError, attempt, statusFailures = 0) {
    setTimeoutFn(() => {
      if (state.activeJobId === jobId) {
        pollJob(jobId, onProgress, onComplete, onError, attempt, statusFailures);
      }
    }, pollIntervalMs);
  }

  async function pollJob(jobId, onProgress, onComplete, onError, attempt = 0, statusFailures = 0) {
    let result;
    try {
      result = await client.get(`/status/${jobId}`);
    } catch (err) {
      result = { ok: false, error: err?.message ?? "Polling error" };
    }
    if (!result.ok) {
      const nextFailures = statusFailures + 1;
      if (nextFailures < maxStatusFailures) {
        schedulePoll(jobId, onProgress, onComplete, onError, attempt, nextFailures);
        return;
      }
      finishError(jobId, { error: result.error ?? "Polling error" }, onError);
      return;
    }

    const job = result.data;
    const status = job.status ?? "running";
    onProgress(
      typeof job.progress === "number" ? job.progress : 0,
      job.message ?? job.msg ?? translate("processing.processing", "Processing..."),
    );
    if (TERMINAL_SUCCESS.has(status)) {
      finishSuccess(jobId, job, onComplete);
      return;
    }
    if (TERMINAL_ERROR.has(status)) {
      finishError(jobId, job, onError);
      return;
    }
    if (attempt >= maxPollAttempts) {
      finishError(jobId, { error: "Polling timed out — the job is still running on the server." }, onError);
      return;
    }
    schedulePoll(jobId, onProgress, onComplete, onError, attempt + 1, 0);
  }

  async function cancel() {
    if (!state.activeJobId) return false;
    const jobId = state.activeJobId;
    closeSse();
    try {
      await client.post(`/cancel/${jobId}`, {});
    } finally {
      clear(jobId);
      fireCompletionHooks();
    }
    return true;
  }

  function poll(jobId) {
    return new Promise((resolve, reject) => {
      if (!state.markJobStarting()) {
        reject(new Error(translate("uxp.runtime.job_already_running", "Another OpenCut job is already running.")));
        return;
      }
      state.trackJob(jobId);
      setLocked(true);
      notifyState();
      const onProgress = () => {};
      const onError = (message) => reject(new Error(message));
      if (EventSourceCtor) trackSse(jobId, onProgress, resolve, onError);
      else pollJob(jobId, onProgress, resolve, onError);
    });
  }

  return {
    start,
    poll,
    cancel,
    resume: (jobId, onProgress, onComplete, onError) => (
      start(`/jobs/${jobId}/resume`, {}, onProgress, onComplete, onError)
    ),
    onJobFinished: (hook) => completionHooks.push(hook),
    hasActiveJob: () => state.hasActiveJob(),
    closeSse,
  };
}

export { TERMINAL_SUCCESS, TERMINAL_ERROR };
