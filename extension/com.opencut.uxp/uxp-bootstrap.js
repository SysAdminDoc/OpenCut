/** Run named bootstrap steps in order and preserve which boundary failed. */
export async function runBootstrapSteps(steps, { onStep, onError } = {}) {
  for (const entry of steps || []) {
    const name = entry?.name || "anonymous";
    try {
      if (typeof onStep === "function") onStep(name);
      await entry.run();
    } catch (error) {
      if (typeof onError === "function") onError(error, name);
      error.bootstrapStep = name;
      throw error;
    }
  }
}

export function bootstrapApplication(init, onError = (error) => console.error(error)) {
  return Promise.resolve()
    .then(() => init())
    .catch((error) => {
      onError(error);
      return null;
    });
}
