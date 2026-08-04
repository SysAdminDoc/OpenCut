/**
 * Keep Sequence Index filter responses ordered without requiring AbortController.
 * UXP hosts can keep an earlier fetch alive after a newer filter request starts,
 * so only the latest generation may update the table or clear its busy state.
 */
export function createSequenceIndexFilterController({
  request,
  getData = (response) => response?.data || {},
  onRows,
  onError,
  setBusy = () => {},
}) {
  let generation = 0;

  function invalidate() {
    generation += 1;
    return generation;
  }

  async function apply(payload) {
    const requestGeneration = invalidate();
    setBusy(true);
    try {
      const response = await request(payload);
      if (requestGeneration !== generation) {
        return { accepted: false, stale: true };
      }
      if (!response || response.error || response.ok === false) {
        onError?.(response);
        return { accepted: false, stale: false };
      }
      const data = getData(response) || {};
      onRows(Array.isArray(data.rows) ? data.rows : []);
      return { accepted: true, stale: false };
    } catch (error) {
      if (requestGeneration !== generation) {
        return { accepted: false, stale: true };
      }
      onError?.(error);
      return { accepted: false, stale: false };
    } finally {
      if (requestGeneration === generation) setBusy(false);
    }
  }

  return { apply, invalidate };
}
