/**
 * Cut-range planning for the UXP timeline write path.
 *
 * `Sequence.rippleDelete()` does not appear in the Premiere 26.3 typings and is
 * reported to return success while changing nothing on that host, so the cut
 * path prefers the typed `SequenceEditor.createRemoveItemsAction`. That action
 * removes whole track items rather than a time range, so a cut is only
 * expressible through it when every item the range touches lies entirely
 * inside it. An item that crosses a boundary would need a razor the typed API
 * does not offer, and removing it whole would delete media the user kept.
 *
 * The decision is pure so it can be tested without a Premiere host: callers
 * read the track-item boundaries and hand them over as plain numbers.
 */

export const CUT_BOUNDARY_TOLERANCE_SECONDS = 1 / 1000;

// `Number(null)` and `Number("")` are 0, which would read a missing boundary as
// a zero-length item at the timeline origin and quietly drop it from the plan.
function _seconds(value) {
  if (value === null || value === undefined || value === "") return NaN;
  return Number(value);
}

/**
 * @param {Array<{start: number, end: number}>} items  track items with seconds bounds
 * @param {number} start  cut range start, seconds
 * @param {number} end    cut range end, seconds
 * @param {number} [tolerance]  boundary slop, seconds
 * @returns {{contained: Array, straddling: Array, unreadable: Array, removable: boolean, reason: string}}
 */
export function planCutRemoval(items, start, end, tolerance = CUT_BOUNDARY_TOLERANCE_SECONDS) {
  const rangeStart = _seconds(start);
  const rangeEnd = _seconds(end);
  const slop = Number.isFinite(Number(tolerance)) ? Number(tolerance) : CUT_BOUNDARY_TOLERANCE_SECONDS;
  const plan = { contained: [], straddling: [], unreadable: [], removable: false, reason: "" };

  if (!Number.isFinite(rangeStart) || !Number.isFinite(rangeEnd) || rangeEnd - rangeStart <= slop) {
    plan.reason = "Cut range is not a finite forward interval.";
    return plan;
  }

  for (const item of items || []) {
    const itemStart = _seconds(item?.start);
    const itemEnd = _seconds(item?.end);
    if (!Number.isFinite(itemStart) || !Number.isFinite(itemEnd)) {
      // Unknown bounds cannot be proven outside the range, so they block the
      // typed path rather than being silently treated as untouched.
      plan.unreadable.push(item);
      continue;
    }
    if (itemEnd <= rangeStart + slop) continue;
    if (itemStart >= rangeEnd - slop) continue;
    if (itemStart >= rangeStart - slop && itemEnd <= rangeEnd + slop) plan.contained.push(item);
    else plan.straddling.push(item);
  }

  if (plan.unreadable.length) {
    plan.reason = `${plan.unreadable.length} track item(s) have unreadable boundaries.`;
  } else if (plan.straddling.length) {
    plan.reason = `${plan.straddling.length} track item(s) cross a cut boundary; `
      + "removing them whole would delete media outside the range.";
  } else if (!plan.contained.length) {
    plan.reason = "No track item overlaps the cut range.";
  } else {
    plan.removable = true;
  }
  return plan;
}
