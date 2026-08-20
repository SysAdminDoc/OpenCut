/**
 * Cut-range planning for the UXP timeline write path.
 *
 * `Sequence.rippleDelete()` does not appear in the Premiere 26.3 typings and is
 * reported to return success while changing nothing on that host, so the cut
 * path prefers typed `SequenceEditor` / `TrackItem` actions.
 *
 * `createRemoveItemsAction` removes whole track items rather than a time range.
 * An item that crosses one cut boundary can still be expressed, by trimming it
 * back to the boundary and pulling its source point in by the same amount, so
 * the media outside the range survives. An item that encloses the whole range
 * cannot: that needs a razor, and 26.3 exposes none.
 *
 * The contract every plan describes is the same on every track the range
 * touches: the cut range ends up empty and nothing moves. Removals therefore do
 * not ripple, which matches the CEP host's `Clip.remove(false, true)` and keeps
 * a trim's predicted boundaries independent of any removal in the same
 * transaction. Closing the resulting gaps is the separate ripple-edit pass.
 *
 * The decision is pure so it can be tested without a Premiere host: callers read
 * the track-item boundaries and hand them over as plain numbers.
 */

export const CUT_BOUNDARY_TOLERANCE_SECONDS = 1 / 1000;

/** Trimming an item's tail back to the cut start; it began before the range. */
export const TRIM_HEAD = "head";
/** Trimming an item's head forward to the cut end; it runs past the range. */
export const TRIM_TAIL = "tail";

// A retimed clip's source points do not advance at one second per second, so
// the trim arithmetic below would silently land on the wrong frame.
const SPEED_TOLERANCE = 1e-6;

// `Number(null)` and `Number("")` are 0, which would read a missing boundary as
// a zero-length item at the timeline origin and quietly drop it from the plan.
function _seconds(value) {
  if (value === null || value === undefined || value === "") return NaN;
  return Number(value);
}

/**
 * Describes how one boundary-crossing item can be trimmed clear of the range,
 * or returns a reason it cannot be.
 *
 * Both trims preserve duration against source range: a head trim shortens the
 * item to `[itemStart, rangeStart]` and pulls the out point back by the same
 * amount; a tail trim shortens it to `[rangeEnd, itemEnd]` and pushes the in
 * point forward by the same amount.
 */
function _planTrim(item, itemStart, itemEnd, rangeStart, rangeEnd, slop) {
  const crossesStart = itemStart < rangeStart - slop;
  const crossesEnd = itemEnd > rangeEnd + slop;

  if (crossesStart && crossesEnd) {
    return { blocked: "encloses the cut range and would need a razor" };
  }

  const speed = _seconds(item?.speed);
  if (Number.isFinite(speed) && Math.abs(speed - 1) > SPEED_TOLERANCE) {
    return { blocked: `is retimed (speed ${speed}), so a source-point trim would land off-frame` };
  }

  // Without the source points the new in/out cannot be stated, and a trim whose
  // outcome cannot be predicted cannot be verified against the read-back.
  const inPoint = _seconds(item?.inPoint);
  const outPoint = _seconds(item?.outPoint);
  if (!Number.isFinite(inPoint) || !Number.isFinite(outPoint)) {
    return { blocked: "has unreadable source in/out points" };
  }

  // A trim leaves the item on the timeline, so proving it landed means finding
  // that same item again afterwards. Removal only has to prove an absence, so
  // it does not need this. An item the read-back cannot name is not trimmed.
  if (!item?.id) {
    return { blocked: "cannot be identified in a re-read, so a trim could not be verified" };
  }

  if (crossesStart) {
    return {
      trim: {
        item,
        kind: TRIM_HEAD,
        from: { start: itemStart, end: itemEnd },
        to: { start: itemStart, end: rangeStart },
        source: { inPoint, outPoint: outPoint - (itemEnd - rangeStart) },
      },
    };
  }
  return {
    trim: {
      item,
      kind: TRIM_TAIL,
      from: { start: itemStart, end: itemEnd },
      to: { start: rangeEnd, end: itemEnd },
      source: { inPoint: inPoint + (rangeEnd - itemStart), outPoint },
    },
  };
}

/**
 * @param {Array<{start: number, end: number, inPoint?: number, outPoint?: number, speed?: number}>} items
 * @param {number} start  cut range start, seconds
 * @param {number} end    cut range end, seconds
 * @param {number} [tolerance]  boundary slop, seconds
 * @returns {{contained: Array, straddling: Array, trims: Array, blocked: Array,
 *           unreadable: Array, removable: boolean, reason: string}}
 */
export function planCutRemoval(items, start, end, tolerance = CUT_BOUNDARY_TOLERANCE_SECONDS) {
  const rangeStart = _seconds(start);
  const rangeEnd = _seconds(end);
  const slop = Number.isFinite(Number(tolerance)) ? Number(tolerance) : CUT_BOUNDARY_TOLERANCE_SECONDS;
  const plan = {
    contained: [],
    straddling: [],
    trims: [],
    blocked: [],
    unreadable: [],
    removable: false,
    reason: "",
  };

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
    if (itemStart >= rangeStart - slop && itemEnd <= rangeEnd + slop) {
      plan.contained.push(item);
      continue;
    }
    plan.straddling.push(item);
    const outcome = _planTrim(item, itemStart, itemEnd, rangeStart, rangeEnd, slop);
    if (outcome.trim) plan.trims.push(outcome.trim);
    else plan.blocked.push({ item, reason: outcome.blocked });
  }

  if (plan.unreadable.length) {
    plan.reason = `${plan.unreadable.length} track item(s) have unreadable boundaries.`;
  } else if (plan.blocked.length) {
    const detail = plan.blocked.map((entry) => entry.reason).join("; ");
    plan.reason = `${plan.blocked.length} track item(s) cross a cut boundary in a way the `
      + `typed API cannot express: ${detail}.`;
  } else if (!plan.contained.length && !plan.trims.length) {
    plan.reason = "No track item overlaps the cut range.";
  } else {
    plan.removable = true;
  }
  return plan;
}

/**
 * The boundaries every affected item is expected to hold once the plan has been
 * applied, for checking against a fresh read of the timeline.
 *
 * Removed items are named rather than described, because their absence is the
 * assertion. Trimmed items carry their predicted sequence bounds, which is what
 * distinguishes a trim from a move: a host whose `createSetStartAction` shifts
 * the item instead of shortening it leaves the end where a trim would not.
 */
export function expectedPostState(plan) {
  return {
    removed: (plan?.contained || []).map((entry) => entry?.id).filter(Boolean),
    trimmed: (plan?.trims || []).map((trim) => ({
      id: trim.item.id,
      kind: trim.kind,
      start: trim.to.start,
      end: trim.to.end,
    })),
  };
}

/**
 * Compares a fresh read of the timeline against `expectedPostState`. Returns
 * the mismatches, so an empty array means the plan landed exactly.
 *
 * This is the check that distinguishes a trim from a move on a host whose
 * `createSetStartAction` turns out to do the other one: a moved item keeps its
 * duration, so its end lands where a trim would never leave it.
 *
 * @param {{removed: Array<string>, trimmed: Array}} expected
 * @param {Array<{id: string, start: number, end: number}>} observed  items still on the timeline
 * @param {number} [tolerance]
 */
export function verifyPostState(expected, observed, tolerance = CUT_BOUNDARY_TOLERANCE_SECONDS) {
  const slop = Number.isFinite(Number(tolerance)) ? Number(tolerance) : CUT_BOUNDARY_TOLERANCE_SECONDS;
  const mismatches = [];
  const survivors = new Map();
  for (const entry of observed || []) {
    if (entry?.id) survivors.set(entry.id, entry);
  }

  for (const id of expected?.removed || []) {
    if (survivors.has(id)) mismatches.push({ kind: "not_removed", id });
  }
  for (const trim of expected?.trimmed || []) {
    const seen = survivors.get(trim.id);
    if (!seen) {
      mismatches.push({ kind: "trimmed_item_vanished", id: trim.id });
      continue;
    }
    const gotStart = _seconds(seen.start);
    const gotEnd = _seconds(seen.end);
    if (!Number.isFinite(gotStart) || !Number.isFinite(gotEnd)) {
      mismatches.push({ kind: "unreadable_after_trim", id: trim.id });
      continue;
    }
    if (Math.abs(gotStart - trim.start) > slop || Math.abs(gotEnd - trim.end) > slop) {
      mismatches.push({
        kind: "trim_landed_elsewhere",
        id: trim.id,
        expected: { start: trim.start, end: trim.end },
        observed: { start: gotStart, end: gotEnd },
      });
    }
  }
  return mismatches;
}
