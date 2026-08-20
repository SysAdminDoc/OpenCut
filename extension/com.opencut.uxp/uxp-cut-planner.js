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
 * The planning half is pure so it can be tested without a Premiere host: callers
 * read the track-item boundaries and hand them over as plain numbers. The
 * executor at the bottom of this file is the half that touches the host; it
 * takes every collaborator as an argument, and it lives here rather than in its
 * own module because the route manifest globs this directory and a new file
 * there cannot be committed while the manifest generator is mid-refactor.
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


// ---------------------------------------------------------------------------
// Applying a plan to the host
// ---------------------------------------------------------------------------

/**
 * @param {object} deps  host-bridge collaborators, all injected so the cut path
 *   can be reasoned about without the 10,000-line entrypoint around it
 */
export function createCutExecutor({
  getPPro,
  getActiveSequence,
  trackListEntries,
  trackItems,
  itemField,
  timeValueToSeconds,
  tickTimeFromSeconds,
  secondsToTicks,
  projectRoot,
  sequenceTrackSnapshot,
  fingerprintDeltaCount,
  fingerprintAddedCount,
  verificationStatus,
  attachHostWriteVerification,
}) {
  /**
   * Reads every track item in the sequence with its sequence-time boundaries.
   * Returns null when no track list is readable, so callers can tell "empty
   * timeline" apart from "cannot see the timeline".
   */
  async function _sequenceTrackItems(seq) {
    const collected = [];
    let readable = false;
    for (const [kind, list] of [
      ["video", await seq?.getVideoTrackList?.()],
      ["audio", await seq?.getAudioTrackList?.()],
    ]) {
      if (list == null) continue;
      readable = true;
      for (const entry of await trackListEntries(list)) {
        for (const item of await trackItems(entry.track)) {
          let start = null;
          let end = null;
          try { start = timeValueToSeconds(await item.getStartTime?.()); } catch (_) { start = null; }
          try { end = timeValueToSeconds(await item.getEndTime?.()); } catch (_) { end = null; }
          // Source points and speed decide whether a boundary-crossing item can
          // be trimmed clear of the range instead of blocking the typed path.
          // An unreadable one leaves the planner to refuse rather than guess.
          let inPoint = null;
          let outPoint = null;
          let speed = null;
          try { inPoint = timeValueToSeconds(await item.getInPoint?.()); } catch (_) { inPoint = null; }
          try { outPoint = timeValueToSeconds(await item.getOutPoint?.()); } catch (_) { outPoint = null; }
          try { speed = Number(await item.getSpeed?.()); } catch (_) { speed = null; }
          // An identity that survives a trim, so the same item can be found in
          // the read-back. Boundaries are deliberately absent: they are what the
          // trim changes. Host objects are re-proxied per read, so object
          // identity is not usable here.
          const nodeId = await itemField(item, ["getNodeId", "getId"], ["nodeId", "id", "guid"]);
          const name = await itemField(item, ["getName"], ["name"]);
          collected.push({
            item,
            start,
            end,
            inPoint,
            outPoint,
            speed,
            id: `${kind}|${entry.index}|${nodeId || ""}|${name || ""}`,
          });
        }
      }
    }
    if (!readable) return null;
    // A name-only identity can collide on one track. An ambiguous id would let
    // the read-back confirm the wrong item, so drop it and let the planner
    // refuse the trim rather than verify against a lookalike.
    const seen = new Map();
    for (const entry of collected) seen.set(entry.id, (seen.get(entry.id) || 0) + 1);
    for (const entry of collected) {
      if (seen.get(entry.id) > 1) entry.id = "";
    }
    return collected;
  }

  /**
   * Applies one planned cut through the typed 26.3 actions: the items the range
   * fully covers are removed, and the items that cross one boundary are trimmed
   * back to it. Everything goes into a single transaction, so a rejected action
   * cannot leave the timeline half cut, and the whole cut is one undo step.
   *
   * Removal does not ripple. That matches the CEP host's `Clip.remove(false,
   * true)`, and it is what makes a trim's predicted boundaries independent of a
   * removal in the same transaction — a rippling removal would shift the very
   * items the read-back is about to check. Closing the gaps is the separate
   * ripple-edit pass.
   */
  async function _writeCutWithEditor(seq, plan) {
    // Read the bridge once per call, and through the same `ppro.` shape the
    // UXP capability scanner recognises.
    const ppro = getPPro();
    const editor = ppro?.SequenceEditor?.getEditor?.(seq);
    if (plan.contained.length && !editor?.createRemoveItemsAction) {
      return { ok: false, reason: "SequenceEditor.createRemoveItemsAction is unavailable in this runtime." };
    }
    if (plan.contained.length && typeof ppro?.TrackItemSelection?.createEmptySelection !== "function") {
      return { ok: false, reason: "TrackItemSelection.createEmptySelection is unavailable in this runtime." };
    }
    for (const trim of plan.trims) {
      const setBoundary = trim.kind === TRIM_HEAD ? "createSetEndAction" : "createSetStartAction";
      const setSource = trim.kind === TRIM_HEAD ? "createSetOutPointAction" : "createSetInPointAction";
      if (typeof trim.item.item?.[setBoundary] !== "function" || typeof trim.item.item?.[setSource] !== "function") {
        return { ok: false, reason: `TrackItem.${setBoundary}/${setSource} is unavailable in this runtime.` };
      }
    }
    const context = await projectRoot();
    if (!context?.proj?.executeTransaction) {
      return { ok: false, reason: "Project.executeTransaction is unavailable in this runtime." };
    }

    let selection = null;
    if (plan.contained.length) {
      ppro.TrackItemSelection.createEmptySelection((created) => { selection = created; });
      if (!selection?.addItem) {
        return { ok: false, reason: "createEmptySelection did not yield a usable selection." };
      }
      for (const entry of plan.contained) {
        // skipDuplicateCheck: the planner already emitted each item once.
        selection.addItem(entry.item, true);
      }
    }

    const mediaType = ppro?.Constants?.MediaType?.ANY ?? 0;
    const run = () => Boolean(context.proj.executeTransaction((compoundAction) => {
      for (const trim of plan.trims) {
        const item = trim.item.item;
        // Sequence boundary first, then the source point. Under the documented
        // reading — `createMoveAction` is the one that moves an item, so these
        // set a boundary — the second write confirms the value the first
        // already implied. A host that disagrees is caught by the read-back
        // rather than left to corrupt the timeline silently.
        if (trim.kind === TRIM_HEAD) {
          compoundAction.addAction(item.createSetEndAction(tickTimeFromSeconds(trim.to.end)));
          compoundAction.addAction(item.createSetOutPointAction(tickTimeFromSeconds(trim.source.outPoint)));
        } else {
          compoundAction.addAction(item.createSetStartAction(tickTimeFromSeconds(trim.to.start)));
          compoundAction.addAction(item.createSetInPointAction(tickTimeFromSeconds(trim.source.inPoint)));
        }
      }
      if (selection) {
        compoundAction.addAction(editor.createRemoveItemsAction(selection, false, mediaType));
      }
    }, "OpenCut apply cuts"));

    let accepted = false;
    if (typeof context.proj.lockedAccess === "function") {
      context.proj.lockedAccess(() => { accepted = run(); });
    } else {
      // eslint-disable-next-line @adobe/premierepro/prefer-locked-access-wrapper -- compatibility fallback when lockedAccess is absent
      accepted = run();
    }
    if (!accepted) {
      return { ok: false, reason: "executeTransaction rejected the cut actions." };
    }

    // The plan states exactly where every affected item should now be, so the
    // read-back can assert that rather than settling for "something changed".
    const after = await _sequenceTrackItems(seq);
    if (!after) {
      return { ok: true, reason: "", verified: null, note: "Track lists became unreadable after the write." };
    }
    const mismatches = verifyPostState(expectedPostState(plan), after);
    if (mismatches.length) {
      return {
        ok: false,
        verified: false,
        reason: `The timeline did not match the cut plan after the write (${_describeMismatches(mismatches)}). `
          + "The cut is one undo step in Premiere.",
      };
    }
    return { ok: true, reason: "", verified: true };
  }

  function _describeMismatches(mismatches) {
    return mismatches.slice(0, 3).map((entry) => {
      if (entry.kind !== "trim_landed_elsewhere") return `${entry.kind} ${entry.id}`;
      return `${entry.id} expected ${entry.expected.start.toFixed(3)}-${entry.expected.end.toFixed(3)}s `
        + `but read back ${entry.observed.start.toFixed(3)}-${entry.observed.end.toFixed(3)}s`;
    }).join("; ");
  }

  /**
   * Marks track items disabled instead of removing them, so a reviewed cut is
   * reversible in Premiere. Runs as one transaction like the removal path.
   */
  async function _disableItemsWithEditor(entries) {
    if (!entries.every((entry) => typeof entry.item?.createSetDisabledAction === "function")) {
      return { ok: false, reason: "TrackItem.createSetDisabledAction is unavailable in this runtime." };
    }
    const context = await projectRoot();
    if (!context?.proj?.executeTransaction) {
      return { ok: false, reason: "Project.executeTransaction is unavailable in this runtime." };
    }
    const run = () => Boolean(context.proj.executeTransaction((compoundAction) => {
      for (const entry of entries) compoundAction.addAction(entry.item.createSetDisabledAction(true));
    }, "OpenCut disable cut ranges"));
    let accepted = false;
    if (typeof context.proj.lockedAccess === "function") {
      context.proj.lockedAccess(() => { accepted = run(); });
    } else {
      // eslint-disable-next-line @adobe/premierepro/prefer-locked-access-wrapper -- compatibility fallback when lockedAccess is absent
      accepted = run();
    }
    return { ok: accepted, reason: accepted ? "" : "executeTransaction rejected the set-disabled action." };
  }

  /**
   * Applies one cut, preferring the typed remove-items action and falling back
   * to the legacy ripple delete when the range cannot be expressed as whole
   * track items or the typed API is absent.
   *
   * In disable mode nothing is ever removed: a range that cannot be expressed
   * as whole track items is reported and skipped, because falling back to
   * ripple delete would destroy media the user explicitly asked to keep.
   */
  async function _applyOneCut(seq, cut, disableMode = false) {
    const items = await _sequenceTrackItems(seq);
    if (!items) {
      if (disableMode) return { method: "skipped", note: "Track lists were not readable." };
      return await _rippleDeleteFallback(seq, cut, "Track lists were not readable.");
    }
    const plan = planCutRemoval(items, cut.start, cut.end);
    if (disableMode) {
      // There is no partial disable: a boundary-crossing item can only be kept
      // whole or trimmed, and trimming destroys the very media disable mode
      // exists to preserve.
      if (!plan.removable) return { method: "skipped", note: plan.reason };
      if (plan.trims.length) {
        return {
          method: "skipped",
          note: `${plan.trims.length} track item(s) cross a cut boundary; disabling them whole `
            + "would mute media outside the range, and trimming would delete it.",
        };
      }
      const disabled = await _disableItemsWithEditor(plan.contained);
      return disabled.ok
        ? { method: "set_disabled_action", note: "" }
        : { method: "skipped", note: disabled.reason };
    }
    if (plan.removable) {
      const typed = await _writeCutWithEditor(seq, plan);
      if (typed.ok) {
        return {
          method: plan.trims.length ? "trim_and_remove_action" : "remove_items_action",
          note: typed.note || "",
        };
      }
      // A read-back that disproved the plan is a wrong result, not a missing
      // capability: repeating the range through the legacy path would edit an
      // already-edited timeline. Report it and let the operator undo.
      if (typed.verified === false) return { method: "failed", note: typed.reason };
      return await _rippleDeleteFallback(seq, cut, typed.reason);
    }
    return await _rippleDeleteFallback(seq, cut, plan.reason);
  }

  async function _rippleDeleteFallback(seq, cut, note) {
    if (typeof seq?.rippleDelete !== "function") {
      throw new Error(`Sequence.rippleDelete is unavailable and the typed path was refused: ${note}`);
    }
    await seq.rippleDelete(secondsToTicks(cut.start), secondsToTicks(cut.end));
    return { method: "ripple_delete", note };
  }

  /**
   * Applies silence-removal cuts, preferring the typed 26.3 remove-items
   * action and falling back to the legacy ripple delete.
   * @param {Array<{start: number, end: number}>} cuts  — times in seconds
   */
  async function applyCuts(cuts, mode = "delete") {
    const seq = await getActiveSequence();
    if (!seq) {
      return { ok: false, reason: "No active sequence or UXP API unavailable." };
    }
    const disableMode = String(mode || "delete").toLowerCase() === "disable";
    try {
      // Sort descending so removal doesn't shift earlier cut points
      const sorted = [...cuts].sort((a, b) => b.start - a.start);
      let beforeAll = await sequenceTrackSnapshot(seq);
      let canVerify = beforeAll !== null;
      let verifiedCuts = 0;
      const methodCounts = {
        remove_items_action: 0,
        trim_and_remove_action: 0,
        ripple_delete: 0,
        set_disabled_action: 0,
        skipped: 0,
        failed: 0,
      };
      const fallbackReasons = [];
      for (const cut of sorted) {
        const beforeCut = canVerify ? await sequenceTrackSnapshot(seq) : null;
        const outcome = await _applyOneCut(seq, cut, disableMode);
        methodCounts[outcome.method] += 1;
        if (outcome.note && !fallbackReasons.includes(outcome.note)) fallbackReasons.push(outcome.note);
        // One cut whose read-back disproved its plan stops the batch. Carrying
        // on would stack further edits on a timeline that is already not what
        // the plan described, and bury the one undo step that reverses it.
        if (outcome.method === "failed") break;
        if (canVerify) {
          const afterCut = await sequenceTrackSnapshot(seq);
          if (afterCut === null) canVerify = false;
          else if (fingerprintDeltaCount(beforeCut, afterCut) > 0 ||
              fingerprintAddedCount(beforeCut, afterCut) > 0) verifiedCuts += 1;
        }
      }
      const afterAll = canVerify ? await sequenceTrackSnapshot(seq) : null;
      if (afterAll === null) canVerify = false;
      const written = sorted.length - methodCounts.skipped - methodCounts.failed;
      const status = methodCounts.failed
        ? "failed"
        : verificationStatus(sorted.length, canVerify ? verifiedCuts : null, canVerify);
      return await attachHostWriteVerification(
        { ok: status !== "failed", applied: written },
        {
          action: "ocApplySequenceCuts",
          attempted: sorted.length,
          reported: written,
          verified: canVerify ? verifiedCuts : null,
          status,
          readBackMethod: canVerify
            ? "video/audio track-item boundary and disabled-state fingerprint diff per cut, plus a per-cut check of every affected item against the boundaries the plan predicted"
            : "unavailable: track-item traversal",
          beforeState: { track_item_count: beforeAll?.length ?? null },
          afterState: {
            track_item_count: afterAll?.length ?? null,
            mode: disableMode ? "disable" : "delete",
            methods: methodCounts,
            fallback_reasons: fallbackReasons,
          },
          detail: "Delete mode runs each cut as one transaction of typed actions: items the range covers are removed with SequenceEditor.createRemoveItemsAction and items crossing one boundary are trimmed back to it, so the cut range ends up empty and nothing moves. Removal does not ripple, matching the CEP host. An item that encloses the range needs a razor the 26.3 API does not offer, and still drops to rippleDelete. Every cut is then checked against the boundaries its plan predicted, which is what tells a trim apart from a move; a mismatch stops the batch rather than editing further. Disable mode sets the track items disabled instead and never falls back, because removing media the user asked to keep is not a degraded form of the same request.",
        }
      );
    } catch (e) {
      console.warn("[PProBridge] applyCuts failed:", e.message);
      return { ok: false, reason: e.message };
    }
  }

  return { applyCuts };
}
