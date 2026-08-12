/** Independent read-back evidence for Premiere UXP host mutations. */
export function createHostWriteVerifier({
  getPPro,
  trackListEntries,
  trackItems,
  itemField,
  timeValueToSeconds,
}) {
  async function hostVersionInfo() {
    const ppro = getPPro?.();
    let version = "";
    try {
      version = String(
        ppro?.app?.version
        ?? ppro?.version
        ?? ""
      );
    } catch (_) { version = ""; }
    return { bridge: "uxp", app_name: "Premiere Pro", version, build: "" };
  }

  function fingerprintDeltaCount(beforeItems, afterItems) {
    const remaining = new Map();
    for (const item of afterItems) remaining.set(item, (remaining.get(item) || 0) + 1);
    let removed = 0;
    for (const item of beforeItems) {
      const count = remaining.get(item) || 0;
      if (count > 0) remaining.set(item, count - 1);
      else removed += 1;
    }
    return removed;
  }

  function fingerprintAddedCount(beforeItems, afterItems) {
    return fingerprintDeltaCount(afterItems, beforeItems);
  }

  function verificationStatus(reported, verified, canVerify = true) {
    if (!canVerify) return "unverified";
    if (Number(reported || 0) > 0 && Number(verified || 0) === 0) return "failed";
    if (Number(verified || 0) < Number(reported || 0)) return "partial";
    return "verified";
  }

  async function attach(result, {
    action,
    attempted,
    reported,
    verified,
    status,
    readBackMethod,
    beforeState = null,
    afterState = null,
    detail = "",
  }) {
    const host = await hostVersionInfo();
    const verification = {
      schema: "opencut.host_write_verification.v1",
      action,
      host_version: host.version,
      host,
      attempted_count: attempted == null ? null : Number(attempted),
      reported_count: reported == null ? null : Number(reported),
      verified_count: verified == null ? null : Number(verified),
      verification_status: status || "unverified",
      read_back_method: readBackMethod || "unavailable",
      before_state: beforeState,
      after_state: afterState,
      detail,
    };
    const attached = {
      ...(result || {}),
      host_write_verification: verification,
      attempted_count: verification.attempted_count,
      reported_count: verification.reported_count,
      verified_count: verification.verified_count,
      verification_status: verification.verification_status,
      read_back_method: verification.read_back_method,
      host_version: verification.host_version,
      host,
    };
    if (verification.verification_status === "failed") {
      attached.ok = false;
      attached.error_code = "HOST_WRITE_NOT_APPLIED";
      attached.reason = attached.reason || "Premiere reported success but the independent read-back found no timeline or project change.";
    } else if (verification.verification_status === "unverified") {
      attached.unverified = true;
      attached.warning = attached.warning || "Premiere accepted the request, but this operation has no independent read-back API and remains unverified.";
    } else if (verification.verification_status === "partial") {
      attached.warning = attached.warning || "Premiere applied only part of the requested host write; review the verified count before continuing.";
    }
    return attached;
  }

  async function sequenceTrackSnapshot(sequence) {
    if (!sequence) return null;
    const fingerprints = [];
    let readable = false;
    for (const [kind, list] of [
      ["video", await sequence?.getVideoTrackList?.()],
      ["audio", await sequence?.getAudioTrackList?.()],
    ]) {
      if (list != null) readable = true;
      const entries = await trackListEntries(list);
      for (const entry of entries) {
        const items = await trackItems(entry.track);
        for (const item of items) {
          const name = await itemField(item, ["getName"], ["name"]);
          const nodeId = await itemField(item, ["getNodeId", "getId"], ["nodeId", "id", "guid"]);
          const start = timeValueToSeconds(await itemField(
            item, ["getStartTime", "getStart", "getInPoint"], ["start", "startTime"]
          )) ?? 0;
          const end = timeValueToSeconds(await itemField(
            item, ["getEndTime", "getEnd", "getOutPoint"], ["end", "endTime"]
          )) ?? 0;
          fingerprints.push(`${kind}|${entry.index}|${nodeId || ""}|${name || ""}|${start.toFixed(6)}|${end.toFixed(6)}`);
        }
      }
    }
    return readable ? fingerprints : null;
  }

  function projectTreeSnapshot(tree) {
    return (tree || []).map((entry) => (
      `${entry.isFolder ? "bin" : "item"}|${entry.nodeId || ""}|${entry.name || ""}|${entry.path || ""}|${entry.mediaPath || ""}`
    ));
  }

  async function projectSequenceSnapshot(project) {
    if (!project?.getSequences) return null;
    try {
      const sequences = await project.getSequences();
      const result = [];
      for (const sequence of (sequences || [])) {
        const name = String(await sequence.getName?.() ?? sequence.name ?? "");
        const id = String(await sequence.getId?.() ?? sequence.sequenceID ?? sequence.id ?? "");
        result.push(`${id}|${name}`);
      }
      return result;
    } catch (_) {
      return null;
    }
  }

  async function verifySubsequenceCreation(result, {
    project,
    beforeSequences,
    rangeVerification,
    restoration,
  }) {
    const afterSequences = await projectSequenceSnapshot(project);
    const canVerify = beforeSequences !== null && afterSequences !== null;
    const verified = canVerify ? Math.min(1, fingerprintAddedCount(beforeSequences, afterSequences)) : null;
    const status = verificationStatus(1, verified, canVerify);
    return await attach(result, {
      action: "ocCreateSubsequenceFromRange",
      attempted: 1,
      reported: 1,
      verified,
      status,
      readBackMethod: canVerify ? "Project.getSequences() identity/name fingerprint diff" : "unavailable: Project.getSequences()",
      beforeState: { sequence_count: beforeSequences?.length ?? null },
      afterState: {
        sequence_count: afterSequences?.length ?? null,
        range_verified: Boolean(rangeVerification?.verified),
        source_range_restored: !restoration?.attempted || Boolean(restoration?.verified),
      },
      detail: "The new subsequence is re-read from the project sequence list after createSubsequence(); temporary source in/out points are also restored and checked.",
    });
  }

  async function ensure(action, result) {
    if (result?.host_write_verification) return result;
    const reported = result?.applied ?? result?.count ?? result?.created ??
      result?.renamed ?? result?.removed ?? result?.imported ?? null;
    return await attach(result || {}, {
      action: action || "unknown",
      attempted: null,
      reported,
      verified: null,
      status: "unverified",
      readBackMethod: "unavailable: bridge operation returned no verification contract",
      detail: "The write was accepted by a bridge path that does not yet expose an independent read-back. It is intentionally not counted as verified.",
    });
  }

  return {
    attach,
    ensure,
    fingerprintAddedCount,
    fingerprintDeltaCount,
    projectTreeSnapshot,
    projectSequenceSnapshot,
    sequenceTrackSnapshot,
    verifySubsequenceCreation,
    verificationStatus,
  };
}
