import { describe, expect, it } from "vitest";

import { createHostWriteVerifier } from "../../com.opencut.uxp/uxp-host-write-verification.js";

function createVerifier() {
  return createHostWriteVerifier({
    getPPro: () => ({
      app: {
        version: "26.3",
      },
    }),
    trackListEntries: async () => [],
    trackItems: async () => [],
    itemField: async () => null,
    timeValueToSeconds: () => null,
  });
}

describe("UXP host-write verification", () => {
  it("fails a reported write when read-back found no mutation", async () => {
    const verifier = createVerifier();
    const result = await verifier.attach(
      { ok: true, applied: 2 },
      {
        action: "ocApplySequenceCuts",
        attempted: 2,
        reported: 2,
        verified: 0,
        status: verifier.verificationStatus(2, 0, true),
        readBackMethod: "track-item fingerprint diff",
      },
    );

    expect(result).toMatchObject({
      ok: false,
      error_code: "HOST_WRITE_NOT_APPLIED",
      host_version: "26.3",
      attempted_count: 2,
      reported_count: 2,
      verified_count: 0,
      verification_status: "failed",
    });
  });

  it("labels bridge writes without a read-back API as unverified", async () => {
    const result = await createVerifier().ensure("ocExportSequenceRange", { ok: true, count: 1 });

    expect(result).toMatchObject({
      ok: true,
      reported_count: 1,
      verified_count: null,
      verification_status: "unverified",
      unverified: true,
    });
  });
});
