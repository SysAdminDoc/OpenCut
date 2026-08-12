import { createRequire } from "node:module";

import { describe, expect, it, vi } from "vitest";

const require = createRequire(import.meta.url);
const verification = require("../client/host-write-verification.js");

describe("CEP host-write verification", () => {
  it("labels successful bridge results without read-back as unverified", () => {
    const notify = vi.fn();
    const result = verification.parse(
      JSON.stringify({ added: 3, host_version: "26.3" }),
      { action: "add_markers" },
      (_key, fallback) => fallback,
      notify,
    );

    expect(result.host_write_verification).toMatchObject({
      action: "add_markers",
      reported_count: 3,
      verified_count: null,
      verification_status: "unverified",
    });
    expect(result.unverified).toBe(true);
    expect(notify).toHaveBeenCalledWith(expect.stringContaining("read-back"), "warning");
  });

  it("turns a reported-success no-op into an explicit failure", () => {
    const notify = vi.fn();
    const hostWrite = {
      host_write_verification: {
        verification_status: "failed",
        reported_count: 2,
        verified_count: 0,
      },
    };

    const result = verification.ensure(hostWrite, {}, (_key, fallback) => fallback, notify);

    expect(result.error_code).toBe("HOST_WRITE_NOT_APPLIED");
    expect(result.error).toContain("reported success");
    expect(notify).toHaveBeenCalledWith(expect.stringContaining("reported success"), "error");
    expect(verification.latest()).toBe(hostWrite.host_write_verification);
  });
});
