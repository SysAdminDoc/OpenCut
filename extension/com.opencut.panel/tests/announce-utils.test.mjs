import { describe, expect, it } from "vitest";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const announce = require("../client/announce-utils.js");

function makeRegion() {
  return { textContent: "" };
}

function makeRegions() {
  return { polite: makeRegion(), assertive: makeRegion() };
}

describe("setLiveRegionMessage", () => {
  it("writes the message", () => {
    const node = makeRegion();
    announce.setLiveRegionMessage(node, "Run finished.");
    expect(node.textContent).toBe("Run finished.");
  });

  it("re-announces an identical repeat by clearing first", () => {
    // Screen readers ignore a write that does not change the text, so two
    // identical runs would announce only once without the clear.
    const writes = [];
    const node = {
      set textContent(value) {
        writes.push(value);
      },
      get textContent() {
        return writes[writes.length - 1] ?? "";
      },
    };
    announce.setLiveRegionMessage(node, "Run finished.");
    announce.setLiveRegionMessage(node, "Run finished.");
    expect(writes).toEqual(["", "Run finished.", "", "Run finished."]);
  });

  it("coerces null to an empty string", () => {
    const node = makeRegion();
    announce.setLiveRegionMessage(node, null);
    expect(node.textContent).toBe("");
  });

  it("tolerates a missing node", () => {
    expect(() => announce.setLiveRegionMessage(null, "x")).not.toThrow();
  });
});

describe("announceResult", () => {
  it("routes success to the polite region only", () => {
    const regions = makeRegions();
    const used = announce.announceResult(regions, "polite", "Run finished.");
    expect(used).toBe("polite");
    expect(regions.polite.textContent).toBe("Run finished.");
    expect(regions.assertive.textContent).toBe("");
  });

  it("routes failure to the assertive region only", () => {
    const regions = makeRegions();
    const used = announce.announceResult(regions, "error", "Run failed: disk full");
    expect(used).toBe("assertive");
    expect(regions.assertive.textContent).toBe("Run failed: disk full");
    expect(regions.polite.textContent).toBe("");
  });

  it("never leaves both regions populated across a success then failure", () => {
    const regions = makeRegions();
    announce.announceResult(regions, "polite", "Run finished.");
    announce.announceResult(regions, "error", "Run failed.");
    expect(regions.polite.textContent).toBe("");
    expect(regions.assertive.textContent).toBe("Run failed.");
  });

  it("clears both regions", () => {
    const regions = makeRegions();
    announce.announceResult(regions, "error", "Run failed.");
    announce.clearAnnouncements(regions);
    expect(regions.polite.textContent).toBe("");
    expect(regions.assertive.textContent).toBe("");
  });

  it("tolerates missing regions", () => {
    expect(() => announce.announceResult(null, "polite", "x")).not.toThrow();
    expect(() => announce.clearAnnouncements(undefined)).not.toThrow();
  });
});

describe("focusWasStranded", () => {
  const doc = { body: { nodeType: 1 }, documentElement: { nodeType: 1 } };

  function liveElement(extra = {}) {
    return {
      nodeType: 1,
      offsetParent: { nodeType: 1 },
      hasAttribute: () => false,
      getAttribute: () => null,
      ...extra,
    };
  }

  it("treats a lost focus as stranded", () => {
    expect(announce.focusWasStranded(null, doc)).toBe(true);
    expect(announce.focusWasStranded(doc.body, doc)).toBe(true);
    expect(announce.focusWasStranded(doc.documentElement, doc)).toBe(true);
  });

  it("treats a now-disabled control as stranded", () => {
    expect(announce.focusWasStranded(liveElement({ disabled: true }), doc)).toBe(true);
  });

  it("treats a hidden control as stranded", () => {
    const hidden = liveElement({
      offsetParent: null,
      getBoundingClientRect: () => ({ width: 0, height: 0 }),
    });
    expect(announce.focusWasStranded(hidden, doc)).toBe(true);
  });

  it("leaves a still-usable control alone", () => {
    // The whole point: do not yank focus away from where the user is.
    expect(announce.focusWasStranded(liveElement(), doc)).toBe(false);
  });

  it("keeps a fixed-position control that still has a box", () => {
    const fixed = liveElement({
      offsetParent: null,
      getBoundingClientRect: () => ({ width: 120, height: 32 }),
    });
    expect(announce.focusWasStranded(fixed, doc)).toBe(false);
  });
});

describe("focusResultsRegion", () => {
  it("adds a programmatic-only tab stop and focuses", () => {
    let focused = 0;
    const attrs = {};
    const node = {
      nodeType: 1,
      getAttribute: (name) => (name in attrs ? attrs[name] : null),
      setAttribute: (name, value) => {
        attrs[name] = value;
      },
      focus: () => {
        focused += 1;
      },
    };
    expect(announce.focusResultsRegion(node)).toBe(true);
    expect(attrs.tabindex).toBe("-1");
    expect(focused).toBe(1);
  });

  it("does not overwrite an existing tabindex", () => {
    const attrs = { tabindex: "0" };
    const node = {
      nodeType: 1,
      getAttribute: (name) => (name in attrs ? attrs[name] : null),
      setAttribute: (name, value) => {
        attrs[name] = value;
      },
      focus: () => {},
    };
    announce.focusResultsRegion(node);
    expect(attrs.tabindex).toBe("0");
  });

  it("falls back when focus options are unsupported", () => {
    let calls = 0;
    const node = {
      nodeType: 1,
      getAttribute: () => "-1",
      setAttribute: () => {},
      focus: (options) => {
        calls += 1;
        if (options) throw new TypeError("options unsupported");
      },
    };
    expect(announce.focusResultsRegion(node)).toBe(true);
    expect(calls).toBe(2);
  });

  it("tolerates a missing node", () => {
    expect(announce.focusResultsRegion(null)).toBe(false);
  });
});
