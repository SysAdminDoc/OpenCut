import { describe, expect, it } from "vitest";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const reasons = require("../client/disabled-reasons.js");

function makeControl(attributes = {}, disabled = true) {
  return {
    attributes: { ...attributes },
    disabled,
    setAttribute(name, value) {
      this.attributes[name] = String(value);
    },
    getAttribute(name) {
      return Object.prototype.hasOwnProperty.call(this.attributes, name)
        ? this.attributes[name]
        : null;
    },
    hasAttribute(name) {
      return Object.prototype.hasOwnProperty.call(this.attributes, name);
    },
    removeAttribute(name) {
      delete this.attributes[name];
    },
  };
}

const REASON = "Connect the backend and load project items first.";

describe("syncControl", () => {
  it("shows the reason while the control is disabled", () => {
    const control = makeControl({ "data-disabled-title": REASON });
    expect(reasons.syncControl(control)).toBe(true);
    expect(control.getAttribute("title")).toBe(REASON);
  });

  it("puts back the working title once the control is usable", () => {
    const control = makeControl({
      "data-disabled-title": REASON,
      title: "Rename every project item",
    });
    reasons.syncControl(control);
    expect(control.getAttribute("title")).toBe(REASON);

    control.disabled = false;
    reasons.syncControl(control);
    expect(control.getAttribute("title")).toBe("Rename every project item");
    expect(control.hasAttribute("data-enabled-title")).toBe(false);
  });

  it("leaves no tooltip behind on a control that never had one", () => {
    const control = makeControl({ "data-disabled-title": REASON });
    reasons.syncControl(control);
    control.disabled = false;
    reasons.syncControl(control);
    expect(control.hasAttribute("title")).toBe(false);
  });

  it("survives repeated syncs without capturing the reason as the working title", () => {
    const control = makeControl({
      "data-disabled-title": REASON,
      title: "Rename every project item",
    });
    reasons.syncControl(control);
    reasons.syncControl(control);
    control.disabled = false;
    reasons.syncControl(control);
    expect(control.getAttribute("title")).toBe("Rename every project item");
  });

  it("ignores controls that declare no reason", () => {
    const control = makeControl({ title: "Something" });
    expect(reasons.syncControl(control)).toBe(false);
    expect(control.getAttribute("title")).toBe("Something");
  });
});

describe("syncDisabledReasons", () => {
  it("counts only the controls it annotated", () => {
    const annotated = makeControl({ "data-disabled-title": REASON });
    const plain = makeControl({});
    const scope = { querySelectorAll: () => [annotated, plain] };
    expect(reasons.syncDisabledReasons(scope)).toBe(1);
  });

  it("returns zero for a scope with no query support", () => {
    expect(reasons.syncDisabledReasons({})).toBe(0);
  });
});

describe("observeDisabledReasons", () => {
  it("syncs immediately and re-syncs whatever the observer reports", () => {
    const control = makeControl({ "data-disabled-title": REASON });
    let handler = null;
    let observed = null;
    class FakeObserver {
      constructor(callback) {
        handler = callback;
      }
      observe(target, options) {
        observed = { target, options };
      }
    }
    const scope = { querySelectorAll: () => [control] };
    const observer = reasons.observeDisabledReasons(scope, FakeObserver);

    expect(observer).toBeInstanceOf(FakeObserver);
    expect(control.getAttribute("title")).toBe(REASON);
    // The translated reason arrives after the locale loads, so the attribute
    // that carries it has to be watched alongside `disabled`.
    expect(observed.options.attributeFilter).toContain("disabled");
    expect(observed.options.attributeFilter).toContain("data-disabled-title");

    control.disabled = false;
    handler([{ target: control }]);
    expect(control.hasAttribute("title")).toBe(false);
  });

  it("does nothing without an observer implementation", () => {
    expect(reasons.observeDisabledReasons({ querySelectorAll: () => [] }, null)).toBeNull();
  });
});
