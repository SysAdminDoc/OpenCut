import { describe, expect, it } from "vitest";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const collapsible = require("../client/collapsible-cards.js");

function makeElement(className) {
  const classes = new Set(className ? className.split(" ") : []);
  return {
    attributes: {},
    style: {},
    children: [],
    listeners: {},
    classList: {
      contains: (name) => classes.has(name),
      toggle: (name, force) => {
        const next = force === undefined ? !classes.has(name) : Boolean(force);
        if (next) classes.add(name);
        else classes.delete(name);
        return next;
      },
    },
    setAttribute(name, value) {
      this.attributes[name] = value;
    },
    getAttribute(name) {
      return Object.prototype.hasOwnProperty.call(this.attributes, name) ? this.attributes[name] : null;
    },
    addEventListener(type, handler) {
      (this.listeners[type] = this.listeners[type] || []).push(handler);
    },
    fire(type, event) {
      for (const handler of this.listeners[type] || []) handler(event || {});
    },
  };
}

function makeCard() {
  const header = makeElement("card-header");
  const body = makeElement("card-body");
  const footer = makeElement("card-footer");
  const card = makeElement("card");
  card.children = [header, body, footer];
  header.closest = (selector) => (selector === ".card" ? card : null);
  return { card, header, body, footer };
}

function makeScope(headers) {
  return { querySelectorAll: () => headers };
}

describe("initCollapsibleCards", () => {
  it("gives every header the button semantics a keyboard user needs", () => {
    const { header } = makeCard();
    expect(collapsible.initCollapsibleCards(makeScope([header]))).toBe(1);
    expect(header.getAttribute("role")).toBe("button");
    expect(header.getAttribute("tabindex")).toBe("0");
    expect(header.getAttribute("aria-expanded")).toBe("true");
  });

  it("binds each header once", () => {
    const { header } = makeCard();
    const scope = makeScope([header]);
    collapsible.initCollapsibleCards(scope);
    expect(collapsible.initCollapsibleCards(scope)).toBe(0);
    expect(header.listeners.click).toHaveLength(1);
  });

  it("reports collapsed state when the header starts collapsed", () => {
    const { header } = makeCard();
    header.classList.toggle("collapsed", true);
    collapsible.initCollapsibleCards(makeScope([header]));
    expect(header.getAttribute("aria-expanded")).toBe("false");
  });

  it("survives a scope with no query support", () => {
    expect(collapsible.initCollapsibleCards({})).toBe(0);
  });
});

describe("toggling", () => {
  it("hides every sibling after the header and announces the new state", () => {
    const { header, body, footer } = makeCard();
    collapsible.initCollapsibleCards(makeScope([header]));

    header.fire("click");
    expect(body.style.display).toBe("none");
    expect(footer.style.display).toBe("none");
    expect(header.getAttribute("aria-expanded")).toBe("false");

    header.fire("click");
    expect(body.style.display).toBe("");
    expect(header.getAttribute("aria-expanded")).toBe("true");
  });

  it("toggles on Enter and Space and swallows the key", () => {
    const { header, body } = makeCard();
    collapsible.initCollapsibleCards(makeScope([header]));

    let prevented = 0;
    const press = (key) => header.fire("keydown", { key, preventDefault: () => { prevented += 1; } });

    press("Enter");
    expect(body.style.display).toBe("none");
    press(" ");
    expect(body.style.display).toBe("");
    press("Spacebar");
    expect(body.style.display).toBe("none");
    expect(prevented).toBe(3);
  });

  it("ignores other keys", () => {
    const { header, body } = makeCard();
    collapsible.initCollapsibleCards(makeScope([header]));
    header.fire("keydown", { key: "a", preventDefault: () => { throw new Error("must not prevent"); } });
    expect(body.style.display).toBeUndefined();
  });

  it("leaves a header outside a card alone", () => {
    const header = makeElement("card-header");
    header.closest = () => null;
    collapsible.initCollapsibleCards(makeScope([header]));
    header.fire("click");
    expect(header.getAttribute("aria-expanded")).toBe("false");
  });
});
