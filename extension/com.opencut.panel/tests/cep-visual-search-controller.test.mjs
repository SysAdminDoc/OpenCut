import { createRequire } from "node:module";

import { describe, expect, it } from "vitest";

const require = createRequire(import.meta.url);
const OpenCutFormat = require("../client/format-utils.js");
const { createVisualSearchController } = require("../client/visual-search-controller.js");
const { replaceTemplateValue } = require("../client/string-utils.js");

const JPEG_URI = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ==";

class FakeClassList {
  constructor() {
    this.names = new Set();
  }
  add(name) {
    this.names.add(name);
  }
  remove(name) {
    this.names.delete(name);
  }
  contains(name) {
    return this.names.has(name);
  }
  toggle(name, force) {
    const on = force === undefined ? !this.names.has(name) : Boolean(force);
    if (on) this.names.add(name);
    else this.names.delete(name);
    return on;
  }
}

class FakeElement {
  constructor(tagName) {
    this.tagName = tagName.toUpperCase();
    this.children = [];
    this.attributes = {};
    this.dataset = {};
    this.classList = new FakeClassList();
    this.ownText = "";
    this.value = "";
    this.disabled = false;
    this.title = "";
  }
  set className(value) {
    this.classList = new FakeClassList();
    String(value).split(/\s+/).filter(Boolean).forEach((name) => this.classList.add(name));
  }
  get className() {
    return [...this.classList.names].join(" ");
  }
  set textContent(value) {
    this.ownText = String(value);
    this.children = [];
  }
  get textContent() {
    return this.ownText + this.children.map((child) => child.textContent).join("");
  }
  set innerHTML(_value) {
    throw new Error("visual search must not build markup from strings");
  }
  appendChild(child) {
    if (child.isFragment) child.children.forEach((node) => this.children.push(node));
    else this.children.push(child);
    return child;
  }
  setAttribute(name, value) {
    this.attributes[name] = String(value);
  }
  getAttribute(name) {
    return name in this.attributes ? this.attributes[name] : null;
  }
  find(className) {
    if (this.classList.contains(className)) return this;
    for (const child of this.children) {
      const hit = child.find(className);
      if (hit) return hit;
    }
    return null;
  }
}

function setup({ mode = "visual", connected = true, busy = false, media, projectDir = "", maxValue = "10", indexed = 0 } = {}) {
  const elements = {
    footageSearchMode: new FakeElement("select"),
    footageSearchMax: new FakeElement("input"),
    footageSearchResults: new FakeElement("div"),
    runFootageSearchBtn: new FakeElement("button"),
    buildVisualIndexBtn: new FakeElement("button"),
  };
  elements.footageSearchMode.value = mode;
  elements.footageSearchMax.value = maxValue;
  elements.buildVisualIndexBtn.classList.add("hidden");
  elements.runFootageSearchBtn.ownText = "Search";
  const calls = { status: [], empty: [], alerts: [], toasts: [], jobs: [] };
  const startJob = (route) => (payload, handlers) => calls.jobs.push({ route, payload, handlers });
  const controller = createVisualSearchController({
    documentRef: {
      getElementById: (id) => elements[id] || null,
      createElement: (tagName) => new FakeElement(tagName),
      createDocumentFragment: () => Object.assign(new FakeElement("fragment"), { isFragment: true }),
    },
    format: OpenCutFormat,
    t: (_key, fallback) => fallback,
    replaceTemplateValue,
    setStatusLine: (id, text, state, title) => calls.status.push({ id, text, state, title }),
    renderSearchResultsEmpty: (title, copy, tone) => calls.empty.push({ title, copy, tone }),
    formatSearchProjectClips: (count) => `${count} project clip${count === 1 ? "" : "s"}`,
    rememberButtonText: (btn) => (btn ? btn.textContent : ""),
    setButtonText: (btn, text) => {
      btn.textContent = text;
    },
    showAlert: (message) => calls.alerts.push(message),
    showToast: (message, tone) => calls.toasts.push({ message, tone }),
    isConnected: () => connected,
    isBusy: () => busy,
    getProjectMedia: () =>
      media ?? [
        { name: "beach.mp4", path: "C:\\media\\beach.mp4" },
        { name: "voice.wav", path: "C:\\media\\voice.wav" },
        { name: "dusk.mov", path: "D:/media/dusk.mov" },
      ],
    getProjectDir: () => projectDir,
    getIndexedFileCount: () => indexed,
    isSelectedPath: (path) => path === "D:/media/dusk.mov",
    startSearch: startJob("/search/semantic"),
    startIndex: startJob("/search/semantic/index"),
  });
  return { controller, elements, calls };
}

describe("CEP visual search mode", () => {
  it("reads visual only from the Search by select", () => {
    expect(setup({ mode: "visual" }).controller.getMode()).toBe("visual");
    expect(setup({ mode: "transcript" }).controller.getMode()).toBe("transcript");
    expect(setup({ mode: "other" }).controller.getMode()).toBe("transcript");
  });

  it("shows the index button and the visual empty state in visual mode", () => {
    const { controller, elements, calls } = setup();
    controller.applyMode();
    expect(elements.buildVisualIndexBtn.classList.contains("hidden")).toBe(false);
    expect(calls.empty.at(-1).title).toBe("Search by what's on screen");
    expect(calls.status.at(-1).state).toBe("idle");

    const offline = setup({ connected: false });
    offline.controller.applyMode();
    expect(offline.calls.status.at(-1)).toMatchObject({ state: "warning" });
  });

  it("goes back to the transcript library state", () => {
    const { controller, elements, calls } = setup({ mode: "transcript", indexed: 4 });
    elements.buildVisualIndexBtn.classList.remove("hidden");
    controller.applyMode();
    expect(elements.buildVisualIndexBtn.classList.contains("hidden")).toBe(true);
    expect(calls.empty.at(-1).title).toBe("Search the footage library");
    expect(calls.status.at(-1).state).toBe("ready");
  });
});

describe("CEP visual search jobs", () => {
  it("posts the project's videos and a saved project's folder", () => {
    const { controller, elements, calls } = setup({ projectDir: "C:\\Projects\\Doc", maxValue: "500" });
    controller.search("red car at night");
    expect(calls.jobs).toHaveLength(1);
    expect(calls.jobs[0].route).toBe("/search/semantic");
    expect(calls.jobs[0].payload).toEqual({
      query: "red car at night",
      clip_paths: ["C:\\media\\beach.mp4", "D:/media/dusk.mov"],
      max_results: 50,
      no_input: true,
      project_dir: "C:\\Projects\\Doc",
    });
    expect(elements.runFootageSearchBtn.disabled).toBe(true);
    expect(elements.runFootageSearchBtn.textContent).toBe("Searching…");
    expect(calls.empty.at(-1).copy).toContain("2 project clips");

    calls.jobs[0].handlers.onFinally();
    expect(elements.runFootageSearchBtn.disabled).toBe(false);
    expect(elements.runFootageSearchBtn.textContent).toBe("Search");
  });

  it("leaves project_dir out for an unsaved project and defaults a bad limit", () => {
    const { controller, calls } = setup({ maxValue: "lots" });
    controller.search("beach");
    expect(calls.jobs[0].payload).toEqual({
      query: "beach",
      clip_paths: ["C:\\media\\beach.mp4", "D:/media/dusk.mov"],
      max_results: 10,
      no_input: true,
    });
  });

  it("starts nothing offline, without video clips or while another job runs", () => {
    const offline = setup({ connected: false });
    offline.controller.search("beach");
    expect(offline.calls.jobs).toHaveLength(0);
    expect(offline.calls.status.at(-1).text).toBe("Reconnect the backend to search the footage library.");

    const noVideo = setup({ media: [{ path: "C:\\media\\voice.wav" }] });
    noVideo.controller.search("beach");
    noVideo.controller.buildIndex();
    expect(noVideo.calls.jobs).toHaveLength(0);
    expect(noVideo.calls.alerts).toEqual([
      "There are no video clips in the project yet. Import footage into Premiere, then try again.",
      "There are no video clips in the project yet. Import footage into Premiere, then try again.",
    ]);

    const busy = setup({ busy: true });
    busy.controller.buildIndex();
    expect(busy.calls.jobs).toHaveLength(0);
    expect(busy.calls.status.at(-1).state).toBe("warning");
  });

  it("renders hits as text with the delegation data main.js reads", () => {
    const { controller, elements, calls } = setup();
    controller.search("beach");
    calls.jobs[0].handlers.onComplete({
      results: [
        { clip_path: "C:\\media\\beach.mp4", clip_name: "<img src=x onerror=alert(1)>", timestamp: 83.4, score: 0.3127, thumbnail: JPEG_URI },
        { clip_path: "D:/media/dusk.mov", timestamp: 3723, score: 0.2, thumbnail: "C:/Users/me/.opencut/thumb.jpg" },
        { timestamp: 5 },
      ],
    });

    const items = elements.footageSearchResults.children;
    expect(items).toHaveLength(3);
    expect(items[0].className).toBe("footage-result-item has-thumb");
    expect(items[0].dataset).toEqual({ path: "C:\\media\\beach.mp4", name: "<img src=x onerror=alert(1)>" });
    expect(items[0].find("footage-result-thumb").src).toBe(JPEG_URI);
    expect(items[0].find("footage-result-title").textContent).toBe("<img src=x onerror=alert(1)>");
    expect(items[0].find("footage-result-range").textContent).toBe("01:23");
    expect(items[0].find("footage-result-score").textContent).toBe("31% match");
    expect(items[0].getAttribute("aria-label")).toBe("<img src=x onerror=alert(1)> at 01:23, 31% match");

    expect(items[1].find("footage-result-thumb")).toBeNull();
    expect(items[1].classList.contains("is-selected")).toBe(true);
    expect(items[1].find("footage-result-title").textContent).toBe("dusk.mov");
    expect(items[1].find("footage-result-range").textContent).toBe("1:02:03");

    expect(items[2].disabled).toBe(true);
    expect(items[2].find("footage-result-score")).toBeNull();
    expect(items[2].getAttribute("aria-label")).toBe("Unavailable clip result");

    expect(calls.status.at(-1)).toMatchObject({ text: "3 matches ready. Click a result to load it into the workspace.", state: "ready" });
  });

  it("shows the no-match state and job errors", () => {
    const { controller, calls } = setup();
    controller.search("unicorn");
    calls.jobs[0].handlers.onComplete({ results: [] });
    expect(calls.empty.at(-1)).toMatchObject({ title: "No matches yet", tone: "warning" });
    expect(calls.status.at(-1).text).toBe("No clips matched that description.");

    calls.jobs[0].handlers.onError({ error: "CLIP model is not installed" });
    expect(calls.empty.at(-1)).toMatchObject({
      copy: "The visual search didn't finish. CLIP model is not installed",
      tone: "error",
    });
    expect(calls.status.at(-1)).toMatchObject({ state: "error", title: "CLIP model is not installed" });
  });

  it("builds the visual index with the same inputs", () => {
    const { controller, elements, calls } = setup({ projectDir: "/Users/me/Edit" });
    controller.buildIndex();
    expect(calls.jobs[0].route).toBe("/search/semantic/index");
    expect(calls.jobs[0].payload).toEqual({
      clip_paths: ["C:\\media\\beach.mp4", "D:/media/dusk.mov"],
      no_input: true,
      project_dir: "/Users/me/Edit",
    });
    expect(elements.buildVisualIndexBtn.disabled).toBe(true);
    expect(calls.status.at(-1).text).toBe("Indexing the key frames of 2 videos. The first run takes a while.");

    calls.jobs[0].handlers.onComplete({ clip_count: 1 });
    expect(calls.toasts.at(-1)).toEqual({
      message: "Visual index ready. 1 video can be searched by what's on screen.",
      tone: "success",
    });

    calls.jobs[0].handlers.onError({ message: "disk full" });
    expect(calls.status.at(-1)).toMatchObject({ text: "Couldn't build the visual index. disk full", state: "error" });

    calls.jobs[0].handlers.onStartError();
    expect(elements.buildVisualIndexBtn.disabled).toBe(false);
    expect(calls.status.at(-1).state).toBe("error");
  });
});
