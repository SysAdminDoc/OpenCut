import { describe, expect, it } from "vitest";

import { createVisualSearchController } from "../../com.opencut.uxp/uxp-visual-search-controller.js";

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
    this.listeners = {};
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
    this.children.push(child);
    return child;
  }
  setAttribute(name, value) {
    this.attributes[name] = String(value);
  }
  getAttribute(name) {
    return name in this.attributes ? this.attributes[name] : null;
  }
  addEventListener(type, listener) {
    (this.listeners[type] ||= []).push(listener);
  }
  click() {
    (this.listeners.click || []).forEach((listener) => listener());
  }
  find(className) {
    if (this.classList.contains(className)) return this;
    for (const child of this.children) {
      const hit = child.find?.(className);
      if (hit) return hit;
    }
    return null;
  }
  findAll(className, out = []) {
    if (this.classList.contains(className)) out.push(this);
    this.children.forEach((child) => child.findAll?.(className, out));
    return out;
  }
}

function fill(template, values = {}) {
  return template.replace(/\{(\w+)\}/g, (match, key) => (key in values ? String(values[key]) : match));
}

function setup({ mode = "visual", online = true, clips, projectDir = "" } = {}) {
  const elements = {
    searchMode: new FakeElement("select"),
    runFootageSearchBtn: new FakeElement("button"),
    runVisualIndexBtn: new FakeElement("button"),
    searchResultList: new FakeElement("div"),
  };
  elements.searchMode.value = mode;
  const calls = { toasts: [], statuses: [], loading: [], resets: [], loads: [], syncs: 0, jobs: [] };
  const status = {};
  const ui = {
    showToast: (message, tone) => calls.toasts.push({ message, tone }),
    setStatus: (message, tone) => calls.statuses.push({ message, tone }),
    setButtonLoading: (id, on) => calls.loading.push([id, on]),
    showProcessing: () => {},
    hideProcessing: () => {},
    setProgress: () => {},
    setProcessingMsg: () => {},
  };
  const startJob = (route) => (payload, onProgress, onComplete, onError) => {
    calls.jobs.push({ route, payload, onProgress, onComplete, onError });
    return Promise.resolve();
  };
  const controller = createVisualSearchController({
    documentRef: {
      getElementById: (id) => elements[id] || null,
      createElement: (tagName) => new FakeElement(tagName),
    },
    translate: (_key, fallback) => fallback,
    formatTranslate: (_key, fallback, values) => fill(fallback, values),
    formatCountTranslate: (count, _oneKey, one, _manyKey, many, values = {}) =>
      fill(count === 1 ? one : many, { ...values, count }),
    ui,
    setTextAndTitle: (id, text, title) => {
      status[id] = { text, title };
    },
    resetSearchResults: (kicker, message) => calls.resets.push({ kicker, message }),
    syncSearchPanelState: () => {
      calls.syncs += 1;
    },
    isBackendConnected: () => online,
    getSearchResultScoreLabel: (_item, index) => `Result ${index + 1}`,
    onLoadResult: (list, el, path, label) => calls.loads.push({ sameList: list === elements.searchResultList, el, path, label }),
    getInputs: async () => ({
      clips: clips ?? [
        { name: "beach", path: "C:\\media\\beach.mp4" },
        { name: "voice", path: "C:\\media\\voice.wav" },
        { name: "dusk", path: "D:/media/dusk.mov" },
      ],
      projectDir,
    }),
    startSearch: startJob("/search/semantic"),
    startIndex: startJob("/search/semantic/index"),
  });
  return { controller, elements, calls, status };
}

describe("UXP visual search mode", () => {
  it("reads visual only from the Search by select", () => {
    expect(setup({ mode: "visual" }).controller.getMode()).toBe("visual");
    expect(setup({ mode: "transcript" }).controller.getMode()).toBe("transcript");
    expect(setup({ mode: "something else" }).controller.getMode()).toBe("transcript");
  });

  it("owns the search button and shows the index button in visual mode", () => {
    const { controller, elements } = setup();
    expect(controller.syncButtons(true, "")).toBe(true);
    expect(elements.runFootageSearchBtn.disabled).toBe(true);
    expect(elements.runVisualIndexBtn.classList.contains("oc-hidden")).toBe(false);
    expect(elements.runVisualIndexBtn.disabled).toBe(false);

    controller.syncButtons(true, "red car at night");
    expect(elements.runFootageSearchBtn.disabled).toBe(false);

    controller.syncButtons(false, "red car at night");
    expect(elements.runFootageSearchBtn.disabled).toBe(true);
    expect(elements.runVisualIndexBtn.disabled).toBe(true);
    expect(elements.runVisualIndexBtn.title).toBe("Reconnect the backend before building the visual index.");
  });

  it("leaves the search button to the transcript rules and hides the index button", () => {
    const { controller, elements } = setup({ mode: "transcript" });
    elements.runFootageSearchBtn.title = "transcript title";
    expect(controller.syncButtons(true, "")).toBe(false);
    expect(elements.runFootageSearchBtn.title).toBe("transcript title");
    expect(elements.runVisualIndexBtn.classList.contains("oc-hidden")).toBe(true);
  });

  it("swaps the empty state when the mode changes", () => {
    const { controller, calls, status } = setup();
    controller.applyMode();
    expect(calls.resets.at(-1).kicker).toBe("Search by what's on screen");
    expect(status.searchStatus.text).toContain("Visual search reads the pictures");
    expect(calls.syncs).toBe(1);
  });
});

describe("UXP visual search jobs", () => {
  it("posts the project's videos and a saved project's folder", async () => {
    const { controller, calls, elements } = setup({ projectDir: "C:\\Projects\\Doc" });
    await controller.search("red car at night", 500);
    expect(calls.jobs).toHaveLength(1);
    expect(calls.jobs[0].route).toBe("/search/semantic");
    expect(calls.jobs[0].payload).toEqual({
      query: "red car at night",
      clip_paths: ["C:\\media\\beach.mp4", "D:/media/dusk.mov"],
      max_results: 100,
      project_dir: "C:\\Projects\\Doc",
    });
    expect(calls.loading).toEqual([["runFootageSearchBtn", true]]);
    expect(elements.searchResultList.children).toHaveLength(0);
  });

  it("leaves project_dir out for an unsaved project and defaults the limit", async () => {
    const { controller, calls } = setup();
    await controller.search("beach", Number.NaN);
    expect(calls.jobs[0].payload).toEqual({
      query: "beach",
      clip_paths: ["C:\\media\\beach.mp4", "D:/media/dusk.mov"],
      max_results: 10,
    });
  });

  it("starts nothing without a query or without video clips", async () => {
    const empty = setup();
    await empty.controller.search("", 10);
    expect(empty.calls.jobs).toHaveLength(0);
    expect(empty.calls.toasts.at(-1)).toEqual({ message: "Please enter a search query.", tone: "warning" });

    const noVideo = setup({ clips: [{ path: "C:\\media\\voice.wav" }] });
    await noVideo.controller.search("beach", 10);
    await noVideo.controller.buildIndex();
    expect(noVideo.calls.jobs).toHaveLength(0);
    expect(noVideo.status.searchStatus.text).toBe(
      "There are no video clips in the project yet. Import footage into Premiere, then try again.",
    );
  });

  it("renders hits as text, keeps only inline JPEG thumbnails and loads the clicked clip", async () => {
    const { controller, calls, elements, status } = setup();
    await controller.search("beach", 10);
    calls.jobs[0].onComplete({
      results: [
        { clip_path: "C:\\media\\beach.mp4", clip_name: "<img src=x onerror=alert(1)>", timestamp: 83.4, score: 0.3127, thumbnail: JPEG_URI },
        { clip_path: "D:/media/dusk.mov", timestamp: 3723, score: 0.2, thumbnail: "C:/Users/me/.opencut/thumb.jpg" },
        { timestamp: 5 },
      ],
    });

    const items = elements.searchResultList.children;
    expect(items).toHaveLength(3);
    expect(items[0].className).toBe("oc-result-list-item is-featured");
    expect(items[0].find("oc-result-list-item-title").textContent).toBe("<img src=x onerror=alert(1)>");
    expect(items[0].find("oc-result-list-item-meta").textContent).toBe("31% match");
    expect(items[0].find("oc-result-list-item-thumb").src).toBe(JPEG_URI);
    expect(items[0].getAttribute("aria-label")).toBe("<img src=x onerror=alert(1)> at 01:23, 31% match");
    expect(items[1].find("oc-result-list-item-thumb")).toBeNull();
    expect(items[1].find("oc-result-list-item-title").textContent).toBe("dusk.mov");
    expect(items[1].findAll("oc-result-list-item-badge").map((badge) => badge.textContent)).toEqual(["Visual", "1:02:03"]);
    expect(items[2].find("oc-result-list-item-meta").textContent).toBe("Result 3");

    items[1].click();
    expect(calls.loads).toHaveLength(1);
    expect(calls.loads[0].sameList).toBe(true);
    expect(calls.loads[0].el === items[1]).toBe(true);
    expect(calls.loads[0].path).toBe("D:/media/dusk.mov");
    expect(calls.loads[0].label).toBe("dusk.mov");

    items[2].click();
    expect(calls.loads).toHaveLength(1);
    expect(calls.toasts.at(-1).tone).toBe("warning");

    expect(status.searchStatus.text).toContain("3 matches are ready. Start with <img src=x onerror=alert(1)>");
    expect(calls.loading.at(-1)).toEqual(["runFootageSearchBtn", false]);
  });

  it("shows the no-match state for an empty result", async () => {
    const { controller, calls, status } = setup();
    await controller.search("unicorn", 10);
    calls.jobs[0].onComplete({ results: [] });
    expect(calls.resets.at(-1).kicker).toBe("No matches yet");
    expect(status.searchStatus.text).toBe("No clips matched that description.");
  });

  it("reports a failed search job", async () => {
    const { controller, calls, status } = setup();
    await controller.search("beach", 10);
    calls.jobs[0].onError("CLIP model is not installed");
    expect(status.searchStatus.text).toBe("Could not search the library. CLIP model is not installed");
    expect(calls.toasts.at(-1)).toEqual({ message: "Search error: CLIP model is not installed", tone: "error" });
    expect(calls.loading.at(-1)).toEqual(["runFootageSearchBtn", false]);
  });

  it("builds the visual index with the same inputs", async () => {
    const { controller, calls, status } = setup({ projectDir: "/Users/me/Edit" });
    await controller.buildIndex();
    expect(calls.jobs[0].route).toBe("/search/semantic/index");
    expect(calls.jobs[0].payload).toEqual({
      clip_paths: ["C:\\media\\beach.mp4", "D:/media/dusk.mov"],
      project_dir: "/Users/me/Edit",
    });
    expect(status.searchStatus.text).toBe("Indexing the key frames of 2 videos. The first run takes a while.");

    calls.jobs[0].onComplete({ clip_count: 1 });
    expect(status.searchStatus.text).toBe("Visual index ready. 1 video can be searched by what's on screen.");
    expect(calls.loading.at(-1)).toEqual(["runVisualIndexBtn", false]);

    calls.jobs[0].onError("disk full");
    expect(calls.toasts.at(-1)).toEqual({ message: "Couldn't build the visual index. disk full", tone: "error" });
  });

  it("asks for the backend before indexing", async () => {
    const { controller, calls } = setup({ online: false });
    await controller.buildIndex();
    expect(calls.jobs).toHaveLength(0);
    expect(calls.toasts.at(-1)).toEqual({
      message: "Reconnect the backend before building the visual index.",
      tone: "warning",
    });
  });
});
