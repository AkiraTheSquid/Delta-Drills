#!/usr/bin/env node
"use strict";

/* Dependency-free behavior tests for practice/lessons.js. Uses tiny DOM
   doubles so CI/local validation needs Node only, not a browser package. */

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const SOURCE = fs.readFileSync(
  path.resolve(__dirname, "../../../Local_Deployed_Shared/practice/lessons.js"),
  "utf8",
);

/* index.html loads practice/config.js before lessons.js, and lessons.js reads
   from it at evaluation time (DEFAULT_EDITOR_CODE) as well as at call time
   (displayTopic). Evaluate it into the same sandbox for the same reason the
   page does, rather than stubbing its values here — a stub would let the two
   copies drift, which is the exact bug config.js's own comments record. */
const CONFIG_SOURCE = fs.readFileSync(
  path.resolve(__dirname, "../../../Local_Deployed_Shared/practice/config.js"),
  "utf8",
);

class FakeClassList {
  constructor() { this.values = new Set(); }
  add(...names) { names.forEach((name) => this.values.add(name)); }
  remove(...names) { names.forEach((name) => this.values.delete(name)); }
  contains(name) { return this.values.has(name); }
  replaceFrom(value) { this.values = new Set(String(value).split(/\s+/).filter(Boolean)); }
}

class FakeElement {
  constructor(tag, document) {
    this.tagName = tag.toUpperCase();
    this.ownerDocument = document;
    this.classList = new FakeClassList();
    this.attributes = {};
    this.listeners = {};
    this.children = [];
    this.hidden = false;
    this.disabled = false;
    this._innerHTML = "";
  }
  set className(value) { this.classList.replaceFrom(value); }
  get className() { return [...this.classList.values].join(" "); }
  set innerHTML(value) { this._innerHTML = value; }
  get innerHTML() { return this._innerHTML; }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  getAttribute(name) { return this.attributes[name]; }
  appendChild(child) { this.children.push(child); return child; }
  addEventListener(name, fn) { this.listeners[name] = fn; }
  focus() { this.ownerDocument.activeElement = this; }
  contains(element) {
    return element === this || this.children.includes(element) ||
      element === this._body || element === this._button || element === this._title;
  }
  querySelector(selector) {
    if (selector === "#lesson-gate-body") {
      this._body ||= new FakeElement("div", this.ownerDocument);
      this._body.id = "lesson-gate-body";
      return this._body;
    }
    if (selector === "#lesson-gate-continue") {
      this._button ||= new FakeElement("button", this.ownerDocument);
      this._button.id = "lesson-gate-continue";
      return this._button;
    }
    if (selector === "#lesson-gate-title") {
      this._title ||= new FakeElement("h2", this.ownerDocument);
      this._title.id = "lesson-gate-title";
      return this._title;
    }
    return null;
  }
  querySelectorAll() {
    return this._button ? [this._button] : [];
  }
}

function makeHarness({ mode = "local", qmatrix = {}, lessons = [], fetchFailure = false } = {}) {
  const timers = [];
  const apiCalls = [];
  const storage = new Map();
  /* lessons.js reaches for page furniture it does not own — #code-editor and
     #output-area, which live in index.html — to reset them when the gate
     closes. Mint one element per id on demand and keep it, so a test can read
     back what the gate wrote there. */
  const byId = new Map();
  const document = {
    activeElement: null,
    createElement(tag) { return new FakeElement(tag, document); },
    getElementById(id) {
      if (!byId.has(id)) {
        const el = new FakeElement("div", document);
        el.id = id;
        byId.set(id, el);
      }
      return byId.get(id);
    },
  };
  document.body = new FakeElement("body", document);
  const trigger = new FakeElement("button", document);
  trigger.focus();

  const localStorage = {
    getItem(key) { return storage.has(key) ? storage.get(key) : null; },
    setItem(key, value) { storage.set(key, String(value)); },
  };
  const fetch = async (requestedPath) => {
    if (fetchFailure) throw new Error("offline");
    const data = requestedPath.includes("qmatrix_tags")
      ? qmatrix
      : { lessons };
    return { ok: true, status: 200, async json() { return data; } };
  };
  const context = {
    window: {}, document, localStorage, fetch,
    /* lessons.js ends with a second IIFE that reads `?lesson=<kc>` at load.
       Without these the whole file throws before a single assertion runs —
       the suite had been dead on arrival rather than failing loudly. Empty
       search = the ordinary practice flow, which is what these tests cover. */
    location: { search: "" },
    URLSearchParams,
    practiceMode: mode,
    getPracticeStorageKey: () => "test_user",
    apiFetch: async (...args) => { apiCalls.push(args); return { status: 500, ok: false }; },
    handleExpiredToken: () => {},
    setTimeout: (fn) => { timers.push(fn); return timers.length; },
    console: { warn: () => {} },
  };
  vm.createContext(context);
  vm.runInContext(CONFIG_SOURCE, context, { filename: "config.js" });
  vm.runInContext(SOURCE, context, { filename: "lessons.js" });
  return {
    gate: context.window.LessonGate,
    document,
    overlay: () => document.body.children[0],
    storage,
    apiCalls,
    timers,
    trigger,
  };
}

const lessonFixture = {
  id: "lesson-1",
  title: "Lesson One",
  topic: "numpy",
  kps: [
    { kc: "kc.one", title: "KC One", concept_markdown: "Concept", worked_example_markdown: "Example" },
    { kc: "kc.two", title: "KC Two", concept_markdown: "Concept", worked_example_markdown: "Example" },
  ],
};

async function run() {
  let h = makeHarness({
    qmatrix: { "7": { target_kcs: ["kc.one", "kc.one", "kc.two"] } },
    lessons: [lessonFixture],
  });
  let done = 0;
  assert.equal(await h.gate.maybeShow({ question_id: 7 }, () => { done++; }), true);
  const overlay = h.overlay();
  const button = overlay.querySelector("#lesson-gate-continue");
  assert.equal(overlay.getAttribute("role"), "dialog");
  assert.equal(overlay.getAttribute("aria-modal"), "true");
  assert.equal(overlay.getAttribute("aria-hidden"), "false");
  assert.equal(h.document.activeElement.id, "lesson-gate-title");
  assert.equal(h.document.body.classList.contains("lesson-gate-open"), true);
  let prevented = false;
  overlay.listeners.keydown({
    key: "Tab", shiftKey: true,
    preventDefault() { prevented = true; },
  });
  assert.equal(prevented, true);
  assert.equal(h.document.activeElement.id, "lesson-gate-continue");

  button.onclick();
  button.onclick();
  assert.equal(done, 0, "double click must not skip second concept");
  let exposed = JSON.parse(h.storage.get("test_user_kc_exposure"));
  assert.deepEqual(Object.keys(exposed), ["kc.one"]);
  h.timers.shift()();
  button.onclick();
  button.onclick();
  assert.equal(done, 1, "completion callback must run once");
  exposed = JSON.parse(h.storage.get("test_user_kc_exposure"));
  assert.deepEqual(Object.keys(exposed), ["kc.one", "kc.two"]);
  assert.equal(overlay.classList.contains("hidden"), true);
  assert.equal(overlay.getAttribute("aria-hidden"), "true");
  assert.equal(h.document.body.classList.contains("lesson-gate-open"), false);

  h = makeHarness({ lessons: [lessonFixture] });
  assert.equal(await h.gate.maybeShow({ diagnostic_active: true }, () => {}), false);
  assert.equal(h.overlay(), undefined, "diagnostic must not build overlay");

  h = makeHarness({ mode: "backend", lessons: [lessonFixture] });
  done = 0;
  assert.equal(await h.gate.maybeShow({
    question_id: 8,
    lesson_gate: [{ kc: "kc.one" }, { kc: "kc.one" }],
  }, () => { done++; }), true);
  h.overlay().querySelector("#lesson-gate-continue").onclick();
  await Promise.resolve();
  assert.equal(done, 1);
  assert.equal(h.apiCalls.length, 1, "backend exposure must post once per unique KC");

  h = makeHarness({ fetchFailure: true, qmatrix: { "9": { target_kcs: ["kc.one"] } } });
  assert.equal(await h.gate.maybeShow({ question_id: 9 }, () => {}), false);

  console.log("ALL PASS");
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});
