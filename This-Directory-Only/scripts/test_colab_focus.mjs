#!/usr/bin/env node
/**
 * The Colab focus tagger, against a stub DOM.
 *
 * `extension/content/colab_focus.js` decides which notebook cells disappear. It
 * only runs on colab.research.google.com behind a Google login, so the browser
 * checks that cover the rest of this project cannot reach it — and its failure
 * mode is a blank notebook, which looks exactly like a page that did not load.
 *
 * So the tagging is exercised here instead: a fake cell list, a fake fragment,
 * and assertions on the classes that CSS keys off. No Colab, no network.
 *
 * Run: node This-Directory-Only/scripts/test_colab_focus.mjs
 */
import { readFileSync } from "node:fs";
import { createContext, runInContext } from "node:vm";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const REPO = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const SOURCE = readFileSync(join(REPO, "extension", "content", "colab_focus.js"), "utf8");

function classList() {
  const set = new Set();
  return {
    set,
    add: (c) => set.add(c),
    remove: (c) => set.delete(c),
    contains: (c) => set.has(c),
    toggle: (c, on) => (on ? set.add(c) : set.delete(c)),
  };
}

function makeCell(anchor) {
  return {
    id: anchor ? `cell-${anchor}` : "",
    classList: classList(),
    querySelector: () => null,
  };
}

/**
 * Run the content script over one notebook and hand back the resulting classes.
 * `document.body` is null on purpose: the toggle panel bails without it, which
 * keeps this harness to the tagging and off the widget's DOM.
 */
function run({ anchors, hash, settings }) {
  const cells = anchors.map(makeCell);
  const root = { classList: classList() };
  const context = {
    console,
    setTimeout,
    document: {
      documentElement: root,
      body: null,
      querySelectorAll: (sel) => (sel === "div.cell" ? cells : []),
      createElement: () => ({ classList: classList(), querySelectorAll: () => [], querySelector: () => null }),
    },
    location: { hash },
    window: { addEventListener: () => {} },
    MutationObserver: class { observe() {} },
    chrome: {
      storage: {
        local: {
          get: (_key, cb) => cb({ dd_colab_view: settings }),
          set: () => {},
        },
      },
    },
  };
  runInContext(SOURCE, createContext(context));
  return {
    focusOn: root.classList.contains("dd-focus"),
    themeOn: root.classList.contains("dd-theme"),
    hidden: anchors.filter((_, i) => cells[i].classList.contains("dd-out-of-focus")),
    inFocus: anchors.filter((_, i) => cells[i].classList.contains("dd-in-focus")),
  };
}

const NOTEBOOK = ["dd-setup", "dd-lesson-np-1", "dd-kp-numpy-slicing", "dd-q12", "dd-q12-code", "dd-q123", "dd-q123-hints", "dd-q123-code"];
const ON = { theme: true, focus: true };

let failures = 0;
function check(label, actual, expected) {
  const a = JSON.stringify(actual);
  const b = JSON.stringify(expected);
  if (a === b) {
    console.log(`  ok  ${label}`);
  } else {
    console.log(`  FAIL ${label}\n       expected ${b}\n       actual   ${a}`);
    failures += 1;
  }
}

console.log("focus on a problem that is in the notebook:");
const focused = run({ anchors: NOTEBOOK, hash: "#scrollTo=dd-q123", settings: ON });
check("focus is live", focused.focusOn, true);
check("the problem's three cells are in focus", focused.inFocus, ["dd-q123", "dd-q123-hints", "dd-q123-code"]);
// dd-q12 must NOT come along: a prefix test without a digit boundary would drag
// it in, and the student would see two problems.
check("everything else is hidden, except setup", focused.hidden,
  ["dd-lesson-np-1", "dd-kp-numpy-slicing", "dd-q12", "dd-q12-code"]);
check("the setup cell is never hidden", focused.hidden.includes("dd-setup"), false);

console.log("the neighbouring problem, to prove the boundary both ways:");
const twelve = run({ anchors: NOTEBOOK, hash: "#scrollTo=dd-q12", settings: ON });
check("only dd-q12's own cells", twelve.inFocus, ["dd-q12", "dd-q12-code"]);

console.log("a problem that is NOT in this notebook:");
const absent = run({ anchors: NOTEBOOK, hash: "#scrollTo=dd-q999", settings: ON });
check("focus stays off", absent.focusOn, false);
check("nothing is hidden", absent.hidden, []);

console.log("an ARENA notebook — nbformat 4.2, no cell ids at all:");
const arena = run({ anchors: ["", "", ""], hash: "#scrollTo=dd-q123", settings: ON });
check("focus stays off", arena.focusOn, false);
check("nothing is hidden", arena.hidden, []);

console.log("no problem in the URL:");
const bare = run({ anchors: NOTEBOOK, hash: "", settings: ON });
check("nothing is hidden", bare.hidden, []);

console.log("focus switched off, theme left on:");
const themeOnly = run({ anchors: NOTEBOOK, hash: "#scrollTo=dd-q123", settings: { theme: true, focus: false } });
check("nothing is hidden", themeOnly.hidden, []);
check("the theme is still applied", themeOnly.themeOn, true);

console.log("both switched off:");
const off = run({ anchors: NOTEBOOK, hash: "#scrollTo=dd-q123", settings: { theme: false, focus: false } });
check("no theme class", off.themeOn, false);
check("no focus class", off.focusOn, false);

if (failures) {
  console.log(`\nFAILED ${failures} check(s)`);
  process.exit(1);
}
console.log("\nPASS colab focus tagging");
