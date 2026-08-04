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
    // What a check cell reads as once `dd_check` has printed into it. The
    // content script parses THIS — the notebook has no other way to report.
    textContent: "",
  };
}

/**
 * Run the content script over one notebook and hand back the resulting classes.
 * `document.body` is null on purpose: the toggle panel bails without it, which
 * keeps this harness to the tagging and off the widget's DOM.
 */
function run({ anchors, hash, settings, outputs }) {
  const cells = anchors.map(makeCell);
  // Output already in the notebook when the script first runs — a saved .ipynb
  // reopened, or a page the extension was reloaded onto mid-session.
  Object.entries(outputs || {}).forEach(([anchor, text]) => {
    cells[anchors.indexOf(anchor)].textContent = `dd_check(1)\n${text}`;
  });
  const root = { classList: classList() };
  const sent = [];
  // The Gemini policy the MAIN-world script (`colab_no_ai.js`) would have heard.
  // It is a DOM event rather than a message because that half cannot use
  // `chrome.*` at all, so this is the only place the wiring can be checked
  // without a browser.
  const gemini = [];
  const context = {
    console,
    setTimeout,
    CustomEvent: class { constructor(type) { this.type = type; } },
    document: {
      documentElement: root,
      body: null,
      querySelectorAll: (sel) => (sel === "div.cell" ? cells : []),
      createElement: () => ({ classList: classList(), querySelectorAll: () => [], querySelector: () => null }),
      dispatchEvent: (event) => gemini.push(event.type),
    },
    location: { hash },
    window: { addEventListener: () => {} },
    MutationObserver: class { observe() {} },
    chrome: {
      runtime: {
        lastError: undefined,
        sendMessage: (msg, cb) => {
          sent.push(msg);
          if (cb) cb();
        },
      },
      storage: {
        local: {
          get: (_key, cb) => cb({ dd_colab_view: settings }),
          set: () => {},
        },
      },
    },
  };
  runInContext(SOURCE, createContext(context));
  const read = () => ({
    focusOn: root.classList.contains("dd-focus"),
    themeOn: root.classList.contains("dd-theme"),
    hideSolutions: root.classList.contains("dd-hide-solutions"),
    noAi: root.classList.contains("dd-no-ai"),
    gemini: [...gemini],
    hidden: anchors.filter((_, i) => cells[i].classList.contains("dd-out-of-focus")),
    inFocus: anchors.filter((_, i) => cells[i].classList.contains("dd-in-focus")),
    solutions: anchors.filter((_, i) => cells[i].classList.contains("dd-solution")),
    shown: anchors.filter((_, i) => cells[i].classList.contains("dd-solution-shown")),
  });
  // `reveal` is what the panel reaches through colab.js's message switch.
  // `check` writes a verdict into a cell the way a finished dd_check does, then
  // re-runs the tagging pass the MutationObserver would have run.
  return {
    ...read(),
    read,
    sent,
    reveal: (n) => context.window.__ddFocus.reveal(n),
    check: (anchor, text) => {
      cells[anchors.indexOf(anchor)].textContent = `dd_check(1)\n${text}`;
      context.window.__ddFocus.rescan();
      return read();
    },
  };
}

const NOTEBOOK = [
  "dd-setup", "dd-checker", "dd-lesson-np-1", "dd-kp-numpy-slicing",
  "dd-q12", "dd-q12-code", "dd-q12-check", "dd-q12-solution",
  "dd-q123", "dd-q123-hints", "dd-q123-code", "dd-q123-check", "dd-q123-solution",
];
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
check("the problem's cells are in focus", focused.inFocus,
  ["dd-q123", "dd-q123-hints", "dd-q123-code", "dd-q123-check", "dd-q123-solution"]);
// dd-q12 must NOT come along: a prefix test without a digit boundary would drag
// it in, and the student would see two problems.
check("everything else is hidden, except setup and the checker", focused.hidden,
  ["dd-lesson-np-1", "dd-kp-numpy-slicing", "dd-q12", "dd-q12-code", "dd-q12-check", "dd-q12-solution"]);
check("the setup cell is never hidden", focused.hidden.includes("dd-setup"), false);
// The checker defines dd_check. Hidden by focus, every check cell below it is a
// NameError, which reads as broken starter code.
check("the checker cell is never hidden", focused.hidden.includes("dd-checker"), false);

console.log("the neighbouring problem, to prove the boundary both ways:");
const twelve = run({ anchors: NOTEBOOK, hash: "#scrollTo=dd-q12", settings: ON });
check("only dd-q12's own cells", twelve.inFocus,
  ["dd-q12", "dd-q12-code", "dd-q12-check", "dd-q12-solution"]);

console.log("solutions stay hidden until the learner has answered:");
const gated = run({ anchors: NOTEBOOK, hash: "#scrollTo=dd-q123", settings: ON });
check("the gate is on", gated.hideSolutions, true);
check("both answer cells are tagged", gated.solutions, ["dd-q12-solution", "dd-q123-solution"]);
check("neither is shown yet", gated.shown, []);
const revealed = gated.reveal("123");
check("reveal reports the problem", revealed.ok && revealed.problem, "123");
check("only that problem's answer opens", gated.read().shown, ["dd-q123-solution"]);
check("the other one stays shut", gated.read().shown.includes("dd-q12-solution"), false);
// The panel forwards whatever the page sent, so the number is checked here.
check("a non-numeric problem is refused", gated.reveal("../../etc").ok, false);
check("and nothing opened", gated.read().shown, ["dd-q123-solution"]);

console.log("there is no way to switch the gate off:");
const noToggle = run({ anchors: NOTEBOOK, hash: "", settings: { theme: false, focus: false, solutions: true } });
check("a stale 'solutions' setting cannot open it", noToggle.hideSolutions, true);
check("and nothing is shown", noToggle.shown, []);

console.log("running the check IS the submit:");
const ran = run({ anchors: NOTEBOOK, hash: "#scrollTo=dd-q123", settings: ON });
check("nothing reported before a check runs", ran.sent, []);
const passed = ran.check("dd-q123-check", "✅ Problem 123 — 5/5 cases passed.");
check("the answer opens", passed.shown, ["dd-q123-solution"]);
check("and the app is told", ran.sent, [{ type: "dd:check-result", problem: "123", correct: true }]);
// Re-rendering a cell must not re-report a grade the app already has.
ran.check("dd-q123-check", "✅ Problem 123 — 5/5 cases passed.");
check("an unchanged verdict is not re-sent", ran.sent.length, 1);
const failed = ran.check("dd-q123-check", "❌ Problem 123 — 2 of 5 cases failed.");
check("a re-run with a new verdict is", ran.sent.length, 2);
check("and it carries the failure", ran.sent[1].correct, false);
check("the answer stays open", failed.shown, ["dd-q123-solution"]);

console.log("a notebook reopened with its outputs still in it:");
// The FIRST pass only records what is already on screen. Replaying it would
// post a grade for work the learner did in some earlier session — and unlock
// the answer to a problem they have not looked at yet.
const reopened = run({
  anchors: NOTEBOOK,
  hash: "#scrollTo=dd-q123",
  settings: ON,
  outputs: { "dd-q123-check": "✅ Problem 123 — 5/5 cases passed." },
});
check("a saved verdict is not replayed", reopened.sent, []);
check("and it does not open the answer", reopened.shown, []);
check("a re-render changes nothing", reopened.read().shown, []);
// Re-running it in this session does report, though.
reopened.check("dd-q123-check", "❌ Problem 123 — 1 of 5 cases failed.");
check("a fresh run is reported", reopened.sent.length, 1);

console.log("a cell's own source can never look like a verdict:");
const sourceOnly = run({ anchors: NOTEBOOK, hash: "#scrollTo=dd-q123", settings: ON });
sourceOnly.check("dd-q123-check", "# Did it work? Run this.");
check("nothing reported", sourceOnly.sent, []);
check("and the answer stays hidden", sourceOnly.read().shown, []);

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

// ── Gemini's shadow text ─────────────────────────────────────────────
// Colab completes the learner's code with the answer unless something turns it
// off. Two ways that goes wrong, both silent: leaving it on (the learner is
// handed the solution on a problem the ladder says they cannot do yet), and
// turning it off too widely (the extension quietly disables a Google feature on
// every unrelated Colab notebook the browser ever opens). The class and the
// event have to agree, because the CSS backstop hides the ghost text while the
// MAIN-world script is what stops Tab from accepting it — hidden-but-live is
// the one state that must not exist.
console.log("Gemini autocomplete on one of our notebooks:");
const noAi = run({ anchors: NOTEBOOK, hash: "#scrollTo=dd-q123", settings: ON });
check("the CSS backstop is on", noAi.noAi, true);
check("and the editor was told, once", noAi.gemini, ["dd:gemini-off"]);
check("a re-render does not re-send it", noAi.read().gemini, ["dd:gemini-off"]);

console.log("Gemini autocomplete switched back on by the learner:");
const allowAi = run({
  anchors: NOTEBOOK, hash: "#scrollTo=dd-q123", settings: { ...ON, gemini: true },
});
check("no CSS backstop", allowAi.noAi, false);
check("and the editor is told to hand them back", allowAi.gemini, ["dd:gemini-on"]);

console.log("somebody else's Colab notebook:");
// No `dd-` anchor anywhere: not ours, so nothing of ours applies. An ARENA
// notebook is the real instance of this — 458 of them, nbformat 4.2, no ids.
const foreign = run({ anchors: ["", "", ""], hash: "", settings: ON });
check("its completions are left alone", foreign.noAi, false);
check("and nothing suppresses them", foreign.gemini, ["dd:gemini-on"]);

if (failures) {
  console.log(`\nFAILED ${failures} check(s)`);
  process.exit(1);
}
console.log("\nPASS colab focus tagging");
