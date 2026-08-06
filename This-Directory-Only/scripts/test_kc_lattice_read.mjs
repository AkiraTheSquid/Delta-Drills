#!/usr/bin/env node
/**
 * Where the knowledge graph gets a concept's number from.
 *
 * This is the guard on a failure that ran in production for a month without
 * raising anything. The graph read its learner model out of
 * `localStorage.adaptive_state_<email>`, which backend mode never writes, and
 * the server's own per-concept mastery — which it was already fetching for the
 * gate rings — went unread. Meanwhile the backend image was missing
 * `kc_atom_crosswalk.json`, so its answer for all 63 concepts was the bare
 * prior. Both sides had a reason to return a constant, and a constant looks
 * exactly like a learner who has not practised.
 *
 * So the precedence is pinned here: server first, browser crosswalk second,
 * and a `topic-proxy` row never presented as this concept's own measurement.
 *
 * Run: node This-Directory-Only/scripts/test_kc_lattice_read.mjs
 */
import { readFileSync } from "node:fs";
import { createContext, runInContext } from "node:vm";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const REPO = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const SOURCE = readFileSync(
  join(REPO, "Local_Deployed_Shared", "concept-graph", "kc_lattice_read.js"),
  "utf8",
);

let failures = 0;
function check(name, fn) {
  try {
    fn();
    console.log(`  ok   ${name}`);
  } catch (err) {
    failures += 1;
    console.log(`  FAIL ${name}\n       ${err.message}`);
  }
}
function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

/* A stub of just the DOM the note-renderer touches. `#kg-cy` is the cytoscape
   container; the note is inserted before it, in its parent. */
function makeDom() {
  const children = [];
  const parent = {
    insertBefore: (node) => { children.push(node); },
    removeChild: (node) => {
      const i = children.indexOf(node);
      if (i >= 0) children.splice(i, 1);
    },
  };
  const host = { id: "kg-cy", parentNode: parent };
  return {
    children,
    document: {
      getElementById: (id) =>
        id === "kg-cy" ? host : children.find((c) => c.id === id) || null,
      createElement: () => ({ id: "", style: { cssText: "" }, textContent: "", parentNode: parent }),
    },
  };
}

function load() {
  const dom = makeDom();
  const win = {};
  const ctx = createContext({ window: win, document: dom.document, Number, Math, Object, Date });
  ctx.window.document = dom.document;
  runInContext(SOURCE, ctx);
  return { win, dom };
}

/* A measured row the server is confident about, and the same concept as a
   topic proxy — the two cases whose difference is the whole point. */
const MEASURED = { tier: "measured", evidenced: true, mastery: 0.894, covered_w: 1 };
const PROXY = { tier: "topic-proxy", evidenced: true, mastery: 0.093, covered_w: 1 };
const UNMAPPED = { tier: "unmapped", evidenced: false, mastery: 0.02, covered_w: 0 };

const STATE = {
  atom_mastery: { "ndarray-shape": 0.4 },
  atom_last_ts: { "ndarray-shape": "2026-08-06T17:00:00Z" },
};

// Stand-in for kc_crosswalk_mastery.js, which is a separate file with its own
// tests — what matters here is only WHETHER it gets consulted.
function withCrosswalk(win, result) {
  win.kcCrosswalkReadiness = () => result;
}

console.log("kc_lattice_read");

check("the server's measured reading wins", () => {
  const { win } = load();
  withCrosswalk(win, { r: 0.11, atoms: ["x"], ts: null, coveredW: 0.9 });
  const out = win.kcLatticeReadiness("numpy.ndarray-model", { kcs: { "numpy.ndarray-model": MEASURED } }, STATE, (v) => v);
  assert(out.r === 0.894, `expected the lattice's 0.894, got ${out && out.r}`);
  assert(out.source === "atom", "a measured server reading is an atom-level source");
  assert(out.server === true, "the read must be marked as the server's");
});

check("a topic proxy is not passed off as this concept's own", () => {
  const { win } = load();
  withCrosswalk(win, null);   // the browser declines proxies for the same reason
  const out = win.kcLatticeReadiness("einops.merge-axes", { kcs: { "einops.merge-axes": PROXY } }, STATE, (v) => v);
  assert(out === null, `a topic-proxy row must fall through, got ${JSON.stringify(out)}`);
});

check("an unevidenced concept falls through", () => {
  const { win } = load();
  withCrosswalk(win, null);
  const row = { ...MEASURED, evidenced: false };
  const out = win.kcLatticeReadiness("numpy.ndarray-model", { kcs: { "numpy.ndarray-model": row } }, STATE, (v) => v);
  assert(out === null, "covered weight below the server's cut is not a measurement");
});

check("a guest with no lattice still reads their own crosswalk", () => {
  const { win } = load();
  withCrosswalk(win, { r: 0.55, atoms: ["ndarray-shape"], ts: "2026-08-06T17:00:00Z", coveredW: 0.8 });
  const out = win.kcLatticeReadiness("numpy.ndarray-model", null, STATE, (v) => v);
  assert(out.r === 0.55, `offline read expected 0.55, got ${out && out.r}`);
  assert(out.server === undefined, "an offline read is not the server's");
  assert(out.via[0] === "ndarray-shape", "the offline read reports which atoms it used");
});

check("no evidence anywhere returns null, never a number", () => {
  const { win } = load();
  withCrosswalk(win, null);
  assert(win.kcLatticeReadiness("numpy.sorting", null, null, (v) => v) === null,
    "with nothing to read the caller must reach its labelled fallback");
});

check("an all-unmapped lattice says so on screen", () => {
  const { win, dom } = load();
  const data = { kcs: { a: UNMAPPED, b: UNMAPPED }, next_kc: "a" };
  const lattice = win.kcLatticeNote(data);
  assert(lattice === data, "the note must pass the lattice through to the caller");
  assert(dom.children.length === 1, "a hollow lattice must render the warning");
  assert(/starting prior/.test(dom.children[0].textContent),
    `the warning must name the cause, got: ${dom.children[0].textContent}`);
});

check("a healthy lattice renders no warning, and clears a stale one", () => {
  const { win, dom } = load();
  win.kcLatticeNote({ kcs: { a: UNMAPPED, b: UNMAPPED } });
  assert(dom.children.length === 1, "precondition: the warning is up");
  const lattice = win.kcLatticeNote({ kcs: { a: MEASURED, b: PROXY } });
  assert(lattice !== null, "a healthy lattice is returned");
  assert(dom.children.length === 0, "the warning must come down once the server can measure again");
});

check("a guest's absent lattice is not reported as a failure", () => {
  const { win, dom } = load();
  assert(win.kcLatticeNote(null) === null, "no lattice is null, not an object");
  assert(dom.children.length === 0,
    "guests have no server to ask — that is not a degraded server");
});

if (failures) {
  console.log(`\nFAIL — ${failures} check(s)`);
  process.exit(1);
}
console.log("\nPASS");
