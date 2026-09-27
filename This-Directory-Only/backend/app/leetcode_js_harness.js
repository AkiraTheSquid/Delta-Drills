/* JavaScript test harness for the LeetCode drills (driven by leetcode_js.py).

   node leetcode_js_harness.js <payload.json>
   payload = {code, cases: [{call, expected}], caseTimeoutMs}

   The learner's code runs once in a fresh vm context; then each case's `call`
   (a JS expression translated from the bank's Python call) runs in that same
   context and is compared with `expected` (JSON of the Python literal). The
   result rows go to stdout after the `__DELTA_TESTS__` marker, one per case in
   bank order, shaped like code_runner.run_function_tests' rows. If the code
   itself does not load, the error goes to stderr and the process exits 1, the
   way a Python submission that fails to import answers with its traceback.

   Equality mirrors code_runner._delta_equal: arrays element-wise, floats with
   rtol 1e-5 / atol 1e-6, integers exact, true == 1 (Python's bool is an int),
   and undefined == null (a JS method that returns nothing is Python's None). */
"use strict";

const fs = require("fs");
const util = require("util");
const vm = require("vm");

const payload = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const CASE_TIMEOUT_MS = payload.caseTimeoutMs || 5000;
const OUTPUT_LIMIT = 64 * 1024;
const RTOL = 1e-5;
const ATOL = 1e-6;

let printed = 0;
const write = (stream, args) => {
  if (printed > OUTPUT_LIMIT) return;
  const text = util.format(...args) + "\n";
  printed += text.length;
  stream.write(printed > OUTPUT_LIMIT ? "… output truncated\n" : text);
};
const sandboxConsole = {
  log: (...a) => write(process.stdout, a),
  info: (...a) => write(process.stdout, a),
  debug: (...a) => write(process.stdout, a),
  warn: (...a) => write(process.stdout, a),
  error: (...a) => write(process.stdout, a),
  table: (...a) => write(process.stdout, a),
};

/* Fixture helpers — the JS twins of the Python setup_code helpers the bank's
   test calls use (list_node, tree_node, is_same_tree, _lc_design, sorted…).
   Defined INSIDE the context so they build the same ListNode/TreeNode the
   learner's code sees. */
const PRELUDE = `
function ListNode(val, next) { this.val = (val === undefined ? 0 : val); this.next = (next === undefined ? null : next); }
function TreeNode(val, left, right) {
  this.val = (val === undefined ? 0 : val);
  this.left = (left === undefined ? null : left);
  this.right = (right === undefined ? null : right);
}
var __listNode = (values) => {
  let head = null, cur = null;
  for (const v of values) { const n = new ListNode(v); if (!head) head = cur = n; else { cur.next = n; cur = n; } }
  return head;
};
var __listNodeCycle = ([values, pos]) => {
  const head = __listNode(values);
  if (head && pos >= 0) {
    let tail = null, target = null, i = 0, cur = head;
    while (cur) { if (i === pos) target = cur; tail = cur; cur = cur.next; i++; }
    tail.next = target;
  }
  return head;
};
var __treeNode = (values) => {
  if (!values.length || values[0] === null) return null;
  const root = new TreeNode(values[0]);
  const queue = [root]; let i = 1, qi = 0;
  while (qi < queue.length && i < values.length) {
    const node = queue[qi++];
    if (i < values.length && values[i] !== null) { node.left = new TreeNode(values[i]); queue.push(node.left); }
    i++;
    if (i < values.length && values[i] !== null) { node.right = new TreeNode(values[i]); queue.push(node.right); }
    i++;
  }
  return root;
};
var __fromListNode = (node) => {
  const out = [];
  while (node != null && out.length < 100000) { out.push(node.val); node = node.next; }
  return out;
};
var __fromTreeNode = (root) => {
  const out = [], queue = [root]; let qi = 0;
  while (qi < queue.length) {
    const node = queue[qi++];
    if (node == null) { out.push(null); continue; }
    out.push(node.val); queue.push(node.left, node.right);
  }
  while (out.length && out[out.length - 1] === null) out.pop();
  return out;
};
var __isSameList = (a, b) => {
  while (a && b) { if (a.val !== b.val) return false; a = a.next; b = b.next; }
  return !a && !b;
};
var __isSameTree = (p, q) => {
  if (!p && !q) return true;
  if (!p || !q || p.val !== q.val) return false;
  return __isSameTree(p.left, q.left) && __isSameTree(p.right, q.right);
};
var __pyCompare = (a, b) => {
  if (Array.isArray(a) && Array.isArray(b)) {
    for (let i = 0; i < Math.min(a.length, b.length); i++) {
      const c = __pyCompare(a[i], b[i]);
      if (c) return c;
    }
    return a.length - b.length;
  }
  if (typeof a === "string" && typeof b === "string") return a < b ? -1 : a > b ? 1 : 0;
  return Number(a) - Number(b);
};
var __sorted = (xs) => Array.from(xs).sort(__pyCompare);
var __design = (Cls, ops, args) => {
  let obj = null; const outs = [];
  ops.forEach((op, i) => {
    if (obj === null) { obj = new Cls(...args[i]); outs.push(null); return; }
    if (typeof obj[op] !== "function") throw new TypeError(op + " is not a method of " + ops[0]);
    const r = obj[op](...args[i]);
    outs.push(r === undefined ? null : r);
  });
  return outs;
};
var __solution = (name, fn, Sol) => {
  if (typeof fn === "function") return fn;
  if (typeof Sol === "function") return (...a) => new Sol()[name](...a);
  throw new ReferenceError(name + " is not defined — declare \`var " + name + " = function(...) {...}\`");
};
`;

const isInt = (x) => typeof x === "number" && Number.isInteger(x);
const numeric = (x) => (typeof x === "boolean" ? Number(x) : x);

function equal(a, b) {
  if (a === undefined) a = null;
  if (b === undefined) b = null;
  if (ArrayBuffer.isView(a) && !(a instanceof DataView)) a = Array.from(a);
  if (Array.isArray(a) || Array.isArray(b)) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false;
    // A plain loop, not a.every: `a` comes from the learner's realm, whose
    // Array.prototype.every they can replace with () => true.
    for (let i = 0; i < a.length; i++) if (!equal(a[i], b[i])) return false;
    return true;
  }
  const na = numeric(a), nb = numeric(b);
  if (typeof na === "number" && typeof nb === "number") {
    if (isInt(na) && isInt(nb)) return na === nb;
    if (Number.isNaN(na) && Number.isNaN(nb)) return true;
    return Math.abs(na - nb) <= ATOL + RTOL * Math.abs(nb);
  }
  return a === b;
}

function show(value) {
  if (value === undefined) return "undefined";
  try {
    const text = JSON.stringify(value, (_k, v) => (v === undefined ? null : v));
    return text === undefined ? String(value) : text;
  } catch (_) {
    return util.inspect(value, { depth: 3, breakLength: Infinity });
  }
}

function errorText(err) {
  if (err && typeof err === "object" && "name" in err && "message" in err) {
    if (err.name === "SyntaxError" && typeof err.stack === "string") {
      const where = err.stack.split("\n")[0];
      return `SyntaxError: ${err.message}${/solution\.js:\d+/.test(where) ? ` (${where.trim()})` : ""}`;
    }
    return `${err.name}: ${err.message}`;
  }
  return `Error: ${String(err)}`;
}

async function settle(value) {
  if (!value || typeof value.then !== "function") return value;
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`timed out after ${CASE_TIMEOUT_MS} ms`)), CASE_TIMEOUT_MS);
  });
  try { return await Promise.race([value, timeout]); } finally { clearTimeout(timer); }
}

async function main() {
  const context = vm.createContext({ console: sandboxConsole });
  vm.runInContext(PRELUDE, context, { filename: "prelude.js" });
  /* The grading helpers are read-only to the learner: `__isSameTree = () =>
     true` would otherwise pass every tree case. ListNode/TreeNode stay
     writable — pasting LeetCode's own `function ListNode(...)` is normal. */
  vm.runInContext(
    `for (const k of Object.getOwnPropertyNames(globalThis)) if (k.startsWith("__")) Object.defineProperty(globalThis, k, { value: globalThis[k], writable: false });`,
    context,
    { filename: "prelude.js" },
  );
  try {
    vm.runInContext(payload.code, context, { filename: "solution.js", timeout: CASE_TIMEOUT_MS });
  } catch (err) {
    process.stderr.write(errorText(err) + "\n");
    process.exit(1);
  }
  const results = [];
  /* A case that hits the clock is almost always an infinite loop that every
     later case would hit too; running them anyway spends 5 s each and the
     whole run dies at the server's timeout with no rows at all. */
  let timedOut = false;
  for (const tc of payload.cases || []) {
    const expected = JSON.parse(tc.expected);
    if (timedOut) {
      results.push({ passed: false, actual: "", expected: show(expected), error: "Not run: an earlier case timed out.", note: "" });
      continue;
    }
    try {
      const actual = await settle(vm.runInContext(tc.call, context, { filename: "test.js", timeout: CASE_TIMEOUT_MS }));
      results.push({ passed: equal(actual, expected), actual: show(actual), expected: show(expected), error: "", note: "" });
    } catch (err) {
      if (err && err.code === "ERR_SCRIPT_EXECUTION_TIMEOUT" || /timed out after/.test(String(err && err.message))) timedOut = true;
      results.push({ passed: false, actual: "", expected: show(expected), error: errorText(err), note: "" });
    }
  }
  process.stdout.write("\n__DELTA_TESTS__" + JSON.stringify(results) + "\n");
}

main().catch((err) => {
  process.stderr.write(errorText(err) + "\n");
  process.exit(1);
});
