"""JavaScript answers on the LeetCode drills (app/leetcode_js.py).

Run: This-Directory-Only/backend/.venv/bin/python scripts/test_leetcode_js.py

Checks, against the real bank:
  - every LeetCode drill translates to JS except the two whose cases hold an
    integer past 2^53 (60121 reverseBits, 60137 isPalindrome);
  - a hand-written JS answer passes every case for each call shape the bank
    uses: Solution method, `class Solution`, bare function, design class,
    linked list in/out, list cycle, tree in, tree out, sorted(), float result;
  - the generated JS starter passes NO case (a stub must not score);
  - /check and grade_submission route `language="javascript"` to node;
  - a syntax error, an undefined function and an infinite loop come back as
    failed rows, not a crash or a 20 s hang.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import leetcode_js as L  # noqa: E402
from app.practice.check_router import check_answer, js_starter  # noqa: E402
from app.practice.grading import grade_submission  # noqa: E402
from app.practice_schemas import CheckRequest  # noqa: E402
from app.questions import get_all_questions, get_question_by_id  # noqa: E402

ANSWERS = {
    60000: """var gcdOfStrings = function(a, b) {
        if (a + b !== b + a) return "";
        const g = (x, y) => y ? g(y, x % y) : x;
        return a.slice(0, g(a.length, b.length));
    };""",
    60029: """class Solution { findTarget(root, k) {
        const seen = new Set(), st = [root];
        while (st.length) { const n = st.pop(); if (!n) continue;
            if (seen.has(k - n.val)) return true; seen.add(n.val); st.push(n.left, n.right); }
        return false; } }""",
    60323: """class MyCircularQueue {
        constructor(k) { this.k = k; this.q = []; }
        enQueue(v) { if (this.q.length === this.k) return false; this.q.push(v); return true; }
        deQueue() { if (!this.q.length) return false; this.q.shift(); return true; }
        Front() { return this.q.length ? this.q[0] : -1; }
        Rear() { return this.q.length ? this.q[this.q.length - 1] : -1; }
        isEmpty() { return this.q.length === 0; }
        isFull() { return this.q.length === this.k; } }""",
    60341: """function subarrayXor(arr, k) { const m = new Map([[0, 1]]); let x = 0, c = 0;
        for (const v of arr) { x ^= v; c += m.get(x ^ k) || 0; m.set(x, (m.get(x) || 0) + 1); } return c; }""",
    60060: """var insertGreatestCommonDivisors = function(head) {
        const g = (a, b) => b ? g(b, a % b) : a; let c = head;
        while (c && c.next) { c.next = new ListNode(g(c.val, c.next.val), c.next); c = c.next.next; }
        return head; };""",
    60360: """function findLengthOfLoop(head) { let s = head, f = head;
        while (f && f.next) { s = s.next; f = f.next.next;
            if (s === f) { let n = 1, p = s.next; while (p !== s) { p = p.next; n++; } return n; } }
        return 0; }""",
    60366: """function powerSet(nums) { const out = [[]];
        for (const n of nums) { const L = out.length; for (let i = 0; i < L; i++) out.push([...out[i], n]); }
        return out; }""",
    60394: """function lowestCommonAncestor(root, p, q) {
        if (!root || root.val === p.val || root.val === q.val) return root;
        const l = lowestCommonAncestor(root.left, p, q), r = lowestCommonAncestor(root.right, p, q);
        return l && r ? root : l || r; }""",
    60413: """function fractionalKnapsack(val, wt, cap) {
        const it = val.map((v, i) => [v, wt[i]]).sort((a, b) => b[0] / b[1] - a[0] / a[1]);
        let t = 0; for (const [v, w] of it) { if (cap >= w) { t += v; cap -= w; } else { t += v * cap / w; break; } }
        return t; }""",
}
USER = SimpleNamespace(id="test-leetcode-js", email="test@example.com")
failures: list[str] = []


def check(ok: bool, label: str) -> None:
    print(("PASS " if ok else "FAIL ") + label)
    if not ok:
        failures.append(label)


get_all_questions()
bank = [q for q in get_all_questions() if L.is_leetcode(q)]
unsupported = sorted(q.id for q in bank if not L.supports_js(q))
check(len(bank) >= 380 and unsupported == [60121, 60137], f"{len(bank)} LeetCode drills, JS-unsupported {unsupported}")

for qid, code in ANSWERS.items():
    q = get_question_by_id(qid)
    results, execution, cases = L.run_js_tests(code, q)
    check(len(results) == len(cases) and all(r.passed for r in results),
          f"q{qid} JS answer passes {sum(r.passed for r in results)}/{len(cases)} {execution.stderr[:120]}")
    starter_results, _, _ = L.run_js_tests(L.starter_js(q), q)
    check(not any(r.passed for r in starter_results), f"q{qid} JS starter passes no case")

q = get_question_by_id(60000)
resp = check_answer(CheckRequest(question_id=60000, user_code=ANSWERS[60000], language="javascript"), USER)
check(resp.supported and resp.correct and resp.tests[0]["call"].startswith("gcdOfStrings("), "/check routes javascript to node, shows the plain call")
correct, _, _, failed = grade_submission(q, ANSWERS[60000], USER, "javascript")
check(correct and not failed, "grade_submission grades a correct JS answer correct")
correct, _, _, failed = grade_submission(q, "var gcdOfStrings = function(a, b) { return a; };", USER, "javascript")
check(not correct and failed, "grade_submission grades a wrong JS answer wrong")
check(js_starter(60000, USER)["supported"] and js_starter(60121, USER)["supported"] is False, "/js-starter says which drills take JS")

err = L.run_js_tests("var x = ;", q)[0][0].error
check(err.startswith("SyntaxError"), f"syntax error surfaces: {err}")
err = L.run_js_tests("var y = 1;", q)[0][0].error
check("gcdOfStrings is not defined" in err, f"missing function surfaces: {err}")
t0 = time.time()
rows = L.run_js_tests("var gcdOfStrings = function() { while (true) {} };", q)[0]
check(time.time() - t0 < 10 and "timed out" in rows[0].error and "Not run" in rows[-1].error,
      f"infinite loop stops after one case ({time.time() - t0:.1f}s)")
check(L.run_js("console.log('hi', [1, 2])").stdout == "hi [ 1, 2 ]", "run_js returns console output")

# Adversarial: forging a pass, escaping to the host, crashing after the rows.
check(L.run_js("var __isSameTree = () => 'forged'; console.log(__isSameTree(null, new TreeNode(1)))").stdout == "false",
      "the grading helpers are read-only to learner code")
forged = "Array.prototype.every = () => true; " \
         "function powerSet(nums) { return Array.from({length: 2 ** nums.length}, () => [-99]); }"
rows = L.run_js_tests(forged, get_question_by_id(60366))[0]
check(not any(r.passed for r in rows), "replacing Array.prototype.every forges no pass (right length, wrong items)")
escape = "const p = console.log.constructor('return process')(); " \
         "try { p.mainModule.require('fs').writeFileSync('/tmp/dd_js_escape', 'x'); console.log('WROTE'); } " \
         "catch (e) { console.log(e.code || e.message); }"
out = L.run_js(escape).stdout
check(L._permission_flag(L.node_binary()) is not None and "WROTE" not in out and not Path("/tmp/dd_js_escape").exists(),
      f"host fs write from learner code is denied: {out[:80]}")
late = ANSWERS[60000] + "\nsetTimeout(() => Promise.reject(new Error('late')), 0);"
rows = L.run_js_tests(late, q)[0]
check(not all(r.passed for r in rows), "a process that crashes after the rows does not grade correct")

fake = "console.log.constructor('return process')().stdout.write('\\n__DELTA_TESTS__[]\\n'); " \
       "console.log.constructor('return process')().exit(0);"
rows = L.run_js_tests(fake, q)[0]
check(not all(r.passed for r in rows), "an empty forged report does not grade correct")
try:
    L._literal(__import__("ast").parse("{1, 2}", mode="eval").body)
    check(False, "a set literal is refused")
except L.Untranslatable:
    check(True, "a set literal is refused as Untranslatable, not a 500")

print(f"\n{len(failures)} failure(s)")
sys.exit(1 if failures else 0)
