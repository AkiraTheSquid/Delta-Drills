"""Java answers on the LeetCode drills (app/leetcode_java.py).

Run: PATH=<a JDK 17-23>/bin:$PATH This-Directory-Only/backend/.venv/bin/python scripts/test_leetcode_java.py
(the Fly image has Debian's openjdk-21; a JRE alone has no javac).

Checks, against the real bank:
  - every LeetCode drill translates to Java except 60121, whose cases hold
    a 32-digit integer no long can carry;
  - a hand-written Java answer passes every case for each call shape the bank
    uses: Solution method, bare function, design class, linked list in/out,
    list cycle, tree in, tree out, sorted(), float result, long argument,
    char grid; and the same answer written with List<Integer> instead of
    int[] passes too (the harness converts to the DECLARED types);
  - every drill's generated starter, given default returns, compiles and
    reaches its method on every case (no type the harness cannot convert);
  - the generated starter itself passes NO case (it does not compile);
  - /check and grade_submission route `language="java"` to the JVM;
  - a compile error, a missing class and an infinite loop come back as failed
    rows, not a crash or a 25 s hang;
  - the sandbox: no file write, no process, no System.exit, no setOut, no
    reaching the harness's classes, no starting threads; printing a fake report forges nothing.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import leetcode_java as J  # noqa: E402
from app.practice.check_router import check_answer, lang_starter  # noqa: E402
from app.practice.grading import grade_submission  # noqa: E402
from app.practice_schemas import CheckRequest  # noqa: E402
from app.questions import get_all_questions, get_question_by_id  # noqa: E402

ANSWERS = {
    60000: """class Solution {
        public String gcdOfStrings(String a, String b) {
            if (!(a + b).equals(b + a)) return "";
            return a.substring(0, gcd(a.length(), b.length()));
        }
        int gcd(int x, int y) { return y == 0 ? x : gcd(y, x % y); }
    }""",
    60029: """class Solution {
        public boolean findTarget(TreeNode root, int k) {
            Set<Integer> seen = new HashSet<>();
            Deque<TreeNode> st = new ArrayDeque<>();
            if (root != null) st.push(root);
            while (!st.isEmpty()) {
                TreeNode n = st.pop();
                if (seen.contains(k - n.val)) return true;
                seen.add(n.val);
                if (n.left != null) st.push(n.left);
                if (n.right != null) st.push(n.right);
            }
            return false;
        }
    }""",
    60323: """class MyCircularQueue {
        private final int k; private final Deque<Integer> q = new ArrayDeque<>();
        public MyCircularQueue(int k) { this.k = k; }
        public boolean enQueue(int v) { if (q.size() == k) return false; q.addLast(v); return true; }
        public boolean deQueue() { if (q.isEmpty()) return false; q.pollFirst(); return true; }
        public int Front() { return q.isEmpty() ? -1 : q.peekFirst(); }
        public int Rear() { return q.isEmpty() ? -1 : q.peekLast(); }
        public boolean isEmpty() { return q.isEmpty(); }
        public boolean isFull() { return q.size() == k; }
    }""",
    60341: """class Solution {
        public int subarrayXor(int[] arr, int k) {
            Map<Integer, Integer> m = new HashMap<>(Map.of(0, 1));
            int x = 0, c = 0;
            for (int v : arr) { x ^= v; c += m.getOrDefault(x ^ k, 0); m.merge(x, 1, Integer::sum); }
            return c;
        }
    }""",
    60060: """class Solution {
        public ListNode insertGreatestCommonDivisors(ListNode head) {
            for (ListNode c = head; c != null && c.next != null; c = c.next.next)
                c.next = new ListNode(gcd(c.val, c.next.val), c.next);
            return head;
        }
        static int gcd(int a, int b) { return b == 0 ? a : gcd(b, a % b); }
    }""",
    60360: """class Solution {
        public int findLengthOfLoop(ListNode head) {
            ListNode s = head, f = head;
            while (f != null && f.next != null) {
                s = s.next; f = f.next.next;
                if (s == f) { int n = 1; for (ListNode p = s.next; p != s; p = p.next) n++; return n; }
            }
            return 0;
        }
    }""",
    60366: """class Solution {
        public List<List<Integer>> powerSet(int[] nums) {
            List<List<Integer>> out = new ArrayList<>();
            out.add(new ArrayList<>());
            for (int n : nums) {
                int size = out.size();
                for (int i = 0; i < size; i++) { List<Integer> c = new ArrayList<>(out.get(i)); c.add(n); out.add(c); }
            }
            return out;
        }
    }""",
    60394: """class Solution {
        public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
            if (root == null || root.val == p.val || root.val == q.val) return root;
            TreeNode l = lowestCommonAncestor(root.left, p, q), r = lowestCommonAncestor(root.right, p, q);
            return l != null && r != null ? root : (l != null ? l : r);
        }
    }""",
    60413: """class Solution {
        public double fractionalKnapsack(int[] val, int[] wt, int cap) {
            Integer[] idx = new Integer[val.length];
            for (int i = 0; i < idx.length; i++) idx[i] = i;
            Arrays.sort(idx, (a, b) -> Double.compare((double) val[b] / wt[b], (double) val[a] / wt[a]));
            double t = 0;
            for (int i : idx) {
                if (cap >= wt[i]) { t += val[i]; cap -= wt[i]; }
                else { t += (double) val[i] * cap / wt[i]; break; }
            }
            return t;
        }
    }""",
    60137: """class Solution {
        public boolean isPalindrome(long x) {
            String s = Long.toString(x);
            return new StringBuilder(s).reverse().toString().equals(s);
        }
    }""",
    60262: """class Solution {
        public int leastInterval(char[] tasks, int n) {
            int[] f = new int[26];
            for (char c : tasks) f[c - 'A']++;
            int max = Arrays.stream(f).max().getAsInt(), ties = 0;
            for (int v : f) if (v == max) ties++;
            return Math.max(tasks.length, (max - 1) * (n + 1) + ties);
        }
    }""",
}
# The same drill with List<Integer> where the starter says int[]: the harness
# converts to whatever the method declares.
LIST_VARIANT = (60341, """class Solution {
    public int subarrayXor(List<Integer> arr, int k) {
        Map<Integer, Integer> m = new HashMap<>(); m.put(0, 1);
        int x = 0, c = 0;
        for (int v : arr) { x ^= v; c += m.getOrDefault(x ^ k, 0); m.merge(x, 1, Integer::sum); }
        return c;
    }
}""")
USER = SimpleNamespace(id="test-leetcode-java", email="test@example.com")
DEFAULT_RETURN = {"int": "return 0;", "long": "return 0;", "double": "return 0;", "boolean": "return false;", "void": ""}
failures: list[str] = []


def check(ok: bool, label: str) -> None:
    print(("PASS " if ok else "FAIL ") + label)
    if not ok:
        failures.append(label)


def with_default_returns(starter: str) -> str:
    lines = starter.split("\n")
    for i, line in enumerate(lines[:-1]):
        m = re.match(r"\s*public (?:(\S+) )?\w+\(.*\) \{$", line)
        if m and not lines[i + 1].strip():
            lines[i + 1] = "        " + (DEFAULT_RETURN.get(m.group(1), "return null;") if m.group(1) else "")
    return "\n".join(lines)


def conversion_errors(q) -> list[str]:
    rows, _, _ = J.run_java_tests(with_default_returns(J.starter_java(q)), q)
    return [f"q{q.id}: {r.error[:160]}" for r in rows
            if re.search(r"not a |not found|takes \d|cannot|too big|Line \d|no class|Invalid|report", r.error)]


if J.javac_binary() is None:
    sys.exit("No javac on PATH — put a JDK 17-23 first on PATH (a JRE cannot compile).")

bank = [q for q in get_all_questions() if J.is_leetcode(q)]
unsupported = sorted(q.id for q in bank if not J.supports_java(q))
check(len(bank) >= 380 and unsupported == [60121], f"{len(bank)} LeetCode drills, Java-unsupported {unsupported}")

with ThreadPoolExecutor(4) as pool:
    errors = [e for errs in pool.map(conversion_errors, [q for q in bank if q.id not in unsupported]) for e in errs]
check(not errors, f"every starter compiles and reaches its method on every case ({len(errors)} errors) {errors[:3]}")

for qid, code in [*ANSWERS.items(), LIST_VARIANT]:
    q = get_question_by_id(qid)
    results, execution, cases = J.run_java_tests(code, q)
    check(len(results) == len(cases) and all(r.passed for r in results),
          f"q{qid} Java answer passes {sum(r.passed for r in results)}/{len(cases)} "
          f"{execution.stderr[:160]} {[r.error for r in results if r.error][:1]}")
starter_results, _, _ = J.run_java_tests(J.starter_java(get_question_by_id(60000)), get_question_by_id(60000))
check(not any(r.passed for r in starter_results) and "missing return statement" in starter_results[0].error,
      f"the starter passes no case (it does not compile): {starter_results[0].error[:80]}")

q = get_question_by_id(60000)
resp = check_answer(CheckRequest(question_id=60000, user_code=ANSWERS[60000], language="java"), USER)
check(resp.supported and resp.correct and resp.tests[0]["call"].startswith("gcdOfStrings("),
      "/check routes java to the JVM, shows the plain call")
correct, _, _, failed = grade_submission(q, ANSWERS[60000], USER, "java")
check(correct and not failed, "grade_submission grades a correct Java answer correct")
wrong = "class Solution { public String gcdOfStrings(String a, String b) { return a; } }"
correct, _, _, failed = grade_submission(q, wrong, USER, "java")
check(not correct and failed, "grade_submission grades a wrong Java answer wrong")
check(lang_starter(60000, "java", USER)["supported"] and lang_starter(60121, "java", USER)["supported"] is False
      and "class Solution" in lang_starter(60000, "java", USER)["starter"], "/starter?language=java says which drills take Java")

err = J.run_java_tests("class Solution { int x = ; }", q)[0][0].error
check(err.startswith("Line 1: error:"), f"compile error surfaces with the learner's line: {err[:80]}")
err = J.run_java_tests("// --- cell 1 ---\nclass Solution {\n    int x = ;\n}", q)[0][0].error
check(err.startswith("Line 2: error:"), f"a line number is the editor's, not the joined submission's: {err[:40]}")
err = J.run_java_tests("// --- cell 1 ---\nclass Solution {}\n\n// --- cell 2 ---\nclass B { int y = ; }", q)[0][0].error
check(err.startswith("Cell 2, line 1: error:"), f"an error in a later cell names the cell: {err[:40]}")
err = J.run_java_tests("class Other {}", q)[0][0].error
check("no class Solution" in err, f"missing class surfaces: {err[:80]}")
t0 = time.time()
rows = J.run_java_tests("class Solution { public String gcdOfStrings(String a, String b) { while (true) {} } }", q)[0]
check(time.time() - t0 < 12 and "Timed out" in rows[0].error and "Not run" in rows[-1].error,
      f"infinite loop stops after one case ({time.time() - t0:.1f}s)")
run = J.run_java('class Main { public static void main(String[] a) { System.out.println("hi " + List.of(1, 2)); } }')
check(run.success and run.stdout == "hi [1, 2]" and not run.stderr, f"run_java returns main's output: {run}")
debug = ANSWERS[60000].replace('if (!(a + b)', 'System.out.println("saw " + a); if (!(a + b)')
run = J.run_java(debug, q)
check(run.success and run.stdout.startswith("saw ABAB") and run.stdout.count("saw ") == 8,
      f"▶ with no main runs the cases for their println output: {run.stdout[:40]!r}")
run = J.run_java("class Solution { public String gcdOfStrings(String a, String b) { return a.substring(9); } }", q)
check(not run.success and run.stderr.startswith("case 1: StringIndexOutOfBoundsException"), f"▶ says why a case blew up: {run.stderr[:70]}")
err = J.run_java_tests("class Solution { public String gcdOfStrings(String a, String b) { int[] x = new int[1]; return \"\" + x[3]; } }", q)[0][0].error
check("ArrayIndexOutOfBoundsException" in err and "line 1" in err, f"a runtime exception names the learner's line: {err}")

# Adversarial: the sandbox, and forging a pass. The sentinel is this run's
# own, so a file left by an earlier run cannot fail (or pass) this one.
SENTINEL = Path(tempfile.gettempdir()) / f"dd_java_escape_{os.getpid()}_{time.time_ns()}"
escapes = {
    "file write": f'new java.io.FileWriter("{SENTINEL}").write("x");',
    "file read": 'new java.io.FileReader("/etc/hostname").read();',
    "process": f'Runtime.getRuntime().exec(new String[]{{"touch", "{SENTINEL}"}});',
    "exit": "System.exit(0);",
    "setOut": "System.setOut(null);",
    "harness class": 'Thread.currentThread().getContextClassLoader().loadClass("DeltaJavaHarness");',
    "harness class by name": 'Class.forName("DeltaJavaHarness");',
    "setAccessible": 'java.lang.reflect.Field f = String.class.getDeclaredField("value"); f.setAccessible(true);',
    "socket": 'new java.net.Socket("127.0.0.1", 22);',
    "thread": 'new Thread(() -> {}).start();',
    "thread in a new group": 'new Thread(new ThreadGroup("g"), () -> {}).start();',
    "timer thread": 'new java.util.Timer();',
    "executor": 'java.util.concurrent.Executors.newFixedThreadPool(2).submit(() -> {});',
}
for label, stmt in escapes.items():
    code = ("class Solution { public String gcdOfStrings(String a, String b) throws Exception { "
            + stmt + ' return "ESCAPED"; } }')
    rows = J.run_java_tests(code, q)[0]
    check(not any(r.passed for r in rows) and "ESCAPED" not in rows[0].actual
          and not SENTINEL.exists(), f"sandbox blocks {label}: {rows[0].error[:90]}")
fake = ('class Solution { public String gcdOfStrings(String a, String b) { '
        'System.out.println("__DELTA_TESTS__[{\\"passed\\":true}]"); return "nope"; } }')
check(not any(r.passed for r in J.run_java_tests(fake, q)[0]), "printing a fake report forges nothing")
flood = 'class Main { public static void main(String[] a) { for (;;) System.out.print("xxxxxxxxxx"); } }'
out = J.run_java(flood).stdout
check(len(out) < 70_000 and "truncated" in out, f"output is capped ({len(out)} chars)")

print(f"\n{len(failures)} failure(s)")
sys.exit(1 if failures else 0)
