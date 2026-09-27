"""JavaScript as a second answer language for the LeetCode Patterns drills.

The bank (lessons/leetcode/problems.json) is authored in Python: a Python
starter, Python test calls (`Solution().twoSum(nums=[2, 7], target=9)`) and
Python-literal expected values. Nothing is re-authored for JavaScript. Instead
this module derives, from the SAME rows:

  starter_js(question)   the LeetCode-style JS stub for the Python signature
                         (`var twoSum = function(nums, target) {}`, a bare
                         `function`, or a `class` for a design problem)
  js_cases(question)     every test call translated to a JS expression, plus
                         the expected value as JSON (ast.literal_eval of the
                         Python literal, so both languages grade against the
                         identical answer)
  run_js_tests(...)      the learner's JS + those cases through `node`
                         (leetcode_js_harness.js), returning the same
                         TestCaseResult rows run_function_tests does, so
                         /check and /submit treat the two languages alike.

A question whose calls use anything the translator does not know (an unknown
helper, an int past 2^53 that a JS number cannot hold) is NOT supported in JS:
`js_cases` raises Untranslatable and the endpoints say so rather than grading
against a mangled test.

Trust: `vm` is not a boundary; the node PROCESS is, fenced by node's permission
model (no fs writes, no child_process, reads only its own temp dir, scrubbed
env) — tighter than the Python runner, which has no fence. As in Python, a
learner who escapes the vm can still forge their OWN grade (patch the host's
stdout); only an OS sandbox would stop that. An async answer that loops after
an `await` starves the per-case timer and ends at the 20 s process timeout:
graded wrong, never right.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

from app.code_runner import ExecutionResult, TestCaseResult

logger = logging.getLogger(__name__)

LANGUAGE = "javascript"
HARNESS_PATH = Path(__file__).with_name("leetcode_js_harness.js")
LEETCODE_ID_FLOOR = 60000
MAX_SAFE_INT = 2**53 - 1
TIMEOUT_SECONDS = 20
CASE_TIMEOUT_MS = 4000
# node needs a PATH and a HOME and nothing else; no secret reaches the child.
_ENV_KEYS = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "TEMP", "TMP")

# Python fixture helpers the bank's setup_code defines -> the harness's JS twins.
HELPERS = {
    "list_node": "__listNode",
    "list_node_cycle": "__listNodeCycle",
    "tree_node": "__treeNode",
    "_from_list_node": "__fromListNode",
    "_from_tree_node": "__fromTreeNode",
    "is_same_list": "__isSameList",
    "is_same_tree": "__isSameTree",
    "sorted": "__sorted",
}
JS_RESERVED = {
    "arguments", "await", "break", "case", "catch", "class", "const", "continue", "debugger",
    "default", "delete", "do", "else", "enum", "eval", "export", "extends", "false", "finally",
    "for", "function", "if", "implements", "import", "in", "instanceof", "interface", "let",
    "new", "null", "package", "private", "protected", "public", "return", "static", "super",
    "switch", "this", "throw", "true", "try", "typeof", "var", "void", "while", "with", "yield",
}


class Untranslatable(ValueError):
    """The question cannot be graded in JavaScript."""


def is_leetcode(question) -> bool:
    return getattr(question, "id", 0) >= LEETCODE_ID_FLOOR and getattr(question, "topic", "") == "LeetCode"


def node_binary() -> str | None:
    return shutil.which("node") or shutil.which("nodejs")


@lru_cache(maxsize=1)
def _permission_flag(node: str) -> str | None:
    """node's permission model: no fs writes, no child_process, no workers.

    `vm` is not a boundary (learner code can reach the host `process`), so the
    PROCESS is the boundary, as it is for the Python runner — this makes it a
    tighter one. node 22+ spells it `--permission`, node 20 (Debian's, the Fly
    image) `--experimental-permission`, and each rejects the other's spelling,
    so ask the binary. None = neither works, and _run_node then refuses to run:
    learner code never runs on a node that cannot be fenced."""
    for flag in ("--permission", "--experimental-permission"):
        try:
            ok = subprocess.run([node, flag, "-e", "0"], capture_output=True, timeout=10).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            ok = False
        if ok:
            return flag
    logger.warning("node %s has no permission model; JavaScript answers are refused", node)
    return None


def _js_name(name: str) -> str:
    return f"{name}_" if name in JS_RESERVED else name


# --- signature ---------------------------------------------------------------

def _params(fn: ast.FunctionDef, drop_self: bool) -> list[tuple[str, str]]:
    args = fn.args.args[1:] if drop_self else fn.args.args
    return [(a.arg, ast.unparse(a.annotation) if a.annotation else "") for a in args]


@lru_cache(maxsize=1024)
def _signature(starter: str, function_name: str) -> dict:
    """What the Python starter asks for: a `Solution` method, a bare function,
    or a design class, with its parameters in declaration order (the order a
    JS call passes them, since JS has no keyword arguments)."""
    try:
        tree = ast.parse(starter)
    except SyntaxError as exc:
        raise Untranslatable(f"starter does not parse: {exc}") from exc
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    uses = {"ListNode": "ListNode" in classes, "TreeNode": "TreeNode" in classes}
    solution = classes.get("Solution")
    if solution is not None:
        methods = {n.name: n for n in solution.body if isinstance(n, ast.FunctionDef)}
        if function_name in methods:
            fn = methods[function_name]
            return {"kind": "solution", "name": function_name, "params": _params(fn, True),
                    "returns": ast.unparse(fn.returns) if fn.returns else "", "uses": uses}
    if function_name in funcs:
        fn = funcs[function_name]
        return {"kind": "function", "name": function_name, "params": _params(fn, False),
                "returns": ast.unparse(fn.returns) if fn.returns else "", "uses": uses}
    if function_name in classes:
        cls = classes[function_name]
        methods = []
        for n in cls.body:
            if isinstance(n, ast.FunctionDef):
                methods.append({"name": "constructor" if n.name == "__init__" else n.name,
                                "params": _params(n, True)})
        return {"kind": "design", "name": function_name, "methods": methods, "uses": uses}
    raise Untranslatable(f"no `{function_name}` in the starter")


# --- starter -----------------------------------------------------------------

_TYPE_MAP = {"int": "number", "float": "number", "str": "string", "bool": "boolean", "None": "void"}


def _js_type(annotation: str) -> str:
    """`List[List[int]]` -> `number[][]`, `Optional[TreeNode]` -> `TreeNode`.
    Only for the JSDoc comment; nothing is checked against it."""
    t = annotation.replace("typing.", "").strip()
    if not t:
        return "*"
    if t.startswith("Optional[") and t.endswith("]"):
        return _js_type(t[9:-1])
    if t.lower().startswith("list[") and t.endswith("]"):
        return _js_type(t[5:-1]) + "[]"
    return _TYPE_MAP.get(t, t)


_NODE_DOCS = {
    "ListNode": (
        "/**\n * Definition for singly-linked list (provided by the grader).\n"
        " * function ListNode(val, next) {\n"
        " *     this.val = (val===undefined ? 0 : val)\n"
        " *     this.next = (next===undefined ? null : next)\n"
        " * }\n */\n"
    ),
    "TreeNode": (
        "/**\n * Definition for a binary tree node (provided by the grader).\n"
        " * function TreeNode(val, left, right) {\n"
        " *     this.val = (val===undefined ? 0 : val)\n"
        " *     this.left = (left===undefined ? null : left)\n"
        " *     this.right = (right===undefined ? null : right)\n"
        " * }\n */\n"
    ),
}


def starter_js(question) -> str:
    sig = _signature(question.starter_code or "", question.function_name or "")
    head = "".join(doc for name, doc in _NODE_DOCS.items() if sig["uses"][name])
    if sig["kind"] == "design":
        body = []
        for m in sig["methods"]:
            args = ", ".join(_js_name(p) for p, _ in m["params"])
            body.append(f"    {m['name']}({args}) {{\n        \n    }}")
        return f"{head}class {sig['name']} {{\n" + "\n\n".join(body) + "\n}\n"
    args = ", ".join(_js_name(p) for p, _ in sig["params"])
    # JSDoc only when the Python starter was typed (the LeetCode rows are;
    # the Striver-style bare functions are not, and `{*}` says nothing).
    if any(t for _, t in sig["params"]) or sig["returns"]:
        doc = ["/**"]
        doc += [f" * @param {{{_js_type(t)}}} {_js_name(p)}" for p, t in sig["params"]]
        doc += [f" * @return {{{_js_type(sig['returns'])}}}", " */"]
        head += "\n".join(doc) + "\n"
    if sig["kind"] == "solution":
        return f"{head}var {sig['name']} = function({args}) {{\n    \n}};\n"
    return f"{head}function {sig['name']}({args}) {{\n    \n}}\n"


# --- test translation ---------------------------------------------------------

def _literal(node: ast.AST) -> str:
    value = ast.literal_eval(node)
    _check_ints(value)
    try:
        return json.dumps(value)
    except TypeError as exc:  # a set / bytes literal has no JSON form
        raise Untranslatable(f"{type(value).__name__} values are not translated") from exc


def _check_ints(value) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, int) and abs(value) > MAX_SAFE_INT:
        raise Untranslatable(f"integer {value} does not fit a JS number")
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        raise Untranslatable("inf/nan do not survive JSON")
    if isinstance(value, (list, tuple, set)):
        for v in value:
            _check_ints(v)
    if isinstance(value, dict):
        raise Untranslatable("dict values are not translated")


def _ordered_args(call: ast.Call, params: list[tuple[str, str]]) -> list[ast.AST]:
    """Positional args, then keywords placed by the Python signature's order."""
    names = [p for p, _ in params]
    out: list[ast.AST | None] = list(call.args) + [None] * max(0, len(names) - len(call.args))
    for kw in call.keywords:
        if kw.arg is None or kw.arg not in names:
            raise Untranslatable(f"keyword {kw.arg!r} is not a parameter")
        out[names.index(kw.arg)] = kw.value
    while out and out[-1] is None:
        out.pop()
    if any(v is None for v in out):
        raise Untranslatable("a middle argument is missing")
    return out  # type: ignore[return-value]


def _expr(node: ast.AST, sig: dict) -> str:
    try:
        return _literal(node)
    except (ValueError, TypeError, SyntaxError):
        pass
    if not isinstance(node, ast.Call):
        raise Untranslatable(f"cannot translate {ast.unparse(node)!r}")
    func = node.func
    # Solution().method(...)
    if (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Call)
            and isinstance(func.value.func, ast.Name) and func.value.func.id == "Solution"):
        if sig["kind"] != "solution" or func.attr != sig["name"]:
            raise Untranslatable(f"Solution().{func.attr} is not the drill's method")
        args = ", ".join(_expr(a, sig) for a in _ordered_args(node, sig["params"]))
        # The learner may write LeetCode's `var fn = function` or a
        # `class Solution { fn() {} }`; __solution takes whichever exists.
        return (f"__solution({json.dumps(func.attr)}, typeof {func.attr} === 'undefined' ? undefined : {func.attr}, "
                f"typeof Solution === 'undefined' ? undefined : Solution)({args})")
    if not isinstance(func, ast.Name):
        raise Untranslatable(f"cannot translate call {ast.unparse(node)!r}")
    if node.keywords and func.id != sig.get("name"):
        raise Untranslatable(f"keywords on helper {func.id}")
    if func.id == "_lc_design":
        if sig["kind"] != "design" or len(node.args) != 2:
            raise Untranslatable("_lc_design on a non-design drill")
        return f"__design({sig['name']}, {_literal(node.args[0])}, {_literal(node.args[1])})"
    if func.id in HELPERS:
        return f"{HELPERS[func.id]}(" + ", ".join(_expr(a, sig) for a in node.args) + ")"
    if sig["kind"] == "function" and func.id == sig["name"]:
        return f"{func.id}(" + ", ".join(_expr(a, sig) for a in _ordered_args(node, sig["params"])) + ")"
    raise Untranslatable(f"unknown call {func.id}")


def js_cases(question) -> list[dict]:
    """Every test case as {call, expected}: a JS expression and the expected
    value as JSON text. Raises Untranslatable if any case cannot be carried."""
    if not is_leetcode(question) or question.submission_mode != "function" or not question.test_cases:
        raise Untranslatable("only LeetCode function drills run in JavaScript")
    return _js_cases_cached(question.id, question.starter_code or "", question.function_name or "",
                            json.dumps(question.test_cases, sort_keys=True))


@lru_cache(maxsize=1024)
def _js_cases_cached(_qid: int, starter: str, function_name: str, cases_json: str) -> list[dict]:
    sig = _signature(starter, function_name)
    out = []
    for case in json.loads(cases_json):
        if case.get("assert_code") or case.get("expected_setup_code"):
            raise Untranslatable("assert_code / expected_setup_code are Python-only")
        try:
            call = ast.parse(case["call"], mode="eval").body
            expected = ast.parse(case["expected_expr"], mode="eval").body
        except SyntaxError as exc:
            raise Untranslatable(str(exc)) from exc
        js = _expr(call, sig)
        out.append({"call": js, "display": _display(js), "expected": _literal(expected)})
    return out


_SOLUTION_CALL = re.compile(
    r'__solution\("(\w+)", typeof \w+ === \'undefined\' \? undefined : \w+, '
    r"typeof Solution === 'undefined' \? undefined : Solution\)\("
)


def _display(js: str) -> str:
    """The call as the learner would write it: `gcdOfStrings("AB", "A")`,
    `treeNode([1, 2])` — not the harness's dispatch wrapper."""
    return _SOLUTION_CALL.sub(r"\1(", js).replace("__", "")


def supports_js(question) -> bool:
    try:
        js_cases(question)
        starter_js(question)
        return True
    except Untranslatable:
        return False


# --- execution ------------------------------------------------------------------

def _run_node(payload: dict, timeout: int) -> ExecutionResult:
    node = node_binary()
    if node is None:
        return ExecutionResult(stdout="", stderr="JavaScript runtime (node) is not installed on the server.", success=False)
    with tempfile.TemporaryDirectory(prefix="practice_js_") as tmp:
        payload_path = os.path.join(tmp, "payload.json")
        with open(payload_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        # The harness is copied next to the payload so ONE read path covers both
        # (node 20 and 24 disagree on how several --allow-fs-read paths are given).
        harness = shutil.copy(HARNESS_PATH, os.path.join(tmp, "harness.js"))
        flag = _permission_flag(node)
        if flag is None:
            return ExecutionResult(stdout="", stderr="This server's node (needs 20+) cannot sandbox JavaScript.", success=False)
        try:
            # --no-warnings: node 20 prints an ExperimentalWarning for the flag
            # on stderr, which the cell's ▶ would show the learner on every run.
            proc = subprocess.run(
                [node, flag, f"--allow-fs-read={tmp}", "--no-warnings", "--max-old-space-size=256", harness, payload_path],
                capture_output=True, text=True, timeout=timeout, cwd=tmp, env={k: os.environ[k] for k in _ENV_KEYS if k in os.environ},
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(stdout="", stderr=f"Execution timed out after {timeout} seconds", success=False)
        return ExecutionResult(stdout=proc.stdout, stderr=proc.stderr, success=proc.returncode == 0)


def run_js(user_code: str, timeout: int = TIMEOUT_SECONDS) -> ExecutionResult:
    """Run the learner's JS on its own (the cell's ▶): console output only."""
    result = _run_node({"code": user_code, "cases": [], "caseTimeoutMs": CASE_TIMEOUT_MS}, timeout)
    result.stdout = result.stdout.partition("__DELTA_TESTS__")[0].rstrip()
    return result


def run_js_tests(user_code: str, question, timeout: int = TIMEOUT_SECONDS) -> tuple[list[TestCaseResult], ExecutionResult, list[dict]]:
    """Grade JS against the question's cases. Returns the per-case results, the
    run, and the translated cases (their `call` is what the ▶ block shows)."""
    cases = js_cases(question)
    execution = _run_node({"code": user_code, "cases": cases, "caseTimeoutMs": CASE_TIMEOUT_MS}, timeout)
    marker = "__DELTA_TESTS__"
    if marker in execution.stdout:
        prefix, _, suffix = execution.stdout.partition(marker)
        execution.stdout = prefix.rstrip()
        try:
            rows = [TestCaseResult(**row) for row in json.loads(suffix.strip())]
            if not cases or len(rows) != len(cases):
                # all([]) is True: a short or empty report must never read as a pass.
                rows.append(TestCaseResult(passed=False, actual="", expected="",
                                           error=f"Test report had {len(rows)} rows for {len(cases)} cases."))
            if not execution.success:
                # The rows were written, then the process died (a stray promise
                # rejecting after the last case): a crashed run is not a pass.
                crash = execution.stderr.strip() or "JavaScript process exited with an error."
                rows.append(TestCaseResult(passed=False, actual="", expected="", error=crash))
            return rows, execution, cases
        except Exception as exc:  # noqa: BLE001 — a malformed payload is a failed grade
            return [TestCaseResult(passed=False, actual="", expected="", error=f"Invalid test payload: {exc}")], execution, cases
    error = execution.stderr.strip() or execution.stdout.strip() or "JavaScript test harness failed."
    return [TestCaseResult(passed=False, actual="", expected="", error=error)], execution, cases
