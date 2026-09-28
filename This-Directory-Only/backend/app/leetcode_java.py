"""Java as an answer language for the LeetCode Patterns drills.

The bank (lessons/leetcode/problems.json) is authored in Python. As with
JavaScript (app/leetcode_js.py), nothing is re-authored: this module derives,
from the SAME rows,

  starter_java(question)  the LeetCode-style `class Solution { public ... }`
                          stub (or the design class). Java needs types, so
                          they come from the Python annotations where the row
                          has them (`List[int]` -> `int[]`) and are inferred
                          from the test values where it does not (the
                          Striver-style bare functions), widening int to long
                          when a test value needs it
  java_cases(question)    every test call as a small expression tree the
                          harness evaluates ({"call": ..., "args": [...]},
                          list_node / tree_node / sorted helpers, the design
                          driver), plus the expected value as JSON of the
                          Python literal: both languages grade against the
                          identical answer
  run_java_tests(...)     the learner's Java + those cases through the JVM
                          (leetcode_java_harness.java), returning the same
                          TestCaseResult rows run_function_tests does.

The harness converts each argument to whatever type the learner's method
DECLARES, so the starter's choice of `int[]` vs `List<Integer>` is a
suggestion, not a contract: either compiles and grades.

Trust: the learner's classes get a ProtectionDomain with no permissions under
a SecurityManager (no files, sockets, processes, System.exit or setOut), and
the rows travel through a report file the learner cannot write, so printing a
fake report forges nothing. SecurityManager exists through JDK 23; on 24+ the
harness refuses to run rather than run unfenced. Pin the image's JDK
(Dockerfile: openjdk-21).
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
from functools import lru_cache
from pathlib import Path

from app.code_runner import ExecutionResult, TestCaseResult
from app.leetcode_js import ENV_KEYS, Untranslatable, is_leetcode, ordered_args, signature

logger = logging.getLogger(__name__)

LANGUAGE = "java"
HARNESS_PATH = Path(__file__).with_name("leetcode_java_harness.java")
HARNESS_CLASS = "DeltaJavaHarness"
INT_MIN, INT_MAX = -2**31, 2**31 - 1
LONG_MIN, LONG_MAX = -2**63, 2**63 - 1
TIMEOUT_SECONDS = 25
CASE_TIMEOUT_MS = 4000
EXIT_NO_SANDBOX = 3
# Each JVM holds ~300 MB at its limits; more learners than this at once wait
# their turn rather than take the machine's memory down with them.
MAX_CONCURRENT_JVMS = 3
_jvm_slots = threading.BoundedSemaphore(MAX_CONCURRENT_JVMS)
JVM_FLAGS = [
    "-Djava.security.manager=allow",  # JDK 18-23 need it to allow setSecurityManager
    "-Xmx256m", "-XX:+UseSerialGC", "-XX:TieredStopAtLevel=1", "-XX:-UsePerfData",
    "-XX:ActiveProcessorCount=1", "-Xshare:auto",
    # -Xmx bounds only the heap; cap the other native pools an answer can grow.
    "-XX:MaxDirectMemorySize=32m", "-XX:MaxMetaspaceSize=128m", "-XX:ReservedCodeCacheSize=32m",
    "-Dfile.encoding=UTF-8", "-Dstdout.encoding=UTF-8", "-Dstderr.encoding=UTF-8",
]

# The bank's Python fixture helpers the harness implements (leetcode_js.HELPERS'
# keys); the display names are what the ▶ block shows.
HELPERS = {
    "list_node": "listNode",
    "list_node_cycle": "listNodeCycle",
    "tree_node": "treeNode",
    "_from_list_node": "fromListNode",
    "_from_tree_node": "fromTreeNode",
    "is_same_list": "isSameList",
    "is_same_tree": "isSameTree",
    "sorted": "sorted",
}
NODE_BUILDERS = {"list_node": "ListNode", "list_node_cycle": "ListNode", "tree_node": "TreeNode"}
JAVA_RESERVED = {
    "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char", "class", "const",
    "continue", "default", "do", "double", "else", "enum", "extends", "false", "final", "finally",
    "float", "for", "goto", "if", "implements", "import", "instanceof", "int", "interface", "long",
    "native", "new", "null", "package", "private", "protected", "public", "return", "short", "static",
    "strictfp", "super", "switch", "synchronized", "this", "throw", "throws", "transient", "true", "try",
    "void", "volatile", "while", "var", "record", "yield", "_",
}


def java_binary() -> str | None:
    return shutil.which("java")


def javac_binary() -> str | None:
    return shutil.which("javac")


def _java_name(name: str) -> str:
    return f"{name}_" if name in JAVA_RESERVED else name


# --- literals -------------------------------------------------------------------

def _value(node: ast.AST):
    """The Python literal as a JSON-able value (tuples become lists)."""
    return _jsonable(ast.literal_eval(node))


def _jsonable(value):
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        if not LONG_MIN <= value <= LONG_MAX:
            raise Untranslatable(f"integer {value} does not fit a Java long")
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise Untranslatable("inf/nan do not survive JSON")
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    raise Untranslatable(f"{type(value).__name__} values are not translated")


def _is_literal(node: ast.AST) -> bool:
    try:
        ast.literal_eval(node)
        return True
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return False


# --- test translation -------------------------------------------------------------

def _expr(node: ast.AST, sig: dict) -> dict:
    if _is_literal(node):
        return {"v": _value(node)}
    if not isinstance(node, ast.Call):
        raise Untranslatable(f"cannot translate {ast.unparse(node)!r}")
    func = node.func
    if (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Call)
            and isinstance(func.value.func, ast.Name) and func.value.func.id == "Solution"):
        if sig["kind"] != "solution" or func.attr != sig["name"]:
            raise Untranslatable(f"Solution().{func.attr} is not the drill's method")
        return {"call": func.attr, "args": [_expr(a, sig) for a in ordered_args(node, sig["params"])]}
    if not isinstance(func, ast.Name):
        raise Untranslatable(f"cannot translate call {ast.unparse(node)!r}")
    if node.keywords and func.id != sig.get("name"):
        raise Untranslatable(f"keywords on helper {func.id}")
    if func.id == "_lc_design":
        if sig["kind"] != "design" or len(node.args) != 2:
            raise Untranslatable("_lc_design on a non-design drill")
        return {"design": sig["name"], "ops": _value(node.args[0]), "args": _value(node.args[1])}
    if func.id in HELPERS:
        return {"h": func.id, "args": [_expr(a, sig) for a in node.args]}
    if sig["kind"] == "function" and func.id == sig["name"]:
        return {"call": func.id, "args": [_expr(a, sig) for a in ordered_args(node, sig["params"])]}
    raise Untranslatable(f"unknown call {func.id}")


def _display(expr: dict) -> str:
    """The call as the learner would read it: `twoSum([2,7,11,15], 9)`."""
    if "v" in expr:
        return json.dumps(expr["v"], separators=(",", ":"))
    if "call" in expr:
        return f"{expr['call']}(" + ", ".join(_display(a) for a in expr["args"]) + ")"
    if "design" in expr:
        return (f"design({expr['design']}, {json.dumps(expr['ops'], separators=(',', ':'))}, "
                f"{json.dumps(expr['args'], separators=(',', ':'))})")
    return f"{HELPERS[expr['h']]}(" + ", ".join(_display(a) for a in expr["args"]) + ")"


def java_cases(question) -> list[dict]:
    """Every test case as {expr, display, expected}. Raises Untranslatable if
    any case cannot be carried."""
    if not is_leetcode(question) or question.submission_mode != "function" or not question.test_cases:
        raise Untranslatable("only LeetCode function drills run in Java")
    return _java_cases_cached(question.id, question.starter_code or "", question.function_name or "",
                              json.dumps(question.test_cases, sort_keys=True))


@lru_cache(maxsize=1024)
def _java_cases_cached(_qid: int, starter: str, function_name: str, cases_json: str) -> list[dict]:
    sig = signature(starter, function_name)
    out = []
    for case in json.loads(cases_json):
        if case.get("assert_code") or case.get("expected_setup_code"):
            raise Untranslatable("assert_code / expected_setup_code are Python-only")
        try:
            call = ast.parse(case["call"], mode="eval").body
            expected = ast.parse(case["expected_expr"], mode="eval").body
        except SyntaxError as exc:
            raise Untranslatable(str(exc)) from exc
        if not _is_literal(expected):
            raise Untranslatable("expected value is not a literal")
        expr = _expr(call, sig)
        out.append({"expr": expr, "display": _display(expr), "expected": json.dumps(_value(expected))})
    return out


def supports_java(question) -> bool:
    try:
        java_cases(question)
        starter_java(question)
        return True
    except Untranslatable:
        return False


# --- types ------------------------------------------------------------------------
# A structural type: ("int"|"long"|"double"|"boolean"|"String"|"char"|"void"|
# "Object"|"ListNode"|"TreeNode",) or ("list", inner).

_ANNOTATION = {"int": ("int",), "float": ("double",), "str": ("String",), "bool": ("boolean",),
               "None": ("void",), "ListNode": ("ListNode",), "TreeNode": ("TreeNode",)}


class _Node:
    """A test argument or result that is a linked list / tree, not a literal."""

    def __init__(self, kind: str):
        self.kind = kind


def _from_annotation(annotation: str):
    t = annotation.replace("typing.", "").replace(" ", "")
    if not t:
        return None
    if t.startswith("Optional[") and t.endswith("]"):
        return _from_annotation(t[9:-1])
    if t.lower().startswith("list[") and t.endswith("]"):
        inner = _from_annotation(t[5:-1])
        return ("list", inner) if inner else None
    return _ANNOTATION.get(t)


def _infer(values: list):
    """The narrowest type every observed value fits; None when nothing was seen."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    if any(isinstance(v, _Node) for v in vals):
        kinds = {v.kind for v in vals if isinstance(v, _Node)}
        return (kinds.pop(),) if len(kinds) == 1 and all(isinstance(v, _Node) for v in vals) else ("Object",)
    if all(isinstance(v, bool) for v in vals):
        return ("boolean",)
    if all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
        return ("long",) if any(not INT_MIN <= v <= INT_MAX for v in vals) else ("int",)
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
        return ("double",)
    if all(isinstance(v, str) for v in vals):
        return ("String",)
    if all(isinstance(v, list) for v in vals):
        return ("list", _infer([x for v in vals for x in v]) or ("int",))
    return ("Object",)


def _refine(t, values: list, role: str):
    """Widen int to long where a value needs it; a parameter list of one-letter
    strings is a char grid (`char[][] board`), as LeetCode's Java has it."""
    if t is None:
        return _infer(values) or (("void",) if role == "return" else ("Object",))
    vals = [v for v in values if v is not None and not isinstance(v, _Node)]
    if t == ("int",) and any(isinstance(v, int) and not isinstance(v, bool) and not INT_MIN <= v <= INT_MAX for v in vals):
        return ("long",)
    if t[0] == "list":
        inner_vals = [x for v in vals if isinstance(v, list) for x in v]
        inner = _refine(t[1], inner_vals, role) if t[1] else (_infer(inner_vals) or ("int",))
        if (role == "param" and inner == ("String",) and inner_vals
                and all(isinstance(x, str) and len(x) == 1 for x in inner_vals)):
            inner = ("char",)
        return ("list", inner)
    return t


_BOX = {"int": "Integer", "long": "Long", "double": "Double", "boolean": "Boolean", "char": "Character"}


def _render(t, role: str) -> str:
    """Parameters take arrays (`int[]`, `char[][]`), as LeetCode's Java mostly
    does; results are Lists (`List<List<Integer>>`). Either is accepted."""
    if t[0] != "list":
        return t[0]
    inner = t[1]
    if role == "param" and (inner[0] in _BOX or inner[0] == "String" or inner[0] == "list"):
        rendered = _render(inner, role)
        if "<" not in rendered:
            return rendered + "[]"
    return f"List<{_boxed(inner)}>"


def _boxed(t) -> str:
    if t[0] == "list":
        return f"List<{_boxed(t[1])}>"
    return _BOX.get(t[0], t[0])


# --- observed values per parameter ---------------------------------------------------

def _arg_value(node: ast.AST):
    if _is_literal(node):
        return _value(node)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in NODE_BUILDERS:
        return _Node(NODE_BUILDERS[node.func.id])
    return None


def _drill_call(node: ast.AST, sig: dict):
    """(the call of the drill's own method inside `node`, the helper wrapping it)."""
    if not isinstance(node, ast.Call):
        return None, None
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr == sig.get("name"):
        return node, None
    if isinstance(func, ast.Name):
        if func.id == sig.get("name"):
            return node, None
        for arg in node.args:
            inner, _ = _drill_call(arg, sig)
            if inner is not None:
                return inner, func.id
    return None, None


def _observed(question, sig: dict) -> tuple[list[list], list]:
    params = [[] for _ in sig.get("params", [])]
    returns = []
    for case in question.test_cases or []:
        try:
            call = ast.parse(case["call"], mode="eval").body
            expected = ast.parse(case["expected_expr"], mode="eval").body
        except (SyntaxError, KeyError):
            continue
        inner, wrapper = _drill_call(call, sig)
        if inner is None:
            continue
        try:
            for i, arg in enumerate(ordered_args(inner, sig["params"])):
                params[i].append(_arg_value(arg))
        except Untranslatable:
            continue
        if wrapper in ("is_same_list", "_from_list_node"):
            returns.append(_Node("ListNode"))
        elif wrapper in ("is_same_tree", "_from_tree_node"):
            returns.append(_Node("TreeNode"))
        elif _is_literal(expected):
            returns.append(_value(expected))
    return params, returns


def _observed_design(question, sig: dict) -> dict:
    """{method: (per-parameter values, results)} from the _lc_design calls."""
    seen: dict[str, tuple[list[list], list]] = {}
    for case in question.test_cases or []:
        try:
            call = ast.parse(case["call"], mode="eval").body
            ops, args = (_value(a) for a in call.args)
            results = _value(ast.parse(case["expected_expr"], mode="eval").body)
        except Exception:  # noqa: BLE001 — a case we cannot read adds no evidence
            continue
        for i, (op, a) in enumerate(zip(ops, args)):
            name = "constructor" if i == 0 else op
            ps, rs = seen.setdefault(name, ([], []))
            while len(ps) < len(a):
                ps.append([])
            for j, v in enumerate(a):
                ps[j].append(v)
            if i > 0 and isinstance(results, list) and i < len(results):
                rs.append(results[i])
    return seen


# --- starter ------------------------------------------------------------------------

_NODE_DOCS = {
    "ListNode": (
        "/**\n * Definition for singly-linked list (provided by the grader).\n"
        " * class ListNode {\n"
        " *     int val;\n"
        " *     ListNode next;\n"
        " *     ListNode() {}\n"
        " *     ListNode(int val) { this.val = val; }\n"
        " *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }\n"
        " * }\n */\n"
    ),
    "TreeNode": (
        "/**\n * Definition for a binary tree node (provided by the grader).\n"
        " * class TreeNode {\n"
        " *     int val;\n"
        " *     TreeNode left;\n"
        " *     TreeNode right;\n"
        " *     TreeNode() {}\n"
        " *     TreeNode(int val) { this.val = val; }\n"
        " *     TreeNode(int val, TreeNode left, TreeNode right) {\n"
        " *         this.val = val;\n"
        " *         this.left = left;\n"
        " *         this.right = right;\n"
        " *     }\n"
        " * }\n */\n"
    ),
}


def _method(name: str, params: list[tuple[str, str]], param_types: list, ret: str | None) -> str:
    args = ", ".join(f"{_render(t, 'param')} {_java_name(p)}" for (p, _), t in zip(params, param_types))
    head = f"public {name}({args})" if ret is None else f"public {ret} {name}({args})"
    return f"    {head} {{\n        \n    }}"


def starter_java(question) -> str:
    starter = question.starter_code or ""
    sig = signature(starter, question.function_name or "")
    used: set[str] = {k for k, v in sig["uses"].items() if v}
    if sig["kind"] == "design":
        seen = _observed_design(question, sig)
        body = []
        for m in sig["methods"]:
            ps, rs = seen.get(m["name"], ([], []))
            types = [_refine(_from_annotation(a), ps[i] if i < len(ps) else [], "param")
                     for i, (_, a) in enumerate(m["params"])]
            if m["name"] == "constructor":
                body.append(_method(sig["name"], m["params"], types, None))
                continue
            ann = next((n for n in _design_returns(starter, sig["name"]) if n[0] == m["name"]), (None, ""))[1]
            ret = _refine(_from_annotation(ann), rs, "return")
            body.append(_method(m["name"], m["params"], types, _render(ret, "return")))
        return f"class {sig['name']} {{\n" + "\n\n".join(body) + "\n}\n"
    params, returns = _observed(question, sig)
    types = [_refine(_from_annotation(a), params[i], "param") for i, (_, a) in enumerate(sig["params"])]
    ret = _refine(_from_annotation(sig["returns"]), returns, "return")
    used |= {t[0] for t in types + [ret] if t[0] in _NODE_DOCS}
    head = "".join(doc for name, doc in _NODE_DOCS.items() if name in used)
    return f"{head}class Solution {{\n{_method(sig['name'], sig['params'], types, _render(ret, 'return'))}\n}}\n"


@lru_cache(maxsize=256)
def _design_returns(starter: str, cls_name: str) -> tuple[tuple[str, str], ...]:
    for node in ast.parse(starter).body:
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            return tuple((n.name, ast.unparse(n.returns) if n.returns else "")
                         for n in node.body if isinstance(n, ast.FunctionDef))
    return ()


# --- execution ------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _harness_dir() -> str | None:
    """The harness, compiled once per source version into a shared temp dir."""
    source = HARNESS_PATH.read_bytes()
    out = Path(tempfile.gettempdir()) / f"delta_java_harness_{hashlib.sha256(source).hexdigest()[:16]}"
    if (out / f"{HARNESS_CLASS}.class").exists():
        return str(out)
    javac = javac_binary()
    if javac is None:
        return None
    staging = tempfile.mkdtemp(prefix="delta_java_harness_build_")
    proc = subprocess.run([javac, "-nowarn", "-Xlint:none", "-d", staging, str(HARNESS_PATH)],
                          capture_output=True, text=True, timeout=180)
    if proc.returncode != 0:
        logger.error("Java harness does not compile: %s", proc.stderr[-2000:])
        shutil.rmtree(staging, ignore_errors=True)
        return None
    try:
        os.replace(staging, out)
    except OSError:  # another worker got there first
        shutil.rmtree(staging, ignore_errors=True)
    return str(out)


_SM_BANNER = re.compile(r"^WARNING: (A terminally deprecated method in java\.lang\.System|System::setSecurityManager|Please consider reporting).*\n?", re.M)


def _run_jvm(payload: dict, timeout: int) -> tuple[ExecutionResult, str | None]:
    """(the run, the report file's text or None)."""
    java = java_binary()
    if java is None:
        return ExecutionResult(stdout="", stderr="Java is not installed on the server.", success=False), None
    if javac_binary() is None:
        return ExecutionResult(stdout="", stderr="This server cannot compile Java (no javac).", success=False), None
    harness = _harness_dir()
    if harness is None:
        return ExecutionResult(stdout="", stderr="The Java grader failed to build on the server.", success=False), None
    with tempfile.TemporaryDirectory(prefix="practice_java_") as tmp:
        payload_path = os.path.join(tmp, "payload.json")
        report_path = os.path.join(tmp, "report.json")
        with open(payload_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        try:
            with _jvm_slots:
                proc = subprocess.run(
                    [java, *JVM_FLAGS, f"-Djava.io.tmpdir={tmp}", "-cp", harness, HARNESS_CLASS, payload_path, report_path],
                    capture_output=True, text=True, timeout=timeout, cwd=tmp,
                    env={k: os.environ[k] for k in ENV_KEYS if k in os.environ},
                )
        except subprocess.TimeoutExpired:
            return ExecutionResult(stdout="", stderr=f"Execution timed out after {timeout} seconds", success=False), None
        report = None
        if os.path.exists(report_path):
            with open(report_path, encoding="utf-8") as fh:
                report = fh.read() or None  # opened before compiling: empty = never reached
        # JDK 17-23 announce the deprecated SecurityManager on the process's
        # own stderr; the learner's ▶ would show it on every run.
        stderr = _SM_BANNER.sub("", proc.stderr)
        if proc.returncode == EXIT_NO_SANDBOX:
            logger.warning("java %s cannot install a SecurityManager; Java answers are refused", java)
        return ExecutionResult(stdout=proc.stdout.rstrip(), stderr=stderr, success=proc.returncode == 0), report


def run_java(user_code: str, question=None, timeout: int = TIMEOUT_SECONDS) -> ExecutionResult:
    """The cell's ▶: compile, then run `main` if there is one; otherwise run
    the drill's cases for what they PRINT (a println inside the method is how
    a Java learner debugs). The verdict is /check's job; rows are dropped."""
    try:
        cases = [{"expr": c["expr"], "expected": c["expected"]} for c in java_cases(question)] if question else []
    except Untranslatable:
        cases = []
    result, _ = _run_jvm({"code": user_code, "mode": "run", "cases": cases, "caseTimeoutMs": CASE_TIMEOUT_MS}, timeout)
    return result


def run_java_tests(user_code: str, question, timeout: int = TIMEOUT_SECONDS) -> tuple[list[TestCaseResult], ExecutionResult, list[dict]]:
    """Grade Java against the question's cases: per-case results, the run, and
    the translated cases (their `display` is what the ▶ block shows)."""
    cases = java_cases(question)
    execution, report = _run_jvm(
        {"code": user_code, "mode": "tests", "caseTimeoutMs": CASE_TIMEOUT_MS,
         "cases": [{"expr": c["expr"], "expected": c["expected"]} for c in cases]},
        timeout,
    )
    if report is None:
        error = execution.stderr.strip() or execution.stdout.strip() or "Java test harness failed."
        return [TestCaseResult(passed=False, actual="", expected="", error=error)], execution, cases
    try:
        rows = [TestCaseResult(**row) for row in json.loads(report)]
    except Exception as exc:  # noqa: BLE001 — a malformed report is a failed grade
        return [TestCaseResult(passed=False, actual="", expected="", error=f"Invalid test report: {exc}")], execution, cases
    if not cases or len(rows) != len(cases):
        # all([]) is True: a short or empty report must never read as a pass.
        rows.append(TestCaseResult(passed=False, actual="", expected="",
                                   error=f"Test report had {len(rows)} rows for {len(cases)} cases."))
    if not execution.success:
        crash = execution.stderr.strip() or "Java process exited with an error."
        rows.append(TestCaseResult(passed=False, actual="", expected="", error=crash))
    return rows, execution, cases
