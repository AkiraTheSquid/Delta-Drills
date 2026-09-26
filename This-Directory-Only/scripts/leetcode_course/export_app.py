"""Turn the finished bank into what the app reads.

Writes three things, each owned wholesale for its `leetcode.*` / lc-1 slice:
  - Local_Deployed_Shared/lessons/leetcode/problems.json: flat drill rows in
    questions.json's shape, which pipeline/export_questions_json.py appends.
  - lessons/kc_registry.json: the lc-1 lesson row + one row per concept that
    has at least one drill (a drill-less concept in a standalone course would
    leave the picker a frontier node it can never serve).
  - lessons/qmatrix_tags.json: question id -> its one target concept.

Ids: lessons/leetcode/ids.json maps problem id -> question id, append-only
from ID_FLOOR, so a re-export never renumbers a drill a learner has answered.

Tests: the dataset's `check(candidate)` asserts become test cases one to one.
`assert candidate(...) == X` -> call `Solution().m(...)`, expected `X`; any
other assert shape -> the whole expression as the call, expected `True`.
Every exported row is re-run through the backend grader before it is written
(the reference passes all cases, the bare starter fails at least one).
"""
from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

from drill_format import clock_secs, statement_markdown

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
COURSE_DIR = REPO / "This-Directory-Only" / "leetcode_course"
LESSONS = REPO / "Local_Deployed_Shared" / "lessons"
OUT_DIR = LESSONS / "leetcode"
ID_FLOOR = 60000
LESSON = {"id": "lc-1", "topic": "LeetCode", "title": "LeetCode Patterns", "subtopic_key": "LeetCode: Patterns"}
MAX_CASES = 8
MAX_CASE_CHARS = 700
CLOCK_SECS = 1200  # the concept's PLACEMENT clock (table ceiling); practice uses each drill's secs_allowed

# Runtime names LeetCode's judge provides implicitly. In setup_code (runs after
# the learner's code, same globals) so a solution that uses deque/heappush/inf
# without importing them still works, exactly as on LeetCode.
RUNTIME = (
    "import collections, heapq, bisect, math, itertools, functools, string\n"
    "from typing import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\n"
    "from itertools import *\nfrom functools import *\nfrom math import *\ninf = float('inf')\n"
)
LIST_NODE = (
    "class ListNode:\n    def __init__(self, val=0, next=None):\n        self.val = val\n        self.next = next\n"
)
TREE_NODE = (
    "class TreeNode:\n    def __init__(self, val=0, left=None, right=None):\n"
    "        self.val = val\n        self.left = left\n        self.right = right\n"
)


def _helper_defs(preamble: str) -> dict[str, str]:
    """The dataset preamble's test helpers (list_node, tree_node, is_same_*), by name."""
    tree = ast.parse(preamble)
    return {n.name: ast.get_source_segment(preamble, n) for n in tree.body
            if isinstance(n, ast.FunctionDef)}


def _cases(check: str, entry: str) -> list[tuple[str, str]]:
    fn = ast.parse(check).body[0]
    out = []
    for st in fn.body:
        if not isinstance(st, ast.Assert):
            continue
        test = st.test
        if (isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq)):
            call, expected = ast.unparse(test.left), ast.unparse(test.comparators[0])
        else:
            call, expected = ast.unparse(test), "True"
        call = call.replace("candidate(", entry + "(")
        if "candidate" in call or len(call) + len(expected) > MAX_CASE_CHARS:
            continue
        out.append((call, expected))
        if len(out) == MAX_CASES:
            break
    return out


def _header(code: str) -> str:
    parts = ["from typing import List, Optional\n"]
    if "ListNode" in code:
        parts.append("\n# Definition for singly-linked list.\n" + LIST_NODE)
    if "TreeNode" in code:
        parts.append("\n# Definition for a binary tree node.\n" + TREE_NODE)
    return "".join(parts) + "\n"


def _label(score: int) -> str:
    return "easy" if score < 40 else "medium" if score < 70 else "hard"


DESIGN_RUNNER = '''def _lc_design(ops, args):
    """Replay a design problem's operations; the first op constructs the object."""
    obj, outs = None, []
    for op, a in zip(ops, args):
        if obj is None:
            obj = globals()[op](*a)
            outs.append(None)
        else:
            outs.append(getattr(obj, op)(*a))
    return outs'''


def _generated_cases(t: dict, entry: str) -> list[tuple[str, str, list[str]]]:
    """harness.py-shaped inputs/outputs -> (call, expected, helper names).
    Argument encodings become the builder call (`tree_node([...])`), so the
    learner reads the case as it would appear on LeetCode."""
    out = []
    for inp, want in zip(t["inputs"], t["outputs"]):
        if t["shape"] == "design":
            call, need = f"_lc_design({inp['ops']!r}, {inp['args']!r})", ["_lc_design"]
        else:
            types = t.get("arg_types") or []
            args, need = [], []
            for i, a in enumerate(inp):
                kind = types[i] if i < len(types) else "plain"
                args.append(f"{kind}({a!r})" if kind != "plain" else repr(a))
                need += [kind] if kind != "plain" else []
            call = f"{entry}({', '.join(args)})"
            if t.get("return_type") in ("list_node", "tree_node"):
                conv = "_from_" + t["return_type"]
                call, need = f"{conv}({call})", need + [conv]
            if t.get("compare") == "unordered":
                call, want = f"sorted({call})", sorted(want)
        expected = repr(want)
        if len(call) + len(expected) > MAX_CASE_CHARS:
            continue
        # list_node_cycle builds on list_node.
        need += ["list_node"] if "list_node_cycle" in need else []
        out.append((call, expected, need))
        if len(out) == MAX_CASES:
            break
    return out


def _test_cases(p: dict) -> list[dict]:
    entry = p["entry_point"]
    if p["tests"]["kind"] == "assert":
        helpers = _helper_defs(p["preamble"])
        cases = [(c, e, [n for n in ("list_node", "is_same_list", "tree_node", "is_same_tree") if n in c + e])
                 for c, e in _cases(p["tests"]["check"], entry)]
    else:
        import harness
        helpers = {**_helper_defs(harness.HELPERS), "_lc_design": DESIGN_RUNNER}
        cases = _generated_cases(p["tests"], entry)
    # The bank audit judges the first case alone: a `None` answer there reads as
    # a placeholder the bare starter passes, so lead with a real value.
    cases.sort(key=lambda c: c[1] == "None")
    return [{"setup_code": RUNTIME + "\n\n".join(helpers[n] for n in dict.fromkeys(need)),
             "call": call, "expected_expr": expected} for call, expected, need in cases]


def drill_row(p: dict, qid: int, concept_title: str) -> dict | None:
    test_cases = _test_cases(p)
    if len(test_cases) < 3:
        return None
    # The builders in the cases (list_node, tree_node) need the node classes
    # even when neither the starter nor the solution names them.
    header = _header(p["starter_code"] + p["solution"]
                     + json.dumps(test_cases).replace("list_node", "ListNode").replace("tree_node", "TreeNode"))
    starter = p["starter_code"].rstrip()
    if p["tests"]["kind"] == "assert":  # the dataset's starter stops at the signature
        starter += "\n        pass"
    starter += "\n"
    entry = p["entry_point"]
    score = int(p.get("difficulty_score", 50))
    return {
        "id": qid,
        "topic": LESSON["topic"],
        "subtopic": "Patterns",
        "subtopic_key": LESSON["subtopic_key"],
        "question_text": f"**{p['title']}**\n\n{statement_markdown(p['statement'].strip())}",
        "answer_code": header + p["solution"].strip() + "\n",
        "difficulty_score": score,
        "difficulty_label": _label(score),
        "expected_output": "",
        "language": "python",
        "primary_library": "python",
        "task_type": "function_impl",
        "function_name": entry.rsplit(".", 1)[-1],
        "starter_code": header + starter,
        "test_cases": test_cases,
        "submission_mode": "function",
        "wrong_examples": [],
        "provenance": {"source": ("newfacade/LeetCodeDataset (Apache-2.0)" if p["tests"]["kind"] == "assert"
                                  else "generated: outputs from two agreeing reference solutions"),
                       "problem": p["id"], "url": p.get("url")},
        "expected_artifact_type": "value",
        "supports_visual_output": False,
        "source_type": "leetcode_json",
        "source_path": "Local_Deployed_Shared/lessons/leetcode/problems.json",
        "leetcode_kc": p["concept"],
        "concept_title": concept_title,
        # The drill's own answer clock (drill_format.clock_secs); the backend
        # serves it in place of the concept's 5:00-clamped table clock.
        "secs_allowed": clock_secs(p),
    }


def _grader():
    path = REPO / "This-Directory-Only" / "backend" / "app" / "code_runner.py"
    spec = importlib.util.spec_from_file_location("lc_code_runner", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _grades(runner, row: dict) -> bool:
    good, _ = runner.run_function_tests(row["answer_code"], row["test_cases"], timeout=30)
    if not good or not all(r.passed for r in good):
        return False
    bad, _ = runner.run_function_tests(row["starter_code"], row["test_cases"], timeout=30)
    return not all(r.passed for r in bad)


def _sync_side_files(kept: list[dict]) -> None:
    """The two per-concept tables the lesson/practice watches hold every
    registry KC to: glossary.js kcLesson (popup 'Taught in' line) and
    placement_time_caps.json (the answer clock)."""
    gl_path = LESSONS / "glossary.js"
    lines = [l for l in gl_path.read_text().split("\n") if not l.startswith('  "leetcode.')]
    at = next(i for i, l in enumerate(lines) if l.startswith("window.DD_GLOSSARY.kcLesson = {")) + 1
    lines[at:at] = [f"  {json.dumps(k['id'])}: {json.dumps([LESSON['title'], k['title']])}," for k in kept]
    gl_path.write_text("\n".join(lines))
    caps_path = LESSONS / "placement_time_caps.json"
    caps = json.loads(caps_path.read_text())
    caps["kcs"] = {k: v for k, v in caps["kcs"].items() if not k.startswith("leetcode.")}
    caps["kcs"].update({k["id"]: CLOCK_SECS for k in kept})
    caps_path.write_text(json.dumps(caps, indent=2, ensure_ascii=False) + "\n")


ATOM_GRAPH = REPO / "This-Directory-Only" / "backend" / "app" / "data" / "concept_graphs" / "arena_drillable_v1.json"
ATOM_TAGS = REPO / "This-Directory-Only" / "backend" / "app" / "data" / "question_atom_tags.jsonl"


def atom_of(kc: str) -> str:
    return "lc-" + kc.split(".", 1)[1]


def _sync_atoms(kept: list[dict], rows: list[dict]) -> None:
    """Mastery is BKT over atoms, and a drill's atoms come from its tag row:
    without these an answer moves nothing. One atom per concept, the atom
    graph mirroring the concept graph edge for edge (encompassing weight =
    propagation weight), the course root as an intentional root atom."""
    text = ATOM_GRAPH.read_text(encoding="utf-8")
    graph = json.loads(text)
    ours = lambda a: a.startswith("lc-")
    graph["concepts"] = [c for c in graph["concepts"] if not ours(c["id"])] + [
        {"id": atom_of(k["id"]), "title": k["title"], "topic": LESSON["topic"], "subtopic": "Patterns",
         "node_type": "concept", "kind": "skill", "tier": "core",
         "description": f"LeetCode Patterns course: assessed by coding problems via {k['id']}.",
         "tags": ["drill-atom", "leetcode"]} for k in kept]
    edges = [e for e in graph["prerequisite_edges"] if not ours(e["dependent_id"])]
    for k in kept:
        enc = k.get("encompassing", {})
        for pre in k["prereqs"]:
            edges.append({"prerequisite_id": atom_of(pre), "dependent_id": atom_of(k["id"]), "weight": 1,
                          "confidence": 0.7, "is_hard_gate": True, "is_encompassing": pre in enc,
                          "propagation_weight": enc.get(pre, 0.0),
                          "rationale": "LeetCode Patterns graph (This-Directory-Only/leetcode_course/patterns.json)."})
    graph["prerequisite_edges"] = edges
    roots = [atom_of(k["id"]) for k in kept if not k["prereqs"]]
    graph["intentional_root_atoms"] = [a for a in graph["intentional_root_atoms"] if not ours(a)] + roots
    out = json.dumps(graph, indent=2, ensure_ascii=False)
    old = json.dumps(json.loads(text), indent=2, ensure_ascii=False)
    ATOM_GRAPH.write_text(out + text[len(old):], encoding="utf-8")  # keep the file's own trailing bytes

    lines = [l for l in ATOM_TAGS.read_text(encoding="utf-8").splitlines()
             if l.strip() and not (ID_FLOOR <= json.loads(l)["question_id"] < ID_FLOOR + 10000)]
    lines += [json.dumps({"question_id": r["id"], "atoms": [{"atom_id": atom_of(r["leetcode_kc"]), "confidence": 1}]},
                         separators=(",", ":")) for r in rows]
    ATOM_TAGS.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    graph = json.loads((COURSE_DIR / "patterns.json").read_text())["kcs"]
    titles = {k["id"]: k["title"] for k in graph}
    bank = [json.loads(line) for line in open(COURSE_DIR / "bank.jsonl")]
    ids_path = OUT_DIR / "ids.json"
    ids = json.loads(ids_path.read_text()) if ids_path.exists() else {}
    runner = _grader()
    rows, skipped = [], []
    for p in bank:
        if p["id"] not in ids:
            ids[p["id"]] = max(ids.values(), default=ID_FLOOR - 1) + 1
        row = drill_row(p, ids[p["id"]], titles[p["concept"]])
        if row is None or not _grades(runner, row):
            skipped.append(p["id"])
            continue
        rows.append(row)
    OUT_DIR.mkdir(exist_ok=True)
    ids_path.write_text(json.dumps(ids, indent=1) + "\n")
    (OUT_DIR / "problems.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n")

    served = {r["leetcode_kc"] for r in rows}
    reg_path = LESSONS / "kc_registry.json"
    reg = json.loads(reg_path.read_text())
    reg["lessons"] = [l for l in reg["lessons"] if l.get("id") != LESSON["id"]] + [LESSON]
    kept = []
    for k in graph:
        if k["id"] not in served:
            continue
        pre = [x for x in k["prereqs"] if x in served]
        row = {"id": k["id"], "lesson": LESSON["id"], "title": k["title"], "syntax": False, "prereqs": pre}
        enc = {x: w for x, w in k.get("encompassing", {}).items() if x in served}
        if enc:
            row["encompassing"] = enc
        kept.append(row)
    reg["kcs"] = [k for k in reg["kcs"] if not k["id"].startswith("leetcode.")] + kept
    reg_path.write_text(json.dumps(reg, indent=1) + "\n")

    qm_path = LESSONS / "qmatrix_tags.json"
    qm = {k: v for k, v in json.loads(qm_path.read_text()).items() if not (ID_FLOOR <= int(k) < ID_FLOOR + 10000)}
    for r in rows:
        qm[str(r["id"])] = {"target_kcs": [r["leetcode_kc"]], "supporting_kcs": [], "new_syntax": [],
                            "source": "kp-independent"}
    # Same key order as scripts/build_qmatrix.py, which rebuilds this file whole.
    qm_path.write_text(json.dumps({k: qm[k] for k in sorted(qm, key=int)}, indent=1))
    _sync_side_files(kept)
    _sync_atoms(kept, rows)

    dropped = [k["id"] for k in graph if k["id"] not in served]
    print(f"{len(rows)} drills -> {OUT_DIR / 'problems.json'}; skipped {len(skipped)}: {skipped[:20]}")
    print(f"{len(kept)} concepts registered; no drills (left out): {dropped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
