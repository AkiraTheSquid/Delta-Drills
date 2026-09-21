#!/usr/bin/env python3
"""Validate the math (multiple-choice) content lane. Spec: docs/spec-math-mc-backbone.md.

A math KP is `lessons/<topic>/kp-<slug>.md` with `kind: math` in its
frontmatter plus `kp-<slug>.problems.json` beside it (format: lessons/math_bank.py).
This is the gate that lets Fable author one and know it will serve:

Per problems file:
  1. shape — `kc` matches the KP beside it and exists in the registry; the
     KP's lesson is a Mathematics lesson
  2. per problem — id >= MATH_ID_FLOOR and unique across every math file and
     the CSV bank; kind in KINDS; difficulty in 10..100; prompt, solution and
     every choice text non-empty; choice keys A.. in order; texts distinct;
     answer is one of the keys
  3. SymPy — required for `compute` and `derivation-step`: `verify.sympy.truth`
     evaluates, equals the keyed choice's `value`, and differs from EVERY
     distractor's `value`. Forbidden on `statement` (nothing to verify).
     Each problem gets SYMPY_TIMEOUT_SECS; a hang is a failure, never a pass.
     🔴 `sympy.sympify` EVALUATES the strings as Python — a problems file is
     code, trusted exactly as far as the ```python fences validate_lessons.py
     already executes from the same tree. Review a problems file from
     outside the repo before running the gate on it.
  4. rung placement — the KP frontmatter (faded/independent/integrated) lists
     exactly this file's ids: every problem sits on ONE rung, no rung names
     a problem that does not exist, `guided` (legacy) stays empty. A targeted
     run still sees the ids every other math file owns.
  5. atom tags — every problem has a row in backend/app/data/question_atom_tags.jsonl
     whose atoms are nodes of the concept graph (the audit gate blocks a
     placed-but-untagged question; say so here, at authoring time, by id)
  6. WARN — the KC has no entry in lessons/placement_time_caps.json (default
     clock applies)

Per math KP markdown (called from validate_lessons.py for `kind: math` pages,
in place of the code checks): every segment has a non-empty Concept and
Worked example; no runnable ```python fence (a math page has no kernel —
use ```python no-run for illustration); a KP with 2+ segments has at least
one `## Faded practice` item per segment (the segment gate's carrier);
frontmatter/section ids agree.

Usage: python3 scripts/validate_math.py [file.problems.json ...]
Exit 0 = pass. Stdlib + sympy (already in the backend venv).
"""
from __future__ import annotations

import json
import re
import signal
import sys
from pathlib import Path

from lesson_lib import LESSONS_DIR, REPO, code_fences, load_registry, parse_frontmatter, parse_kp, split_items

sys.path.insert(0, str(LESSONS_DIR))
import math_bank  # noqa: E402

MATH_TOPIC = "Mathematics"
SYMPY_TIMEOUT_SECS = 5
ATOM_TAGS_PATH = REPO / "This-Directory-Only" / "backend" / "app" / "data" / "question_atom_tags.jsonl"
ATOM_GRAPH_PATH = REPO / "This-Directory-Only" / "backend" / "app" / "data" / "concept_graphs" / "arena_drillable_v1.json"
TIME_CAPS_PATH = LESSONS_DIR / "placement_time_caps.json"
CSV_BANK_PATH = REPO / "Local_Deployed_Shared" / "questions_structured.json"
WARNINGS: list[str] = []


# ---------------------------------------------------------------------------
# SymPy equality with a per-problem clock
# ---------------------------------------------------------------------------

class _Timeout(Exception):
    pass


def _alarm(_signum, _frame):
    raise _Timeout()


def _sympify(text: str, ns: dict):
    import sympy

    return sympy.sympify(text, locals=ns)


def _equal(a, b) -> bool:
    """Structural-then-symbolic equality; a type mismatch is a difference."""
    import sympy

    is_mat = lambda x: isinstance(x, sympy.MatrixBase)  # noqa: E731
    if is_mat(a) != is_mat(b):
        return False
    if is_mat(a):
        if a.shape != b.shape:
            return False
        return (a - b).applyfunc(sympy.simplify).is_zero_matrix is True
    if isinstance(a, (bool, sympy.logic.boolalg.Boolean)) or isinstance(b, (bool, sympy.logic.boolalg.Boolean)):
        # `bool(Eq(x, 1))` raises on an unresolved relation; compare the
        # simplified forms structurally instead (Eq(x, 1) vs Eq(1, x) is
        # canonicalised by simplify; true/false fold to BooleanAtoms).
        if not isinstance(a, (bool, sympy.logic.boolalg.Boolean)) or not isinstance(b, (bool, sympy.logic.boolalg.Boolean)):
            return False
        sa, sb = sympy.simplify(a), sympy.simplify(b)
        if sa == sb:
            return True
        return sympy.simplify(sympy.Equivalent(sa, sb)) is sympy.true
    if isinstance(a, (set, frozenset, sympy.Set)) or isinstance(b, (set, frozenset, sympy.Set)):
        return a == b
    try:
        if a.equals(b):
            return True
    except Exception:
        pass
    try:
        return sympy.simplify(a - b) == 0
    except Exception:
        return False


def sympy_findings(problem: dict, label: str) -> list[str]:
    """Errors for one problem's `verify.sympy` block (empty = verified)."""
    kind = problem.get("kind")
    verify = problem.get("verify") if isinstance(problem.get("verify"), dict) else {}
    spec = verify.get("sympy")
    if kind not in math_bank.SYMPY_KINDS:
        if "sympy" in verify:
            return [f"{label}: kind `{kind}` must not carry verify.sympy (nothing to verify)"]
        return []
    if not isinstance(spec, dict) or not str(spec.get("truth") or "").strip():
        return [f"{label}: kind `{kind}` needs verify.sympy.truth (the author's own derivation of the answer)"]
    import sympy

    ns: dict = {}
    names = str(spec.get("symbols") or "").strip()
    if names:
        syms = sympy.symbols(names)
        for s in (syms if isinstance(syms, (tuple, list)) else (syms,)):
            ns[str(s)] = s
    for extra in ("Matrix", "Rational", "sqrt", "pi", "E", "I", "oo", "exp", "log", "sin", "cos", "tan",
                  "eye", "zeros", "ones", "diag", "det", "Abs", "factorial", "binomial", "Sum", "Integral",
                  "diff", "integrate", "limit", "simplify", "expand", "factor", "Symbol", "symbols",
                  "Eq", "S", "nsimplify", "Piecewise", "Max", "Min", "floor", "ceiling", "Transpose",
                  "MatrixSymbol", "Identity", "trace", "gcd", "lcm", "Mod", "sign", "atan2", "asin", "acos", "atan"):
        ns.setdefault(extra, getattr(sympy, extra))

    errors: list[str] = []
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(SYMPY_TIMEOUT_SECS)
    try:
        try:
            truth = _sympify(str(spec["truth"]), ns)
        except Exception as exc:
            return [f"{label}: verify.sympy.truth does not evaluate: {type(exc).__name__}: {exc}"]
        answer = str(problem.get("answer") or "")
        for choice in problem.get("choices") or []:
            if not isinstance(choice, dict):
                continue  # problem_findings already reported the shape
            key = str(choice.get("key") or "")
            raw = choice.get("value")
            if raw is None or not str(raw).strip():
                errors.append(f"{label}: choice {key} has no `value` (SymPy form) — every choice of a `{kind}` problem needs one")
                continue
            try:
                value = _sympify(str(raw), ns)
            except Exception as exc:
                errors.append(f"{label}: choice {key} value does not evaluate: {type(exc).__name__}: {exc}")
                continue
            try:
                same = _equal(truth, value)
            except _Timeout:
                raise
            except Exception as exc:  # a comparison that blows up is unverified, not passed
                errors.append(f"{label}: choice {key} could not be compared with the truth: "
                              f"{type(exc).__name__}: {exc}")
                continue
            if key == answer and not same:
                errors.append(f"{label}: keyed choice {key} ({raw}) != truth ({spec['truth']})")
            if key != answer and same:
                errors.append(f"{label}: distractor {key} ({raw}) equals the truth — two correct answers")
    except _Timeout:
        errors.append(f"{label}: SymPy check exceeded {SYMPY_TIMEOUT_SECS}s — unverified, not passed")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    return errors


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------

def _atom_rows() -> dict[int, list[str]]:
    rows: dict[int, list[str]] = {}
    if not ATOM_TAGS_PATH.exists():
        return rows
    for line in ATOM_TAGS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        rows[int(rec["question_id"])] = [a.get("atom_id") for a in rec.get("atoms", [])]
    return rows


def _atom_nodes() -> set[str]:
    if not ATOM_GRAPH_PATH.exists():
        return set()
    graph = json.loads(ATOM_GRAPH_PATH.read_text(encoding="utf-8"))
    return {c["id"] for c in graph.get("concepts", [])}


def _csv_ids() -> set[int]:
    if not CSV_BANK_PATH.exists():
        return set()
    bank = json.loads(CSV_BANK_PATH.read_text(encoding="utf-8"))
    return {int(q["id"]) for q in bank if (q.get("source") or {}).get("type") != "math_json"}


def problem_findings(problem: dict, label: str) -> list[str]:
    errors: list[str] = []
    pid = problem.get("id")
    if not isinstance(pid, int) or pid < math_bank.MATH_ID_FLOOR:
        errors.append(f"{label}: id must be an integer >= {math_bank.MATH_ID_FLOOR}")
    if problem.get("kind") not in math_bank.KINDS:
        errors.append(f"{label}: kind must be one of {list(math_bank.KINDS)}")
    diff = problem.get("difficulty")
    lo, hi = math_bank.DIFFICULTY_RANGE
    if not isinstance(diff, int) or not lo <= diff <= hi:
        errors.append(f"{label}: difficulty must be an integer in {lo}..{hi}")
    for field in ("prompt", "solution"):
        if not str(problem.get(field) or "").strip():
            errors.append(f"{label}: `{field}` is empty")
    choices = problem.get("choices")
    if not isinstance(choices, list) or not math_bank.MIN_CHOICES <= len(choices) <= math_bank.MAX_CHOICES:
        errors.append(f"{label}: needs {math_bank.MIN_CHOICES}..{math_bank.MAX_CHOICES} choices")
        return errors
    keys = [str(c.get("key") or "") if isinstance(c, dict) else "" for c in choices]
    expected = [chr(ord("A") + i) for i in range(len(choices))]
    if keys != expected:
        errors.append(f"{label}: choice keys must be {expected} in order, got {keys}")
    texts = [str(c.get("text") or "").strip() if isinstance(c, dict) else "" for c in choices]
    if any(not t for t in texts):
        errors.append(f"{label}: every choice needs non-empty `text`")
    if len(set(texts)) != len(texts):
        errors.append(f"{label}: two choices show the same text")
    if str(problem.get("answer") or "") not in keys:
        errors.append(f"{label}: answer must be one of the choice keys")
    if "lean" in (problem.get("verify") or {}):
        WARNINGS.append(f"{label}: verify.lean is reserved for a later pass and is not checked")
    return errors


def check_problem_file(path: Path, registry: dict, seen_ids: dict[int, str], csv_ids: set[int],
                       atom_rows: dict[int, list[str]], atom_nodes: set[str], errors: list[str]) -> None:
    name = path.name
    try:
        data = math_bank.read_problem_file(path)
    except Exception as exc:
        errors.append(f"{name}: {exc}")
        return
    kc = str(data["kc"])
    kcs = {k["id"]: k for k in registry.get("kcs", [])}
    lessons = {l["id"]: l for l in registry.get("lessons", [])}
    if kc not in kcs:
        errors.append(f"{name}: kc `{kc}` not in kc_registry.json")
    else:
        lesson = lessons.get(kcs[kc].get("lesson")) or {}
        if lesson.get("topic") != MATH_TOPIC:
            errors.append(f"{name}: kc `{kc}` belongs to lesson `{kcs[kc].get('lesson')}` whose topic is "
                          f"`{lesson.get('topic')}`, not `{MATH_TOPIC}`")
    md_path = math_bank.kp_markdown_for(path)
    meta = {}
    if not md_path.exists():
        errors.append(f"{name}: no KP markdown beside it ({md_path.name})")
    else:
        try:
            meta, _ = parse_frontmatter(md_path.read_text(encoding="utf-8"), md_path)
        except Exception as exc:
            errors.append(f"{md_path.name}: {exc}")
        if meta.get("kind") != "math":
            errors.append(f"{md_path.name}: frontmatter needs `kind: math` to pair with {name}")
        if meta.get("kc") != kc:
            errors.append(f"{name}: kc `{kc}` != {md_path.name} kc `{meta.get('kc')}`")

    ids_here: list[int] = []
    for i, problem in enumerate(data["problems"]):
        if not isinstance(problem, dict):
            errors.append(f"{name}: problem #{i} is not an object")
            continue
        label = f"{name} q{problem.get('id', '?')}"
        errors.extend(problem_findings(problem, label))
        pid = problem.get("id")
        if isinstance(pid, int):
            if pid in seen_ids:
                errors.append(f"{label}: id already used in {seen_ids[pid]}")
            elif pid in csv_ids:
                errors.append(f"{label}: id collides with a drill in the CSV bank")
            seen_ids[pid] = name
            ids_here.append(pid)
            atoms = atom_rows.get(pid)
            if not atoms:
                errors.append(f"{label}: no row in {ATOM_TAGS_PATH.name} — a placed question without atoms "
                              "moves no mastery and the deploy audit blocks it")
            else:
                for atom in atoms:
                    if atom not in atom_nodes:
                        errors.append(f"{label}: atom `{atom}` is not a node of {ATOM_GRAPH_PATH.name}")
        errors.extend(sympy_findings(problem, label))

    if meta:
        if meta.get("guided"):
            errors.append(f"{md_path.name}: `guided` is a legacy rung with no section on a math page — "
                          "use faded / independent / integrated")
        placed: list[int] = []
        for role in ("faded", "guided", "independent", "integrated"):
            placed += [int(x) for x in (meta.get(role) or []) if isinstance(x, int)]
        not_placed = sorted(set(ids_here) - set(placed))
        unknown = sorted(set(placed) - set(ids_here))
        # Exactly once: a problem on two rungs would be served under two
        # ladder roles and emitted twice into the notebook.
        twice = sorted({x for x in placed if placed.count(x) > 1})
        if not_placed:
            errors.append(f"{name}: problems on no rung of {md_path.name}: {not_placed}")
        if unknown:
            errors.append(f"{md_path.name}: rung ids with no problem in {name}: {unknown}")
        if twice:
            errors.append(f"{md_path.name}: ids placed on more than one rung: {twice}")
    if TIME_CAPS_PATH.exists():
        caps = json.loads(TIME_CAPS_PATH.read_text(encoding="utf-8")).get("kcs", {})
        if kc not in caps:
            WARNINGS.append(f"{name}: kc `{kc}` has no entry in placement_time_caps.json — default clock applies")


# ---------------------------------------------------------------------------
# The KP markdown (called by validate_lessons.py for kind: math pages)
# ---------------------------------------------------------------------------

_RUNNABLE_FENCE = re.compile(r"^[ \t]*```python[ \t]*$", re.M)
_ITEM_HEADING = re.compile(r"^### q(\d+)\s*$", re.M)


def check_math_kp(kp: dict, path: Path, bank: dict, errors: list[str]) -> None:
    name = path.name
    problems_path = path.with_name(path.name[:-3] + ".problems.json")
    if not problems_path.exists():
        errors.append(f"{name}: kind: math but no {problems_path.name} beside it")
    text = path.read_text(encoding="utf-8")
    if _RUNNABLE_FENCE.search(text):
        errors.append(f"{name}: a math page has no kernel — use ```python no-run for illustrative code")
    for si, seg in enumerate(kp["segments"]):
        seg_label = f"segment {si + 1}" + (f" ({seg['title']})" if seg["title"] else "")
        if not seg["concept"].strip():
            errors.append(f"{name}: {seg_label} has an empty '## Concept'")
        if not seg["worked"].strip():
            errors.append(f"{name}: {seg_label} has no worked example (prose + LaTeX, step by step)")
        if len(kp["segments"]) > 1 and not split_items(seg["faded"]):
            errors.append(f"{name}: {seg_label} needs at least one `## Faded practice` item — the segment "
                          "gate serves it as the carrier for this page")
    for title in ("Faded practice", "Solo practice", "Integrated practice"):
        heads = [int(x) for x in _ITEM_HEADING.findall(kp["sections"].get(title, ""))]
        dup = sorted({x for x in heads if heads.count(x) > 1})
        if dup:
            errors.append(f"{name}: {title} lists {dup} more than once")
    if kp.get("guided"):
        errors.append(f"{name}: `guided` is a legacy rung with no section on a math page — "
                      "use faded / independent / integrated")
    faded_ids = {qid for seg in kp["segments"] for qid in split_items(seg["faded"])}
    if set(kp["faded"]) != faded_ids:
        errors.append(f"{name}: frontmatter faded {sorted(kp['faded'])} != sections {sorted(faded_ids)}")
    solo_ids = set(split_items(kp["sections"].get("Solo practice", "")))
    if solo_ids != set(kp["independent"]):
        errors.append(f"{name}: frontmatter independent {sorted(kp['independent'])} != Solo practice {sorted(solo_ids)}")
    integrated_ids = set(split_items(kp["sections"].get("Integrated practice", "")))
    if integrated_ids != set(kp.get("integrated") or []):
        errors.append(f"{name}: frontmatter integrated {sorted(kp.get('integrated') or [])} != sections {sorted(integrated_ids)}")
    for a, b, la, lb in ((faded_ids, solo_ids, "Faded", "Solo"), (faded_ids, integrated_ids, "Faded", "Integrated"),
                         (solo_ids, integrated_ids, "Solo", "Integrated")):
        if a & b:
            errors.append(f"{name}: {sorted(a & b)} listed under both {la} and {lb} practice — one rung per problem")
    for qid in sorted(faded_ids | solo_ids | integrated_ids):
        if qid not in bank:
            errors.append(f"{name}: q{qid} not in the bank — run pipeline/export_questions_json.py after editing "
                          f"{problems_path.name}")
        elif not math_bank.is_math_row(bank[qid].get("exercise") or {}):
            errors.append(f"{name}: q{qid} is a coding drill; a math page may only reference math problems")
    for fence in code_fences(text, "python worked"):
        if fence.strip():
            errors.append(f"{name}: ```python worked fences are for coding pages — put the example in prose")


def check_all(errors: list[str], files: list[Path] | None = None) -> int:
    registry = load_registry()
    seen: dict[int, str] = {}
    csv_ids = _csv_ids()
    atom_rows = _atom_rows()
    atom_nodes = _atom_nodes()
    every = math_bank.problem_files(LESSONS_DIR)
    paths = files or every
    # A targeted run still has to see the ids the OTHER files own, or a
    # cross-file duplicate slips through until the next full run.
    wanted = {p.resolve() for p in paths}
    for other in every:
        if other.resolve() in wanted:
            continue
        try:
            for problem in math_bank.read_problem_file(other)["problems"]:
                if isinstance(problem, dict) and isinstance(problem.get("id"), int):
                    seen.setdefault(problem["id"], other.name)
        except Exception:
            pass  # the full run reports that file
    for path in paths:
        check_problem_file(path, registry, seen, csv_ids, atom_rows, atom_nodes, errors)
    return len(paths)


def main(argv: list[str]) -> int:
    files = [Path(a).resolve() for a in argv if a.endswith(".json")]
    errors: list[str] = []
    n = check_all(errors, files or None)
    for w in WARNINGS:
        print(f"WARN — {w}")
    if errors:
        print(f"FAIL — {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"PASS — {n} math problem file(s) validated")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
