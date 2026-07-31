#!/usr/bin/env python3
"""Validate first-encounter lesson content (spec: docs/spec-first-encounter-course-content.md).

Checks, per KP markdown file:
  1. frontmatter kc exists in kc_registry.json; supporting KCs exist; sections known
  2. every segment has one non-empty Concept, one Python Worked example,
     and one Faded exercise
  3. every plain ```python fence in Concept/Worked example EXECUTES, with a fresh
     namespace PER SEGMENT (```python no-run fences are skipped)
  4. every Faded-practice ### q<id> has starter + solution fences, qid exists in the
     bank, and the solution PASSES the bank question's test_cases
  5. guided/independent qids exist in the bank

Registry-level checks:
  6. KC prereq graph is acyclic; prereqs/lessons resolve
  7. --coverage: every registry KC has exactly one KP file; every easy-topic bank
     question appears in qmatrix_tags.json; every tagged target KC has a KP

Usage: python3 scripts/validate_lessons.py [--coverage] [file.md ...]
Exit 0 = all pass.
"""
import contextlib
import io
import json
import sys
import traceback
from pathlib import Path

from lesson_lib import (LESSONS_DIR, REPO, all_kp_paths, code_fences, load_bank,
                        load_registry, parse_kp, split_items)

EASY_TOPICS = ("Numpy", "Einsum", "Einops")


def run_code(code, ns):
    with contextlib.redirect_stdout(io.StringIO()):
        exec(code, ns)


def values_equal(got, expected):
    """Robust equality: numpy arrays (exact then float-tolerant), tuples/lists recursively."""
    import numpy as np
    if isinstance(expected, (list, tuple)) and isinstance(got, (list, tuple)):
        return len(got) == len(expected) and all(
            values_equal(g, e) for g, e in zip(got, expected))
    try:
        if bool(np.array_equal(got, expected)):
            return True
    except Exception:
        pass
    try:
        return bool(np.allclose(got, expected, equal_nan=True))
    except Exception:
        pass
    try:
        return bool(got == expected)
    except Exception:
        return False


NUMBERS_NPY = str(REPO / "Local_Deployed_Shared" / "delta_numbers.npy")


def grade_against_bank(solution_code, question):
    """Run solution against the bank question's test_cases. Returns list of failures."""
    failures = []
    for i, tc in enumerate(question["exercise"]["test_cases"]):
        # The runtime grader always injects numpy (code_runner.CODE_PREAMBLE),
        # so fixtures may use it even in a torch drill — np.load is the only
        # way to reach the ARENA image. Mirror that, or torch lessons fail here
        # for a reason that cannot happen in the sandbox.
        ns = {}
        try:
            run_code("import numpy as np", ns)
            run_code(solution_code, ns)
            # Bank fixtures use the Docker-image path; point at the local copy.
            setup = tc.get("setup_code", "").replace("/delta_numbers.npy", NUMBERS_NPY)
            run_code(setup, ns)
            got = eval(tc["call"], ns)
            # Mirror the JS/backend graders: expected_setup falls back to
            # re-running setup_code, so expected_expr never sees fixtures the
            # solution mutated in place (e.g. out=-style drills).
            run_code((tc.get("expected_setup_code") or setup), ns)
            expected = eval(tc["expected_expr"], ns)
            if not values_equal(got, expected):
                failures.append(f"case {i}: got {got!r} expected {expected!r}")
        except Exception as e:
            failures.append(f"case {i}: {type(e).__name__}: {e}")
    return failures


def check_kp(path, registry, bank, errors):
    kc_ids = {kc["id"] for kc in registry["kcs"]}
    try:
        kp = parse_kp(path)
    except Exception as e:
        errors.append(f"{path.name}: parse error: {e}")
        return
    name = path.name
    if kp["kc"] not in kc_ids:
        errors.append(f"{name}: kc '{kp['kc']}' not in registry")
    for s in kp["supporting"]:
        if s not in kc_ids:
            errors.append(f"{name}: supporting kc '{s}' not in registry")
    if kp["concepts"]:
        if len(kp["concepts"]) != len(kp["segments"]):
            errors.append(
                f"{name}: frontmatter concepts declares {len(kp['concepts'])} "
                f"atomic concepts but file has {len(kp['segments'])} segments")
        if len(set(kp["concepts"])) != len(kp["concepts"]):
            errors.append(f"{name}: frontmatter concepts must be unique")
        for si, seg in enumerate(kp["segments"]):
            if not seg["title"]:
                errors.append(
                    f"{name}: declared concept {kp['concepts'][si] if si < len(kp['concepts']) else si + 1} "
                    "needs a titled '## Concept: ...' segment")
    for sec in ("Concept", "Worked example"):
        if not kp["sections"].get(sec):
            errors.append(f"{name}: empty/missing '## {sec}'")

    # 3. executable prose/worked-example code — run fences in DOCUMENT order
    # within each segment, against a namespace that is FRESH PER SEGMENT.
    #
    # This mirrors what the learner gets. practice/notebook.js turns every one
    # of these fences into a runnable notebook cell, and the lesson screen shows
    # ONE SEGMENT PER PAGE — so the only cells in scope when someone presses Run
    # are that segment's. A fence that quietly depended on a name defined in an
    # earlier segment would pass a KP-wide namespace here and hand the learner a
    # NameError, which is the failure this check exists to make impossible.
    #
    # Practically it costs nothing: every segment already opens with its own
    # import (122 of 122 passed the day this was tightened).
    for si, seg in enumerate(kp["segments"]):
        ns = {}
        for label, text in (("Concept", seg["concept"]), ("Worked example", seg["worked"])):
            for code in code_fences(text, "python"):
                try:
                    run_code(code, ns)
                except Exception:
                    tb = traceback.format_exc().strip().splitlines()[-1]
                    errors.append(f"{name}: segment {si + 1} [{label}] fence failed: {tb}")

    # 4. faded solutions pass bank tests; every segment teaches ONE concept
    # then pairs one worked example with a fading SERIES of one or two items.
    #
    # Two, not one, because of what audit_ladder_pairing.py measures: a first
    # completion item sitting adjacent to the example is correct fading, but a
    # series that never grows past it is transcription, and the ladder promotes
    # on that. The second item is where the distance lives. Two is the ceiling
    # on purpose — a segment teaches one concept, and a third completion of the
    # same concept is drill, which is what the independent rung is for.
    faded_ids = set()
    for si, seg in enumerate(kp["segments"]):
        seg_label = f"segment {si + 1}" + (f" ({seg['title']})" if seg["title"] else "")
        if not seg["concept"]:
            errors.append(f"{name}: {seg_label} has an empty '## Concept'")
        if not seg["worked"]:
            errors.append(f"{name}: {seg_label} has no worked example")
        elif len(code_fences(seg["worked"], "python")) != 1:
            errors.append(f"{name}: {seg_label} must have exactly one Python worked example")
        items = split_items(seg["faded"])
        if not 1 <= len(items) <= 2:
            errors.append(f"{name}: {seg_label} must have one or two faded exercises")
        for qid, content in items.items():
            if qid in faded_ids:
                errors.append(f"{name}: faded q{qid} appears in more than one segment")
            faded_ids.add(qid)
            if qid not in bank:
                errors.append(f"{name}: faded q{qid} not in bank")
                continue
            starters = code_fences(content, "python starter")
            solutions = code_fences(content, "python solution")
            if not starters or not solutions:
                errors.append(f"{name}: faded q{qid} missing starter/solution fence")
                continue
            for fail in grade_against_bank(solutions[0], bank[qid]):
                errors.append(f"{name}: faded q{qid} solution FAILED {fail}")

    # 5. refs exist and are consistent
    if set(kp["faded"]) != faded_ids:
        errors.append(f"{name}: frontmatter faded {sorted(kp['faded'])} != sections {sorted(faded_ids)}")
    guided_ids = set(split_items(kp["sections"].get("Guided practice", "")).keys())
    if set(kp["guided"]) != guided_ids:
        errors.append(f"{name}: frontmatter guided {sorted(kp['guided'])} != sections {sorted(guided_ids)}")
    for qid in list(kp["guided"]) + list(kp["independent"]):
        if qid not in bank:
            errors.append(f"{name}: referenced q{qid} not in bank")
    return kp


def check_registry(registry, errors):
    kc_ids = {kc["id"] for kc in registry["kcs"]}
    lesson_ids = {l["id"] for l in registry["lessons"]}
    graph = {}
    for kc in registry["kcs"]:
        if kc["lesson"] not in lesson_ids:
            errors.append(f"registry: kc {kc['id']} has unknown lesson {kc['lesson']}")
        for p in kc["prereqs"]:
            if p not in kc_ids:
                errors.append(f"registry: kc {kc['id']} has unknown prereq {p}")
        graph[kc["id"]] = list(kc["prereqs"])
    # cycle check (iterative DFS)
    WHITE, GRAY, BLACK = 0, 1, 2
    color = dict.fromkeys(graph, WHITE)
    for start in graph:
        if color[start] != WHITE:
            continue
        stack = [(start, iter(graph[start]))]
        color[start] = GRAY
        while stack:
            node, it = stack[-1]
            for nxt in it:
                if color.get(nxt, BLACK) == GRAY:
                    errors.append(f"registry: prereq cycle through {nxt}")
                elif color.get(nxt) == WHITE:
                    color[nxt] = GRAY
                    stack.append((nxt, iter(graph[nxt])))
                    break
            else:
                color[node] = BLACK
                stack.pop()


def check_coverage(registry, bank, kps, errors):
    kp_kcs = [kp["kc"] for kp in kps if kp]
    dupes = {k for k in kp_kcs if kp_kcs.count(k) > 1}
    for d in dupes:
        errors.append(f"coverage: kc {d} has multiple KP files")
    missing = {kc["id"] for kc in registry["kcs"]} - set(kp_kcs)
    for m in sorted(missing):
        errors.append(f"coverage: kc {m} has no KP file")
    tags_path = LESSONS_DIR / "qmatrix_tags.json"
    if not tags_path.exists():
        errors.append("coverage: qmatrix_tags.json missing")
        return
    tags = json.loads(tags_path.read_text())
    kc_ids = {kc["id"] for kc in registry["kcs"]}
    easy_ids = {qid for qid, q in bank.items() if q["curriculum"]["topic"] in EASY_TOPICS}
    untagged = easy_ids - {int(k) for k in tags}
    if untagged:
        errors.append(f"coverage: {len(untagged)} easy questions untagged (e.g. {sorted(untagged)[:8]})")
    for qid, t in tags.items():
        for kc in t.get("target_kcs", []) + t.get("supporting_kcs", []):
            if kc not in kc_ids:
                errors.append(f"coverage: q{qid} tagged with unknown kc {kc}")
        if not t.get("target_kcs"):
            errors.append(f"coverage: q{qid} has no target_kcs")


def main(argv):
    coverage = "--coverage" in argv
    files = [Path(a) for a in argv if a.endswith(".md")]
    registry = load_registry()
    bank = load_bank()
    errors = []
    check_registry(registry, errors)
    paths = files or all_kp_paths()
    kps = [check_kp(p, registry, bank, errors) for p in paths]
    if coverage:
        check_coverage(registry, bank, kps, errors)
    if errors:
        print(f"FAIL — {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        return 1
    n = len(paths)
    print(f"PASS — {n} KP file(s) validated" + (" + coverage" if coverage else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
