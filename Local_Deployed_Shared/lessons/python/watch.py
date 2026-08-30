"""watch.py — health checks for lessons/python (the py-0 prerequisite floor).

py-0 is the seven concepts that sit BELOW numpy.ndarray-model: values and
names, types, lists and tuples, indexing, calling functions, defining
functions, and dots/imports. It exists so the first thing a learner reads has
something prior to it — before 2026-08-28 the course had exactly one root,
numpy.ndarray-model, and every piece of jargon on that page pointed FORWARD
into lessons the learner had not reached.

Every way this floor breaks is quiet, which is what the checks below are for:

  - a page here reaching for torch/numpy/einops teaches above its own level,
    and nothing else in the pipeline would notice;
  - the floor drifting out from under np-1 (numpy.ndarray-model losing its
    python prereqs) puts the root back where it was, silently;
  - 🔴 wiring a python ATOM as a prerequisite of tensor-wraps-ndarray locks
    every existing account out of the whole numpy course. bkt_mastery's
    item_is_unlocked needs EVERY gating prerequisite atom ready, `is_hard_gate`
    is not read at runtime, and no learner has python evidence yet. This was
    tried, measured, and reverted the same hour; the check is the memory of it.

Runs via `mod watch` — exit 0 = PASS, exit non-zero = FAIL.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)

LESSON_ID = "py-0"
ROOT_KC = "python.values-and-names"


def _registry():
    with open(os.path.join(HERE, "..", "kc_registry.json"), encoding="utf-8") as f:
        return json.load(f)


def _pages():
    return sorted(n for n in os.listdir(HERE) if n.startswith("kp-") and n.endswith(".md"))


def _frontmatter(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        text = f.read()
    head = text.split("---", 2)[1]
    return head, text


def check_every_page_is_a_registered_py0_concept():
    """Each kp-*.md here declares a python.* KC that the registry files under py-0."""
    reg = _registry()
    kcs = {kc["id"]: kc for kc in reg["kcs"]}
    assert any(l["id"] == LESSON_ID for l in reg["lessons"]), \
        "kc_registry.json has no %s lesson — the floor has no shelf to sit on" % LESSON_ID
    pages = _pages()
    assert pages, "lessons/python holds no KP pages at all"
    declared = set()
    for name in pages:
        head, _ = _frontmatter(name)
        m = re.search(r"^kc:\s*(\S+)", head, re.M)
        assert m, "%s has no kc: in its frontmatter" % name
        kc = m.group(1)
        assert kc.startswith("python."), \
            "%s declares %s — pages in lessons/python teach the python.* floor" % (name, kc)
        assert kc in kcs, "%s declares %s, which kc_registry.json does not hold" % (name, kc)
        assert kcs[kc]["lesson"] == LESSON_ID, \
            "%s is filed under lesson %s, not %s" % (kc, kcs[kc]["lesson"], LESSON_ID)
        declared.add(kc)
    registered = {k for k, v in kcs.items() if v["lesson"] == LESSON_ID}
    missing = sorted(registered - declared)
    assert not missing, \
        "registry files %s under %s but no page teaches them" % (", ".join(missing), LESSON_ID)


def check_the_floor_stays_under_the_numpy_course():
    """One root, and it is the python one; np-1's first concept still stands on py-0."""
    kcs = {kc["id"]: kc for kc in _registry()["kcs"]}
    roots = sorted(k for k, v in kcs.items() if not v["prereqs"])
    assert roots == [ROOT_KC], (
        "the course's root concept(s) are %s. There must be exactly one, and it "
        "must be %s — a second root is a lesson with nothing prior to it, which "
        "is the state this whole folder exists to end" % (roots, ROOT_KC))
    first_numpy = kcs.get("numpy.ndarray-model")
    assert first_numpy, "numpy.ndarray-model is gone from the registry"
    prereqs = first_numpy["prereqs"]
    assert prereqs and all(p.startswith("python.") for p in prereqs), (
        "numpy.ndarray-model's prereqs are %s — the floor has drifted out from "
        "under it and the first numpy page is a root again" % prereqs)


def check_pages_teach_plain_python():
    """No page here reaches for a library. The floor is what comes BEFORE them."""
    banned = re.compile(r"\b(import\s+torch|torch\.|import\s+numpy|np\.|einops|einsum)\b")
    for name in _pages():
        _, text = _frontmatter(name)
        body = re.sub(r"^---.*?^---", "", text, flags=re.S | re.M)
        hit = banned.search(body)
        assert not hit, (
            "%s mentions %r. py-0 runs before any library exists for the learner; "
            "a page here that reaches for one is teaching above its own level"
            % (name, hit.group(0)))


def check_no_python_atom_gates_the_numpy_floor():
    """🔴 The lock trap. A python atom wired into tensor-wraps-ndarray would
    lock EVERY existing account out of the numpy course: item_is_unlocked wants
    every gating prerequisite atom ready, is_hard_gate is not read at runtime,
    and nobody has python evidence. Tried and reverted 2026-08-28."""
    path = os.path.join(REPO, "This-Directory-Only", "backend", "app", "data",
                        "concept_graphs", "arena_drillable_v1.json")
    with open(path, encoding="utf-8") as f:
        graph = json.load(f)
    bad = [e for e in graph.get("prerequisite_edges", [])
           if str(e.get("prerequisite_id", "")).startswith("python-")
           and not str(e.get("dependent_id", "")).startswith("python-")]
    assert not bad, (
        "%d atom edge(s) gate non-python atoms behind the python floor (%s). "
        "That locks the numpy course for every account with no python evidence "
        "— which is all of them. Keep the floor's ordering in kc_registry "
        "prereqs, not in the BKT lattice"
        % (len(bad), ", ".join("%s -> %s" % (e["prerequisite_id"], e["dependent_id"])
                               for e in bad[:3])))
    ids = {c["id"] for c in graph.get("concepts", [])}
    roots = set(graph.get("intentional_root_atoms", []))
    for atom in sorted(a for a in ids if a.startswith("python-")):
        gated = any(e["dependent_id"] == atom for e in graph["prerequisite_edges"])
        assert gated or atom in roots, (
            "atom %s has no incoming prerequisite edge and is not declared in "
            "intentional_root_atoms — audit_question_bank flags that as "
            "atom_tag_unwired and the gate refuses the deploy" % atom)



# ── The two standing content guards ───────────
# Filled 2026-08-29 on Seth's instruction. This folder is where the guards bite
# hardest: py-0 is the FLOOR, so a drill here reaching for an untaught
# construct has nothing earlier to be taught by. The three open cases recorded
# in README.md (q578 membership, q585 generator expression + set, q603 an `if`)
# are exactly this rule, found by hand before it was mechanised.
# ARENA grounding is scoped to `python.` too, and is expected to be silent
# here: the floor teaches no library at all, and a python page that starts
# declaring torch symbols has stopped being the floor.
import importlib.util as _ilu
_GUARD = os.path.join(REPO, 'scripts', 'guard_checks.py')
_spec = _ilu.spec_from_file_location('guard_checks', _GUARD)
_guard = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_guard)

_CONTENT_GUARDS = _guard.run('python.')

if __name__ == '__main__':
    checks = [
        check_every_page_is_a_registered_py0_concept,
        check_the_floor_stays_under_the_numpy_course,
        check_pages_teach_plain_python,
        check_no_python_atom_gates_the_numpy_floor,
    ] + _CONTENT_GUARDS
    for fn in checks:
        try:
            fn()
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}", file=sys.stderr)
            sys.exit(1)
