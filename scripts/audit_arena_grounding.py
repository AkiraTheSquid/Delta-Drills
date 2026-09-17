#!/usr/bin/env python3
"""Every function we teach or drill must be a function ARENA actually uses.

The rule, stated the way it is enforced
---------------------------------------
The course exists to prepare people for ARENA. So ARENA is the source of
truth for what is worth teaching, and the test is empirical, not editorial: a
library symbol that appears in **zero** of the 458 ARENA exercise notebooks is
something no learner will ever need, and putting it in the graph spends a
learner's attention on it anyway.

Two surfaces are checked, because they fail differently:

* **declarations** — a KP page's `new_syntax:` frontmatter. This is the graph
  saying "this is a thing to learn". An ungrounded declaration is a dead node:
  it takes lattice space, gates other concepts, and gets drilled forever.
* **drills** — the symbols a question's solution and starter actually use. A
  drill can reach for an ungrounded symbol even when its page is clean, and
  that is the sneakier one: the learner spends the attempt on it regardless of
  what the page claims to teach.

`torch.einsum` is the case that motivated this. Our whole einsum course is
written in it — 69 drill solutions — and the entire ARENA corpus does not
contain a single notebook that uses it. ARENA writes `einops.einsum`, which
takes its arguments in a different order. Nothing failed; the audit that found
it did not exist yet.

Spellings are FOLDED before the lookup
--------------------------------------
`torch.argmax(x)` and `x.argmax()` are one operation to learn. ARENA
overwhelmingly writes the method where our pages declare the function, so
comparing raw spellings reported half the curriculum as dead weight. The fold
is the same one `audit_arena_frequency.operation()` applies, using the member
list frozen into the index — recomputing it here would need torch, and a
reader that folded differently from the audit that produced the numbers would
answer a different question than the one it appears to answer.

Where the numbers come from
---------------------------
`scripts/arena_symbol_index.json`, written by
`audit_arena_frequency.py --write-index`. The scan needs torch and the
vendored corpus; this audit needs neither, which is what lets the fast
structural watchers run it. `--check-index` verifies the frozen file still
describes the corpus on disk.

Baseline
--------
There is an existing backlog — 42% of the operations we teach are ungrounded
— so this ships as a RATCHET, exactly like the prerequisite audit:
`scripts/arena_grounding_baseline.json` records today's debt, the watchers
fail on anything NOT in it, and the file is meant to shrink. Re-recording is
admitting debt, not fixing it, and it shows up in a diff.

Usage
-----
    python3 scripts/audit_arena_grounding.py                  # full report
    python3 scripts/audit_arena_grounding.py --summary
    python3 scripts/audit_arena_grounding.py --surface declarations
    python3 scripts/audit_arena_grounding.py --kc-prefix einsum.,einops.
    python3 scripts/audit_arena_grounding.py --new
    python3 scripts/audit_arena_grounding.py --write-baseline
    python3 scripts/audit_arena_grounding.py --check-index
Exit 1 when anything is reported (or, with --new, when anything is NEW).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_lesson_syntax import lesson_order  # noqa: E402
from audit_solution_prereqs import (  # noqa: E402
    QMATRIX, QUESTIONS, declaring_kcs, question_symbols,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
INDEX = HERE / "arena_symbol_index.json"
BASELINE = HERE / "arena_grounding_baseline.json"

# The prompt is part of the PROBLEM the learner is handed, so the grounding
# rule reaches it for the same reason the prerequisite rule does: a prompt that
# names an API nobody uses still sends the learner to look it up.
SURFACES = ("declarations", "solution", "starter", "prompt")

# Environment setup, not curriculum: the device/precision preamble every ARENA
# file opens with. Grounded by construction (it is in nearly every notebook)
# and listed here only so the report can say so.
_ENV_NOTE = "boilerplate"


def corpus_fingerprint(root: Path, excluded: str) -> str:
    """A hash of the corpus an index was built from.

    Counting notebooks is not enough to know a measurement is still valid: a
    notebook can be rewritten, replaced or renamed without the count moving,
    and the frozen frequencies then describe a corpus nobody has. Path and byte
    length per file catch every one of those — a content edit that preserves
    the exact byte count of every file it touches is not a failure mode anyone
    has. The bytes themselves are never read; this runs on every edit.

    It lives HERE rather than beside the scan that uses it because
    `audit_arena_frequency` exits at import time without torch, and the
    watchers that check freshness run under bare python3. One definition, on
    the side that can always be loaded.
    """
    h = hashlib.sha256()
    for p in sorted(root.rglob("*.ipynb")):
        if excluded in p.parts:
            continue
        h.update(str(p.relative_to(root)).encode("utf-8"))
        h.update(b"\0")
        h.update(str(p.stat().st_size).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def load_index() -> dict:
    if not INDEX.exists():
        raise SystemExit(
            f"{INDEX.name} is missing — the ARENA grounding guard has no corpus "
            "to check against. Regenerate it with "
            "This-Directory-Only/backend/.venv/bin/python "
            "scripts/audit_arena_frequency.py --write-index")
    return json.loads(INDEX.read_text(encoding="utf-8"))


class Corpus:
    """The frozen ARENA measurement, with the fold applied on lookup."""

    def __init__(self, index: dict):
        self.notebooks = index["notebooks"]
        self.operation_df = index["operation_df"]
        self.raw_df = index["raw_df"]
        self.members = set(index["tensor_members"])
        self.ambiguous = set(index["ambiguous_members"])
        self.boilerplate = set(index["boilerplate"])

    def is_library(self, sym: str) -> bool:
        """Symbols this audit has an opinion about: library API, not language.

        `syntax.for` and `builtin.len` are Python. ARENA obviously uses them,
        and asking the corpus about them would only add noise — the
        prerequisite audit is what governs those.
        """
        base = sym.split("#")[0]
        if base.startswith("Tensor."):
            return base[len("Tensor."):] in self.members
        return base.startswith(("torch.", "einops.", "numpy.", "np."))

    def fold(self, sym: str) -> str:
        base = sym.split("#")[0]
        for prefix in ("torch.", "Tensor."):
            if base.startswith(prefix):
                tail = base[len(prefix):]
                if "." not in tail and tail in self.members:
                    return "op." + tail
        return base

    def df(self, sym: str) -> int:
        """Notebooks containing this operation, in either spelling."""
        return int(self.operation_df.get(self.fold(sym), 0))

    def note(self, sym: str) -> str:
        base = sym.split("#")[0]
        if base in self.boilerplate:
            return _ENV_NOTE
        if base.startswith("Tensor.") and base[len("Tensor."):] in self.ambiguous:
            # The count is an upper bound: a static pass cannot tell `x.split()`
            # on a tensor from `s.split()` on a string. Erring toward "grounded"
            # is deliberate — this guard should flag clear dead weight, not
            # argue with a name collision.
            return "name shared with str/dict/list/Path — DF is an upper bound"
        return ""


def declaration_violations(corpus: Corpus, rank: dict[str, int]) -> list[dict]:
    """KP pages declaring a symbol ARENA never uses."""
    declared, kc_of_page = declaring_kcs()
    out = []
    for sym, kc in sorted(declared.items()):
        if not corpus.is_library(sym):
            continue
        if corpus.df(sym) > 0:
            continue
        out.append({
            "surface": "declarations", "kc": kc, "symbol": sym,
            "folded": corpus.fold(sym), "df": 0, "qid": None,
            "rank": rank.get(kc, -1), "note": corpus.note(sym),
        })
    return out


def drill_violations(corpus: Corpus, surfaces: tuple[str, ...],
                     only_qid: int | None = None) -> list[dict]:
    """Questions whose solution or starter uses a symbol ARENA never uses."""
    bank = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    questions = bank if isinstance(bank, list) else bank.get("questions", bank)
    qmatrix = json.loads(QMATRIX.read_text(encoding="utf-8"))

    out = []
    for q in questions:
        qid = q.get("id")
        if only_qid is not None and int(qid) != only_qid:
            continue
        targets = (qmatrix.get(str(qid)) or {}).get("target_kcs") or []
        for surface in surfaces:
            if surface == "declarations":
                continue
            for sym in sorted(question_symbols(q, surface)):
                if not corpus.is_library(sym) or corpus.df(sym) > 0:
                    continue
                out.append({
                    "surface": surface, "kc": targets[0] if targets else "",
                    "symbol": sym, "folded": corpus.fold(sym), "df": 0,
                    "qid": qid, "rank": -1, "note": corpus.note(sym),
                })
    return out


def find(surfaces: tuple[str, ...], only_qid: int | None = None) -> list[dict]:
    corpus = Corpus(load_index())
    _, kc_of_page = declaring_kcs()
    rank = lesson_order(kc_of_page)
    out = []
    if "declarations" in surfaces and only_qid is None:
        out += declaration_violations(corpus, rank)
    out += drill_violations(corpus, surfaces, only_qid)
    return out


def key(v: dict) -> str:
    """Stable identity for the baseline.

    A declaration is keyed by its CONCEPT, a drill by its question id. Keying a
    declaration by page filename would let a rename launder the debt.
    """
    if v["surface"] == "declarations":
        return f"decl|{v['kc']}|{v['symbol']}"
    return f"{v['qid']}|{v['surface']}|{v['symbol']}"


def load_baseline() -> set[str] | None:
    """Known debt, or None when the file is ABSENT.

    None and the empty set are different answers: an empty file means the debt
    is genuinely zero, a missing file means the ratchet has nothing to compare
    against and must say so rather than pass.
    """
    if not BASELINE.exists():
        return None
    data = json.loads(BASELINE.read_text(encoding="utf-8"))
    return set(data.get("violations") or [])


def write_baseline(violations: list[dict]) -> None:
    BASELINE.write_text(json.dumps({
        "_": "Known ARENA-grounding violations: symbols we teach or drill that "
             "appear in ZERO of the 458 ARENA notebooks. watch.py fails on "
             "anything NOT listed here. Shrink it; re-recording is admitting "
             "the debt, not fixing it, so say why in the commit.",
        "surfaces": list(SURFACES),
        "count": len(violations),
        "violations": sorted({key(v) for v in violations}),
    }, indent=1) + "\n", encoding="utf-8")


def check_index_fresh() -> None:
    """The frozen index must still describe the corpus on disk.

    A guard whose evidence has silently gone stale is worse than no guard: it
    keeps answering, from a corpus that no longer exists. This is the cheap
    half of the check — notebook COUNT — because recomputing the symbols needs
    torch. It catches the case that actually happens: the vendored corpus is
    updated, or moved, and nobody re-ran the scan.
    """
    index = load_index()
    root = ROOT / index["corpus_root"]
    excluded = index["excluded_dir"]
    if not root.exists():
        raise AssertionError(
            f"{index['corpus_root']} is not on disk — the corpus the grounding "
            "guard measures has moved or was never checked out")
    on_disk = sum(1 for p in root.rglob("*.ipynb") if excluded not in p.parts)
    if on_disk != index["notebooks"]:
        raise AssertionError(
            f"arena_symbol_index.json was built from {index['notebooks']} "
            f"notebooks but {on_disk} are on disk — the grounding guard is "
            "answering from a corpus that no longer exists. Re-run: "
            "This-Directory-Only/backend/.venv/bin/python "
            "scripts/audit_arena_frequency.py --write-index")

    # Count alone would pass a corpus whose notebooks were rewritten or
    # swapped, which is the same lie in a quieter form. The fingerprint is
    # path + byte length per file, so it moves on any of those.
    frozen = index.get("corpus_fingerprint")
    if frozen:
        if corpus_fingerprint(root, excluded) != frozen:
            raise AssertionError(
                "the ARENA corpus has CHANGED since arena_symbol_index.json "
                "was written — same notebook count, different contents. The "
                "frozen frequencies no longer describe it. Re-run: "
                "This-Directory-Only/backend/.venv/bin/python "
                "scripts/audit_arena_frequency.py --write-index")


def report(violations: list[dict], args) -> int:
    corpus = Corpus(load_index())
    if args.kc_prefix:
        prefixes = tuple(p.strip() for p in args.kc_prefix.split(",") if p.strip())
        violations = [v for v in violations if v["kc"].startswith(prefixes)]

    known = load_baseline() or set()
    if args.new:
        violations = [v for v in violations if key(v) not in known]

    by_surface = Counter(v["surface"] for v in violations)
    symbols = Counter(v["symbol"] for v in violations)
    kcs = Counter(v["kc"] for v in violations if v["kc"])

    print(f"ARENA corpus: {corpus.notebooks} notebooks, "
          f"{len(corpus.operation_df)} distinct operations")
    print(f"ungrounded   : {len(violations)}"
          + (f"  ({sum(1 for v in violations if v['note'])} carry a caveat)"
             if violations else ""))
    for s in SURFACES:
        if by_surface.get(s):
            print(f"  {s:13s}: {by_surface[s]}")
    print(f"distinct symbols  : {len(symbols)}")
    if kcs:
        worst = ", ".join(f"{k}={n}" for k, n in kcs.most_common(5))
        print(f"worst concepts    : {worst}")
    base = load_baseline()
    if base is None:
        print("baseline          : ABSENT")
    else:
        allv = {key(v) for v in find(tuple(args.surface.split(",")))}
        print(f"baseline          : {len(base)} known, "
              f"{len(allv - base)} NEW, {len(base - allv)} stale")

    if not args.summary:
        if args.by_symbol:
            print()
            for sym, n in symbols.most_common(args.top):
                note = corpus.note(sym)
                print(f"  {sym:34s} {n:4d} uses" + (f"   [{note}]" if note else ""))
        else:
            print()
            grouped = defaultdict(list)
            for v in violations:
                grouped[(v["surface"], v["kc"])].append(v)
            for (surface, kc), items in sorted(grouped.items()):
                where = f"{surface} / {kc or '(untagged)'}"
                print(f"  {where}")
                for v in sorted(items, key=lambda x: (str(x["qid"]), x["symbol"])):
                    qid = f"q{v['qid']} " if v["qid"] is not None else ""
                    note = f"   [{v['note']}]" if v["note"] else ""
                    print(f"      {qid}{v['symbol']}  (folded {v['folded']}, "
                          f"0 notebooks){note}")

    return 1 if violations else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--surface", default=",".join(SURFACES),
                    help="declarations,solution,starter")
    ap.add_argument("--qid", type=int, default=None)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--by-symbol", action="store_true")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--kc-prefix", default=None,
                    help="restrict to concepts whose id starts with one of "
                         "these, comma separated (einsum.,einops.)")
    ap.add_argument("--new", action="store_true",
                    help="only what the baseline does not know")
    ap.add_argument("--write-baseline", action="store_true")
    ap.add_argument("--check-index", action="store_true",
                    help="verify the frozen index still matches the corpus")
    args = ap.parse_args()

    if args.check_index:
        check_index_fresh()
        print("arena_symbol_index.json matches the corpus on disk")
        return 0

    surfaces = tuple(s.strip() for s in args.surface.split(",") if s.strip())
    unknown = set(surfaces) - set(SURFACES)
    if unknown:
        raise SystemExit(f"unknown surface(s): {sorted(unknown)}")

    violations = find(surfaces, args.qid)
    if args.write_baseline:
        # A subset written over the canonical file would reclassify every
        # violation it omitted as NEW, and the next watcher run would fail on
        # the entire pre-existing backlog. The prerequisite audit refuses the
        # same way.
        if args.qid is not None or surfaces != SURFACES or args.kc_prefix:
            raise SystemExit("--write-baseline records every surface of the "
                             "whole bank; drop --qid/--surface/--kc-prefix")
        write_baseline(violations)
        print(f"baseline written: {len(violations)} violation(s) -> {BASELINE.name}")
        return 0
    return report(violations, args)


if __name__ == "__main__":
    sys.exit(main())
