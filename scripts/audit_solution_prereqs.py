#!/usr/bin/env python3
"""Every call in a drill's SOLUTION must have a lesson before that drill.

The rule, stated the way it is enforced
---------------------------------------
A drill is served for one concept. Its solution is what the learner has to
produce. So every symbol that solution uses — every function, every method,
every attribute, every language construct — must be declared by the
`new_syntax:` of a KP page whose KC sits **at or before** the drill's own KC in
the prerequisite lattice. A symbol no page declares, or one first declared
after the drill, means the drill asks for something the graph has not taught.

`a.T` is the case that motivated this. It appeared in eight faded drills on
`numpy.ndarray-model` — the first concept of the course — while the only page
that mentioned transposition sat four lessons later. Nothing failed. A learner
hit it on their second day and had to go ask a different model what the
question meant.

Why it is not `audit_question_syntax.py`
----------------------------------------
That audit asks the same question of three surfaces, and is deliberately
generous about what counts as syntax: its `ASSUMED` set exempts `len`, `for`,
comprehensions, comparison and a dozen more as "the audience already writes
plain Python". That was true before the course had a Python floor. It is not
true now — `py-0` teaches names, types, lists, indexing and functions, so a
drill reaching for a set comprehension is reaching past its own curriculum in
exactly the way `a.T` did. This audit exempts NOTHING: every construct is
looked up, and the ones the old audit would have waved through are still
reported, tagged `assumed`, so the two numbers can be compared.

Baseline
--------
The corpus has a large existing backlog. Shipping this as a hard failure would
paint the build red and teach everyone to ignore it, so the check is a
RATCHET: `scripts/solution_prereq_baseline.json` records today's violations,
`watch.py` fails on anything not in it, and the file is meant to shrink.
`--write-baseline` re-records. Re-recording is a content decision — it is how
new debt gets admitted, so it should show up in a diff and be argued for.

Usage
-----
    python3 scripts/audit_solution_prereqs.py                  # full report
    python3 scripts/audit_solution_prereqs.py --summary        # counts
    python3 scripts/audit_solution_prereqs.py --by-symbol      # worst symbols
    python3 scripts/audit_solution_prereqs.py --qid 535
    python3 scripts/audit_solution_prereqs.py --new            # vs baseline
    python3 scripts/audit_solution_prereqs.py --write-baseline
Exit 1 when anything is reported (or, with --new, when anything is NEW).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_lesson_syntax import ASSUMED, LESSONS, lesson_order, page_symbols  # noqa: E402
from solution_symbols import aliases, bound_names, collect  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "Local_Deployed_Shared"
QUESTIONS = SHARED / "questions.json"
QMATRIX = SHARED / "lessons" / "qmatrix_tags.json"
BASELINE = Path(__file__).resolve().parent / "solution_prereq_baseline.json"

# The learner writes `solve`; the bank's own scaffold names the fixture and the
# result. Counting those as untaught API was the one false positive this audit
# produced on every single question.
SELF_DEFINED = {"builtin.solve", "builtin.example", "builtin.result"}

SURFACES = ("solution", "starter", "prompt")


def declaring_kcs() -> tuple[dict[str, str], dict[str, str]]:
    """(symbol -> the EARLIEST KC that declares it, page filename -> its KC).

    Earliest by lattice rank, not by filename: two pages may declare the same
    symbol, and taking whichever sorted first read `Tensor.sum` as taught by a
    page four lessons after the one that actually introduces it.
    """
    declared_all: dict[str, list[str]] = {}
    kc_of_page: dict[str, str] = {}
    for path in sorted(LESSONS.rglob("kp-*.md")):
        fm, _used, _errs = page_symbols(path)
        kc = fm.get("kc")
        if not kc:
            continue
        kc_of_page[path.name] = kc
        for sym in fm.get("new_syntax") or []:
            declared_all.setdefault(sym, []).append(kc)
    rank = lesson_order(kc_of_page)
    declared = {sym: min(kcs, key=lambda k: rank.get(k, 10**6))
                for sym, kcs in declared_all.items()}
    return declared, kc_of_page


def question_symbols(q: dict, surface: str) -> set[str]:
    """Symbols one surface of one question requires."""
    known: set[str] = set()
    for case in q.get("test_cases") or []:
        known |= bound_names(case.get("setup_code") or "")
    if surface == "solution":
        return collect(q.get("answer_code") or "", known)
    if surface == "starter":
        return collect(q.get("starter_code") or "", known)
    if surface == "prompt":
        import re
        out: set[str] = set()
        for span in re.findall(r"`([^`\n]+)`", q.get("question_text") or ""):
            out |= collect(span.strip(), known)
        return out
    raise ValueError(surface)


def owner_of(sym: str, declared: dict[str, str],
              rank: dict[str, int] | None = None) -> str | None:
    """The KC that teaches `sym`, under any spelling a page may have used."""
    for cand in aliases(sym):
        if cand in declared:
            return declared[cand]
    # `torch.arange#dtype` with no page declaring the kwarg still has a home if
    # the call itself is taught; report it against that page rather than as
    # "nobody teaches this", which would be false.
    base = sym.split("#")[0]
    if base != sym:
        for cand in aliases(base):
            if cand in declared:
                return declared[cand]
    # `import.einops` is not a symbol anyone declares, and demanding that they
    # do would be paperwork: a page that teaches `einops.rearrange` cannot show
    # it without the import. So the import is owned by the earliest page that
    # teaches anything from that module.
    if sym.startswith("import."):
        mod = sym[len("import."):].split(".")[0]
        owners = [kc for s2, kc in declared.items() if s2.startswith(mod + ".")]
        if owners:
            return min(owners, key=lambda k: (rank or {}).get(k, 0))
    return None


def find(surfaces: tuple[str, ...], only_qid: int | None = None) -> list[dict]:
    bank = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    questions = bank if isinstance(bank, list) else bank.get("questions", bank)
    qmatrix = json.loads(QMATRIX.read_text(encoding="utf-8"))
    declared, kc_of_page = declaring_kcs()
    rank = lesson_order(kc_of_page)

    out: list[dict] = []
    for q in questions:
        qid = q.get("id")
        if only_qid is not None and qid != only_qid:
            continue
        targets = (qmatrix.get(str(qid)) or {}).get("target_kcs") or []
        ranked = [(rank[k], k) for k in targets if k in rank]
        if not ranked:
            continue
        my_rank, my_kc = min(ranked)
        for surface in surfaces:
            for sym in sorted(question_symbols(q, surface)):
                if sym in SELF_DEFINED:
                    continue
                owner = owner_of(sym, declared, rank)
                if owner is None:
                    kind = "unowned"
                elif rank.get(owner, -1) > my_rank:
                    kind = "late"
                else:
                    continue
                out.append({
                    "qid": qid, "kc": my_kc, "surface": surface, "symbol": sym,
                    "kind": kind, "owner": owner,
                    "assumed": sym in ASSUMED,
                    "topic": q.get("topic"),
                })
    return out


def key(v: dict) -> str:
    return f"{v['qid']}|{v['surface']}|{v['symbol']}"


def load_baseline() -> set[str] | None:
    """Known debt, or None when the file is absent.

    None and empty are different answers: an empty `known` list is the state
    this audit is trying to reach, while a missing file means the ratchet has
    nothing to compare against and must not report a clean bill.
    """
    if not BASELINE.exists():
        return None
    return set(json.loads(BASELINE.read_text(encoding="utf-8")).get("known") or [])


def write_baseline(violations: list[dict]) -> None:
    payload = {
        "_": "Known solution/prereq violations. watch.py fails on anything NOT "
             "listed here. Shrink it; re-record only with a reason.",
        "count": len(violations),
        "known": sorted({key(v) for v in violations}),
    }
    BASELINE.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def report(violations: list[dict], args) -> int:
    known = load_baseline() or set()
    new = [v for v in violations if key(v) not in known]
    stale = known - {key(v) for v in violations}
    shown = new if args.new else violations

    if not args.summary and not args.by_symbol:
        for v in sorted(shown, key=lambda v: (v["qid"], v["surface"], v["symbol"])):
            tag = "ASSUMED " if v["assumed"] else ""
            where = f"first taught by {v['owner']}" if v["kind"] == "late" \
                else "taught by NO lesson"
            print(f"q{v['qid']:<4} [{v['kc']}] {v['surface']:8s} {tag}{v['symbol']} — {where}")
        print()

    if args.by_symbol:
        per = Counter(v["symbol"] for v in shown)
        owners = {v["symbol"]: (v["kind"], v["owner"]) for v in shown}
        for sym, n in per.most_common():
            kind, owner = owners[sym]
            where = f"late — {owner}" if kind == "late" else "UNOWNED"
            flag = " (ASSUMED)" if sym in ASSUMED else ""
            print(f"{n:5d}  {sym:38s} {where}{flag}")
        print()

    by_kind = Counter(v["kind"] for v in violations)
    by_surface = Counter(v["surface"] for v in violations)
    per_kc = Counter(v["kc"] for v in violations)
    print(f"violations      : {len(violations)}  "
          f"(unowned {by_kind['unowned']}, late {by_kind['late']})")
    print(f"  of which the old ASSUMED list would have hidden: "
          f"{sum(1 for v in violations if v['assumed'])}")
    for s in SURFACES:
        if s in by_surface:
            print(f"  {s:9s}: {by_surface[s]}")
    print(f"questions affected: {len({v['qid'] for v in violations})}")
    print(f"distinct symbols  : {len({v['symbol'] for v in violations})}")
    print(f"worst concepts    : "
          + ", ".join(f"{k}={n}" for k, n in per_kc.most_common(5)))
    if args.kc_prefix or args.qid is not None:
        print(f"baseline          : {len(new)} NEW (filtered run — stale not meaningful)")
    else:
        print(f"baseline          : {len(known)} known, {len(new)} NEW, {len(stale)} stale")
    if stale and not args.new and not args.kc_prefix and args.qid is None:
        print("  (stale = fixed since the baseline was written; re-record it)")
    if args.new:
        return 1 if new else 0
    return 1 if violations else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # All three by default since 2026-08-29. The rule Seth stated is "every
    # function utilised in the solution OR THE PROBLEM", and the problem is
    # what the learner is handed: the starter they type into and the prompt
    # they read. A starter is not merely a faded solution — at the worked rung
    # it IS the code, and it carries scaffold lines the solution never shows.
    ap.add_argument("--surface", default=",".join(SURFACES),
                    help="comma-separated: solution,starter,prompt")
    ap.add_argument("--qid", type=int, default=None)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--by-symbol", action="store_true")
    ap.add_argument("--kc-prefix", default=None,
                    help="only drills whose concept starts with this "
                         "(e.g. numpy. einops. einsum., or python. for the floor)")
    ap.add_argument("--new", action="store_true", help="only what the baseline does not know")
    ap.add_argument("--write-baseline", action="store_true")
    args = ap.parse_args()

    surfaces = tuple(s.strip() for s in args.surface.split(",") if s.strip())
    for s in surfaces:
        if s not in SURFACES:
            raise SystemExit(f"unknown surface {s!r}; pick from {SURFACES}")

    violations = find(surfaces, args.qid)
    if args.kc_prefix:
        pre = tuple(p.strip() for p in args.kc_prefix.split(",") if p.strip())
        violations = [v for v in violations if v["kc"].startswith(pre)]
    if args.write_baseline:
        if args.qid is not None or tuple(surfaces) != SURFACES:
            raise SystemExit("--write-baseline records every surface of the "
                             "whole bank; drop --qid/--surface")
        write_baseline(violations)
        print(f"baseline written: {len({key(v) for v in violations})} entries")
        return 0
    return report(violations, args)


if __name__ == "__main__":
    raise SystemExit(main())
