#!/usr/bin/env python3
"""Every symbol a concept CLAIMS to teach must be drilled on that concept.

The rule, stated the way it is enforced
---------------------------------------
A KP page's `new_syntax:` frontmatter is the graph saying "mastery of this
concept includes mastery of this symbol". The mastery models take that claim
literally: BKT and the logistic engine both estimate one number per concept
and gate the lattice on it. So a symbol that is declared but never drilled on
its own node is a claim the app can never have evidence for — the learner is
marked as having learned it on the strength of drills about something else.

    every symbol in KC K's `new_syntax` must appear in the SOLUTION of at
    least MIN_DRILLS_PER_SYMBOL distinct drills tagged to K.

Two halves of that sentence carry weight.

* **SOLUTION, not starter or prompt.** Evidence is what the learner had to
  write. A starter that hands them the call, or a prompt that names it, is
  the opposite of evidence. (Measured on the current bank: starter and prompt
  add coverage for exactly zero symbols, so this costs nothing and means
  something.)
* **tagged to K.** A drill tagged to a later concept that happens to use the
  symbol does not update K's mastery estimate, so it cannot back K's claim.
  Attributing it there would be counting evidence the model never sees.

What it actually finds
----------------------
Under-drilled symbols cluster on the nodes that were never split — the
"blob" concepts that declare eight or ten symbols and carry three drills.
`numpy.random-generator` declares ten and drills five across three questions;
`numpy.stack-concat-interleave` declares ten and covers none of them twice.
That is the measurement behind `SPEC_NODE_SPLITTING.md`: the fix is both more
drills and fewer claims per node.

Baseline
--------
51 of 144 declared symbols are under the floor today, 19 of them at zero, so
this ships as a RATCHET like the other two guards: `symbol_coverage_baseline.json`
records today's debt and the watchers fail on anything NOT in it.

It records the COUNT, not just the symbol. A symbol sitting at one drill is
already in the baseline; if someone deletes that drill it goes to zero and a
key-only ratchet would stay silent, because the key is still "known". Debt is
allowed to stay where it is. It is not allowed to get worse.

Usage
-----
    python3 scripts/audit_symbol_coverage.py                 # full report
    python3 scripts/audit_symbol_coverage.py --summary
    python3 scripts/audit_symbol_coverage.py --kc-prefix numpy.
    python3 scripts/audit_symbol_coverage.py --new
    python3 scripts/audit_symbol_coverage.py --write-baseline
Exit 1 when anything is reported (or, with --new, when anything is NEW).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_lesson_syntax import lesson_order  # noqa: E402
from audit_solution_prereqs import (  # noqa: E402
    QMATRIX, QUESTIONS, declaring_kcs, question_symbols,
)
from solution_symbols import aliases  # noqa: E402

BASELINE = Path(__file__).resolve().parent / "symbol_coverage_baseline.json"

# Two, not one. One drill is a coin flip the learner can pass by luck or by
# reading the starter, and the ladder promotes on it; two is the smallest
# number that can distinguish "knows it" from "got it once". It is also the
# floor the ladder already assumes elsewhere (Faded >= 2 per segment).
MIN_DRILLS_PER_SYMBOL = 2

# The learner writes the answer. That is the only surface that is evidence.
EVIDENCE_SURFACE = "solution"


def _alias_index(declared: dict[str, str]) -> dict[str, str]:
    """Any spelling a drill might use -> the declared spelling it satisfies.

    A page declares `torch.reshape`; the drill writes `x.reshape(...)`, which
    the symbol collector reports as `Tensor.reshape`. Comparing raw spellings
    would report the symbol as undrilled while the drill for it sits right
    there. `aliases()` is the same fold `audit_solution_prereqs.owner_of` uses
    to decide who teaches a symbol, so coverage and ownership agree by
    construction rather than by coincidence.
    """
    index: dict[str, str] = {}
    for sym in declared:
        for spelling in aliases(sym):
            index.setdefault(spelling, sym)
    return index


def _declared_used(q: dict, index: dict[str, str]) -> set[str]:
    """The declared symbols this question's solution actually exercises."""
    out: set[str] = set()
    for used in question_symbols(q, EVIDENCE_SURFACE):
        candidates = list(aliases(used))
        # `torch.rand#generator` is a claim about the kwarg. A drill that
        # passes the kwarg exercises the bare call too, so the bare spelling
        # is checked as a fallback -- but NOT the reverse: a drill that calls
        # `torch.rand(...)` with no generator is no evidence for the kwarg.
        base = used.split("#")[0]
        if base != used:
            candidates += list(aliases(base))
        for spelling in candidates:
            if spelling in index:
                out.add(index[spelling])
                break
    return out


def coverage() -> dict[str, dict]:
    """declared symbol -> {kc, drills (on-node), elsewhere (other nodes)}."""
    declared, kc_of_page = declaring_kcs()
    index = _alias_index(declared)

    bank = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    questions = bank if isinstance(bank, list) else bank.get("questions", bank)
    qmatrix = json.loads(QMATRIX.read_text(encoding="utf-8"))

    out = {sym: {"kc": kc, "drills": set(), "elsewhere": set()}
           for sym, kc in declared.items()}
    for q in questions:
        qid = q.get("id")
        tags = set((qmatrix.get(str(qid)) or {}).get("target_kcs") or [])
        for sym in _declared_used(q, index):
            bucket = "drills" if out[sym]["kc"] in tags else "elsewhere"
            out[sym][bucket].add(qid)
    return out


def find(kc_prefix=None) -> list[dict]:
    _declared, kc_of_page = declaring_kcs()
    rank = lesson_order(kc_of_page)
    cov = coverage()
    prefixes = None
    if kc_prefix:
        prefixes = tuple(p.strip() for p in kc_prefix.split(",") if p.strip())

    out = []
    for sym, row in sorted(cov.items()):
        if len(row["drills"]) >= MIN_DRILLS_PER_SYMBOL:
            continue
        if prefixes and not row["kc"].startswith(prefixes):
            continue
        out.append({
            "kc": row["kc"], "symbol": sym,
            "drills": sorted(row["drills"], key=lambda x: int(x)),
            "elsewhere": sorted(row["elsewhere"], key=lambda x: int(x)),
            "count": len(row["drills"]),
            "rank": rank.get(row["kc"], -1),
        })
    return out


def key(v: dict) -> str:
    """Stable identity: the CONCEPT and the symbol.

    Not the page filename — renaming a page would launder the debt — and not
    the drill ids, which are positional in the CSVs and move whenever a row is
    retired.
    """
    return f"{v['kc']}|{v['symbol']}"


def load_baseline() -> dict[str, int] | None:
    """Known debt as {key: count at record time}, or None when ABSENT.

    None and empty are different answers: empty means the debt is genuinely
    zero, missing means the ratchet has nothing to compare against and must
    say so rather than pass.
    """
    if not BASELINE.exists():
        return None
    data = json.loads(BASELINE.read_text(encoding="utf-8"))
    return dict(data.get("violations") or {})


def regressions(violations: list[dict], known: dict[str, int]) -> dict[str, str]:
    """{key: why} for what the baseline does NOT already excuse.

    Two ways to fail: a symbol the baseline never heard of, and a symbol it
    knows about that has FEWER drills than when the debt was recorded.
    """
    out: dict[str, str] = {}
    for v in violations:
        k = key(v)
        if k not in known:
            out[k] = f"{k} — NEW, {v['count']}/{MIN_DRILLS_PER_SYMBOL} drill(s)"
        elif v["count"] < known[k]:
            out[k] = (f"{k} — LOST a drill, {known[k]} at baseline, "
                      f"{v['count']} now")
    return out


def write_baseline(violations: list[dict]) -> None:
    BASELINE.write_text(json.dumps({
        "_": "Known symbol-coverage debt: symbols a concept declares in "
             f"new_syntax but drills fewer than {MIN_DRILLS_PER_SYMBOL} times "
             "on its own node. The value is the drill count when the debt was "
             "recorded; the guard fails on a NEW key or on a count that has "
             "gone DOWN. Shrink it. Re-recording is admitting the debt, not "
             "fixing it, so say why in the commit.",
        "min_drills_per_symbol": MIN_DRILLS_PER_SYMBOL,
        "surface": EVIDENCE_SURFACE,
        "count": len(violations),
        "violations": {key(v): v["count"] for v in sorted(violations, key=key)},
    }, indent=1) + "\n", encoding="utf-8")


def report(violations: list[dict], args) -> int:
    known = load_baseline()
    if args.new:
        bad = regressions(violations, known or {})
        violations = [v for v in violations if key(v) in bad]

    by_kc = Counter(v["kc"] for v in violations)
    at_zero = [v for v in violations if v["count"] == 0]
    print(f"floor             : {MIN_DRILLS_PER_SYMBOL} drill(s) per declared "
          f"symbol, on its own concept, in the {EVIDENCE_SURFACE}")
    print(f"under the floor   : {len(violations)} symbol(s) "
          f"across {len(by_kc)} concept(s)")
    print(f"  of those at zero: {len(at_zero)}")
    if known is None:
        print("baseline          : ABSENT")
    else:
        allv = find()
        print(f"baseline          : {len(known)} known, "
              f"{len(regressions(allv, known))} regression(s), "
              f"{len(set(known) - {key(v) for v in allv})} stale")

    if not args.summary:
        print()
        grouped = defaultdict(list)
        for v in violations:
            grouped[(v["rank"], v["kc"])].append(v)
        for (_rank, kc), items in sorted(grouped.items()):
            print(f"  {kc}  ({len(items)} under)")
            for v in sorted(items, key=lambda x: x["symbol"]):
                where = (f"drilled on {','.join('q' + str(d) for d in v['drills'])}"
                         if v["drills"] else "NEVER drilled here")
                other = (f"; used by q{',q'.join(str(d) for d in v['elsewhere'][:4])}"
                         " on other concepts" if v["elsewhere"] else "")
                print(f"      {v['symbol']:34s} {v['count']}/"
                      f"{MIN_DRILLS_PER_SYMBOL}  {where}{other}")
    return 1 if violations else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--kc-prefix", default=None,
                    help="restrict to concepts whose id starts with one of "
                         "these, comma separated (numpy.,einops.)")
    ap.add_argument("--new", action="store_true",
                    help="only what the baseline does not already excuse")
    ap.add_argument("--write-baseline", action="store_true")
    args = ap.parse_args()

    if args.write_baseline:
        # A scoped subset written over the canonical file would reclassify
        # every symbol it omitted as NEW, and the next watcher run would fail
        # on the whole pre-existing backlog. Both sibling audits refuse the
        # same way.
        if args.kc_prefix:
            raise SystemExit("--write-baseline records the whole bank; "
                             "drop --kc-prefix")
        violations = find()
        write_baseline(violations)
        print(f"baseline written: {len(violations)} symbol(s) under the floor "
              f"-> {BASELINE.name}")
        return 0

    return report(find(args.kc_prefix), args)


if __name__ == "__main__":
    raise SystemExit(main())
