#!/usr/bin/env python3
"""Syntax-coverage audit for BANK QUESTIONS, across all three surfaces.

`audit_lesson_syntax.py` checks that a lesson page never *shows* syntax an
earlier lesson has not declared. That is only half the guarantee. A learner also
meets syntax in the drills, and a drill has three surfaces that can each leak
untaught material:

  * **prompt**   — `question_text`. Naming `t.arange` in the prose teaches
                   nothing but assumes it; the learner has to know it to parse
                   the task.
  * **starter**  — `starter_code`. This is handed to the learner as given, so
                   every symbol in it is shown to them.
  * **solution** — `answer_code`. This is the one that actually bites: it is
                   what the learner must PRODUCE. Auditing only the starter
                   passes a drill whose starter is a bare `return None` and
                   whose required answer needs three unlearned functions.

A question is a violation when any of those three uses a symbol whose declaring
lesson sits at or after the question's own KC in the prerequisite lattice — or
that no lesson declares at all. Ordering comes from `kc_registry.json`, so the
checker cannot disagree with the map the tutor gates on.

Usage:
    python3 scripts/audit_question_syntax.py            # report
    python3 scripts/audit_question_syntax.py --summary  # counts only
    python3 scripts/audit_question_syntax.py --qid 73   # one question
Exit code is 1 when any violation is found, so this can gate a build.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_lesson_syntax import (  # noqa: E402
    ASSUMED,
    Collector,
    LESSONS,
    REGISTRY,
    lesson_order,
    page_symbols,
)

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "Local_Deployed_Shared"
QUESTIONS = SHARED / "questions.json"
QMATRIX = SHARED / "lessons" / "qmatrix_tags.json"

# Backticked spans in a prompt are the prose surface's code. Anything else in
# the prose is English and is not held to the syntax contract.
BACKTICK = re.compile(r"`([^`\n]+)`")

# Names the question itself defines are not syntax anyone has to have been
# taught. `solve` is the function the learner writes, and `example`/`result`
# are the scaffold's own locals — counting them as untaught API was the one
# false positive this audit produced on every single question.
SELF_DEFINED = {"builtin.solve", "builtin.example", "builtin.result"}


def code_symbols(src: str) -> set[str]:
    """Symbols shown by a chunk of Python. Unparseable chunks yield nothing —
    a faded starter with `_____` is not valid Python by design, and a blank is
    the answer the learner supplies rather than syntax shown to them."""
    if not src or not src.strip():
        return set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return set()
    c = Collector()
    c.visit(tree)
    return {s for s in c.symbols if "_____" not in s}


def prompt_symbols(text: str) -> set[str]:
    """Symbols a prompt names in backticks. Parsed as expressions where
    possible so `t.arange` and `x.sum(dim=0)` resolve the same way they would
    in code; bare prose in backticks (a variable name, a shape) yields nothing
    of interest and is dropped by the same ASSUMED filter as everything else."""
    out: set[str] = set()
    for span in BACKTICK.findall(text or ""):
        out |= code_symbols(span.strip())
    return out


def declaring_lessons() -> tuple[dict[str, str], dict[str, str]]:
    """(symbol -> KC that first declares it, page name -> KC).

    A symbol is 'declared' by the `new_syntax:` frontmatter of the KP page that
    teaches it — the same contract the lesson audit uses, so a symbol taught
    once is taught for both audits.
    """
    declared: dict[str, str] = {}
    kc_of_page: dict[str, str] = {}
    for path in sorted(LESSONS.rglob("kp-*.md")):
        fm, _used, _errs = page_symbols(path)
        kc = fm.get("kc")
        if not kc:
            continue
        kc_of_page[path.name] = kc
        for sym in fm.get("new_syntax") or []:
            declared.setdefault(sym, kc)
    return declared, kc_of_page


def audit(only_qid: int | None = None, summary: bool = False) -> int:
    bank = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    questions = bank if isinstance(bank, list) else bank.get("questions", bank)
    qmatrix = json.loads(QMATRIX.read_text(encoding="utf-8"))

    declared, kc_of_page = declaring_lessons()
    rank = lesson_order(kc_of_page)

    violations: list[tuple[int, str, str, str]] = []
    untagged = 0
    checked = 0

    for q in questions:
        qid = q.get("id")
        if only_qid is not None and qid != only_qid:
            continue
        row = qmatrix.get(str(qid)) or {}
        targets = row.get("target_kcs") or []
        if not targets:
            untagged += 1
            continue
        kc = targets[0]
        my_rank = rank.get(kc)
        if my_rank is None:
            continue
        checked += 1

        surfaces = {
            "prompt": prompt_symbols(q.get("question_text") or ""),
            "starter": code_symbols(q.get("starter_code") or ""),
            "solution": code_symbols(q.get("answer_code") or ""),
        }
        for surface, syms in surfaces.items():
            for sym in sorted(syms):
                if sym in ASSUMED or sym in SELF_DEFINED:
                    continue
                owner = declared.get(sym)
                if owner is None:
                    violations.append((qid, kc, surface, f"{sym} — taught by NO lesson"))
                    continue
                owner_rank = rank.get(owner)
                if owner_rank is not None and owner_rank > my_rank:
                    violations.append(
                        (qid, kc, surface,
                         f"{sym} — first taught in {owner} (rank {owner_rank} > {my_rank})")
                    )

    by_surface: dict[str, int] = {}
    for _qid, _kc, surface, _msg in violations:
        by_surface[surface] = by_surface.get(surface, 0) + 1

    if not summary:
        for qid, kc, surface, msg in violations:
            print(f"q{qid} [{kc}] {surface}: {msg}")
        print()

    print(f"questions checked : {checked}  (untagged, skipped: {untagged})")
    print(f"violations        : {len(violations)}")
    for surface in ("prompt", "starter", "solution"):
        print(f"  {surface:9s}: {by_surface.get(surface, 0)}")
    print(f"distinct questions affected: {len({v[0] for v in violations})}")
    return 1 if violations else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qid", type=int, default=None, help="audit one question")
    ap.add_argument("--summary", action="store_true", help="counts only")
    args = ap.parse_args()
    return audit(only_qid=args.qid, summary=args.summary)


if __name__ == "__main__":
    raise SystemExit(main())
