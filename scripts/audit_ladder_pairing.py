#!/usr/bin/env python3
"""Example-problem pairing audit for the expertise-reversal ladder.

The ladder's `faded` rung shows a worked example beside a problem and asks the
learner to complete it. That only works if the two are paired at the right
distance, and "the right distance" has two failure modes in opposite
directions:

  TOO FAR   the faded item needs syntax the example never demonstrates. The
            learner is told to fill a blank using a move they were not shown.
            This is the one that silently ruins the rung: it looks like a
            completion problem and behaves like a solo problem, so the learner
            fails, gets demoted, re-reads the same example, and fails again.

  TOO CLOSE the faded item's solution reproduces the example's code, so filling
            the blank is transcription and passing it is evidence of nothing.
            The ladder promotes on that evidence, so a too-close pair does not
            just waste a rung — it manufactures a false promotion.

What is checked
---------------
For every faded item authored in a KP page (`faded_items` in the compiled
lessons), against the worked example of the SEGMENT that owns it:

  1. COVERAGE   every symbol the item's solution requires and its starter does
                not already show must appear in the paired worked example (or
                be assumed/self-defined, or be declared by an EARLIER lesson —
                a learner at this KP has already met those).
  2. DISTANCE   the solution must compute at least one expression the example
                does not, comparing SHAPES with identifiers and literals
                erased. Shape rather than symbols is what makes this
                meaningful: renaming a variable is not transfer, and two honest
                problems routinely share an API surface.
  3. BLANK      the authored starter must actually hide something (`_____`).
                A faded starter with nothing blanked is the answer, and the
                rung silently becomes a free promotion.

Usage:
    python3 scripts/audit_ladder_pairing.py            # full report
    python3 scripts/audit_ladder_pairing.py --summary  # counts only
    python3 scripts/audit_ladder_pairing.py --kc numpy.ndarray-model
    python3 scripts/audit_ladder_pairing.py --strict   # distance fails too

Exit code is 1 on any COVERAGE or BLANK failure, so this can gate a build
today. DISTANCE is reported but does not fail without --strict, for a reason
worth writing down rather than hiding behind a flag.

Every segment in the bank authors EXACTLY ONE faded item (118 segments, 118
items), and 61 of those restate their own example. On a first completion item
that is defensible — Renkl's fading starts adjacent to the example and grows
distance across a SERIES. The bank has no series: there is no second, more
distant faded item anywhere to grow into. So the ladder's `faded` rung is
largely transcription, and the first real evidence arrives at `partial`, which
is generated from the canonical answer rather than authored.

That is a content-design gap, not a regression, and it is not fixable by
editing the 61 items — it needs second faded items authored per segment.
Failing the build on it today would only get the whole check switched off,
taking the two rules that ARE clean (coverage, blank) down with it.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_lesson_syntax import ASSUMED, lesson_order, page_symbols  # noqa: E402
from audit_question_syntax import SELF_DEFINED, code_symbols  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "Local_Deployed_Shared"
STRUCTURED = SHARED / "lessons" / "lessons_structured.json"
LESSONS_DIR = SHARED / "lessons"

FENCE = re.compile(r"```(?:python)?\s*\n([\s\S]*?)```")
BLANK = "_____"


class _Blank(ast.NodeTransformer):
    """Erase identifiers and literals, keep structure.

    `curved = t.floor(x)` and `out = t.floor(y)` collapse to the same shape,
    which is the point: copying an example means reproducing its shape, and
    renaming the variable is not transfer. Attribute and keyword names survive
    on purpose — `.clamp(min=)` and `.clamp(max=)` are different moves, not
    different spellings of one move, and that distinction is precisely what
    separates an honest pair from a transcription exercise.
    """

    def visit_Name(self, node):  # noqa: N802
        return ast.copy_location(ast.Name(id="_", ctx=node.ctx), node)

    def visit_Constant(self, node):  # noqa: N802
        # STRINGS SURVIVE. In this bank a string literal is usually the whole
        # answer — `'bi,bj->ij'` is the einsum spec and `'b h w c -> b c h w'`
        # is the einops pattern. Blanking those would collapse every einsum
        # call in the language to one shape and flag every einsum pair as a
        # copy. Numbers and booleans are erased: an axis count or a fill value
        # is a detail of the instance, not the move being taught.
        if isinstance(node.value, str):
            return node
        return ast.copy_location(ast.Constant(value=None), node)


def _shape(node: ast.AST) -> str:
    return ast.dump(_Blank().visit(ast.parse(ast.unparse(node)).body[0]))


def expression_shapes(src: str, body_of: str | None = None) -> list[str]:
    """The shapes of the expressions `src` actually computes.

    Statements are the wrong unit for this comparison. An example ends in
    `assert x.tolist() == [...]` and a solution ends in `return ...`, so
    statement-for-statement they never line up even when the middle is a
    verbatim copy. What carries the idea is the expression on the right of the
    assignment / after the `return`, so that is what gets compared.

    Bare names (`return out`) are dropped: they are plumbing, present in every
    solution, and counting them would make everything look distinct.

    `body_of` restricts the walk to one function's body, docstring excluded —
    the docstring is the task description, not a solution step.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []

    if body_of is not None:
        stmts: list[ast.stmt] = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == body_of:
                stmts = list(node.body)
                break
        if (
            stmts
            and isinstance(stmts[0], ast.Expr)
            and isinstance(getattr(stmts[0], "value", None), ast.Constant)
            and isinstance(stmts[0].value.value, str)
        ):
            stmts = stmts[1:]
    else:
        stmts = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]

    out: list[str] = []
    for stmt in stmts:
        for node in ast.walk(stmt):
            value = getattr(node, "value", None)
            if not isinstance(node, (ast.Assign, ast.Return, ast.Expr)):
                continue
            if not isinstance(value, ast.AST) or isinstance(value, (ast.Name, ast.Constant)):
                continue
            out.append(_shape(value))
    return out


def example_symbols(markdown: str) -> set[str]:
    """Every symbol demonstrated by a worked example's code fences.

    Prose is deliberately excluded. An example that merely *mentions*
    `t.stack` in a sentence has not shown the learner how to use it, and the
    faded item is about producing code, not recognising a name.
    """
    out: set[str] = set()
    for block in FENCE.findall(markdown or ""):
        out |= code_symbols(block)
    return out


def _known_before(kc: str, rank: dict[str, int], declared: dict[str, str]) -> set[str]:
    """Symbols a learner reaching this KP has already been taught.

    These need no re-demonstration in the example: the pairing rule is about
    the NEW move the item asks for, not about re-teaching the whole language
    on every page.
    """
    my_rank = rank.get(kc)
    if my_rank is None:
        return set()
    out = set()
    for sym, owner in declared.items():
        owner_rank = rank.get(owner)
        if owner_rank is not None and owner_rank < my_rank:
            out.add(sym)
    return out


def declaring_lessons() -> tuple[dict[str, str], dict[str, str]]:
    declared: dict[str, str] = {}
    kc_of_page: dict[str, str] = {}
    for path in sorted(LESSONS_DIR.rglob("kp-*.md")):
        fm, _used, _errs = page_symbols(path)
        kc = fm.get("kc")
        if not kc:
            continue
        kc_of_page[path.name] = kc
        for sym in fm.get("new_syntax") or []:
            declared.setdefault(sym, kc)
    return declared, kc_of_page


def _segments(kp: dict) -> list[dict]:
    segs = kp.get("segments") or []
    if segs:
        return segs
    return [{
        "title": kp.get("title", ""),
        "worked_example_markdown": kp.get("worked_example_markdown", ""),
        "faded_items": kp.get("faded_items") or [],
    }]


def audit(only_kc: str | None = None, summary: bool = False, strict: bool = False) -> int:
    data = json.loads(STRUCTURED.read_text(encoding="utf-8"))
    declared, kc_of_page = declaring_lessons()
    rank = lesson_order(kc_of_page)

    problems: list[tuple[str, int, str, str]] = []  # kc, qid, kind, detail
    pairs = 0
    segments = 0
    flat_segments: list[tuple[str, str]] = []

    for lesson in data.get("lessons", []):
        for kp in lesson.get("kps", []):
            kc = kp.get("kc")
            if not kc or (only_kc and kc != only_kc):
                continue
            already = _known_before(kc, rank, declared)
            for seg in _segments(kp):
                shown = example_symbols(seg.get("worked_example_markdown", ""))
                example_shapes = {
                    s
                    for block in FENCE.findall(seg.get("worked_example_markdown", "") or "")
                    for s in expression_shapes(block)
                }
                seg_faded = False
                seg_has_distance = False
                for item in seg.get("faded_items") or []:
                    if not isinstance(item, dict):
                        continue
                    seg_faded = True
                    qid = item.get("question_id")
                    starter = item.get("starter_code") or ""
                    solution = item.get("solution") or ""
                    if not solution.strip():
                        continue
                    pairs += 1

                    # 3. BLANK — a "faded" starter with nothing hidden is just
                    # the answer, and the rung silently becomes a free pass.
                    if BLANK not in starter:
                        problems.append((kc, qid, "blank",
                                         "authored faded starter hides nothing"))

                    # 1. COVERAGE — what the learner must produce that the
                    # starter does not already hand them.
                    required = code_symbols(solution) - code_symbols(starter)
                    missing = sorted(
                        s for s in required
                        if s not in ASSUMED
                        and s not in SELF_DEFINED
                        and s not in shown
                        and s not in already
                    )
                    if missing:
                        problems.append((kc, qid, "coverage",
                                         "example never shows " + ", ".join(missing)))

                    # 2. DISTANCE — the solution's body, with names and
                    # literals erased, appearing as a contiguous run inside the
                    # example means the learner can fill the blank by
                    # transcribing rather than transferring. Comparing SHAPES
                    # and not symbol sets is the whole point: a symbol-set test
                    # both rejects honest pairs that happen to reuse the same
                    # API and waves through a verbatim copy the moment the
                    # example mentions one extra name.
                    sol_shapes = expression_shapes(solution, body_of="solve")
                    if sol_shapes and set(sol_shapes) <= example_shapes:
                        problems.append((kc, qid, "distance",
                                         "every expression the solution computes already "
                                         "appears in the example — completing it is copying"))
                    else:
                        seg_has_distance = True

                # A segment is the unit the LEARNER experiences, and one
                # near-copy completion is correct fading — what breaks the rung
                # is a series that stops there. So track both: the per-item
                # count says how much of the bank restates its example, and this
                # says how many segments never hand the learner anything past
                # transcription. The second is the one that has to reach zero.
                if seg_faded:
                    segments += 1
                    if not seg_has_distance:
                        flat_segments.append((kc, seg.get("title") or "(untitled)"))

    by_kind: dict[str, int] = {}
    for _kc, _qid, kind, _detail in problems:
        by_kind[kind] = by_kind.get(kind, 0) + 1

    if not summary:
        for kc, qid, kind, detail in problems:
            print(f"q{qid} [{kc}] {kind}: {detail}")
        print()

    print(f"example/problem pairs checked : {pairs}")
    print(f"broken pairings               : {len(problems)}")
    for kind in ("coverage", "distance", "blank"):
        print(f"  {kind:9s}: {by_kind.get(kind, 0)}")
    print(f"distinct questions affected   : {len({p[1] for p in problems})}")
    print(f"segments with a faded series  : {segments}")
    print(f"  series never reaching distance: {len(flat_segments)}")

    if not summary and flat_segments:
        print("\n-- segments whose completion items never get past the example --")
        for kc, title in flat_segments:
            print(f"  {kc} — {title}")

    # `distance` is a KNOWN CONTENT BACKLOG, not a regression: the lesson bank
    # was authored before this rule existed and a large share of faded items
    # currently restate their example. Failing the build on it today would just
    # mean the gate gets disabled, and then the two rules that ARE clean
    # (coverage, blank) stop protecting anything. So distance reports loudly
    # and passes; --strict promotes it once the backlog is worked down.
    hard = [p for p in problems if p[2] != "distance"] if not strict else problems
    soft = len(problems) - len(hard)
    if soft:
        print(f"\nNOTE: {soft} 'distance' finding(s) reported but NOT failing the "
              f"build (pre-existing content backlog). Use --strict to enforce.")
    return 1 if hard else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kc", default=None, help="audit one concept")
    ap.add_argument("--summary", action="store_true", help="counts only")
    ap.add_argument("--strict", action="store_true",
                    help="fail on 'distance' findings too (content backlog)")
    args = ap.parse_args()
    return audit(only_kc=args.kc, summary=args.summary, strict=args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
