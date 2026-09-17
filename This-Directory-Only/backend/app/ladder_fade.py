"""Backward fading — build a partially-worked starter from a canonical answer.

Renkl & Atkinson's backward fading: when you take support away, remove the LAST
solution step first and work backwards, rather than blanking the first step.
The reason is asymmetric difficulty. With the worked example fresh, the final
step is the one a learner can most plausibly attempt — it is closest to the
goal and every input it needs is already on the page. Blank the FIRST step
instead and the learner cannot even begin, so the item stops being a completion
problem and becomes a solo problem with extra clutter.

Two reveal depths, matching the ladder's two scaffolded rungs:

    "most"  (faded rung)   — every body line but the last is shown.
    "half"  (partial rung) — only the first half of the body lines is shown.

This is generated, not authored. 63 KP pages hand-author one faded item each,
which is the reason `partial` could not simply be added as a third authored
list: it would need 63 more hand-written variants. Deriving the fade from
`answer_code` makes the rung available on every question in the bank at once,
and keeps it honest — the shown lines ARE the canonical solution's lines, so a
faded starter can never drift from the answer it is fading.

An authored faded starter (the `_____` blanks in KP frontmatter) still wins at
the `faded` rung when one exists; this fills the gap everywhere else.
"""

from __future__ import annotations

import ast
import re
from typing import List, Optional

BLANK_LINE = "    # ... your code here"

# Lines that are setup rather than solution: imports, and the trailing
# example-run block every starter carries. Fading those teaches nothing.
_IMPORT = re.compile(r"^\s*(import|from)\s")


def _function_span(src: str, fn_name: str) -> Optional[tuple]:
    """(start_line, end_line, body_start_line) for `fn_name`, 1-based inclusive.

    Returns None when the source will not parse or the function is absent —
    callers then leave the starter alone rather than emitting a mangled one.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            body = [n for n in node.body]
            if not body:
                return None
            # Skip a leading docstring: it is the spec, not a step to remove.
            first = body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(getattr(first, "value", None), ast.Constant)
                and isinstance(first.value.value, str)
                and len(body) > 1
            ):
                body_start = body[1].lineno
            else:
                body_start = first.lineno
            end = max(getattr(n, "end_lineno", n.lineno) for n in node.body)
            return node.lineno, end, body_start
    return None


def fade(answer_code: str, fn_name: str, reveal: str = "most") -> Optional[str]:
    """A starter with the tail of `fn_name`'s body removed.

    `reveal="most"` keeps all but the final statement; `reveal="half"` keeps the
    first half. Returns None when the answer cannot be parsed, has too few
    statements to fade meaningfully, or lacks the function — in every one of
    those cases the caller should fall back to the question's own starter, which
    is always safe.
    """
    if not answer_code or not fn_name:
        return None
    span = _function_span(answer_code, fn_name)
    if not span:
        return None
    _def_line, end_line, body_start = span

    lines = answer_code.splitlines()
    tree = ast.parse(answer_code)  # already known to parse — _function_span ran
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn_name
    )
    steps = [n for n in fn.body if n.lineno >= body_start]
    # A one-statement body faded to nothing is just the solo problem with a
    # comment in it — no support at all, so do not pretend otherwise.
    if len(steps) < 2:
        return None

    keep = len(steps) - 1 if reveal == "most" else max(1, len(steps) // 2)
    if keep >= len(steps):
        return None
    cut_from = steps[keep].lineno  # first line of the removed tail

    indent = re.match(r"^[ \t]*", lines[body_start - 1]).group(0) or "    "
    faded: List[str] = []
    placed = False
    for i, line in enumerate(lines, start=1):
        if i < cut_from:
            faded.append(line)
        elif i <= end_line:
            # Inside the removed tail: emit the blank once, drop the rest.
            if not placed:
                faded.append(f"{indent}# ... finish the function from here")
                placed = True
        else:
            # Past the function — the example-run block stays, so the learner
            # can run their attempt and see output immediately.
            faded.append(line)
    return "\n".join(faded).rstrip() + "\n"
