#!/usr/bin/env python3
"""Prove that no lesson shows syntax the learner has not been taught yet.

The curriculum's promise is that every piece of PyTorch a learner meets has a
lesson behind it. Nothing enforced that promise: `new_syntax:` exists in every
KP page's frontmatter and is empty almost everywhere, so a page could introduce
`t.cummax(...).values` — a call, an attribute and a kwarg — with no lesson for
any of them, and the only thing that would catch it is a human reading closely.
This is that check, done mechanically instead.

How it works
------------
1. Every fenced ```python block in every KP page is parsed with `ast`. Not by
   regex: a regex cannot tell `z.sum` (a method the learner must have met) from
   `q.sum` inside a string, and getting this wrong in the permissive direction
   defeats the point.
2. Each block yields a set of SYMBOLS at three grains, because "syntax" is not
   only function names:
       torch.cumsum          a torch-level function
       Tensor.sum            a tensor method
       torch.cumsum#dim      a PARAMETER of that call, checked separately —
                             `dim=` is exactly the kind of thing a learner is
                             expected to absorb by osmosis and does not
       builtin.assert        a language construct or builtin
       syntax.slice          notation with no name at all
3. Lessons are ordered by the KC prerequisite lattice (topological over
   kc_registry.json), which is the order the graph itself claims. A symbol is
   INTRODUCED by the page whose `new_syntax:` declares it.
4. A page using a symbol that no page declares, or that is only declared by a
   page at-or-after it in the order, is a violation.

`--suggest` prints the `new_syntax:` each page would need for the corpus to
pass. That is a starting point for authoring, not an answer: declaring a symbol
means a human wrote a lesson for it, and the whole point of this tool is that
the declaration is a claim someone stands behind.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LESSONS = ROOT / "Local_Deployed_Shared" / "lessons"
REGISTRY = LESSONS / "kc_registry.json"

FENCE = re.compile(r"```python[a-z ]*\n(.*?)```", re.S)
FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.S)

# Assumed known before the first lesson. Every entry here is a promise that the
# learner already has it, so the list is deliberately short and explicit rather
# than a convenient catch-all: anything quietly added here stops being checked.
ASSUMED = {
    "builtin.def", "builtin.return", "builtin.print", "builtin.len",
    "builtin.range", "builtin.int", "builtin.float", "builtin.str",
    "builtin.list", "builtin.tuple", "builtin.dict", "builtin.bool",
    "builtin.import", "syntax.arith", "syntax.compare", "syntax.call",
    "syntax.attribute", "syntax.subscript", "syntax.assign", "syntax.docstring",
    # Plain Python, and the course's stated audience already writes it. These
    # earn their place by WHAT THEY ARE FOR here: `assert` and `AssertionError`
    # are how every worked example proves its claim, `type`/`isinstance`/`repr`
    # are how a print shows what a result IS, and `round`/`abs` are arithmetic.
    # None of them is a tensor idea, so no KP could honestly claim to teach one
    # — declaring them somewhere would be filing paperwork, not writing a
    # lesson. (`getattr` is deliberately NOT here: kp-dtype-astype teaches it
    # as the way to turn a dtype NAME into a dtype, which is a real move.)
    "builtin.assert", "builtin.AssertionError", "builtin.type",
    "builtin.isinstance", "builtin.repr", "builtin.round", "builtin.abs",
    "syntax.comprehension", "syntax.for", "syntax.fstring",
}

TORCH_ALIASES = {"t", "torch"}

# einops is a second library with its own surface, and `einops.rearrange` is a
# different thing to learn from anything on a tensor. Without this the callee
# rule below — "an attribute on a non-torch base is a tensor method" — files
# every pattern call in the einops chapters under `Tensor.rearrange`, a method
# that does not exist, and the einops KP that DOES teach it cannot declare it
# under a name matching what the audit reports.
LIBRARY_ALIASES = {"einops": "einops"}


class Collector(ast.NodeVisitor):
    """Every symbol one code block puts in front of the learner."""

    def __init__(self) -> None:
        self.symbols: set[str] = set()
        # Names the block defines for itself. A demo that writes a helper and
        # then calls it is not showing the learner an unexplained builtin —
        # the definition is right there, three lines up.
        self.local_defs: set[str] = set()

    # -- helpers ---------------------------------------------------------
    def _callee(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Attribute):
            # Dunders are the object protocol, not curriculum. `type(x).__name__`
            # in a print is not a tensor method and there is no lesson to write
            # for it.
            if node.attr.startswith("__"):
                return None
            # Walk the whole dotted chain rather than one or two links: this is
            # what tells `t.nn.functional.pad` (one function, with a submodule
            # path) from a method named `pad` on a tensor. Stopping early filed
            # it as `Tensor.pad` AND invented `torch.nn` / `torch.nn.functional`
            # as separate things to teach.
            parts = [node.attr]
            base = node.value
            while isinstance(base, ast.Attribute):
                if base.attr.startswith("__"):
                    return None
                parts.append(base.attr)
                base = base.value
            if isinstance(base, ast.Name):
                if base.id in TORCH_ALIASES:
                    return "torch." + ".".join(reversed(parts))
                if base.id in LIBRARY_ALIASES:
                    return LIBRARY_ALIASES[base.id] + "." + ".".join(reversed(parts))
            return f"Tensor.{node.attr}"
        if isinstance(node, ast.Name):
            return f"builtin.{node.id}"
        return None

    # -- visitors --------------------------------------------------------
    def visit_Call(self, node: ast.Call) -> None:
        name = self._callee(node.func)
        if name:
            self.symbols.add(name)
            # Parameters are their own lesson content — `dim=`, `keepdim=`,
            # `correction=` change the ANSWER, not just the spelling.
            #
            # einops is the exception: its keywords ARE the pattern's axis
            # names, which the author invents per call (`h=2`, `two=2`, `g1=3`).
            # Treating those as API surface would demand a lesson for every
            # letter anyone ever picked, and teach nothing — the thing to learn
            # is that factors arrive as keywords, which is the pattern lesson.
            if not name.startswith("einops."):
                for kw in node.keywords:
                    if kw.arg:
                        self.symbols.add(f"{name}#{kw.arg}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Bare attribute access that is not a call: .shape, .dtype, .T, .values
        if not isinstance(getattr(node, "_is_callee", None), bool):
            name = self._callee(node)
            if name:
                self.symbols.add(name)
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        self.symbols.add("builtin.assert")
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        sl = node.slice
        if isinstance(sl, ast.Slice):
            self.symbols.add("syntax.slice")
            if sl.step is not None:
                self.symbols.add("syntax.slice-step")
        elif isinstance(sl, ast.Tuple):
            self.symbols.add("syntax.multi-axis-index")
        self.generic_visit(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self.symbols.add("syntax.comprehension")
        self.generic_visit(node)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        self.symbols.add("syntax.fstring")
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.symbols.add("syntax.for")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self.local_defs.add(node.name)
        self.generic_visit(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.symbols.add("syntax.lambda")
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.MatMult):
            self.symbols.add("syntax.matmul")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.symbols.add(f"import.{node.module}")
        self.generic_visit(node)


def parse_frontmatter(text: str) -> dict:
    m = FRONTMATTER.match(text)
    if not m:
        return {}
    out: dict = {}
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        key, _, raw = line.partition(":")
        raw = raw.strip()
        if raw.startswith("["):
            items = [x.strip().strip("'\"") for x in raw[1:-1].split(",")]
            out[key.strip()] = [x for x in items if x]
        else:
            out[key.strip()] = raw
    return out


def page_symbols(path: Path) -> tuple[dict, set[str], list[str]]:
    """(frontmatter, symbols used, parse errors) for one KP page."""
    text = path.read_text(encoding="utf-8")
    fm = parse_frontmatter(text)
    used: set[str] = set()
    errors: list[str] = []
    for block in FENCE.findall(text):
        try:
            tree = ast.parse(block)
        except SyntaxError as exc:
            # A faded starter with `t._____(x)` is not valid Python by design.
            # Blanked syntax is exactly what the learner must supply, so it is
            # not "shown" to them and is skipped rather than reported.
            if "_____" not in block:
                errors.append(f"{path.name}: unparseable block — {exc}")
            continue
        c = Collector()
        c.visit(tree)
        # `t._____(x)` PARSES — an underscore run is a legal identifier — so the
        # SyntaxError path above never sees it. A blank is the answer the
        # learner supplies, not syntax shown to them.
        local = {f"builtin.{n}" for n in c.local_defs}
        used |= {s for s in c.symbols if "_____" not in s and s not in local}
    return fm, used, errors


# Worked examples are read as much as run, so a line without a reason attached
# teaches only that the line exists. This checks the rule directly: in a
# `## Worked example` block, every executable line needs a trailing comment or
# a comment on the line above it.
def uncommented_example_lines(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    out: list[str] = []
    for section in re.split(r"^## ", text, flags=re.M)[1:]:
        if not section.lower().startswith("worked example"):
            continue
        for block in FENCE.findall(section):
            lines = block.splitlines()
            for i, raw in enumerate(lines):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                # An import line and a bare closing bracket carry no idea worth
                # explaining; everything else does.
                if line.startswith(("import ", "from ")) or line in (")", "]", "}"):
                    continue
                # A `#` inside a string literal is not a comment. Counting
                # quotes before the first `#` is enough for lesson code.
                before = raw.split("#")[0]
                has_trailing = (
                    "#" in raw
                    and before.count('"') % 2 == 0
                    and before.count("'") % 2 == 0
                )
                has_above = i > 0 and lines[i - 1].strip().startswith("#")
                if not (has_trailing or has_above):
                    out.append(f"{path.name}:{i + 1}  {line[:72]}")
    return out


# A worked example that only asserts runs green and shows the learner nothing:
# "Ran successfully (no printed output)" is the same screen whether the concept
# is obvious or baffling. An example earns its place by DISPLAYING the thing it
# is teaching, so every value the example computes has to be printed.
def silent_example_gaps(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    out: list[str] = []
    for section in re.split(r"^## ", text, flags=re.M)[1:]:
        if not section.lower().startswith("worked example"):
            continue
        for block in FENCE.findall(section):
            if "_____" in block:
                continue
            try:
                tree = ast.parse(block)
            except SyntaxError:
                continue
            assigned: list[str] = []
            printed: set[str] = set()
            prints = 0
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for tgt in node.targets:
                        if isinstance(tgt, ast.Name):
                            assigned.append(tgt.id)
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                        and node.func.id == "print":
                    prints += 1
                    for arg in node.args:
                        for sub in ast.walk(arg):
                            if isinstance(sub, ast.Name):
                                printed.add(sub.id)
            if prints == 0:
                out.append(f"{path.name}: worked example prints nothing at all")
                continue
            unseen = [n for n in dict.fromkeys(assigned) if n not in printed]
            if unseen:
                out.append(
                    f"{path.name}: computes but never shows {', '.join(unseen[:6])}"
                )
    return out


def lesson_order(kc_of_page: dict[str, str]) -> dict[str, int]:
    """Rank of each KC in the order a learner actually walks the course.

    That order is `kc_registry.json`'s KC sequence — the authoring source of
    truth, and what `compile_lessons.py` sorts pages by. It is used here in
    place of the prerequisite lattice's DEPTH, which was the earlier rule, for
    a concrete reason: depth is not a total order. Nine of np-1's twelve KCs
    sit at depth 0 or 1, so "is this symbol taught before that page?" was
    unanswerable across most of a chapter — same-depth pages compared equal and
    every use inside the tie looked fine.

    Registry order does not contradict the lattice: it is verified below to be
    a linear extension of it (every prerequisite appears earlier), so any
    "shown before it is taught" this reports is one the graph agrees with. If
    that ever stops holding, the assertion fires rather than the checker
    quietly ranking a KC before its own prerequisite.
    """
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    rank = {kc["id"]: i for i, kc in enumerate(reg["kcs"])}
    for kc in reg["kcs"]:
        for prereq in kc.get("prereqs") or []:
            if prereq in rank and rank[prereq] >= rank[kc["id"]]:
                raise SystemExit(
                    f"kc_registry.json: {kc['id']} is listed before its "
                    f"prerequisite {prereq}; fix the order or the prereq edge."
                )
    return rank


def audit(suggest: bool = False) -> int:
    pages = sorted(LESSONS.rglob("kp-*.md"))
    if not pages:
        print("no KP pages found", file=sys.stderr)
        return 1

    info = {}
    errors: list[str] = []
    for p in pages:
        fm, used, errs = page_symbols(p)
        errors.extend(errs)
        info[p] = (fm, used)

    rank = lesson_order({})
    ordered = sorted(
        pages,
        key=lambda p: (rank.get(str(info[p][0].get("kc") or ""), 99), p.name),
    )

    # Who claims to teach what.
    introduced_by: dict[str, list[Path]] = defaultdict(list)
    for p in ordered:
        for sym in info[p][0].get("new_syntax") or []:
            introduced_by[sym].append(p)

    if suggest:
        seen: set[str] = set()
        for p in ordered:
            fresh = sorted(s for s in info[p][1] if s not in seen and s not in ASSUMED)
            seen |= set(fresh)
            if fresh:
                print(f"\n# {p.relative_to(ROOT)}  (kc: {info[p][0].get('kc')})")
                print(f"new_syntax: [{', '.join(fresh)}]")
        return 0

    undeclared: dict[str, list[str]] = defaultdict(list)
    too_late: list[str] = []
    for idx, p in enumerate(ordered):
        for sym in sorted(info[p][1]):
            if sym in ASSUMED:
                continue
            homes = introduced_by.get(sym) or []
            if not homes:
                undeclared[sym].append(p.name)
                continue
            if all(ordered.index(h) > idx for h in homes):
                too_late.append(
                    f"{p.name} uses {sym}, first taught later in {homes[0].name}"
                )

    uncommented = {p: uncommented_example_lines(p) for p in ordered}
    uncommented = {p: v for p, v in uncommented.items() if v}
    silent = {p: silent_example_gaps(p) for p in ordered}
    silent = {p: v for p, v in silent.items() if v}

    print(f"KP pages audited: {len(pages)}")
    print(f"distinct symbols shown to learners: "
          f"{len({s for _, u in info.values() for s in u} - ASSUMED)}")
    print(f"symbols with NO lesson declaring them: {len(undeclared)}")
    print(f"uses that precede their own lesson: {len(too_late)}")
    print(f"worked-example lines with no explanation: "
          f"{sum(len(v) for v in uncommented.values())} across {len(uncommented)} page(s)")
    print(f"worked examples that show the learner nothing: "
          f"{sum(len(v) for v in silent.values())} across {len(silent)} page(s)")
    if errors:
        print(f"\nunparseable code blocks: {len(errors)}")
        for e in errors[:10]:
            print(f"  {e}")

    if undeclared:
        print("\n-- taught nowhere (symbol: pages that show it) --")
        for sym, where in sorted(undeclared.items(), key=lambda kv: -len(kv[1]))[:40]:
            shown = ", ".join(where[:3]) + ("…" if len(where) > 3 else "")
            print(f"  {sym:38s} {len(where):3d} page(s)  {shown}")
    if too_late:
        print("\n-- shown before it is taught --")
        for line in too_late[:40]:
            print(f"  {line}")

    if uncommented:
        print("\n-- worked-example lines with no comment explaining them --")
        for p, lines in list(uncommented.items())[:8]:
            print(f"  {p.name}")
            for line in lines[:6]:
                print(f"      {line}")

    if silent:
        print("\n-- worked examples that print nothing, or hide what they compute --")
        for p, lines in list(silent.items())[:10]:
            for line in lines[:3]:
                print(f"  {line}")

    return 1 if (undeclared or too_late or uncommented or silent) else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suggest", action="store_true",
                    help="print the new_syntax: each page would need to pass")
    args = ap.parse_args()
    sys.exit(audit(suggest=args.suggest))
