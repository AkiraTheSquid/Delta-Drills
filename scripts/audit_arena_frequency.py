#!/usr/bin/env python3
"""What does ARENA actually use? Frequency audit against the real curriculum.

Our graph decides what gets taught and how often it gets drilled. Nothing has
ever checked that decision against the curriculum the course exists to prepare
people for. `arena-book-colab/ARENA_5.0` is 458 exercise notebooks — that is
the population, and every symbol in it has an observable frequency.

Two numbers per symbol, and they answer different questions:

* **calls**  — how many times it is written. Skewed by one notebook that calls
               `t.zeros` forty times in a loop body.
* **DF**     — in how many of the 458 notebooks it appears at all. This is the
               one to rank by: it measures "how much of the curriculum you
               cannot read without knowing this", which is exactly what a
               prerequisite graph is supposed to encode.

`ch-1-foundations/` is EXCLUDED. That directory is Delta Drills' own published
notebooks; measuring it would be measuring ourselves and calling it evidence.

Usage:
    python3 scripts/audit_arena_frequency.py                 # the distribution
    python3 scripts/audit_arena_frequency.py --coverage      # us vs ARENA
    python3 scripts/audit_arena_frequency.py --chapter 0     # one chapter
    python3 scripts/audit_arena_frequency.py --json out.json
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from solution_symbols import collect  # noqa: E402
from audit_solution_prereqs import declaring_kcs  # noqa: E402
from audit_lesson_syntax import lesson_order  # noqa: E402
# Defined in the GROUNDING module, not here: the watchers that verify
# freshness run under bare python3, and this module exits at import time
# without torch. One definition, on the side that can always load it.
from audit_arena_grounding import corpus_fingerprint  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARENA = ROOT / "arena-book-colab" / "ARENA_5.0"
QUESTIONS = ROOT / "Local_Deployed_Shared" / "questions.json"

# Our own published notebooks live here. Including them would measure us.
OURS = "ch-1-foundations"

# A notebook cell is not always Python: `%pip install`, `!wget`, and IPython
# magics make `ast.parse` throw and would silently drop the whole cell.
MAGIC = re.compile(r"^\s*[%!]")

# ARENA keeps its worked solutions in MARKDOWN, inside fenced blocks under a
# `<details><summary>Solution</summary>` — the code cell next to them is the
# empty stub the learner fills in. Reading only `cell_type == "code"` therefore
# measures the STUBS and calls it the curriculum: 437 of the 458 notebooks
# carry fenced code in markdown, 1040 blocks of it labelled `python`, and
# `nn.Linear` came back at 1 notebook when the corpus has it in nine.
FENCE = re.compile(r"```([A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)```", re.S)
_PY_LABELS = {"python", "py", "python3"}

# ARENA's prose fences contain LaTeX (`\ `, `\d`) inside string literals, and
# parsing them emits a SyntaxWarning per occurrence — on stdout, from a
# watcher, about somebody else's notebook. Nothing is being hidden: an invalid
# escape is still valid Python, and a block that genuinely fails to parse is
# dropped by `_is_python` either way.
warnings.filterwarnings("ignore", category=SyntaxWarning)


def _is_python(block: str) -> bool:
    """Does an UNLABELLED fence hold code, or output?

    1704 of the fences carry no language tag and they are a mix: real snippets
    alongside pasted `tensor([1., 2.])` output, which parses perfectly well as
    Python and would inject a call to `tensor` that nobody wrote. So a block
    counts only when it parses AND contains a statement that is not a bare
    expression — an import, an assignment, a def, a loop. Output dumps are bare
    expressions and drop out; snippets keep.

    Only ever asked of an UNLABELLED fence. A block the author labelled `yaml`
    or `bash` is not Python no matter how it parses, and the difference is not
    academic: `model: gpt-4` is a perfectly good Python AnnAssign, and the
    corpus has 462 yaml fences. Trusting the label where there is one, and
    guessing only where there is not, is the whole rule.
    """
    try:
        tree = ast.parse(block)
    except SyntaxError:
        return False
    return any(not isinstance(node, ast.Expr) for node in tree.body)


def cell_source(nb_path: Path) -> list[str]:
    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []
    out = []
    for cell in nb.get("cells") or []:
        src = cell.get("source")
        src = "".join(src) if isinstance(src, list) else (src or "")
        if cell.get("cell_type") != "code":
            for label, block in FENCE.findall(src):
                if not block.strip():
                    continue
                label = label.lower()
                if label in _PY_LABELS or (not label and _is_python(block)):
                    out.append(block)
            continue
        # Replaced with `pass`, not deleted: ARENA's setup cell is
        # `try: import x / except ImportError: %pip install x`, and DELETING
        # the magic left an empty except block that failed to parse. That was
        # 9.5% of all cells — the setup cell of nearly every notebook.
        src = "\n".join(
            (ln[:len(ln) - len(ln.lstrip())] + "pass") if MAGIC.match(ln) else ln
            for ln in src.splitlines())
        if src.strip():
            out.append(src)
    return out


IMPORT_LINE = re.compile(r"^\s*(import|from)\s")


def preamble(cells: list[str]) -> str:
    """Every import in the notebook, to prepend to each cell before parsing.

    A notebook's imports live in its first cell and its code lives in the rest,
    so parsing a cell alone leaves `rearrange(...)` looking like a call to a
    name from nowhere. That is not a rounding error: it reported einops as
    ENTIRELY ABSENT from ARENA, which would have been a spectacular thing to
    act on. Prepending the import lines is safer than concatenating whole
    cells, which can put a stray `else:` at module level.
    """
    keep: list[str] = []
    for src in cells:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            # Fall back to the line scan for a cell that will not parse at all.
            for ln in src.splitlines():
                ln = ln.strip()
                if not IMPORT_LINE.match(ln):
                    continue
                try:
                    ast.parse(ln)
                except SyntaxError:
                    continue
                keep.append(ln)
            continue
        # Whole AST nodes, so a wrapped `from x import (\n a, b)` survives, and
        # an import nested in a `try:` arrives DEDENTED — pasted at its own
        # indentation it makes the preamble itself unparseable, which silently
        # zeroes every cell it is prepended to. (It did: 4363 symbol-notebook
        # pairs collapsed to 74 before that was caught.)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                keep.append(ast.unparse(node))
    return "\n".join(keep)


def notebooks(chapter: str | None = None) -> list[Path]:
    paths = [p for p in sorted(ARENA.rglob("*.ipynb")) if OURS not in p.parts]
    if chapter:
        paths = [p for p in paths if f"chapter{chapter}" in str(p)]
    return paths


# symbol -> the notebooks it appears in. Populated by `scan`, read by
# `write_index`, which needs notebook IDENTITIES to fold two spellings of one
# operation into a single document frequency.
_NOTEBOOKS: dict[str, set[str]] = defaultdict(set)


def scan(paths: list[Path]) -> tuple[Counter, Counter, dict[str, set[str]]]:
    """(cell-uses, document-frequency, symbol -> chapters it appears in).

    The first counter is CELLS THAT USE the symbol, not call sites: `collect`
    returns a set per cell. DF is the number to rank by either way.
    """
    uses: Counter = Counter()
    df: Counter = Counter()
    where: dict[str, set[str]] = defaultdict(set)
    _NOTEBOOKS.clear()
    for p in paths:
        chapter = next((part for part in p.parts if part.startswith("chapter")), "?")
        cells = cell_source(p)
        head = preamble(cells)
        here: set[str] = set()
        for src in cells:
            found = collect(head + "\n" + src)
            here |= found
            # `collect` returns a SET, so this counts cells-that-use rather
            # than raw call sites. Named `calls` because that is what it
            # approximates; DF is the number to rank by.
            for sym in found:
                uses[sym] += 1
        for sym in here:
            df[sym] += 1
            where[sym].add(chapter)
            # The notebook IDENTITIES, not just a count. Folding two spellings
            # of one operation needs the UNION of the notebooks that use
            # either, and a union cannot be recovered from two totals after
            # the fact: `max` is only right when one spelling's notebooks are
            # a subset of the other's, and there is no reason they would be.
            _NOTEBOOKS[sym].add(str(p))
    return uses, df, where


try:
    import torch as _torch
except ImportError:  # pragma: no cover
    raise SystemExit(
        "this audit needs torch to tell a tensor method from a string method — "
        "run it with This-Directory-Only/backend/.venv/bin/python")

# A static pass cannot know a receiver's type, so `_callee` files every
# attribute on a non-module base as `Tensor.<attr>`. In a 6-line drill the
# receiver really is a tensor. In an ARENA notebook it is `Path.cwd()`,
# `cfg.n_heads`, `msg.content`, `d.items()` — and the raw ranking came back
# with `Tensor.read`, `Tensor.parents` and `Tensor.cwd` at 100% of notebooks,
# which is the boilerplate header of every file and not curriculum at all.
# Intersecting with the real member list is the cheap correction.
TENSOR_MEMBERS = {m for m in dir(_torch.Tensor) if not m.startswith("_")}
_OTHER_MEMBERS = set()
for _typ in (str, dict, list, set, tuple):
    _OTHER_MEMBERS |= {m for m in dir(_typ) if not m.startswith("_")}
import pathlib as _pl
_OTHER_MEMBERS |= {m for m in dir(_pl.Path) if not m.startswith("_")}

# Present in most notebooks and taught by nobody: the device/precision preamble
# every ARENA file opens with. Real usage, but it is environment setup, so it
# is marked rather than allowed to head the "what should we teach" ranking.
BOILERPLATE = {
    "torch.cuda", "torch.cuda.is_available", "torch.backends",
    "torch.backends.mps", "torch.backends.mps.is_available", "torch.device",
    "torch.set_grad_enabled", "torch.cuda.empty_cache", "torch.cuda.manual_seed",
    "torch.cuda.manual_seed_all",
}


def library(sym: str) -> bool:
    """Symbols a curriculum could teach: library API, not language glue.

    A `Tensor.<attr>` only counts when torch.Tensor actually has that member —
    see TENSOR_MEMBERS for why.
    """
    if sym.startswith("Tensor."):
        return sym.split("#")[0][len("Tensor."):] in TENSOR_MEMBERS
    return sym.startswith(("torch.", "einops.")) or sym == "syntax.matmul"


def operation(sym: str) -> str:
    """Fold the two spellings of one operation.

    `torch.argmax(x)` and `x.argmax()` are the same thing to learn, and ARENA
    overwhelmingly writes the method while our pages declare the function. Left
    unfolded, half our curriculum looked like dead weight: `torch.argmax` was
    reported as "ARENA never uses it" while `Tensor.argmax` sat at 51
    notebooks. Comparing operations rather than spellings is the honest join;
    which spelling to TEACH is then a separate, real question.
    """
    base = sym.split("#")[0]
    for prefix in ("torch.", "Tensor."):
        if base.startswith(prefix):
            tail = base[len(prefix):]
            if "." not in tail and tail in TENSOR_MEMBERS:
                return "op." + tail
    return base


def ambiguous(sym: str) -> bool:
    """A tensor method whose name is also a str/dict/list/Path method.

    `x.split`, `d.values`, `p.name`: the count is an upper bound, not a fact.
    """
    return (sym.startswith("Tensor.")
            and sym.split("#")[0][len("Tensor."):] in _OTHER_MEMBERS)


INDEX = Path(__file__).resolve().parent / "arena_symbol_index.json"



def write_index(n_notebooks: int, df: Counter) -> None:
    """Freeze the corpus summary the watchers read.

    The scan itself needs torch (to tell `x.split()` on a tensor from
    `s.split()` on a string) and the 458-notebook tree, neither of which a
    fast structural watcher can assume: `arena-book-colab/` is a vendored
    corpus, and most watchers run under bare `python3`. So the answer to "does
    ARENA use this?" is computed here, once, and committed. The watchers read
    the file and nothing else.

    `tensor_members` and `ambiguous_members` ride along because `operation()`
    and `ambiguous()` cannot be recomputed without torch — a reader that
    guessed at them would fold the spellings differently from the audit that
    produced the numbers, which is the join bug this file exists to avoid.
    """
    # UNION of the notebooks, not the max of the two counts. `torch.argmax` in
    # notebooks {A, B} and `x.argmax()` in {C, D} is one operation appearing in
    # four notebooks; `max` calls it two. The two spellings' notebook sets have
    # no reason to nest, so the union is the only correct fold — which is why
    # `scan` keeps identities and not just totals.
    folded: dict[str, set[str]] = {}
    for sym in df:
        if library(sym) and "#" not in sym:
            folded.setdefault(operation(sym), set()).update(_NOTEBOOKS[sym])
    ops = Counter({op: len(nbs) for op, nbs in folded.items()})
    INDEX.write_text(json.dumps({
        "_": "Frozen summary of the ARENA corpus — what the curriculum may "
             "teach. Regenerate with: This-Directory-Only/backend/.venv/bin/"
             "python scripts/audit_arena_frequency.py --write-index. Do not "
             "hand-edit: every number here is a measurement.",
        "corpus": "arena-book-colab/ARENA_5.0 (ch-1-foundations excluded — ours)",
        # Machine-readable so the freshness check can recount without importing
        # this module — which would drag in torch, and the watchers that most
        # need the check are the ones running under bare python3.
        "corpus_root": str(ARENA.relative_to(ROOT)),
        "excluded_dir": OURS,
        "notebooks": n_notebooks,
        "corpus_fingerprint": corpus_fingerprint(ARENA, OURS),
        # DF per FOLDED operation: torch.argmax and x.argmax() are one thing
        # to learn, and ARENA writes the method where our pages declare the
        # function. Ranking the spellings separately made half the curriculum
        # look dead.
        "operation_df": dict(sorted(ops.items(), key=lambda kv: (-kv[1], kv[0]))),
        # Raw, unfolded, library symbols only — kept so a reader can ask which
        # SPELLING ARENA prefers, which is a separate question from whether
        # the operation appears at all.
        "raw_df": dict(sorted(
            ((s, d) for s, d in df.items() if library(s) and "#" not in s),
            key=lambda kv: (-kv[1], kv[0]))),
        "tensor_members": sorted(TENSOR_MEMBERS),
        "ambiguous_members": sorted(TENSOR_MEMBERS & _OTHER_MEMBERS),
        "boilerplate": sorted(BOILERPLATE),
    }, indent=1) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chapter", default=None)
    ap.add_argument("--top", type=int, default=60)
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--write-index", action="store_true",
                    help="refresh scripts/arena_symbol_index.json, the frozen "
                         "corpus summary the watchers read")
    args = ap.parse_args()

    paths = notebooks(args.chapter)
    n = len(paths)
    cell_uses, df, where = scan(paths)
    # `Tensor.name`, `Tensor.split`, `Tensor.values`: torch.Tensor has all
    # three, and so do Path/str/dict. Without type inference the count is an
    # upper bound, and `Tensor.name` came back at 98% of notebooks — which is
    # `Path(...).name` in the setup cell, not a tensor. Held out of the
    # headline numbers and reported on its own line.
    lib = {s: c for s, c in df.items()
           if library(s) and "#" not in s and not ambiguous(s)}
    ambig = {s: c for s, c in df.items()
             if library(s) and "#" not in s and ambiguous(s)}
    total_df = sum(lib.values())

    print(f"ARENA notebooks scanned: {n}  (ch-1-foundations excluded — ours)")
    print(f"distinct library symbols: {len(lib)}   total symbol-notebook pairs: {total_df}")
    print(f"held out as unattributable (name shared with str/dict/list/Path): "
          f"{len(ambig)} symbols, {sum(ambig.values())} pairs")
    print()

    if args.json:
        Path(args.json).write_text(json.dumps({
            "notebooks": n,
            "df": dict(df), "cell_uses": dict(cell_uses),
            "chapters": {k: sorted(v) for k, v in where.items()},
        }, indent=1) + "\n", encoding="utf-8")
        print(f"written: {args.json}")

    if args.write_index:
        write_index(n, df)
        print(f"written: {INDEX}")

    if not args.coverage:
        print(f"{'symbol':30s} {'DF':>5s} {'%nb':>6s} {'cells':>6s}  chapters")
        for sym, d in sorted(lib.items(), key=lambda kv: -kv[1])[:args.top]:
            chs = ",".join(sorted(c.replace("chapter", "") for c in where[sym]))
            flag = " [env]" if sym in BOILERPLATE else (" [?]" if ambiguous(sym) else "")
            print(f"{sym:30s} {d:5d} {100*d/n:5.1f}% {cell_uses[sym]:6d}  {chs}{flag}")
        return 0

    # ---- us vs them -------------------------------------------------------
    declared, kc_of_page = declaring_kcs()
    rank = lesson_order(kc_of_page)
    taught = {s for s in declared if library(s) and "#" not in s}

    bank = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    drilled: Counter = Counter()
    for q in bank:
        for sym in collect(q.get("answer_code") or ""):
            if library(sym) and "#" not in sym:
                drilled[sym] += 1

    # Fold both sides to operations before comparing. ARENA's side folds by
    # UNION of notebooks, never by summing the two spellings' DFs: a notebook
    # that writes both `torch.argmax` and `x.argmax()` would otherwise be
    # counted twice in the very total everything else is a percentage of.
    arena_nb: dict[str, set[str]] = {}
    for sym in lib:
        arena_nb.setdefault(operation(sym), set()).update(_NOTEBOOKS[sym])
    arena_op: Counter = Counter({o: len(n) for o, n in arena_nb.items()})
    total_df = sum(arena_op.values())
    taught_op = {operation(s) for s in taught}
    drilled_op: Counter = Counter()
    for sym, c in drilled.items():
        drilled_op[operation(sym)] += c

    covered_mass = sum(d for o, d in arena_op.items() if o in taught_op)
    env_ops = {operation(s) for s in BOILERPLATE}
    env_mass = sum(d for o, d in arena_op.items() if o in env_ops)
    # Both sides net of the preamble, or the ratio compares different universes.
    covered_ex_env = sum(d for o, d in arena_op.items()
                         if o in taught_op and o not in env_ops)
    print(f"operations we teach: {len(taught_op)}   ARENA uses: {len(arena_op)}")
    print(f"overlap: {len(taught_op & set(arena_op))} operations, carrying "
          f"{100*covered_mass/total_df:.1f}% of ARENA's usage mass "
          f"({100*covered_ex_env/max(total_df-env_mass, 1):.1f}% once the "
          f"device/dtype preamble is set aside)")
    ranked = sorted(arena_op.items(), key=lambda kv: -kv[1])
    for cut in (10, 25, 50, 100):
        head = ranked[:cut]
        mine = sum(d for o, d in head if o in taught_op)
        mass = sum(d for _o, d in head)
        print(f"  top {cut:3d} ARENA operations = {100*mass/total_df:4.1f}% of usage; "
              f"we teach {sum(1 for o, _ in head if o in taught_op):3d}/{len(head)}"
              f" ({100*mine/max(mass,1):4.1f}% of that mass)")
    print()

    print("=== TOP ARENA OPERATIONS WE DO NOT TEACH (the gaps) ===")
    for sym, d in sorted(lib.items(), key=lambda kv: -kv[1]):
        # Folded: `Tensor.topk` is not a gap when `torch.topk` is taught.
        if operation(sym) in taught_op:
            continue
        if d < 8:
            break
        flag = " [env]" if sym in BOILERPLATE else (" [?]" if ambiguous(sym) else "")
        print(f"  {d:4d} nb ({100*d/n:4.1f}%)  {sym:28s} drilled {drilled.get(sym, 0)}x{flag}")
    print()

    print("=== WE TEACH IT, ARENA NEVER USES IT — EITHER SPELLING (cut candidates) ===")
    dead_ops = sorted({operation(s) for s in taught} - set(arena_op))
    by_op: dict[str, list[str]] = defaultdict(list)
    for s2 in taught:
        by_op[operation(s2)].append(s2)
    for op in dead_ops:
        syms = sorted(by_op[op])
        owner = declared[syms[0]]
        uses = sum(drilled.get(x, 0) for x in syms)
        print(f"  {'/'.join(syms):40s} {owner:32s} drilled {uses}x")
    print(f"  ({len(dead_ops)} of {len(taught_op)} operations = "
          f"{100*len(dead_ops)/max(len(taught_op),1):.0f}% of what we teach)")
    print()

    print("=== WE DRILL IT, ARENA NEVER USES IT ===")
    dead_drills = sorted((c, s) for s, c in drilled.items()
                         if operation(s) not in arena_op)
    print(f"  {sum(c for c, _ in dead_drills)} solution-uses over "
          f"{len(dead_drills)} symbols")
    for c, s in sorted(dead_drills, reverse=True)[:20]:
        print(f"  {c:4d}x  {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
