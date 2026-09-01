#!/usr/bin/env python3
"""Every named thing a chunk of Python puts in front of a learner.

`audit_lesson_syntax.Collector` answers "which library API does this code
show?". That is the right question for a lesson page and the wrong one for a
DRILL SOLUTION, because a solution is what the learner has to WRITE. Writing
`tuple(len(row) for row in rows)` needs a generator expression, a call, and
`len` — none of which is library API, all of which is syntax somebody has to
have taught. The base collector reports one symbol for that line.

So this module widens the net to *every* construct with a name a lesson could
own, using the SAME vocabulary the KP pages already declare in `new_syntax:`
(`syntax.index`, `syntax.list-literal`, `list.append`, `math.sqrt`, …) so that
a symbol reported here can actually be matched against a declaration.

Three things it fixes beyond breadth:

* **Imports are resolved.** The base collector files any attribute on a
  non-torch base as `Tensor.<attr>`, so `math.sqrt(2)` was reported as
  `Tensor.sqrt` — a method that does not exist, and unmatchable against the
  `math.sqrt` that `kp-dots-and-imports` declares. Here `import math` /
  `import numpy as np` / `from einops import rearrange` are tracked and the
  callee is resolved through them.
* **Names the code binds are not API.** `solve`, the fixture `example`, a
  helper `def pool(...)`, a loop variable — the base collector reported all of
  them as `builtin.<name>`. Bound names are collected and subtracted.
* **A method has more than one legal spelling.** Static analysis cannot tell
  `xs.append` (a list) from `x.append` (nothing), so `Tensor.append` also
  matches a declaration of `list.append`. `aliases()` returns every spelling a
  page may legitimately have used.

Nothing here changes `Collector`. The lesson audit, the question audit and
`validate_lessons.check_previews` all read the base class and keep their
current behaviour exactly.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_lesson_syntax import Collector, TORCH_ALIASES  # noqa: E402

# Receiver types whose methods a KP may declare as `<type>.<method>`. Used only
# to widen matching: a name in `dir(list)` reported as `Tensor.append` is also
# accepted as `list.append`.
RECEIVER_TYPES = ("list", "str", "dict", "tuple", "set", "int", "float")
_RECEIVER_METHODS = {
    name: {m for m in dir(__builtins__[name] if isinstance(__builtins__, dict)
                          else getattr(__builtins__, name))
           if not m.startswith("_")}
    for name in RECEIVER_TYPES
}

_BUILTIN_NAMES = set(dir(__builtins__)) if not isinstance(__builtins__, dict) \
    else set(__builtins__)


class StrictCollector(Collector):
    """Collector + language constructs + import resolution + name binding."""

    def __init__(self, known_names: set[str] | None = None) -> None:
        super().__init__()
        # alias in this file -> the module it actually is
        self.modules: dict[str, str] = {}
        # `from math import sqrt` -> {"sqrt": "math.sqrt"}
        self.from_imports: dict[str, str] = {}
        # every name this chunk binds: params, assignments, defs, loop vars.
        self.bound: set[str] = set(known_names or ())

    # -- callee resolution -----------------------------------------------
    def _callee(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                return None
            parts = [node.attr]
            base = node.value
            while isinstance(base, ast.Attribute):
                if base.attr.startswith("__"):
                    return None
                parts.append(base.attr)
                base = base.value
            dotted = ".".join(reversed(parts))
            if isinstance(base, ast.Name):
                if base.id in TORCH_ALIASES or self.modules.get(base.id) == "torch":
                    return f"torch.{dotted}"
                root = self.modules.get(base.id)
                if root:
                    return f"{root}.{dotted}"
                # A base the chunk never binds and never imports is not a
                # tensor with a method — it is a name that does not exist.
                # Reporting `np.load` as `Tensor.load` hid 24 solutions left
                # half-converted from numpy, all of them dead on the first line.
                if base.id not in self.bound:
                    return f"undefined.{base.id}.{dotted}"
            return f"Tensor.{node.attr}"
        if isinstance(node, ast.Name):
            if node.id in self.from_imports:
                return self.from_imports[node.id]
            return f"builtin.{node.id}"
        return None

    # -- binding ---------------------------------------------------------
    def _bind(self, target: ast.AST) -> None:
        for n in ast.walk(target):
            if isinstance(n, ast.Name):
                self.bound.add(n.id)

    # -- imports ---------------------------------------------------------
    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        self.symbols.add("syntax.import")
        for a in node.names:
            # `import torch.nn as nn` binds `nn` to the WHOLE dotted path; only
            # a plain `import torch.nn` binds the root. Collapsing both to the
            # root read `nn.Linear` as `torch.Linear`, a function that does not
            # exist and that no page could ever declare.
            local = a.asname or a.name.split(".")[0]
            self.modules[local] = a.name if a.asname else a.name.split(".")[0]
            self.bound.add(local)
            if a.asname:
                self.symbols.add("syntax.import-as")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        self.symbols.add("syntax.import")
        if node.module:
            self.symbols.add(f"import.{node.module}")
        for a in node.names:
            local = a.asname or a.name
            self.bound.add(local)
            if node.module:
                self.from_imports[local] = f"{node.module}.{a.name}"
            if a.asname:
                self.symbols.add("syntax.import-as")
        self.generic_visit(node)

    # -- definitions -----------------------------------------------------
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self.symbols.add("syntax.def")
        self.local_defs.add(node.name)
        self.bound.add(node.name)
        a = node.args
        for arg in [*a.posonlyargs, *a.args, *a.kwonlyargs,
                    a.vararg, a.kwarg]:
            if arg is not None:
                self.bound.add(arg.arg)
        if a.defaults or any(d is not None for d in a.kw_defaults):
            self.symbols.add("syntax.default-argument")
        if a.vararg or a.kwarg:
            self.symbols.add("syntax.star-args")
        if node.decorator_list:
            self.symbols.add("syntax.decorator")
        first = node.body[0] if node.body else None
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            self.symbols.add("syntax.docstring")
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self.symbols.add("syntax.class")
        self.local_defs.add(node.name)
        self.bound.add(node.name)
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:  # noqa: N802
        self.symbols.add("syntax.return")
        self.generic_visit(node)

    # -- assignment ------------------------------------------------------
    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        self.symbols.add("syntax.assign")
        for tgt in node.targets:
            if isinstance(tgt, (ast.Tuple, ast.List)):
                self.symbols.add("syntax.unpack")
            self._bind(tgt)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:  # noqa: N802
        self.symbols.add("syntax.aug-assign")
        self._bind(node.target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
        self.symbols.add("syntax.assign")
        self.symbols.add("syntax.annotation")
        self._bind(node.target)
        self.generic_visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:  # noqa: N802
        self.symbols.add("syntax.walrus")
        self._bind(node.target)
        self.generic_visit(node)

    # -- literals --------------------------------------------------------
    def visit_List(self, node: ast.List) -> None:  # noqa: N802
        if isinstance(node.ctx, ast.Load):
            self.symbols.add("syntax.list-literal")
        self.generic_visit(node)

    def visit_Tuple(self, node: ast.Tuple) -> None:  # noqa: N802
        if isinstance(node.ctx, ast.Load):
            self.symbols.add("syntax.tuple")
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:  # noqa: N802
        self.symbols.add("syntax.dict-literal")
        self.generic_visit(node)

    def visit_Set(self, node: ast.Set) -> None:  # noqa: N802
        self.symbols.add("syntax.set-literal")
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:  # noqa: N802
        if node.value is True or node.value is False:
            self.symbols.add("syntax.bool-literal")
        elif node.value is None:
            self.symbols.add("syntax.none")
        self.generic_visit(node)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:  # noqa: N802
        self.symbols.add("syntax.fstring")
        # Do NOT descend: the pieces are formatting, and an f-string's own
        # `{x:.2f}` spec is not a construct anyone teaches separately.
        for v in node.values:
            if isinstance(v, ast.FormattedValue):
                self.visit(v.value)

    # -- indexing --------------------------------------------------------
    def visit_Subscript(self, node: ast.Subscript) -> None:  # noqa: N802
        sl = node.slice
        if isinstance(sl, ast.Slice):
            self.symbols.add("syntax.slice")
            if sl.step is not None:
                self.symbols.add("syntax.slice-step")
        elif isinstance(sl, ast.Tuple):
            self.symbols.add("syntax.multi-axis-index")
        else:
            self.symbols.add("syntax.index")
            if (isinstance(sl, ast.UnaryOp) and isinstance(sl.op, ast.USub)):
                self.symbols.add("syntax.negative-index")
        if isinstance(node.value, ast.Subscript):
            self.symbols.add("syntax.nested-index")
        if isinstance(sl, ast.Constant) and sl.value is None:
            self.symbols.add("none-newaxis-indexing")
        self.generic_visit(node)

    def visit_Starred(self, node: ast.Starred) -> None:  # noqa: N802
        self.symbols.add("syntax.star-args")
        self.generic_visit(node)

    # -- operators -------------------------------------------------------
    # `&` `|` `~` on a tensor are boolean MASKING, which is a tensor idea and
    # not arithmetic — filing them under syntax.arith would let a drill that
    # combines two masks pass on a page that only ever taught `+`.
    _BITWISE = (ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift)

    def visit_BinOp(self, node: ast.BinOp) -> None:  # noqa: N802
        if isinstance(node.op, ast.MatMult):
            self.symbols.add("syntax.matmul")
        elif isinstance(node.op, self._BITWISE):
            self.symbols.add("syntax.bitwise")
        else:
            self.symbols.add("syntax.arith")
        self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:  # noqa: N802
        if isinstance(node.op, ast.Not):
            self.symbols.add("syntax.not")
        elif isinstance(node.op, ast.Invert):
            self.symbols.add("syntax.bitwise")
        else:
            self.symbols.add("syntax.arith")
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:  # noqa: N802
        self.symbols.add("syntax.and" if isinstance(node.op, ast.And) else "syntax.or")
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:  # noqa: N802
        for op in node.ops:
            if isinstance(op, (ast.In, ast.NotIn)):
                self.symbols.add("syntax.in")
            elif isinstance(op, (ast.Is, ast.IsNot)):
                self.symbols.add("syntax.is")
            elif isinstance(op, (ast.Eq, ast.NotEq)):
                self.symbols.add("syntax.equality")
            else:
                self.symbols.add("syntax.compare")
        self.generic_visit(node)

    # -- control flow ----------------------------------------------------
    def visit_If(self, node: ast.If) -> None:  # noqa: N802
        self.symbols.add("syntax.if")
        if node.orelse:
            self.symbols.add("syntax.else")
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:  # noqa: N802
        self.symbols.add("syntax.ternary")
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:  # noqa: N802
        self.symbols.add("syntax.for")
        self._bind(node.target)
        if node.orelse:
            self.symbols.add("syntax.else")
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:  # noqa: N802
        self.symbols.add("syntax.while")
        self.generic_visit(node)

    def visit_Break(self, node: ast.Break) -> None:  # noqa: N802
        self.symbols.add("syntax.break")

    def visit_Continue(self, node: ast.Continue) -> None:  # noqa: N802
        self.symbols.add("syntax.continue")

    def visit_With(self, node: ast.With) -> None:  # noqa: N802
        self.symbols.add("syntax.with")
        for item in node.items:
            if item.optional_vars is not None:
                self._bind(item.optional_vars)
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:  # noqa: N802
        self.symbols.add("syntax.try")
        for h in node.handlers:
            if h.name:
                self.bound.add(h.name)
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:  # noqa: N802
        self.symbols.add("syntax.raise")
        self.generic_visit(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
        self.symbols.add("syntax.lambda")
        a = node.args
        for arg in [*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg]:
            if arg is not None:
                self.bound.add(arg.arg)
        self.generic_visit(node)

    def visit_Yield(self, node: ast.Yield) -> None:  # noqa: N802
        self.symbols.add("syntax.yield")
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:  # noqa: N802
        self.symbols.add("syntax.global")

    # -- comprehensions --------------------------------------------------
    def _comp(self, node: ast.AST, label: str) -> None:
        self.symbols.add(label)
        for gen in node.generators:  # type: ignore[attr-defined]
            self._bind(gen.target)
            if gen.ifs:
                self.symbols.add("syntax.comprehension-filter")
        self.generic_visit(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:  # noqa: N802
        self._comp(node, "syntax.comprehension")

    def visit_SetComp(self, node: ast.SetComp) -> None:  # noqa: N802
        self._comp(node, "syntax.set-comprehension")

    def visit_DictComp(self, node: ast.DictComp) -> None:  # noqa: N802
        self._comp(node, "syntax.dict-comprehension")

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:  # noqa: N802
        self._comp(node, "syntax.generator-expression")

    # -- the two the base class already emits, kept explicit --------------
    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        self.symbols.add("syntax.call")
        if node.keywords:
            self.symbols.add("syntax.keyword-argument")
        super().visit_Call(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        self.symbols.add("syntax.attribute")
        # `type(x).__name__` is the readable word for a type, and
        # `python.types-and-conversion` declares it as `python.type-name`. The
        # receiver is a Call, so `_callee` resolves nothing and the move landed
        # as bare `syntax.attribute` — a name no page declares and every
        # attribute access satisfies. Left that way the declared symbol sat at
        # 0/2 coverage permanently: the two drills that DO teach it (q574,
        # q578) could never be counted, and no new drill could ever fix it.
        if node.attr == "__name__" and _is_type_call(node.value):
            self.symbols.add("python.type-name")
        super().visit_Attribute(node)


def _is_type_call(node: ast.AST) -> bool:
    """True for `type(x)` written inline, which is the only spelling credited.

    Matching every `__name__` would credit `module.__name__` and
    `some_function.__name__` as evidence of the type-name move, so a drill
    could satisfy the coverage floor without ever showing it. The two-step
    spelling (`kind = type(x)` then `kind.__name__`) is deliberately not
    credited either: missing credit surfaces as a coverage violation, which is
    loud, while false credit is silent.
    """
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "type"
    )


def aliases(symbol: str) -> set[str]:
    """Every spelling a KP page may legitimately have declared this symbol as.

    Only one rule so far, and it exists because a static pass cannot know a
    receiver's type: `xs.append(3)` is reported `Tensor.append`, and the page
    that teaches it declares `list.append`.
    """
    out = {symbol}
    base, _, kwarg = symbol.partition("#")
    if base.startswith("Tensor."):
        attr = base[len("Tensor."):]
        for typ, methods in _RECEIVER_METHODS.items():
            if attr in methods:
                out.add(f"{typ}.{attr}" + (f"#{kwarg}" if kwarg else ""))
    return out


def prebind(tree: ast.AST, into: set[str]) -> None:
    """Every name the chunk itself binds, gathered BEFORE the main pass.

    A single pass cannot do this: `_callee` has to know whether `np` is a local
    the code defined or a name from nowhere, and the binding may sit on a later
    line than the use (a helper called above its own `def`).
    """
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            into.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            into.add(n.name)
            args = getattr(n, "args", None)
            if args is not None:
                for a in [*args.posonlyargs, *args.args, *args.kwonlyargs,
                          args.vararg, args.kwarg]:
                    if a is not None:
                        into.add(a.arg)
        elif isinstance(n, ast.Lambda):
            a = n.args
            for arg in [*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg]:
                if arg is not None:
                    into.add(arg.arg)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                into.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.ExceptHandler) and n.name:
            into.add(n.name)


def collect(source: str, known_names: set[str] | None = None) -> set[str]:
    """Symbols a solution requires the learner to be able to write.

    Unparseable source yields nothing — a faded starter's `_____` is the answer
    the learner supplies, not syntax anyone showed them.
    """
    if not source or not source.strip():
        return set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    c = StrictCollector(known_names)
    prebind(tree, c.bound)
    c.visit(tree)
    bound = c.bound | c.local_defs
    out = set()
    for s in c.symbols:
        if "_____" in s:
            continue
        # A name this very chunk binds is not API the learner must have met.
        if s.startswith("builtin."):
            name = s.split("#")[0][len("builtin."):]
            if name in bound:
                continue
            if name not in _BUILTIN_NAMES:
                # Not a builtin and not bound anywhere: the solution calls
                # something that does not exist.
                out.add("undefined." + s.split(".", 1)[1])
                continue
        out.add(s)
    return out


def bound_names(source: str) -> set[str]:
    """Names a fixture/setup chunk defines, for the caller to pass forward."""
    try:
        tree = ast.parse(source or "")
    except SyntaxError:
        return set()
    c = StrictCollector()
    prebind(tree, c.bound)
    c.visit(tree)
    return c.bound | c.local_defs


# Node types that carry no teachable idea of their own: they are the grammar's
# glue (a Load context, an argument slot), or an operator whose MEANING is
# reported by the visitor that owns its parent (`ast.Add` is read by
# `visit_BinOp`). Everything not listed here must have a visitor, or the
# collector is walking past a construct a learner has to be able to write.
STRUCTURAL = frozenset({
    "Module", "Interactive", "Expression", "Expr", "Load", "Store", "Del",
    "Name", "arguments", "arg", "keyword", "alias", "comprehension",
    "FormattedValue", "Slice", "Index", "withitem", "ExceptHandler", "Pass",
    "Ellipsis", "TypeIgnore", "MatchValue", "MatchAs",
    # operators, read by visit_BinOp / visit_UnaryOp / visit_BoolOp / visit_Compare
    "Add", "Sub", "Mult", "Div", "FloorDiv", "Mod", "Pow", "MatMult",
    "LShift", "RShift", "BitOr", "BitXor", "BitAnd", "UAdd", "USub", "Not",
    "Invert", "And", "Or", "Eq", "NotEq", "Lt", "LtE", "Gt", "GtE",
    "Is", "IsNot", "In", "NotIn",
})


def unhandled_node_types(sources) -> set[str]:
    """Node types present in `sources` that nothing here looks at.

    This is the completeness claim, checked rather than asserted: "every
    function whatsoever" is only true while no construct walks past unseen.
    A new drill written with `match`, a decorator, a `while`/`else` — anything
    the collector has no visitor for — shows up here instead of silently
    counting as taught.
    """
    handled = {m[len("visit_"):] for m in dir(StrictCollector)
               if m.startswith("visit_")}
    seen: set[str] = set()
    for src in sources:
        try:
            tree = ast.parse(src or "")
        except SyntaxError:
            continue
        seen |= {type(n).__name__ for n in ast.walk(tree)}
    return seen - handled - STRUCTURAL
