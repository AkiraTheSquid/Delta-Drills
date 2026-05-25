#!/usr/bin/env python3
"""Author 8 ex2 deepening drills (batch 11).

Atoms (2 geometry/linalg + 6 hparam-config):
    - triangle-barycentric            (ex2: convert Cartesian (x,y) -> (u,v) via 2x2 solve)
    - try-except-solve                (ex2: list-batched safe-solve via try/except, returns list[Optional[Tensor]])
    - dataclasses-replace-args        (ex2: 2-D grid sweep via nested replace over (lr, batch_size))
    - hparam-precedence-merge         (ex2: filter sentinel None from cli before merging so user-typed --flag stays)
    - nested-param-group-loop         (ex2: manual SGD-with-weight-decay per-group via nested loop)
    - optimizer-class-dispatch        (ex2: register a new optimizer at runtime + dispatch)
    - param-group-dict-list           (ex2: three-group split — no-decay biases + decay-weights + head LR)
    - params-iterable-vs-groups       (ex2: polymorphic dispatch with pre-existing nn.Parameter as Tensor subclass)

Each ex2 hits a DISTINCT facet from ex1: different cognitive operation, surface
context, or constraint set. ONE LO + ONE Bloom + <=2 KCs per drill.

Verification re-runs each spec's solution against its test_body inside the
build venv (torch 2.12.0+cpu) before any notebook is emitted.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

TOPIC_GEOM = "prereqs_geometry_cnn"
TOPIC_HPARAM = "prereqs_hparam_config"


# ---------------------------------------------------------------------------
# Per-atom deepening recaps (build on ex1; tight focus on the ex2 facet).
# ---------------------------------------------------------------------------

RECAP_TRIANGLE_BARY_DEEP = (
    "## Barycentric coords — inverting Cartesian → (u, v)\n"
    "\n"
    "Ex1 used the predicate `u >= 0 & v >= 0 & u + v <= 1` on coordinates "
    "that were ALREADY in barycentric form. Real pipelines arrive at "
    "barycentrics by SOLVING for them: given a point `P` and triangle "
    "`ABC`, find the `(u, v)` such that\n"
    "\n"
    "```\n"
    "P - A = u * (B - A) + v * (C - A)\n"
    "```\n"
    "\n"
    "This is a 2×2 linear system in `(u, v)`:\n"
    "\n"
    "```\n"
    "[ (B-A).x  (C-A).x ] [u]   [(P-A).x]\n"
    "[ (B-A).y  (C-A).y ] [v] = [(P-A).y]\n"
    "```\n"
    "\n"
    "**Why a 2×2 not a 3×3 solve.** In 2-D the system has exactly 2 "
    "unknowns and 2 equations. In 3-D (Möller-Trumbore) the system "
    "becomes 3×3 because the ray parameter `s` joins `(u, v)` as a "
    "third unknown.\n"
    "\n"
    "**Composition with ex1's predicate.** Once you have `(u, v)`, the "
    "inside test is *exactly* ex1's three inequalities. Cartesian "
    "inside-triangle = solve + predicate."
)

RECAP_TRY_EXCEPT_BATCH = (
    "## try/except solve — list-batched form\n"
    "\n"
    "Ex1 wrapped ONE call in `try/except RuntimeError` and returned "
    "`None` on failure. The natural extension is a PYTHON-LEVEL batch: "
    "you have a `list[Tensor]` of `(n, n)` matrices and a parallel "
    "`list[Tensor]` of RHS vectors, and you want one solution-or-None "
    "per pair without letting one bad matrix kill the rest:\n"
    "\n"
    "```python\n"
    "def safe_solve_list(As, bs):\n"
    "    out = []\n"
    "    for A, b in zip(As, bs):\n"
    "        try:\n"
    "            out.append(t.linalg.solve(A, b))\n"
    "        except RuntimeError:\n"
    "            out.append(None)\n"
    "    return out\n"
    "```\n"
    "\n"
    "**Why a Python loop, not `t.linalg.solve` on a stacked tensor.** A "
    "stacked solve fails ALL slices the moment one is singular. The "
    "loop form is the right tool when failures are per-item and you "
    "want surviving results to come through.\n"
    "\n"
    "**Contrast with `singular-matrix-mask-trick`.** The mask trick "
    "stays fully vectorized but always returns a same-shape tensor "
    "(NaN-marked at bad slices). The list-of-Optional form is for the "
    "smaller-N case where Python overhead is fine and the caller "
    "wants explicit per-slot None."
)

RECAP_DATACLASS_GRID = (
    "## `dataclasses.replace` — 2-D grid sweep\n"
    "\n"
    "Ex1 swept ONE axis (LR). The natural deepening is the 2-D grid: "
    "for every `(lr, batch_size)` cross-pair, build one variant. The "
    "idiom is a nested comprehension that passes BOTH overrides into "
    "a single `replace` call:\n"
    "\n"
    "```python\n"
    "variants = [\n"
    "    replace(base, lr=lr, batch_size=bs)\n"
    "    for lr in lrs\n"
    "    for bs in bss\n"
    "]\n"
    "```\n"
    "\n"
    "**Why ONE `replace` per pair (not nested replaces).** "
    "`replace(replace(base, lr=lr), batch_size=bs)` works but pays "
    "two `__post_init__` runs per variant and produces an "
    "intermediate object. The single-call form is the same result, "
    "one validation, one allocation.\n"
    "\n"
    "**Ordering convention.** Outer loop is the SLOWER-varying axis "
    "(matches numpy / itertools.product) — first vary `bs` for each "
    "fixed `lr`, then advance `lr`. Match this when feeding a sweep "
    "harness so logs are sorted in a predictable order."
)

RECAP_PRECEDENCE_FILTER = (
    "## Precedence merge — filter sentinel `None` from CLI\n"
    "\n"
    "Ex1 implemented the strict `defaults < file < cli` chain — every "
    "key in `cli_args` wins, including explicit `None`. That works "
    "when argparse populates `cli_args` only for flags the user "
    "actually typed.\n"
    "\n"
    "Many real CLIs (esp. fire / typer / hand-rolled argparse) emit "
    "`None` as a sentinel meaning 'flag not provided'. In that case "
    "you DO want to drop Nones from `cli_args` BEFORE merging, so an "
    "un-passed `--lr` doesn't blast away a value the YAML file just "
    "set:\n"
    "\n"
    "```python\n"
    "def merge_with_cli_sentinel(defaults, file_cfg, cli_args):\n"
    "    cli_filtered = {k: v for k, v in cli_args.items() if v is not None}\n"
    "    out = {}\n"
    "    out.update(defaults)\n"
    "    out.update(file_cfg)\n"
    "    out.update(cli_filtered)\n"
    "    return out\n"
    "```\n"
    "\n"
    "**Two different defaults — pick by convention.** Strict "
    "later-wins (ex1) is right when None is a legitimate value. "
    "Sentinel-filter (ex2) is right when None means 'unset'. The "
    "decision belongs at the CLI-parser layer, not buried in the "
    "merge function — but the helper must MATCH the parser's "
    "convention or you get silent precedence bugs.\n"
    "\n"
    "**Why filter only the CLI layer, not the file.** A YAML user "
    "who types `lr: null` is making an explicit choice; the file is "
    "expressive enough to mean what it says. The CLI layer has the "
    "shape problem because argparse defaults to None for absent flags."
)

RECAP_NESTED_LOOP_WD = (
    "## Manual SGD step with weight decay — per-group hparams\n"
    "\n"
    "Ex1 read ONE per-group hparam (`lr`) inside the nested loop. The "
    "natural deepening: also read `weight_decay`. Weight decay adds a "
    "`-lr * wd * p` term to every parameter (or equivalently, an "
    "`L2` penalty on `p` in the loss):\n"
    "\n"
    "```python\n"
    "for group in optimizer.param_groups:\n"
    "    lr = group['lr']\n"
    "    wd = group.get('weight_decay', 0.0)\n"
    "    for p in group['params']:\n"
    "        if p.grad is None:\n"
    "            continue\n"
    "        g = p.grad\n"
    "        if wd != 0.0:\n"
    "            g = g + wd * p.data       # decoupled decay term\n"
    "        p.data.add_(g, alpha=-lr)\n"
    "```\n"
    "\n"
    "**Why `group.get('weight_decay', 0.0)`.** Not every group has "
    "weight_decay set (some groups, e.g. biases / LayerNorm params, "
    "opt out). `.get(..., 0.0)` is the standard 'missing means no "
    "decay' convention.\n"
    "\n"
    "**Sign convention.** Decay is `+wd * p` ADDED to the grad — "
    "since the step subtracts `lr * grad`, that pulls `p` toward zero. "
    "Match the sign convention used in `torch.optim.SGD(weight_decay=...)`.\n"
    "\n"
    "**Vanilla SGD weight-decay vs AdamW decoupled decay.** The form "
    "above is the SGD style: decay is folded into the grad before the "
    "step. AdamW decouples decay so it's applied as a separate term "
    "outside the moment estimates — same arithmetic, different timing."
)

RECAP_OPTIMIZER_REGISTER = (
    "## Optimizer dispatch — runtime registration\n"
    "\n"
    "Ex1 hard-coded three entries in `OPTIMIZER_CLASSES`. The next "
    "design step is to let CALLERS register new optimizers at runtime "
    "(plugin systems, experiments, custom optimizers from research "
    "code):\n"
    "\n"
    "```python\n"
    "OPTIMIZER_CLASSES = {\n"
    "    'sgd':   t.optim.SGD,\n"
    "    'adam':  t.optim.Adam,\n"
    "    'adamw': t.optim.AdamW,\n"
    "}\n"
    "\n"
    "def register_optimizer(name: str, cls):\n"
    "    OPTIMIZER_CLASSES[name] = cls\n"
    "```\n"
    "\n"
    "**Why mutate the module-level dict.** This is the simplest "
    "registry pattern — it lets `register_optimizer('lion', Lion)` "
    "from a plugin module take effect everywhere that imports "
    "`OPTIMIZER_CLASSES`. No singleton object needed; the dict IS "
    "the registry.\n"
    "\n"
    "**Validation at registration, not at dispatch.** Reject a "
    "non-class value at registration time (`isinstance(cls, type)` + "
    "subclass of `torch.optim.Optimizer`). This shifts the failure "
    "to the plugin loader, where the stack trace points at the bad "
    "registration — not later when an unrelated training run dies on "
    "`opt.step()`.\n"
    "\n"
    "**Overwrite-vs-error policy.** Default to OVERWRITE (`d[name] = "
    "cls`) so users can intentionally swap implementations (e.g. a "
    "fused-CUDA AdamW for the default). If you want strict-add, the "
    "caller can check `name in OPTIMIZER_CLASSES` first."
)

RECAP_THREE_GROUP_NO_DECAY = (
    "## Param groups — three-way split for no-decay biases\n"
    "\n"
    "Ex1 built two groups by MODULE (encoder vs head) with different "
    "LRs. The standard production-quality split is BY PARAMETER ROLE "
    "WITHIN A MODULE: weights get weight_decay, biases and "
    "normalization params don't. Combined with ex1's encoder/head LR "
    "split that becomes a three-group setup:\n"
    "\n"
    "```python\n"
    "encoder_decay, encoder_no_decay = split_params(encoder)\n"
    "head_params = list(head.parameters())\n"
    "\n"
    "groups = [\n"
    "    {'params': encoder_decay,    'lr': enc_lr, 'weight_decay': wd},\n"
    "    {'params': encoder_no_decay, 'lr': enc_lr, 'weight_decay': 0.0},\n"
    "    {'params': head_params,      'lr': head_lr, 'weight_decay': wd},\n"
    "]\n"
    "```\n"
    "\n"
    "**The split rule.** A param is 'no-decay' if `p.dim() <= 1` "
    "(biases are rank-1; weight matrices are rank-2+). LayerNorm "
    "weight is rank-1 too, so the same rule catches it. This is the "
    "convention used by transformers / vit / nanoGPT.\n"
    "\n"
    "**Why three groups, not two.** Two groups (decay vs no-decay) "
    "loses the encoder/head LR distinction; two groups (encoder vs "
    "head) loses the decay distinction. The full transfer-learning "
    "fine-tune wants BOTH knobs.\n"
    "\n"
    "**Iteration count check.** Sum of `len(g['params'])` across all "
    "groups MUST equal the total parameter count of the model — "
    "otherwise some params silently aren't being trained."
)

RECAP_PARAMS_DISPATCH_DEEP = (
    "## params dispatch — `nn.Parameter` subclass + reused materialization\n"
    "\n"
    "Ex1 implemented the polymorphic `tensor` vs `dict` dispatch from "
    "scratch. The deepening focuses on TWO subtle facets that bite real "
    "callers:\n"
    "\n"
    "1. **`nn.Parameter` IS a `Tensor` subclass** — `isinstance(p, "
    "t.Tensor)` is True for any `nn.Parameter`. Code that special-cases "
    "`type(first) is t.Tensor` (using `is` instead of `isinstance`) "
    "WOULD FAIL when a caller passes `module.parameters()`. Always use "
    "`isinstance`.\n"
    "\n"
    "2. **The dispatch must re-materialize the iterable AFTER peeking.** "
    "If you do `first = next(iter(params))` to inspect type, the original "
    "iterator has already advanced — the first element is lost on the "
    "next pass. Materialize ONCE with `list(params)` up front, then peek "
    "at `materialized[0]`. (PyTorch's own source does this.)\n"
    "\n"
    "**Robust skeleton:**\n"
    "```python\n"
    "def normalize(params):\n"
    "    materialized = list(params)         # one-shot consume\n"
    "    if not materialized:\n"
    "        raise ValueError('empty')\n"
    "    if isinstance(materialized[0], t.Tensor):   # catches nn.Parameter too\n"
    "        return [{'params': materialized}]\n"
    "    if isinstance(materialized[0], dict):\n"
    "        return [dict(g) for g in materialized]\n"
    "    raise TypeError(...)\n"
    "```"
)


# ---------------------------------------------------------------------------
# Specs.
# ---------------------------------------------------------------------------

SPECS = [
    # ─────────────────────────────────────────────────────────────────────
    # 1. triangle-barycentric  (ex2: invert Cartesian -> (u,v) via 2x2 solve)
    # ─────────────────────────────────────────────────────────────────────
    {
        "atom_id": "triangle-barycentric",
        "subtopic": "Geometry: Barycentric coords",
        "topic_folder": TOPIC_GEOM,
        "atom_recap_md": RECAP_TRIANGLE_BARY_DEEP,
        "exercise_index": 2,
        "exercise_title": "Cartesian to (u, v) via 2x2 solve",
        "slug": "cartesian-to-uv-via-2x2-solve",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["barycentric", "linalg-solve", "2x2", "inversion"],
        "kcs": ["barycentric-cartesian-solve", "barycentric-edge-basis"],
        "lo": (
            "Apply a 2x2 `t.linalg.solve` against the edge-basis matrix "
            "`[B-A | C-A]` to convert a Cartesian point `P` into "
            "barycentric coordinates `(u, v)` w.r.t. triangle `ABC`."
        ),
        "prompt_body": (
            "Implement `ex2_uv_from_cartesian(A, B, C, P)`.\n\n"
            "Inputs (all `(2,)` float tensors): triangle vertices `A`, "
            "`B`, `C` and a query point `P`. Return `(u, v)` as a "
            "tensor of shape `(2,)` such that "
            "`P = A + u*(B-A) + v*(C-A)`.\n\n"
            "**Algorithm.** Stack the edge vectors as columns of a "
            "2x2 matrix `M = [B-A | C-A]` and solve `M @ [u, v] = "
            "P - A`. One call to `t.linalg.solve(M, P - A)`.\n\n"
            "**Hint.** `t.stack([B - A, C - A], dim=1)` gives the "
            "2x2 with edges as columns (NOT rows — that would be "
            "the wrong system)."
        ),
        "stub": (
            "def ex2_uv_from_cartesian(A: Tensor, B: Tensor, C: Tensor, P: Tensor) -> Tensor:\n"
            '    """Solve P = A + u*(B-A) + v*(C-A) for (u, v). Returns (2,)."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Canonical right triangle A=(0,0), B=(1,0), C=(0,1).\n"
            "A = t.tensor([0.0, 0.0])\n"
            "B = t.tensor([1.0, 0.0])\n"
            "C = t.tensor([0.0, 1.0])\n"
            "\n"
            "# Centroid -> u=v=1/3.\n"
            "P_cent = (A + B + C) / 3\n"
            "uv = ex2_uv_from_cartesian(A, B, C, P_cent)\n"
            "assert uv.shape == (2,), f'expected (2,), got {tuple(uv.shape)}'\n"
            "assert t.allclose(uv, t.tensor([1/3, 1/3]), atol=1e-5), f'centroid uv: {uv}'\n"
            "\n"
            "# Vertex A -> u=v=0; vertex B -> u=1,v=0; vertex C -> u=0,v=1.\n"
            "assert t.allclose(ex2_uv_from_cartesian(A, B, C, A), t.zeros(2), atol=1e-5)\n"
            "assert t.allclose(ex2_uv_from_cartesian(A, B, C, B), t.tensor([1.0, 0.0]), atol=1e-5)\n"
            "assert t.allclose(ex2_uv_from_cartesian(A, B, C, C), t.tensor([0.0, 1.0]), atol=1e-5)\n"
            "\n"
            "# Non-canonical triangle: A=(2,1), B=(5,1), C=(2,4).\n"
            "A2 = t.tensor([2.0, 1.0])\n"
            "B2 = t.tensor([5.0, 1.0])\n"
            "C2 = t.tensor([2.0, 4.0])\n"
            "# Test that round-trip reconstructs P.\n"
            "for true_u, true_v in [(0.2, 0.5), (0.1, 0.1), (0.8, 0.0), (0.0, 0.9)]:\n"
            "    P = A2 + true_u * (B2 - A2) + true_v * (C2 - A2)\n"
            "    uv = ex2_uv_from_cartesian(A2, B2, C2, P)\n"
            "    assert t.allclose(uv, t.tensor([true_u, true_v]), atol=1e-5), (\n"
            "        f'roundtrip failed: expected ({true_u}, {true_v}), got {uv}'\n"
            "    )\n"
            "\n"
            "# Reconstruction check: A + u*(B-A) + v*(C-A) must equal P.\n"
            "P_query = t.tensor([3.0, 2.0])\n"
            "uv = ex2_uv_from_cartesian(A2, B2, C2, P_query)\n"
            "P_back = A2 + uv[0] * (B2 - A2) + uv[1] * (C2 - A2)\n"
            "assert t.allclose(P_back, P_query, atol=1e-5), (\n"
            "    f'reconstruction failed: P_back={P_back}, P={P_query}'\n"
            ")"
        ),
        "solution_body": (
            "def ex2_uv_from_cartesian(A: Tensor, B: Tensor, C: Tensor, P: Tensor) -> Tensor:\n"
            "    M = t.stack([B - A, C - A], dim=1)   # (2, 2), edges as columns\n"
            "    return t.linalg.solve(M, P - A)       # (2,)"
        ),
        "solution_notes": (
            "**Why columns, not rows.** Writing the system "
            "`u*(B-A) + v*(C-A) = P-A` in matrix form makes `(B-A)` "
            "and `(C-A)` the COLUMNS of `M` so that `M @ [u, v]^T` "
            "produces the RHS. `t.stack([..., ...], dim=1)` builds "
            "the columns; `dim=0` would build rows and silently "
            "solve the wrong system.\n\n"
            "**Composition with ex1.** Once you have `(u, v)`, the "
            "ex1 predicate (`u >= 0 & v >= 0 & u + v <= 1`) is the "
            "inside test. Together: cartesian-inside-triangle = "
            "this solve + that predicate.\n\n"
            "**Failure mode.** A degenerate triangle (three collinear "
            "vertices) makes `M` singular and the solve raises "
            "`_LinAlgError`. Wrap in `try/except` — exactly the "
            "facet drilled in `try-except-solve`."
        ),
        "extra_imports": [],
    },

    # ─────────────────────────────────────────────────────────────────────
    # 2. try-except-solve  (ex2: list-batched safe-solve, list[Optional[Tensor]])
    # ─────────────────────────────────────────────────────────────────────
    {
        "atom_id": "try-except-solve",
        "subtopic": "LinAlg: try/except solve",
        "topic_folder": TOPIC_GEOM,
        "atom_recap_md": RECAP_TRY_EXCEPT_BATCH,
        "exercise_index": 2,
        "exercise_title": "list-batched safe solve returning list[Optional[Tensor]]",
        "slug": "list-batched-safe-solve",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["try-except", "linalg-solve", "list-batch", "graceful-failure"],
        "kcs": ["per-item-try-except-loop", "return-none-on-singular"],
        "lo": (
            "Apply a `try/except RuntimeError` inside a Python loop over "
            "paired `(A, b)` lists to produce a `list[Optional[Tensor]]` "
            "where each None marks a singular input without aborting the "
            "rest of the batch."
        ),
        "prompt_body": (
            "Implement `ex2_safe_solve_list(As, bs)`.\n\n"
            "- `As`: a `list[Tensor]` of `(n, n)` matrices (n may differ "
            "  across list items).\n"
            "- `bs`: a parallel `list[Tensor]` of RHS vectors.\n"
            "- Return `list[Optional[Tensor]]` of the same length: each "
            "  entry is either `t.linalg.solve(A, b)` or `None` if that "
            "  particular solve raised `RuntimeError`.\n\n"
            "Constraints:\n"
            "- One singular matrix must NOT abort the rest of the loop.\n"
            "- Output length must equal input length.\n"
            "- Order preserved (entry `i` of output corresponds to "
            "  `(As[i], bs[i])`).\n"
            "- Don't try to fix singular inputs — just record None."
        ),
        "stub": (
            "from typing import Optional\n"
            "\n"
            "def ex2_safe_solve_list(As: list, bs: list) -> list:\n"
            '    """Per-item safe solve. Returns list[Optional[Tensor]]."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# Three solves: well-conditioned, singular, well-conditioned.\n"
            "A_good = t.tensor([[3.0, 1.0], [1.0, 2.0]])\n"
            "A_sing = t.tensor([[1.0, 2.0], [2.0, 4.0]])   # rank-1\n"
            "A_id   = t.eye(3)\n"
            "b_good = t.tensor([9.0, 8.0])\n"
            "b_sing = t.tensor([3.0, 6.0])\n"
            "b_id   = t.tensor([1.0, -2.0, 5.0])\n"
            "out = ex2_safe_solve_list([A_good, A_sing, A_id], [b_good, b_sing, b_id])\n"
            "assert isinstance(out, list), f'must return a list, got {type(out).__name__}'\n"
            "assert len(out) == 3, f'len must be 3, got {len(out)}'\n"
            "\n"
            "# Entry 0: well-conditioned -> tensor solution.\n"
            "assert out[0] is not None, 'well-conditioned must not be None'\n"
            "assert t.allclose(A_good @ out[0], b_good, atol=1e-5)\n"
            "# Entry 1: singular -> None.\n"
            "assert out[1] is None, f'singular must be None, got {out[1]!r}'\n"
            "# Entry 2: identity -> b unchanged.\n"
            "assert out[2] is not None\n"
            "assert t.allclose(out[2], b_id, atol=1e-6)\n"
            "\n"
            "# Order preservation: shuffle inputs, output must follow.\n"
            "out2 = ex2_safe_solve_list([A_sing, A_good, A_sing], [b_sing, b_good, b_sing])\n"
            "assert out2[0] is None\n"
            "assert out2[1] is not None and t.allclose(A_good @ out2[1], b_good, atol=1e-5)\n"
            "assert out2[2] is None\n"
            "\n"
            "# Empty input -> empty list (loop boundary).\n"
            "assert ex2_safe_solve_list([], []) == []\n"
            "\n"
            "# All-singular input -> all-None output.\n"
            "out_all_bad = ex2_safe_solve_list([A_sing, A_sing], [b_sing, b_sing])\n"
            "assert out_all_bad == [None, None]\n"
            "\n"
            "# Heterogeneous sizes: a 2x2 followed by a 4x4 should both succeed.\n"
            "A4 = t.eye(4) * 2 + t.ones(4, 4) * 0.05\n"
            "b4 = t.tensor([1.0, 2.0, 3.0, 4.0])\n"
            "out_mixed = ex2_safe_solve_list([A_good, A4], [b_good, b4])\n"
            "assert out_mixed[0] is not None and out_mixed[0].shape == (2,)\n"
            "assert out_mixed[1] is not None and out_mixed[1].shape == (4,)\n"
            "assert t.allclose(A4 @ out_mixed[1], b4, atol=1e-4)\n"
            "\n"
            "# Singular in the MIDDLE must not break the loop on the tail.\n"
            "out_mid = ex2_safe_solve_list(\n"
            "    [A_good, A_sing, A_good, A_sing, A_id],\n"
            "    [b_good, b_sing, b_good, b_sing, b_id],\n"
            ")\n"
            "assert [x is None for x in out_mid] == [False, True, False, True, False]"
        ),
        "solution_body": (
            "from typing import Optional\n"
            "\n"
            "def ex2_safe_solve_list(As, bs):\n"
            "    out = []\n"
            "    for A, b in zip(As, bs):\n"
            "        try:\n"
            "            out.append(t.linalg.solve(A, b))\n"
            "        except RuntimeError:\n"
            "            out.append(None)\n"
            "    return out"
        ),
        "solution_notes": (
            "**Why the try lives INSIDE the loop.** A single "
            "`try` around the whole loop would abort on the first "
            "singular matrix — exactly the behavior we're trying to "
            "avoid. Per-iteration `try/except` is the unit of "
            "graceful-failure here.\n\n"
            "**Why `zip(As, bs)`.** Mixed sizes per iteration means "
            "you can't stack into a batched tensor anyway — the "
            "list loop is the natural shape. If sizes WERE uniform "
            "you'd want the `singular-matrix-mask-trick` atom "
            "instead.\n\n"
            "**Python overhead is fine here.** N is small (a few "
            "hundred at most). The savings from vectorizing don't "
            "offset the API ugliness of NaN-marked tensors. Use the "
            "list-of-Optional form when N < ~10k."
        ),
        "extra_imports": [],
    },

    # ─────────────────────────────────────────────────────────────────────
    # 3. dataclasses-replace-args  (ex2: 2-D grid sweep)
    # ─────────────────────────────────────────────────────────────────────
    {
        "atom_id": "dataclasses-replace-args",
        "subtopic": "Config: dataclasses.replace args",
        "topic_folder": TOPIC_HPARAM,
        "atom_recap_md": RECAP_DATACLASS_GRID,
        "exercise_index": 2,
        "exercise_title": "2-D grid sweep over (lr, batch_size) via replace",
        "slug": "2d-grid-sweep-over-lr-and-batch-size",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["dataclass", "replace", "grid-sweep", "cartesian-product"],
        "kcs": ["dataclasses-replace-keyword-overrides", "grid-sweep-row-major-order"],
        "lo": (
            "Apply a single `dataclasses.replace` call inside a nested "
            "comprehension to produce a 2-D Cartesian-product sweep of "
            "training-args variants ordered row-major over (lr, batch_size)."
        ),
        "prompt_body": (
            "Implement `ex2_make_grid_sweep(base, lrs, batch_sizes)`.\n\n"
            "For every `(lr, bs)` pair in the Cartesian product, build a "
            "fresh `TrainingArgs` with BOTH overrides set in one "
            "`replace` call. Order is **row-major**: outer loop is `lr`, "
            "inner loop is `batch_size`. So `lrs=[a, b], batch_sizes=[1, "
            "2, 3]` produces 6 variants in order: "
            "`(a,1), (a,2), (a,3), (b,1), (b,2), (b,3)`.\n\n"
            "Constraints:\n"
            "- Total variants = `len(lrs) * len(batch_sizes)`.\n"
            "- Each variant uses ONE `replace` call (not nested replaces).\n"
            "- `base` must NOT be mutated.\n"
            "- `__post_init__` validation must still fire (a bad lr "
            "  anywhere in the grid raises `ValueError`).\n"
            "- Empty axis -> empty grid."
        ),
        "stub": (
            "from dataclasses import dataclass, replace\n"
            "\n"
            "@dataclass\n"
            "class TrainingArgs:\n"
            "    lr: float = 1e-3\n"
            "    batch_size: int = 32\n"
            "    epochs: int = 10\n"
            "    optimizer_name: str = 'adam'\n"
            "\n"
            "    def __post_init__(self):\n"
            "        if self.lr <= 0:\n"
            "            raise ValueError(f'lr must be > 0, got {self.lr}')\n"
            "        if self.batch_size < 1:\n"
            "            raise ValueError(f'batch_size must be >= 1, got {self.batch_size}')\n"
            "\n"
            "def ex2_make_grid_sweep(base: 'TrainingArgs', lrs: list, batch_sizes: list) -> list:\n"
            '    """Cartesian-product sweep over (lr, batch_size), row-major."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "base = TrainingArgs(lr=1e-3, batch_size=32, epochs=20, optimizer_name='adamw')\n"
            "\n"
            "# Basic 2x3 grid.\n"
            "lrs = [1e-4, 1e-3]\n"
            "bss = [16, 32, 64]\n"
            "grid = ex2_make_grid_sweep(base, lrs, bss)\n"
            "assert isinstance(grid, list)\n"
            "assert len(grid) == 6, f'2x3 must give 6 variants, got {len(grid)}'\n"
            "\n"
            "# Row-major order: outer=lr, inner=batch_size.\n"
            "expected_pairs = [\n"
            "    (1e-4, 16), (1e-4, 32), (1e-4, 64),\n"
            "    (1e-3, 16), (1e-3, 32), (1e-3, 64),\n"
            "]\n"
            "for v, (lr, bs) in zip(grid, expected_pairs):\n"
            "    assert v.lr == lr, f'order wrong: got lr={v.lr}, expected {lr}'\n"
            "    assert v.batch_size == bs, f'order wrong: got bs={v.batch_size}, expected {bs}'\n"
            "\n"
            "# Untouched fields copy from base.\n"
            "for v in grid:\n"
            "    assert v.epochs == 20\n"
            "    assert v.optimizer_name == 'adamw'\n"
            "    assert isinstance(v, TrainingArgs)\n"
            "    assert v is not base\n"
            "\n"
            "# base not mutated.\n"
            "assert base.lr == 1e-3 and base.batch_size == 32\n"
            "\n"
            "# Distinct objects.\n"
            "ids = [id(v) for v in grid]\n"
            "assert len(set(ids)) == 6, 'each grid variant must be a distinct object'\n"
            "\n"
            "# Validation re-runs (bad lr in the grid raises).\n"
            "try:\n"
            "    ex2_make_grid_sweep(base, [1e-4, -1.0], [16, 32])\n"
            "except ValueError:\n"
            "    pass\n"
            "else:\n"
            "    raise AssertionError('expected ValueError for bad lr in grid')\n"
            "\n"
            "# Empty lr axis -> empty grid.\n"
            "assert ex2_make_grid_sweep(base, [], [16, 32]) == []\n"
            "# Empty bs axis -> empty grid.\n"
            "assert ex2_make_grid_sweep(base, [1e-4, 1e-3], []) == []\n"
            "\n"
            "# Single-element axes -> 1x1 grid of one variant.\n"
            "tiny = ex2_make_grid_sweep(base, [5e-4], [8])\n"
            "assert len(tiny) == 1\n"
            "assert tiny[0].lr == 5e-4 and tiny[0].batch_size == 8"
        ),
        "solution_body": (
            "from dataclasses import dataclass, replace\n"
            "\n"
            "@dataclass\n"
            "class TrainingArgs:\n"
            "    lr: float = 1e-3\n"
            "    batch_size: int = 32\n"
            "    epochs: int = 10\n"
            "    optimizer_name: str = 'adam'\n"
            "\n"
            "    def __post_init__(self):\n"
            "        if self.lr <= 0:\n"
            "            raise ValueError(f'lr must be > 0, got {self.lr}')\n"
            "        if self.batch_size < 1:\n"
            "            raise ValueError(f'batch_size must be >= 1, got {self.batch_size}')\n"
            "\n"
            "def ex2_make_grid_sweep(base, lrs, batch_sizes):\n"
            "    return [\n"
            "        replace(base, lr=lr, batch_size=bs)\n"
            "        for lr in lrs\n"
            "        for bs in batch_sizes\n"
            "    ]"
        ),
        "solution_notes": (
            "**Why row-major over column-major.** Matches "
            "`itertools.product(lrs, batch_sizes)` and numpy "
            "`meshgrid(..., indexing='ij').reshape(-1, 2)`. When "
            "you log sweep results, the natural sort key is "
            "`(lr, bs)` tuples in lexicographic order — that's "
            "exactly what row-major produces.\n\n"
            "**One `replace` per pair, two kwargs.** "
            "`replace(base, lr=lr, batch_size=bs)` passes BOTH "
            "overrides in one call — one allocation, one "
            "`__post_init__` invocation. Nested calls "
            "`replace(replace(base, lr=lr), batch_size=bs)` work "
            "but double the cost.\n\n"
            "**Empty axis -> empty grid.** Either factor zero "
            "makes the Cartesian product empty — the comprehension "
            "naturally returns `[]` because the outer or inner "
            "loop has nothing to iterate. No special case needed."
        ),
        "extra_imports": [],
    },

    # ─────────────────────────────────────────────────────────────────────
    # 4. hparam-precedence-merge  (ex2: filter sentinel None from CLI)
    # ─────────────────────────────────────────────────────────────────────
    {
        "atom_id": "hparam-precedence-merge",
        "subtopic": "Config: hparam precedence merge",
        "topic_folder": TOPIC_HPARAM,
        "atom_recap_md": RECAP_PRECEDENCE_FILTER,
        "exercise_index": 2,
        "exercise_title": "filter sentinel None from CLI before merging",
        "slug": "filter-sentinel-none-from-cli-before-merging",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["config", "precedence", "sentinel-none", "argparse-default"],
        "kcs": ["dict-comprehension-filter-none", "dict-update-later-wins"],
        "lo": (
            "Apply a sentinel-None dict-comprehension filter to "
            "`cli_args` BEFORE chained `dict.update` so that argparse-"
            "style 'flag not provided' Nones don't overwrite values set "
            "by the YAML config layer."
        ),
        "prompt_body": (
            "Implement `ex2_merge_filter_cli(defaults, file_cfg, cli_args)`.\n\n"
            "Same three-layer chain as ex1, but with a twist: in this "
            "merge, `cli_args` came from argparse where ABSENT flags "
            "default to `None`. We don't want those Nones blasting away "
            "values set in the file layer.\n\n"
            "Algorithm:\n"
            "1. Build `cli_filtered = {k: v for k, v in cli_args.items() "
            "  if v is not None}`.\n"
            "2. `out = {}`; `out.update(defaults)`; `out.update(file_cfg)`; "
            "  `out.update(cli_filtered)`.\n"
            "3. Return `out`.\n\n"
            "Constraints:\n"
            "- No mutation of any input dict (including `cli_args`).\n"
            "- Non-None CLI values STILL override file (later-wins).\n"
            "- A file value of `None` should pass through (only CLI is "
            "  filtered).\n"
            "- A file value that's overridden by a non-None CLI value "
            "  must be replaced.\n"
            "- A CLI value of `None` must NOT override the file."
        ),
        "stub": (
            "def ex2_merge_filter_cli(defaults: dict, file_cfg: dict, cli_args: dict) -> dict:\n"
            '    """Three-layer merge; CLI Nones treated as \'flag not provided\'."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# === Sentinel None in CLI does NOT override file ===\n"
            "defaults = {'lr': 1e-3, 'batch_size': 32}\n"
            "file_cfg = {'lr': 3e-4, 'batch_size': 64}\n"
            "cli_args = {'lr': None, 'batch_size': 128}\n"
            "merged = ex2_merge_filter_cli(defaults, file_cfg, cli_args)\n"
            "assert merged == {'lr': 3e-4, 'batch_size': 128}, (\n"
            "    f'CLI None must not override file; got {merged}'\n"
            ")\n"
            "\n"
            "# === Inputs not mutated ===\n"
            "assert defaults == {'lr': 1e-3, 'batch_size': 32}\n"
            "assert file_cfg == {'lr': 3e-4, 'batch_size': 64}\n"
            "assert cli_args == {'lr': None, 'batch_size': 128}\n"
            "\n"
            "# === Non-None CLI value DOES override ===\n"
            "merged = ex2_merge_filter_cli({'lr': 1e-3}, {'lr': 3e-4}, {'lr': 1e-5})\n"
            "assert merged == {'lr': 1e-5}, f'CLI non-None must win, got {merged}'\n"
            "\n"
            "# === All-None CLI -> file wins ===\n"
            "merged = ex2_merge_filter_cli({'a': 1}, {'a': 2, 'b': 7}, {'a': None, 'b': None})\n"
            "assert merged == {'a': 2, 'b': 7}, f'all-None CLI should be noop, got {merged}'\n"
            "\n"
            "# === None in FILE layer passes through (only CLI is filtered) ===\n"
            "merged = ex2_merge_filter_cli({'a': 1}, {'a': None}, {})\n"
            "assert merged == {'a': None}, f'file None should pass, got {merged}'\n"
            "\n"
            "# === CLI introduces a new key with non-None value ===\n"
            "merged = ex2_merge_filter_cli({'a': 1}, {}, {'b': 2})\n"
            "assert merged == {'a': 1, 'b': 2}\n"
            "\n"
            "# === CLI introduces a new key but value is None -> dropped ===\n"
            "merged = ex2_merge_filter_cli({'a': 1}, {}, {'b': None})\n"
            "assert merged == {'a': 1}, f'CLI-introduced None must drop, got {merged}'\n"
            "\n"
            "# === Three-layer override chain still works for non-None CLI ===\n"
            "merged = ex2_merge_filter_cli({'lr': 1.0}, {'lr': 2.0}, {'lr': 3.0})\n"
            "assert merged['lr'] == 3.0\n"
            "\n"
            "# === Result is a fresh dict ===\n"
            "d = {'x': 1}\n"
            "merged = ex2_merge_filter_cli(d, {}, {})\n"
            "merged['y'] = 99\n"
            "assert 'y' not in d\n"
            "\n"
            "# === Mixed: file sets 3 keys, CLI overrides 1 and Nones 1 and adds 1 ===\n"
            "out = ex2_merge_filter_cli(\n"
            "    {'lr': 1e-3, 'bs': 16, 'opt': 'adam'},\n"
            "    {'lr': 3e-4, 'bs': 64, 'opt': 'adamw', 'wd': 1e-2},\n"
            "    {'lr': None, 'bs': 128, 'opt': None, 'epochs': 5},\n"
            ")\n"
            "assert out == {'lr': 3e-4, 'bs': 128, 'opt': 'adamw', 'wd': 1e-2, 'epochs': 5}, out"
        ),
        "solution_body": (
            "def ex2_merge_filter_cli(defaults, file_cfg, cli_args):\n"
            "    cli_filtered = {k: v for k, v in cli_args.items() if v is not None}\n"
            "    out = {}\n"
            "    out.update(defaults)\n"
            "    out.update(file_cfg)\n"
            "    out.update(cli_filtered)\n"
            "    return out"
        ),
        "solution_notes": (
            "**Why filter only the CLI layer.** A YAML config value "
            "of `None` is an explicit user choice — the file format "
            "is expressive enough to mean what it says. The CLI "
            "layer has the sentinel-None problem because argparse "
            "defaults absent flags to None.\n\n"
            "**Comprehension vs `{**d}` spread.** "
            "`{k: v for k, v in cli.items() if v is not None}` is "
            "the clearest filter. `{k: v for k, v in cli.items() if "
            "v != None}` would catch numpy/pandas NaT-style "
            "objects too — usually not what you want.\n\n"
            "**Choosing between ex1 (strict) and ex2 (sentinel) "
            "behavior.** Match your CLI parser. argparse with "
            "default=None -> ex2. Hand-rolled parsers that only "
            "include user-typed flags -> ex1. Picking the wrong "
            "one is one of the most common silent precedence bugs "
            "in training scripts."
        ),
        "extra_imports": [],
    },

    # ─────────────────────────────────────────────────────────────────────
    # 5. nested-param-group-loop  (ex2: SGD with weight decay per group)
    # ─────────────────────────────────────────────────────────────────────
    {
        "atom_id": "nested-param-group-loop",
        "subtopic": "Config: nested param-group loop",
        "topic_folder": TOPIC_HPARAM,
        "atom_recap_md": RECAP_NESTED_LOOP_WD,
        "exercise_index": 2,
        "exercise_title": "manual SGD-with-weight-decay using per-group hparams",
        "slug": "manual-sgd-with-weight-decay-per-group",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["param-groups", "weight-decay", "sgd", "per-group-hparams"],
        "kcs": ["read-multiple-group-hparams", "weight-decay-folded-into-grad"],
        "lo": (
            "Apply the nested `param_groups → params` loop to manually "
            "perform one SGD step that reads BOTH `lr` and "
            "`weight_decay` per group and folds the decay term into the "
            "gradient before the update."
        ),
        "prompt_body": (
            "Implement `ex2_manual_sgd_wd_step(optimizer)`.\n\n"
            "Like ex1, but each group now ALSO has a `weight_decay` "
            "hparam. The step rule per parameter is:\n"
            "```\n"
            "g_effective = p.grad + wd * p.data       (only if wd != 0)\n"
            "p.data <- p.data - lr * g_effective\n"
            "```\n"
            "Algorithm:\n"
            "1. Outer loop over `optimizer.param_groups`.\n"
            "2. Read `lr = group['lr']` and `wd = group.get('weight_decay', 0.0)`.\n"
            "3. Inner loop over `group['params']`.\n"
            "4. Skip `p.grad is None`.\n"
            "5. If `wd != 0`, use `g = p.grad + wd * p.data`; "
            "  otherwise `g = p.grad`.\n"
            "6. `p.data.add_(g, alpha=-lr)`.\n\n"
            "Do NOT call `optimizer.step()`. Output: `None` "
            "(in-place mutation)."
        ),
        "stub": (
            "def ex2_manual_sgd_wd_step(optimizer):\n"
            '    """Manual SGD step with per-group weight decay."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# === Single group, weight_decay=0 ===\n"
            "p = t.nn.Parameter(t.tensor([1.0, 2.0, 3.0]))\n"
            "opt = t.optim.SGD([p], lr=0.1, weight_decay=0.0)\n"
            "p.grad = t.tensor([10.0, 20.0, 30.0])\n"
            "ex2_manual_sgd_wd_step(opt)\n"
            "expected = t.tensor([1.0, 2.0, 3.0]) - 0.1 * t.tensor([10.0, 20.0, 30.0])\n"
            "assert t.allclose(p.detach(), expected, atol=1e-6), (\n"
            "    f'wd=0 step wrong: got {p.detach()}, expected {expected}'\n"
            ")\n"
            "\n"
            "# === Single group, weight_decay=0.1 ===\n"
            "p = t.nn.Parameter(t.tensor([1.0, 2.0, 3.0]))\n"
            "opt = t.optim.SGD([p], lr=0.1, weight_decay=0.1)\n"
            "p.grad = t.tensor([1.0, 1.0, 1.0])\n"
            "ex2_manual_sgd_wd_step(opt)\n"
            "# g_eff = grad + wd * p = [1,1,1] + 0.1*[1,2,3] = [1.1, 1.2, 1.3]\n"
            "# new p = old p - lr * g_eff = [1,2,3] - 0.1*[1.1,1.2,1.3]\n"
            "expected = t.tensor([1.0, 2.0, 3.0]) - 0.1 * t.tensor([1.1, 1.2, 1.3])\n"
            "assert t.allclose(p.detach(), expected, atol=1e-6), (\n"
            "    f'wd=0.1 step wrong: got {p.detach()}, expected {expected}'\n"
            ")\n"
            "\n"
            "# === Two groups with DIFFERENT weight decay ===\n"
            "p1 = t.nn.Parameter(t.tensor([2.0, 2.0]))\n"
            "p2 = t.nn.Parameter(t.tensor([2.0, 2.0]))\n"
            "opt = t.optim.SGD([\n"
            "    {'params': [p1], 'lr': 0.1, 'weight_decay': 0.0},\n"
            "    {'params': [p2], 'lr': 0.1, 'weight_decay': 0.5},\n"
            "])\n"
            "p1.grad = t.zeros(2)                    # only decay drives p1\n"
            "p2.grad = t.zeros(2)                    # only decay drives p2\n"
            "ex2_manual_sgd_wd_step(opt)\n"
            "# p1: g_eff = 0 + 0 * p1 = 0 -> p1 unchanged.\n"
            "# p2: g_eff = 0 + 0.5 * p2 = [1, 1]; new p2 = [2,2] - 0.1*[1,1] = [1.9, 1.9].\n"
            "assert t.allclose(p1.detach(), t.tensor([2.0, 2.0]), atol=1e-6), (\n"
            "    f'wd=0 group should not move: got {p1.detach()}'\n"
            ")\n"
            "assert t.allclose(p2.detach(), t.tensor([1.9, 1.9]), atol=1e-6), (\n"
            "    f'wd=0.5 group wrong: got {p2.detach()}'\n"
            ")\n"
            "\n"
            "# === grad=None must be SKIPPED ===\n"
            "p_a = t.nn.Parameter(t.tensor([5.0]))\n"
            "p_b = t.nn.Parameter(t.tensor([5.0]))\n"
            "opt = t.optim.SGD([p_a, p_b], lr=0.1, weight_decay=0.2)\n"
            "p_a.grad = t.tensor([1.0])\n"
            "# p_b.grad left as None\n"
            "ex2_manual_sgd_wd_step(opt)\n"
            "# p_a: g_eff = 1 + 0.2 * 5 = 2; new p_a = 5 - 0.1*2 = 4.8.\n"
            "assert t.allclose(p_a.detach(), t.tensor([4.8]), atol=1e-6), p_a.detach()\n"
            "assert t.allclose(p_b.detach(), t.tensor([5.0]), atol=1e-6), (\n"
            "    f'p_b grad=None should not move, got {p_b.detach()}'\n"
            ")\n"
            "\n"
            "# === group.get('weight_decay', 0.0) — groups without WD key default to 0 ===\n"
            "# Construct two groups, the second one without weight_decay specified.\n"
            "p_x = t.nn.Parameter(t.tensor([1.0, 1.0]))\n"
            "p_y = t.nn.Parameter(t.tensor([1.0, 1.0]))\n"
            "# Build manually so one group is missing 'weight_decay'.\n"
            "opt = t.optim.SGD([\n"
            "    {'params': [p_x], 'lr': 0.1, 'weight_decay': 0.0},\n"
            "    {'params': [p_y], 'lr': 0.1, 'weight_decay': 0.0},\n"
            "])\n"
            "# Now simulate a group WITHOUT weight_decay key by popping it:\n"
            "del opt.param_groups[1]['weight_decay']\n"
            "p_x.grad = t.ones(2)\n"
            "p_y.grad = t.ones(2)\n"
            "ex2_manual_sgd_wd_step(opt)\n"
            "# Both groups should behave as wd=0.\n"
            "assert t.allclose(p_x.detach(), t.tensor([0.9, 0.9]), atol=1e-6), p_x.detach()\n"
            "assert t.allclose(p_y.detach(), t.tensor([0.9, 0.9]), atol=1e-6), (\n"
            "    f'missing weight_decay key should default to 0; got {p_y.detach()}'\n"
            ")\n"
            "\n"
            "# === In-place — same id and storage ===\n"
            "p = t.nn.Parameter(t.zeros(4))\n"
            "orig_ptr = p.data_ptr()\n"
            "opt = t.optim.SGD([p], lr=0.1, weight_decay=0.01)\n"
            "p.grad = t.ones(4)\n"
            "ex2_manual_sgd_wd_step(opt)\n"
            "assert p.data_ptr() == orig_ptr, 'param storage was reallocated'"
        ),
        "solution_body": (
            "def ex2_manual_sgd_wd_step(optimizer):\n"
            "    for group in optimizer.param_groups:\n"
            "        lr = group['lr']\n"
            "        wd = group.get('weight_decay', 0.0)\n"
            "        for p in group['params']:\n"
            "            if p.grad is None:\n"
            "                continue\n"
            "            g = p.grad\n"
            "            if wd != 0.0:\n"
            "                g = g + wd * p.data\n"
            "            p.data.add_(g, alpha=-lr)"
        ),
        "solution_notes": (
            "**`group.get('weight_decay', 0.0)` not `group["
            "'weight_decay']`.** Some groups (e.g. the no-decay "
            "bias group) intentionally omit `weight_decay`. The "
            "`.get` form gives 0 as the natural default and "
            "doesn't crash with KeyError.\n\n"
            "**Why `g = p.grad + wd * p.data`, not `p.grad.add_("
            "wd * p.data)`.** The latter mutates the user's "
            "gradient tensor — a side effect that breaks "
            "gradient accumulation and gradient-norm logging. "
            "Allocate a fresh `g`; the step's in-place mutation "
            "is the only side effect this function should have.\n\n"
            "**Why fold decay into grad, not into the step.** "
            "Mathematically equivalent for vanilla SGD, but "
            "matches the structure of `torch.optim.SGD`'s own "
            "source — easier to read against the reference. "
            "AdamW takes the opposite choice (decoupled decay) "
            "for very different numerical reasons."
        ),
        "extra_imports": [],
    },

    # ─────────────────────────────────────────────────────────────────────
    # 6. optimizer-class-dispatch  (ex2: register a new optimizer at runtime)
    # ─────────────────────────────────────────────────────────────────────
    {
        "atom_id": "optimizer-class-dispatch",
        "subtopic": "Config: Optimizer class dispatch",
        "topic_folder": TOPIC_HPARAM,
        "atom_recap_md": RECAP_OPTIMIZER_REGISTER,
        "exercise_index": 2,
        "exercise_title": "register a new optimizer at runtime",
        "slug": "register-a-new-optimizer-at-runtime",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["dispatch-table", "registry", "plugin", "registration"],
        "kcs": ["registry-mutation-by-name", "registration-time-validation"],
        "lo": (
            "Apply a runtime-registration function that mutates the "
            "module-level dispatch dict so that subsequent "
            "construction calls can resolve the new optimizer name "
            "without modifying the factory function."
        ),
        "prompt_body": (
            "Implement TWO things:\n\n"
            "1. `ex2_register_optimizer(name, cls)` — mutate the "
            "  module-level `OPTIMIZER_CLASSES` dict to add (or "
            "  overwrite) the entry `{name: cls}`. Validate that "
            "  `cls` is a class AND a subclass of "
            "  `torch.optim.Optimizer`; reject anything else with "
            "  `TypeError`.\n"
            "2. `ex2_build_optimizer(name, params, lr)` — the same "
            "  factory as ex1 but reading from the (potentially "
            "  extended) `OPTIMIZER_CLASSES` registry.\n\n"
            "Constraints:\n"
            "- `OPTIMIZER_CLASSES` must start with the 3 base "
            "  entries (`sgd`, `adam`, `adamw`).\n"
            "- Registration must be IDEMPOTENT for the same `(name, "
            "  cls)` pair.\n"
            "- Overwriting an existing entry is allowed (e.g. swap "
            "  in a custom AdamW).\n"
            "- Unknown name at build time raises `KeyError`.\n"
            "- Non-class or non-Optimizer-subclass `cls` raises "
            "  `TypeError` at REGISTRATION (not at build)."
        ),
        "stub": (
            "OPTIMIZER_CLASSES = {\n"
            "    'sgd':   t.optim.SGD,\n"
            "    'adam':  t.optim.Adam,\n"
            "    'adamw': t.optim.AdamW,\n"
            "}\n"
            "\n"
            "def ex2_register_optimizer(name: str, cls):\n"
            '    """Add or overwrite OPTIMIZER_CLASSES[name] = cls."""\n'
            "    raise NotImplementedError()\n"
            "\n"
            "def ex2_build_optimizer(name: str, params, lr: float):\n"
            '    """Dispatch using the (extended) registry."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# === Registry starts with the 3 base entries ===\n"
            "assert set(OPTIMIZER_CLASSES.keys()) >= {'sgd', 'adam', 'adamw'}, (\n"
            "    f'base entries missing: {set(OPTIMIZER_CLASSES.keys())}'\n"
            ")\n"
            "\n"
            "# === Register a custom optimizer (a real torch.optim subclass) ===\n"
            "class MyCustomSGD(t.optim.SGD):\n"
            "    pass\n"
            "\n"
            "ex2_register_optimizer('mycustom', MyCustomSGD)\n"
            "assert 'mycustom' in OPTIMIZER_CLASSES\n"
            "assert OPTIMIZER_CLASSES['mycustom'] is MyCustomSGD\n"
            "\n"
            "# === Build using the newly-registered name ===\n"
            "param = t.nn.Parameter(t.randn(3))\n"
            "opt = ex2_build_optimizer('mycustom', [param], lr=0.01)\n"
            "assert isinstance(opt, MyCustomSGD), (\n"
            "    f'expected MyCustomSGD instance, got {type(opt).__name__}'\n"
            ")\n"
            "assert opt.param_groups[0]['lr'] == 0.01\n"
            "\n"
            "# === Idempotent re-registration with same (name, cls) ===\n"
            "ex2_register_optimizer('mycustom', MyCustomSGD)\n"
            "ex2_register_optimizer('mycustom', MyCustomSGD)\n"
            "assert OPTIMIZER_CLASSES['mycustom'] is MyCustomSGD\n"
            "\n"
            "# === Overwrite an existing entry ===\n"
            "class MyAdamW(t.optim.AdamW):\n"
            "    pass\n"
            "\n"
            "old = OPTIMIZER_CLASSES['adamw']\n"
            "ex2_register_optimizer('adamw', MyAdamW)\n"
            "assert OPTIMIZER_CLASSES['adamw'] is MyAdamW, 'overwrite must take effect'\n"
            "# Restore for downstream sanity.\n"
            "ex2_register_optimizer('adamw', old)\n"
            "assert OPTIMIZER_CLASSES['adamw'] is old\n"
            "\n"
            "# === Non-class value -> TypeError ===\n"
            "try:\n"
            "    ex2_register_optimizer('bad1', 'not a class')\n"
            "except TypeError:\n"
            "    pass\n"
            "else:\n"
            "    raise AssertionError('expected TypeError for non-class registration')\n"
            "assert 'bad1' not in OPTIMIZER_CLASSES, 'failed registration must not leak'\n"
            "\n"
            "# === Class that is NOT an Optimizer subclass -> TypeError ===\n"
            "class NotAnOptimizer:\n"
            "    pass\n"
            "\n"
            "try:\n"
            "    ex2_register_optimizer('bad2', NotAnOptimizer)\n"
            "except TypeError:\n"
            "    pass\n"
            "else:\n"
            "    raise AssertionError('expected TypeError for non-Optimizer class')\n"
            "assert 'bad2' not in OPTIMIZER_CLASSES, 'failed registration must not leak'\n"
            "\n"
            "# === Unknown name at build time -> KeyError ===\n"
            "try:\n"
            "    ex2_build_optimizer('does-not-exist', [t.nn.Parameter(t.randn(2))], lr=1e-3)\n"
            "except KeyError as e:\n"
            "    assert 'does-not-exist' in str(e), f'KeyError should mention name, got {e!r}'\n"
            "else:\n"
            "    raise AssertionError('expected KeyError for unknown name')\n"
            "\n"
            "# === Built optimizer actually steps ===\n"
            "p = t.nn.Parameter(t.zeros(2))\n"
            "opt = ex2_build_optimizer('sgd', [p], lr=0.1)\n"
            "p.grad = t.tensor([1.0, -2.0])\n"
            "opt.step()\n"
            "assert t.allclose(p.detach(), t.tensor([-0.1, 0.2]), atol=1e-6)"
        ),
        "solution_body": (
            "OPTIMIZER_CLASSES = {\n"
            "    'sgd':   t.optim.SGD,\n"
            "    'adam':  t.optim.Adam,\n"
            "    'adamw': t.optim.AdamW,\n"
            "}\n"
            "\n"
            "def ex2_register_optimizer(name, cls):\n"
            "    if not isinstance(cls, type):\n"
            "        raise TypeError(\n"
            "            f'cls must be a class, got {type(cls).__name__}'\n"
            "        )\n"
            "    if not issubclass(cls, t.optim.Optimizer):\n"
            "        raise TypeError(\n"
            "            f'cls must subclass torch.optim.Optimizer, got {cls.__name__}'\n"
            "        )\n"
            "    OPTIMIZER_CLASSES[name] = cls\n"
            "\n"
            "def ex2_build_optimizer(name, params, lr):\n"
            "    cls = OPTIMIZER_CLASSES[name]\n"
            "    return cls(params, lr=lr)"
        ),
        "solution_notes": (
            "**Validation at registration, not at build.** The "
            "registration call is where the bad class entered the "
            "system — the stack trace there points at the plugin "
            "loader. Validating at build time hides the cause "
            "behind whatever training run hits the bad name first.\n\n"
            "**Why two `TypeError` branches.** "
            "`isinstance(cls, type)` catches non-classes (strings, "
            "functions, instances) — `issubclass` would raise its "
            "own `TypeError` for those, but the message is cryptic. "
            "The explicit guard gives a clear error.\n\n"
            "**Idempotence is free.** `d[name] = cls` is "
            "idempotent for the same `(name, cls)` pair — re-"
            "registration just rewrites the same value. No need "
            "for an explicit 'is this already there?' check.\n\n"
            "**The registry IS the API.** Plugins can `from "
            "your_module import OPTIMIZER_CLASSES` and use the "
            "registry directly (e.g. to enumerate available "
            "optimizers for a CLI `--help`). That's the win over "
            "an `if/elif` chain — the data is inspectable."
        ),
        "extra_imports": [],
    },

    # ─────────────────────────────────────────────────────────────────────
    # 7. param-group-dict-list  (ex2: three-group no-decay biases split)
    # ─────────────────────────────────────────────────────────────────────
    {
        "atom_id": "param-group-dict-list",
        "subtopic": "Config: param-group dict list",
        "topic_folder": TOPIC_HPARAM,
        "atom_recap_md": RECAP_THREE_GROUP_NO_DECAY,
        "exercise_index": 2,
        "exercise_title": "three-group split — no-decay biases + decay-weights + head LR",
        "slug": "three-group-no-decay-biases-plus-head-lr",
        "bloom_level": "Apply",
        "difficulty_num": 4,
        "difficulty_dots": "🔴🔴🔴🔴⚪",
        "keywords": ["param-groups", "no-decay-biases", "weight-decay", "fine-tuning"],
        "kcs": ["rank-based-param-split", "param-group-multi-hparam-override"],
        "lo": (
            "Apply the `[{'params': ..., 'lr': ..., 'weight_decay': "
            "...}, ...]` construction to split a model into THREE "
            "param groups (encoder-decay, encoder-no-decay, head) "
            "using `p.dim() > 1` as the decay-eligibility rule."
        ),
        "prompt_body": (
            "Implement `ex2_make_three_groups(encoder, head, "
            "encoder_lr, head_lr, weight_decay)`.\n\n"
            "Build a 3-element list of param-group dicts:\n"
            "1. **encoder-decay**: encoder params with "
            "  `p.dim() > 1` (weight matrices), `lr=encoder_lr`, "
            "  `weight_decay=weight_decay`.\n"
            "2. **encoder-no-decay**: encoder params with "
            "  `p.dim() <= 1` (biases, LayerNorm scale), "
            "  `lr=encoder_lr`, `weight_decay=0.0`.\n"
            "3. **head**: ALL head params (no further split), "
            "  `lr=head_lr`, `weight_decay=weight_decay`.\n\n"
            "Order matters: encoder-decay first, then encoder-"
            "no-decay, then head.\n\n"
            "Constraints:\n"
            "- Sum of `len(g['params'])` across the 3 groups must "
            "  equal total params in `encoder` + `head` (no param "
            "  silently dropped, none duplicated).\n"
            "- Empty group lists are allowed (e.g. an encoder with "
            "  no biases produces an empty group 2)."
        ),
        "stub": (
            "def ex2_make_three_groups(encoder, head, encoder_lr: float,\n"
            "                         head_lr: float, weight_decay: float):\n"
            '    """Three-group param-list split: decay/no-decay encoder + head."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "import torch.nn as nn\n"
            "\n"
            "# Encoder: 2 Linears (4 params: 2 weights rank-2, 2 biases rank-1).\n"
            "encoder = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 16))\n"
            "head = nn.Linear(16, 3)   # 1 weight (rank 2) + 1 bias (rank 1)\n"
            "\n"
            "groups = ex2_make_three_groups(\n"
            "    encoder, head,\n"
            "    encoder_lr=1e-4, head_lr=1e-2, weight_decay=0.01,\n"
            ")\n"
            "assert isinstance(groups, list), f'must return a list, got {type(groups).__name__}'\n"
            "assert len(groups) == 3, f'must return 3 groups, got {len(groups)}'\n"
            "\n"
            "# === Each group is a dict with the right keys ===\n"
            "for i, g in enumerate(groups):\n"
            "    assert isinstance(g, dict)\n"
            "    for key in ['params', 'lr', 'weight_decay']:\n"
            "        assert key in g, f'group {i} missing {key!r}'\n"
            "\n"
            "# === Hparam assignment ===\n"
            "# group 0 = encoder-decay\n"
            "assert groups[0]['lr'] == 1e-4\n"
            "assert groups[0]['weight_decay'] == 0.01\n"
            "# group 1 = encoder-no-decay\n"
            "assert groups[1]['lr'] == 1e-4\n"
            "assert groups[1]['weight_decay'] == 0.0\n"
            "# group 2 = head\n"
            "assert groups[2]['lr'] == 1e-2\n"
            "assert groups[2]['weight_decay'] == 0.01\n"
            "\n"
            "# === Rank-based split: only rank>1 params in decay groups ===\n"
            "for p in groups[0]['params']:\n"
            "    assert p.dim() > 1, f'decay group should only have rank>1; got dim={p.dim()}'\n"
            "for p in groups[1]['params']:\n"
            "    assert p.dim() <= 1, f'no-decay group should only have rank<=1; got dim={p.dim()}'\n"
            "\n"
            "# === Counts match the model structure ===\n"
            "# Encoder: 2 weight matrices (rank-2), 2 biases (rank-1).\n"
            "assert len(groups[0]['params']) == 2, f'encoder-decay count: {len(groups[0][\"params\"])}'\n"
            "assert len(groups[1]['params']) == 2, f'encoder-no-decay count: {len(groups[1][\"params\"])}'\n"
            "# Head: 1 weight + 1 bias = 2 params total.\n"
            "assert len(groups[2]['params']) == 2, f'head count: {len(groups[2][\"params\"])}'\n"
            "\n"
            "# === Sum of params equals total model params ===\n"
            "total_split = sum(len(g['params']) for g in groups)\n"
            "total_model = len(list(encoder.parameters())) + len(list(head.parameters()))\n"
            "assert total_split == total_model, (\n"
            "    f'param count mismatch: split has {total_split}, model has {total_model}'\n"
            ")\n"
            "\n"
            "# === Plug into a real optimizer; per-group hparams round-trip ===\n"
            "opt = t.optim.AdamW(groups)\n"
            "assert len(opt.param_groups) == 3\n"
            "assert opt.param_groups[0]['weight_decay'] == 0.01\n"
            "assert opt.param_groups[1]['weight_decay'] == 0.0\n"
            "assert opt.param_groups[2]['weight_decay'] == 0.01\n"
            "assert opt.param_groups[0]['lr'] == 1e-4\n"
            "assert opt.param_groups[2]['lr'] == 1e-2\n"
            "\n"
            "# === No duplicates: every param identity appears in exactly one group ===\n"
            "all_param_ids = []\n"
            "for g in groups:\n"
            "    for p in g['params']:\n"
            "        all_param_ids.append(id(p))\n"
            "assert len(all_param_ids) == len(set(all_param_ids)), 'param appears in multiple groups'\n"
            "\n"
            "# === Encoder without biases (LayerNorm-only, or biases=False) produces empty group 2 ===\n"
            "encoder_no_bias = nn.Sequential(nn.Linear(8, 16, bias=False), nn.Linear(16, 16, bias=False))\n"
            "head2 = nn.Linear(16, 3)\n"
            "groups2 = ex2_make_three_groups(encoder_no_bias, head2, 1e-4, 1e-2, 0.01)\n"
            "assert len(groups2[1]['params']) == 0, (\n"
            "    f'bias-less encoder should produce empty no-decay group, got {len(groups2[1][\"params\"])}'\n"
            ")\n"
            "# Decay group still has the 2 weight matrices.\n"
            "assert len(groups2[0]['params']) == 2"
        ),
        "solution_body": (
            "def ex2_make_three_groups(encoder, head, encoder_lr,\n"
            "                         head_lr, weight_decay):\n"
            "    enc_params = list(encoder.parameters())\n"
            "    head_params = list(head.parameters())\n"
            "    enc_decay = [p for p in enc_params if p.dim() > 1]\n"
            "    enc_no_decay = [p for p in enc_params if p.dim() <= 1]\n"
            "    return [\n"
            "        {'params': enc_decay,    'lr': encoder_lr, 'weight_decay': weight_decay},\n"
            "        {'params': enc_no_decay, 'lr': encoder_lr, 'weight_decay': 0.0},\n"
            "        {'params': head_params,  'lr': head_lr,    'weight_decay': weight_decay},\n"
            "    ]"
        ),
        "solution_notes": (
            "**Why `p.dim() > 1` as the decay rule.** Biases are "
            "rank-1 (shape `(out_features,)`), LayerNorm `weight` "
            "is rank-1 (shape `(features,)`), and BatchNorm `weight` "
            "/ `bias` are rank-1. Weight MATRICES are rank-2 or "
            "higher (Linear weight is `(out, in)`; Conv2d weight is "
            "`(out, in, kh, kw)`). The `> 1` rule cleanly separates "
            "scale/shift params from learned-projection params.\n\n"
            "**Why the head isn't split.** Conventionally, the head "
            "is small enough that splitting weight-decay inside it "
            "doesn't measurably help — and a fresh head is "
            "initialized with random values that wants the regularize. "
            "If you DO want a four-group split (head-decay + head-"
            "no-decay), the same rule applies — just two more "
            "comprehensions.\n\n"
            "**Empty groups are FINE.** Empty `'params': []` is "
            "valid input to `torch.optim` — the group is just a "
            "no-op every step. Letting an empty group survive "
            "(rather than special-casing it out) keeps the code "
            "structure regular and lets logging code count three "
            "groups every time."
        ),
        "extra_imports": [],
    },

    # ─────────────────────────────────────────────────────────────────────
    # 8. params-iterable-vs-groups  (ex2: nn.Parameter subclass + reuse)
    # ─────────────────────────────────────────────────────────────────────
    {
        "atom_id": "params-iterable-vs-groups",
        "subtopic": "Config: params iterable vs groups",
        "topic_folder": TOPIC_HPARAM,
        "atom_recap_md": RECAP_PARAMS_DISPATCH_DEEP,
        "exercise_index": 2,
        "exercise_title": "polymorphic dispatch handles nn.Parameter and re-iterates safely",
        "slug": "polymorphic-dispatch-handles-nn-parameter",
        "bloom_level": "Analyze",
        "difficulty_num": 4,
        "difficulty_dots": "🔴🔴🔴🔴⚪",
        "keywords": ["polymorphism", "isinstance", "nn-Parameter", "iterator-exhaustion"],
        "kcs": ["isinstance-vs-type-for-tensor-dispatch", "materialize-iterator-before-peek"],
        "lo": (
            "Analyze the `params=` dispatch corner cases so that "
            "`nn.Parameter` (a Tensor SUBCLASS) routes through the "
            "Tensor branch via `isinstance`, single-use generators "
            "are materialized once before peeking, and the returned "
            "groups are fresh dicts (no aliasing the caller's input)."
        ),
        "prompt_body": (
            "Implement `ex2_normalize_params_robust(params, "
            "default_lr)`.\n\n"
            "Same job as ex1's `ex1_normalize_params` but pinned on "
            "TWO subtle correctness facets the ex1 spec didn't probe:\n\n"
            "1. **`nn.Parameter` MUST route through the Tensor "
            "  branch.** Use `isinstance(first, t.Tensor)` (not "
            "  `type(first) is t.Tensor`) so that `nn.Parameter` "
            "  (a Tensor subclass) is treated as a tensor.\n"
            "2. **Materialize the iterable BEFORE peeking.** If "
            "  `params` is a single-use generator, calling "
            "  `next(iter(params))` then iterating again would lose "
            "  the first element. Materialize to a list once.\n"
            "3. **Group dicts in the output are FRESH dicts (shallow "
            "  copies) — the caller's input dicts MUST NOT be "
            "  mutated.** Adding a fallback `'lr'` key to a "
            "  caller-owned dict is a side effect we want to avoid.\n\n"
            "Inputs / outputs match ex1: returns `list[dict]` where "
            "each dict has at least `'params'` and `'lr'`. Empty "
            "input -> `ValueError('empty')`. Wrong element type -> "
            "`TypeError`."
        ),
        "stub": (
            "def ex2_normalize_params_robust(params, default_lr: float):\n"
            '    """Polymorphic params dispatch (ex2 deepening)."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# === nn.Parameter is a Tensor subclass -> Tensor branch ===\n"
            "p1 = t.nn.Parameter(t.randn(3))\n"
            "p2 = t.nn.Parameter(t.randn(2, 4))\n"
            "out = ex2_normalize_params_robust([p1, p2], default_lr=1e-3)\n"
            "assert isinstance(out, list)\n"
            "assert len(out) == 1, 'flat-tensor mode should yield 1 group'\n"
            "assert out[0]['lr'] == 1e-3\n"
            "assert out[0]['params'][0] is p1\n"
            "assert out[0]['params'][1] is p2\n"
            "\n"
            "# Make sure the dispatch isn't using `type(...) is t.Tensor` —\n"
            "# nn.Parameter has `type() is t.nn.Parameter`, NOT `t.Tensor`.\n"
            "assert type(p1) is not t.Tensor, 'sanity: nn.Parameter has its own type'\n"
            "assert isinstance(p1, t.Tensor), 'sanity: nn.Parameter is-a Tensor'\n"
            "\n"
            "# Module.parameters() generator -> still works.\n"
            "mod = t.nn.Linear(4, 6)\n"
            "out = ex2_normalize_params_robust(mod.parameters(), default_lr=5e-4)\n"
            "assert len(out) == 1\n"
            "assert len(out[0]['params']) == 2, 'Linear has weight + bias'\n"
            "assert out[0]['lr'] == 5e-4\n"
            "# All entries are nn.Parameter instances.\n"
            "for p in out[0]['params']:\n"
            "    assert isinstance(p, t.nn.Parameter)\n"
            "\n"
            "# === Single-use generator must be materialized once ===\n"
            "gen = (t.nn.Parameter(t.randn(2)) for _ in range(4))\n"
            "out = ex2_normalize_params_robust(gen, default_lr=1e-3)\n"
            "assert len(out[0]['params']) == 4, (\n"
            "    f'generator should yield 4 params; got {len(out[0][\"params\"])}'\n"
            ")\n"
            "\n"
            "# === Group dicts: missing lr falls through to default_lr ===\n"
            "p3 = t.nn.Parameter(t.randn(3))\n"
            "p4 = t.nn.Parameter(t.randn(3))\n"
            "g1 = {'params': [p3], 'lr': 1e-4}\n"
            "g2 = {'params': [p4]}\n"
            "out = ex2_normalize_params_robust([g1, g2], default_lr=2e-3)\n"
            "assert len(out) == 2\n"
            "assert out[0]['lr'] == 1e-4\n"
            "assert out[1]['lr'] == 2e-3\n"
            "\n"
            "# === Caller's input dicts MUST NOT be mutated ===\n"
            "assert 'lr' not in g2, (\n"
            "    f'g2 should be unchanged; got keys {list(g2.keys())} — '\n"
            "    'normalize must shallow-copy each group dict, not mutate in place'\n"
            ")\n"
            "# Output group is a DIFFERENT dict than the input.\n"
            "assert out[1] is not g2, 'output group must be a fresh dict'\n"
            "# But the 'params' list reference can be shared (no spec on that).\n"
            "\n"
            "# === Empty -> ValueError ===\n"
            "try:\n"
            "    ex2_normalize_params_robust([], default_lr=1e-3)\n"
            "except ValueError:\n"
            "    pass\n"
            "else:\n"
            "    raise AssertionError('expected ValueError on empty')\n"
            "\n"
            "# === Bad element type -> TypeError ===\n"
            "try:\n"
            "    ex2_normalize_params_robust([42, 43], default_lr=1e-3)\n"
            "except TypeError:\n"
            "    pass\n"
            "else:\n"
            "    raise AssertionError('expected TypeError for int element')\n"
            "\n"
            "# === Feeds into a real torch.optim, both forms ===\n"
            "p5 = t.nn.Parameter(t.randn(3))\n"
            "p6 = t.nn.Parameter(t.randn(3))\n"
            "# Flat form\n"
            "out_flat = ex2_normalize_params_robust([p5, p6], default_lr=1e-2)\n"
            "opt_flat = t.optim.SGD(out_flat)\n"
            "# Group form with one missing-lr group\n"
            "out_dict = ex2_normalize_params_robust([{'params': [p5, p6]}], default_lr=1e-2)\n"
            "opt_dict = t.optim.SGD(out_dict)\n"
            "assert opt_flat.param_groups[0]['lr'] == 1e-2\n"
            "assert opt_dict.param_groups[0]['lr'] == 1e-2"
        ),
        "solution_body": (
            "def ex2_normalize_params_robust(params, default_lr):\n"
            "    materialized = list(params)               # single-use safe\n"
            "    if not materialized:\n"
            "        raise ValueError('optimizer got an empty parameter list')\n"
            "    first = materialized[0]\n"
            "    if isinstance(first, t.Tensor):           # catches nn.Parameter too\n"
            "        return [{'params': materialized, 'lr': default_lr}]\n"
            "    if isinstance(first, dict):\n"
            "        out = []\n"
            "        for group in materialized:\n"
            "            g = dict(group)                    # shallow copy — no caller mutation\n"
            "            if 'lr' not in g:\n"
            "                g['lr'] = default_lr\n"
            "            out.append(g)\n"
            "        return out\n"
            "    raise TypeError(\n"
            "        f'params must be an iterable of Tensors or dicts, '\n"
            "        f'got first element of type {type(first).__name__}'\n"
            "    )"
        ),
        "solution_notes": (
            "**`isinstance` vs `type(...) is`.** "
            "`isinstance(first, t.Tensor)` returns True for "
            "`nn.Parameter` (a subclass). `type(first) is t.Tensor` "
            "would return False — a real bug, since every call site "
            "that passes `module.parameters()` would route through "
            "the dict branch and crash. This is the #1 polymorphism "
            "footgun in homemade optimizer code.\n\n"
            "**Why `list(params)` BEFORE peeking.** Two reasons: "
            "(1) generators are single-use, and (2) some iterables "
            "(e.g. a one-shot `chain()`) advance internally when "
            "you call `iter()` on them. `list(...)` is the only "
            "safe primitive that gives you both 'first element' "
            "and 'all elements'.\n\n"
            "**`dict(group)` not in-place mutation.** Mutating the "
            "caller's group dicts (`group['lr'] = default_lr`) is "
            "a silent side effect that breaks any code that "
            "introspects the dicts post-construction. PyTorch's "
            "own `Optimizer.__init__` shallow-copies each group "
            "for exactly this reason — match the source."
        ),
        "extra_imports": [],
    },
]


# ---------------------------------------------------------------------------
# Verifier — exec stub + solution + test_body inside a single namespace.
# ---------------------------------------------------------------------------

def _verify_all(specs):
    import torch as t
    import torch.nn as nn
    import numpy as np
    from torch import Tensor
    import einops
    from einops import rearrange, reduce, repeat

    passed = 0
    failed = []

    for spec in specs:
        ex_id = f"ex{spec['exercise_index']}"
        tag = f"{spec['atom_id']}/{ex_id}"
        ns = {
            "t": t,
            "nn": nn,
            "np": np,
            "Tensor": Tensor,
            "einops": einops,
            "rearrange": rearrange,
            "reduce": reduce,
            "repeat": repeat,
            "_dd_passed": set(),
            "__name__": "__main__",
        }
        t.manual_seed(0)
        np.random.seed(0)

        try:
            exec(spec["stub"], ns)
        except Exception:
            pass

        try:
            exec(spec["solution_body"], ns)
            exec(spec["test_body"], ns)
        except Exception as e:
            failed.append((tag, repr(e), traceback.format_exc()))
            continue
        passed += 1
        print(f"  [verify] {tag}: ok")

    print(f"\n[verify] {passed}/{len(specs)} specs passed")
    if failed:
        for tag, err, tb in failed:
            print(f"\n--- FAILED: {tag} ---")
            print(err)
            print(tb)
        raise SystemExit(1)


def main():
    print(f"[deepening_u_batch11] Verifying {len(SPECS)} specs...")
    _verify_all(SPECS)

    print(f"\n[deepening_u_batch11] All verified — emitting notebooks.")
    for spec in SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        print(f"  wrote {rel}")
    print(f"\n[deepening_u_batch11] {len(SPECS)} notebooks emitted.")


if __name__ == "__main__":
    main()
