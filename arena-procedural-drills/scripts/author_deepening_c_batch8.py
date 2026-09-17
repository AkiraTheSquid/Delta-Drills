#!/usr/bin/env python3
"""Author ex2 deepening standalones for 8 high-ARENA single-drill atoms.

Each atom already has ex1 authored; ex2 here probes a DISTINCT facet — either
a different Bloom (Apply vs Analyze), a different surface context, or a
same-Bloom-different-mechanism. Subtopic key is re-used verbatim so the
EWMA mapping stays single-source-per-atom.

For wandb atoms we keep the `unittest.mock.patch`-style sys.modules injection
since backend venv lacks `wandb`. tqdm IS installed in the venv (4.67.3).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone


# ===========================================================================
# ATOM 1 — inplace-param-update / prereqs_training_loop
# ex1: Analyze identity of `theta -= lr * g` vs `theta = theta - lr * g`
# ex2 NEW facet: Apply — manual SGD over a *parameter list* using `.data`
#                in-place subtract; verify each Parameter's storage survives.
# ===========================================================================
SPEC_INPLACE_PARAM = {
    "atom_id": "inplace-param-update",
    "subtopic": "PyTorch: In-place param update",
    "topic_folder": "prereqs_training_loop",
    "atom_recap_md": (
        "## in-place parameter update — quick refresher\n"
        "\n"
        "A hand-rolled optimizer step has to mutate the existing parameter "
        "tensor — NOT rebind a Python name. Two equivalent in-place patterns:\n"
        "\n"
        "- `param.data -= lr * grad` — works on `nn.Parameter` (the `.data` "
        "alias keeps autograd-tracked Parameter wrapper intact).\n"
        "- `param.data.sub_(grad, alpha=lr)` — same effect, slightly faster, "
        "matches what `torch.optim.SGD.step` actually does.\n"
        "\n"
        "**Why `.data` and not just `param -= lr * grad`.** On a leaf "
        "Parameter that requires grad, you can't run in-place ops in a way "
        "autograd would track without `.data` (or a `with torch.no_grad()` "
        "block). Both bypass autograd; `.data` is the older idiom, `no_grad` "
        "is what new code uses.\n"
        "\n"
        "**Storage identity = registered-buffer link.** If you rebind a "
        "Parameter (`param = param - lr * g`), the module's `parameters()` "
        "list still points at the OLD tensor — the new one is orphaned. "
        "Optimizers + checkpoint save/load all break."
    ),
    "exercise_index": 2,
    "exercise_title": "hand-rolled SGD over a parameter list using .data.sub_",
    "slug": "hand-rolled-sgd-over-a-parameter-list",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["sgd", "parameter-list", "data-sub_", "storage-identity"],
    "kcs": ["inplace-update-mutates-storage", "param-data-bypass-autograd"],
    "lo": (
        "Apply `param.data.sub_(grad, alpha=lr)` across a list of "
        "`nn.Parameter` objects so that every parameter's storage pointer "
        "is preserved after the step."
    ),
    "prompt_body": (
        "Implement `ex2_sgd_step(params, grads, lr)`. The minimal hand-"
        "rolled SGD update over an arbitrary parameter list:\n\n"
        "1. `params` is a list of `nn.Parameter` (each requires_grad=True).\n"
        "2. `grads` is a list of `Tensor` of matching shapes — pretend these "
        "came from a backward pass (we pass them in directly to keep the "
        "test deterministic).\n"
        "3. `lr` is a Python float.\n"
        "4. For each `(p, g)` pair: do `p.data.sub_(g, alpha=lr)` (in-place).\n"
        "5. Return the list `params` itself (same Python list, same elements).\n\n"
        "**The test will record each parameter's `data_ptr()` BEFORE the call "
        "and assert they are identical AFTER.** A rebinding implementation "
        "(e.g. `p = p - lr * g`) will fail because the storage changes.\n\n"
        "You are NOT inside a `torch.no_grad()` block in this function — "
        "`.data` is doing the autograd-bypass for you."
    ),
    "stub": (
        "import torch.nn as nn\n"
        "\n"
        "def ex2_sgd_step(params: list, grads: list, lr: float) -> list:\n"
        '    """In-place SGD step over a parameter list. Returns the same list."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "import torch.nn as nn\n"
        "\n"
        "# Build a tiny parameter list — bias + weight shaped like a linear layer.\n"
        "p_w = nn.Parameter(t.tensor([[1.0, 2.0], [3.0, 4.0]]))\n"
        "p_b = nn.Parameter(t.tensor([0.5, -0.5]))\n"
        "params = [p_w, p_b]\n"
        "grads = [t.tensor([[0.1, 0.2], [0.3, 0.4]]),\n"
        "         t.tensor([0.05, -0.05])]\n"
        "lr = 0.1\n"
        "\n"
        "# Snapshot storage pointers BEFORE the step.\n"
        "before_ptrs = [p.data_ptr() for p in params]\n"
        "# Snapshot expected post-step values out of place (oracle).\n"
        "expected = [p.detach().clone() - lr * g for p, g in zip(params, grads)]\n"
        "\n"
        "ret = ex2_sgd_step(params, grads, lr)\n"
        "\n"
        "# Return must be the same Python list object (identity, not equality).\n"
        "assert ret is params, 'must return the same params list (identity)'\n"
        "\n"
        "# Each parameter must have moved by -lr * g.\n"
        "for i, (p, exp) in enumerate(zip(params, expected)):\n"
        "    assert t.allclose(p.data, exp, atol=1e-6), (\n"
        "        f'param {i} value wrong:\\n got {p.data}\\n exp {exp}'\n"
        "    )\n"
        "\n"
        "# Storage pointer MUST be identical — proves in-place update.\n"
        "for i, (p, before) in enumerate(zip(params, before_ptrs)):\n"
        "    assert p.data_ptr() == before, (\n"
        "        f'param {i} storage changed — looks like you rebound the variable '\n"
        "        f'instead of mutating in place'\n"
        "    )\n"
        "\n"
        "# Parameter wrapper must still be a Parameter (not a plain Tensor).\n"
        "for i, p in enumerate(params):\n"
        "    assert isinstance(p, nn.Parameter), (\n"
        "        f'param {i} lost its nn.Parameter wrapper — did you assign p.data = ...?'\n"
        "    )\n"
        "    assert p.requires_grad, f'param {i} lost requires_grad'\n"
        "\n"
        "# Second step uses different lr → values keep accumulating in place.\n"
        "ex2_sgd_step(params, grads, lr=0.5)\n"
        "expected2 = [exp - 0.5 * g for exp, g in zip(expected, grads)]\n"
        "for i, (p, exp) in enumerate(zip(params, expected2)):\n"
        "    assert t.allclose(p.data, exp, atol=1e-6), (\n"
        "        f'after 2nd step param {i} value wrong'\n"
        "    )"
    ),
    "solution_body": (
        "import torch.nn as nn\n"
        "\n"
        "def ex2_sgd_step(params: list, grads: list, lr: float) -> list:\n"
        "    for p, g in zip(params, grads):\n"
        "        p.data.sub_(g, alpha=lr)\n"
        "    return params"
    ),
    "solution_notes": (
        "**`sub_(g, alpha=lr)` not `data -= lr * g`.** Both work, but `sub_` "
        "fuses the scale+subtract into one kernel call and matches what "
        "`torch.optim.SGD` does internally (no momentum/weight-decay here).\n\n"
        "**Never `p.data = ...`.** Assigning a new tensor to `p.data` keeps "
        "the Parameter wrapper but swaps its underlying storage — anything "
        "holding a reference to the OLD storage (saved activations, "
        "optimizer state, hooks) is now stale. `.sub_` mutates the existing "
        "storage in place.\n\n"
        "**Why `.data` not `with no_grad()`.** On a leaf Parameter that "
        "requires grad, in-place ops on the Parameter itself raise. `.data` "
        "is the untracked view; `with no_grad():` is the modern equivalent. "
        "ARENA training loops historically use `.data`, so we drilled that."
    ),
}


# ===========================================================================
# ATOM 2 — buffer-copy_-inplace / prereqs_backprop
# ex1: Apply — copy_ running_mean preserves storage.
# ex2 NEW facet: Analyze — copy_ broadcasting + dtype coercion edge cases.
# ===========================================================================
SPEC_BUFFER_COPY = {
    "atom_id": "buffer-copy_-inplace",
    "subtopic": "PyTorch: in-place buffer copy",
    "topic_folder": "prereqs_backprop",
    "atom_recap_md": (
        "## `tensor.copy_(other)` — quick refresher\n"
        "\n"
        "`dst.copy_(src)` is the canonical in-place \"overwrite my contents "
        "with src's contents\" op. It:\n"
        "\n"
        "- preserves `dst.data_ptr()` (no reallocation),\n"
        "- preserves `dst.dtype` (casts `src` to `dst.dtype` if they differ),\n"
        "- **broadcasts** `src` against `dst.shape` (so a scalar into a vector works),\n"
        "- requires `src` to be broadcastable to `dst` — a `(3,)` src into a `(4,)` dst RAISES.\n"
        "\n"
        "**What it is NOT.** It is not `dst = src` (rebinds the Python name) "
        "and not `dst.data = src` (swaps storage, breaks registered-buffer "
        "links). For module buffers / parameters, `copy_` is the only safe "
        "in-place overwrite."
    ),
    "exercise_index": 2,
    "exercise_title": "predict copy_ behavior across dtype + shape combinations",
    "slug": "predict-copy_-behavior-across-dtype-and-shape",
    "bloom_level": "Analyze",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["copy_", "broadcasting", "dtype-coercion", "in-place-rules"],
    "kcs": ["buffer-copy_-inplace", "copy_-broadcasts-and-coerces"],
    "lo": (
        "Analyze the broadcasting and dtype-coercion rules of `tensor.copy_` "
        "to predict, for each (dst, src) pair, whether the call succeeds + "
        "what dtype/shape/storage the destination ends up with."
    ),
    "prompt_body": (
        "Implement `ex2_predict_copy_outcome(dst, src)`. You will NOT call "
        "`.copy_` — you will REASON about what it would do, and return a "
        "dict describing the predicted outcome:\n\n"
        "Return `{'ok': bool, 'dtype': torch.dtype | None, 'shape': tuple | None}`:\n\n"
        "- `'ok'` — True iff `dst.copy_(src)` would succeed (i.e. `src` is "
        "broadcastable to `dst.shape`). False otherwise.\n"
        "- `'dtype'` — `dst.dtype` if ok (copy_ keeps dst dtype), else None.\n"
        "- `'shape'` — `tuple(dst.shape)` if ok, else None.\n\n"
        "**Broadcast rule:** `src` is broadcastable to `dst` iff, aligned "
        "right, every src dim is 1 or equal to the matching dst dim, and "
        "`src.dim() <= dst.dim()`.\n\n"
        "After computing your prediction, the test will actually run "
        "`dst_copy.copy_(src)` (on a clone of dst) and assert your prediction "
        "matches reality — including the success/failure side."
    ),
    "stub": (
        "def ex2_predict_copy_outcome(dst: Tensor, src: Tensor) -> dict:\n"
        '    """Predict the result of dst.copy_(src) without calling it."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "def _broadcastable(src_shape, dst_shape):\n"
        "    # Reference oracle for broadcastability of src into dst.\n"
        "    if len(src_shape) > len(dst_shape):\n"
        "        return False\n"
        "    for s, d in zip(reversed(src_shape), reversed(dst_shape)):\n"
        "        if s != 1 and s != d:\n"
        "            return False\n"
        "    return True\n"
        "\n"
        "cases = [\n"
        "    # (label, dst, src)\n"
        "    ('same-shape-same-dtype',  t.zeros(3, 4, dtype=t.float32),\n"
        "                                t.ones(3, 4, dtype=t.float32)),\n"
        "    ('same-shape-cross-dtype', t.zeros(3, 4, dtype=t.float32),\n"
        "                                t.ones(3, 4, dtype=t.float64)),\n"
        "    ('scalar-into-vector',     t.zeros(5, dtype=t.float32),\n"
        "                                t.tensor(7.0, dtype=t.float32)),\n"
        "    ('row-into-matrix',        t.zeros(3, 4, dtype=t.float32),\n"
        "                                t.ones(4, dtype=t.float32)),\n"
        "    ('mismatched-vector',      t.zeros(4, dtype=t.float32),\n"
        "                                t.ones(3, dtype=t.float32)),\n"
        "    ('extra-leading-dim',      t.zeros(4, dtype=t.float32),\n"
        "                                t.ones(1, 4, dtype=t.float32)),\n"
        "    ('int-into-float',         t.zeros(2, 3, dtype=t.float32),\n"
        "                                t.ones(2, 3, dtype=t.int64)),\n"
        "]\n"
        "\n"
        "for label, dst, src in cases:\n"
        "    pred = ex2_predict_copy_outcome(dst, src)\n"
        "    assert set(pred.keys()) == {'ok', 'dtype', 'shape'}, (\n"
        "        f'{label}: dict must have exactly keys ok/dtype/shape, got {pred.keys()}'\n"
        "    )\n"
        "    expected_ok = _broadcastable(tuple(src.shape), tuple(dst.shape))\n"
        "    assert pred['ok'] == expected_ok, (\n"
        "        f'{label}: ok prediction wrong — got {pred[\"ok\"]}, expected {expected_ok}'\n"
        "    )\n"
        "    if expected_ok:\n"
        "        assert pred['dtype'] == dst.dtype, (\n"
        "            f'{label}: dtype should equal dst.dtype={dst.dtype}, got {pred[\"dtype\"]}'\n"
        "        )\n"
        "        assert pred['shape'] == tuple(dst.shape), (\n"
        "            f'{label}: shape should equal {tuple(dst.shape)}, got {pred[\"shape\"]}'\n"
        "        )\n"
        "    else:\n"
        "        assert pred['dtype'] is None and pred['shape'] is None, (\n"
        "            f'{label}: on failure dtype + shape must both be None'\n"
        "        )\n"
        "\n"
        "    # Cross-check against reality.\n"
        "    dst_copy = dst.clone()\n"
        "    try:\n"
        "        dst_copy.copy_(src)\n"
        "        real_ok = True\n"
        "    except RuntimeError:\n"
        "        real_ok = False\n"
        "    assert real_ok == expected_ok, (\n"
        "        f'{label}: oracle/reality mismatch — fix the test'\n"
        "    )"
    ),
    "solution_body": (
        "def ex2_predict_copy_outcome(dst: Tensor, src: Tensor) -> dict:\n"
        "    src_shape = tuple(src.shape)\n"
        "    dst_shape = tuple(dst.shape)\n"
        "    if len(src_shape) > len(dst_shape):\n"
        "        return {'ok': False, 'dtype': None, 'shape': None}\n"
        "    for s, d in zip(reversed(src_shape), reversed(dst_shape)):\n"
        "        if s != 1 and s != d:\n"
        "            return {'ok': False, 'dtype': None, 'shape': None}\n"
        "    return {'ok': True, 'dtype': dst.dtype, 'shape': dst_shape}"
    ),
    "solution_notes": (
        "**Two-rule broadcast.** Aligned right, every src dim must be 1 or "
        "equal to the matching dst dim, AND `src.dim() <= dst.dot.dim()`. The "
        "second rule is easy to forget — `(1, 4).copy_into((4,))` fails "
        "because src has more dims than dst.\n\n"
        "**Dtype is dst-driven.** Unlike `t.add` (which promotes), `copy_` "
        "keeps dst's dtype and silently casts src — `int64 → float32` is "
        "legal and lossless for small ints. `float64 → float32` is also "
        "legal but loses precision, no warning.\n\n"
        "**Why not just call copy_.** This is an Analyze drill — the value "
        "is internalizing the rule so you can predict downstream behavior "
        "without trial-and-error in a Jupyter cell."
    ),
}


# ===========================================================================
# ATOM 3 — dataclass-training-args / prereqs_hparam_config
# ex1: Apply — TrainingArgs with __post_init__ validate + asdict round-trip
# ex2 NEW facet: Analyze — predict mutable-default trap + how frozen=True
#                + field(default_factory=...) fixes it.
# ===========================================================================
SPEC_DATACLASS_ARGS = {
    "atom_id": "dataclass-training-args",
    "subtopic": "Config: @dataclass training args",
    "topic_folder": "prereqs_hparam_config",
    "atom_recap_md": (
        "## `@dataclass` mutable-default trap — quick refresher\n"
        "\n"
        "Python dataclasses REJECT a mutable default on a field:\n"
        "\n"
        "```python\n"
        "@dataclass\n"
        "class Args:\n"
        "    layer_dims: list = [256, 128]   # ValueError at class-creation\n"
        "```\n"
        "\n"
        "**The fix:** `field(default_factory=lambda: [256, 128])`. The "
        "factory runs once per *instance*, so two `Args()` calls get two "
        "independent lists.\n"
        "\n"
        "**Why this matters for training args.** If you accidentally share "
        "a list across runs (e.g. by patching defaults at module load), "
        "mutating it in one run silently corrupts the other. The "
        "`default_factory` rule prevents this at class-definition time.\n"
        "\n"
        "**`frozen=True` is the heavier hammer.** Forbids ALL attribute "
        "assignment after `__init__`. Useful for args that should be "
        "immutable for the entire run; not useful for args you mutate from "
        "a sweep agent."
    ),
    "exercise_index": 2,
    "exercise_title": "diagnose mutable-default trap + repair with default_factory",
    "slug": "diagnose-mutable-default-trap-and-repair",
    "bloom_level": "Analyze",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["dataclass", "default-factory", "mutable-default", "frozen"],
    "kcs": ["dataclass-default-factory", "dataclass-frozen-immutability"],
    "lo": (
        "Analyze a broken @dataclass definition to detect the mutable-"
        "default pitfall and return a repaired class that uses "
        "`field(default_factory=...)` for the list default."
    ),
    "prompt_body": (
        "Implement `ex2_build_safe_args_class()`. Build (and RETURN) a "
        "dataclass `SafeArgs` that:\n\n"
        "1. Has fields:\n"
        "   - `lr: float = 1e-3`\n"
        "   - `batch_size: int = 32`\n"
        "   - `layer_dims: list[int]` — defaults to `[256, 128]` BUT via "
        "`field(default_factory=...)`, so two instances get independent "
        "lists.\n"
        "2. Is NOT frozen (the test will mutate `lr` on an instance).\n"
        "3. Returns the class object itself, not an instance.\n\n"
        "The test will verify:\n"
        "- Two `SafeArgs()` instances have `layer_dims` lists with "
        "DIFFERENT identities (`is not`).\n"
        "- Mutating one instance's `layer_dims.append(64)` does NOT affect "
        "the other instance's `layer_dims`.\n"
        "- Mutating `lr` on an instance succeeds (proves it is not frozen).\n"
        "- A naive class definition `layer_dims: list = [256, 128]` would "
        "have raised `ValueError` at class-creation time — your repaired "
        "version doesn't.\n\n"
        "**Hint:** import `field` from `dataclasses`."
    ),
    "stub": (
        "def ex2_build_safe_args_class():\n"
        '    """Return a dataclass that avoids the mutable-default trap."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "from dataclasses import is_dataclass, fields, FrozenInstanceError\n"
        "\n"
        "SafeArgs = ex2_build_safe_args_class()\n"
        "\n"
        "# Class-level checks.\n"
        "assert is_dataclass(SafeArgs), 'must be a dataclass'\n"
        "field_names = {f.name for f in fields(SafeArgs)}\n"
        "assert field_names == {'lr', 'batch_size', 'layer_dims'}, (\n"
        "    f'expected fields lr/batch_size/layer_dims, got {field_names}'\n"
        ")\n"
        "\n"
        "# Default-factory test: two instances → two independent lists.\n"
        "a = SafeArgs()\n"
        "b = SafeArgs()\n"
        "assert a.layer_dims == [256, 128], f'default layer_dims wrong: {a.layer_dims}'\n"
        "assert b.layer_dims == [256, 128]\n"
        "assert a.layer_dims is not b.layer_dims, (\n"
        "    'two instances share the same list — you used `= [256, 128]` directly '\n"
        "    'instead of `field(default_factory=...)`'\n"
        ")\n"
        "\n"
        "# Mutating one must not affect the other.\n"
        "a.layer_dims.append(64)\n"
        "assert a.layer_dims == [256, 128, 64]\n"
        "assert b.layer_dims == [256, 128], 'shared list leaked across instances'\n"
        "\n"
        "# Not frozen: lr is assignable.\n"
        "a.lr = 5e-4\n"
        "assert a.lr == 5e-4\n"
        "\n"
        "# Numeric defaults intact.\n"
        "assert SafeArgs().lr == 1e-3\n"
        "assert SafeArgs().batch_size == 32"
    ),
    "solution_body": (
        "from dataclasses import dataclass, field\n"
        "\n"
        "def ex2_build_safe_args_class():\n"
        "    @dataclass\n"
        "    class SafeArgs:\n"
        "        lr: float = 1e-3\n"
        "        batch_size: int = 32\n"
        "        layer_dims: list = field(default_factory=lambda: [256, 128])\n"
        "    return SafeArgs"
    ),
    "solution_notes": (
        "**Why `default_factory` exists.** Without it, the default value "
        "would be evaluated once at class-creation time and SHARED across "
        "every instance — classic Python mutable-default-argument bug. "
        "Dataclasses detect this and raise `ValueError` rather than let "
        "you ship a footgun.\n\n"
        "**Lambda vs `list`.** `field(default_factory=list)` gives `[]`. "
        "For a non-empty default, pass a lambda returning the literal: "
        "`lambda: [256, 128]`.\n\n"
        "**`frozen=True` trade-off.** Freezing prevents accidental mutation "
        "(good) but also prevents sweep agents from doing "
        "`args.lr = sweep_lr` (bad). ARENA convention: NOT frozen — "
        "training args are mutable for sweep overrides."
    ),
}


# ===========================================================================
# ATOM 4 — wandb-init-run / prereqs_logging_instr
# ex1: Apply — wandb.init(project=, name=, config=) from args object.
# ex2 NEW facet: Apply, different mechanism — merge defaults + override dict
#                BEFORE init (sweep-agent pattern).
# ===========================================================================
SPEC_WANDB_INIT = {
    "atom_id": "wandb-init-run",
    "subtopic": "Logging: wandb.init run",
    "topic_folder": "prereqs_logging_instr",
    "atom_recap_md": (
        "## `wandb.init(...)` with sweep overrides — quick refresher\n"
        "\n"
        "Sweep agents (and Optuna / Hydra / etc.) pass per-trial "
        "hyperparameter overrides as a plain dict. The ARENA pattern is to "
        "merge them into a default args dataclass BEFORE calling "
        "`wandb.init`, so the wandb `config=` snapshot reflects the *actual* "
        "hparams used by training — not the unmodified defaults.\n"
        "\n"
        "**Two failure modes to avoid:**\n"
        "\n"
        "- Calling `wandb.init(config=defaults)` and THEN overriding "
        "`args.lr = sweep_lr` — the dashboard logs `defaults.lr`, not the "
        "real lr. Filtering / comparing runs becomes wrong.\n"
        "- Mutating the defaults dict in place — leaks state into the next "
        "sweep trial.\n"
        "\n"
        "**The merge recipe.** `merged = {**asdict(defaults), **overrides}`. "
        "Right-side wins. Pass `merged` (a dict) to `config=`."
    ),
    "exercise_index": 2,
    "exercise_title": "merge default + override hparams before wandb.init",
    "slug": "merge-default-and-override-hparams-before-init",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["wandb", "init", "sweep", "config-merge", "mock"],
    "kcs": ["wandb-init-config-dict", "config-override-merge"],
    "lo": (
        "Apply the `{**asdict(defaults), **overrides}` merge before "
        "`wandb.init(config=merged)` so the wandb config snapshot reflects "
        "the actual hparams used by the sweep trial."
    ),
    "prompt_body": (
        "Implement `ex2_init_with_overrides(defaults, overrides)`. The "
        "sweep-trial wandb-init recipe:\n\n"
        "1. `defaults` is a dataclass instance (has fields `wandb_project`, "
        "`wandb_name`, `lr`, `batch_size`, `epochs`).\n"
        "2. `overrides` is a plain dict of per-trial overrides — e.g. "
        "`{'lr': 1e-4, 'batch_size': 128}`. May be empty.\n"
        "3. Build the merged config: `merged = {**asdict(defaults), "
        "**overrides}` so override values WIN over defaults.\n"
        "4. Call `wandb.init(project=defaults.wandb_project, "
        "name=defaults.wandb_name, config=merged)`. Note `config=` is the "
        "MERGED DICT, not the defaults object — this is the whole point.\n"
        "5. Return the merged dict.\n\n"
        "The test mocks `wandb` via `sys.modules.setdefault`, then inspects "
        "`wandb.init.call_args` to verify the merged dict landed in "
        "`config=`. Defaults must NOT be mutated (caller-side assertion)."
    ),
    "stub": (
        "import sys\n"
        "from unittest.mock import MagicMock\n"
        "sys.modules.setdefault('wandb', MagicMock())\n"
        "import wandb\n"
        "\n"
        "def ex2_init_with_overrides(defaults, overrides: dict) -> dict:\n"
        '    """Merge overrides into defaults, init wandb with the merged config."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "from dataclasses import dataclass, asdict\n"
        "import copy\n"
        "\n"
        "@dataclass\n"
        "class DefaultArgs:\n"
        "    wandb_project: str = 'arena-sweep'\n"
        "    wandb_name: str = 'trial-0'\n"
        "    lr: float = 1e-3\n"
        "    batch_size: int = 32\n"
        "    epochs: int = 3\n"
        "\n"
        "wandb.init.reset_mock()\n"
        "defaults = DefaultArgs()\n"
        "defaults_snapshot = copy.deepcopy(defaults)\n"
        "overrides = {'lr': 1e-4, 'batch_size': 128}\n"
        "merged = ex2_init_with_overrides(defaults, overrides)\n"
        "\n"
        "# Return is the merged dict.\n"
        "assert isinstance(merged, dict), f'must return a dict, got {type(merged)}'\n"
        "assert merged['lr'] == 1e-4, f'override should win on lr, got {merged[\"lr\"]}'\n"
        "assert merged['batch_size'] == 128, f'override should win on batch_size, got {merged[\"batch_size\"]}'\n"
        "assert merged['epochs'] == 3, f'unspecified field should fall back to default, got {merged[\"epochs\"]}'\n"
        "assert merged['wandb_project'] == 'arena-sweep'\n"
        "\n"
        "# Defaults must not be mutated.\n"
        "assert defaults == defaults_snapshot, 'defaults mutated — use {**asdict(d), **overrides}, NOT d.lr = ...'\n"
        "\n"
        "# wandb.init received the merged dict in config=.\n"
        "assert wandb.init.call_count == 1\n"
        "kw = wandb.init.call_args.kwargs\n"
        "assert kw.get('project') == 'arena-sweep'\n"
        "assert kw.get('name') == 'trial-0'\n"
        "cfg = kw.get('config')\n"
        "assert isinstance(cfg, dict), f'config kwarg must be the merged dict, got {type(cfg)}'\n"
        "assert cfg.get('lr') == 1e-4 and cfg.get('batch_size') == 128, (\n"
        "    f'config snapshot does not reflect overrides — got {cfg}'\n"
        ")\n"
        "\n"
        "# Empty overrides -> merged == asdict(defaults).\n"
        "wandb.init.reset_mock()\n"
        "merged2 = ex2_init_with_overrides(DefaultArgs(), {})\n"
        "assert merged2 == asdict(DefaultArgs()), 'empty overrides must yield the plain defaults dict'"
    ),
    "solution_body": (
        "import sys\n"
        "from unittest.mock import MagicMock\n"
        "sys.modules.setdefault('wandb', MagicMock())\n"
        "import wandb\n"
        "from dataclasses import asdict\n"
        "\n"
        "def ex2_init_with_overrides(defaults, overrides: dict) -> dict:\n"
        "    merged = {**asdict(defaults), **overrides}\n"
        "    wandb.init(\n"
        "        project=defaults.wandb_project,\n"
        "        name=defaults.wandb_name,\n"
        "        config=merged,\n"
        "    )\n"
        "    return merged"
    ),
    "solution_notes": (
        "**Right-side wins.** Python `{**a, **b}` literally builds a new "
        "dict and re-inserts `b`'s keys last — later writes overwrite "
        "earlier ones. Perfect for overrides.\n\n"
        "**Why pass the dict, not the dataclass.** `wandb.init(config=...)` "
        "snapshots WHATEVER you give it. Pass the dataclass and wandb sees "
        "the unmodified defaults; pass the merged dict and wandb sees the "
        "true trial config. The dashboard filter `lr=1e-4` will only "
        "surface this run if config['lr'] is 1e-4.\n\n"
        "**Never mutate defaults.** If you set `defaults.lr = overrides['lr']`, "
        "the next sweep trial inherits that mutation. Always rebuild a "
        "fresh dict per trial."
    ),
}


# ===========================================================================
# ATOM 5 — wandb-finish / prereqs_logging_instr
# ex1: Apply — init -> loop -> finish (lifecycle pair).
# ex2 NEW facet: Analyze — diagnose a missing finally; show why a raised
#                exception skips finish without try/finally, then fix it.
# ===========================================================================
SPEC_WANDB_FINISH = {
    "atom_id": "wandb-finish",
    "subtopic": "Logging: wandb.finish",
    "topic_folder": "prereqs_logging_instr",
    "atom_recap_md": (
        "## `wandb.finish()` with try/finally — quick refresher\n"
        "\n"
        "If your training loop raises (CUDA OOM, dataset NaN, you Ctrl-C), "
        "the default ARENA `train()` skips `wandb.finish` — the run sits in "
        "`running` state on the dashboard forever. Worse, the next "
        "`wandb.init` may silently append to the dead run instead of "
        "opening a new one.\n"
        "\n"
        "**The fix is one keyword pair:**\n"
        "\n"
        "```python\n"
        "wandb.init(...)\n"
        "try:\n"
        "    for step in range(n_steps):\n"
        "        train_step(...)\n"
        "finally:\n"
        "    wandb.finish()\n"
        "```\n"
        "\n"
        "**`finally` runs even on exception.** Whether the loop completes "
        "normally or blows up, `wandb.finish` is called exactly once. The "
        "exception still propagates after the finally block — you don't "
        "swallow it."
    ),
    "exercise_index": 2,
    "exercise_title": "guarantee wandb.finish even on training-loop exception",
    "slug": "guarantee-wandb-finish-on-exception",
    "bloom_level": "Analyze",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["wandb", "finish", "try-finally", "exception-safety", "mock"],
    "kcs": ["wandb-finish-after-train", "try-finally-cleanup"],
    "lo": (
        "Analyze the failure mode of an init/finish lifecycle that lacks "
        "try/finally and repair it so `wandb.finish` always runs, even "
        "when the inner loop raises."
    ),
    "prompt_body": (
        "Implement `ex2_safe_train(n_steps, raise_at)`. A wandb-instrumented "
        "fake-train that uses try/finally to guarantee `finish` runs:\n\n"
        "1. Call `wandb.init(project='arena', name='safe-run')`.\n"
        "2. Wrap a `for step in range(n_steps)` loop in `try:` ... "
        "`finally: wandb.finish()`.\n"
        "3. Inside the loop, if `raise_at is not None and step == raise_at`, "
        "raise `RuntimeError(f'simulated failure at step {step}')`.\n"
        "4. Otherwise the loop is a no-op.\n"
        "5. **Return** the number of steps completed BEFORE the raise (or "
        "`n_steps` if no raise happened). The exception must still "
        "propagate out of the function — do NOT swallow it; the test "
        "expects it via `with pytest.raises(...)`-style assert.\n\n"
        "The test will run TWO scenarios:\n"
        "- `raise_at=None` — normal completion. Assert `wandb.init` + "
        "`wandb.finish` each called once.\n"
        "- `raise_at=2` — exception path. Assert RuntimeError propagated, "
        "`wandb.finish` STILL called exactly once."
    ),
    "stub": (
        "import sys\n"
        "from unittest.mock import MagicMock\n"
        "sys.modules.setdefault('wandb', MagicMock())\n"
        "import wandb\n"
        "\n"
        "def ex2_safe_train(n_steps: int, raise_at) -> int:\n"
        '    """Init wandb, fake-train, finish in try/finally. Returns steps completed."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# Scenario 1 — normal completion.\n"
        "wandb.init.reset_mock(); wandb.finish.reset_mock()\n"
        "out = ex2_safe_train(n_steps=5, raise_at=None)\n"
        "assert out == 5, f'normal path must return n_steps, got {out}'\n"
        "assert wandb.init.call_count == 1, f'init expected 1x, got {wandb.init.call_count}'\n"
        "assert wandb.finish.call_count == 1, f'finish expected 1x, got {wandb.finish.call_count}'\n"
        "\n"
        "# Scenario 2 — raise mid-loop. Exception must propagate, finish must STILL run.\n"
        "wandb.init.reset_mock(); wandb.finish.reset_mock()\n"
        "raised = False\n"
        "try:\n"
        "    ex2_safe_train(n_steps=5, raise_at=2)\n"
        "except RuntimeError as e:\n"
        "    raised = True\n"
        "    assert 'simulated failure at step 2' in str(e), f'wrong message: {e}'\n"
        "assert raised, 'RuntimeError must propagate out of ex2_safe_train — do not swallow it'\n"
        "assert wandb.init.call_count == 1\n"
        "assert wandb.finish.call_count == 1, (\n"
        "    f'wandb.finish must run via the finally block even on exception, '\n"
        "    f'got call_count={wandb.finish.call_count}'\n"
        ")\n"
        "\n"
        "# Scenario 3 — raise_at=0 (very first step). Finish must still run.\n"
        "wandb.init.reset_mock(); wandb.finish.reset_mock()\n"
        "try:\n"
        "    ex2_safe_train(n_steps=10, raise_at=0)\n"
        "except RuntimeError:\n"
        "    pass\n"
        "assert wandb.finish.call_count == 1, 'finish must run even if loop raises on step 0'"
    ),
    "solution_body": (
        "import sys\n"
        "from unittest.mock import MagicMock\n"
        "sys.modules.setdefault('wandb', MagicMock())\n"
        "import wandb\n"
        "\n"
        "def ex2_safe_train(n_steps: int, raise_at) -> int:\n"
        "    wandb.init(project='arena', name='safe-run')\n"
        "    completed = 0\n"
        "    try:\n"
        "        for step in range(n_steps):\n"
        "            if raise_at is not None and step == raise_at:\n"
        "                raise RuntimeError(f'simulated failure at step {step}')\n"
        "            completed += 1\n"
        "    finally:\n"
        "        wandb.finish()\n"
        "    return completed"
    ),
    "solution_notes": (
        "**`finally` runs unconditionally.** Normal return, raised "
        "exception, even `return` inside the `try` — `finally` still runs. "
        "It is the right tool when you want \"this cleanup MUST happen, "
        "no matter what.\"\n\n"
        "**Don't catch + re-raise.** `try: ... except: wandb.finish(); "
        "raise` works but is more code and is wrong if you raise something "
        "the except clause doesn't catch. `finally` covers every "
        "exception class for free.\n\n"
        "**Why not a context manager?** Wandb DOES expose a context-"
        "manager API (`with wandb.init(...) as run:`), which is nicer. "
        "ARENA's code base predates it, so the explicit try/finally is "
        "still the prevailing idiom in the codebase."
    ),
}


# ===========================================================================
# ATOM 6 — wandb-log-step / prereqs_logging_instr
# ex1: Apply — log({'loss': loss}, step=examples_seen) per step.
# ex2 NEW facet: Apply, different mechanism — accumulate train metrics and a
#                periodic val metric on the SAME step axis using commit=
#                semantics to flush exactly once per step.
# ===========================================================================
SPEC_WANDB_LOG = {
    "atom_id": "wandb-log-step",
    "subtopic": "Logging: wandb.log step",
    "topic_folder": "prereqs_logging_instr",
    "atom_recap_md": (
        "## `wandb.log({...}, step=..., commit=...)` — quick refresher\n"
        "\n"
        "Wandb buffers `log` calls per step. By default each `log` call "
        "auto-commits — the buffer is flushed and the step advances. If "
        "you need to log train + val metrics on the SAME step, you must "
        "tell wandb \"don't commit yet\" with `commit=False`, then flush "
        "with a final `commit=True` (or just omit it on the last call):\n"
        "\n"
        "```python\n"
        "wandb.log({'train/loss': train_loss}, step=ex_seen, commit=False)\n"
        "wandb.log({'val/loss':   val_loss},   step=ex_seen, commit=True)\n"
        "```\n"
        "\n"
        "**Why this matters.** Without `commit=False`, the train metric "
        "and val metric land on different internal steps even though you "
        "passed the same `step=`. Filters comparing train vs val at the "
        "same x-axis position break.\n"
        "\n"
        "**The rule.** All-but-last log on a given step → `commit=False`. "
        "Last log on a given step → `commit=True` (the default)."
    ),
    "exercise_index": 2,
    "exercise_title": "log train + val on the same step using commit=False/True",
    "slug": "log-train-plus-val-on-the-same-step",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["wandb", "log", "commit", "train-val", "mock"],
    "kcs": ["wandb-log-step-kwarg", "wandb-log-commit-semantics"],
    "lo": (
        "Apply `wandb.log(..., commit=False)` for non-final metric dicts "
        "and `commit=True` for the final one per step so train + val "
        "metrics share a single x-axis position."
    ),
    "prompt_body": (
        "Implement `ex2_log_train_and_val(train_losses, val_loss, "
        "batch_size)`. A fake epoch where every step logs train metrics, "
        "and the LAST step ALSO logs a val metric — all on the same step "
        "axis as the train metric:\n\n"
        "1. Maintain `examples_seen` starting at 0, incrementing by "
        "`batch_size` per step.\n"
        "2. For each `train_loss` at index `i`:\n"
        "   - `examples_seen += batch_size`.\n"
        "   - If `i < len(train_losses) - 1` (not the last step):\n"
        "     Call `wandb.log({'train/loss': train_loss}, step=examples_seen)` "
        "(auto-commit is fine; step advances naturally).\n"
        "   - If `i == len(train_losses) - 1` (last step):\n"
        "     Call `wandb.log({'train/loss': train_loss}, step=examples_seen, "
        "commit=False)` first, THEN call "
        "`wandb.log({'val/loss': val_loss}, step=examples_seen, commit=True)` "
        "to flush both metrics on the SAME step.\n"
        "3. Return `examples_seen`.\n\n"
        "The test inspects `wandb.log.call_args_list` to verify per-call "
        "kwargs."
    ),
    "stub": (
        "import sys\n"
        "from unittest.mock import MagicMock\n"
        "sys.modules.setdefault('wandb', MagicMock())\n"
        "import wandb\n"
        "\n"
        "def ex2_log_train_and_val(train_losses: list, val_loss: float, batch_size: int) -> int:\n"
        '    """Log train per step; on last step also log val with commit semantics."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "wandb.log.reset_mock()\n"
        "train_losses = [0.9, 0.7, 0.5]\n"
        "val_loss = 0.42\n"
        "batch_size = 16\n"
        "final = ex2_log_train_and_val(train_losses, val_loss, batch_size)\n"
        "assert final == len(train_losses) * batch_size, (\n"
        "    f'examples_seen wrong: got {final}, expected {len(train_losses) * batch_size}'\n"
        ")\n"
        "\n"
        "calls = wandb.log.call_args_list\n"
        "# 2 inner-step train logs + 1 final train (commit=False) + 1 val (commit=True) = 4.\n"
        "assert len(calls) == 4, f'expected 4 wandb.log calls, got {len(calls)}'\n"
        "\n"
        "def _metrics(c):\n"
        "    return c.args[0] if c.args else c.kwargs.get('metrics') or c.kwargs.get('data')\n"
        "\n"
        "# Calls 0, 1 — intermediate train metrics; step monotone; commit not False.\n"
        "for i in (0, 1):\n"
        "    m = _metrics(calls[i])\n"
        "    assert isinstance(m, dict) and 'train/loss' in m\n"
        "    assert abs(m['train/loss'] - train_losses[i]) < 1e-9\n"
        "    expected_step = (i + 1) * batch_size\n"
        "    assert calls[i].kwargs.get('step') == expected_step, (\n"
        "        f'call {i}: step expected {expected_step}, got {calls[i].kwargs.get(\"step\")}'\n"
        "    )\n"
        "    # Auto-commit is fine on intermediate steps — accept absence OR True.\n"
        "    commit_val = calls[i].kwargs.get('commit', True)\n"
        "    assert commit_val is True, f'intermediate step call must commit (auto or explicit)'\n"
        "\n"
        "# Call 2 — final train metric, commit=False (do not flush yet).\n"
        "m2 = _metrics(calls[2])\n"
        "assert 'train/loss' in m2 and abs(m2['train/loss'] - train_losses[-1]) < 1e-9\n"
        "assert calls[2].kwargs.get('step') == final\n"
        "assert calls[2].kwargs.get('commit') is False, (\n"
        "    'last train log on the joint step must use commit=False to coalesce with val'\n"
        ")\n"
        "\n"
        "# Call 3 — val metric, commit=True flushes the step.\n"
        "m3 = _metrics(calls[3])\n"
        "assert 'val/loss' in m3 and abs(m3['val/loss'] - val_loss) < 1e-9\n"
        "assert calls[3].kwargs.get('step') == final, 'val log step must match the joint train step'\n"
        "assert calls[3].kwargs.get('commit') is True, 'val log must commit=True to flush both metrics'"
    ),
    "solution_body": (
        "import sys\n"
        "from unittest.mock import MagicMock\n"
        "sys.modules.setdefault('wandb', MagicMock())\n"
        "import wandb\n"
        "\n"
        "def ex2_log_train_and_val(train_losses: list, val_loss: float, batch_size: int) -> int:\n"
        "    examples_seen = 0\n"
        "    n = len(train_losses)\n"
        "    for i, train_loss in enumerate(train_losses):\n"
        "        examples_seen += batch_size\n"
        "        if i < n - 1:\n"
        "            wandb.log({'train/loss': train_loss}, step=examples_seen)\n"
        "        else:\n"
        "            wandb.log({'train/loss': train_loss}, step=examples_seen, commit=False)\n"
        "            wandb.log({'val/loss': val_loss}, step=examples_seen, commit=True)\n"
        "    return examples_seen"
    ),
    "solution_notes": (
        "**Why two calls, not one merged dict.** You CAN merge train+val "
        "into one log call (`{'train/loss': ..., 'val/loss': ...}`). The "
        "two-call commit pattern shines when the val computation happens "
        "in a SEPARATE code path (separate eval function) — you don't "
        "want to thread the train metric all the way down into the eval "
        "function just to flush them together.\n\n"
        "**`commit=False` is sticky to the step.** Once you log with "
        "`commit=False` on step S, wandb holds the buffer until a "
        "`commit=True` (or another `log` call that auto-commits) for "
        "step S. Any LATER `log` for the same step lands in the same "
        "buffer.\n\n"
        "**Real-world variant.** Validation often runs once per epoch, "
        "not once per step. Same pattern — `commit=False` on the train "
        "tail, then a separate `eval()` call logs val with `commit=True`."
    ),
}


# ===========================================================================
# ATOM 7 — tqdm-postfix-metrics / prereqs_logging_instr
# ex1: Apply — tqdm(iter) + set_postfix per step.
# ex2 NEW facet: Apply, different tqdm-API mechanism — manual progress with
#                total= + pbar.update(n) + dynamic set_description per phase.
# ===========================================================================
SPEC_TQDM_POSTFIX = {
    "atom_id": "tqdm-postfix-metrics",
    "subtopic": "Logging: tqdm postfix metrics",
    "topic_folder": "prereqs_logging_instr",
    "atom_recap_md": (
        "## `tqdm(total=N)` + `pbar.update(n)` + `set_description` — quick refresher\n"
        "\n"
        "Wrapping an iterable in `tqdm(...)` is the common case. Sometimes "
        "you don't have a single iterable — you have multiple phases of "
        "work (train batches, val batches, log flush) and want ONE bar "
        "showing total progress. That calls for the manual API:\n"
        "\n"
        "```python\n"
        "pbar = tqdm(total=total_units)\n"
        "for phase, units in phases:\n"
        "    pbar.set_description(f'Phase: {phase}')\n"
        "    for _ in range(units):\n"
        "        do_work()\n"
        "        pbar.update(1)         # advance by 1 unit\n"
        "        pbar.set_postfix(loss=...)\n"
        "pbar.close()\n"
        "```\n"
        "\n"
        "**`set_description` sets the LEFT label** (replaces `desc=` you'd "
        "pass at construction). **`update(n)` advances the bar by `n` "
        "units** — you control how many units per logical step.\n"
        "\n"
        "**Always `close()` a manual `tqdm`.** Wrapping-an-iterable closes "
        "automatically when the iterator exhausts; the manual API does not."
    ),
    "exercise_index": 2,
    "exercise_title": "manual tqdm with phase-labeled set_description across train+val",
    "slug": "manual-tqdm-phase-labeled-train-plus-val",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["tqdm", "manual-update", "set-description", "phase-labels"],
    "kcs": ["tqdm-manual-update", "tqdm-set-description-phase"],
    "lo": (
        "Apply `tqdm(total=N)` + `pbar.update(1)` + `pbar.set_description` "
        "to drive a single progress bar across multiple phases (train + "
        "val), with the bar's left label switching per phase."
    ),
    "prompt_body": (
        "Implement `ex2_manual_tqdm_phases(train_losses, val_losses)`. A "
        "single tqdm bar that spans train + val with dynamic phase "
        "labels:\n\n"
        "1. Construct `pbar = tqdm(total=len(train_losses) + len(val_losses))`.\n"
        "2. **Train phase**: `pbar.set_description('train')`. Loop over "
        "`train_losses`; for each loss, call "
        "`pbar.set_postfix(loss=f'{loss:.3f}')` and `pbar.update(1)`.\n"
        "3. **Val phase**: `pbar.set_description('val')`. Loop over "
        "`val_losses`; for each loss, call "
        "`pbar.set_postfix(loss=f'{loss:.3f}')` and `pbar.update(1)`.\n"
        "4. Call `pbar.close()`.\n"
        "5. Return a dict: `{'final_n': pbar.n, 'final_desc': <phase>}`. "
        "tqdm stores the current bar position in `pbar.n` and the current "
        "description in `pbar.desc` — note that newer tqdm versions append "
        "`': '` to the desc, so **`rstrip(': ')` the value** before "
        "returning it so the caller sees just the phase name.\n\n"
        "**Why not two separate bars?** A single bar makes the train+val "
        "ratio visually obvious. ARENA's eval-during-train loops use this "
        "to surface val cost vs train cost.\n\n"
        "The test uses `disable=True`-style introspection — it inspects "
        "`pbar.n`, `pbar.total`, and the call history; nothing prints to "
        "the terminal during testing."
    ),
    "stub": (
        "from tqdm import tqdm\n"
        "\n"
        "def ex2_manual_tqdm_phases(train_losses: list, val_losses: list) -> dict:\n"
        '    """Single manual tqdm bar across train + val phases."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "out = ex2_manual_tqdm_phases([0.9, 0.7, 0.5], [0.4, 0.35])\n"
        "assert isinstance(out, dict), f'must return a dict, got {type(out)}'\n"
        "assert set(out.keys()) == {'final_n', 'final_desc'}, f'keys: {out.keys()}'\n"
        "\n"
        "# All 5 units must have been advanced.\n"
        "assert out['final_n'] == 5, f'expected pbar.n=5, got {out[\"final_n\"]}'\n"
        "\n"
        "# Last set_description call was 'val' (val is the second phase).\n"
        "assert out['final_desc'] == 'val', (\n"
        "    f'last set_description should be \"val\", got {out[\"final_desc\"]!r} — '\n"
        "    f'did you forget to set_description before the val phase?'\n"
        ")\n"
        "\n"
        "# Empty val phase — bar stops at len(train), desc stays at last set value (train).\n"
        "out2 = ex2_manual_tqdm_phases([1.0, 0.8], [])\n"
        "assert out2['final_n'] == 2, f'empty val: expected n=2, got {out2[\"final_n\"]}'\n"
        "# When val phase is empty, val's set_description still ran (loop entered the val branch unconditionally).\n"
        "# Either 'train' (if you guard) or 'val' (if you unconditionally set) is acceptable —\n"
        "# the prompt says unconditionally set, so assert 'val'.\n"
        "assert out2['final_desc'] == 'val', (\n"
        "    f'final_desc should be \"val\" (prompt says always set it before the val loop), '\n"
        "    f'got {out2[\"final_desc\"]!r}'\n"
        ")\n"
        "\n"
        "# Empty everything — n=0, total=0.\n"
        "out3 = ex2_manual_tqdm_phases([], [])\n"
        "assert out3['final_n'] == 0"
    ),
    "solution_body": (
        "from tqdm import tqdm\n"
        "\n"
        "def ex2_manual_tqdm_phases(train_losses: list, val_losses: list) -> dict:\n"
        "    pbar = tqdm(total=len(train_losses) + len(val_losses))\n"
        "    pbar.set_description('train')\n"
        "    for loss in train_losses:\n"
        "        pbar.set_postfix(loss=f'{loss:.3f}')\n"
        "        pbar.update(1)\n"
        "    pbar.set_description('val')\n"
        "    for loss in val_losses:\n"
        "        pbar.set_postfix(loss=f'{loss:.3f}')\n"
        "        pbar.update(1)\n"
        "    # tqdm stores desc with a trailing ': ' separator on some versions;\n"
        "    # rstrip it so the caller-facing label is just the phase name.\n"
        "    desc = pbar.desc.rstrip(': ') if pbar.desc else ''\n"
        "    out = {'final_n': pbar.n, 'final_desc': desc}\n"
        "    pbar.close()\n"
        "    return out"
    ),
    "solution_notes": (
        "**`pbar.desc` vs `pbar.set_description(...)`.** `set_description` "
        "is the setter (it also re-renders the bar). `pbar.desc` is the "
        "attribute it writes to — fine to read directly. Newer tqdm "
        "versions append `': '` automatically when you `set_description`, "
        "so we `rstrip(': ')` in the return path to give callers just the "
        "phase name.\n\n"
        "**`update(1)` vs ` update(n)`.** Pass `n` whenever a logical step "
        "covers `n` units — e.g. `pbar.update(batch_size)` for an "
        "examples-seen bar. Here we treat each loss as one unit.\n\n"
        "**Why manual instead of wrapping a chained iterator.** "
        "`itertools.chain(train_losses, val_losses)` works for the bar "
        "itself but loses the phase boundary — you can't `set_description` "
        "mid-iteration without an external counter. Manual is clearer."
    ),
}


# ===========================================================================
# ATOM 8 — tensor-wraps-ndarray / prereqs_tensor_mechanics
# ex1: Analyze — from_numpy shares memory vs t.tensor copies (mutate source).
# ex2 NEW facet: Apply, different mechanism — write a dtype/device routing
#                policy that picks from_numpy/as_tensor/tensor based on
#                input dtype, with aliasing semantics documented.
# ===========================================================================
SPEC_TENSOR_FROM_NDARRAY = {
    "atom_id": "tensor-wraps-ndarray",
    "subtopic": "PyTorch: tensor from ndarray",
    "topic_folder": "prereqs_tensor_mechanics",
    "atom_recap_md": (
        "## `from_numpy` / `as_tensor` / `tensor` — quick refresher\n"
        "\n"
        "Three ways to wrap a numpy array, with very different aliasing "
        "semantics:\n"
        "\n"
        "- `t.from_numpy(arr)` — **always shares memory** with `arr`. "
        "Same dtype as arr. Raises if arr dtype is unsupported (e.g. "
        "uint16). Mutating one mutates the other.\n"
        "- `t.as_tensor(arr)` — **shares memory IFF dtype/device match the "
        "target**; otherwise copies. Effectively `from_numpy` when types "
        "align, `tensor` when they don't.\n"
        "- `t.tensor(arr)` — **always copies**. Independent storage. "
        "Mutating one does NOT affect the other.\n"
        "\n"
        "**Why a policy matters.** ARENA pipelines load int32 numpy arrays "
        "from disk but train with int64 indices and float32 features. A "
        "policy that picks the right wrapper per dtype avoids the "
        "\"silently aliased buffer\" footgun while still saving a copy "
        "when the dtypes match."
    ),
    "exercise_index": 2,
    "exercise_title": "wrap ndarray with dtype-aware routing policy",
    "slug": "wrap-ndarray-with-dtype-aware-routing-policy",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["from_numpy", "as_tensor", "tensor", "dtype-routing", "aliasing"],
    "kcs": ["from-numpy-shares-storage", "as-tensor-conditional-copy"],
    "lo": (
        "Apply `t.from_numpy` / `t.as_tensor` / `t.tensor` selection based "
        "on input ndarray dtype + a target dtype, producing the correct "
        "aliasing-or-copy outcome per case."
    ),
    "prompt_body": (
        "Implement `ex2_wrap_ndarray(arr, target_dtype)`. A small routing "
        "policy that picks the right wrapper:\n\n"
        "- `arr` is a numpy ndarray.\n"
        "- `target_dtype` is a torch dtype (e.g. `t.float32`, `t.int64`).\n\n"
        "Rules:\n"
        "1. If the numpy dtype already corresponds to `target_dtype` "
        "(e.g. arr is `float32` and target is `t.float32`), use "
        "`t.from_numpy(arr)` — fast, **aliased**, no copy.\n"
        "2. Otherwise, use `t.tensor(arr, dtype=target_dtype)` — explicit "
        "copy + cast. The result must NOT alias `arr`.\n\n"
        "**Use `t.from_numpy(arr).dtype` to check the would-be dtype** of "
        "a numpy array without doing the wrap yet (or check "
        "`arr.dtype.name` against torch dtype names — both work).\n\n"
        "Return a dict: `{'tensor': out_tensor, 'aliased': bool}` where "
        "`aliased` is True iff the returned tensor shares memory with "
        "`arr` (i.e. you went down branch 1).\n\n"
        "The test will verify aliasing the hard way: mutate `arr` "
        "in-place and check whether `out_tensor` reflects the change."
    ),
    "stub": (
        "def ex2_wrap_ndarray(arr: np.ndarray, target_dtype: t.dtype) -> dict:\n"
        '    """Route to from_numpy (alias) or tensor (copy+cast) per dtype match."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# Case 1 — float32 ndarray, target float32 → from_numpy → aliased.\n"
        "arr1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)\n"
        "out1 = ex2_wrap_ndarray(arr1, t.float32)\n"
        "assert set(out1.keys()) == {'tensor', 'aliased'}\n"
        "ten1 = out1['tensor']\n"
        "assert ten1.dtype == t.float32, f'expected float32, got {ten1.dtype}'\n"
        "assert out1['aliased'] is True, 'matching dtype should alias (from_numpy)'\n"
        "# Mutate source — tensor should reflect it.\n"
        "arr1[0] = 99.0\n"
        "assert ten1[0].item() == 99.0, 'aliased tensor must reflect ndarray mutation'\n"
        "\n"
        "# Case 2 — int32 ndarray, target int64 → tensor → copy.\n"
        "arr2 = np.array([1, 2, 3], dtype=np.int32)\n"
        "out2 = ex2_wrap_ndarray(arr2, t.int64)\n"
        "ten2 = out2['tensor']\n"
        "assert ten2.dtype == t.int64, f'expected int64, got {ten2.dtype}'\n"
        "assert out2['aliased'] is False, 'dtype mismatch should COPY, not alias'\n"
        "arr2[0] = 999\n"
        "assert ten2[0].item() == 1, 'copied tensor must NOT reflect ndarray mutation'\n"
        "\n"
        "# Case 3 — float64 ndarray, target float64 → from_numpy → aliased.\n"
        "arr3 = np.array([0.1, 0.2], dtype=np.float64)\n"
        "out3 = ex2_wrap_ndarray(arr3, t.float64)\n"
        "ten3 = out3['tensor']\n"
        "assert ten3.dtype == t.float64\n"
        "assert out3['aliased'] is True\n"
        "arr3[1] = -1.5\n"
        "assert ten3[1].item() == -1.5\n"
        "\n"
        "# Case 4 — float64 ndarray, target float32 → tensor → copy + downcast.\n"
        "arr4 = np.array([1.5, 2.5], dtype=np.float64)\n"
        "out4 = ex2_wrap_ndarray(arr4, t.float32)\n"
        "ten4 = out4['tensor']\n"
        "assert ten4.dtype == t.float32, f'expected float32 after downcast, got {ten4.dtype}'\n"
        "assert out4['aliased'] is False\n"
        "arr4[0] = 100.0\n"
        "assert abs(ten4[0].item() - 1.5) < 1e-6, 'downcast result should NOT alias'\n"
        "\n"
        "# Case 5 — int64 ndarray, target int64 → from_numpy → aliased.\n"
        "arr5 = np.array([10, 20, 30], dtype=np.int64)\n"
        "out5 = ex2_wrap_ndarray(arr5, t.int64)\n"
        "assert out5['aliased'] is True\n"
        "arr5[2] = 777\n"
        "assert out5['tensor'][2].item() == 777"
    ),
    "solution_body": (
        "def ex2_wrap_ndarray(arr: np.ndarray, target_dtype: t.dtype) -> dict:\n"
        "    natural = t.from_numpy(arr).dtype\n"
        "    if natural == target_dtype:\n"
        "        return {'tensor': t.from_numpy(arr), 'aliased': True}\n"
        "    return {'tensor': t.tensor(arr, dtype=target_dtype), 'aliased': False}"
    ),
    "solution_notes": (
        "**Why check via `t.from_numpy(arr).dtype` not `arr.dtype.name`.** "
        "Torch and numpy don't always share dtype spellings (`np.int64` vs "
        "`t.int64` happen to align, but `np.bool_` vs `t.bool` don't on "
        "all builds). Routing through `from_numpy` makes torch tell you "
        "the canonical mapping. The check is cheap because `from_numpy` "
        "just inspects metadata — no data copy at that moment.\n\n"
        "**Why not `t.as_tensor`?** `as_tensor` already does the routing "
        "we built — same dtype → share, different dtype → copy. We "
        "built the policy ourselves to drive home what `as_tensor` is "
        "actually doing under the hood (Apply-level mastery), and to "
        "return the explicit `'aliased'` flag callers want.\n\n"
        "**The aliasing hazard.** Aliased tensors are dangerous if you "
        "later `.to('cuda')` (which copies, breaking alias) or pass to a "
        "module that calls `.contiguous()` (also a copy). For training "
        "data loaded once at startup, alias is fine; for online data "
        "streams, prefer the explicit copy."
    ),
}


# ===========================================================================
SPECS = [
    SPEC_INPLACE_PARAM,
    SPEC_BUFFER_COPY,
    SPEC_DATACLASS_ARGS,
    SPEC_WANDB_INIT,
    SPEC_WANDB_FINISH,
    SPEC_WANDB_LOG,
    SPEC_TQDM_POSTFIX,
    SPEC_TENSOR_FROM_NDARRAY,
]


for spec in SPECS:
    path = emit_standalone(spec)
    rel = path.relative_to(path.parents[3])
    print(f"wrote {rel}")
