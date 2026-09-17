#!/usr/bin/env python3
"""Author 8 standalone Colab drills for the hyperparam/config-management
family of atoms (batch-5).

Atoms covered (each drill = ONE LO + ONE Bloom level, max 2 concurrent KCs):

  dataclass-training-args         — 1 drill (ex1)
  optimizer-class-dispatch        — 1 drill (ex1)
  param-group-dict-list           — 1 drill (ex1)
  hparam-precedence-merge         — 1 drill (ex1)
  params-iterable-vs-groups       — 1 drill (ex1)
  sweep-config-dict               — 1 drill (ex1)
  sweep-hparam-distribution       — 1 drill (ex1)
  nested-param-group-loop         — 1 drill (ex1)

These are SMALLER constituent config skills the ARENA chap-0 part-3
optimization material and the chap-3 transformer-training scaffolding
both assume the learner can already perform in isolation. They cover
the "stitch a training run together" wiring: dataclasses for args,
optimizer dispatch tables, differential-LR param groups, the wandb
sweep config dict format, and the polymorphic optimizer `params=`
signature.

Each spec is verified by re-running its solution against its test_body
inside the build venv (torch 2.12.0+cpu) before emission. Any failure
aborts the build.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

TOPIC = "prereqs_hparam_config"


# ---------------------------------------------------------------------------
# Per-atom recap blocks.
# ---------------------------------------------------------------------------

RECAP_DATACLASS_ARGS = (
    "## Config: `@dataclass` training args — quick refresher\n"
    "\n"
    "Modern training-loop code groups all hyperparameters into a single "
    "args object instead of dragging dozens of kwargs through every "
    "function. The idiom:\n"
    "\n"
    "```python\n"
    "from dataclasses import dataclass, asdict\n"
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
    "```\n"
    "\n"
    "**Why a dataclass and not a plain dict.** Dataclasses give you "
    "type annotations (IDE autocomplete + static checking), default "
    "values, AND `__post_init__` for cheap validation. They keep the "
    "schema in one place — change a default and every call site picks "
    "it up.\n"
    "\n"
    "**`frozen=False` is the right default for training args.** ARENA "
    "and most reference implementations leave the dataclass mutable so "
    "you can override fields from the CLI / config file *after* "
    "construction. `frozen=True` would force you to build a fresh "
    "object for every override, which clashes with the precedence-"
    "merge pattern (see `hparam-precedence-merge`).\n"
    "\n"
    "**`dataclasses.asdict(args)`** is how you flatten the object back "
    "to a dict — required for wandb config logging and for JSON "
    "checkpoints. It's recursive: nested dataclasses get unrolled too."
)

RECAP_OPTIMIZER_DISPATCH = (
    "## Config: Optimizer class dispatch — quick refresher\n"
    "\n"
    "Switching optimizers based on a config string is universal in "
    "training scripts. The idiom:\n"
    "\n"
    "```python\n"
    "OPTIMIZER_CLASSES = {\n"
    "    'sgd':  torch.optim.SGD,\n"
    "    'adam': torch.optim.Adam,\n"
    "    'adamw': torch.optim.AdamW,\n"
    "}\n"
    "optimizer = OPTIMIZER_CLASSES[cfg.optimizer_name](\n"
    "    model.parameters(), lr=cfg.lr,\n"
    ")\n"
    "```\n"
    "\n"
    "**Why a dict, not `if/elif/else`.** The dict is data — you can "
    "register a new optimizer from a plugin, iterate the keys for a "
    "CLI `--help`, or test the dispatch table without instantiating "
    "any optimizer. An `if/elif` chain hides this in control flow.\n"
    "\n"
    "**Store CLASSES, not instances.** The dict holds the class "
    "itself (`SGD`, not `SGD(...)`), because the instance needs "
    "`model.parameters()` and per-run kwargs. The class is the "
    "factory; you call it at construction time.\n"
    "\n"
    "**Why a `KeyError` is the right failure mode.** If the user "
    "passes `--optimizer rmsprop` and you forgot to register it, "
    "`KeyError('rmsprop')` halts the run cleanly with the offending "
    "name in the message. Wrapping it in a try/except that returns "
    "`None` produces a much more confusing downstream failure."
)

RECAP_PARAM_GROUP_DICTS = (
    "## Config: param-group dict list — quick refresher\n"
    "\n"
    "PyTorch's optimizer constructor accepts EITHER a flat parameter "
    "iterable OR a list of dicts. The dict form lets you set "
    "different hyperparameters per group (different LR for backbone "
    "vs head, no weight decay on biases, etc.):\n"
    "\n"
    "```python\n"
    "optimizer = torch.optim.Adam([\n"
    "    {'params': encoder.parameters(), 'lr': 1e-4},\n"
    "    {'params': head.parameters(),    'lr': 1e-2},\n"
    "], lr=1e-3)  # ← default LR (used when a group omits the key)\n"
    "```\n"
    "\n"
    "**Each dict MUST have a `params` key.** Other keys (`lr`, "
    "`weight_decay`, `momentum`, `betas`) override the constructor "
    "defaults for THAT group only.\n"
    "\n"
    "**Keys you don't set fall through to the top-level default.** "
    "If a group dict has `{'params': ..., 'lr': 1e-4}` and you pass "
    "`weight_decay=0.01` at the constructor level, that group gets "
    "`weight_decay=0.01`. The optimizer fills in missing keys.\n"
    "\n"
    "**Common pattern — no decay on biases/LayerNorm.** Two groups, "
    "same LR, different `weight_decay`. The split is structural, "
    "not by optimizer choice.\n"
    "\n"
    "**Calling `.parameters()` returns a GENERATOR**, which is "
    "single-use. If you build two groups from the same module you'll "
    "exhaust the iterator. Wrap with `list(model.parameters())` if "
    "you need to introspect or reuse."
)

RECAP_HPARAM_PRECEDENCE = (
    "## Config: hparam precedence merge — quick refresher\n"
    "\n"
    "Real training scripts pull config from multiple sources with a "
    "clear precedence chain. The standard order (lowest → highest):\n"
    "\n"
    "```\n"
    "defaults  <  config file  <  CLI overrides  <  programmatic\n"
    "```\n"
    "\n"
    "Implementation is just chained `dict.update()`:\n"
    "\n"
    "```python\n"
    "def merge_config(defaults, file_cfg, cli_args):\n"
    "    out = {}\n"
    "    out.update(defaults)\n"
    "    out.update(file_cfg)   # file overrides defaults\n"
    "    out.update(cli_args)   # CLI overrides file\n"
    "    return out\n"
    "```\n"
    "\n"
    "**Why later-wins is the right semantic.** The user typed the CLI "
    "args most recently — they're the strongest expression of intent. "
    "A YAML config they wrote yesterday is a weaker signal. Defaults "
    "are the weakest signal of all (we picked them).\n"
    "\n"
    "**`dict.update(other)` is shallow.** If a value is a nested "
    "dict (e.g. `{'optimizer': {'lr': 1e-3}}`), `update` REPLACES "
    "the whole subdict instead of merging into it. For nested "
    "configs you want a recursive merge (Hydra, OmegaConf). But for "
    "flat training args the shallow merge is correct.\n"
    "\n"
    "**Don't mutate the inputs.** Start with `out = {}` so the "
    "caller's `defaults` dict isn't polluted across runs."
)

RECAP_PARAMS_VS_GROUPS = (
    "## Config: params iterable vs groups — quick refresher\n"
    "\n"
    "`torch.optim.Optimizer.__init__` accepts a single positional "
    "argument `params` whose type is polymorphic:\n"
    "\n"
    "1. **Flat iterable of `Tensor`** — every param shares the "
    "constructor-level hyperparameters.\n"
    "2. **Iterable of `dict[str, Any]`** — each dict is a parameter "
    "GROUP with its own per-group hyperparameters.\n"
    "\n"
    "Internally the optimizer normalizes everything to "
    "`self.param_groups: list[dict]`. The flat form is just a "
    "one-group convenience.\n"
    "\n"
    "**How the optimizer tells them apart.** It peeks at the first "
    "element: if it's a `Tensor` → flat-iterable mode; if it's a "
    "`dict` → group mode. (You'll see this in the PyTorch source: "
    "`if isinstance(param_groups[0], Tensor): param_groups = "
    "[{'params': param_groups}]`.)\n"
    "\n"
    "**You CAN'T mix them.** A list like "
    "`[tensor_a, {'params': [tensor_b]}]` is invalid — the dispatch "
    "looks at element 0 only.\n"
    "\n"
    "**After construction, both forms produce the same shape.** "
    "`optimizer.param_groups` is always a `list[dict]`. The flat "
    "form just has `len(param_groups) == 1`.\n"
    "\n"
    "**Why this matters for debugging.** When you set "
    "`for group in optimizer.param_groups: ...`, the loop works "
    "either way. Code that assumed a flat list of params would "
    "silently break the moment you added a second group."
)

RECAP_SWEEP_CONFIG = (
    "## Config: wandb sweep config dict — quick refresher\n"
    "\n"
    "A wandb sweep is configured by a Python dict (or equivalent "
    "YAML) with a fixed top-level schema:\n"
    "\n"
    "```python\n"
    "sweep_config = {\n"
    "    'method': 'bayes',                 # 'grid' | 'random' | 'bayes'\n"
    "    'metric': {                         # what bayes optimizes\n"
    "        'name': 'val/loss',\n"
    "        'goal': 'minimize',\n"
    "    },\n"
    "    'parameters': {\n"
    "        'lr':         {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-1},\n"
    "        'batch_size': {'values': [16, 32, 64]},\n"
    "        'optimizer':  {'value': 'adam'},\n"
    "    },\n"
    "}\n"
    "```\n"
    "\n"
    "**Four keys you must understand:**\n"
    "- `method` — the search algorithm. `grid` enumerates every "
    "combination, `random` samples i.i.d., `bayes` builds a Gaussian-"
    "process surrogate on `metric`.\n"
    "- `metric` — only consumed by `bayes`. `grid` and `random` "
    "ignore it but still log it.\n"
    "- `parameters` — a dict-of-dicts. Each inner dict specifies "
    "EITHER a distribution to sample from OR a discrete `values` "
    "list OR a fixed `value`.\n"
    "- (optional) `name`, `program`, `early_terminate` — not "
    "required for a minimal sweep.\n"
    "\n"
    "**You don't need the wandb package to BUILD the config.** It's "
    "just a Python dict — exercising the shape doesn't require any "
    "external dep."
)

RECAP_SWEEP_DISTRIBUTION = (
    "## Config: sweep hparam distribution — quick refresher\n"
    "\n"
    "Inside a wandb sweep's `parameters` block, each hparam picks "
    "ONE of these specifiers:\n"
    "\n"
    "| specifier | when to use | example |\n"
    "|---|---|---|\n"
    "| `{'value': X}` | fixed — don't sweep this | `{'value': 'adam'}` |\n"
    "| `{'values': [...]}` | small discrete set | `{'values': [16, 32, 64]}` |\n"
    "| `{'distribution': 'uniform', 'min': ..., 'max': ...}` | uniformly on a linear scale | learning rate FOR a model where lr scale is small (rare) |\n"
    "| `{'distribution': 'log_uniform_values', 'min': ..., 'max': ...}` | uniformly on log scale | LR, weight decay, anything that spans orders of magnitude |\n"
    "| `{'distribution': 'int_uniform', 'min': ..., 'max': ...}` | integer-valued | num_layers, batch size if continuous |\n"
    "| `{'distribution': 'categorical', 'values': [...]}` | unordered categories | optimizer name, activation function |\n"
    "\n"
    "**`log_uniform_values` is the right default for LR.** A linear "
    "uniform sample between `1e-5` and `1e-1` wastes ~99% of its "
    "samples on the large-LR end (anything above `~1e-2` for most "
    "vision tasks is divergent). Log-uniform gives even coverage of "
    "the decade scale.\n"
    "\n"
    "**`uniform` vs `log_uniform_values` is the most common bug.** "
    "Picking `uniform` for LR gives you a sweep that's effectively "
    "just trying lr in `[5e-2, 1e-1]` with one or two stray samples "
    "in the productive range.\n"
    "\n"
    "**`int_uniform` rounds at sample time** — you don't have to "
    "wrap it in `int(...)` yourself."
)

RECAP_NESTED_PARAM_GROUP_LOOP = (
    "## Config: nested param-group loop — quick refresher\n"
    "\n"
    "Every PyTorch optimizer normalizes its params into "
    "`self.param_groups: list[dict]`. To touch every parameter "
    "tensor you nest two loops:\n"
    "\n"
    "```python\n"
    "for group in optimizer.param_groups:\n"
    "    lr = group['lr']\n"
    "    for p in group['params']:\n"
    "        if p.grad is None:\n"
    "            continue\n"
    "        p.add_(p.grad, alpha=-lr)\n"
    "```\n"
    "\n"
    "**This is the universal optimizer step pattern.** SGD, Adam, "
    "RMSprop, AdamW — every reference implementation in "
    "`torch/optim/*.py` starts with this nested loop.\n"
    "\n"
    "**Outer loop = per-group hparam fetch.** `lr`, `weight_decay`, "
    "`momentum`, `betas` all live on the group dict. Naive flat-"
    "iteration `for p in optimizer.parameters()` doesn't exist on "
    "the optimizer and would lose the per-group hparams.\n"
    "\n"
    "**Skip `p.grad is None` parameters.** When `set_to_none=True` "
    "is the zero_grad mode (PyTorch 1.7+ default), un-touched "
    "params have `grad=None` and you must not try to read it. The "
    "check is one line and prevents crashes on partially-frozen "
    "models.\n"
    "\n"
    "**Inner-loop variable naming.** `p` for the param, `g` for "
    "`p.grad`, `lr` for the group LR — these names are conventional "
    "across the PyTorch optim source. Match them when you write "
    "custom optimizers so reviewers don't have to context-switch."
)


# ---------------------------------------------------------------------------
# Specs.
# ---------------------------------------------------------------------------

SPECS = [

    # =========================================================
    # dataclass-training-args — ex1
    # =========================================================
    {
        "atom_id": "dataclass-training-args",
        "subtopic": "Config: @dataclass training args",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_DATACLASS_ARGS,
        "exercise_index": 1,
        "exercise_title": "TrainingArgs dataclass with __post_init__ validation + asdict round-trip",
        "slug": "training-args-dataclass-with-post-init-validation-and-asdict",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["dataclass", "config", "validation", "asdict"],
        "kcs": [
            "dataclass-default-fields",
            "post-init-validation-and-asdict",
        ],
        "lo": (
            "Apply the `@dataclass` + `__post_init__` + "
            "`dataclasses.asdict` pattern to define a TrainingArgs "
            "container that validates its fields and round-trips "
            "to a plain dict."
        ),
        "prompt_body": (
            "Implement `ex1_make_training_args` so it returns a "
            "`TrainingArgs` dataclass instance built from the given "
            "overrides. The dataclass must have:\n\n"
            "1. Fields with defaults: `lr: float = 1e-3`, "
            "`batch_size: int = 32`, `epochs: int = 10`, "
            "`optimizer_name: str = 'adam'`.\n"
            "2. A `__post_init__` that raises `ValueError` if "
            "`lr <= 0`, if `batch_size < 1`, if `epochs < 1`, or if "
            "`optimizer_name` is not in `{'sgd', 'adam', 'adamw'}`.\n"
            "3. A method `to_dict(self) -> dict` that returns "
            "`dataclasses.asdict(self)`.\n\n"
            "`ex1_make_training_args` should:\n"
            "- Accept `**overrides`.\n"
            "- Construct a `TrainingArgs(**overrides)` and return "
            "it.\n\n"
            "The test verifies the defaults, the validation, and "
            "the `to_dict()` round-trip."
        ),
        "stub": (
            "from dataclasses import dataclass, asdict\n"
            "\n"
            "@dataclass\n"
            "class TrainingArgs:\n"
            "    # Fill in the fields and __post_init__ here.\n"
            "    pass\n"
            "\n"
            "\n"
            "def ex1_make_training_args(**overrides) -> 'TrainingArgs':\n"
            '    """Build a TrainingArgs with the given overrides."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# === Defaults ===\n"
            "args = ex1_make_training_args()\n"
            "assert args.lr == 1e-3, f'default lr should be 1e-3, got {args.lr}'\n"
            "assert args.batch_size == 32, f'default batch_size should be 32, got {args.batch_size}'\n"
            "assert args.epochs == 10, f'default epochs should be 10, got {args.epochs}'\n"
            "assert args.optimizer_name == 'adam', f'default optimizer_name should be adam, got {args.optimizer_name!r}'\n"
            "\n"
            "# === Overrides ===\n"
            "args2 = ex1_make_training_args(lr=3e-4, optimizer_name='sgd', epochs=5)\n"
            "assert args2.lr == 3e-4\n"
            "assert args2.optimizer_name == 'sgd'\n"
            "assert args2.epochs == 5\n"
            "assert args2.batch_size == 32, 'unset field should still hold the default'\n"
            "\n"
            "# === to_dict round-trip ===\n"
            "d = args2.to_dict()\n"
            "assert isinstance(d, dict), f'to_dict must return dict, got {type(d).__name__}'\n"
            "assert d == {'lr': 3e-4, 'batch_size': 32, 'epochs': 5, 'optimizer_name': 'sgd'}, (\n"
            "    f'to_dict round-trip wrong: {d}'\n"
            ")\n"
            "# Rebuild from dict.\n"
            "args3 = ex1_make_training_args(**d)\n"
            "assert args3.to_dict() == d, 'rebuild-from-dict failed'\n"
            "\n"
            "# === Validation errors ===\n"
            "for bad in [\n"
            "    dict(lr=0.0),\n"
            "    dict(lr=-1e-3),\n"
            "    dict(batch_size=0),\n"
            "    dict(batch_size=-1),\n"
            "    dict(epochs=0),\n"
            "    dict(optimizer_name='rmsprop'),\n"
            "    dict(optimizer_name=''),\n"
            "]:\n"
            "    try:\n"
            "        ex1_make_training_args(**bad)\n"
            "    except ValueError:\n"
            "        pass\n"
            "    else:\n"
            "        raise AssertionError(f'expected ValueError for {bad}, but no error raised')\n"
            "\n"
            "# === Valid optimizer names ===\n"
            "for name in ['sgd', 'adam', 'adamw']:\n"
            "    a = ex1_make_training_args(optimizer_name=name)\n"
            "    assert a.optimizer_name == name\n"
            "\n"
            "# === Dataclass is mutable (frozen=False default) ===\n"
            "args.lr = 5e-4\n"
            "assert args.lr == 5e-4, 'dataclass should be mutable for override-after-construct'\n"
            "\n"
            "# === asdict is recursive but flat here — just sanity check ===\n"
            "from dataclasses import asdict as _asdict\n"
            "assert _asdict(args) == args.to_dict(), 'to_dict must delegate to dataclasses.asdict'"
        ),
        "solution_body": (
            "from dataclasses import dataclass, asdict\n"
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
            "        if self.epochs < 1:\n"
            "            raise ValueError(f'epochs must be >= 1, got {self.epochs}')\n"
            "        if self.optimizer_name not in {'sgd', 'adam', 'adamw'}:\n"
            "            raise ValueError(\n"
            "                f'optimizer_name must be one of sgd/adam/adamw, '\n"
            "                f'got {self.optimizer_name!r}'\n"
            "            )\n"
            "\n"
            "    def to_dict(self):\n"
            "        return asdict(self)\n"
            "\n"
            "\n"
            "def ex1_make_training_args(**overrides):\n"
            "    return TrainingArgs(**overrides)"
        ),
        "solution_notes": (
            "**Why `__post_init__` and not a regular `__init__`.** "
            "Dataclasses generate `__init__` for you. Replacing it "
            "defeats the whole point. `__post_init__` runs AFTER the "
            "generated init has assigned every field, which is "
            "exactly when you want validation to fire.\n\n"
            "**Why `set` for the optimizer-name check.** "
            "`self.optimizer_name not in {'sgd', 'adam', 'adamw'}` is "
            "O(1) and reads like math. A list (`['sgd', 'adam', "
            "'adamw']`) is fine for 3 items but signals you might "
            "care about order — you don't.\n\n"
            "**`asdict` is recursive.** If you nest dataclasses "
            "(e.g. `optimizer_config: OptimizerConfig`), `asdict` "
            "unrolls the inner dataclass into a sub-dict. That's why "
            "real training scripts can dump the whole args object "
            "to JSON or to wandb in one call."
        ),
        "extra_imports": [],
    },

    # =========================================================
    # optimizer-class-dispatch — ex1
    # =========================================================
    {
        "atom_id": "optimizer-class-dispatch",
        "subtopic": "Config: Optimizer class dispatch",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_OPTIMIZER_DISPATCH,
        "exercise_index": 1,
        "exercise_title": "build an optimizer from a config string via a class-dispatch dict",
        "slug": "build-an-optimizer-from-a-config-string-via-a-class-dispatch-dict",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["dispatch-table", "registry", "factory", "optimizer-name"],
        "kcs": [
            "class-dispatch-dict-lookup",
            "optimizer-factory-call",
        ],
        "lo": (
            "Apply the dispatch-dict factory pattern "
            "`OPTIMIZER_CLASSES[name](params, lr=...)` to construct "
            "the correct `torch.optim` optimizer from a string."
        ),
        "prompt_body": (
            "Implement `ex1_build_optimizer(name, params, lr)`. The "
            "config-string → optimizer-instance dispatch.\n\n"
            "1. Define a module-level dict `OPTIMIZER_CLASSES` "
            "mapping `'sgd'`, `'adam'`, `'adamw'` to "
            "`torch.optim.SGD`, `torch.optim.Adam`, "
            "`torch.optim.AdamW` respectively.\n"
            "2. `ex1_build_optimizer` looks up the class by `name`, "
            "calls it with `(params, lr=lr)`, and returns the "
            "instance.\n"
            "3. An unknown `name` should raise `KeyError` with the "
            "offending name in the message. The default dict "
            "indexing already does this.\n\n"
            "Inputs:\n"
            "- `name`: `str` from `{'sgd', 'adam', 'adamw'}`.\n"
            "- `params`: iterable of parameter tensors.\n"
            "- `lr`: float learning rate.\n\n"
            "Output: `torch.optim.Optimizer` instance.\n\n"
            "Why this matters: dispatching by string is how real "
            "training scripts let you sweep optimizer choice from a "
            "config file without code changes."
        ),
        "stub": (
            "OPTIMIZER_CLASSES = {\n"
            "    # Fill in: 'sgd', 'adam', 'adamw' -> torch.optim classes.\n"
            "}\n"
            "\n"
            "def ex1_build_optimizer(name: str, params, lr: float):\n"
            '    """Look up name in OPTIMIZER_CLASSES and instantiate."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# === Module-level dispatch dict ===\n"
            "assert isinstance(OPTIMIZER_CLASSES, dict), 'OPTIMIZER_CLASSES must be a dict'\n"
            "assert set(OPTIMIZER_CLASSES.keys()) == {'sgd', 'adam', 'adamw'}, (\n"
            "    f'OPTIMIZER_CLASSES keys must be {{sgd, adam, adamw}}, got {set(OPTIMIZER_CLASSES.keys())}'\n"
            ")\n"
            "# Dict stores CLASSES, not instances.\n"
            "assert OPTIMIZER_CLASSES['sgd'] is t.optim.SGD\n"
            "assert OPTIMIZER_CLASSES['adam'] is t.optim.Adam\n"
            "assert OPTIMIZER_CLASSES['adamw'] is t.optim.AdamW\n"
            "# These are types, not optimizer objects.\n"
            "for cls in OPTIMIZER_CLASSES.values():\n"
            "    assert isinstance(cls, type), f'expected a class, got {cls!r}'\n"
            "\n"
            "# === Construct each kind ===\n"
            "param = t.nn.Parameter(t.randn(3))\n"
            "for name, expected_cls in [\n"
            "    ('sgd', t.optim.SGD),\n"
            "    ('adam', t.optim.Adam),\n"
            "    ('adamw', t.optim.AdamW),\n"
            "]:\n"
            "    opt = ex1_build_optimizer(name, [param], lr=1e-2)\n"
            "    assert isinstance(opt, expected_cls), (\n"
            "        f'name={name!r}: expected {expected_cls.__name__}, got {type(opt).__name__}'\n"
            "    )\n"
            "    assert opt.param_groups[0]['lr'] == 1e-2, f'lr not propagated for {name}'\n"
            "    assert len(opt.param_groups[0]['params']) == 1\n"
            "\n"
            "# === Round-trip a tiny step to prove the optimizer actually works ===\n"
            "p2 = t.nn.Parameter(t.zeros(2))\n"
            "opt2 = ex1_build_optimizer('sgd', [p2], lr=0.1)\n"
            "p2.grad = t.tensor([1.0, -2.0])\n"
            "opt2.step()\n"
            "# SGD: p_new = p - lr * grad = -0.1 * [1, -2] = [-0.1, 0.2]\n"
            "assert t.allclose(p2.detach(), t.tensor([-0.1, 0.2]), atol=1e-6), (\n"
            "    f'SGD step produced wrong value: {p2.detach()}'\n"
            ")\n"
            "\n"
            "# === Unknown name => KeyError ===\n"
            "try:\n"
            "    ex1_build_optimizer('rmsprop', [t.nn.Parameter(t.randn(2))], lr=1e-3)\n"
            "except KeyError as e:\n"
            "    # The KeyError's args should mention the unknown name.\n"
            "    assert 'rmsprop' in str(e), f'KeyError should mention the bad name, got {e!r}'\n"
            "else:\n"
            "    raise AssertionError('expected KeyError for unknown optimizer name')\n"
            "\n"
            "# === Different LRs construct independent optimizers ===\n"
            "opts = [\n"
            "    ex1_build_optimizer('adam', [t.nn.Parameter(t.randn(3))], lr=lr)\n"
            "    for lr in [1e-4, 3e-4, 1e-3]\n"
            "]\n"
            "lrs = [o.param_groups[0]['lr'] for o in opts]\n"
            "assert lrs == [1e-4, 3e-4, 1e-3], f'per-call lr not isolated, got {lrs}'"
        ),
        "solution_body": (
            "OPTIMIZER_CLASSES = {\n"
            "    'sgd':   t.optim.SGD,\n"
            "    'adam':  t.optim.Adam,\n"
            "    'adamw': t.optim.AdamW,\n"
            "}\n"
            "\n"
            "def ex1_build_optimizer(name, params, lr):\n"
            "    cls = OPTIMIZER_CLASSES[name]\n"
            "    return cls(params, lr=lr)"
        ),
        "solution_notes": (
            "**Why `OPTIMIZER_CLASSES[name]` not `getattr(t.optim, "
            "name)`.** The `getattr` form would let users pass any "
            "attribute of the `torch.optim` module — including "
            "`Optimizer` (the abstract base), `lr_scheduler`, etc. "
            "An explicit registry is a SECURITY/CORRECTNESS "
            "boundary: only the names you registered are reachable.\n\n"
            "**Letting `KeyError` bubble.** A common anti-pattern is "
            "`try: ... except KeyError: return None`. Now the caller "
            "downstream gets `AttributeError: 'NoneType' object has "
            "no attribute 'step'` 10 stack frames deep with no hint "
            "about the config typo. Don't swallow it.\n\n"
            "**Extending the registry.** Want to add Lion or "
            "Sophia? Just `OPTIMIZER_CLASSES['lion'] = LionOptimizer` "
            "before the dispatch runs. The factory function itself "
            "doesn't change — that's the whole win over `if/elif`."
        ),
        "extra_imports": [],
    },

    # =========================================================
    # param-group-dict-list — ex1
    # =========================================================
    {
        "atom_id": "param-group-dict-list",
        "subtopic": "Config: param-group dict list",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_PARAM_GROUP_DICTS,
        "exercise_index": 1,
        "exercise_title": "build differential-LR param groups for encoder vs head",
        "slug": "build-differential-lr-param-groups-for-encoder-vs-head",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["param-groups", "differential-lr", "fine-tuning", "transfer"],
        "kcs": [
            "param-group-dict-with-lr-override",
            "list-vs-generator-parameters",
        ],
        "lo": (
            "Apply the `[{'params': ..., 'lr': ...}, ...]` dict-list "
            "construction to give different learning rates to two "
            "halves of a model, and verify the optimizer reads "
            "back each group's LR correctly."
        ),
        "prompt_body": (
            "Implement `ex1_make_param_groups(encoder, head, "
            "encoder_lr, head_lr)`. The classical transfer-learning "
            "differential-LR setup.\n\n"
            "1. Build a list of TWO dicts:\n"
            "   - `{'params': list(encoder.parameters()), 'lr': "
            "encoder_lr}`\n"
            "   - `{'params': list(head.parameters()), 'lr': "
            "head_lr}`\n"
            "2. Return the list.\n\n"
            "Why wrap in `list(...)`: `.parameters()` returns a "
            "single-use generator. If the optimizer iterates it "
            "more than once (it doesn't normally, but downstream "
            "tooling like checkpoint loaders sometimes does), the "
            "second pass yields nothing. Materializing into a list "
            "is the safe default.\n\n"
            "Inputs:\n"
            "- `encoder`, `head`: `nn.Module` instances.\n"
            "- `encoder_lr`, `head_lr`: floats.\n\n"
            "Output: `list[dict]` ready to pass as the first "
            "argument to any `torch.optim` optimizer."
        ),
        "stub": (
            "def ex1_make_param_groups(encoder, head, encoder_lr: float, head_lr: float):\n"
            '    """Two-group param dict-list with per-group lr."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "import torch.nn as nn\n"
            "\n"
            "encoder = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 16))\n"
            "head = nn.Linear(16, 3)\n"
            "\n"
            "groups = ex1_make_param_groups(encoder, head, encoder_lr=1e-4, head_lr=1e-2)\n"
            "assert isinstance(groups, list), f'must return a list, got {type(groups).__name__}'\n"
            "assert len(groups) == 2, f'must return 2 groups, got {len(groups)}'\n"
            "\n"
            "# === Each group is a dict with the right keys ===\n"
            "for i, g in enumerate(groups):\n"
            "    assert isinstance(g, dict), f'group {i} must be a dict, got {type(g).__name__}'\n"
            "    assert 'params' in g, f'group {i} missing params key'\n"
            "    assert 'lr' in g, f'group {i} missing lr key'\n"
            "\n"
            "# === Group order: encoder first, head second ===\n"
            "assert groups[0]['lr'] == 1e-4, f'group 0 lr should be encoder_lr=1e-4, got {groups[0][\"lr\"]}'\n"
            "assert groups[1]['lr'] == 1e-2, f'group 1 lr should be head_lr=1e-2, got {groups[1][\"lr\"]}'\n"
            "\n"
            "# === Param counts match ===\n"
            "enc_params = list(encoder.parameters())\n"
            "head_params = list(head.parameters())\n"
            "assert len(groups[0]['params']) == len(enc_params), (\n"
            "    f'encoder group has {len(groups[0][\"params\"])} params, expected {len(enc_params)}'\n"
            ")\n"
            "assert len(groups[1]['params']) == len(head_params), (\n"
            "    f'head group has {len(groups[1][\"params\"])} params, expected {len(head_params)}'\n"
            ")\n"
            "# Encoder has 4 tensors (2 linears × {weight,bias}), head has 2.\n"
            "assert len(groups[0]['params']) == 4, f'expected 4 encoder params, got {len(groups[0][\"params\"])}'\n"
            "assert len(groups[1]['params']) == 2, f'expected 2 head params, got {len(groups[1][\"params\"])}'\n"
            "\n"
            "# === params is a LIST (materialized), not a generator ===\n"
            "import types\n"
            "for i, g in enumerate(groups):\n"
            "    assert not isinstance(g['params'], types.GeneratorType), (\n"
            "        f'group {i} params is a generator — should be materialized to a list'\n"
            "    )\n"
            "\n"
            "# === Plug into a real optimizer; read back per-group LR ===\n"
            "opt = t.optim.Adam(groups)\n"
            "assert len(opt.param_groups) == 2\n"
            "assert opt.param_groups[0]['lr'] == 1e-4\n"
            "assert opt.param_groups[1]['lr'] == 1e-2\n"
            "# Other defaults filled in by Adam (e.g. betas).\n"
            "for g in opt.param_groups:\n"
            "    assert 'betas' in g, 'Adam should fill in betas default per group'\n"
            "\n"
            "# === Run one step; encoder moves slower than head ===\n"
            "x = t.randn(4, 8)\n"
            "logits = head(encoder(x))\n"
            "loss = logits.pow(2).mean()\n"
            "# Snapshot a representative param from each group.\n"
            "enc_w = encoder[0].weight.detach().clone()\n"
            "head_w = head.weight.detach().clone()\n"
            "opt.zero_grad()\n"
            "loss.backward()\n"
            "opt.step()\n"
            "enc_delta = (encoder[0].weight - enc_w).abs().mean().item()\n"
            "head_delta = (head.weight - head_w).abs().mean().item()\n"
            "# Both moved, but per-step magnitude is gated by LR.\n"
            "# (Not a strict assertion — gradient magnitudes differ — but encoder LR is 100x smaller.)\n"
            "assert enc_delta > 0, 'encoder weights should have moved at all'\n"
            "assert head_delta > 0, 'head weights should have moved at all'"
        ),
        "solution_body": (
            "def ex1_make_param_groups(encoder, head, encoder_lr, head_lr):\n"
            "    return [\n"
            "        {'params': list(encoder.parameters()), 'lr': encoder_lr},\n"
            "        {'params': list(head.parameters()),    'lr': head_lr},\n"
            "    ]"
        ),
        "solution_notes": (
            "**Why both groups need `'params'`.** It's the ONE "
            "required key in the param-group spec. Forgetting it "
            "raises `ValueError: optimizer got an empty parameter "
            "list` or worse, silently constructs a no-param group.\n\n"
            "**What if I want the same LR for both halves?** Drop "
            "the `'lr'` key — the optimizer's top-level `lr=` "
            "kwarg falls through. Param groups are most useful when "
            "DIFFERENT hparams apply, not for cosmetic grouping.\n\n"
            "**Three-group variant — no-decay biases.** A common "
            "extension is to split `params` into two groups, one "
            "for `weight` tensors (with `weight_decay=0.01`) and "
            "one for biases + LayerNorm (with `weight_decay=0`). "
            "Same mechanism — just three dicts in the list, "
            "filtered by `p.dim() > 1` and name."
        ),
        "extra_imports": [],
    },

    # =========================================================
    # hparam-precedence-merge — ex1
    # =========================================================
    {
        "atom_id": "hparam-precedence-merge",
        "subtopic": "Config: hparam precedence merge",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_HPARAM_PRECEDENCE,
        "exercise_index": 1,
        "exercise_title": "merge defaults < config-file < CLI args via dict.update",
        "slug": "merge-defaults-config-file-cli-args-via-dict-update",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["config", "precedence", "merge", "cli"],
        "kcs": [
            "dict-update-later-wins",
            "input-isolation-no-mutation",
        ],
        "lo": (
            "Apply chained `dict.update` to merge three config "
            "layers (defaults, file, CLI) with strict "
            "later-overrides-earlier semantics, without mutating "
            "any input."
        ),
        "prompt_body": (
            "Implement `ex1_merge_config(defaults, file_cfg, "
            "cli_args)`. The standard training-script config "
            "merge.\n\n"
            "Precedence (lowest → highest):\n"
            "  `defaults  <  file_cfg  <  cli_args`\n"
            "\n"
            "So CLI overrides file overrides defaults. Same as "
            "argparse + YAML.\n\n"
            "Algorithm:\n"
            "1. Start with an empty dict `out = {}`.\n"
            "2. `out.update(defaults)` (lowest precedence — fills "
            "every default key).\n"
            "3. `out.update(file_cfg)` (file overrides).\n"
            "4. `out.update(cli_args)` (CLI overrides — wins).\n"
            "5. Return `out`.\n\n"
            "Constraints:\n"
            "- Must NOT mutate any of the three input dicts.\n"
            "- A `None` value in `cli_args` should STILL override "
            "(it's an explicit choice). Don't filter Nones — that's "
            "a different design.\n"
            "- A key that appears ONLY in `file_cfg` (not in "
            "defaults) is allowed — pass it through.\n\n"
            "Output: merged `dict`."
        ),
        "stub": (
            "def ex1_merge_config(defaults: dict, file_cfg: dict, cli_args: dict) -> dict:\n"
            '    """Merge three config layers; later overrides earlier."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# === Basic precedence chain ===\n"
            "defaults = {'lr': 1e-3, 'batch_size': 32, 'epochs': 10}\n"
            "file_cfg = {'lr': 3e-4, 'epochs': 5}\n"
            "cli_args = {'lr': 1e-4}\n"
            "\n"
            "merged = ex1_merge_config(defaults, file_cfg, cli_args)\n"
            "assert merged == {'lr': 1e-4, 'batch_size': 32, 'epochs': 5}, (\n"
            "    f'precedence wrong: {merged}\\n'\n"
            "    f'expected lr from CLI, batch_size from defaults, epochs from file'\n"
            ")\n"
            "\n"
            "# === Inputs are NOT mutated ===\n"
            "assert defaults == {'lr': 1e-3, 'batch_size': 32, 'epochs': 10}, 'defaults was mutated'\n"
            "assert file_cfg == {'lr': 3e-4, 'epochs': 5}, 'file_cfg was mutated'\n"
            "assert cli_args == {'lr': 1e-4}, 'cli_args was mutated'\n"
            "\n"
            "# === All three layers empty ===\n"
            "assert ex1_merge_config({}, {}, {}) == {}\n"
            "\n"
            "# === Only defaults ===\n"
            "assert ex1_merge_config({'a': 1}, {}, {}) == {'a': 1}\n"
            "\n"
            "# === Only file ===\n"
            "assert ex1_merge_config({}, {'a': 1}, {}) == {'a': 1}\n"
            "\n"
            "# === CLI introduces a new key ===\n"
            "assert ex1_merge_config({'a': 1}, {}, {'b': 2}) == {'a': 1, 'b': 2}\n"
            "\n"
            "# === None in CLI overrides ===\n"
            "out = ex1_merge_config({'lr': 1e-3}, {}, {'lr': None})\n"
            "assert out == {'lr': None}, f'None should override, got {out}'\n"
            "\n"
            "# === File-only key passes through ===\n"
            "out = ex1_merge_config({'a': 1}, {'b': 2}, {})\n"
            "assert out == {'a': 1, 'b': 2}\n"
            "\n"
            "# === Result is a fresh dict (not an alias) ===\n"
            "d = {'x': 1}\n"
            "out = ex1_merge_config(d, {}, {})\n"
            "out['y'] = 99\n"
            "assert 'y' not in d, 'mutating the result must not back-propagate to defaults'\n"
            "\n"
            "# === Order matters — three-way tie test ===\n"
            "# All three layers set 'lr'. CLI must win.\n"
            "out = ex1_merge_config({'lr': 1.0}, {'lr': 2.0}, {'lr': 3.0})\n"
            "assert out['lr'] == 3.0, f'CLI must win three-way; got {out}'\n"
            "# Without CLI, file wins.\n"
            "out = ex1_merge_config({'lr': 1.0}, {'lr': 2.0}, {})\n"
            "assert out['lr'] == 2.0\n"
            "# Without file, defaults.\n"
            "out = ex1_merge_config({'lr': 1.0}, {}, {})\n"
            "assert out['lr'] == 1.0"
        ),
        "solution_body": (
            "def ex1_merge_config(defaults, file_cfg, cli_args):\n"
            "    out = {}\n"
            "    out.update(defaults)\n"
            "    out.update(file_cfg)\n"
            "    out.update(cli_args)\n"
            "    return out"
        ),
        "solution_notes": (
            "**`dict.update(other)` is the right primitive.** It "
            "writes every key from `other` into `self`, OVERWRITING "
            "existing values. That's exactly the later-wins "
            "semantic.\n\n"
            "**Alternative one-liner: `{**defaults, **file_cfg, "
            "**cli_args}`.** Same semantics; sometimes considered "
            "more Pythonic. The explicit `update` form scales better "
            "if you add a fourth layer (e.g. env vars) — just one "
            "more line.\n\n"
            "**Don't filter out `None` from `cli_args`.** It's "
            "tempting to write `{k: v for k, v in cli_args.items() "
            "if v is not None}` to support `--lr` (no value = "
            "'use default'). But that conflates 'not provided' with "
            "'explicit None'. argparse handles this at the parser "
            "level by only adding a key if it was provided — use "
            "that mechanism, not post-hoc None filtering.\n\n"
            "**Shallow merge limitation.** If `defaults['optimizer'] "
            "= {'lr': 1e-3, 'beta1': 0.9}` and `file_cfg['optimizer'] "
            "= {'lr': 3e-4}`, the merge gives `{'optimizer': "
            "{'lr': 3e-4}}` — `beta1` is lost. Deep configs need "
            "OmegaConf or Hydra; flat configs don't."
        ),
        "extra_imports": [],
    },

    # =========================================================
    # params-iterable-vs-groups — ex1
    # =========================================================
    {
        "atom_id": "params-iterable-vs-groups",
        "subtopic": "Config: params iterable vs groups",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_PARAMS_VS_GROUPS,
        "exercise_index": 1,
        "exercise_title": "polymorphic params= signature: flat iterable vs dict-list both normalize to param_groups",
        "slug": "polymorphic-params-signature-flat-vs-dict-list-normalize",
        "bloom_level": "Analyze",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["polymorphism", "param_groups", "normalization", "isinstance-dispatch"],
        "kcs": [
            "params-dispatch-tensor-vs-dict",
            "param-groups-always-list-of-dict",
        ],
        "lo": (
            "Analyze how `torch.optim.Optimizer` normalizes its "
            "polymorphic `params=` argument: detect whether the "
            "caller passed a flat tensor iterable or a list of "
            "param-group dicts, and produce the canonical "
            "`list[dict]` form."
        ),
        "prompt_body": (
            "Implement `ex1_normalize_params(params, default_lr)`. "
            "A from-scratch reimplementation of the dispatch logic "
            "that PyTorch's `Optimizer.__init__` uses.\n\n"
            "1. Materialize `params` to a list (it may be a "
            "generator).\n"
            "2. Look at the FIRST element:\n"
            "   - If it's a `torch.Tensor` → flat-iterable mode. "
            "Wrap everything in a single group: `[{'params': "
            "[...all tensors...], 'lr': default_lr}]`.\n"
            "   - If it's a `dict` → group mode. Pass through, but "
            "fill in `lr=default_lr` for any group that lacks an "
            "`'lr'` key.\n"
            "3. Empty list → `ValueError('optimizer got an empty "
            "parameter list')` (matches the canonical PyTorch "
            "error).\n"
            "4. First element is neither tensor nor dict → "
            "`TypeError`.\n"
            "\n"
            "Output: `list[dict]` where each dict has at least "
            "`'params'` and `'lr'`."
        ),
        "stub": (
            "def ex1_normalize_params(params, default_lr: float):\n"
            '    """Normalize the polymorphic params= argument to list[dict]."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# === Flat iterable of tensors → single group ===\n"
            "p1 = t.nn.Parameter(t.randn(3))\n"
            "p2 = t.nn.Parameter(t.randn(2, 4))\n"
            "out = ex1_normalize_params([p1, p2], default_lr=1e-3)\n"
            "assert isinstance(out, list), f'must return list, got {type(out).__name__}'\n"
            "assert len(out) == 1, f'flat mode should produce 1 group, got {len(out)}'\n"
            "assert isinstance(out[0], dict)\n"
            "assert out[0]['lr'] == 1e-3\n"
            "assert len(out[0]['params']) == 2\n"
            "assert out[0]['params'][0] is p1 and out[0]['params'][1] is p2, (\n"
            "    'tensors must pass through by reference'\n"
            ")\n"
            "\n"
            "# === Generator of tensors (single-use) → works ===\n"
            "ps_gen = (t.nn.Parameter(t.randn(2)) for _ in range(3))\n"
            "out = ex1_normalize_params(ps_gen, default_lr=5e-4)\n"
            "assert len(out) == 1\n"
            "assert len(out[0]['params']) == 3\n"
            "assert out[0]['lr'] == 5e-4\n"
            "\n"
            "# === Dict-list (group mode) → each group preserved ===\n"
            "p3 = t.nn.Parameter(t.randn(4))\n"
            "p4 = t.nn.Parameter(t.randn(5))\n"
            "groups_in = [\n"
            "    {'params': [p3], 'lr': 1e-4},   # explicit lr\n"
            "    {'params': [p4]},                # falls through to default_lr\n"
            "]\n"
            "out = ex1_normalize_params(groups_in, default_lr=1e-3)\n"
            "assert len(out) == 2, f'group mode should preserve count, got {len(out)}'\n"
            "assert out[0]['lr'] == 1e-4, 'explicit per-group lr must be kept'\n"
            "assert out[1]['lr'] == 1e-3, 'missing lr should fall back to default'\n"
            "assert out[0]['params'] == [p3]\n"
            "assert out[1]['params'] == [p4]\n"
            "\n"
            "# === Empty list → ValueError ===\n"
            "try:\n"
            "    ex1_normalize_params([], default_lr=1e-3)\n"
            "except ValueError as e:\n"
            "    assert 'empty' in str(e).lower(), f'ValueError msg should mention empty, got {e!r}'\n"
            "else:\n"
            "    raise AssertionError('expected ValueError on empty params list')\n"
            "\n"
            "# === Wrong element type → TypeError ===\n"
            "try:\n"
            "    ex1_normalize_params(['not a tensor', 'or dict'], default_lr=1e-3)\n"
            "except TypeError:\n"
            "    pass\n"
            "else:\n"
            "    raise AssertionError('expected TypeError on non-tensor non-dict element')\n"
            "\n"
            "# === Both forms produce equivalent optimizer.param_groups when fed to torch.optim ===\n"
            "p5 = t.nn.Parameter(t.randn(3))\n"
            "p6 = t.nn.Parameter(t.randn(3))\n"
            "# Flat form\n"
            "norm_flat = ex1_normalize_params([p5, p6], default_lr=1e-2)\n"
            "opt_flat = t.optim.SGD(norm_flat)\n"
            "# Dict form\n"
            "norm_dict = ex1_normalize_params([{'params': [p5, p6]}], default_lr=1e-2)\n"
            "opt_dict = t.optim.SGD(norm_dict)\n"
            "# Both should produce a single group with lr=1e-2.\n"
            "assert len(opt_flat.param_groups) == 1\n"
            "assert len(opt_dict.param_groups) == 1\n"
            "assert opt_flat.param_groups[0]['lr'] == 1e-2\n"
            "assert opt_dict.param_groups[0]['lr'] == 1e-2\n"
            "\n"
            "# === Dispatch decision relies on FIRST element only ===\n"
            "# (We're not testing the 'can't mix' case — the spec says don't, but the\n"
            "#  dispatch only peeks at element 0, so a mixed list would be parsed as\n"
            "#  whatever element 0 is. Just confirm pure-dict still works with 3 groups.)\n"
            "groups3 = [\n"
            "    {'params': [t.nn.Parameter(t.randn(2))]},\n"
            "    {'params': [t.nn.Parameter(t.randn(3))], 'lr': 5e-5},\n"
            "    {'params': [t.nn.Parameter(t.randn(4))]},\n"
            "]\n"
            "out = ex1_normalize_params(groups3, default_lr=2e-3)\n"
            "assert len(out) == 3\n"
            "assert out[0]['lr'] == 2e-3\n"
            "assert out[1]['lr'] == 5e-5\n"
            "assert out[2]['lr'] == 2e-3"
        ),
        "solution_body": (
            "def ex1_normalize_params(params, default_lr):\n"
            "    materialized = list(params)\n"
            "    if not materialized:\n"
            "        raise ValueError('optimizer got an empty parameter list')\n"
            "    first = materialized[0]\n"
            "    if isinstance(first, t.Tensor):\n"
            "        return [{'params': materialized, 'lr': default_lr}]\n"
            "    if isinstance(first, dict):\n"
            "        out = []\n"
            "        for group in materialized:\n"
            "            g = dict(group)\n"
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
            "**`isinstance(first, t.Tensor)` is correct even for "
            "`nn.Parameter`.** `nn.Parameter` IS a `torch.Tensor` "
            "subclass — `isinstance` returns True. The dispatch "
            "works on the base class.\n\n"
            "**Why we `dict(group)` instead of mutating in place.** "
            "The caller's dict shouldn't be mutated as a side effect "
            "of optimizer construction. `dict(group)` makes a "
            "shallow copy; adding `'lr'` to the copy doesn't affect "
            "the original. PyTorch does the same.\n\n"
            "**This is real PyTorch source.** Read "
            "`torch/optim/optimizer.py` `Optimizer.__init__` and "
            "you'll see almost exactly this dispatch — minus the "
            "extra validation hooks (per-param NaN check, "
            "fused-vs-foreach branching, etc.). The polymorphism is "
            "intentional and stable across PyTorch versions."
        ),
        "extra_imports": [],
    },

    # =========================================================
    # sweep-config-dict — ex1
    # =========================================================
    {
        "atom_id": "sweep-config-dict",
        "subtopic": "Config: wandb sweep config dict",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_SWEEP_CONFIG,
        "exercise_index": 1,
        "exercise_title": "build a wandb sweep config: bayes + metric + parameters",
        "slug": "build-a-wandb-sweep-config-bayes-metric-parameters",
        "bloom_level": "Apply",
        "difficulty_num": 2,
        "difficulty_dots": "🔴🔴⚪⚪⚪",
        "keywords": ["wandb", "sweep", "bayes", "config-schema"],
        "kcs": [
            "sweep-config-top-level-schema",
            "bayes-method-metric-block",
        ],
        "lo": (
            "Apply the wandb sweep config dict schema to specify a "
            "bayes-optimized minimization sweep with a mix of "
            "fixed, discrete, and continuous hyperparameters."
        ),
        "prompt_body": (
            "Implement `ex1_build_sweep_config(metric_name)`. "
            "Construct a valid wandb sweep config dict (you don't "
            "need wandb installed — this exercises the dict shape "
            "only).\n\n"
            "The returned dict must contain:\n"
            "\n"
            "1. `'method': 'bayes'` — bayesian optimization.\n"
            "2. `'metric': {'name': metric_name, 'goal': "
            "'minimize'}` — what to optimize.\n"
            "3. `'parameters'`: a dict containing exactly these "
            "four entries:\n"
            "   - `'lr'`: log-uniform distribution between `1e-5` "
            "and `1e-1`. Use `'log_uniform_values'`.\n"
            "   - `'batch_size'`: discrete values `[16, 32, 64, "
            "128]`.\n"
            "   - `'optimizer'`: fixed at `'adam'`. Use `'value'`, "
            "not `'values'`.\n"
            "   - `'weight_decay'`: log-uniform between `1e-6` and "
            "`1e-2`.\n\n"
            "No wandb import needed; we exercise the dict shape "
            "only.\n\n"
            "Output: `dict` matching the wandb sweep schema."
        ),
        "stub": (
            "def ex1_build_sweep_config(metric_name: str) -> dict:\n"
            '    """Build a wandb sweep config dict (bayes + metric + parameters)."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "cfg = ex1_build_sweep_config('val/loss')\n"
            "\n"
            "# === Top-level schema ===\n"
            "assert isinstance(cfg, dict), f'must return dict, got {type(cfg).__name__}'\n"
            "assert cfg['method'] == 'bayes', f\"method should be 'bayes', got {cfg.get('method')!r}\"\n"
            "\n"
            "# === Metric block ===\n"
            "assert 'metric' in cfg, 'missing metric key'\n"
            "m = cfg['metric']\n"
            "assert isinstance(m, dict), 'metric must be a dict'\n"
            "assert m == {'name': 'val/loss', 'goal': 'minimize'}, f'metric block wrong: {m}'\n"
            "\n"
            "# === Parameters block ===\n"
            "assert 'parameters' in cfg, 'missing parameters key'\n"
            "p = cfg['parameters']\n"
            "assert isinstance(p, dict)\n"
            "assert set(p.keys()) == {'lr', 'batch_size', 'optimizer', 'weight_decay'}, (\n"
            "    f\"parameters keys must be exactly {{lr, batch_size, optimizer, weight_decay}}, got {set(p.keys())}\"\n"
            ")\n"
            "\n"
            "# === lr: log-uniform ===\n"
            "assert p['lr']['distribution'] == 'log_uniform_values', (\n"
            "    f\"lr distribution must be 'log_uniform_values', got {p['lr'].get('distribution')!r}\"\n"
            ")\n"
            "assert p['lr']['min'] == 1e-5\n"
            "assert p['lr']['max'] == 1e-1\n"
            "\n"
            "# === batch_size: discrete values ===\n"
            "assert p['batch_size'] == {'values': [16, 32, 64, 128]}, (\n"
            "    f'batch_size spec wrong: {p[\"batch_size\"]}'\n"
            ")\n"
            "assert 'value' not in p['batch_size'], 'use values (plural) for the discrete set, not value'\n"
            "\n"
            "# === optimizer: fixed ===\n"
            "assert p['optimizer'] == {'value': 'adam'}, (\n"
            "    f'optimizer spec wrong: {p[\"optimizer\"]}'\n"
            ")\n"
            "assert 'values' not in p['optimizer'], 'use value (singular) for a fixed param, not values'\n"
            "\n"
            "# === weight_decay: log-uniform ===\n"
            "assert p['weight_decay']['distribution'] == 'log_uniform_values'\n"
            "assert p['weight_decay']['min'] == 1e-6\n"
            "assert p['weight_decay']['max'] == 1e-2\n"
            "\n"
            "# === Pluggable metric_name ===\n"
            "cfg2 = ex1_build_sweep_config('test/accuracy')\n"
            "assert cfg2['metric']['name'] == 'test/accuracy'\n"
            "assert cfg2['metric']['goal'] == 'minimize', 'goal hard-coded as minimize per the spec'\n"
            "\n"
            "# === Result is JSON-serializable (wandb requires this) ===\n"
            "import json\n"
            "_ = json.dumps(cfg)  # would raise if non-serializable"
        ),
        "solution_body": (
            "def ex1_build_sweep_config(metric_name):\n"
            "    return {\n"
            "        'method': 'bayes',\n"
            "        'metric': {\n"
            "            'name': metric_name,\n"
            "            'goal': 'minimize',\n"
            "        },\n"
            "        'parameters': {\n"
            "            'lr': {\n"
            "                'distribution': 'log_uniform_values',\n"
            "                'min': 1e-5,\n"
            "                'max': 1e-1,\n"
            "            },\n"
            "            'batch_size': {'values': [16, 32, 64, 128]},\n"
            "            'optimizer': {'value': 'adam'},\n"
            "            'weight_decay': {\n"
            "                'distribution': 'log_uniform_values',\n"
            "                'min': 1e-6,\n"
            "                'max': 1e-2,\n"
            "            },\n"
            "        },\n"
            "    }"
        ),
        "solution_notes": (
            "**`value` vs `values` is the #1 sweep-config typo.** "
            "Singular `value` = fixed (don't sweep). Plural "
            "`values` = discrete set (do sweep, sample uniformly). "
            "Get this wrong and wandb either runs your sweep over a "
            "single point (`values: 'adam'` parses as a 4-character "
            "list...) or treats your discrete set as a fixed string.\n\n"
            "**Why `log_uniform_values` not `log_uniform`.** wandb "
            "has two log-uniform distributions: `log_uniform` "
            "(min/max in log-space — confusing) and "
            "`log_uniform_values` (min/max in linear value space, "
            "log-uniformly sampled — what you actually want). The "
            "`_values` suffix means 'specify the bounds as actual "
            "values, not as logs'.\n\n"
            "**JSON-serializability is a hard requirement.** wandb "
            "ships the sweep config to its server; numpy floats, "
            "tensors, dataclasses, etc. all break it. Stick to "
            "ints, floats, strs, lists, dicts."
        ),
        "extra_imports": [],
    },

    # =========================================================
    # sweep-hparam-distribution — ex1
    # =========================================================
    {
        "atom_id": "sweep-hparam-distribution",
        "subtopic": "Config: sweep hparam distribution",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_SWEEP_DISTRIBUTION,
        "exercise_index": 1,
        "exercise_title": "pick the right wandb sweep distribution for each hparam type",
        "slug": "pick-the-right-wandb-sweep-distribution-for-each-hparam-type",
        "bloom_level": "Analyze",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["distributions", "log-uniform", "categorical", "int-uniform"],
        "kcs": [
            "log-uniform-for-orders-of-magnitude-hparams",
            "categorical-vs-values-for-strings",
        ],
        "lo": (
            "Analyze hparam types (LR, dropout, num_layers, "
            "activation) and select the appropriate wandb sweep "
            "distribution specifier for each."
        ),
        "prompt_body": (
            "Implement `ex1_distribution_spec(name)`. Given an "
            "hparam name, return the correct wandb-sweep "
            "distribution spec dict.\n\n"
            "Mapping required:\n"
            "\n"
            "| name | distribution | spec dict |\n"
            "|---|---|---|\n"
            "| `'lr'` | log-uniform value-space | `{'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-1}` |\n"
            "| `'weight_decay'` | log-uniform value-space | `{'distribution': 'log_uniform_values', 'min': 1e-6, 'max': 1e-2}` |\n"
            "| `'dropout'` | linear uniform on `[0.0, 0.5]` | `{'distribution': 'uniform', 'min': 0.0, 'max': 0.5}` |\n"
            "| `'num_layers'` | integer uniform on `[2, 12]` | `{'distribution': 'int_uniform', 'min': 2, 'max': 12}` |\n"
            "| `'activation'` | categorical from "
            "`['relu', 'gelu', 'silu']` | `{'distribution': 'categorical', 'values': ['relu', 'gelu', 'silu']}` |\n"
            "| `'optimizer'` | discrete (small set, no need for "
            "'categorical') | `{'values': ['sgd', 'adam', 'adamw']}` |\n"
            "\n"
            "Any other name raises `KeyError`.\n\n"
            "The reasoning isn't tested directly — the test checks "
            "you produced exactly the right spec dict for each "
            "input. The mapping above IS the analysis: which "
            "distribution fits which kind of hparam.\n\n"
            "Output: `dict`."
        ),
        "stub": (
            "def ex1_distribution_spec(name: str) -> dict:\n"
            '    """Return the wandb sweep distribution spec for the given hparam name."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# === log_uniform_values for LR ===\n"
            "spec = ex1_distribution_spec('lr')\n"
            "assert spec == {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-1}, (\n"
            "    f'lr spec wrong: {spec}; LR spans orders of magnitude → use log_uniform_values'\n"
            ")\n"
            "\n"
            "# === log_uniform_values for weight_decay (also spans orders of magnitude) ===\n"
            "spec = ex1_distribution_spec('weight_decay')\n"
            "assert spec == {'distribution': 'log_uniform_values', 'min': 1e-6, 'max': 1e-2}, (\n"
            "    f'weight_decay spec wrong: {spec}'\n"
            ")\n"
            "\n"
            "# === linear uniform for dropout (bounded to [0, 0.5], linear scale fine) ===\n"
            "spec = ex1_distribution_spec('dropout')\n"
            "assert spec == {'distribution': 'uniform', 'min': 0.0, 'max': 0.5}, (\n"
            "    f'dropout spec wrong: {spec}; bounded linear-scale param → uniform'\n"
            ")\n"
            "\n"
            "# === int_uniform for num_layers (must be integer) ===\n"
            "spec = ex1_distribution_spec('num_layers')\n"
            "assert spec == {'distribution': 'int_uniform', 'min': 2, 'max': 12}, (\n"
            "    f'num_layers spec wrong: {spec}; integer-valued → int_uniform'\n"
            ")\n"
            "\n"
            "# === categorical for activation (unordered strings) ===\n"
            "spec = ex1_distribution_spec('activation')\n"
            "assert spec == {'distribution': 'categorical', 'values': ['relu', 'gelu', 'silu']}, (\n"
            "    f'activation spec wrong: {spec}; unordered string options → categorical'\n"
            ")\n"
            "\n"
            "# === discrete 'values' for optimizer (small set, no need for the categorical wrapper) ===\n"
            "spec = ex1_distribution_spec('optimizer')\n"
            "assert spec == {'values': ['sgd', 'adam', 'adamw']}, (\n"
            "    f'optimizer spec wrong: {spec}'\n"
            ")\n"
            "\n"
            "# === Unknown name => KeyError ===\n"
            "try:\n"
            "    ex1_distribution_spec('mystery_param')\n"
            "except KeyError as e:\n"
            "    assert 'mystery_param' in str(e), f'KeyError should mention the bad name, got {e!r}'\n"
            "else:\n"
            "    raise AssertionError('expected KeyError for unknown hparam name')\n"
            "\n"
            "# === Same call returns equal dicts (no shared mutable state) ===\n"
            "s1 = ex1_distribution_spec('lr')\n"
            "s2 = ex1_distribution_spec('lr')\n"
            "s1['min'] = 999  # mutate one\n"
            "assert s2['min'] == 1e-5, 'returned specs must not share mutable state'\n"
            "\n"
            "# === All specs are JSON-serializable ===\n"
            "import json\n"
            "for name in ['lr', 'weight_decay', 'dropout', 'num_layers', 'activation', 'optimizer']:\n"
            "    _ = json.dumps(ex1_distribution_spec(name))"
        ),
        "solution_body": (
            "def ex1_distribution_spec(name):\n"
            "    table = {\n"
            "        'lr':           {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-1},\n"
            "        'weight_decay': {'distribution': 'log_uniform_values', 'min': 1e-6, 'max': 1e-2},\n"
            "        'dropout':      {'distribution': 'uniform',            'min': 0.0,  'max': 0.5},\n"
            "        'num_layers':   {'distribution': 'int_uniform',        'min': 2,    'max': 12},\n"
            "        'activation':   {'distribution': 'categorical', 'values': ['relu', 'gelu', 'silu']},\n"
            "        'optimizer':    {'values': ['sgd', 'adam', 'adamw']},\n"
            "    }\n"
            "    if name not in table:\n"
            "        raise KeyError(name)\n"
            "    # Return a copy so callers can mutate without affecting the table.\n"
            "    spec = dict(table[name])\n"
            "    if 'values' in spec and isinstance(spec['values'], list):\n"
            "        spec['values'] = list(spec['values'])\n"
            "    return spec"
        ),
        "solution_notes": (
            "**Why LR is log-uniform and dropout is linear.** LR "
            "spans 4-5 orders of magnitude in viable ranges (`1e-5` "
            "to `1e-1`). Sampling uniformly on the LINEAR scale "
            "would waste ~99% of trials on `[1e-2, 1e-1]`. Dropout "
            "is bounded to `[0, 0.5]` and varies linearly in its "
            "effect — linear uniform is correct.\n\n"
            "**Why `int_uniform` for num_layers.** wandb's "
            "`int_uniform` rounds at sample time. If you used "
            "`uniform` and then `int(...)`-cast in your training "
            "script, you'd bias toward the lower bound (a sample "
            "of `4.7` becomes `4`, not `5`, by `int()`'s truncation "
            "rule).\n\n"
            "**`categorical` vs `values`.** Plain `{'values': [...]}` "
            "samples uniformly from the list. `{'distribution': "
            "'categorical', 'values': [...]}` is equivalent for "
            "equal weights but lets you add a `probabilities` "
            "key for non-uniform priors (e.g. weight adam at 0.7, "
            "sgd at 0.2, adamw at 0.1). For the activation "
            "function we want the option of biasing the prior; "
            "for the optimizer the plain form is fine."
        ),
        "extra_imports": [],
    },

    # =========================================================
    # nested-param-group-loop — ex1
    # =========================================================
    {
        "atom_id": "nested-param-group-loop",
        "subtopic": "Config: nested param-group loop",
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP_NESTED_PARAM_GROUP_LOOP,
        "exercise_index": 1,
        "exercise_title": "manual SGD step via the nested param_groups loop",
        "slug": "manual-sgd-step-via-the-nested-param-groups-loop",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["param-groups", "sgd-step", "manual-optimizer", "in-place"],
        "kcs": [
            "outer-loop-on-param-groups",
            "inner-loop-skip-grad-none",
        ],
        "lo": (
            "Apply the canonical "
            "`for group in optimizer.param_groups: for p in "
            "group['params']: ...` nested loop to manually perform "
            "an SGD step that honors per-group learning rates and "
            "skips `grad=None` params."
        ),
        "prompt_body": (
            "Implement `ex1_manual_sgd_step(optimizer)`. The "
            "no-momentum SGD step, hand-written, using the "
            "optimizer's `param_groups` structure.\n\n"
            "Algorithm:\n"
            "```\n"
            "for group in optimizer.param_groups:\n"
            "    lr = group['lr']\n"
            "    for p in group['params']:\n"
            "        if p.grad is None:\n"
            "            continue\n"
            "        p.data.add_(p.grad, alpha=-lr)\n"
            "```\n"
            "\n"
            "1. Outer loop iterates over `optimizer.param_groups` "
            "(a `list[dict]`).\n"
            "2. Read `lr = group['lr']` ONCE per group.\n"
            "3. Inner loop iterates over `group['params']`.\n"
            "4. If `p.grad is None`, skip — don't try to read it.\n"
            "5. Otherwise update in place: `p.data.add_(p.grad, "
            "alpha=-lr)`.\n"
            "\n"
            "Do NOT call `optimizer.step()`. You're reimplementing "
            "step from scratch.\n\n"
            "Output: `None` (in-place mutation of params)."
        ),
        "stub": (
            "def ex1_manual_sgd_step(optimizer):\n"
            '    """Manually perform one SGD step using optimizer.param_groups."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "# === Single group, simple step ===\n"
            "p = t.nn.Parameter(t.tensor([1.0, 2.0, 3.0]))\n"
            "opt = t.optim.SGD([p], lr=0.1)\n"
            "p.grad = t.tensor([10.0, 20.0, 30.0])\n"
            "ex1_manual_sgd_step(opt)\n"
            "expected = t.tensor([1.0, 2.0, 3.0]) - 0.1 * t.tensor([10.0, 20.0, 30.0])\n"
            "assert t.allclose(p.detach(), expected, atol=1e-6), (\n"
            "    f'single-group step wrong: got {p.detach()}, expected {expected}'\n"
            ")\n"
            "\n"
            "# === Two groups with DIFFERENT lr ===\n"
            "p1 = t.nn.Parameter(t.zeros(3))\n"
            "p2 = t.nn.Parameter(t.zeros(3))\n"
            "opt = t.optim.SGD([\n"
            "    {'params': [p1], 'lr': 0.01},  # slow group\n"
            "    {'params': [p2], 'lr': 1.0},   # fast group\n"
            "])\n"
            "p1.grad = t.ones(3)\n"
            "p2.grad = t.ones(3)\n"
            "ex1_manual_sgd_step(opt)\n"
            "# p1 moved by 0.01 * 1 = 0.01\n"
            "# p2 moved by 1.0  * 1 = 1.0\n"
            "assert t.allclose(p1.detach(), -0.01 * t.ones(3)), f'slow group wrong: {p1.detach()}'\n"
            "assert t.allclose(p2.detach(), -1.0  * t.ones(3)), f'fast group wrong: {p2.detach()}'\n"
            "\n"
            "# === grad=None must be SKIPPED, not crash ===\n"
            "p_a = t.nn.Parameter(t.tensor([5.0]))\n"
            "p_b = t.nn.Parameter(t.tensor([5.0]))\n"
            "opt = t.optim.SGD([p_a, p_b], lr=0.1)\n"
            "p_a.grad = t.tensor([1.0])\n"
            "# p_b.grad intentionally left as None\n"
            "ex1_manual_sgd_step(opt)\n"
            "assert t.allclose(p_a.detach(), t.tensor([4.9])), 'p_a should have stepped'\n"
            "assert t.allclose(p_b.detach(), t.tensor([5.0])), (\n"
            "    f'p_b had grad=None; should not have moved, got {p_b.detach()}'\n"
            ")\n"
            "\n"
            "# === In-place: id and storage preserved ===\n"
            "p = t.nn.Parameter(t.zeros(4))\n"
            "orig_id = id(p)\n"
            "orig_ptr = p.data_ptr()\n"
            "opt = t.optim.SGD([p], lr=0.1)\n"
            "p.grad = t.ones(4)\n"
            "ex1_manual_sgd_step(opt)\n"
            "assert id(p) == orig_id, 'param object was rebound'\n"
            "assert p.data_ptr() == orig_ptr, 'param storage reallocated'\n"
            "\n"
            "# === Verify the loop reads lr from the GROUP, not from a constructor capture ===\n"
            "# Mutate the group's lr between construction and step.\n"
            "p = t.nn.Parameter(t.zeros(3))\n"
            "opt = t.optim.SGD([p], lr=0.1)\n"
            "opt.param_groups[0]['lr'] = 10.0  # caller bumped lr (e.g. LR schedule)\n"
            "p.grad = t.ones(3)\n"
            "ex1_manual_sgd_step(opt)\n"
            "assert t.allclose(p.detach(), -10.0 * t.ones(3)), (\n"
            "    f'lr should be read from group dict at step time; got {p.detach()}, expected -10'\n"
            ")\n"
            "\n"
            "# === Multi-param group: every param in the group gets the group's lr ===\n"
            "pa = t.nn.Parameter(t.zeros(2))\n"
            "pb = t.nn.Parameter(t.zeros(3))\n"
            "opt = t.optim.SGD([{'params': [pa, pb], 'lr': 0.5}])\n"
            "pa.grad = t.ones(2)\n"
            "pb.grad = t.ones(3) * 2\n"
            "ex1_manual_sgd_step(opt)\n"
            "assert t.allclose(pa.detach(), -0.5 * t.ones(2)), f'pa wrong: {pa.detach()}'\n"
            "assert t.allclose(pb.detach(), -1.0 * t.ones(3)), f'pb wrong: {pb.detach()}'\n"
            "\n"
            "# === Function returns None (sentinel for 'in-place side effect') ===\n"
            "p = t.nn.Parameter(t.zeros(2))\n"
            "opt = t.optim.SGD([p], lr=0.1)\n"
            "p.grad = t.ones(2)\n"
            "ret = ex1_manual_sgd_step(opt)\n"
            "assert ret is None, f'should return None (in-place), got {ret!r}'"
        ),
        "solution_body": (
            "def ex1_manual_sgd_step(optimizer):\n"
            "    for group in optimizer.param_groups:\n"
            "        lr = group['lr']\n"
            "        for p in group['params']:\n"
            "            if p.grad is None:\n"
            "                continue\n"
            "            p.data.add_(p.grad, alpha=-lr)"
        ),
        "solution_notes": (
            "**Why `p.data.add_(p.grad, alpha=-lr)` and not "
            "`p -= lr * p.grad`.** Both work, but the fused form "
            "is what real PyTorch optim source uses: it avoids the "
            "intermediate `lr * p.grad` allocation. `add_(x, "
            "alpha=k)` computes `self += k * x` in one CUDA kernel.\n\n"
            "**Why read `lr` once per group, not once per param.** "
            "If you write `for p in group['params']: lr = "
            "group['lr']; ...`, you're paying the dict lookup "
            "per-param. For a 100-param model with a 10-step inner "
            "loop, that's 1000 lookups instead of 10. Microscopic, "
            "but the convention is 'hoist what's group-scoped to "
            "the outer loop'.\n\n"
            "**`p.data` vs `p` for the in-place mutation.** Inside "
            "an optimizer step you usually wrap with "
            "`torch.no_grad()` so autograd doesn't track the "
            "mutation. PyTorch's optim source uses `p.data.add_` "
            "as belt-and-suspenders — it strips the autograd "
            "tracking even without the context manager. Either is "
            "acceptable in your own code as long as you're "
            "consistent."
        ),
        "extra_imports": [],
    },
]


# ---------------------------------------------------------------------------
# Verifier — exec every (solution + test_body) in a fresh namespace.
# Aborts the whole build if any spec fails.
# ---------------------------------------------------------------------------

def _verify_all(specs):
    import torch as t
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
    print(f"[batch5] Verifying {len(SPECS)} specs against torch backend...")
    _verify_all(SPECS)

    print(f"\n[batch5] All verified — emitting notebooks.")
    for spec in SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        print(f"  wrote {rel}")
    print(f"\n[batch5] {len(SPECS)} notebooks emitted.")


if __name__ == "__main__":
    main()
