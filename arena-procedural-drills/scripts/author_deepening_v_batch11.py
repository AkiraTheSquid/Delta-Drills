#!/usr/bin/env python3
"""Author 8 ex2 deepening drills (batch 11).

Atoms (2 hparam-config + 4 logging-instr + 2 misc-cleanup):
    - sweep-config-dict             (ex2: validate a sweep config + flag schema errors)
    - sweep-hparam-distribution     (ex2: convert a (low, high, kind) tuple registry to distribution specs)
    - log-samples-eval-callback     (ex2: callback respects start_step offset + skips when sink is full)
    - time-stage-instrumentation    (ex2: per-stage context-manager accumulator + nested-call safety)
    - wandb-config-into-args        (ex2: precedence merge — sweep config beats defaults but not explicit CLI)
    - wandb-watch-model             (ex2: watch only modules with trainable params + return qnames)
    - any-reduce-axis               (ex2: per-batch ANY along arbitrary dim with keepdim broadcast back)
    - functional-module-wrap        (ex2: parametric wrap of F.leaky_relu with negative_slope + inplace flag)

Each ex2 hits a DISTINCT facet from ex1. ONE LO + ONE Bloom + <=2 KCs per drill.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

TOPIC_HPARAM = "prereqs_hparam_config"
TOPIC_LOG = "prereqs_logging_instr"
TOPIC_MISC = "prereqs_misc_cleanup"


# ---------------------------------------------------------------------------
# Recap blocks
# ---------------------------------------------------------------------------

RECAP_SWEEP_VALIDATE = (
    "## Sweep config dict — schema validation\n"
    "\n"
    "Ex1 BUILT a sweep config. The deepening move is to VALIDATE one — given an "
    "arbitrary user-supplied dict, return the list of schema errors. This is "
    "what `wandb.sweep(cfg)` does internally before posting to the server.\n"
    "\n"
    "**Required keys:** `'method'` (one of `'grid'`, `'random'`, `'bayes'`) and "
    "`'parameters'` (a non-empty dict-of-dicts).\n"
    "\n"
    "**Conditional requirement:** if `method == 'bayes'`, `'metric'` MUST be "
    "present and contain `'name'` (str) + `'goal'` (`'minimize'` | `'maximize'`).\n"
    "\n"
    "**Per-parameter shape:** every value in `parameters` must be a dict with "
    "EXACTLY ONE of: `'value'`, `'values'`, or `'distribution'` keys. More than "
    "one specifier is the typo trap (`{'value': 'adam', 'values': ['sgd']}`).\n"
    "\n"
    "The validator returns a `list[str]` — empty means valid. Listing errors "
    "instead of raising on the first one lets the user fix all problems at once."
)

RECAP_DIST_REGISTRY = (
    "## Distribution registry — (low, high, kind) tuple to wandb spec\n"
    "\n"
    "Ex1 hand-mapped hparam name → spec. The deepening move is a generic "
    "registry: given `(low, high, kind)`, emit the right distribution spec.\n"
    "\n"
    "```python\n"
    "spec = build_spec(low=1e-5, high=1e-1, kind='log_float')\n"
    "# → {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-1}\n"
    "```\n"
    "\n"
    "**Kinds:** `'float'` → `'uniform'`; `'log_float'` → `'log_uniform_values'`; "
    "`'int'` → `'int_uniform'`. Unknown kind raises `ValueError`.\n"
    "\n"
    "**Why a registry over named-hparam dispatch.** When you sweep 30 hparams "
    "across 3 projects, hand-listing each name is fragile. A `(low, high, kind)` "
    "tuple keeps the SHAPE separate from the NAME — projects share kind logic.\n"
    "\n"
    "**Validation: low < high.** A flipped range is the most common typo. "
    "Detect it up front; downstream wandb won't give a helpful error."
)

RECAP_CALLBACK_OFFSET = (
    "## Eval callback with start_step offset + capacity cap\n"
    "\n"
    "Ex1 fired at `step % K == 0`. Two real-world extensions:\n"
    "\n"
    "1. **`start_step` offset** — skip eval during the warm-up phase. Fire only "
    "when `step >= start_step AND (step - start_step) % K == 0`. Lets you skip "
    "the noisy first 1000 steps without changing K.\n"
    "2. **Capacity cap** — the sink has a `max_entries`. Once full, no more "
    "appends (downstream storage limit, wandb table cap, etc.).\n"
    "\n"
    "```python\n"
    "for step in range(n_steps):\n"
    "    if step >= start and (step - start) % K == 0 and len(sink) < cap:\n"
    "        sink.append({'step': step, 'samples': sample()})\n"
    "```\n"
    "\n"
    "**Why aligned to `start_step`, not absolute step.** A warm-up of 100 steps "
    "with K=50 should fire at 100, 150, 200, ... — not at 100, 150, 200 only "
    "because they happen to be K-multiples. The offset shifts the cadence origin."
)

RECAP_STAGE_CTX = (
    "## Per-stage context manager + nested re-entry\n"
    "\n"
    "Ex1 used inline `t0 = perf_counter()` ... `acc += perf_counter() - t0`. The "
    "deepening move is a context-manager helper:\n"
    "\n"
    "```python\n"
    "@contextmanager\n"
    "def stage(name, acc):\n"
    "    t0 = time.perf_counter()\n"
    "    try:\n"
    "        yield\n"
    "    finally:\n"
    "        acc[name] = acc.get(name, 0.0) + (time.perf_counter() - t0)\n"
    "```\n"
    "\n"
    "**Why `finally`.** A raised exception inside the block still records the "
    "partial elapsed time. Without `finally`, an OOM mid-forward leaves the "
    "accumulator silently missing that iteration's contribution.\n"
    "\n"
    "**Re-entrant safety.** Calling `stage('forward', acc)` twice in nested "
    "scopes accumulates BOTH elapsed times into the same key — total is the "
    "sum of (outer block + inner block). For non-overlapping stages this is "
    "what you want; for overlapping stages you need per-frame keys."
)

RECAP_CONFIG_PRECEDENCE = (
    "## wandb.config + CLI overrides — three-tier precedence\n"
    "\n"
    "Ex1 overwrote args from `wandb.config` unconditionally. Real ARENA sweep "
    "scripts have a THIRD tier: explicit CLI flags that should win over the "
    "sweep agent (debug-rerun a sweep config with a hand-tweaked lr).\n"
    "\n"
    "Precedence (lowest → highest):\n"
    "```\n"
    "defaults  <  wandb.config (sweep)  <  cli_overrides (explicit)\n"
    "```\n"
    "\n"
    "**Same `setattr` + `hasattr` skeleton.** Apply each layer in order. The "
    "later layer wins on shared keys. Unknown keys (e.g. wandb metadata) "
    "still get silently skipped per ex1's contract.\n"
    "\n"
    "**Why CLI beats sweep.** A sweep is automated exploration; a CLI override "
    "is a deliberate human intervention. Production sweep harnesses (Hydra, "
    "MMCV) all follow the same ordering."
)

RECAP_WATCH_TRAINABLE = (
    "## `wandb.watch` only the trainable submodules\n"
    "\n"
    "Ex1 watched a fixed `model.out_layers[-1]`. The deepening move: "
    "AUTO-DETECT which submodules to watch by checking "
    "`any(p.requires_grad for p in m.parameters(recurse=False))`. This way a "
    "fine-tune script that freezes/unfreezes layers at runtime still watches "
    "the right thing without code edits.\n"
    "\n"
    "```python\n"
    "watched = []\n"
    "for qname, m in model.named_modules():\n"
    "    if qname == '':\n"
    "        continue\n"
    "    own = list(m.parameters(recurse=False))\n"
    "    if own and any(p.requires_grad for p in own):\n"
    "        wandb.watch(m, log='all', log_freq=K)\n"
    "        watched.append(qname)\n"
    "```\n"
    "\n"
    "**`recurse=False` is load-bearing.** Without it, every ancestor of a "
    "trainable leaf shows up as 'has a trainable param' and you double-hook. "
    "`recurse=False` gives only the params OWNED by the module itself.\n"
    "\n"
    "**Skip the root.** `named_modules()` yields `('', model)` first — same "
    "filter as the named-modules report pattern."
)

RECAP_ANY_KEEPDIM = (
    "## `.any(dim=k, keepdim=True)` + broadcast back\n"
    "\n"
    "Ex1 reduced to a flat `(N,)` mask. The deepening move keeps the rank for "
    "broadcasting:\n"
    "\n"
    "```python\n"
    "mask = (x > 0)                              # (N, M)\n"
    "row_has_pos = mask.any(dim=1, keepdim=True) # (N, 1)\n"
    "x_blanked = t.where(row_has_pos, x, t.zeros_like(x))\n"
    "```\n"
    "\n"
    "**Why `keepdim=True`.** Lets you broadcast the per-row decision back "
    "against the full `(N, M)` tensor without manual `unsqueeze`. The "
    "broadcasting machinery just sees `(N, 1)` and replicates.\n"
    "\n"
    "**Arbitrary-axis variant.** `.any(dim=k, keepdim=True)` works for any "
    "rank — collapses axis `k` to size 1. ARENA's image masks (`(B, C, H, W)`) "
    "use this to flag, e.g., 'this image has any non-zero pixel' at "
    "`dim=(1, 2, 3)` reduce, kept at `(B, 1, 1, 1)` for broadcast."
)

RECAP_FUNC_PARAMETRIC = (
    "## Parametric functional wrap — `F.leaky_relu(negative_slope, inplace)`\n"
    "\n"
    "Ex1 wrapped the parameter-free `F.relu`. The deepening move handles a "
    "PARAMETRIC functional — `F.leaky_relu(x, negative_slope=0.01, "
    "inplace=False)`. The Module must STORE the parameters (not the tensor "
    "weights — these aren't learnable, they're hparams) and pass them through:\n"
    "\n"
    "```python\n"
    "class MyLeakyReLU(nn.Module):\n"
    "    def __init__(self, negative_slope=0.01, inplace=False):\n"
    "        super().__init__()\n"
    "        self.negative_slope = negative_slope\n"
    "        self.inplace = inplace\n"
    "\n"
    "    def forward(self, x):\n"
    "        return F.leaky_relu(x, self.negative_slope, self.inplace)\n"
    "\n"
    "    def extra_repr(self):\n"
    "        return f'negative_slope={self.negative_slope}, inplace={self.inplace}'\n"
    "```\n"
    "\n"
    "**`extra_repr` over `__repr__`.** `nn.Module.__repr__` already handles "
    "the class name + children; `extra_repr` slots your hparams into that "
    "format. This is how `nn.LeakyReLU(negative_slope=0.2)` shows up in "
    "`print(model)`.\n"
    "\n"
    "**Still no parameters.** Hparams are stored as plain Python attrs, not "
    "`nn.Parameter`. `model.parameters()` stays empty — these are config."
)


# ---------------------------------------------------------------------------
# SPEC 1 — sweep-config-dict ex2
# ---------------------------------------------------------------------------

SPEC_SWEEP_VALIDATE = {
    "atom_id": "sweep-config-dict",
    "subtopic": "Config: wandb sweep config dict",
    "topic_folder": TOPIC_HPARAM,
    "atom_recap_md": RECAP_SWEEP_VALIDATE,
    "exercise_index": 2,
    "exercise_title": "validate a wandb sweep config dict and report all schema errors",
    "slug": "validate-wandb-sweep-config-and-report-errors",
    "bloom_level": "Analyze",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["wandb", "sweep", "validation", "schema"],
    "kcs": [
        "sweep-config-top-level-schema",
        "per-parameter-specifier-exactly-one",
    ],
    "lo": (
        "Analyze an arbitrary user-supplied dict against the wandb sweep "
        "schema and return the full list of error messages — including "
        "missing required keys, bad method values, missing bayes metric, "
        "and per-parameter specifier mistakes."
    ),
    "prompt_body": (
        "Implement `ex2_validate_sweep_config(cfg)`. Return a `list[str]` of "
        "schema error messages. Empty list = valid.\n\n"
        "Checks (in this order, but collect ALL failures, do not short-"
        "circuit):\n\n"
        "1. `'method'` key present AND in `{'grid', 'random', 'bayes'}`. "
        "If missing: append `'missing required key: method'`. If wrong "
        "value: append `'method must be one of grid/random/bayes, got "
        "<value>'`.\n"
        "2. If `method == 'bayes'`:\n"
        "   - `'metric'` key must be present. If missing: append "
        "`'bayes method requires metric block'`.\n"
        "   - If present, `metric` must be a dict containing `'name'` "
        "(str) AND `'goal'` in `{'minimize', 'maximize'}`. Append "
        "`'metric missing name'` and/or `'metric goal must be minimize "
        "or maximize'` as applicable.\n"
        "3. `'parameters'` key present AND a non-empty dict. Missing → "
        "`'missing required key: parameters'`; empty → `'parameters dict "
        "is empty'`.\n"
        "4. For each `(pname, pspec)` in `parameters`:\n"
        "   - `pspec` must be a dict. If not → append `'parameters.<pname>: "
        "spec must be a dict'`.\n"
        "   - Count how many of `'value'`, `'values'`, `'distribution'` "
        "keys appear. Must be EXACTLY 1. If 0 → append "
        "`'parameters.<pname>: needs one of value/values/distribution'`. "
        "If >1 → append `'parameters.<pname>: only one of "
        "value/values/distribution allowed'`.\n\n"
        "Input: `cfg` — arbitrary dict.\n"
        "Output: `list[str]` of error messages (any order acceptable; tests "
        "check `set` equality)."
    ),
    "stub": (
        "def ex2_validate_sweep_config(cfg: dict) -> list:\n"
        '    """Return all schema errors as a list of strings (empty = valid)."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# === Valid config → no errors ===\n"
        "good = {\n"
        "    'method': 'bayes',\n"
        "    'metric': {'name': 'val/loss', 'goal': 'minimize'},\n"
        "    'parameters': {\n"
        "        'lr': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-1},\n"
        "        'batch_size': {'values': [16, 32, 64]},\n"
        "        'optimizer': {'value': 'adam'},\n"
        "    },\n"
        "}\n"
        "errs = ex2_validate_sweep_config(good)\n"
        "assert errs == [], f'valid config should have no errors, got {errs}'\n"
        "\n"
        "# === Missing method ===\n"
        "bad = {'parameters': {'lr': {'value': 1e-3}}}\n"
        "errs = ex2_validate_sweep_config(bad)\n"
        "assert any('missing required key: method' in e for e in errs), f'expected missing-method error, got {errs}'\n"
        "\n"
        "# === Bad method value ===\n"
        "bad = {'method': 'genetic', 'parameters': {'lr': {'value': 1e-3}}}\n"
        "errs = ex2_validate_sweep_config(bad)\n"
        "assert any('method must be one of' in e and 'genetic' in e for e in errs), f'expected bad-method error, got {errs}'\n"
        "\n"
        "# === Bayes missing metric ===\n"
        "bad = {'method': 'bayes', 'parameters': {'lr': {'value': 1e-3}}}\n"
        "errs = ex2_validate_sweep_config(bad)\n"
        "assert any('bayes method requires metric block' in e for e in errs), f'expected bayes-metric error, got {errs}'\n"
        "\n"
        "# === Bayes with bad goal ===\n"
        "bad = {\n"
        "    'method': 'bayes',\n"
        "    'metric': {'name': 'loss', 'goal': 'reduce'},\n"
        "    'parameters': {'lr': {'value': 1e-3}},\n"
        "}\n"
        "errs = ex2_validate_sweep_config(bad)\n"
        "assert any('metric goal must be' in e for e in errs), f'expected bad-goal error, got {errs}'\n"
        "\n"
        "# === Bayes metric missing name ===\n"
        "bad = {\n"
        "    'method': 'bayes',\n"
        "    'metric': {'goal': 'minimize'},\n"
        "    'parameters': {'lr': {'value': 1e-3}},\n"
        "}\n"
        "errs = ex2_validate_sweep_config(bad)\n"
        "assert any('metric missing name' in e for e in errs), f'expected metric-name error, got {errs}'\n"
        "\n"
        "# === Missing parameters ===\n"
        "bad = {'method': 'random'}\n"
        "errs = ex2_validate_sweep_config(bad)\n"
        "assert any('missing required key: parameters' in e for e in errs), f'expected missing-parameters error, got {errs}'\n"
        "\n"
        "# === Empty parameters ===\n"
        "bad = {'method': 'random', 'parameters': {}}\n"
        "errs = ex2_validate_sweep_config(bad)\n"
        "assert any('parameters dict is empty' in e for e in errs), f'expected empty-parameters error, got {errs}'\n"
        "\n"
        "# === Per-param: zero specifiers ===\n"
        "bad = {'method': 'random', 'parameters': {'lr': {}}}\n"
        "errs = ex2_validate_sweep_config(bad)\n"
        "assert any('parameters.lr: needs one of' in e for e in errs), f'expected zero-spec error, got {errs}'\n"
        "\n"
        "# === Per-param: more than one specifier ===\n"
        "bad = {\n"
        "    'method': 'random',\n"
        "    'parameters': {'lr': {'value': 1e-3, 'values': [1e-3, 1e-4]}},\n"
        "}\n"
        "errs = ex2_validate_sweep_config(bad)\n"
        "assert any('parameters.lr: only one of' in e for e in errs), f'expected multi-spec error, got {errs}'\n"
        "\n"
        "# === Per-param: spec is not a dict ===\n"
        "bad = {'method': 'random', 'parameters': {'lr': 1e-3}}\n"
        "errs = ex2_validate_sweep_config(bad)\n"
        "assert any('parameters.lr: spec must be a dict' in e for e in errs), f'expected non-dict-spec error, got {errs}'\n"
        "\n"
        "# === Multiple errors collected, not short-circuited ===\n"
        "bad = {\n"
        "    'method': 'jelly',\n"
        "    'parameters': {\n"
        "        'a': {},\n"
        "        'b': {'value': 1, 'values': [1, 2]},\n"
        "    },\n"
        "}\n"
        "errs = ex2_validate_sweep_config(bad)\n"
        "assert len(errs) >= 3, f'expected at least 3 errors (method + a + b), got {len(errs)}: {errs}'\n"
        "assert any('jelly' in e for e in errs)\n"
        "assert any('parameters.a:' in e for e in errs)\n"
        "assert any('parameters.b:' in e for e in errs)\n"
        "\n"
        "# === Returns a list, not a generator or set ===\n"
        "assert isinstance(ex2_validate_sweep_config(good), list)\n"
        "assert isinstance(ex2_validate_sweep_config(bad), list)"
    ),
    "solution_body": (
        "def ex2_validate_sweep_config(cfg):\n"
        "    errs = []\n"
        "    # method\n"
        "    if 'method' not in cfg:\n"
        "        errs.append('missing required key: method')\n"
        "    else:\n"
        "        if cfg['method'] not in {'grid', 'random', 'bayes'}:\n"
        "            errs.append(\n"
        "                f\"method must be one of grid/random/bayes, got {cfg['method']!r}\"\n"
        "            )\n"
        "    # bayes → metric\n"
        "    if cfg.get('method') == 'bayes':\n"
        "        if 'metric' not in cfg:\n"
        "            errs.append('bayes method requires metric block')\n"
        "        else:\n"
        "            m = cfg['metric']\n"
        "            if not isinstance(m, dict) or 'name' not in m or not isinstance(m.get('name'), str):\n"
        "                errs.append('metric missing name')\n"
        "            if not isinstance(m, dict) or m.get('goal') not in {'minimize', 'maximize'}:\n"
        "                errs.append('metric goal must be minimize or maximize')\n"
        "    # parameters\n"
        "    if 'parameters' not in cfg:\n"
        "        errs.append('missing required key: parameters')\n"
        "    else:\n"
        "        p = cfg['parameters']\n"
        "        if not isinstance(p, dict):\n"
        "            errs.append('parameters must be a dict')\n"
        "        elif len(p) == 0:\n"
        "            errs.append('parameters dict is empty')\n"
        "        else:\n"
        "            for pname, pspec in p.items():\n"
        "                if not isinstance(pspec, dict):\n"
        "                    errs.append(f'parameters.{pname}: spec must be a dict')\n"
        "                    continue\n"
        "                n = sum(k in pspec for k in ('value', 'values', 'distribution'))\n"
        "                if n == 0:\n"
        "                    errs.append(f'parameters.{pname}: needs one of value/values/distribution')\n"
        "                elif n > 1:\n"
        "                    errs.append(f'parameters.{pname}: only one of value/values/distribution allowed')\n"
        "    return errs"
    ),
    "solution_notes": (
        "**Collect all errors, never short-circuit.** Reporting one error at "
        "a time forces the user into N round-trips for N typos. A list of "
        "errors lets them fix everything in one pass — the same UX as a "
        "compiler's diagnostics buffer.\n\n"
        "**`sum(k in pspec for k in (...))` is the exactly-one count.** "
        "Booleans add as 0/1. Three keys → counts 0..3. The check `n == 1` "
        "is the canonical 'mutually exclusive' validator.\n\n"
        "**Why `pspec.get('goal') not in {...}` even after `isinstance`.** "
        "Defensive against `metric={'goal': None}` — `None not in {'minimize', "
        "'maximize'}` is True, so the error fires. Without the set check, "
        "`None == 'minimize'` would be False but no error would be raised."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 2 — sweep-hparam-distribution ex2
# ---------------------------------------------------------------------------

SPEC_DIST_REGISTRY = {
    "atom_id": "sweep-hparam-distribution",
    "subtopic": "Config: sweep hparam distribution",
    "topic_folder": TOPIC_HPARAM,
    "atom_recap_md": RECAP_DIST_REGISTRY,
    "exercise_index": 2,
    "exercise_title": "build sweep distribution specs from (low, high, kind) tuples",
    "slug": "build-sweep-distribution-specs-from-tuples",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["distribution", "registry", "tuple-dispatch", "validation"],
    "kcs": [
        "kind-to-distribution-dispatch",
        "range-validation-low-less-than-high",
    ],
    "lo": (
        "Apply a kind-dispatch table (`'float' | 'log_float' | 'int'`) to "
        "convert `(low, high, kind)` tuples into the correct wandb "
        "distribution spec dicts, validating that `low < high` and raising "
        "`ValueError` on unknown kinds."
    ),
    "prompt_body": (
        "Implement `ex2_build_distribution_spec(low, high, kind)`. A generic "
        "registry over the ex1 mapping.\n\n"
        "Kind → distribution:\n"
        "- `'float'` → `{'distribution': 'uniform', 'min': low, 'max': high}`\n"
        "- `'log_float'` → `{'distribution': 'log_uniform_values', 'min': low, 'max': high}`\n"
        "- `'int'` → `{'distribution': 'int_uniform', 'min': low, 'max': high}`\n\n"
        "Validation:\n"
        "1. `low < high` — strict. If violated: raise `ValueError` whose "
        "message contains both `'low'` and `'high'` (case-insensitive).\n"
        "2. `kind in {'float', 'log_float', 'int'}` — else raise `ValueError` "
        "whose message contains the offending kind.\n"
        "3. For `'log_float'` only: BOTH `low > 0` and `high > 0`. Otherwise "
        "log-uniform is undefined — raise `ValueError` containing `'log'`.\n"
        "4. For `'int'`: `low` and `high` must both be `int`. If either is a "
        "non-int (including bool), raise `TypeError` containing `'int'`.\n\n"
        "Output: `dict` matching wandb's distribution spec schema."
    ),
    "stub": (
        "def ex2_build_distribution_spec(low, high, kind: str) -> dict:\n"
        '    """Convert (low, high, kind) into the wandb distribution spec dict."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# === float kind ===\n"
        "spec = ex2_build_distribution_spec(0.0, 0.5, 'float')\n"
        "assert spec == {'distribution': 'uniform', 'min': 0.0, 'max': 0.5}, f'float spec wrong: {spec}'\n"
        "\n"
        "# === log_float kind ===\n"
        "spec = ex2_build_distribution_spec(1e-5, 1e-1, 'log_float')\n"
        "assert spec == {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-1}, f'log_float spec wrong: {spec}'\n"
        "\n"
        "# === int kind ===\n"
        "spec = ex2_build_distribution_spec(2, 12, 'int')\n"
        "assert spec == {'distribution': 'int_uniform', 'min': 2, 'max': 12}, f'int spec wrong: {spec}'\n"
        "\n"
        "# === Unknown kind → ValueError naming the kind ===\n"
        "try:\n"
        "    ex2_build_distribution_spec(0.0, 1.0, 'cauchy')\n"
        "except ValueError as e:\n"
        "    assert 'cauchy' in str(e), f'error must name the bad kind, got {e!r}'\n"
        "else:\n"
        "    raise AssertionError('expected ValueError for unknown kind')\n"
        "\n"
        "# === low >= high → ValueError mentioning both labels ===\n"
        "for low, high in [(1.0, 0.5), (5.0, 5.0)]:\n"
        "    try:\n"
        "        ex2_build_distribution_spec(low, high, 'float')\n"
        "    except ValueError as e:\n"
        "        msg = str(e).lower()\n"
        "        assert 'low' in msg and 'high' in msg, f'range error must mention low and high, got {e!r}'\n"
        "    else:\n"
        "        raise AssertionError(f'expected ValueError for low={low} high={high}')\n"
        "\n"
        "# === log_float with non-positive low → ValueError mentioning 'log' ===\n"
        "for low, high in [(0.0, 1.0), (-1e-3, 1.0)]:\n"
        "    try:\n"
        "        ex2_build_distribution_spec(low, high, 'log_float')\n"
        "    except ValueError as e:\n"
        "        assert 'log' in str(e).lower(), f'log_float positivity error must mention log, got {e!r}'\n"
        "    else:\n"
        "        raise AssertionError(f'expected ValueError for log_float with low={low}')\n"
        "\n"
        "# === int with float bounds → TypeError mentioning 'int' ===\n"
        "for low, high in [(2.0, 12), (2, 12.0), (1.5, 5.5)]:\n"
        "    try:\n"
        "        ex2_build_distribution_spec(low, high, 'int')\n"
        "    except TypeError as e:\n"
        "        assert 'int' in str(e).lower(), f'int type error must mention int, got {e!r}'\n"
        "    else:\n"
        "        raise AssertionError(f'expected TypeError for int with non-int bounds {low}, {high}')\n"
        "\n"
        "# === Result is always a fresh dict (mutation isolation) ===\n"
        "s1 = ex2_build_distribution_spec(0.0, 1.0, 'float')\n"
        "s2 = ex2_build_distribution_spec(0.0, 1.0, 'float')\n"
        "s1['min'] = 999\n"
        "assert s2['min'] == 0.0, 'each call must return a fresh dict'\n"
        "\n"
        "# === Each result is JSON-serializable ===\n"
        "import json\n"
        "json.dumps(ex2_build_distribution_spec(1e-6, 1e-2, 'log_float'))\n"
        "json.dumps(ex2_build_distribution_spec(0, 100, 'int'))\n"
        "json.dumps(ex2_build_distribution_spec(0.0, 1.0, 'float'))"
    ),
    "solution_body": (
        "def ex2_build_distribution_spec(low, high, kind):\n"
        "    if kind not in {'float', 'log_float', 'int'}:\n"
        "        raise ValueError(\n"
        "            f'unknown kind {kind!r}, must be one of float/log_float/int'\n"
        "        )\n"
        "    if kind == 'int':\n"
        "        # bool is a subclass of int; exclude it explicitly.\n"
        "        if not (type(low) is int and type(high) is int):\n"
        "            raise TypeError(\n"
        "                f'kind=int requires int bounds, got low={type(low).__name__} high={type(high).__name__}'\n"
        "            )\n"
        "    if not (low < high):\n"
        "        raise ValueError(f'low must be < high, got low={low} high={high}')\n"
        "    if kind == 'log_float' and (low <= 0 or high <= 0):\n"
        "        raise ValueError(\n"
        "            f'log distribution requires low>0 and high>0, got low={low} high={high}'\n"
        "        )\n"
        "    dist_name = {\n"
        "        'float': 'uniform',\n"
        "        'log_float': 'log_uniform_values',\n"
        "        'int': 'int_uniform',\n"
        "    }[kind]\n"
        "    return {'distribution': dist_name, 'min': low, 'max': high}"
    ),
    "solution_notes": (
        "**Validation order matters.** Check kind FIRST — otherwise an "
        "unknown kind with bad bounds raises the bound error instead of the "
        "more informative kind error.\n\n"
        "**`type(low) is int` not `isinstance(low, int)`.** `bool` is a "
        "subclass of `int` in Python — `isinstance(True, int) == True`. The "
        "stricter `type() is int` check rejects booleans, which is almost "
        "always what you want for a bounds value.\n\n"
        "**Why a fresh dict per call.** Returning a shared dict (e.g. from "
        "a cached table) breaks downstream code that mutates the spec — "
        "the next call gets the mutated version. Constructing a new dict "
        "literal sidesteps the issue without needing `copy.deepcopy`."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 3 — log-samples-eval-callback ex2
# ---------------------------------------------------------------------------

SPEC_CALLBACK_OFFSET = {
    "atom_id": "log-samples-eval-callback",
    "subtopic": "Logging: log-samples eval callback",
    "topic_folder": TOPIC_LOG,
    "atom_recap_md": RECAP_CALLBACK_OFFSET,
    "exercise_index": 2,
    "exercise_title": "eval callback with start_step warm-up offset and sink capacity cap",
    "slug": "eval-callback-with-start-step-and-capacity-cap",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["eval", "callback", "warmup", "capacity"],
    "kcs": [
        "offset-aligned-cadence",
        "capacity-bounded-sink",
    ],
    "lo": (
        "Apply an offset-aligned modulo-K cadence with a `max_entries` "
        "guard so an eval callback skips a warm-up window AND stops "
        "appending once the sink hits capacity."
    ),
    "prompt_body": (
        "Implement `ex2_run_with_offset_callback(n_steps, eval_every, "
        "start_step, n_eval, sink, max_entries)`. The deepening variant of "
        "ex1's callback.\n\n"
        "Behaviour:\n\n"
        "1. Loop `for step in range(n_steps)`.\n"
        "2. Fire iff:\n"
        "   - `step >= start_step` (warm-up gate), AND\n"
        "   - `(step - start_step) % eval_every == 0` (offset-aligned "
        "cadence), AND\n"
        "   - `len(sink) < max_entries` (capacity guard).\n"
        "3. When firing, append `{'step': step, 'samples': [f'step={step}-"
        "sample={i}' for i in range(n_eval)]}` to `sink`.\n"
        "4. Return the number of `sink.append` calls performed by THIS "
        "function (so a sink that was non-empty going in still counts only "
        "the fires this call made).\n\n"
        "Edge cases:\n"
        "- `start_step >= n_steps` → no fires.\n"
        "- `max_entries == 0` → no fires regardless of cadence.\n"
        "- A `sink` that's already at or above `max_entries` → no fires."
    ),
    "stub": (
        "def ex2_run_with_offset_callback(n_steps: int, eval_every: int,\n"
        "                                  start_step: int, n_eval: int,\n"
        "                                  sink: list, max_entries: int) -> int:\n"
        '    """Offset-aligned eval callback with a sink-capacity cap."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# === Basic offset: warm-up 5, K=3, 15 steps → fires at 5, 8, 11, 14 ===\n"
        "sink = []\n"
        "n = ex2_run_with_offset_callback(n_steps=15, eval_every=3, start_step=5, n_eval=2, sink=sink, max_entries=100)\n"
        "assert n == 4, f'expected 4 fires, got {n}'\n"
        "assert [e['step'] for e in sink] == [5, 8, 11, 14], f'fire steps wrong: {[e[\"step\"] for e in sink]}'\n"
        "for entry in sink:\n"
        "    assert len(entry['samples']) == 2\n"
        "    assert entry['samples'][0] == f\"step={entry['step']}-sample=0\"\n"
        "\n"
        "# === start_step == 0 reduces to plain modulo-K (matches ex1 semantics) ===\n"
        "sink = []\n"
        "n = ex2_run_with_offset_callback(n_steps=10, eval_every=3, start_step=0, n_eval=1, sink=sink, max_entries=100)\n"
        "assert [e['step'] for e in sink] == [0, 3, 6, 9], f'start=0 should match ex1; got {[e[\"step\"] for e in sink]}'\n"
        "assert n == 4\n"
        "\n"
        "# === Capacity cap stops further fires ===\n"
        "sink = []\n"
        "n = ex2_run_with_offset_callback(n_steps=100, eval_every=10, start_step=0, n_eval=1, sink=sink, max_entries=3)\n"
        "assert n == 3, f'capacity cap=3 should give exactly 3 fires, got {n}'\n"
        "assert len(sink) == 3\n"
        "assert [e['step'] for e in sink] == [0, 10, 20]\n"
        "\n"
        "# === max_entries=0 → no fires regardless of cadence ===\n"
        "sink = []\n"
        "n = ex2_run_with_offset_callback(n_steps=10, eval_every=1, start_step=0, n_eval=4, sink=sink, max_entries=0)\n"
        "assert n == 0 and sink == [], f'max_entries=0 must produce no fires; got n={n}, sink={sink}'\n"
        "\n"
        "# === start_step >= n_steps → no fires ===\n"
        "sink = []\n"
        "n = ex2_run_with_offset_callback(n_steps=5, eval_every=1, start_step=10, n_eval=1, sink=sink, max_entries=100)\n"
        "assert n == 0 and sink == []\n"
        "\n"
        "# === Pre-populated sink that's already at capacity → no further fires ===\n"
        "sink = [{'step': -1, 'samples': []}, {'step': -2, 'samples': []}, {'step': -3, 'samples': []}]\n"
        "n = ex2_run_with_offset_callback(n_steps=20, eval_every=2, start_step=0, n_eval=1, sink=sink, max_entries=3)\n"
        "assert n == 0, f'sink already at cap should yield no new fires; got n={n}'\n"
        "assert len(sink) == 3, f'sink length should stay at cap; got {len(sink)}'\n"
        "\n"
        "# === Partially-filled sink fills up to cap, then stops ===\n"
        "sink = [{'step': -1, 'samples': []}]  # 1 entry, cap 4 → 3 more allowed\n"
        "n = ex2_run_with_offset_callback(n_steps=50, eval_every=5, start_step=0, n_eval=1, sink=sink, max_entries=4)\n"
        "assert n == 3, f'should add 3 new entries to reach cap 4, got {n}'\n"
        "assert len(sink) == 4\n"
        "# The new entries' steps are 0, 5, 10 (first three K-aligned).\n"
        "new_steps = [sink[i]['step'] for i in range(1, 4)]\n"
        "assert new_steps == [0, 5, 10]\n"
        "\n"
        "# === n_steps=0 → no fires ===\n"
        "sink = []\n"
        "n = ex2_run_with_offset_callback(n_steps=0, eval_every=1, start_step=0, n_eval=2, sink=sink, max_entries=10)\n"
        "assert n == 0 and sink == []"
    ),
    "solution_body": (
        "def ex2_run_with_offset_callback(n_steps, eval_every, start_step, n_eval, sink, max_entries):\n"
        "    n_fires = 0\n"
        "    for step in range(n_steps):\n"
        "        if step < start_step:\n"
        "            continue\n"
        "        if (step - start_step) % eval_every != 0:\n"
        "            continue\n"
        "        if len(sink) >= max_entries:\n"
        "            break  # sink full, no further fires will succeed either\n"
        "        samples = [f'step={step}-sample={i}' for i in range(n_eval)]\n"
        "        sink.append({'step': step, 'samples': samples})\n"
        "        n_fires += 1\n"
        "    return n_fires"
    ),
    "solution_notes": (
        "**Why `break` instead of `continue` on cap-hit.** Once `len(sink) "
        ">= max_entries`, no later step will pass the guard either — the "
        "sink only grows. `break` is correct AND faster than walking the "
        "remaining n_steps doing nothing.\n\n"
        "**Offset alignment, not absolute alignment.** Ex1's `step % K == 0` "
        "fires at 0, K, 2K. With `start_step=5, K=3`, the fires are 5, 8, "
        "11, 14 — aligned to the offset origin, NOT to the absolute clock. "
        "This is what users want when they say 'eval every K steps after "
        "warm-up'.\n\n"
        "**Don't double-count.** The return value is fires from THIS call. "
        "A pre-populated sink stays unchanged in the counter. Real ARENA "
        "callbacks track per-call fires so a downstream throttle can "
        "back-off."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 4 — time-stage-instrumentation ex2
# ---------------------------------------------------------------------------

SPEC_STAGE_CTX = {
    "atom_id": "time-stage-instrumentation",
    "subtopic": "Logging: time-stage instrumentation",
    "topic_folder": TOPIC_LOG,
    "atom_recap_md": RECAP_STAGE_CTX,
    "exercise_index": 2,
    "exercise_title": "context-manager `stage` helper that accumulates per-name elapsed time across nested calls",
    "slug": "stage-context-manager-accumulator-nested-safe",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["context-manager", "perf_counter", "accumulate", "exception-safe"],
    "kcs": [
        "contextmanager-finally-accumulate",
        "dict-get-default-accumulator",
    ],
    "lo": (
        "Apply `@contextmanager` + `try/finally` over `time.perf_counter()` "
        "to build a `stage(name, acc)` helper that accumulates per-stage "
        "seconds even when the wrapped block raises, and supports nested "
        "(non-overlapping) re-entry into the same accumulator."
    ),
    "prompt_body": (
        "Implement `ex2_stage(name, acc)`. A reusable context manager for "
        "per-stage timing with crash-safe accumulation.\n\n"
        "Contract:\n\n"
        "1. Decorated with `@contextlib.contextmanager`.\n"
        "2. Records `t0 = time.perf_counter()` at entry.\n"
        "3. `yield` (no value needed).\n"
        "4. In a `finally` block: compute `elapsed = time.perf_counter() - "
        "t0` and do `acc[name] = acc.get(name, 0.0) + elapsed`. Use "
        "`.get(name, 0.0)` so a fresh accumulator dict works.\n"
        "5. If an exception is raised inside the block, the `finally` "
        "still records the elapsed time, then the exception propagates "
        "naturally (do NOT swallow it).\n\n"
        "Inputs:\n"
        "- `name`: `str`, key into `acc`.\n"
        "- `acc`: `dict[str, float]`, mutable accumulator.\n\n"
        "Output: context manager (you don't return a value from `yield`).\n\n"
        "The test exercises: (a) basic per-name sum, (b) exception "
        "propagation with accumulation preserved, (c) nested same-name "
        "re-entry summing both inner and outer elapsed."
    ),
    "stub": (
        "import time\n"
        "import contextlib\n"
        "\n"
        "@contextlib.contextmanager\n"
        "def ex2_stage(name: str, acc: dict):\n"
        '    """Accumulate elapsed seconds for `name` into `acc` (safe under exceptions)."""\n'
        "    raise NotImplementedError()\n"
        "    yield  # unreachable; for the @contextmanager decorator"
    ),
    "test_body": (
        "import time\n"
        "\n"
        "# === Basic per-name accumulation ===\n"
        "acc = {}\n"
        "with ex2_stage('forward', acc):\n"
        "    time.sleep(0.010)\n"
        "assert 'forward' in acc, f'forward must be recorded; got {acc}'\n"
        "assert acc['forward'] >= 0.010 - 1e-4, f'forward total too low: {acc[\"forward\"]}'\n"
        "assert acc['forward'] < 0.5, f'forward total absurdly high: {acc[\"forward\"]}'\n"
        "\n"
        "# === Multi-stage sum into same dict ===\n"
        "acc = {}\n"
        "for _ in range(3):\n"
        "    with ex2_stage('forward', acc):\n"
        "        time.sleep(0.005)\n"
        "    with ex2_stage('backward', acc):\n"
        "        time.sleep(0.002)\n"
        "assert set(acc.keys()) == {'forward', 'backward'}\n"
        "assert acc['forward'] >= 3 * 0.005 - 1e-4, f'forward sum too low: {acc[\"forward\"]}'\n"
        "assert acc['backward'] >= 3 * 0.002 - 1e-4, f'backward sum too low: {acc[\"backward\"]}'\n"
        "assert acc['forward'] > acc['backward'], (\n"
        "    f'forward should dominate; got forward={acc[\"forward\"]:.4f}, backward={acc[\"backward\"]:.4f}'\n"
        ")\n"
        "\n"
        "# === Exception still records elapsed AND re-raises ===\n"
        "acc = {}\n"
        "raised = False\n"
        "try:\n"
        "    with ex2_stage('crash', acc):\n"
        "        time.sleep(0.005)\n"
        "        raise RuntimeError('boom')\n"
        "except RuntimeError as e:\n"
        "    raised = True\n"
        "    assert 'boom' in str(e), f'exception must propagate verbatim, got {e!r}'\n"
        "assert raised, 'exception must propagate (not swallowed)'\n"
        "assert 'crash' in acc, 'finally must record even when block raises'\n"
        "assert acc['crash'] >= 0.005 - 1e-4, f'crash recorded too little: {acc[\"crash\"]}'\n"
        "\n"
        "# === Nested same-name → both elapsed times sum ===\n"
        "acc = {}\n"
        "with ex2_stage('forward', acc):\n"
        "    time.sleep(0.010)\n"
        "    with ex2_stage('forward', acc):\n"
        "        time.sleep(0.005)\n"
        "# Outer block runs for ~15ms total (including the inner sleep);\n"
        "# inner block runs for ~5ms. Sum is ~20ms.\n"
        "assert acc['forward'] >= 0.020 - 1e-4, f'nested sum should be >=20ms, got {acc[\"forward\"]:.4f}'\n"
        "assert acc['forward'] < 1.0, f'nested sum absurdly large: {acc[\"forward\"]:.4f}'\n"
        "\n"
        "# === Pre-existing key value is ADDED to, not overwritten ===\n"
        "acc = {'forward': 100.0}  # pretend we already had elapsed from earlier\n"
        "with ex2_stage('forward', acc):\n"
        "    time.sleep(0.005)\n"
        "assert acc['forward'] >= 100.0 + 0.005 - 1e-4, f'must accumulate onto pre-existing value, got {acc[\"forward\"]:.4f}'\n"
        "assert acc['forward'] < 101.0\n"
        "\n"
        "# === Context manager yields nothing ===\n"
        "acc = {}\n"
        "with ex2_stage('noop', acc) as ret:\n"
        "    yielded = ret\n"
        "assert yielded is None, f'context manager should yield nothing, got {yielded!r}'"
    ),
    "solution_body": (
        "import time\n"
        "import contextlib\n"
        "\n"
        "@contextlib.contextmanager\n"
        "def ex2_stage(name, acc):\n"
        "    t0 = time.perf_counter()\n"
        "    try:\n"
        "        yield\n"
        "    finally:\n"
        "        elapsed = time.perf_counter() - t0\n"
        "        acc[name] = acc.get(name, 0.0) + elapsed"
    ),
    "solution_notes": (
        "**`try/finally` is the load-bearing structural choice.** A "
        "try/except would let you handle the exception inside the CM, but "
        "you'd have to re-raise to preserve user semantics. `finally` "
        "guarantees the accumulation regardless of how the block exits — "
        "normal return, exception, or even `return` inside a function "
        "wrapping the `with`.\n\n"
        "**`acc.get(name, 0.0)` over `if name in acc`.** One line. Default "
        "handling baked in. No race between the existence check and the "
        "assignment (irrelevant single-threaded, but a habit worth keeping).\n\n"
        "**Nested same-name accumulates by design.** Outer block's "
        "elapsed includes the inner block's elapsed (because perf_counter "
        "ticks the whole time). Inner block also adds its own elapsed. "
        "Total = outer + inner. For non-overlapping nested stages this is "
        "the correct sum; for overlapping you'd want a different design."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 5 — wandb-config-into-args ex2
# ---------------------------------------------------------------------------

SPEC_CONFIG_PRECEDENCE = {
    "atom_id": "wandb-config-into-args",
    "subtopic": "Logging: wandb.config into args",
    "topic_folder": TOPIC_LOG,
    "atom_recap_md": RECAP_CONFIG_PRECEDENCE,
    "exercise_index": 2,
    "exercise_title": "three-tier precedence: defaults < wandb.config < cli_overrides",
    "slug": "three-tier-precedence-defaults-sweep-cli",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["wandb", "sweep", "cli", "precedence"],
    "kcs": [
        "ordered-setattr-precedence",
        "skip-unknown-keys-defensive",
    ],
    "lo": (
        "Apply chained `setattr(args, k, v) if hasattr(args, k)` in the "
        "order defaults → wandb.config → cli_overrides, so the highest-"
        "precedence layer's values are the ones that stick on the "
        "dataclass `args`."
    ),
    "prompt_body": (
        "Implement `ex2_apply_three_tier_config(args, sweep_cfg, "
        "cli_overrides)`. Ex1 collapsed sweep into args; ex2 adds a third "
        "tier on top.\n\n"
        "Inputs:\n"
        "- `args`: a dataclass instance with hparam fields.\n"
        "- `sweep_cfg`: `dict` from `dict(wandb.config)` — may include "
        "wandb metadata keys (`'_wandb'`, `'_runtime'`) that aren't fields "
        "on args.\n"
        "- `cli_overrides`: `dict` from argparse-style explicit user "
        "flags.\n\n"
        "Algorithm:\n"
        "1. For each `(k, v)` in `sweep_cfg.items()`: if `hasattr(args, "
        "k)`, `setattr(args, k, v)`. Else skip.\n"
        "2. For each `(k, v)` in `cli_overrides.items()`: same — `setattr` "
        "only if `hasattr(args, k)`. Skip unknown.\n"
        "3. Return the (mutated) `args`.\n\n"
        "Precedence: cli_overrides applied LAST wins. The `defaults` tier "
        "is implicit in the dataclass field defaults — already on `args` "
        "at entry.\n\n"
        "Return the SAME args instance (mutation contract from ex1)."
    ),
    "stub": (
        "def ex2_apply_three_tier_config(args, sweep_cfg: dict, cli_overrides: dict):\n"
        '    """Apply sweep then CLI on top of args; skip unknown keys; return args."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass\n"
        "class FakeArgs:\n"
        "    lr: float = 1e-3\n"
        "    batch_size: int = 64\n"
        "    optimizer: str = 'adam'\n"
        "    weight_decay: float = 0.0\n"
        "    project: str = 'arena-default'\n"
        "\n"
        "# === Sweep alone overwrites defaults ===\n"
        "args = FakeArgs()\n"
        "out = ex2_apply_three_tier_config(args, {'lr': 5e-4, 'batch_size': 128}, {})\n"
        "assert out is args, 'must mutate-and-return the same instance'\n"
        "assert args.lr == 5e-4 and args.batch_size == 128\n"
        "assert args.optimizer == 'adam', 'untouched field stays default'\n"
        "\n"
        "# === CLI override BEATS sweep ===\n"
        "args = FakeArgs()\n"
        "ex2_apply_three_tier_config(\n"
        "    args,\n"
        "    {'lr': 5e-4, 'batch_size': 128},          # sweep\n"
        "    {'lr': 1e-2},                              # CLI\n"
        ")\n"
        "assert args.lr == 1e-2, f'CLI lr (1e-2) must win over sweep lr (5e-4), got {args.lr}'\n"
        "assert args.batch_size == 128, 'batch_size from sweep (no CLI override) must persist'\n"
        "\n"
        "# === Sweep beats default; CLI beats both ===\n"
        "args = FakeArgs()\n"
        "ex2_apply_three_tier_config(\n"
        "    args,\n"
        "    {'weight_decay': 0.01},                   # sweep\n"
        "    {'weight_decay': 0.0001},                  # CLI\n"
        ")\n"
        "assert args.weight_decay == 0.0001\n"
        "\n"
        "# === Unknown keys silently skipped in BOTH layers ===\n"
        "args = FakeArgs()\n"
        "ex2_apply_three_tier_config(\n"
        "    args,\n"
        "    {'lr': 5e-4, '_wandb': {'v': '0.17'}, 'mystery_a': 1},\n"
        "    {'optimizer': 'sgd', '_runtime': 100, 'mystery_b': 2},\n"
        ")\n"
        "assert args.lr == 5e-4 and args.optimizer == 'sgd'\n"
        "assert not hasattr(args, '_wandb')\n"
        "assert not hasattr(args, 'mystery_a')\n"
        "assert not hasattr(args, '_runtime')\n"
        "assert not hasattr(args, 'mystery_b')\n"
        "\n"
        "# === Both dicts empty → args unchanged ===\n"
        "args = FakeArgs(lr=2e-3)\n"
        "ex2_apply_three_tier_config(args, {}, {})\n"
        "assert args.lr == 2e-3 and args.batch_size == 64\n"
        "\n"
        "# === Only CLI provided → CLI overrides defaults directly ===\n"
        "args = FakeArgs()\n"
        "ex2_apply_three_tier_config(args, {}, {'project': 'arena-cli'})\n"
        "assert args.project == 'arena-cli'\n"
        "\n"
        "# === Only sweep provided → sweep overrides defaults ===\n"
        "args = FakeArgs()\n"
        "ex2_apply_three_tier_config(args, {'project': 'arena-sweep'}, {})\n"
        "assert args.project == 'arena-sweep'\n"
        "\n"
        "# === CLI=None still overrides (explicit None choice) ===\n"
        "args = FakeArgs()\n"
        "ex2_apply_three_tier_config(args, {'lr': 5e-4}, {'lr': None})\n"
        "assert args.lr is None, f'CLI None must win, got {args.lr!r}'\n"
        "\n"
        "# === Inputs dicts are not mutated ===\n"
        "args = FakeArgs()\n"
        "sw = {'lr': 5e-4}\n"
        "cli = {'lr': 1e-2}\n"
        "ex2_apply_three_tier_config(args, sw, cli)\n"
        "assert sw == {'lr': 5e-4} and cli == {'lr': 1e-2}, 'input dicts must not be mutated'"
    ),
    "solution_body": (
        "def ex2_apply_three_tier_config(args, sweep_cfg, cli_overrides):\n"
        "    for k, v in sweep_cfg.items():\n"
        "        if hasattr(args, k):\n"
        "            setattr(args, k, v)\n"
        "    for k, v in cli_overrides.items():\n"
        "        if hasattr(args, k):\n"
        "            setattr(args, k, v)\n"
        "    return args"
    ),
    "solution_notes": (
        "**Order is the only thing that matters.** Both layers use the "
        "same `hasattr` + `setattr` skeleton; the precedence comes "
        "entirely from APPLICATION ORDER. Last write wins. Reversing the "
        "two `for` loops would make sweep beat CLI — the opposite of what "
        "production sweep harnesses expect.\n\n"
        "**Why not merge dicts first, then setattr.** You COULD do "
        "`merged = {**sweep_cfg, **cli_overrides}; for k, v in merged...`. "
        "Same end state. The explicit two-pass form scales better when "
        "you add a fourth tier (env vars, file config) and is easier to "
        "step-debug.\n\n"
        "**CLI=None as a legitimate override.** A user passing `--lr=None` "
        "explicitly is making a choice — the apply layer shouldn't filter "
        "it. If the downstream training code can't handle `lr=None`, "
        "that's a validation step at the dataclass `__post_init__`, not "
        "here."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 6 — wandb-watch-model ex2
# ---------------------------------------------------------------------------

SPEC_WATCH_TRAINABLE = {
    "atom_id": "wandb-watch-model",
    "subtopic": "Logging: wandb.watch model",
    "topic_folder": TOPIC_LOG,
    "atom_recap_md": RECAP_WATCH_TRAINABLE,
    "exercise_index": 2,
    "exercise_title": "auto-watch every submodule with at least one trainable own-parameter",
    "slug": "auto-watch-trainable-submodules-with-named-modules",
    "bloom_level": "Analyze",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["wandb", "watch", "named-modules", "trainable", "requires_grad"],
    "kcs": [
        "named-modules-with-recurse-false-params",
        "trainable-param-filter-any",
    ],
    "lo": (
        "Analyze a model by walking `named_modules()` and watch (via "
        "`wandb.watch`) only the submodules that own at least one "
        "trainable parameter, returning the sorted list of watched qnames."
    ),
    "prompt_body": (
        "Implement `ex2_watch_trainable_modules(model, log_freq)`. The "
        "auto-watch generalization of ex1.\n\n"
        "Algorithm:\n"
        "1. Walk `model.named_modules()`.\n"
        "2. SKIP entries where `qname == ''` (the root container).\n"
        "3. For each remaining `(qname, m)`:\n"
        "   - Get `own = list(m.parameters(recurse=False))` — params "
        "OWNED by `m`, not its children.\n"
        "   - If `own` is empty: skip (no parameters of its own — it's a "
        "wrapper like `nn.Sequential`).\n"
        "   - If `any(p.requires_grad for p in own)`: call "
        "`wandb.watch(m, log='all', log_freq=log_freq)` AND append `qname` "
        "to a `watched` list.\n"
        "   - Else (all own params are frozen): skip (don't watch frozen "
        "layers — they produce zero-gradient histograms).\n"
        "4. Return `sorted(watched)`.\n\n"
        "Inputs:\n"
        "- `model`: `nn.Module`.\n"
        "- `log_freq`: int.\n\n"
        "Output: sorted `list[str]` of watched qnames.\n\n"
        "The test mocks `wandb.watch` to verify exactly which modules "
        "got hooked + that `log='all'` + `log_freq` are forwarded "
        "correctly."
    ),
    "stub": (
        "import sys\n"
        "from unittest.mock import MagicMock\n"
        "sys.modules.setdefault('wandb', MagicMock())\n"
        "import wandb\n"
        "import torch.nn as nn\n"
        "\n"
        "def ex2_watch_trainable_modules(model: nn.Module, log_freq: int) -> list:\n"
        '    """Watch every submodule with at least one trainable own-param; return sorted qnames."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "import torch.nn as nn\n"
        "import sys\n"
        "from unittest.mock import MagicMock\n"
        "if 'wandb' in sys.modules and not isinstance(sys.modules['wandb'], MagicMock):\n"
        "    del sys.modules['wandb']\n"
        "sys.modules.setdefault('wandb', MagicMock())\n"
        "import wandb\n"
        "\n"
        "# === Build a small model: encoder frozen, head trainable ===\n"
        "class Head(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc1 = nn.Linear(16, 32)\n"
        "        self.fc2 = nn.Linear(32, 10)\n"
        "\n"
        "class Net(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.encoder = nn.Sequential(\n"
        "            nn.Linear(8, 16),\n"
        "            nn.ReLU(),\n"
        "            nn.Linear(16, 16),\n"
        "        )\n"
        "        self.head = Head()\n"
        "        # Freeze the encoder.\n"
        "        for p in self.encoder.parameters():\n"
        "            p.requires_grad_(False)\n"
        "\n"
        "wandb.watch.reset_mock()\n"
        "model = Net()\n"
        "watched = ex2_watch_trainable_modules(model, log_freq=50)\n"
        "\n"
        "# === Watched list is sorted and contains exactly the trainable-owning leaves ===\n"
        "assert isinstance(watched, list), f'must return list, got {type(watched).__name__}'\n"
        "assert watched == sorted(watched), f'must be sorted, got {watched}'\n"
        "# Encoder's Linear children OWN params but they're frozen → not watched.\n"
        "# ReLU has no params → not watched.\n"
        "# Sequential is a wrapper with no own params → not watched.\n"
        "# Head itself has no own params (its fcN do) → not watched.\n"
        "# So watched = ['head.fc1', 'head.fc2'].\n"
        "assert watched == ['head.fc1', 'head.fc2'], f'watched qnames wrong: {watched}'\n"
        "\n"
        "# === wandb.watch called exactly len(watched) times ===\n"
        "assert wandb.watch.call_count == 2, f'expected 2 wandb.watch calls, got {wandb.watch.call_count}'\n"
        "\n"
        "# === Each call uses log='all' and the passed log_freq ===\n"
        "for call in wandb.watch.call_args_list:\n"
        "    assert call.kwargs.get('log') == 'all', f\"log must be 'all', got {call.kwargs.get('log')!r}\"\n"
        "    assert call.kwargs.get('log_freq') == 50, f'log_freq must propagate, got {call.kwargs.get(\"log_freq\")!r}'\n"
        "    # Positional arg = the module itself.\n"
        "    assert isinstance(call.args[0], nn.Linear), f'must watch the actual leaf module, got {type(call.args[0]).__name__}'\n"
        "\n"
        "# === Unfreezing the encoder grows the watched set ===\n"
        "wandb.watch.reset_mock()\n"
        "for p in model.encoder.parameters():\n"
        "    p.requires_grad_(True)\n"
        "watched2 = ex2_watch_trainable_modules(model, log_freq=100)\n"
        "# Encoder Linears (encoder.0, encoder.2) now also trainable.\n"
        "assert watched2 == ['encoder.0', 'encoder.2', 'head.fc1', 'head.fc2'], (\n"
        "    f'unfreezing should add 2 more watched modules, got {watched2}'\n"
        ")\n"
        "assert wandb.watch.call_count == 4\n"
        "# log_freq propagates the new value.\n"
        "for call in wandb.watch.call_args_list:\n"
        "    assert call.kwargs.get('log_freq') == 100\n"
        "\n"
        "# === Fully-frozen model → empty watched list, zero wandb.watch calls ===\n"
        "wandb.watch.reset_mock()\n"
        "for p in model.parameters():\n"
        "    p.requires_grad_(False)\n"
        "watched3 = ex2_watch_trainable_modules(model, log_freq=50)\n"
        "assert watched3 == [], f'all-frozen model should yield empty list, got {watched3}'\n"
        "assert wandb.watch.call_count == 0\n"
        "\n"
        "# === Root module ('') excluded even if it has own params ===\n"
        "# Build a model where the root itself has a Parameter directly attached.\n"
        "wandb.watch.reset_mock()\n"
        "class WithRootParam(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.bias = nn.Parameter(t.zeros(4))\n"
        "        self.fc = nn.Linear(4, 4)\n"
        "m2 = WithRootParam()\n"
        "watched4 = ex2_watch_trainable_modules(m2, log_freq=10)\n"
        "# Even though root has a trainable own-param, qname=='' is filtered.\n"
        "assert '' not in watched4, 'root qname must be filtered'\n"
        "assert watched4 == ['fc'], f'expected [\"fc\"], got {watched4}'"
    ),
    "solution_body": (
        "import sys\n"
        "from unittest.mock import MagicMock\n"
        "sys.modules.setdefault('wandb', MagicMock())\n"
        "import wandb\n"
        "import torch.nn as nn\n"
        "\n"
        "def ex2_watch_trainable_modules(model, log_freq):\n"
        "    watched = []\n"
        "    for qname, m in model.named_modules():\n"
        "        if qname == '':\n"
        "            continue\n"
        "        own = list(m.parameters(recurse=False))\n"
        "        if not own:\n"
        "            continue\n"
        "        if any(p.requires_grad for p in own):\n"
        "            wandb.watch(m, log='all', log_freq=log_freq)\n"
        "            watched.append(qname)\n"
        "    return sorted(watched)"
    ),
    "solution_notes": (
        "**`recurse=False` is the structural choice.** Without it, every "
        "ancestor of a trainable leaf reports `requires_grad=True` and "
        "gets a hook — you'd double- or triple-hook the same tensor "
        "histograms. `recurse=False` walks down to leaves and the wrapper "
        "modules drop out naturally.\n\n"
        "**Why `any` not `all`.** A LayerNorm with frozen weight but "
        "trainable bias should still be watched — `any` captures the "
        "'at least one trainable scalar' semantic. `all` would miss "
        "partially-frozen modules, which is the realistic case in "
        "fine-tuning.\n\n"
        "**Sorted output for stable tests.** `named_modules` walks in "
        "insertion order, which is reproducible but order-dependent on "
        "model construction. Sorting the output makes the watched-list "
        "diffable across model edits."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 7 — any-reduce-axis ex2
# ---------------------------------------------------------------------------

SPEC_ANY_KEEPDIM = {
    "atom_id": "any-reduce-axis",
    "subtopic": "Numpy: any() reduce along axis",
    "topic_folder": TOPIC_MISC,
    "atom_recap_md": RECAP_ANY_KEEPDIM,
    "exercise_index": 2,
    "exercise_title": "any(dim=k, keepdim=True) and broadcast-blank rows that contain no positives",
    "slug": "any-keepdim-and-broadcast-blank-rows",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["any", "keepdim", "broadcast", "torch.where"],
    "kcs": [
        "any-with-keepdim-preserves-rank",
        "broadcast-mask-back-via-where",
    ],
    "lo": (
        "Apply `.any(dim=k, keepdim=True)` to produce a rank-preserving "
        "row-decision tensor, then broadcast it back via `torch.where` to "
        "blank rows that fail the predicate."
    ),
    "prompt_body": (
        "Implement `ex2_blank_rows_without_positive(x)`. The deepening "
        "variant of ex1.\n\n"
        "Inputs:\n"
        "- `x`: `(N, M)` float tensor.\n\n"
        "Algorithm:\n"
        "1. Build a `(N, M)` bool mask: `mask = (x > 0)`.\n"
        "2. Compute `row_has_pos = mask.any(dim=1, keepdim=True)` — shape "
        "`(N, 1)`, dtype bool.\n"
        "3. Return `t.where(row_has_pos, x, t.zeros_like(x))` — rows with "
        "at least one positive entry are passed through; rows with no "
        "positives are zeroed out (broadcast from the `(N, 1)` mask).\n\n"
        "Constraints:\n"
        "- DO NOT use a Python for-loop.\n"
        "- DO NOT use `keepdim=False` then `unsqueeze` — exercise the "
        "`keepdim=True` shape directly.\n"
        "- Preserve the input dtype.\n\n"
        "Output: `(N, M)` tensor, same dtype as `x`."
    ),
    "stub": (
        "def ex2_blank_rows_without_positive(x: Tensor) -> Tensor:\n"
        '    """Zero out any row of x that contains no positive entry; preserve others."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# === Hand-traced reference ===\n"
        "x = t.tensor([\n"
        "    [-1.0, -2.0,  3.0],   # row 0: has positive\n"
        "    [-1.0, -2.0, -3.0],   # row 1: all negative → blank\n"
        "    [ 0.0,  0.0,  0.0],   # row 2: no STRICT positive → blank\n"
        "    [ 1.0, -1.0,  0.0],   # row 3: has positive\n"
        "])\n"
        "out = ex2_blank_rows_without_positive(x)\n"
        "expected = t.tensor([\n"
        "    [-1.0, -2.0,  3.0],\n"
        "    [ 0.0,  0.0,  0.0],\n"
        "    [ 0.0,  0.0,  0.0],\n"
        "    [ 1.0, -1.0,  0.0],\n"
        "])\n"
        "assert out.shape == x.shape, f'shape must match input, got {tuple(out.shape)}'\n"
        "assert out.dtype == x.dtype, f'dtype must match input, got {out.dtype}'\n"
        "assert t.equal(out, expected), f'mismatch\\nexpected={expected}\\nactual={out}'\n"
        "\n"
        "# === Original x is NOT mutated ===\n"
        "x_clone = x.clone()\n"
        "_ = ex2_blank_rows_without_positive(x_clone)\n"
        "assert t.equal(x_clone, x), 'must not mutate the input tensor'\n"
        "\n"
        "# === All-positive matrix → identity ===\n"
        "x = t.ones(5, 4)\n"
        "assert t.equal(ex2_blank_rows_without_positive(x), x)\n"
        "\n"
        "# === All-negative matrix → all zeros ===\n"
        "x = -t.ones(5, 4)\n"
        "assert t.equal(ex2_blank_rows_without_positive(x), t.zeros(5, 4))\n"
        "\n"
        "# === Single-row + single-col edge cases ===\n"
        "x = t.tensor([[1.0, -1.0]])\n"
        "assert t.equal(ex2_blank_rows_without_positive(x), x)\n"
        "x = t.tensor([[-1.0, -1.0]])\n"
        "assert t.equal(ex2_blank_rows_without_positive(x), t.zeros_like(x))\n"
        "x = t.tensor([[5.0], [-3.0], [0.0]])\n"
        "expected = t.tensor([[5.0], [0.0], [0.0]])\n"
        "assert t.equal(ex2_blank_rows_without_positive(x), expected)\n"
        "\n"
        "# === dtype preservation: int input → int output ===\n"
        "x = t.tensor([[1, -2, 3], [-1, -2, -3]])\n"
        "out = ex2_blank_rows_without_positive(x)\n"
        "assert out.dtype == x.dtype, f'int input must give int output, got {out.dtype}'\n"
        "assert t.equal(out, t.tensor([[1, -2, 3], [0, 0, 0]]))\n"
        "\n"
        "# === Larger random check vs explicit row loop ===\n"
        "rng = t.Generator().manual_seed(0)\n"
        "big = t.randn(64, 32, generator=rng) - 0.3  # bias toward negative\n"
        "out = ex2_blank_rows_without_positive(big)\n"
        "for i in range(big.shape[0]):\n"
        "    row = big[i]\n"
        "    if (row > 0).any():\n"
        "        assert t.equal(out[i], row), f'row {i}: positive-bearing must pass through'\n"
        "    else:\n"
        "        assert t.all(out[i] == 0), f'row {i}: no-positive must be zero'\n"
        "\n"
        "# === Strictly > 0, not >= 0 (zero alone does not save a row) ===\n"
        "x = t.tensor([[0.0, 0.0, 0.0]])\n"
        "assert t.equal(ex2_blank_rows_without_positive(x), t.zeros_like(x))"
    ),
    "solution_body": (
        "def ex2_blank_rows_without_positive(x):\n"
        "    row_has_pos = (x > 0).any(dim=1, keepdim=True)  # (N, 1) bool\n"
        "    return t.where(row_has_pos, x, t.zeros_like(x))"
    ),
    "solution_notes": (
        "**Why `keepdim=True` over `unsqueeze`.** Both end up at `(N, 1)`, "
        "but `keepdim=True` says it at the reduce site — one operation "
        "instead of two, and the intent ('I want to broadcast this back') "
        "is read off the `any` line.\n\n"
        "**`torch.where(cond, a, b)` is the broadcasting select.** The "
        "condition broadcasts against `a` and `b`; output has the "
        "broadcast shape. `(N, 1)` × `(N, M)` → `(N, M)`, exactly the "
        "result we want.\n\n"
        "**Strict `>` matters for the spec.** `x >= 0` would let an "
        "all-zero row pass — different semantic. The exercise picks `> 0` "
        "explicitly to drive home that `any` is dispatching a Python "
        "boolean comparison and the comparison choice is yours."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# SPEC 8 — functional-module-wrap ex2
# ---------------------------------------------------------------------------

SPEC_FUNC_PARAMETRIC = {
    "atom_id": "functional-module-wrap",
    "subtopic": "PyTorch: functional module wrap",
    "topic_folder": TOPIC_MISC,
    "atom_recap_md": RECAP_FUNC_PARAMETRIC,
    "exercise_index": 2,
    "exercise_title": "MyLeakyReLU — parametric F.leaky_relu wrap with extra_repr",
    "slug": "myleakyrelu-parametric-wrap-with-extra-repr",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["leaky-relu", "module", "extra_repr", "parametric"],
    "kcs": [
        "parametric-functional-wrap",
        "extra_repr-hparam-display",
    ],
    "lo": (
        "Apply the parametric functional-to-Module wrap pattern: store "
        "`negative_slope` and `inplace` as plain attributes, delegate "
        "`forward` to `F.leaky_relu(x, negative_slope, inplace)`, and "
        "expose the hparams via `extra_repr`."
    ),
    "prompt_body": (
        "Implement `MyLeakyReLU(nn.Module)`. Like ex1's MyReLU but "
        "PARAMETRIC.\n\n"
        "Constraints:\n"
        "1. Subclass `nn.Module`. Call `super().__init__()` in "
        "`__init__`.\n"
        "2. `__init__(self, negative_slope=0.01, inplace=False)` stores "
        "BOTH parameters as plain Python attributes (NOT as "
        "`nn.Parameter` — these are config, not learnables).\n"
        "3. `forward(x)` returns `F.leaky_relu(x, self.negative_slope, "
        "self.inplace)`. Do NOT delegate to `nn.LeakyReLU` internally — "
        "exercise the functional wrap directly.\n"
        "4. `extra_repr(self)` returns "
        "`f'negative_slope={self.negative_slope}, "
        "inplace={self.inplace}'` so `print(model)` shows the hparams.\n\n"
        "Output: an `nn.Module` subclass that matches `nn.LeakyReLU` "
        "numerically and has zero `model.parameters()` entries."
    ),
    "stub": (
        "import torch.nn as nn\n"
        "import torch.nn.functional as F\n"
        "\n"
        "class MyLeakyReLU(nn.Module):\n"
        "    def __init__(self, negative_slope: float = 0.01, inplace: bool = False):\n"
        "        raise NotImplementedError()\n"
        "\n"
        "    def forward(self, x):\n"
        "        raise NotImplementedError()\n"
        "\n"
        "    def extra_repr(self):\n"
        "        raise NotImplementedError()"
    ),
    "test_body": (
        "import torch.nn as nn\n"
        "import torch.nn.functional as F\n"
        "\n"
        "# === Default slope matches nn.LeakyReLU on a mix of signs ===\n"
        "x = t.tensor([-2.0, -0.5, 0.0, 0.5, 3.0])\n"
        "my = MyLeakyReLU()\n"
        "ref = nn.LeakyReLU()\n"
        "assert t.allclose(my(x), ref(x)), f'default-slope mismatch: my={my(x)}, ref={ref(x)}'\n"
        "\n"
        "# === Custom slope matches F.leaky_relu directly ===\n"
        "for slope in [0.0, 0.01, 0.2, 0.5]:\n"
        "    my = MyLeakyReLU(negative_slope=slope)\n"
        "    out = my(x)\n"
        "    ref = F.leaky_relu(x, slope)\n"
        "    assert t.allclose(out, ref), f'slope={slope}: my={out}, ref={ref}'\n"
        "\n"
        "# === Subclass + no parameters (config-only) ===\n"
        "assert isinstance(my, nn.Module), 'MyLeakyReLU must subclass nn.Module'\n"
        "params = list(my.parameters())\n"
        "assert params == [], f'no learnable params expected, got {params}'\n"
        "buffers = list(my.buffers())\n"
        "assert buffers == [], f'no buffers expected, got {buffers}'\n"
        "\n"
        "# === Hparams stored as plain attributes ===\n"
        "m = MyLeakyReLU(negative_slope=0.2, inplace=False)\n"
        "assert m.negative_slope == 0.2, f'negative_slope attr wrong: {m.negative_slope!r}'\n"
        "assert m.inplace is False, f'inplace attr wrong: {m.inplace!r}'\n"
        "# Hparams are NOT nn.Parameter instances.\n"
        "assert not isinstance(m.negative_slope, nn.Parameter), 'negative_slope must be plain float, not nn.Parameter'\n"
        "\n"
        "# === extra_repr exposes both hparams in the printed form ===\n"
        "r = MyLeakyReLU(negative_slope=0.3, inplace=True).extra_repr()\n"
        "assert isinstance(r, str), f'extra_repr must return str, got {type(r).__name__}'\n"
        "assert 'negative_slope=0.3' in r, f'extra_repr must contain negative_slope=0.3, got {r!r}'\n"
        "assert 'inplace=True' in r, f'extra_repr must contain inplace=True, got {r!r}'\n"
        "# And it shows up in repr(model).\n"
        "model_str = repr(MyLeakyReLU(negative_slope=0.25))\n"
        "assert 'negative_slope=0.25' in model_str, f'repr should embed extra_repr, got {model_str!r}'\n"
        "\n"
        "# === Inplace mode actually mutates ===\n"
        "x_mut = t.tensor([-1.0, 2.0, -3.0])\n"
        "orig_ptr = x_mut.data_ptr()\n"
        "MyLeakyReLU(negative_slope=0.1, inplace=True)(x_mut)\n"
        "assert x_mut.data_ptr() == orig_ptr, 'inplace must reuse storage'\n"
        "assert t.allclose(x_mut, t.tensor([-0.1, 2.0, -0.3])), f'inplace result wrong: {x_mut}'\n"
        "\n"
        "# === Composes inside nn.Sequential and matches LeakyReLU ===\n"
        "t.manual_seed(42)\n"
        "net1 = nn.Sequential(nn.Linear(4, 3), MyLeakyReLU(negative_slope=0.2), nn.Linear(3, 2))\n"
        "t.manual_seed(42)\n"
        "net2 = nn.Sequential(nn.Linear(4, 3), nn.LeakyReLU(negative_slope=0.2), nn.Linear(3, 2))\n"
        "x = t.randn(8, 4)\n"
        "assert t.allclose(net1(x), net2(x), atol=1e-6), 'composed nets must match nn.LeakyReLU exactly'\n"
        "\n"
        "# === No child modules (do not delegate to nn.LeakyReLU) ===\n"
        "children = list(MyLeakyReLU().children())\n"
        "assert children == [], f'MyLeakyReLU should not delegate to a child nn.LeakyReLU; got {children}'\n"
        "\n"
        "# === Gradient flows correctly with the slope on the negative side ===\n"
        "x = t.tensor([-2.0, -1.0, 1.0, 2.0], requires_grad=True)\n"
        "y = MyLeakyReLU(negative_slope=0.25)(x).sum()\n"
        "y.backward()\n"
        "# Gradient of leaky_relu wrt x: 1 where x>0, slope where x<=0.\n"
        "expected_grad = t.where(x.detach() > 0, t.ones_like(x.detach()), t.full_like(x.detach(), 0.25))\n"
        "assert t.allclose(x.grad, expected_grad), f'grad mismatch: got {x.grad}, expected {expected_grad}'"
    ),
    "solution_body": (
        "import torch.nn as nn\n"
        "import torch.nn.functional as F\n"
        "\n"
        "class MyLeakyReLU(nn.Module):\n"
        "    def __init__(self, negative_slope=0.01, inplace=False):\n"
        "        super().__init__()\n"
        "        self.negative_slope = negative_slope\n"
        "        self.inplace = inplace\n"
        "\n"
        "    def forward(self, x):\n"
        "        return F.leaky_relu(x, self.negative_slope, self.inplace)\n"
        "\n"
        "    def extra_repr(self):\n"
        "        return f'negative_slope={self.negative_slope}, inplace={self.inplace}'"
    ),
    "solution_notes": (
        "**Hparams as plain attrs, not Parameters.** `nn.Parameter` would "
        "make `negative_slope` show up in `model.parameters()` and "
        "downstream optimizer construction would try to update it. The "
        "intent is config: the slope is fixed at construction time.\n\n"
        "**`extra_repr` returns a single line.** `nn.Module.__repr__` "
        "wraps it in the `ClassName(...)` envelope and handles "
        "indentation for nested modules. You just supply the comma-joined "
        "hparam string.\n\n"
        "**`inplace=True` is a performance flag.** It writes the result "
        "into the input's storage instead of allocating a new tensor. "
        "Saves memory in long chains of activations but breaks autograd "
        "if the input is needed for the backward pass — that's why the "
        "default is `False`."
    ),
    "extra_imports": [],
}


# ---------------------------------------------------------------------------
# All specs
# ---------------------------------------------------------------------------

SPECS = [
    SPEC_SWEEP_VALIDATE,
    SPEC_DIST_REGISTRY,
    SPEC_CALLBACK_OFFSET,
    SPEC_STAGE_CTX,
    SPEC_CONFIG_PRECEDENCE,
    SPEC_WATCH_TRAINABLE,
    SPEC_ANY_KEEPDIM,
    SPEC_FUNC_PARAMETRIC,
]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

def _verify_all(specs):
    import torch as t
    import numpy as np
    import torch.nn as nn
    from torch import Tensor
    import einops
    from einops import rearrange, reduce, repeat

    passed = 0
    failed = []

    for spec in specs:
        ex_id = f"ex{spec['exercise_index']}"
        tag = f"{spec['atom_id']}/{ex_id}"

        # Wandb drills: clear any cached wandb mock from a previous spec so
        # call counts start fresh.
        if "wandb" in sys.modules:
            del sys.modules["wandb"]

        ns = {
            "t": t,
            "np": np,
            "nn": nn,
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
    print(f"[deepening_v_batch11] Verifying {len(SPECS)} specs...")
    _verify_all(SPECS)

    print(f"\n[deepening_v_batch11] All verified — emitting notebooks.")
    for spec in SPECS:
        path = emit_standalone(spec)
        rel = path.relative_to(path.parents[3])
        print(f"  wrote {rel}")
    print(f"\n[deepening_v_batch11] {len(SPECS)} notebooks emitted.")


if __name__ == "__main__":
    main()
