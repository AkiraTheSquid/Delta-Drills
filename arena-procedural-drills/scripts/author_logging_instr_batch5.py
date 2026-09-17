#!/usr/bin/env python3
"""Author 8 standalone Colab drills for the logging + instrumentation atoms
that ARENA chap-3 (transformer-training) leans on.

Each drill = ONE LO + ONE Bloom level, max 2 KCs.

Atoms covered (all single-ex, ex1):

  wandb-init-run             — wandb.init(project=, name=, config=)
  wandb-finish               — wandb.finish() at end of train()
  wandb-log-step             — wandb.log({'loss': ...}, step=step_count)
  wandb-watch-model          — wandb.watch(module, log='all', log_freq=K)
  wandb-config-into-args     — read wandb.config and write onto args dataclass
  tqdm-postfix-metrics       — pbar.set_postfix(loss=..., ex=...)
  log-samples-eval-callback  — every K steps, dump N model outputs to a sink
  time-stage-instrumentation — time.perf_counter() around named stages

The backend venv lacks `wandb` (and may or may not have tqdm). The wandb drills
install a `MagicMock()` into `sys.modules['wandb']` before exec; `wandb.init` /
`wandb.log` / `wandb.watch` / `wandb.finish` become recordable MagicMock calls
that the test cell inspects. The tqdm drill uses real tqdm (it IS present in
the venv) but exercises only the `set_postfix(**metrics)` API — the actual
progress-bar rendering is irrelevant. `log-samples-eval-callback` uses a plain
list as the sink (no wandb dependency at all). `time-stage-instrumentation`
uses real `time.perf_counter` with a tight elapsed-time tolerance.

Each spec is verified by exec-ing (solution_body + test_body) in a fresh
namespace inside the build venv. Any failure aborts the build.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

TOPIC = "prereqs_logging_instr"


# ---------------------------------------------------------------------------
# Per-atom recap blocks.
# ---------------------------------------------------------------------------

RECAP_WANDB_INIT = (
    "## `wandb.init(project=, name=, config=)` — quick refresher\n"
    "\n"
    "Every wandb run starts with exactly one call to `wandb.init(...)`. It "
    "opens a new run in your project, names it (so you can find it in the "
    "dashboard), and snapshots the hyperparameter dict as `config` so the UI "
    "can group runs by hparam value.\n"
    "\n"
    "**Three kwargs that matter for ARENA training loops:**\n"
    "\n"
    "- `project=` — string, the wandb project name. All runs land here.\n"
    "- `name=` — string, the run name. ARENA uses `args.wandb_name`; if you "
    "omit it, wandb auto-generates a silly two-word name.\n"
    "- `config=` — dict (or dataclass), the snapshot of hparams. Wandb stores "
    "it so you can filter / group / compare runs by lr, batch_size, etc.\n"
    "\n"
    "**Where to call it.** Inside `pre_training_setup` (ARENA convention) or "
    "at the very top of `train()`. Never in `__init__` — that runs once at "
    "object construction, before sweep agents have a chance to override "
    "`wandb.config`.\n"
    "\n"
    "**Symmetric with `wandb.finish()`.** Every `wandb.init` should pair "
    "with a `wandb.finish` at the end of training (separate drill)."
)

RECAP_WANDB_FINISH = (
    "## `wandb.finish()` — quick refresher\n"
    "\n"
    "`wandb.finish()` closes the current wandb run: flushes any pending "
    "metrics, uploads the final media + artifacts, and tears down the "
    "background sync thread. After it returns, the run is marked `finished` "
    "in the dashboard and a fresh `wandb.init(...)` will open a new one.\n"
    "\n"
    "**Why you must call it explicitly.**\n"
    "\n"
    "- Inside a Jupyter / Colab notebook the Python process keeps running "
    "after `train()` returns — wandb doesn't know training ended unless you "
    "tell it. The run sits in the `running` state on the dashboard forever.\n"
    "- Inside a sweep, the agent calls `train()` repeatedly. If you don't "
    "finish each run, the second `wandb.init` either errors or silently "
    "appends to the wrong run.\n"
    "- Exceptions skip `finish` unless you wrap in try/finally. ARENA's "
    "default loop puts it at the end of `train()` without a finally — that's "
    "fine for the drill but be aware.\n"
    "\n"
    "**Symmetric pair.** Every `wandb.init(...)` (separate drill) wants a "
    "matching `wandb.finish()`."
)

RECAP_WANDB_LOG_STEP = (
    "## `wandb.log({...}, step=step_count)` — quick refresher\n"
    "\n"
    "`wandb.log(metrics_dict, step=int)` sends one row of scalar metrics to "
    "the wandb dashboard. The `step` kwarg is the x-axis value — wandb plots "
    "`metrics` against it.\n"
    "\n"
    "**Why pass `step` explicitly.** If you omit it, wandb uses an internal "
    "monotonic step counter that increments by 1 per `log` call. This is "
    "fine for single-runs, but disastrous when comparing runs with different "
    "batch sizes or different log frequencies — the x-axes don't line up.\n"
    "\n"
    "**ARENA convention: step = `self.examples_seen`.** Examples-seen is "
    "batch-size-invariant: 50 000 examples is 50 000 examples whether your "
    "batch was 32 or 256. The training-step counter is NOT — step 1000 with "
    "batch=32 has seen 32 000 examples, batch=256 has seen 256 000.\n"
    "\n"
    "**Values must be Python scalars (not tensors).** `loss.item()` not "
    "`loss`. Wandb serializes the dict to JSON; tensors aren't JSON-able. "
    "Pass `loss.item()`, `accuracy` (already a float), etc.\n"
    "\n"
    "**Multiple metrics in one call.** `wandb.log({'loss': l, 'acc': a, "
    "'lr': lr}, step=s)` is preferred over three separate `wandb.log` calls "
    "— one call = one row in the dashboard."
)

RECAP_WANDB_WATCH = (
    "## `wandb.watch(module, log='all', log_freq=K)` — quick refresher\n"
    "\n"
    "`wandb.watch(module, log='all', log_freq=K)` hooks the named module so "
    "wandb auto-logs its **parameters** AND **gradients** every K forward "
    "passes. You point it at a single submodule (or `self.model`) and the "
    "wandb dashboard grows histogram plots for each tracked tensor.\n"
    "\n"
    "**The three kwargs:**\n"
    "\n"
    "- First positional: the `nn.Module` (or list of modules) to watch.\n"
    "- `log='all'` — log both parameters AND gradients. Other options are "
    "`'parameters'`, `'gradients'`, or `None` (disable).\n"
    "- `log_freq=K` — log every K forward passes. ARENA's guidance: make "
    "this LESS than 1 epoch's worth of steps, otherwise the dashboard "
    "shows zero histograms.\n"
    "\n"
    "**Call once per run**, inside `pre_training_setup` after `wandb.init` "
    "and after the model is on the right device. Calling it twice on the "
    "same module duplicates the hooks.\n"
    "\n"
    "**ARENA's choice of submodule.** Watching `self.model` itself logs "
    "every layer's weights — useful but noisy. ARENA's resnet-fine-tune "
    "example watches just the head (`self.model.out_layers[-1]`) because "
    "that's the only module being trained."
)

RECAP_WANDB_CONFIG_ARGS = (
    "## `args = update_args(args, dict(wandb.config))` — quick refresher\n"
    "\n"
    "**The sweep pattern.** A wandb sweep samples hyperparameters and "
    "passes them to your `train()` function via `wandb.config`. Your "
    "dataclass `args` carries the same hparams as fields. You need to "
    "overwrite the args fields with whatever the sweep just sampled — "
    "otherwise `train()` uses its hardcoded defaults and the sweep has no "
    "effect.\n"
    "\n"
    "**The recipe** (ARENA 0_3_8):\n"
    "\n"
    "```python\n"
    "def train():\n"
    "    args = WandbResNetFinetuningArgs()       # hardcoded defaults\n"
    "    wandb.init(...)                          # sweep agent populates wandb.config\n"
    "    args = update_args(args, dict(wandb.config))  # overwrite from sweep\n"
    "    trainer = WandbResNetFinetuner(args)\n"
    "    trainer.train()\n"
    "```\n"
    "\n"
    "**`update_args` is a one-liner.** Loop over the sampled dict; for each "
    "key, `setattr(args, key, value)` if the field exists. Skip unknown "
    "keys (defensive — sweeps sometimes include meta keys like `_wandb`).\n"
    "\n"
    "**`dataclasses.replace` is the prettier alternative.** "
    "`return replace(args, **sampled)` returns a NEW args with overrides "
    "applied — immutable and clean — but errors on unknown keys, so the "
    "mutable `update_args` is more robust to wandb's metadata fields."
)

RECAP_TQDM_POSTFIX = (
    "## `pbar.set_postfix(loss=..., ex=...)` — quick refresher\n"
    "\n"
    "`tqdm` is the progress-bar library used in every ARENA training loop. "
    "Wrapping a DataLoader in `tqdm(...)` gives you the iterating bar; "
    "`pbar.set_postfix(**metrics)` then appends key=value pairs to the "
    "RIGHT side of that bar so you can see live metrics scroll while the "
    "loop runs.\n"
    "\n"
    "**The two-step idiom:**\n"
    "\n"
    "```python\n"
    "pbar = tqdm(self.train_loader, desc='Training')\n"
    "for imgs, labels in pbar:\n"
    "    loss = self.training_step(imgs, labels)\n"
    "    pbar.set_postfix(loss=f'{loss:.3f}', ex_seen=self.examples_seen)\n"
    "```\n"
    "\n"
    "**`set_postfix` overwrites previous postfix** — each call REPLACES the "
    "metrics dict, it doesn't accumulate. Pass every metric you want on the "
    "bar in every call.\n"
    "\n"
    "**Format numbers in the call.** `loss=f'{loss:.3f}'` prints `0.412` "
    "not `0.41234567...`. Pass `int(ex_seen)` if you want no decimal "
    "places. tqdm doesn't auto-format floats nicely.\n"
    "\n"
    "**Different from wandb.log.** `set_postfix` updates the LOCAL "
    "terminal bar; `wandb.log` ships metrics to the cloud dashboard. You "
    "typically call both in the same step."
)

RECAP_LOG_SAMPLES = (
    "## Every-K-steps eval callback — quick refresher\n"
    "\n"
    "Many ARENA training loops periodically dump a small batch of model "
    "outputs to a logging sink — sample completions from a language model, "
    "sample generations from a diffusion model, eval-set predictions, etc.\n"
    "\n"
    "**The pattern:**\n"
    "\n"
    "```python\n"
    "for step, batch in enumerate(loader):\n"
    "    loss = train_step(batch)\n"
    "    if step % args.eval_every == 0:\n"
    "        samples = sample_n_from_model(model, n=args.n_eval)\n"
    "        sink.append({'step': step, 'samples': samples})\n"
    "```\n"
    "\n"
    "**Three knobs:**\n"
    "\n"
    "- `eval_every` (K) — how often to fire the callback. K=100 is common "
    "for cheap eval, K=1000 for expensive generation.\n"
    "- `n_eval` (N) — how many samples to log per fire. Keep small (N≤16) "
    "to avoid blowing up wandb storage / slowing training.\n"
    "- `sink` — where the samples go. Can be `wandb.log({'samples': ...})`, "
    "a `wandb.Table`, or a plain Python list (for offline debugging).\n"
    "\n"
    "**The plain-list sink** is what this drill uses — it lets you exercise "
    "the every-K cadence + N-per-fire counting WITHOUT requiring wandb to "
    "be installed. The cadence logic is identical whatever sink you wire in."
)

RECAP_TIME_STAGE = (
    "## `time.perf_counter()` stage instrumentation — quick refresher\n"
    "\n"
    "Profiling a training loop starts with `time.perf_counter()` around "
    "named stages: data-load, forward, backward, optimizer-step, log. "
    "Sum the elapsed seconds per stage across the run and you get the "
    "wall-clock breakdown — usually data-load and forward dominate.\n"
    "\n"
    "**The recipe:**\n"
    "\n"
    "```python\n"
    "stages = {'forward': 0.0, 'backward': 0.0, 'step': 0.0}\n"
    "for batch in loader:\n"
    "    t0 = time.perf_counter()\n"
    "    out = model(batch)\n"
    "    stages['forward'] += time.perf_counter() - t0\n"
    "\n"
    "    t0 = time.perf_counter()\n"
    "    loss.backward()\n"
    "    stages['backward'] += time.perf_counter() - t0\n"
    "    ...\n"
    "```\n"
    "\n"
    "**Why `perf_counter` not `time.time`.** `perf_counter` is monotonic "
    "and has the highest available resolution on each platform; `time.time` "
    "can go backwards when the system clock adjusts and is coarse-grained "
    "on Windows. ALWAYS use `perf_counter` for elapsed-time measurement.\n"
    "\n"
    "**Sum elapsed inside the loop**, don't store individual samples — a "
    "100k-step training run would balloon to gigabytes if you stored every "
    "per-stage delta. Sum (and optionally count) is enough to compute the "
    "mean later.\n"
    "\n"
    "**Subtraction handles overflow gracefully.** `perf_counter` is "
    "monotonic so `t1 - t0` is always >= 0. No clock-skew edge cases."
)


# ---------------------------------------------------------------------------
# Specs.
# ---------------------------------------------------------------------------

SPEC_WANDB_INIT = {
    "atom_id": "wandb-init-run",
    "subtopic": "Logging: wandb.init run",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_WANDB_INIT,
    "exercise_index": 1,
    "exercise_title": "open a wandb run with project, name, and config",
    "slug": "open-a-wandb-run-with-project-name-and-config",
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "difficulty_dots": "🔴🔴⚪⚪⚪",
    "keywords": ["wandb", "init", "config", "mock"],
    "kcs": ["wandb-init-project-name", "wandb-init-config-dict"],
    "lo": (
        "Apply `wandb.init(project=, name=, config=)` to open a single run "
        "from a dataclass-style args object, verified by mocking the "
        "`wandb` module."
    ),
    "prompt_body": (
        "Implement `ex1_open_wandb_run(args)`. The canonical ARENA "
        "wandb-init recipe:\n\n"
        "1. `args` is a simple object with attributes `wandb_project: str`, "
        "`wandb_name: str`, and a few hparams (`lr: float`, `batch_size: "
        "int`, `epochs: int`).\n"
        "2. Call `wandb.init(...)` with EXACTLY these kwargs:\n"
        "   - `project=args.wandb_project`\n"
        "   - `name=args.wandb_name`\n"
        "   - `config=args` itself (wandb accepts dataclasses + plain "
        "objects; it'll snapshot their attributes).\n"
        "3. Return whatever `wandb.init` returned (a run handle in real "
        "wandb; the mock's MagicMock return value here).\n\n"
        "**The test mocks `wandb`.** Colab and the build venv may not have "
        "wandb installed. The test installs a `MagicMock()` into "
        "`sys.modules['wandb']` BEFORE you import it, then inspects "
        "`wandb.init.call_args` to verify the kwargs."
    ),
    "stub": (
        "import sys\n"
        "from unittest.mock import MagicMock\n"
        "sys.modules.setdefault('wandb', MagicMock())\n"
        "import wandb\n"
        "\n"
        "def ex1_open_wandb_run(args):\n"
        '    """Open a wandb run named args.wandb_name in args.wandb_project."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass\n"
        "class FakeArgs:\n"
        "    wandb_project: str = 'arena-resnet'\n"
        "    wandb_name: str = 'baseline-run-3'\n"
        "    lr: float = 1e-3\n"
        "    batch_size: int = 64\n"
        "    epochs: int = 3\n"
        "\n"
        "# Reset the mock before exercising.\n"
        "wandb.init.reset_mock()\n"
        "args = FakeArgs()\n"
        "ret = ex1_open_wandb_run(args)\n"
        "\n"
        "# Exactly one wandb.init call.\n"
        "assert wandb.init.call_count == 1, f'expected 1 wandb.init call, got {wandb.init.call_count}'\n"
        "ca = wandb.init.call_args\n"
        "kwargs = ca.kwargs\n"
        "assert kwargs.get('project') == 'arena-resnet', f'project kwarg wrong: {kwargs.get(\"project\")!r}'\n"
        "assert kwargs.get('name') == 'baseline-run-3', f'name kwarg wrong: {kwargs.get(\"name\")!r}'\n"
        "assert kwargs.get('config') is args, 'config kwarg must be the args object itself (wandb snapshots its attrs)'\n"
        "# Return value comes from wandb.init.\n"
        "assert ret is wandb.init.return_value, 'must return whatever wandb.init returns (the run handle)'\n"
        "\n"
        "# Second run with a different args — fresh call recorded.\n"
        "wandb.init.reset_mock()\n"
        "args2 = FakeArgs(wandb_project='arena-transformer', wandb_name='sweep-run-7', lr=5e-4)\n"
        "ex1_open_wandb_run(args2)\n"
        "assert wandb.init.call_count == 1\n"
        "k2 = wandb.init.call_args.kwargs\n"
        "assert k2['project'] == 'arena-transformer'\n"
        "assert k2['name'] == 'sweep-run-7'\n"
        "assert k2['config'] is args2"
    ),
    "solution_body": (
        "import sys\n"
        "from unittest.mock import MagicMock\n"
        "sys.modules.setdefault('wandb', MagicMock())\n"
        "import wandb\n"
        "\n"
        "def ex1_open_wandb_run(args):\n"
        "    return wandb.init(\n"
        "        project=args.wandb_project,\n"
        "        name=args.wandb_name,\n"
        "        config=args,\n"
        "    )"
    ),
    "solution_notes": (
        "**Why pass `config=args` (the object), not `config=vars(args)`.** "
        "Wandb accepts dataclasses and plain objects with `__dict__` and "
        "snapshots their attributes itself. Passing the live object means "
        "wandb sees the SAME values your training loop uses — there's no "
        "drift between what wandb thinks the hparams are and what `args.lr` "
        "actually is.\n\n"
        "**Reset before re-exercising.** In a Jupyter session, the mock "
        "accumulates calls across cells. `wandb.init.reset_mock()` at the "
        "top of the test resets `call_count` / `call_args` to a known "
        "state — useful when you want to test exactly one invocation.\n\n"
        "**`sys.modules.setdefault` not `=`.** If a real wandb is installed "
        "later, `setdefault` won't clobber it. The mock only fills the gap "
        "when wandb is absent."
    ),
    "extra_imports": [],
}


SPEC_WANDB_FINISH = {
    "atom_id": "wandb-finish",
    "subtopic": "Logging: wandb.finish",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_WANDB_FINISH,
    "exercise_index": 1,
    "exercise_title": "close a wandb run at the end of train",
    "slug": "close-a-wandb-run-at-the-end-of-train",
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "difficulty_dots": "🔴🔴⚪⚪⚪",
    "keywords": ["wandb", "finish", "lifecycle", "mock"],
    "kcs": ["wandb-finish-after-train", "wandb-init-finish-pair"],
    "lo": (
        "Apply `wandb.init` + `wandb.finish` as a symmetric pair around a "
        "fake training loop, verified by mocking the `wandb` module."
    ),
    "prompt_body": (
        "Implement `ex1_train_with_wandb_lifecycle(args, n_steps)`. A "
        "complete (mocked) train function that:\n\n"
        "1. Opens a wandb run with `wandb.init(project=args.wandb_project, "
        "name=args.wandb_name)`. (No `config` kwarg this time — keep the "
        "signature minimal.)\n"
        "2. Loops `for step in range(n_steps)` doing nothing inside (this "
        "drill is only about the lifecycle).\n"
        "3. Calls `wandb.finish()` AFTER the loop.\n"
        "4. Returns `n_steps`.\n\n"
        "The test asserts the CALL ORDER: `wandb.init` first, then "
        "`wandb.finish`, exactly one of each."
    ),
    "stub": (
        "import sys\n"
        "from unittest.mock import MagicMock\n"
        "sys.modules.setdefault('wandb', MagicMock())\n"
        "import wandb\n"
        "\n"
        "def ex1_train_with_wandb_lifecycle(args, n_steps: int) -> int:\n"
        '    """Open wandb run, fake-train n_steps, close run, return n_steps."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "from dataclasses import dataclass\n"
        "from unittest.mock import call\n"
        "\n"
        "@dataclass\n"
        "class FakeArgs:\n"
        "    wandb_project: str = 'arena-transformer'\n"
        "    wandb_name: str = 'lifecycle-test'\n"
        "\n"
        "# Build a parent mock so we can inspect call ORDER across init+finish.\n"
        "parent = MagicMock()\n"
        "parent.attach_mock(wandb.init, 'init')\n"
        "parent.attach_mock(wandb.finish, 'finish')\n"
        "wandb.init.reset_mock(); wandb.finish.reset_mock(); parent.reset_mock()\n"
        "parent.attach_mock(wandb.init, 'init')\n"
        "parent.attach_mock(wandb.finish, 'finish')\n"
        "\n"
        "args = FakeArgs()\n"
        "out = ex1_train_with_wandb_lifecycle(args, n_steps=5)\n"
        "assert out == 5, f'must return n_steps, got {out!r}'\n"
        "\n"
        "# init called exactly once, with project + name.\n"
        "assert wandb.init.call_count == 1, f'expected 1 wandb.init, got {wandb.init.call_count}'\n"
        "k = wandb.init.call_args.kwargs\n"
        "assert k.get('project') == 'arena-transformer'\n"
        "assert k.get('name') == 'lifecycle-test'\n"
        "# finish called exactly once.\n"
        "assert wandb.finish.call_count == 1, f'expected 1 wandb.finish, got {wandb.finish.call_count}'\n"
        "\n"
        "# Order: init BEFORE finish.\n"
        "names = [c[0] for c in parent.mock_calls]\n"
        "assert 'init' in names and 'finish' in names, f'missing calls: {names}'\n"
        "assert names.index('init') < names.index('finish'), (\n"
        "    f'wandb.init must be called BEFORE wandb.finish, got order: {names}'\n"
        ")\n"
        "\n"
        "# Zero-step run still pairs init + finish.\n"
        "wandb.init.reset_mock(); wandb.finish.reset_mock()\n"
        "ex1_train_with_wandb_lifecycle(args, n_steps=0)\n"
        "assert wandb.init.call_count == 1\n"
        "assert wandb.finish.call_count == 1"
    ),
    "solution_body": (
        "import sys\n"
        "from unittest.mock import MagicMock\n"
        "sys.modules.setdefault('wandb', MagicMock())\n"
        "import wandb\n"
        "\n"
        "def ex1_train_with_wandb_lifecycle(args, n_steps: int) -> int:\n"
        "    wandb.init(project=args.wandb_project, name=args.wandb_name)\n"
        "    for step in range(n_steps):\n"
        "        pass  # fake training step\n"
        "    wandb.finish()\n"
        "    return n_steps"
    ),
    "solution_notes": (
        "**Why ORDER matters.** `wandb.finish()` flushes the buffer for "
        "the run opened by `wandb.init()`. Calling them out of order "
        "(or twice in a row) confuses the wandb sync thread. The test "
        "uses `parent.attach_mock` to record both calls into a single "
        "ordered list so we can assert `init` happened before `finish`.\n\n"
        "**Zero-step edge case.** A run that does no training still needs "
        "the init/finish pair — wandb's run lifecycle is independent of "
        "what happens between. Skipping `finish` when `n_steps == 0` "
        "leaves a stale run on the dashboard.\n\n"
        "**No try/finally here.** ARENA's default loop puts `finish` at "
        "the bare end of `train()` — an exception mid-loop will skip it. "
        "In production you'd wrap with try/finally; the drill keeps the "
        "ARENA pattern faithful."
    ),
    "extra_imports": [],
}


SPEC_WANDB_LOG_STEP = {
    "atom_id": "wandb-log-step",
    "subtopic": "Logging: wandb.log step",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_WANDB_LOG_STEP,
    "exercise_index": 1,
    "exercise_title": "log loss + examples_seen step over a fake epoch",
    "slug": "log-loss-and-examples-seen-step-over-a-fake-epoch",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["wandb", "log", "examples-seen", "step-axis", "mock"],
    "kcs": ["wandb-log-metrics-dict", "wandb-log-step-kwarg"],
    "lo": (
        "Apply `wandb.log({...}, step=examples_seen)` inside a fake "
        "training loop with an examples-seen counter, verified by mocking "
        "the `wandb` module and inspecting every recorded call."
    ),
    "prompt_body": (
        "Implement `ex1_fake_epoch_with_wandb_log(losses, batch_size)`. "
        "The canonical ARENA inner-loop logging recipe:\n\n"
        "1. `losses` is a Python list of floats — one per fake training "
        "step.\n"
        "2. `batch_size` is an int — examples per step.\n"
        "3. Maintain a running `examples_seen` counter, starting at 0.\n"
        "4. For each `loss` in `losses`:\n"
        "   - Increment `examples_seen` by `batch_size` BEFORE logging.\n"
        "   - Call `wandb.log({'loss': loss}, step=examples_seen)`.\n"
        "5. Return the final `examples_seen` value.\n\n"
        "The test inspects `wandb.log.call_args_list` to check every call's "
        "step value lines up with the cumulative examples-seen."
    ),
    "stub": (
        "import sys\n"
        "from unittest.mock import MagicMock\n"
        "sys.modules.setdefault('wandb', MagicMock())\n"
        "import wandb\n"
        "\n"
        "def ex1_fake_epoch_with_wandb_log(losses: list, batch_size: int) -> int:\n"
        '    """Log each loss with step=cumulative examples_seen. Return final examples_seen."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "wandb.log.reset_mock()\n"
        "losses = [0.9, 0.8, 0.7, 0.6, 0.5]\n"
        "batch_size = 32\n"
        "final = ex1_fake_epoch_with_wandb_log(losses, batch_size)\n"
        "\n"
        "# Final examples_seen = len(losses) * batch_size.\n"
        "assert final == 5 * 32, f'expected examples_seen=160, got {final}'\n"
        "\n"
        "# One wandb.log call per loss.\n"
        "calls = wandb.log.call_args_list\n"
        "assert len(calls) == len(losses), f'expected {len(losses)} wandb.log calls, got {len(calls)}'\n"
        "\n"
        "# Each call: positional metrics dict, step kwarg = cumulative.\n"
        "for i, (loss, c) in enumerate(zip(losses, calls)):\n"
        "    args_pos, kwargs = c.args, c.kwargs\n"
        "    # Metrics may be positional or kwarg; accept both.\n"
        "    metrics = args_pos[0] if args_pos else kwargs.get('metrics') or kwargs.get('data')\n"
        "    assert isinstance(metrics, dict), f'call {i}: first arg must be dict, got {type(metrics)}'\n"
        "    assert 'loss' in metrics, f'call {i}: metrics dict missing \"loss\" key'\n"
        "    assert abs(metrics['loss'] - loss) < 1e-9, f'call {i}: loss mismatch'\n"
        "    expected_step = (i + 1) * batch_size\n"
        "    assert kwargs.get('step') == expected_step, (\n"
        "        f'call {i}: step kwarg wrong — expected {expected_step}, got {kwargs.get(\"step\")}'\n"
        "    )\n"
        "\n"
        "# Different batch size → step values rescale.\n"
        "wandb.log.reset_mock()\n"
        "final2 = ex1_fake_epoch_with_wandb_log([0.1, 0.2], batch_size=128)\n"
        "assert final2 == 256\n"
        "steps2 = [c.kwargs.get('step') for c in wandb.log.call_args_list]\n"
        "assert steps2 == [128, 256], f'batch=128 should produce steps [128, 256], got {steps2}'"
    ),
    "solution_body": (
        "import sys\n"
        "from unittest.mock import MagicMock\n"
        "sys.modules.setdefault('wandb', MagicMock())\n"
        "import wandb\n"
        "\n"
        "def ex1_fake_epoch_with_wandb_log(losses, batch_size):\n"
        "    examples_seen = 0\n"
        "    for loss in losses:\n"
        "        examples_seen += batch_size\n"
        "        wandb.log({'loss': loss}, step=examples_seen)\n"
        "    return examples_seen"
    ),
    "solution_notes": (
        "**Why increment BEFORE logging.** Step 1's log call should show "
        "step=batch_size, not step=0 (that would imply the model hasn't "
        "trained yet). The semantics are 'after this batch of "
        "batch_size examples, the loss is X'.\n\n"
        "**Why `examples_seen`, not `step_idx`.** A run with batch=32 vs "
        "batch=256 spends very different wall-clock per step. Plotting on "
        "the examples-seen axis lets you overlay the two and see which "
        "configuration converges faster per example. Step-idx would have "
        "the batch=32 run reach step 1000 in 1/8th the wall-clock.\n\n"
        "**The metrics dict is the FIRST positional arg.** Wandb's API "
        "accepts both `wandb.log(d, step=n)` and `wandb.log(data=d, "
        "step=n)`; ARENA uses positional. The test tolerates both."
    ),
    "extra_imports": [],
}


SPEC_WANDB_WATCH = {
    "atom_id": "wandb-watch-model",
    "subtopic": "Logging: wandb.watch model",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_WANDB_WATCH,
    "exercise_index": 1,
    "exercise_title": "watch the trainable head with parameter+gradient logging",
    "slug": "watch-the-trainable-head-with-parameter-and-gradient-logging",
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "difficulty_dots": "🔴🔴⚪⚪⚪",
    "keywords": ["wandb", "watch", "histogram", "mock"],
    "kcs": ["wandb-watch-target-module", "wandb-watch-log-all-freq"],
    "lo": (
        "Apply `wandb.watch(module, log='all', log_freq=K)` to a specific "
        "submodule (the trainable head) inside a fake setup function, "
        "verified by mocking the `wandb` module."
    ),
    "prompt_body": (
        "Implement `ex1_attach_wandb_watch(model, log_freq)`. The canonical "
        "ARENA `pre_training_setup` line for histogram logging:\n\n"
        "1. `model` is an `nn.Module` with a `.out_layers` "
        "`nn.ModuleList`. The HEAD is `model.out_layers[-1]` — that's the "
        "module we want to watch (not the whole model — too noisy).\n"
        "2. Call `wandb.watch(...)` with EXACTLY these args:\n"
        "   - First positional: `model.out_layers[-1]` (the head module).\n"
        "   - `log='all'` (log both parameters AND gradients).\n"
        "   - `log_freq=log_freq` (passed through as a kwarg).\n"
        "3. Return the head module that was watched (for the caller to "
        "sanity-check).\n\n"
        "The test mocks wandb and inspects `wandb.watch.call_args` to "
        "verify the exact module reference and the kwargs."
    ),
    "stub": (
        "import sys\n"
        "from unittest.mock import MagicMock\n"
        "sys.modules.setdefault('wandb', MagicMock())\n"
        "import wandb\n"
        "import torch as t\n"
        "\n"
        "def ex1_attach_wandb_watch(model: t.nn.Module, log_freq: int) -> t.nn.Module:\n"
        '    """Wandb.watch the head (out_layers[-1]) with log=all and log_freq=K."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "class FakeResNet(t.nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.body = t.nn.Linear(8, 16)\n"
        "        self.out_layers = t.nn.ModuleList([\n"
        "            t.nn.Linear(16, 32),\n"
        "            t.nn.Linear(32, 10),\n"
        "        ])\n"
        "\n"
        "wandb.watch.reset_mock()\n"
        "model = FakeResNet()\n"
        "head = ex1_attach_wandb_watch(model, log_freq=50)\n"
        "\n"
        "# Returned head is the actual final layer.\n"
        "assert head is model.out_layers[-1], 'must return model.out_layers[-1]'\n"
        "\n"
        "# Exactly one wandb.watch call.\n"
        "assert wandb.watch.call_count == 1, f'expected 1 wandb.watch call, got {wandb.watch.call_count}'\n"
        "ca = wandb.watch.call_args\n"
        "\n"
        "# First positional arg must BE the head module (identity, not equality).\n"
        "assert len(ca.args) >= 1, f'wandb.watch must be called with the module as positional arg, got args={ca.args!r}'\n"
        "assert ca.args[0] is model.out_layers[-1], (\n"
        "    'first arg must be model.out_layers[-1] (the head), not the whole model'\n"
        ")\n"
        "\n"
        "# log kwarg must be the string \"all\".\n"
        "assert ca.kwargs.get('log') == 'all', f'log kwarg must be \"all\", got {ca.kwargs.get(\"log\")!r}'\n"
        "# log_freq kwarg must equal what we passed in.\n"
        "assert ca.kwargs.get('log_freq') == 50, f'log_freq must be 50, got {ca.kwargs.get(\"log_freq\")!r}'\n"
        "\n"
        "# Different log_freq propagates through.\n"
        "wandb.watch.reset_mock()\n"
        "ex1_attach_wandb_watch(FakeResNet(), log_freq=200)\n"
        "assert wandb.watch.call_args.kwargs.get('log_freq') == 200"
    ),
    "solution_body": (
        "import sys\n"
        "from unittest.mock import MagicMock\n"
        "sys.modules.setdefault('wandb', MagicMock())\n"
        "import wandb\n"
        "import torch as t\n"
        "\n"
        "def ex1_attach_wandb_watch(model, log_freq):\n"
        "    head = model.out_layers[-1]\n"
        "    wandb.watch(head, log='all', log_freq=log_freq)\n"
        "    return head"
    ),
    "solution_notes": (
        "**Why watch the head, not the whole model.** ARENA's resnet "
        "fine-tune freezes everything except the final classifier. "
        "`wandb.watch(self.model, ...)` would attach hooks to every "
        "frozen layer — they'd record zero-gradient histograms and waste "
        "dashboard real estate. Watching just `out_layers[-1]` keeps the "
        "histograms relevant.\n\n"
        "**`log='all'` is the kitchen sink.** Wandb supports `'parameters'` "
        "(weights only), `'gradients'` (gradients only), or `'all'` (both). "
        "ARENA picks `'all'` — for fine-tuning you want to see both the "
        "weights drift AND the gradient magnitudes evolve.\n\n"
        "**`log_freq` units = forward passes, not training steps.** They "
        "happen to coincide in supervised training (one forward per step), "
        "but if you have an inner-loop training algorithm with multiple "
        "forwards per step, plan accordingly."
    ),
    "extra_imports": [],
}


SPEC_WANDB_CONFIG_ARGS = {
    "atom_id": "wandb-config-into-args",
    "subtopic": "Logging: wandb.config into args",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_WANDB_CONFIG_ARGS,
    "exercise_index": 1,
    "exercise_title": "overwrite dataclass args from a wandb sweep config",
    "slug": "overwrite-dataclass-args-from-a-wandb-sweep-config",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["wandb", "sweep", "config", "dataclass", "args"],
    "kcs": ["wandb-config-overwrite-args", "wandb-config-ignore-unknown"],
    "lo": (
        "Apply `setattr(args, k, v)` over a `wandb.config`-style dict to "
        "overwrite hparams on a dataclass `args` instance, skipping keys "
        "the dataclass doesn't define."
    ),
    "prompt_body": (
        "Implement `ex1_update_args_from_wandb(args, sampled_config)`. The "
        "canonical ARENA 0_3_8 sweep recipe:\n\n"
        "1. `args` is a dataclass instance with hparam fields (`lr`, "
        "`batch_size`, `weight_decay_bool`, plus an unrelated field "
        "`wandb_project`).\n"
        "2. `sampled_config` is a plain dict (what `dict(wandb.config)` "
        "returns inside `train()`): may contain hparams to OVERWRITE on "
        "args, may contain UNKNOWN keys (wandb metadata like `'_wandb'`) "
        "that you should SILENTLY SKIP.\n"
        "3. For each `(k, v)` in `sampled_config.items()`:\n"
        "   - If `args` has an attribute `k` (use `hasattr`), set "
        "`setattr(args, k, v)`.\n"
        "   - Otherwise, skip (don't error).\n"
        "4. Return the (mutated) `args`.\n\n"
        "This is `update_args` from ARENA's sweep cells."
    ),
    "stub": (
        "def ex1_update_args_from_wandb(args, sampled_config: dict):\n"
        '    """Overwrite args fields from sampled_config; skip unknown keys."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "from dataclasses import dataclass\n"
        "\n"
        "@dataclass\n"
        "class FakeArgs:\n"
        "    lr: float = 1e-3\n"
        "    batch_size: int = 64\n"
        "    weight_decay_bool: bool = False\n"
        "    wandb_project: str = 'arena-resnet'\n"
        "\n"
        "# Case 1 — overwrite two known fields, ignore one unknown.\n"
        "args = FakeArgs()\n"
        "sampled = {\n"
        "    'lr': 5e-4,\n"
        "    'batch_size': 128,\n"
        "    '_wandb': {'cli_version': '0.17.0'},  # metadata — must skip\n"
        "    'completely_unknown_field': 'should-not-appear',\n"
        "}\n"
        "out = ex1_update_args_from_wandb(args, sampled)\n"
        "\n"
        "# Returned args must be the same instance (mutation).\n"
        "assert out is args, 'must mutate and return the same args instance'\n"
        "assert args.lr == 5e-4, f'lr must overwrite to 5e-4, got {args.lr}'\n"
        "assert args.batch_size == 128, f'batch_size must overwrite to 128, got {args.batch_size}'\n"
        "# Untouched defaults stay.\n"
        "assert args.weight_decay_bool is False, 'untouched field must stay default'\n"
        "assert args.wandb_project == 'arena-resnet'\n"
        "# Unknown keys must NOT be set as new attrs.\n"
        "assert not hasattr(args, '_wandb'), '_wandb metadata key must be skipped, not set as attr'\n"
        "assert not hasattr(args, 'completely_unknown_field'), 'unknown sweep keys must be skipped'\n"
        "\n"
        "# Case 2 — empty config → no changes.\n"
        "args2 = FakeArgs(lr=2e-3)\n"
        "ex1_update_args_from_wandb(args2, {})\n"
        "assert args2.lr == 2e-3\n"
        "assert args2.batch_size == 64\n"
        "\n"
        "# Case 3 — bool field overwrite (truthy / falsy edge).\n"
        "args3 = FakeArgs()\n"
        "ex1_update_args_from_wandb(args3, {'weight_decay_bool': True})\n"
        "assert args3.weight_decay_bool is True\n"
        "ex1_update_args_from_wandb(args3, {'weight_decay_bool': False})\n"
        "assert args3.weight_decay_bool is False"
    ),
    "solution_body": (
        "def ex1_update_args_from_wandb(args, sampled_config):\n"
        "    for k, v in sampled_config.items():\n"
        "        if hasattr(args, k):\n"
        "            setattr(args, k, v)\n"
        "    return args"
    ),
    "solution_notes": (
        "**Why `hasattr` not `k in args.__dataclass_fields__`.** "
        "Both work for dataclasses, but `hasattr` also handles plain "
        "classes and inherited fields without special-casing. It's the "
        "most permissive check — exactly what you want for a defensive "
        "config-overwrite helper.\n\n"
        "**Why SKIP unknown keys instead of raising.** Wandb sweeps add "
        "metadata keys like `_wandb` and `_runtime` automatically. If you "
        "raise on unknown keys, the sweep crashes on every run. Skipping "
        "is the conservative default.\n\n"
        "**Mutation vs `dataclasses.replace`.** ARENA's example uses "
        "mutation because `update_args` is called inside `train()` and "
        "the args object is local. `dataclasses.replace(args, **sampled)` "
        "returns a NEW args — cleaner, but errors on unknown keys. The "
        "mutable form is what ARENA picks for robustness."
    ),
    "extra_imports": [],
}


SPEC_TQDM_POSTFIX = {
    "atom_id": "tqdm-postfix-metrics",
    "subtopic": "Logging: tqdm postfix metrics",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_TQDM_POSTFIX,
    "exercise_index": 1,
    "exercise_title": "set live loss + examples_seen on a tqdm progress bar",
    "slug": "set-live-loss-and-examples-seen-on-a-tqdm-progress-bar",
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "difficulty_dots": "🔴🔴⚪⚪⚪",
    "keywords": ["tqdm", "postfix", "progress-bar", "live-metrics"],
    "kcs": ["tqdm-wrap-iterable", "tqdm-set-postfix-kwargs"],
    "lo": (
        "Apply `tqdm(iterable, desc=...)` + `pbar.set_postfix(**metrics)` "
        "to display live per-step metrics on a progress bar while "
        "iterating a fake DataLoader."
    ),
    "prompt_body": (
        "Implement `ex1_run_with_tqdm_postfix(losses, batch_size)`. The "
        "canonical ARENA inner-loop progress-bar idiom (without wandb):\n\n"
        "1. Wrap `enumerate(losses)` in `tqdm(...)` with `desc='Training'`. "
        "Capture the wrapped object — you need it to call `set_postfix`.\n"
        "2. For each `(step, loss)` pair from the wrapped iterator:\n"
        "   - Accumulate `examples_seen += batch_size`.\n"
        "   - Call `pbar.set_postfix(loss=f'{loss:.3f}', "
        "examples_seen=examples_seen)`.\n"
        "3. Return the list of postfix dicts that were set "
        "(`pbar.postfix` is set, but tqdm stores it differently across "
        "versions — record it yourself into a sidecar list so the test "
        "can inspect).\n\n"
        "**Wrapping `enumerate(losses)`:** `for step, loss in tqdm(...)` "
        "is exactly what ARENA writes. Don't fight it — wrap the "
        "enumerate, not the bare losses."
    ),
    "stub": (
        "from tqdm import tqdm\n"
        "\n"
        "def ex1_run_with_tqdm_postfix(losses: list, batch_size: int) -> list:\n"
        '    """Iterate losses with a tqdm bar, set per-step postfix. Return list of postfix dicts."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "losses = [0.91, 0.75, 0.42, 0.18, 0.06]\n"
        "batch_size = 64\n"
        "postfix_log = ex1_run_with_tqdm_postfix(losses, batch_size)\n"
        "\n"
        "# One postfix entry per loss.\n"
        "assert len(postfix_log) == len(losses), (\n"
        "    f'expected {len(losses)} postfix entries, got {len(postfix_log)}'\n"
        ")\n"
        "\n"
        "# Each entry is a dict with 'loss' (formatted str) and 'examples_seen' (int).\n"
        "for i, (loss, pf) in enumerate(zip(losses, postfix_log)):\n"
        "    assert isinstance(pf, dict), f'entry {i} not a dict'\n"
        "    assert 'loss' in pf, f'entry {i} missing \"loss\" key'\n"
        "    assert 'examples_seen' in pf, f'entry {i} missing \"examples_seen\" key'\n"
        "    # loss formatted as .3f.\n"
        "    expected_loss_str = f'{loss:.3f}'\n"
        "    assert pf['loss'] == expected_loss_str, (\n"
        "        f'entry {i}: loss must be formatted as .3f — expected {expected_loss_str!r}, got {pf[\"loss\"]!r}'\n"
        "    )\n"
        "    # examples_seen monotone increasing by batch_size each step.\n"
        "    expected_ex = (i + 1) * batch_size\n"
        "    assert pf['examples_seen'] == expected_ex, (\n"
        "        f'entry {i}: examples_seen must be {expected_ex}, got {pf[\"examples_seen\"]}'\n"
        "    )\n"
        "\n"
        "# Empty losses → no postfix, no error.\n"
        "assert ex1_run_with_tqdm_postfix([], 32) == []"
    ),
    "solution_body": (
        "from tqdm import tqdm\n"
        "\n"
        "def ex1_run_with_tqdm_postfix(losses, batch_size):\n"
        "    examples_seen = 0\n"
        "    postfix_log = []\n"
        "    pbar = tqdm(enumerate(losses), desc='Training', total=len(losses))\n"
        "    for step, loss in pbar:\n"
        "        examples_seen += batch_size\n"
        "        pf = dict(loss=f'{loss:.3f}', examples_seen=examples_seen)\n"
        "        pbar.set_postfix(**pf)\n"
        "        postfix_log.append(pf)\n"
        "    return postfix_log"
    ),
    "solution_notes": (
        "**Why record `postfix_log` separately.** tqdm's `pbar.postfix` "
        "attribute stores the LAST postfix you set, not the history. "
        "For the drill we record a sidecar list so the test can inspect "
        "every step's metrics — in real ARENA code you don't need this "
        "because wandb.log captures the history.\n\n"
        "**Why `total=len(losses)`.** When you wrap `enumerate(losses)`, "
        "tqdm can't infer the length (enumerate is a generator). Passing "
        "`total=` makes the bar show '5/100' not '5it'. Optional for "
        "correctness, mandatory for nice UX.\n\n"
        "**`f'{loss:.3f}'` not `loss`.** Pass a STRING to `set_postfix` "
        "when you want exact decimal control. Passing a float lets tqdm "
        "auto-format, which produces inconsistent precision (depends on "
        "the magnitude). ARENA always pre-formats."
    ),
    "extra_imports": [],
}


SPEC_LOG_SAMPLES = {
    "atom_id": "log-samples-eval-callback",
    "subtopic": "Logging: log-samples eval callback",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_LOG_SAMPLES,
    "exercise_index": 1,
    "exercise_title": "every-K-steps eval callback that logs N samples to a sink",
    "slug": "every-k-steps-eval-callback-that-logs-n-samples-to-a-sink",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["eval", "callback", "cadence", "sink"],
    "kcs": ["modulo-k-step-cadence", "n-sample-dump-to-sink"],
    "lo": (
        "Apply the every-K-steps callback pattern to dump N "
        "deterministically-generated model outputs to a plain-list sink, "
        "verified by checking cadence + per-fire sample count."
    ),
    "prompt_body": (
        "Implement `ex1_run_with_sample_callback(n_steps, eval_every, "
        "n_eval, sink)`. The canonical 'log samples every K steps' "
        "training-loop pattern (with a plain-list sink so we don't need "
        "wandb):\n\n"
        "1. Loop `for step in range(n_steps)`.\n"
        "2. At every step where `step % eval_every == 0` (so step 0, K, "
        "2K, ...):\n"
        "   - Generate `n_eval` fake samples — use "
        "`[f'step={step}-sample={i}' for i in range(n_eval)]`.\n"
        "   - Append `{'step': step, 'samples': samples}` to `sink`.\n"
        "3. Return the total number of `sink.append` calls.\n\n"
        "**The sink is a plain Python list.** No wandb dependency. In "
        "real ARENA code you'd swap `sink.append(d)` for `wandb.log(d, "
        "step=step)` — same cadence logic, different destination.\n\n"
        "**Step 0 fires.** `0 % K == 0` for any K, so the initial random "
        "model gets a baseline sample dump before training begins. This "
        "is the ARENA convention."
    ),
    "stub": (
        "def ex1_run_with_sample_callback(n_steps: int, eval_every: int, n_eval: int, sink: list) -> int:\n"
        '    """Fire callback at step 0, K, 2K, ...; each fire appends {step, samples} to sink."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "# Case 1 — K=3, n=2, 10 steps → fires at 0, 3, 6, 9 (4 fires).\n"
        "sink = []\n"
        "n_fires = ex1_run_with_sample_callback(n_steps=10, eval_every=3, n_eval=2, sink=sink)\n"
        "assert n_fires == 4, f'expected 4 fires (steps 0,3,6,9), got {n_fires}'\n"
        "assert len(sink) == 4, f'sink length must equal n_fires, got {len(sink)}'\n"
        "\n"
        "# Each entry: step + samples; samples is a list of length n_eval.\n"
        "for entry, expected_step in zip(sink, [0, 3, 6, 9]):\n"
        "    assert entry['step'] == expected_step, f'step mismatch: {entry[\"step\"]} vs {expected_step}'\n"
        "    assert len(entry['samples']) == 2, f'expected 2 samples per fire, got {len(entry[\"samples\"])}'\n"
        "    # Sample format is deterministic.\n"
        "    assert entry['samples'][0] == f'step={expected_step}-sample=0'\n"
        "    assert entry['samples'][1] == f'step={expected_step}-sample=1'\n"
        "\n"
        "# Case 2 — K=5, n=4, 12 steps → fires at 0, 5, 10 (3 fires).\n"
        "sink2 = []\n"
        "n2 = ex1_run_with_sample_callback(n_steps=12, eval_every=5, n_eval=4, sink=sink2)\n"
        "assert n2 == 3\n"
        "assert [e['step'] for e in sink2] == [0, 5, 10]\n"
        "assert all(len(e['samples']) == 4 for e in sink2)\n"
        "\n"
        "# Case 3 — K=100, 5 steps → only step 0 fires.\n"
        "sink3 = []\n"
        "n3 = ex1_run_with_sample_callback(n_steps=5, eval_every=100, n_eval=1, sink=sink3)\n"
        "assert n3 == 1\n"
        "assert sink3[0]['step'] == 0\n"
        "\n"
        "# Case 4 — n_steps=0 → no fires.\n"
        "sink4 = []\n"
        "n4 = ex1_run_with_sample_callback(n_steps=0, eval_every=3, n_eval=2, sink=sink4)\n"
        "assert n4 == 0\n"
        "assert sink4 == []"
    ),
    "solution_body": (
        "def ex1_run_with_sample_callback(n_steps, eval_every, n_eval, sink):\n"
        "    n_fires = 0\n"
        "    for step in range(n_steps):\n"
        "        if step % eval_every == 0:\n"
        "            samples = [f'step={step}-sample={i}' for i in range(n_eval)]\n"
        "            sink.append({'step': step, 'samples': samples})\n"
        "            n_fires += 1\n"
        "    return n_fires"
    ),
    "solution_notes": (
        "**`step % K == 0` includes step 0.** The first iteration of the "
        "loop, BEFORE any training has happened, captures the "
        "random-init baseline. Skip step 0 if you specifically want "
        "trained-only samples — but ARENA's convention is to include "
        "it.\n\n"
        "**Why a plain list sink.** In ARENA's real loop you'd write "
        "`wandb.log({'samples': wandb.Table(data=...)}, step=step)` here. "
        "By swapping in a list we exercise the SAME cadence logic "
        "(`step % K == 0`) and the SAME N-per-fire counting without "
        "needing wandb installed. The cadence is the load-bearing "
        "primitive; the destination is interchangeable.\n\n"
        "**`n_steps=0` is a legitimate edge case.** Tests that "
        "construct a model and immediately exit (e.g. smoke tests) "
        "should produce zero samples, not crash."
    ),
    "extra_imports": [],
}


SPEC_TIME_STAGE = {
    "atom_id": "time-stage-instrumentation",
    "subtopic": "Logging: time-stage instrumentation",
    "topic_folder": TOPIC,
    "atom_recap_md": RECAP_TIME_STAGE,
    "exercise_index": 1,
    "exercise_title": "accumulate per-stage seconds with time.perf_counter",
    "slug": "accumulate-per-stage-seconds-with-time-perf-counter",
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "difficulty_dots": "🔴🔴🔴⚪⚪",
    "keywords": ["time", "perf_counter", "profile", "stages"],
    "kcs": ["perf-counter-elapsed", "per-stage-sum-accumulator"],
    "lo": (
        "Apply `time.perf_counter()` deltas to accumulate elapsed seconds "
        "per named stage across a loop, returning the per-stage totals "
        "with measured order-of-magnitude correct."
    ),
    "prompt_body": (
        "Implement `ex1_time_three_stages(n_iters, sleep_forward, "
        "sleep_backward, sleep_step)`. The canonical per-stage profiling "
        "recipe:\n\n"
        "1. Initialize `stages = {'forward': 0.0, 'backward': 0.0, "
        "'step': 0.0}`.\n"
        "2. Loop `for _ in range(n_iters)`:\n"
        "   - Time a `time.sleep(sleep_forward)` block; ADD the elapsed "
        "to `stages['forward']`.\n"
        "   - Time a `time.sleep(sleep_backward)` block; add to "
        "`stages['backward']`.\n"
        "   - Time a `time.sleep(sleep_step)` block; add to "
        "`stages['step']`.\n"
        "3. Use `time.perf_counter()` for both endpoints of each timed "
        "block. NOT `time.time()`.\n"
        "4. Return `stages`.\n\n"
        "The test asserts (a) each stage's total is at least "
        "`n_iters * sleep_X` (sleep + measurement overhead means actual "
        "is always >= nominal), (b) the totals are within a generous "
        "upper bound, and (c) the relative ORDER of magnitudes matches "
        "the input sleeps."
    ),
    "stub": (
        "import time\n"
        "\n"
        "def ex1_time_three_stages(n_iters: int,\n"
        "                          sleep_forward: float,\n"
        "                          sleep_backward: float,\n"
        "                          sleep_step: float) -> dict:\n"
        '    """Accumulate per-stage elapsed seconds across n_iters iterations."""\n'
        "    raise NotImplementedError()"
    ),
    "test_body": (
        "import time\n"
        "\n"
        "# Modest sleeps to keep verify fast: 5ms / 10ms / 2ms × 4 iters.\n"
        "stages = ex1_time_three_stages(\n"
        "    n_iters=4,\n"
        "    sleep_forward=0.005,\n"
        "    sleep_backward=0.010,\n"
        "    sleep_step=0.002,\n"
        ")\n"
        "\n"
        "# Three keys, all present.\n"
        "assert set(stages.keys()) == {'forward', 'backward', 'step'}, (\n"
        "    f'expected keys forward/backward/step, got {set(stages.keys())}'\n"
        "    )\n"
        "\n"
        "# Lower bound: each total >= nominal sleep × n_iters (perf_counter\n"
        "# can't measure LESS than the actual sleep).\n"
        "assert stages['forward']  >= 4 * 0.005 - 1e-4, f'forward total too low: {stages[\"forward\"]}'\n"
        "assert stages['backward'] >= 4 * 0.010 - 1e-4, f'backward total too low: {stages[\"backward\"]}'\n"
        "assert stages['step']     >= 4 * 0.002 - 1e-4, f'step total too low: {stages[\"step\"]}'\n"
        "\n"
        "# Upper bound: each total < nominal × 50 (generous — accounts for\n"
        "# CI jitter, GC pauses, etc; even at 100x overhead these stay loose).\n"
        "assert stages['forward']  < 50 * 4 * 0.005, f'forward total absurdly high: {stages[\"forward\"]}'\n"
        "assert stages['backward'] < 50 * 4 * 0.010, f'backward total absurdly high: {stages[\"backward\"]}'\n"
        "assert stages['step']     < 50 * 4 * 0.002, f'step total absurdly high: {stages[\"step\"]}'\n"
        "\n"
        "# Ordering: backward >> step (5x ratio of inputs; expect 3x+ ratio measured).\n"
        "assert stages['backward'] > stages['step'] * 2, (\n"
        "    f'backward should dominate step (ratio expected >=2x), got '\n"
        "    f'backward={stages[\"backward\"]:.4f} step={stages[\"step\"]:.4f}'\n"
        ")\n"
        "\n"
        "# Zero iterations → all stages stay at 0.0 exactly.\n"
        "z = ex1_time_three_stages(n_iters=0, sleep_forward=0.01, sleep_backward=0.01, sleep_step=0.01)\n"
        "assert z == {'forward': 0.0, 'backward': 0.0, 'step': 0.0}"
    ),
    "solution_body": (
        "import time\n"
        "\n"
        "def ex1_time_three_stages(n_iters, sleep_forward, sleep_backward, sleep_step):\n"
        "    stages = {'forward': 0.0, 'backward': 0.0, 'step': 0.0}\n"
        "    for _ in range(n_iters):\n"
        "        t0 = time.perf_counter()\n"
        "        time.sleep(sleep_forward)\n"
        "        stages['forward'] += time.perf_counter() - t0\n"
        "\n"
        "        t0 = time.perf_counter()\n"
        "        time.sleep(sleep_backward)\n"
        "        stages['backward'] += time.perf_counter() - t0\n"
        "\n"
        "        t0 = time.perf_counter()\n"
        "        time.sleep(sleep_step)\n"
        "        stages['step'] += time.perf_counter() - t0\n"
        "    return stages"
    ),
    "solution_notes": (
        "**`perf_counter` for elapsed, `time.time` for wall clock.** "
        "`perf_counter` is monotonic and high-resolution; `time.time` "
        "can jump (NTP adjustment, daylight savings) and is "
        "coarse-grained on Windows. ALWAYS use `perf_counter` for "
        "elapsed-time profiling.\n\n"
        "**Why ACCUMULATE not store every delta.** A 100k-step training "
        "run with 5 stages would store 500k floats — wasteful when you "
        "only need the totals. Summing in place keeps memory O(num_stages) "
        "regardless of training length.\n\n"
        "**The 50x upper bound is conservative.** On a clean CI runner "
        "`time.sleep(0.005)` returns in ~5ms + tiny scheduler overhead, "
        "well under 2x nominal. The 50x cap absorbs container-level "
        "noise and OS jitter without false-failing the test."
    ),
    "extra_imports": [],
}


SPECS = [
    SPEC_WANDB_INIT,
    SPEC_WANDB_FINISH,
    SPEC_WANDB_LOG_STEP,
    SPEC_WANDB_WATCH,
    SPEC_WANDB_CONFIG_ARGS,
    SPEC_TQDM_POSTFIX,
    SPEC_LOG_SAMPLES,
    SPEC_TIME_STAGE,
]


# ---------------------------------------------------------------------------
# Verifier — exec every (solution + test_body) in a fresh namespace.
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

        # Wandb drills: clear any cached wandb mock from a previous spec so
        # call counts start fresh.
        if "wandb" in sys.modules:
            del sys.modules["wandb"]

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
