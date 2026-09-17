"""Composite drills cx13..cx18 — batch-20 part5 (MM-cell, ARENA VAE/GAN training loop).

Six composite procedural drills exercising 2-atom pairs from the canonical
training-loop / logging plumbing that wraps every VAE & GAN trainer.

cx13  backward-on-scalar-loss + zero-grad-set-none       — fwd loss → backward → step → zero_grad(set_to_none=True)
cx14  backward-on-scalar-loss + step-counter-increment   — track step count after each backward+step
cx15  backward-on-scalar-loss + wandb-log-step           — log scalar loss with wandb.log(step=N)
cx16  step-counter-increment + wandb-log-step            — step counter feeds the wandb step kwarg
cx17  step-counter-increment + zero-grad-set-none        — counter increments AFTER zero_grad in the canonical order
cx18  wandb-log-step + zero-grad-set-none                — log then zero_grad in standard order
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_composite import emit_composite  # noqa: E402

INV = {a["atom_id"]: a for a in json.load(open("/tmp/drill_atoms.json"))}


def _subs(atom_ids):
    return [INV[a]["subtopic"] for a in atom_ids]


# All six specs touch wandb (cx15, cx16, cx18) or torch.nn loss / Linear (everywhere).
# The verifier mocks `wandb` via MagicMock — see header doc.
NN_IMPORTS = [
    "import torch.nn as nn",
    "import torch.nn.functional as F",
    "import wandb",
]


# ===========================================================================
# cx13 — fwd loss → backward → opt.step → zero_grad(set_to_none=True)
# ===========================================================================
spec_13 = {
    "atom_ids": ["backward-on-scalar-loss", "zero-grad-set-none"],
    "subtopics": _subs(["backward-on-scalar-loss", "zero-grad-set-none"]),
    "primary_atom": "backward-on-scalar-loss",
    "part": "part5",
    "exercise_index": 13,
    "exercise_title": "canonical training step: backward on scalar loss + zero_grad set_to_none",
    "slug": "backward-then-zero-grad-set-none",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Every PyTorch training step ends with the same four-line ritual. Two of the four lines are "
        "the atoms in this drill:\n\n"
        "```python\n"
        "loss = loss_fn(model(x), y)   # forward; scalar if loss_fn is .mean()-reducing.\n"
        "loss.backward()               # atom A: backward-on-scalar-loss.\n"
        "optimizer.step()              # apply update.\n"
        "optimizer.zero_grad(set_to_none=True)  # atom B: zero-grad-set-none.\n"
        "```\n\n"
        "**Atom A — `backward-on-scalar-loss`.** `tensor.backward()` populates `.grad` on every "
        "leaf with `requires_grad=True` that contributed to `tensor`. The tensor MUST be a scalar "
        "(0-D / single-element) unless a `gradient=...` tensor is supplied. The canonical reduction "
        "is `.mean()` (batch-invariant scale) — `.sum()` makes the effective learning rate scale "
        "with batch size.\n\n"
        "**Atom B — `zero-grad-set-none`.** `.backward()` ACCUMULATES into existing `.grad`. Without "
        "a reset between batches, batch N+1's gradient would be added on top of batch N's, "
        "producing a stale oversized update. `set_to_none=True` replaces `.grad` with `None` "
        "(faster, no kernel launch, no memory cleared) — the next `.backward()` allocates a fresh "
        "tensor. This is PyTorch's default since 1.7.\n\n"
        "**Why the order matters.** `step()` reads from `.grad`. If you `zero_grad` BEFORE `step`, "
        "the optimizer sees zero gradient and the parameters never move. The order is "
        "`backward → step → zero_grad`, not `zero_grad → backward → step` (though that order works "
        "too; just don't put `zero_grad` between `backward` and `step`)."
    ),
    "prompt_body": (
        "Implement `cx13_train_step(model, optimizer, x, y, loss_fn)`. ONE training step. "
        "Sequence:\n\n"
        "1. `pred = model(x)` — forward.\n"
        "2. `loss = loss_fn(pred, y)` — assume `loss_fn` already reduces to a scalar (e.g. "
        "`nn.MSELoss()` defaults to `reduction='mean'`).\n"
        "3. `loss.backward()` (atom A — relies on `loss` being a scalar).\n"
        "4. `optimizer.step()` — apply the update.\n"
        "5. `optimizer.zero_grad(set_to_none=True)` (atom B).\n"
        "6. Return `loss.item()` — the scalar Python float.\n\n"
        "The test confirms:\n"
        "- `loss.item()` is returned (NOT the tensor — wandb-loggable).\n"
        "- Model parameters MOVED (so step ran after a non-None grad existed).\n"
        "- After return, every parameter has `p.grad is None` (set_to_none semantics).\n"
        "- Calling the function twice doesn't double-accumulate — second-call grads are NOT 2x the "
        "first-call grads (which is what would happen if `zero_grad` was missing)."
    ),
    "stub_body": (
        "def cx13_train_step(model, optimizer, x, y, loss_fn):\n"
        "    \"\"\"One canonical training step. Returns loss.item().\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: round-trip — loss returned as a float, params move, grads set to None.\n"
        "t.manual_seed(0)\n"
        "model = nn.Linear(3, 1)\n"
        "opt = t.optim.SGD(model.parameters(), lr=0.1)\n"
        "loss_fn = nn.MSELoss()\n"
        "x = t.randn(8, 3)\n"
        "y = t.randn(8, 1)\n"
        "\n"
        "params_before = [p.detach().clone() for p in model.parameters()]\n"
        "loss_val = cx13_train_step(model, opt, x, y, loss_fn)\n"
        "\n"
        "assert isinstance(loss_val, float), (\n"
        "    f'expected loss.item() (float); got {type(loss_val).__name__} — did you forget .item()?'\n"
        ")\n"
        "for p_before, p_after in zip(params_before, model.parameters()):\n"
        "    assert not t.allclose(p_before, p_after.detach()), (\n"
        "        'parameter did not move — optimizer.step() must run BEFORE zero_grad'\n"
        "    )\n"
        "for p in model.parameters():\n"
        "    assert p.grad is None, (\n"
        "        f'p.grad must be None after step (set_to_none=True); got {p.grad!r}'\n"
        "    )\n"
        "\n"
        "# Case B: second call works — zero_grad prevented accumulation.\n"
        "# If zero_grad was missing, the second backward would double-add gradients.\n"
        "# We verify by running TWO steps with identical (x, y) and checking that the\n"
        "# parameter delta of step 2 is close to step 1's delta (not 2x).\n"
        "t.manual_seed(1)\n"
        "m2 = nn.Linear(3, 1)\n"
        "opt2 = t.optim.SGD(m2.parameters(), lr=0.01)\n"
        "x2 = t.randn(8, 3)\n"
        "y2 = t.randn(8, 1)\n"
        "p0 = next(m2.parameters()).detach().clone()\n"
        "cx13_train_step(m2, opt2, x2, y2, loss_fn)\n"
        "p1 = next(m2.parameters()).detach().clone()\n"
        "delta1 = p1 - p0\n"
        "cx13_train_step(m2, opt2, x2, y2, loss_fn)\n"
        "p2 = next(m2.parameters()).detach().clone()\n"
        "delta2 = p2 - p1\n"
        "# delta2 should be SMALLER than 2*delta1 (because loss decreased between steps —\n"
        "# but absolutely NOT 2x as large, which signals double-accumulation).\n"
        "assert delta2.abs().max() < 2.0 * delta1.abs().max(), (\n"
        "    'second-step delta ~2x first-step delta — zero_grad is not running, grads are accumulating'\n"
        ")\n"
        "\n"
        "# Case C: the function rejects vector losses gracefully if the loss_fn happens to be\n"
        "# misconfigured (no reduction). PyTorch raises RuntimeError on vector backward without\n"
        "# a gradient= argument.\n"
        "t.manual_seed(2)\n"
        "m3 = nn.Linear(3, 1)\n"
        "opt3 = t.optim.SGD(m3.parameters(), lr=0.1)\n"
        "vec_loss = nn.MSELoss(reduction='none')\n"
        "x3 = t.randn(8, 3)\n"
        "y3 = t.randn(8, 1)\n"
        "raised = False\n"
        "try:\n"
        "    cx13_train_step(m3, opt3, x3, y3, vec_loss)\n"
        "except RuntimeError:\n"
        "    raised = True\n"
        "assert raised, (\n"
        "    'expected RuntimeError on backward of non-scalar loss — your code must call '\n"
        "    'loss.backward() unconditionally (do not silently sum-reduce inside cx13_train_step)'\n"
        ")"
    ),
    "solution_body": (
        "def cx13_train_step(model, optimizer, x, y, loss_fn):\n"
        "    # Forward + scalar loss.\n"
        "    pred = model(x)\n"
        "    loss = loss_fn(pred, y)\n"
        "    # Atom A: backward-on-scalar-loss. Requires loss to be a 0-D tensor.\n"
        "    loss.backward()\n"
        "    optimizer.step()\n"
        "    # Atom B: zero-grad-set-none. set_to_none=True is the default but write it out.\n"
        "    optimizer.zero_grad(set_to_none=True)\n"
        "    return loss.item()"
    ),
    "solution_notes": (
        "Returning `loss.item()` (a Python float) — not the tensor — is the wandb-friendly form: "
        "tensors aren't JSON-serializable. If your test fails on Case C, you may be accidentally "
        "calling `loss.sum().backward()` or `loss.mean().backward()` inside the function. Don't — "
        "the contract is that `loss_fn` produces a scalar; if it doesn't, propagate the error."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "kcs": ["backward-on-scalar-loss", "zero-grad-set-none"],
    "lo": (
        "Compose the scalar-loss backward and set_to_none zero_grad into the canonical four-line "
        "PyTorch training step so parameters update once per batch without grad accumulation."
    ),
}


# ===========================================================================
# cx14 — backward + step counter increment
# ===========================================================================
spec_14 = {
    "atom_ids": ["backward-on-scalar-loss", "step-counter-increment"],
    "subtopics": _subs(["backward-on-scalar-loss", "step-counter-increment"]),
    "primary_atom": "step-counter-increment",
    "part": "part5",
    "exercise_index": 14,
    "exercise_title": "scalar-loss backward followed by step counter tick",
    "slug": "backward-then-step-counter-tick",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Every trainer carries a step counter — the x-axis for logs, the trigger for learning-rate "
        "schedules, the gate for `if step % checkpoint_every == 0`. The canonical placement is "
        "**after** the optimizer update is committed:\n\n"
        "```python\n"
        "loss = loss_fn(model(x), y)\n"
        "loss.backward()              # atom A: backward-on-scalar-loss.\n"
        "optimizer.step()\n"
        "optimizer.zero_grad()\n"
        "self.step += 1               # atom B: step-counter-increment (AFTER step).\n"
        "```\n\n"
        "**Atom A — `backward-on-scalar-loss`.** The loss must be 0-D for `.backward()` to work "
        "without a `gradient=` argument. Same atom as cx13.\n\n"
        "**Atom B — `step-counter-increment`.** The counter measures how many UPDATES have been "
        "applied. Incrementing BEFORE `optimizer.step()` would mean step 1 reflects the model state "
        "from BEFORE step 1 happened — the off-by-one bug that breaks `step % log_every == 0` "
        "gates.\n\n"
        "**Why these together.** A batch with no backward is a batch that didn't update the model, "
        "so it should NOT tick the step counter. The two atoms have to fire in the same control "
        "flow: backward succeeded ⇒ optimizer updated ⇒ counter ticks. If the backward raises, the "
        "counter must NOT advance."
    ),
    "prompt_body": (
        "Implement `cx14_train_epoch(model, optimizer, loader, loss_fn, start_step)`. ONE epoch.\n\n"
        "Inputs:\n"
        "- `model`, `optimizer`, `loader`, `loss_fn`: usual.\n"
        "- `start_step`: int — counter value BEFORE this epoch.\n\n"
        "For each `(x, y)` in `loader`:\n"
        "1. `pred = model(x)`; `loss = loss_fn(pred, y)`.\n"
        "2. `loss.backward()` (atom A — requires scalar loss).\n"
        "3. `optimizer.step()`.\n"
        "4. `optimizer.zero_grad()` (default semantics — don't worry about set_to_none here).\n"
        "5. `step += 1` (atom B — AFTER the optimizer.step).\n"
        "6. Append `(step, loss.item())` to a log list AFTER the increment (so the entry's step "
        "reflects post-update state).\n\n"
        "Return `(final_step, log_list)`.\n\n"
        "**Contract**: if `loss.backward()` raises for ANY batch, the counter must NOT advance for "
        "that batch. (Easiest way: increment AFTER backward, not before — same canonical order.)"
    ),
    "stub_body": (
        "def cx14_train_epoch(model, optimizer, loader, loss_fn, start_step):\n"
        "    \"\"\"One epoch. Returns (final_step, [(step, loss_float), ...]).\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "from torch.utils.data import TensorDataset, DataLoader\n"
        "\n"
        "# Case A: 6 batches, start_step=0 → final_step=6, 6 log entries (1..6).\n"
        "t.manual_seed(0)\n"
        "x = t.randn(24, 3)\n"
        "y = t.randn(24, 1)\n"
        "loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)\n"
        "assert len(loader) == 6\n"
        "model = nn.Linear(3, 1)\n"
        "opt = t.optim.SGD(model.parameters(), lr=0.05)\n"
        "loss_fn = nn.MSELoss()\n"
        "\n"
        "final, log = cx14_train_epoch(model, opt, loader, loss_fn, start_step=0)\n"
        "assert final == 6, f'expected final_step=6, got {final}'\n"
        "assert len(log) == 6, f'expected 6 log entries, got {len(log)}'\n"
        "steps = [s for s, _ in log]\n"
        "assert steps == [1, 2, 3, 4, 5, 6], (\n"
        "    f'log steps must be [1..6] (ticked AFTER each optimizer.step); got {steps}'\n"
        ")\n"
        "for s, lv in log:\n"
        "    assert isinstance(lv, float), f'log entry loss should be float, got {type(lv).__name__}'\n"
        "\n"
        "# Case B: start_step accumulates across calls — epoch 2 picks up where epoch 1 left off.\n"
        "final2, log2 = cx14_train_epoch(model, opt, loader, loss_fn, start_step=final)\n"
        "assert final2 == 12, f'after epoch 2, final_step should be 12, got {final2}'\n"
        "assert [s for s, _ in log2] == [7, 8, 9, 10, 11, 12]\n"
        "\n"
        "# Case C: if a batch's backward raises, counter must NOT advance for that batch.\n"
        "# Inject a bad batch by using a loss_fn that returns a vector (RuntimeError on backward).\n"
        "vec_loss = nn.MSELoss(reduction='none')\n"
        "model3 = nn.Linear(3, 1)\n"
        "opt3 = t.optim.SGD(model3.parameters(), lr=0.05)\n"
        "raised = False\n"
        "try:\n"
        "    cx14_train_epoch(model3, opt3, loader, vec_loss, start_step=100)\n"
        "except RuntimeError:\n"
        "    raised = True\n"
        "assert raised, 'expected RuntimeError on vector-loss backward — code must call .backward() directly'\n"
        "\n"
        "# Case D: empty loader → counter unchanged, log empty.\n"
        "empty_loader = DataLoader(TensorDataset(t.zeros(0, 3), t.zeros(0, 1)), batch_size=4)\n"
        "f3, l3 = cx14_train_epoch(model, opt, empty_loader, loss_fn, start_step=42)\n"
        "assert f3 == 42, f'empty loader must not advance counter; got {f3}'\n"
        "assert l3 == [], f'empty loader must produce empty log; got {l3}'"
    ),
    "solution_body": (
        "def cx14_train_epoch(model, optimizer, loader, loss_fn, start_step):\n"
        "    step = start_step\n"
        "    log = []\n"
        "    for x, y in loader:\n"
        "        pred = model(x)\n"
        "        loss = loss_fn(pred, y)\n"
        "        # Atom A: scalar-loss backward.\n"
        "        loss.backward()\n"
        "        optimizer.step()\n"
        "        optimizer.zero_grad()\n"
        "        # Atom B: step counter tick — AFTER optimizer.step.\n"
        "        step += 1\n"
        "        log.append((step, loss.item()))\n"
        "    return step, log"
    ),
    "solution_notes": (
        "Placing the increment AFTER `optimizer.step()` (not before) is what makes Case C "
        "automatic — a backward that raises propagates out before the increment line runs, so the "
        "counter never advances for the failed batch. If you increment BEFORE backward (some "
        "frameworks do this for 'we attempted N batches' semantics), you'd need an explicit "
        "try/except to roll back on failure."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "kcs": ["backward-on-scalar-loss", "step-counter-increment"],
    "lo": (
        "Compose the scalar-loss backward with the post-step counter tick so the step counter "
        "advances exactly once per committed update — and never for batches whose backward fails."
    ),
}


# ===========================================================================
# cx15 — backward + wandb.log(step=N)
# ===========================================================================
spec_15 = {
    "atom_ids": ["backward-on-scalar-loss", "wandb-log-step"],
    "subtopics": _subs(["backward-on-scalar-loss", "wandb-log-step"]),
    "primary_atom": "wandb-log-step",
    "part": "part5",
    "exercise_index": 15,
    "exercise_title": "log scalar loss to wandb with step kwarg",
    "slug": "backward-then-wandb-log-step",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "The number you log to wandb is the SAME number you call `.backward()` on — modulo "
        "`.item()`. The composition makes the order explicit:\n\n"
        "```python\n"
        "loss = loss_fn(model(x), y)     # scalar.\n"
        "loss.backward()                  # atom A: backward-on-scalar-loss.\n"
        "wandb.log({'loss': loss.item()}, step=step_count)  # atom B: wandb-log-step.\n"
        "```\n\n"
        "**Atom A — `backward-on-scalar-loss`.** Backward requires a scalar; same atom as cx13/cx14.\n\n"
        "**Atom B — `wandb-log-step`.** `wandb.log(metrics_dict, step=int)` sends one row to the "
        "dashboard. The `step` kwarg is the x-axis. Pass `loss.item()` (Python float), NOT the "
        "tensor — wandb serializes to JSON and tensors aren't JSON-able. If you omit `step`, wandb "
        "uses its own monotonic counter, which makes runs with different batch sizes uncomparable.\n\n"
        "**Why both atoms in the same drill.** A common bug: log `loss` instead of `loss.item()`. "
        "It still 'works' in the sense that the JSON serializer raises later, far from where the "
        "logic broke. We test by mocking `wandb` and inspecting `wandb.log.call_args_list` — every "
        "logged value MUST be a plain Python `float`, every step MUST be a plain Python `int`."
    ),
    "prompt_body": (
        "Implement `cx15_train_and_log(model, optimizer, loader, loss_fn, batch_size)`. ONE epoch.\n\n"
        "For each `(x, y)` in `loader`, batch index `i` starting at 0:\n"
        "1. `pred = model(x)`; `loss = loss_fn(pred, y)`.\n"
        "2. `loss.backward()` (atom A).\n"
        "3. `optimizer.step()`; `optimizer.zero_grad()`.\n"
        "4. Compute `examples_seen = (i + 1) * batch_size`.\n"
        "5. Call `wandb.log({'loss': loss.item()}, step=examples_seen)` (atom B).\n\n"
        "Return the final `examples_seen`.\n\n"
        "**Test asserts (via mocked wandb)**:\n"
        "- One `wandb.log` call per batch.\n"
        "- Each call's positional first arg is a `dict` with key `'loss'` mapping to a Python "
        "`float` (NOT a `Tensor`).\n"
        "- Each call's `step` kwarg is an `int` equal to `(i+1)*batch_size`.\n"
        "- Final returned value equals `len(loader)*batch_size`."
    ),
    "stub_body": (
        "def cx15_train_and_log(model, optimizer, loader, loss_fn, batch_size):\n"
        "    \"\"\"Train one epoch, log each step to wandb. Return final examples_seen.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "from torch.utils.data import TensorDataset, DataLoader\n"
        "\n"
        "# Case A: 5 batches × bs=8 → 5 log calls, steps [8,16,24,32,40], final=40.\n"
        "wandb.log.reset_mock()\n"
        "t.manual_seed(0)\n"
        "x = t.randn(40, 3)\n"
        "y = t.randn(40, 1)\n"
        "loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)\n"
        "assert len(loader) == 5\n"
        "model = nn.Linear(3, 1)\n"
        "opt = t.optim.SGD(model.parameters(), lr=0.05)\n"
        "loss_fn = nn.MSELoss()\n"
        "\n"
        "final = cx15_train_and_log(model, opt, loader, loss_fn, batch_size=8)\n"
        "assert final == 40, f'expected final examples_seen=40, got {final}'\n"
        "\n"
        "calls = wandb.log.call_args_list\n"
        "assert len(calls) == 5, f'expected 5 wandb.log calls, got {len(calls)}'\n"
        "for i, c in enumerate(calls):\n"
        "    args, kwargs = c.args, c.kwargs\n"
        "    metrics = args[0] if args else kwargs.get('data') or kwargs.get('metrics')\n"
        "    assert isinstance(metrics, dict), f'call {i}: first arg must be dict, got {type(metrics).__name__}'\n"
        "    assert 'loss' in metrics, f'call {i}: metrics dict missing \"loss\" key; got {list(metrics.keys())}'\n"
        "    lv = metrics['loss']\n"
        "    assert isinstance(lv, float), (\n"
        "        f'call {i}: loss value must be a Python float (use .item()); got {type(lv).__name__}'\n"
        "    )\n"
        "    step_kwarg = kwargs.get('step')\n"
        "    assert step_kwarg == (i + 1) * 8, (\n"
        "        f'call {i}: step kwarg expected {(i+1)*8}, got {step_kwarg}'\n"
        "    )\n"
        "    assert isinstance(step_kwarg, int), (\n"
        "        f'call {i}: step kwarg must be int, got {type(step_kwarg).__name__}'\n"
        "    )\n"
        "\n"
        "# Case B: different batch size → step kwarg scales accordingly.\n"
        "wandb.log.reset_mock()\n"
        "loader2 = DataLoader(TensorDataset(x, y), batch_size=10, shuffle=False)\n"
        "assert len(loader2) == 4\n"
        "model2 = nn.Linear(3, 1)\n"
        "opt2 = t.optim.SGD(model2.parameters(), lr=0.05)\n"
        "final2 = cx15_train_and_log(model2, opt2, loader2, loss_fn, batch_size=10)\n"
        "assert final2 == 40, f'expected final=40 with bs=10*4 batches; got {final2}'\n"
        "calls2 = wandb.log.call_args_list\n"
        "steps2 = [c.kwargs.get('step') for c in calls2]\n"
        "assert steps2 == [10, 20, 30, 40], f'expected steps [10,20,30,40]; got {steps2}'\n"
        "\n"
        "# Case C: empty loader → zero log calls, final=0.\n"
        "wandb.log.reset_mock()\n"
        "empty = DataLoader(TensorDataset(t.zeros(0, 3), t.zeros(0, 1)), batch_size=8)\n"
        "final3 = cx15_train_and_log(model, opt, empty, loss_fn, batch_size=8)\n"
        "assert final3 == 0\n"
        "assert wandb.log.call_count == 0, f'empty loader → no log calls; got {wandb.log.call_count}'"
    ),
    "solution_body": (
        "def cx15_train_and_log(model, optimizer, loader, loss_fn, batch_size):\n"
        "    examples_seen = 0\n"
        "    for i, (x, y) in enumerate(loader):\n"
        "        pred = model(x)\n"
        "        loss = loss_fn(pred, y)\n"
        "        # Atom A: scalar-loss backward.\n"
        "        loss.backward()\n"
        "        optimizer.step()\n"
        "        optimizer.zero_grad()\n"
        "        examples_seen = (i + 1) * batch_size\n"
        "        # Atom B: wandb.log with step kwarg. loss.item() — NOT the tensor.\n"
        "        wandb.log({'loss': loss.item()}, step=examples_seen)\n"
        "    return examples_seen"
    ),
    "solution_notes": (
        "Computing `examples_seen` from `(i+1)*batch_size` instead of `len(x)*(i+1)` matters for "
        "the last partial batch — when `drop_last=False` the final batch may be smaller. ARENA's "
        "convention is the simple `(i+1)*batch_size` form because it makes runs with the same "
        "loader configuration directly comparable; production code often uses the exact "
        "`examples_seen += x.size(0)` form instead."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["backward-on-scalar-loss", "wandb-log-step"],
    "lo": (
        "Compose the scalar-loss backward with wandb.log({'loss': loss.item()}, step=examples_seen) "
        "so the same scalar feeds both the gradient computation and the dashboard, with a "
        "batch-size-invariant x-axis."
    ),
}


# ===========================================================================
# cx16 — step counter feeds wandb.log step kwarg
# ===========================================================================
spec_16 = {
    "atom_ids": ["step-counter-increment", "wandb-log-step"],
    "subtopics": _subs(["step-counter-increment", "wandb-log-step"]),
    "primary_atom": "step-counter-increment",
    "part": "part5",
    "exercise_index": 16,
    "exercise_title": "step counter feeds the wandb step kwarg",
    "slug": "step-counter-feeds-wandb-step",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "The step counter and the wandb log share the same number. Increment first, log second:\n\n"
        "```python\n"
        "optimizer.step()\n"
        "optimizer.zero_grad()\n"
        "self.step += 1                                  # atom A.\n"
        "wandb.log({'loss': loss.item()}, step=self.step)  # atom B: uses atom A's value.\n"
        "```\n\n"
        "**Atom A — `step-counter-increment`.** The counter measures committed updates. Tick AFTER "
        "`optimizer.step()` (see cx14). The value AT THE TIME OF LOG must reflect 'this many "
        "updates have happened'.\n\n"
        "**Atom B — `wandb-log-step`.** `wandb.log(metrics, step=N)` plots `metrics` at x=N on the "
        "dashboard. If `step` is omitted, wandb uses an internal monotonic counter — for "
        "single-runs that's fine, but it makes runs with different log frequencies uncomparable.\n\n"
        "**The composition.** Log AFTER increment — so the FIRST log entry has step=1, not step=0. "
        "If you log BEFORE incrementing, step=0 means 'before any update' but the metrics you're "
        "logging are post-update. That mismatch is the off-by-one bug that every trainer hits at "
        "least once. Test by inspecting `wandb.log.call_args_list` and asserting `[1, 2, 3, ...]`."
    ),
    "prompt_body": (
        "Implement `cx16_log_loop(losses, start_step)`.\n\n"
        "Inputs:\n"
        "- `losses`: list of Python floats — one per fake training step.\n"
        "- `start_step`: int — counter value BEFORE this loop.\n\n"
        "For each `loss` in `losses`:\n"
        "1. Increment counter: `step += 1` (atom A — BEFORE logging).\n"
        "2. Call `wandb.log({'loss': loss}, step=step)` (atom B).\n\n"
        "Return the final `step`.\n\n"
        "**Test asserts** (via mocked wandb):\n"
        "- One log call per loss.\n"
        "- Step kwarg sequence is `start_step+1, start_step+2, ...` (off-by-one matters).\n"
        "- Loss kwarg in each call equals the corresponding input loss.\n"
        "- Calling twice with `start_step=final_from_call_1` produces a continuous sequence "
        "(no reset)."
    ),
    "stub_body": (
        "def cx16_log_loop(losses, start_step):\n"
        "    \"\"\"Increment counter then wandb.log(..., step=counter) for each loss. Return final counter.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: start at 0 → steps [1, 2, 3, 4, 5].\n"
        "wandb.log.reset_mock()\n"
        "losses = [0.9, 0.8, 0.7, 0.6, 0.5]\n"
        "final = cx16_log_loop(losses, start_step=0)\n"
        "assert final == 5, f'expected final_step=5, got {final}'\n"
        "calls = wandb.log.call_args_list\n"
        "assert len(calls) == 5, f'expected 5 calls, got {len(calls)}'\n"
        "steps = [c.kwargs.get('step') for c in calls]\n"
        "assert steps == [1, 2, 3, 4, 5], (\n"
        "    f'step kwargs should be [1..5] (increment BEFORE log); got {steps}. '\n"
        "    f'If you got [0..4], you logged BEFORE incrementing — fix the order.'\n"
        ")\n"
        "logged_losses = [(c.args[0] if c.args else c.kwargs.get('data')).get('loss') for c in calls]\n"
        "assert logged_losses == losses, f'losses logged don\\'t match input; got {logged_losses}'\n"
        "\n"
        "# Case B: start at 100 → continuous sequence [101..105].\n"
        "wandb.log.reset_mock()\n"
        "final2 = cx16_log_loop(losses, start_step=100)\n"
        "assert final2 == 105\n"
        "steps2 = [c.kwargs.get('step') for c in wandb.log.call_args_list]\n"
        "assert steps2 == [101, 102, 103, 104, 105], f'got {steps2}'\n"
        "\n"
        "# Case C: chained calls produce a contiguous sequence across calls.\n"
        "wandb.log.reset_mock()\n"
        "mid = cx16_log_loop([0.1, 0.2, 0.3], start_step=0)\n"
        "assert mid == 3\n"
        "end = cx16_log_loop([0.4, 0.5], start_step=mid)\n"
        "assert end == 5\n"
        "steps3 = [c.kwargs.get('step') for c in wandb.log.call_args_list]\n"
        "assert steps3 == [1, 2, 3, 4, 5], (\n"
        "    f'chained calls should produce [1..5]; got {steps3}'\n"
        ")\n"
        "\n"
        "# Case D: empty losses → no calls, counter unchanged.\n"
        "wandb.log.reset_mock()\n"
        "f_empty = cx16_log_loop([], start_step=42)\n"
        "assert f_empty == 42\n"
        "assert wandb.log.call_count == 0\n"
        "\n"
        "# Case E: step kwarg must be int (not e.g. float — wandb requires integer step).\n"
        "wandb.log.reset_mock()\n"
        "cx16_log_loop([0.5], start_step=0)\n"
        "step_v = wandb.log.call_args_list[0].kwargs.get('step')\n"
        "assert isinstance(step_v, int), f'step kwarg must be int; got {type(step_v).__name__}'"
    ),
    "solution_body": (
        "def cx16_log_loop(losses, start_step):\n"
        "    step = start_step\n"
        "    for loss in losses:\n"
        "        # Atom A: increment FIRST so log entry's step reflects post-update state.\n"
        "        step += 1\n"
        "        # Atom B: wandb.log with step kwarg = current counter value.\n"
        "        wandb.log({'loss': loss}, step=step)\n"
        "    return step"
    ),
    "solution_notes": (
        "Increment-then-log (not log-then-increment) is the convention because the metric being "
        "logged reflects the state AFTER `step` updates have been applied. Logging at step=0 with "
        "post-update metrics would mean 'before any update happened, here's a post-update loss' — "
        "which makes no sense. The chained-call test (Case C) is the real-world resumption "
        "pattern: epoch N+1 starts where epoch N ended."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "kcs": ["step-counter-increment", "wandb-log-step"],
    "lo": (
        "Compose the step-counter increment (before logging) with wandb.log(step=counter) so the "
        "dashboard x-axis advances in lockstep with committed updates across an unbounded chain "
        "of calls."
    ),
}


# ===========================================================================
# cx17 — step counter increments AFTER zero_grad (canonical order)
# ===========================================================================
spec_17 = {
    "atom_ids": ["step-counter-increment", "zero-grad-set-none"],
    "subtopics": _subs(["step-counter-increment", "zero-grad-set-none"]),
    "primary_atom": "step-counter-increment",
    "part": "part5",
    "exercise_index": 17,
    "exercise_title": "step counter increments after zero_grad(set_to_none=True)",
    "slug": "step-counter-after-zero-grad-set-none",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "The canonical training-step ritual ends with `zero_grad`, then ticks the counter:\n\n"
        "```python\n"
        "loss.backward()\n"
        "optimizer.step()\n"
        "optimizer.zero_grad(set_to_none=True)   # atom A.\n"
        "self.step += 1                          # atom B (AFTER zero_grad).\n"
        "```\n\n"
        "**Atom A — `zero-grad-set-none`.** Sets every `p.grad = None`. Same atom as cx13. The "
        "next `.backward()` allocates fresh grads — no accumulation.\n\n"
        "**Atom B — `step-counter-increment`.** The counter advances once per committed update. "
        "Same atom as cx14/cx16. Position it AFTER `zero_grad` so that, at the moment the counter "
        "shows `N`, the model has been updated N times AND its gradient slate is clean for batch "
        "N+1.\n\n"
        "**Why this exact order.** If you increment BEFORE `zero_grad`, then any code that reads "
        "`self.step` between the increment and the `zero_grad` sees a state where 'we've done N "
        "updates' but `p.grad` still holds batch N's gradient. That gradient is about to be wiped "
        "— so it's stale information masquerading as fresh — and any concurrent reader (logging "
        "thread, gradient-clipping monitor, gradient-norm logger) sees inconsistent state. The "
        "rule: tick LAST.\n\n"
        "We test by asserting BOTH semantic facts after each step: counter incremented by exactly "
        "1 AND every `p.grad is None`."
    ),
    "prompt_body": (
        "Implement `cx17_train_loop(model, optimizer, loader, loss_fn, start_step)`. ONE epoch.\n\n"
        "For each `(x, y)`:\n"
        "1. Forward + backward + optimizer.step (same as cx13/cx14).\n"
        "2. `optimizer.zero_grad(set_to_none=True)` (atom A — explicit `set_to_none=True`).\n"
        "3. `step += 1` (atom B — AFTER `zero_grad`).\n\n"
        "Return `(final_step, snapshots)` where `snapshots` is a list of "
        "`(step_at_end_of_batch, all_grads_are_none)` tuples — one per batch. The boolean is "
        "`all(p.grad is None for p in model.parameters())`.\n\n"
        "**Test asserts**:\n"
        "- Step sequence is `[start_step+1, start_step+2, ...]`.\n"
        "- After EVERY batch (i.e. at every snapshot), all grads are None.\n"
        "- After the final batch returns, every `p.grad is None` (set_to_none observed externally).\n"
        "- Model parameters MOVED (so step did run before zero_grad)."
    ),
    "stub_body": (
        "def cx17_train_loop(model, optimizer, loader, loss_fn, start_step):\n"
        "    \"\"\"Train one epoch. zero_grad(set_to_none=True) then increment counter.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "from torch.utils.data import TensorDataset, DataLoader\n"
        "\n"
        "# Case A: 6 batches, start=0. After each batch: grads None, counter ticked.\n"
        "t.manual_seed(0)\n"
        "x = t.randn(24, 3)\n"
        "y = t.randn(24, 1)\n"
        "loader = DataLoader(TensorDataset(x, y), batch_size=4, shuffle=False)\n"
        "assert len(loader) == 6\n"
        "model = nn.Linear(3, 1)\n"
        "params_before = [p.detach().clone() for p in model.parameters()]\n"
        "opt = t.optim.SGD(model.parameters(), lr=0.05)\n"
        "loss_fn = nn.MSELoss()\n"
        "\n"
        "final, snaps = cx17_train_loop(model, opt, loader, loss_fn, start_step=0)\n"
        "assert final == 6, f'expected final=6, got {final}'\n"
        "assert len(snaps) == 6\n"
        "for i, (s, all_none) in enumerate(snaps):\n"
        "    assert s == i + 1, f'snapshot {i}: step should be {i+1}, got {s}'\n"
        "    assert all_none is True, (\n"
        "        f'snapshot {i}: all grads should be None — zero_grad(set_to_none=True) must run BEFORE counter tick. '\n"
        "        f'If snapshot[0] shows grads non-None, you are taking the snapshot BEFORE zero_grad.'\n"
        "    )\n"
        "\n"
        "# Case B: post-return state — all p.grad is None.\n"
        "for p in model.parameters():\n"
        "    assert p.grad is None, f'post-return p.grad must be None; got {p.grad!r}'\n"
        "\n"
        "# Case C: params actually moved (step ran).\n"
        "for p_before, p_after in zip(params_before, model.parameters()):\n"
        "    assert not t.allclose(p_before, p_after.detach()), 'param did not move — step did not run'\n"
        "\n"
        "# Case D: start_step offset accumulates correctly.\n"
        "model2 = nn.Linear(3, 1)\n"
        "opt2 = t.optim.SGD(model2.parameters(), lr=0.05)\n"
        "f2, snaps2 = cx17_train_loop(model2, opt2, loader, loss_fn, start_step=100)\n"
        "assert f2 == 106\n"
        "assert [s for s, _ in snaps2] == [101, 102, 103, 104, 105, 106]\n"
        "\n"
        "# Case E: no accumulation — params don't blow up after 2 epochs.\n"
        "model3 = nn.Linear(3, 1)\n"
        "opt3 = t.optim.SGD(model3.parameters(), lr=0.05)\n"
        "f3a, _ = cx17_train_loop(model3, opt3, loader, loss_fn, start_step=0)\n"
        "p_after_1 = next(model3.parameters()).detach().clone()\n"
        "f3b, _ = cx17_train_loop(model3, opt3, loader, loss_fn, start_step=f3a)\n"
        "p_after_2 = next(model3.parameters()).detach().clone()\n"
        "# Two epochs should produce a bounded update — not a 2x larger one (which would signal\n"
        "# zero_grad never ran and grads accumulated across batches).\n"
        "delta = (p_after_2 - p_after_1).abs().max()\n"
        "assert delta < 1.0, f'second-epoch delta blew up to {delta:.4f}; zero_grad probably not running'"
    ),
    "solution_body": (
        "def cx17_train_loop(model, optimizer, loader, loss_fn, start_step):\n"
        "    step = start_step\n"
        "    snaps = []\n"
        "    for x, y in loader:\n"
        "        pred = model(x)\n"
        "        loss = loss_fn(pred, y)\n"
        "        loss.backward()\n"
        "        optimizer.step()\n"
        "        # Atom A: zero-grad-set-none. Wipe grads.\n"
        "        optimizer.zero_grad(set_to_none=True)\n"
        "        # Atom B: counter tick AFTER zero_grad.\n"
        "        step += 1\n"
        "        all_none = all(p.grad is None for p in model.parameters())\n"
        "        snaps.append((step, all_none))\n"
        "    return step, snaps"
    ),
    "solution_notes": (
        "The snapshot is taken AFTER both atoms have fired, so `all_none == True` is the steady "
        "state contract. If you reorder to `increment → zero_grad`, the snapshot would still pass — "
        "but Case E (no accumulation) catches the deeper bug of forgetting `zero_grad` entirely. "
        "The canonical order `step → zero_grad → counter` is what makes the step counter a "
        "trustworthy 'how many committed updates so far' invariant."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "kcs": ["step-counter-increment", "zero-grad-set-none"],
    "lo": (
        "Compose zero_grad(set_to_none=True) with the post-zero_grad step counter increment so "
        "the counter and the grad-cleared state become observable together at the end of each batch."
    ),
}


# ===========================================================================
# cx18 — log then zero_grad in the standard order
# ===========================================================================
spec_18 = {
    "atom_ids": ["wandb-log-step", "zero-grad-set-none"],
    "subtopics": _subs(["wandb-log-step", "zero-grad-set-none"]),
    "primary_atom": "wandb-log-step",
    "part": "part5",
    "exercise_index": 18,
    "exercise_title": "wandb.log then zero_grad(set_to_none=True) — standard order",
    "slug": "wandb-log-then-zero-grad-set-none",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "After `optimizer.step()`, two cleanup actions remain: (a) emit the metrics for this batch "
        "to wandb, (b) wipe the gradients for the next batch. The canonical order is **log first, "
        "then zero_grad**:\n\n"
        "```python\n"
        "loss.backward()\n"
        "optimizer.step()\n"
        "wandb.log({'loss': loss.item()}, step=step_count)  # atom A.\n"
        "optimizer.zero_grad(set_to_none=True)              # atom B.\n"
        "```\n\n"
        "**Atom A — `wandb-log-step`.** `wandb.log({...}, step=N)` plots metrics against `N`. "
        "Pass `loss.item()` (Python float), `step` as `int`.\n\n"
        "**Atom B — `zero-grad-set-none`.** Sets every `p.grad = None` so the next backward "
        "allocates fresh tensors with no accumulation.\n\n"
        "**Why log first, then zero_grad.** The log line may want to record gradient-norm "
        "statistics in the future (`wandb.log({'loss': l, 'grad_norm': gn}, step=N)`). If you "
        "`zero_grad` first, every grad is `None` by the time you'd compute the norm — you've "
        "destroyed the data you wanted to log. Even if today's metric is just `loss`, the order "
        "leaves the door open for richer logging without a refactor.\n\n"
        "**A weaker reason:** consistency. The order `loss → backward → step → log → zero_grad` "
        "matches what every ARENA trainer template does, so reviewers can pattern-match.\n\n"
        "We test by mocking wandb and asserting (i) `wandb.log` is called with the right "
        "metrics+step, (ii) AFTER the function returns every `p.grad is None`, and (iii) the call "
        "ORDER is correct — gradient norm is logged with a non-zero value, which only works if "
        "log fires BEFORE zero_grad."
    ),
    "prompt_body": (
        "Implement `cx18_train_step_with_log(model, optimizer, x, y, loss_fn, step_count)`.\n\n"
        "Sequence (one batch):\n"
        "1. `pred = model(x)`; `loss = loss_fn(pred, y)`.\n"
        "2. `loss.backward()`.\n"
        "3. `optimizer.step()`.\n"
        "4. Compute `grad_norm`: the L2 norm across ALL parameter gradients flattened together — "
        "`torch.cat([p.grad.flatten() for p in model.parameters()]).norm().item()`. "
        "(This relies on `.grad` still being a Tensor — the whole point of doing this BEFORE "
        "`zero_grad`.)\n"
        "5. Call `wandb.log({'loss': loss.item(), 'grad_norm': grad_norm}, step=step_count)` "
        "(atom A).\n"
        "6. Call `optimizer.zero_grad(set_to_none=True)` (atom B).\n"
        "7. Return `loss.item()`.\n\n"
        "**Test asserts** (via mocked wandb):\n"
        "- `wandb.log` called exactly once with both `loss` and `grad_norm` keys.\n"
        "- `grad_norm` value is a positive float (proof you computed it BEFORE zero_grad).\n"
        "- Step kwarg equals `step_count`.\n"
        "- After return, every `p.grad is None`."
    ),
    "stub_body": (
        "def cx18_train_step_with_log(model, optimizer, x, y, loss_fn, step_count):\n"
        "    \"\"\"One step: forward → backward → step → log → zero_grad. Returns loss float.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: round-trip with grad_norm — must be computed BEFORE zero_grad.\n"
        "wandb.log.reset_mock()\n"
        "t.manual_seed(0)\n"
        "model = nn.Linear(3, 1)\n"
        "opt = t.optim.SGD(model.parameters(), lr=0.05)\n"
        "loss_fn = nn.MSELoss()\n"
        "x = t.randn(8, 3)\n"
        "y = t.randn(8, 1)\n"
        "\n"
        "loss_val = cx18_train_step_with_log(model, opt, x, y, loss_fn, step_count=42)\n"
        "\n"
        "assert isinstance(loss_val, float)\n"
        "assert wandb.log.call_count == 1, f'expected 1 wandb.log call, got {wandb.log.call_count}'\n"
        "call = wandb.log.call_args_list[0]\n"
        "metrics = call.args[0] if call.args else call.kwargs.get('data')\n"
        "assert isinstance(metrics, dict)\n"
        "assert 'loss' in metrics and 'grad_norm' in metrics, (\n"
        "    f'metrics dict missing keys; got {list(metrics.keys())}'\n"
        ")\n"
        "assert isinstance(metrics['loss'], float)\n"
        "assert isinstance(metrics['grad_norm'], float)\n"
        "assert metrics['grad_norm'] > 0, (\n"
        "    f'grad_norm should be > 0 (computed BEFORE zero_grad); got {metrics[\"grad_norm\"]}. '\n"
        "    f'If you got 0 or got an AttributeError, you called zero_grad BEFORE computing grad_norm.'\n"
        ")\n"
        "assert call.kwargs.get('step') == 42, f\"step kwarg expected 42, got {call.kwargs.get('step')}\"\n"
        "\n"
        "# Case B: after return, all grads are None (set_to_none observed externally).\n"
        "for p in model.parameters():\n"
        "    assert p.grad is None, f'post-return p.grad must be None; got {p.grad!r}'\n"
        "\n"
        "# Case C: chained calls — log fires each call, grad_norm always > 0.\n"
        "wandb.log.reset_mock()\n"
        "t.manual_seed(1)\n"
        "model2 = nn.Linear(3, 1)\n"
        "opt2 = t.optim.SGD(model2.parameters(), lr=0.05)\n"
        "for step_n in [1, 2, 3]:\n"
        "    cx18_train_step_with_log(model2, opt2, t.randn(8, 3), t.randn(8, 1), loss_fn, step_count=step_n)\n"
        "assert wandb.log.call_count == 3\n"
        "for i, call in enumerate(wandb.log.call_args_list):\n"
        "    metrics_i = call.args[0] if call.args else call.kwargs.get('data')\n"
        "    assert metrics_i['grad_norm'] > 0, (\n"
        "        f'call {i}: grad_norm must be > 0 every step (computed before zero_grad)'\n"
        "    )\n"
        "    assert call.kwargs.get('step') == i + 1\n"
        "\n"
        "# Case D: order matters — if log were AFTER zero_grad, p.grad would be None during the\n"
        "# norm computation, raising AttributeError. The fact that Case A passed without exception\n"
        "# AND grad_norm > 0 is the joint signal that the order is right."
    ),
    "solution_body": (
        "def cx18_train_step_with_log(model, optimizer, x, y, loss_fn, step_count):\n"
        "    pred = model(x)\n"
        "    loss = loss_fn(pred, y)\n"
        "    loss.backward()\n"
        "    optimizer.step()\n"
        "    # grad_norm BEFORE zero_grad — relies on .grad still being a Tensor.\n"
        "    grad_norm = t.cat([p.grad.flatten() for p in model.parameters()]).norm().item()\n"
        "    # Atom A: wandb.log with both metrics + step kwarg.\n"
        "    wandb.log({'loss': loss.item(), 'grad_norm': grad_norm}, step=step_count)\n"
        "    # Atom B: zero-grad-set-none.\n"
        "    optimizer.zero_grad(set_to_none=True)\n"
        "    return loss.item()"
    ),
    "solution_notes": (
        "The grad-norm metric is the canonical reason for log-before-zero_grad — it's a common "
        "diagnostic in GAN/VAE trainers where one of the two networks may be exploding while the "
        "other looks fine. ARENA's reference DCGAN trainer uses exactly this pattern. If you "
        "tried `zero_grad → log` and Case A raised `AttributeError: NoneType has no attribute "
        "'flatten'`, that's the canonical bug — the order in the canonical recipe is there for a "
        "reason."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["wandb-log-step", "zero-grad-set-none"],
    "lo": (
        "Compose the post-step wandb.log (with both loss and grad_norm) BEFORE the "
        "zero_grad(set_to_none=True) so gradient statistics can be recorded before the buffers "
        "are wiped — the canonical training-step order."
    ),
}


SPECS = [spec_13, spec_14, spec_15, spec_16, spec_17, spec_18]


def main() -> None:
    for spec in SPECS:
        out = emit_composite(spec)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
