"""Composite drills cx19..cx24 — batch-20 part5 (NN-cell, ARENA VAE/GAN eval+dataloader).

Six composite procedural drills exercising 2-atom pairs from ARENA part 5 —
the eval/logging/dataloader scaffolding around VAE/GAN training.

cx19  backward-on-scalar-loss + log-samples-eval-callback  — backward step then conditional sample-logging callback
cx20  log-samples-eval-callback + step-counter-increment   — log samples every N steps
cx21  log-samples-eval-callback + wandb-log-step           — log generated samples as wandb.Image
cx22  log-samples-eval-callback + zero-grad-set-none       — eval callback then resume training (zero_grad)
cx23  dataloader-batching + backward-on-scalar-loss        — per-batch backward pass
cx24  dataloader-batching + step-counter-increment         — batch loop with step counter
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_composite import emit_composite  # noqa: E402

INV = {a["atom_id"]: a for a in json.load(open("/tmp/drill_atoms.json"))}


def _subs(atom_ids):
    return [INV[a]["subtopic"] for a in atom_ids]


NN_IMPORTS = [
    "import torch.nn as nn",
    "import torch.nn.functional as F",
    "import wandb",
    "from torch.utils.data import DataLoader, TensorDataset",
]


# ===========================================================================
# cx19 — backward then conditional sample-logging callback
# ===========================================================================
spec_19 = {
    "atom_ids": ["backward-on-scalar-loss", "log-samples-eval-callback"],
    "subtopics": _subs(["backward-on-scalar-loss", "log-samples-eval-callback"]),
    "primary_atom": "backward-on-scalar-loss",
    "part": "part5",
    "exercise_index": 19,
    "exercise_title": "backward on scalar loss, then maybe fire the eval callback",
    "slug": "backward-on-scalar-loss-then-eval-callback",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "A real VAE/GAN training step has two responsibilities per micro-batch:\n\n"
        "1. **Backward pass on the scalar loss.** Per-sample losses must be **reduced** to a 0-dim "
        "scalar before `.backward()`. Calling `.backward()` on a vector raises "
        "`RuntimeError: grad can be implicitly created only for scalar outputs`. The canonical "
        "reduction is `.mean()` (or `.sum()` / explicit weights). Atom: `backward-on-scalar-loss`.\n"
        "2. **Conditional eval/logging callback.** Every `K` steps, the trainer calls a callback "
        "that generates `N` samples (e.g. decoder draws from the prior) and writes them to a sink. "
        "Atom: `log-samples-eval-callback`.\n\n"
        "**Why compose them.** The most common bug at this seam is calling `.backward()` on the "
        "*unreduced* per-sample loss because the eval callback was tacked on AFTER the loop body "
        "without a `.mean()`. The composition forces both steps to happen in the right order:\n\n"
        "```python\n"
        "per_sample_loss = recon_loss + kl_loss   # shape (B,)\n"
        "loss = per_sample_loss.mean()            # backward-on-scalar-loss.\n"
        "loss.backward()\n"
        "if step % log_every == 0:                # log-samples-eval-callback.\n"
        "    samples = sample_fn()\n"
        "    sink.append(samples)\n"
        "```\n\n"
        "We test that `.backward()` is invoked on a scalar (autograd raises otherwise) and that the "
        "callback fires exactly on the documented schedule."
    ),
    "prompt_body": (
        "Implement `cx19_step_then_maybe_log(per_sample_loss, param, log_every, step, sample_fn, sink)`.\n\n"
        "Inputs:\n"
        "- `per_sample_loss` — a `t.Tensor` of shape `(B,)`, where each entry is the loss for one "
        "sample in the batch. Has `requires_grad=True` (descends from `param`).\n"
        "- `param` — a leaf `t.Tensor` with `requires_grad=True`, which `per_sample_loss` was built "
        "from. The function MUST populate `param.grad`.\n"
        "- `log_every` — int. Fire the callback iff `step % log_every == 0`.\n"
        "- `step` — int, the current global step number.\n"
        "- `sample_fn` — a callable `() -> Any`. Call it to obtain a sample batch.\n"
        "- `sink` — a `list`. When the callback fires, append the sample-fn output AND the step "
        "number as a `(step, sample)` tuple.\n\n"
        "Required behaviour, in this order:\n"
        "1. Reduce `per_sample_loss` to a **scalar** via `.mean()` (atom: backward-on-scalar-loss).\n"
        "2. Call `.backward()` on that scalar. After this, `param.grad` is populated.\n"
        "3. If `step % log_every == 0`, call `sample_fn()` and append `(step, sample)` to `sink` "
        "(atom: log-samples-eval-callback).\n"
        "4. Return the scalar loss value (a Python float).\n\n"
        "The test confirms:\n"
        "- `param.grad` is populated (NOT `None`) after the call.\n"
        "- Callback fires exactly when `step % log_every == 0` — not on other steps.\n"
        "- Sink entries are `(step, sample_fn_output)` tuples in step order.\n"
        "- Calling `.backward()` on the unreduced vector would raise — your reduction is load-bearing."
    ),
    "stub_body": (
        "def cx19_step_then_maybe_log(per_sample_loss, param, log_every, step, sample_fn, sink):\n"
        "    \"\"\"Reduce to scalar, backward, then maybe fire the log-samples callback. Returns the loss as float.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: scalar backward populates param.grad; callback skipped when step % log_every != 0.\n"
        "t.manual_seed(0)\n"
        "param = t.randn(3, requires_grad=True)\n"
        "per_sample = (param ** 2).expand(4, 3).clone().sum(dim=1)  # shape (4,)\n"
        "assert per_sample.shape == (4,)\n"
        "sink = []\n"
        "calls = {'n': 0}\n"
        "def _sample_fn():\n"
        "    calls['n'] += 1\n"
        "    return t.full((2, 2), float(calls['n']))\n"
        "loss = cx19_step_then_maybe_log(per_sample, param, log_every=5, step=3, sample_fn=_sample_fn, sink=sink)\n"
        "assert isinstance(loss, float), f'must return a python float; got {type(loss).__name__}'\n"
        "assert param.grad is not None, 'param.grad must be populated by backward'\n"
        "assert calls['n'] == 0, 'sample_fn must not be called when step % log_every != 0'\n"
        "assert sink == [], 'sink must stay empty when step % log_every != 0'\n"
        "\n"
        "# Case B: callback FIRES when step % log_every == 0; sink gets (step, sample) tuple.\n"
        "param2 = t.randn(2, requires_grad=True)\n"
        "per_sample2 = (param2 ** 2).expand(3, 2).clone().sum(dim=1)\n"
        "sink2 = []\n"
        "calls2 = {'n': 0}\n"
        "def _sample_fn2():\n"
        "    calls2['n'] += 1\n"
        "    return f'sample_{calls2[\"n\"]}'\n"
        "loss2 = cx19_step_then_maybe_log(per_sample2, param2, log_every=4, step=8, sample_fn=_sample_fn2, sink=sink2)\n"
        "assert param2.grad is not None\n"
        "assert calls2['n'] == 1, f'sample_fn must fire exactly once on step 8 with log_every=4; got {calls2[\"n\"]}'\n"
        "assert len(sink2) == 1\n"
        "entry = sink2[0]\n"
        "assert isinstance(entry, tuple) and len(entry) == 2, f'sink entry must be (step, sample) tuple; got {entry}'\n"
        "assert entry[0] == 8, f'first element of sink entry must be the step; got {entry[0]}'\n"
        "assert entry[1] == 'sample_1', f'second element must be sample_fn output; got {entry[1]}'\n"
        "\n"
        "# Case C: scalar return value matches per_sample.mean().\n"
        "param3 = t.tensor([1.0, 2.0, 3.0], requires_grad=True)\n"
        "per_sample3 = param3 * 2.0  # shape (3,), values 2,4,6 → mean = 4.\n"
        "loss3 = cx19_step_then_maybe_log(per_sample3, param3, log_every=10, step=0, sample_fn=lambda: None, sink=[])\n"
        "assert abs(loss3 - 4.0) < 1e-6, f'returned loss should equal per_sample.mean()=4.0; got {loss3}'\n"
        "# step=0 % log_every=10 == 0, so callback should fire too.\n"
        "sink_check = []\n"
        "param3b = t.tensor([1.0, 2.0, 3.0], requires_grad=True)\n"
        "per_sample3b = param3b * 2.0\n"
        "cx19_step_then_maybe_log(per_sample3b, param3b, log_every=10, step=0, sample_fn=lambda: 'fired', sink=sink_check)\n"
        "assert sink_check == [(0, 'fired')], f'step=0 % log_every=10 == 0; callback should fire. got {sink_check}'\n"
        "\n"
        "# Case D: gradient is CORRECT, not just non-None. d(mean(2*p))/dp = 2/3 per coord.\n"
        "param4 = t.tensor([1.0, 2.0, 3.0], requires_grad=True)\n"
        "per_sample4 = param4 * 2.0\n"
        "cx19_step_then_maybe_log(per_sample4, param4, log_every=99, step=1, sample_fn=lambda: None, sink=[])\n"
        "expected_grad = t.full((3,), 2.0 / 3.0)\n"
        "assert t.allclose(param4.grad, expected_grad, atol=1e-7), (\n"
        "    f'gradient through mean reduction should be 2/3 per coord; got {param4.grad.tolist()}'\n"
        ")\n"
        "\n"
        "# Case E: callback fires on every multiple of log_every across a sequence.\n"
        "param5 = t.tensor([1.0], requires_grad=True)\n"
        "sink5 = []\n"
        "for s in range(1, 11):\n"
        "    p5 = (param5 * float(s)).expand(2).clone()\n"
        "    cx19_step_then_maybe_log(p5, param5, log_every=3, step=s, sample_fn=lambda s=s: f'sample-{s}', sink=sink5)\n"
        "fired_steps = [e[0] for e in sink5]\n"
        "assert fired_steps == [3, 6, 9], f'expected callbacks on steps [3, 6, 9]; got {fired_steps}'"
    ),
    "solution_body": (
        "def cx19_step_then_maybe_log(per_sample_loss, param, log_every, step, sample_fn, sink):\n"
        "    # Atom A (backward-on-scalar-loss): reduce per-sample loss to a 0-dim scalar.\n"
        "    loss = per_sample_loss.mean()\n"
        "    loss.backward()\n"
        "    # Atom B (log-samples-eval-callback): fire every `log_every` steps.\n"
        "    if step % log_every == 0:\n"
        "        sample = sample_fn()\n"
        "        sink.append((step, sample))\n"
        "    return loss.item()"
    ),
    "solution_notes": (
        "The classic bug here is calling `per_sample_loss.backward()` directly — autograd needs an "
        "explicit `gradient=` argument for non-scalar outputs, and the natural default is "
        "`.mean()`. The callback is a clean side-channel: it doesn't read `loss` itself, only the "
        "step counter — so the order doesn't matter for correctness, but doing backward FIRST "
        "matches how trainers are written (backward → optional callbacks → optimizer step)."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["backward-on-scalar-loss", "log-samples-eval-callback"],
    "lo": (
        "Compose a scalar-reduced backward pass with a step-gated eval/sample-logging callback so a "
        "training step both populates param.grad and fires the periodic sample sink."
    ),
}


# ===========================================================================
# cx20 — log samples every N steps (step-counter + eval callback)
# ===========================================================================
spec_20 = {
    "atom_ids": ["log-samples-eval-callback", "step-counter-increment"],
    "subtopics": _subs(["log-samples-eval-callback", "step-counter-increment"]),
    "primary_atom": "log-samples-eval-callback",
    "part": "part5",
    "exercise_index": 20,
    "exercise_title": "increment step counter, then fire log-samples callback every N",
    "slug": "log-samples-eval-callback-every-n-steps",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "A trainer that logs generated samples every `N` steps depends on two atoms wired in "
        "exactly the right order:\n\n"
        "1. **step-counter-increment** — increment `self.step` (or the local `step` var) AT THE "
        "TOP of each iteration, so the FIRST iteration ends with `step == 1`, not `step == 0`. "
        "The convention `step % N == 0` then fires correctly on iteration `N`, not iteration 0.\n"
        "2. **log-samples-eval-callback** — when `step % N == 0`, call `sample_fn()` and stash the "
        "result in a sink keyed by step.\n\n"
        "**Why order matters.** If you check `step % N == 0` BEFORE incrementing, step 0 (before "
        "any work was done) fires the callback — you log random-init samples. If you check AFTER "
        "incrementing, the callback fires on steps `N, 2N, 3N, ...` — sample sets that reflect "
        "actual training progress.\n\n"
        "**Anatomy.**\n"
        "```python\n"
        "for micro_batch in batches:\n"
        "    step += 1                          # step-counter-increment (top of loop).\n"
        "    # ... train ...\n"
        "    if step % log_every == 0:          # log-samples-eval-callback.\n"
        "        samples = sample_fn()\n"
        "        sink[step] = samples\n"
        "```"
    ),
    "prompt_body": (
        "Implement `cx20_train_with_sample_logging(n_iters, log_every, sample_fn)`.\n\n"
        "Inputs:\n"
        "- `n_iters` — int. Number of training iterations to run.\n"
        "- `log_every` — int. Sample callback fires when `step % log_every == 0`.\n"
        "- `sample_fn` — callable `(step: int) -> Any`. Receives the current step number and "
        "returns a sample.\n\n"
        "Required behaviour:\n"
        "1. Initialise a local `step = 0` counter.\n"
        "2. Initialise a `sink` dict: `{step: sample}` entries appended by the callback.\n"
        "3. For each of `n_iters` iterations:\n"
        "   a. Increment `step` (atom: step-counter-increment) — so iteration 1 ends with step=1.\n"
        "   b. If `step % log_every == 0` (atom: log-samples-eval-callback), call `sample_fn(step)` "
        "and write `sink[step] = sample`.\n"
        "4. Return `(step, sink)` — the final step counter and the populated sink.\n\n"
        "Test checks:\n"
        "- Step counter increments exactly `n_iters` times (final step == n_iters).\n"
        "- Callback fires on steps `log_every, 2*log_every, ..., (n_iters // log_every) * log_every`.\n"
        "- `sink` keys are exactly that arithmetic progression.\n"
        "- `sample_fn` is called with the correct `step` argument each time.\n"
        "- Step 0 does NOT fire the callback (because counter starts at 0 and increments BEFORE the check)."
    ),
    "stub_body": (
        "def cx20_train_with_sample_logging(n_iters, log_every, sample_fn):\n"
        "    \"\"\"Run n_iters of a fake training loop; log samples every log_every steps. Returns (final_step, sink_dict).\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: 10 iters, log every 3 → fires on steps 3, 6, 9.\n"
        "received_steps = []\n"
        "def _sample_fn(step):\n"
        "    received_steps.append(step)\n"
        "    return f'samples@{step}'\n"
        "final_step, sink = cx20_train_with_sample_logging(n_iters=10, log_every=3, sample_fn=_sample_fn)\n"
        "assert final_step == 10, f'final step should equal n_iters=10; got {final_step}'\n"
        "assert sorted(sink.keys()) == [3, 6, 9], f'expected keys [3, 6, 9]; got {sorted(sink.keys())}'\n"
        "assert sink[3] == 'samples@3'\n"
        "assert sink[6] == 'samples@6'\n"
        "assert sink[9] == 'samples@9'\n"
        "assert received_steps == [3, 6, 9], f'sample_fn called with wrong steps; got {received_steps}'\n"
        "\n"
        "# Case B: zero iters → no callback, final step 0.\n"
        "calls = []\n"
        "fs, sk = cx20_train_with_sample_logging(n_iters=0, log_every=2, sample_fn=lambda s: calls.append(s) or 'x')\n"
        "assert fs == 0\n"
        "assert sk == {}\n"
        "assert calls == [], 'no iterations should mean no callback'\n"
        "\n"
        "# Case C: log_every=1 fires every step.\n"
        "fired = []\n"
        "fs2, sk2 = cx20_train_with_sample_logging(\n"
        "    n_iters=5, log_every=1, sample_fn=lambda s: fired.append(s) or s\n"
        ")\n"
        "assert fs2 == 5\n"
        "assert sorted(sk2.keys()) == [1, 2, 3, 4, 5], (\n"
        "    f'log_every=1 should fire on every step; got keys {sorted(sk2.keys())}'\n"
        ")\n"
        "assert fired == [1, 2, 3, 4, 5]\n"
        "\n"
        "# Case D: log_every > n_iters → no callback fires.\n"
        "fs3, sk3 = cx20_train_with_sample_logging(n_iters=4, log_every=10, sample_fn=lambda s: 'x')\n"
        "assert fs3 == 4\n"
        "assert sk3 == {}, 'log_every > n_iters should produce no callbacks'\n"
        "\n"
        "# Case E: counter increments BEFORE the modulo check — so step 0 NEVER fires.\n"
        "# If a buggy impl checked `step % log_every == 0` BEFORE incrementing, step=0 would fire.\n"
        "fired_b = []\n"
        "fs4, sk4 = cx20_train_with_sample_logging(\n"
        "    n_iters=1, log_every=1, sample_fn=lambda s: fired_b.append(s) or s\n"
        ")\n"
        "# Right answer: fires once on step 1; sink == {1: 1}.\n"
        "# Wrong answer (check-before-increment): fires twice — step 0 and step 1.\n"
        "assert fired_b == [1], f'should fire exactly once on step 1; got {fired_b}'\n"
        "assert sk4 == {1: 1}, f'sink should be {{1: 1}}; got {sk4}'"
    ),
    "solution_body": (
        "def cx20_train_with_sample_logging(n_iters, log_every, sample_fn):\n"
        "    step = 0\n"
        "    sink = {}\n"
        "    for _ in range(n_iters):\n"
        "        # Atom A (step-counter-increment): bump BEFORE the modulo check so step 0 never fires.\n"
        "        step += 1\n"
        "        # Atom B (log-samples-eval-callback): every log_every steps, call sample_fn.\n"
        "        if step % log_every == 0:\n"
        "            sink[step] = sample_fn(step)\n"
        "    return step, sink"
    ),
    "solution_notes": (
        "The increment-then-check pattern matches PyTorch's `Optimizer.step()` convention (step "
        "counter incremented inside `.step()` before any logging hooks observe it). The "
        "alternative — check-then-increment — leads to off-by-one bugs where you log untrained "
        "model output at step 0, then log every `log_every` thereafter."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "kcs": ["log-samples-eval-callback", "step-counter-increment"],
    "lo": (
        "Compose step-counter increment with the every-K-steps sample-logging callback so the "
        "callback fires on steps K, 2K, 3K... and never on step 0."
    ),
}


# ===========================================================================
# cx21 — log generated samples as wandb.Image (eval callback + wandb.log)
# ===========================================================================
spec_21 = {
    "atom_ids": ["log-samples-eval-callback", "wandb-log-step"],
    "subtopics": _subs(["log-samples-eval-callback", "wandb-log-step"]),
    "primary_atom": "log-samples-eval-callback",
    "part": "part5",
    "exercise_index": 21,
    "exercise_title": "every-K-steps callback logs generated samples as wandb.Image",
    "slug": "log-samples-callback-as-wandb-image",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "For GAN / VAE training, the canonical eval signal is **generated image samples**. The way "
        "you ship them to wandb is to wrap each tensor in `wandb.Image(...)` and call "
        "`wandb.log({'samples': [...], 'step': step})`.\n\n"
        "**The two atoms.**\n"
        "- **log-samples-eval-callback** — fire every `K` steps; produce `N` samples from the "
        "generator.\n"
        "- **wandb-log-step** — call `wandb.log(payload)` with the step number and the wrapped "
        "samples. `wandb.Image` is `wandb`'s tagged-image type; the dashboard renders it as a "
        "panel of thumbnails.\n\n"
        "**Anatomy.**\n"
        "```python\n"
        "if step % log_every == 0:\n"
        "    samples = generator(z)                # shape (N, C, H, W)\n"
        "    images = [wandb.Image(s) for s in samples]\n"
        "    wandb.log({'samples': images, 'step': step})\n"
        "```\n\n"
        "**Why care about the wrapper.** `wandb.log({'samples': tensor})` silently logs a histogram, "
        "not an image panel. The `wandb.Image(...)` wrap is what tells wandb 'this is a picture, "
        "render it as one'. The test confirms `wandb.Image` was called exactly N times per fire."
    ),
    "prompt_body": (
        "Implement `cx21_log_samples_to_wandb(n_iters, log_every, generator, wandb)`.\n\n"
        "Inputs:\n"
        "- `n_iters` — int.\n"
        "- `log_every` — int. Fire callback when `step % log_every == 0` (and `step != 0`).\n"
        "- `generator` — callable `(step: int) -> t.Tensor`. Returns a batch of shape `(N, C, H, W)`.\n"
        "- `wandb` — the wandb module (mocked in tests).\n\n"
        "Required behaviour:\n"
        "1. Initialise `step = 0`.\n"
        "2. For each iteration:\n"
        "   - Increment `step`.\n"
        "   - If `step % log_every == 0`:\n"
        "     - Call `samples = generator(step)`; iterate the batch dim to get individual sample "
        "tensors.\n"
        "     - Wrap each in `wandb.Image(sample)` (atom: wandb.Image is wandb's image type).\n"
        "     - Call `wandb.log({'samples': images, 'step': step})` (atom: wandb-log-step).\n"
        "3. Return the final step.\n\n"
        "Test checks:\n"
        "- `wandb.log.call_count == n_iters // log_every`.\n"
        "- Each `wandb.log` payload has key `'samples'` (a list) and key `'step'` (the step number).\n"
        "- `wandb.Image` was called once per individual sample (so `N * (n_iters // log_every)` times "
        "total).\n"
        "- `wandb.Image` was called with a tensor of the per-sample shape `(C, H, W)` — not "
        "the full batch."
    ),
    "stub_body": (
        "def cx21_log_samples_to_wandb(n_iters, log_every, generator, wandb):\n"
        "    \"\"\"Every log_every steps, generate samples and log them to wandb as wandb.Image. Returns final step.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "from unittest.mock import MagicMock\n"
        "\n"
        "# Case A: basic — 10 iters, log every 4, 3-sample batches → 2 fires, 6 images logged.\n"
        "wb = MagicMock()\n"
        "def _gen(step):\n"
        "    return t.zeros(3, 1, 4, 4) + float(step)  # (N=3, C=1, H=4, W=4), all entries = step.\n"
        "final_step = cx21_log_samples_to_wandb(n_iters=10, log_every=4, generator=_gen, wandb=wb)\n"
        "assert final_step == 10\n"
        "assert wb.log.call_count == 2, f'should fire on steps 4 and 8; got {wb.log.call_count} fires'\n"
        "\n"
        "# Inspect each log call: payload dict with 'samples' (list) and 'step' (int).\n"
        "for i, call in enumerate(wb.log.call_args_list, start=1):\n"
        "    args, kwargs = call\n"
        "    assert len(args) == 1, f'call {i}: expected one positional dict; got {args}'\n"
        "    payload = args[0]\n"
        "    assert isinstance(payload, dict)\n"
        "    assert 'samples' in payload, f'payload missing key samples; got {list(payload.keys())}'\n"
        "    assert 'step' in payload\n"
        "    samples = payload['samples']\n"
        "    assert isinstance(samples, list), f'samples should be a list; got {type(samples).__name__}'\n"
        "    assert len(samples) == 3, f'expected N=3 wrapped images; got {len(samples)}'\n"
        "    assert payload['step'] == i * 4, f'step should be {i*4}; got {payload[\"step\"]}'\n"
        "\n"
        "# Case B: wandb.Image was called once per individual sample — total 2 fires * 3 samples = 6.\n"
        "assert wb.Image.call_count == 6, f'wandb.Image should be called once per sample; got {wb.Image.call_count}'\n"
        "\n"
        "# Case C: wandb.Image is called with per-sample shape (C, H, W), not the (N, C, H, W) batch.\n"
        "for call in wb.Image.call_args_list:\n"
        "    args, kwargs = call\n"
        "    assert len(args) >= 1, f'wandb.Image must be called with the sample tensor positionally; got args={args}'\n"
        "    arr = args[0]\n"
        "    assert hasattr(arr, 'shape'), f'wandb.Image arg must be a tensor; got {type(arr).__name__}'\n"
        "    assert arr.shape == (1, 4, 4), f'per-sample shape should be (C=1, H=4, W=4); got {tuple(arr.shape)}'\n"
        "\n"
        "# Case D: log_every > n_iters → no fires.\n"
        "wb2 = MagicMock()\n"
        "cx21_log_samples_to_wandb(n_iters=3, log_every=100, generator=_gen, wandb=wb2)\n"
        "assert wb2.log.call_count == 0\n"
        "assert wb2.Image.call_count == 0\n"
        "\n"
        "# Case E: step 0 never fires (counter increments before the modulo check).\n"
        "wb3 = MagicMock()\n"
        "calls_to_gen = []\n"
        "def _gen3(s):\n"
        "    calls_to_gen.append(s)\n"
        "    return t.zeros(2, 3, 8, 8)\n"
        "cx21_log_samples_to_wandb(n_iters=2, log_every=1, generator=_gen3, wandb=wb3)\n"
        "assert calls_to_gen == [1, 2], f'generator should be called with steps 1, 2 — not 0; got {calls_to_gen}'\n"
        "assert wb3.log.call_count == 2"
    ),
    "solution_body": (
        "def cx21_log_samples_to_wandb(n_iters, log_every, generator, wandb):\n"
        "    step = 0\n"
        "    for _ in range(n_iters):\n"
        "        step += 1\n"
        "        # Atom A (log-samples-eval-callback): fire every log_every steps.\n"
        "        if step % log_every == 0:\n"
        "            samples = generator(step)\n"
        "            # Wrap each sample in wandb.Image (one wrap per sample, not per batch).\n"
        "            images = [wandb.Image(s) for s in samples]\n"
        "            # Atom B (wandb-log-step): log payload with samples + step.\n"
        "            wandb.log({'samples': images, 'step': step})\n"
        "    return step"
    ),
    "solution_notes": (
        "Iterating `for s in samples` over a tensor of shape `(N, C, H, W)` yields per-sample "
        "tensors of shape `(C, H, W)` — exactly what `wandb.Image` expects. If you pass the whole "
        "batch as a single `wandb.Image(samples)`, the dashboard tries to render a 4-D array as one "
        "image and either fails or shows nonsense. The 1-wrap-per-sample contract is the test's "
        "main signal."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["log-samples-eval-callback", "wandb-log-step"],
    "lo": (
        "Compose every-K-steps sample-logging with wandb.Image wrapping + wandb.log so generated "
        "image batches show up as a panel of thumbnails in the wandb dashboard."
    ),
}


# ===========================================================================
# cx22 — eval callback then resume training (zero_grad)
# ===========================================================================
spec_22 = {
    "atom_ids": ["log-samples-eval-callback", "zero-grad-set-none"],
    "subtopics": _subs(["log-samples-eval-callback", "zero-grad-set-none"]),
    "primary_atom": "log-samples-eval-callback",
    "part": "part5",
    "exercise_index": 22,
    "exercise_title": "log-samples eval callback then zero_grad(set_to_none=True) to resume training",
    "slug": "eval-callback-then-zero-grad-resume",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "After a generator/decoder eval pass produces sample images, the trainer has to **reset "
        "the gradients** before the next backward — otherwise the next step's `.backward()` "
        "accumulates on top of the LAST training step's `.grad` (PyTorch's `.grad` accumulator is "
        "an *add*, not an overwrite).\n\n"
        "**The two atoms.**\n"
        "- **log-samples-eval-callback** — periodic sample generation; does NOT touch parameter "
        "grads (it should be wrapped in `no_grad` / `inference_mode` to avoid pollution, but even "
        "if not, no `.backward()` is called inside).\n"
        "- **zero-grad-set-none** — set each `p.grad = None` (NOT `p.grad.zero_()`). `set_to_none` "
        "is the modern PyTorch default because it skips the zero-fill and lets autograd allocate "
        "fresh storage next backward. Reads of `.grad` after `set_to_none` return `None`, not a "
        "zero tensor.\n\n"
        "**Why compose them.** If your trainer fires the eval callback INSIDE the same step where "
        "it forgets to call zero_grad, the NEXT training step sees old `.grad` + new `.grad`. With "
        "`set_to_none=True`, the contract is even stricter: after the call, `.grad` IS `None` and "
        "the next backward writes a fresh tensor.\n\n"
        "**Anatomy.**\n"
        "```python\n"
        "if step % log_every == 0:\n"
        "    samples = sample_fn()                   # log-samples-eval-callback.\n"
        "    sink[step] = samples\n"
        "for p in params:                            # zero-grad-set-none.\n"
        "    p.grad = None\n"
        "```"
    ),
    "prompt_body": (
        "Implement `cx22_eval_then_zero_grad(params, step, log_every, sample_fn, sink)`.\n\n"
        "Inputs:\n"
        "- `params` — list of `t.Tensor` leaves with `requires_grad=True`. They may have `.grad` "
        "populated from a prior step.\n"
        "- `step` — int.\n"
        "- `log_every` — int.\n"
        "- `sample_fn` — callable `() -> Any`.\n"
        "- `sink` — list; append `(step, sample)` to it when the callback fires.\n\n"
        "Required behaviour:\n"
        "1. If `step % log_every == 0`, call `sample_fn()` and append `(step, sample)` to `sink` "
        "(atom: log-samples-eval-callback).\n"
        "2. UNCONDITIONALLY (every call), for each `p in params`, set `p.grad = None` (atom: "
        "zero-grad-set-none). `None`, not a zero tensor.\n"
        "3. Return `None`.\n\n"
        "Test checks:\n"
        "- After the call, every `p.grad is None` — even if it was populated before.\n"
        "- Callback fires iff `step % log_every == 0`.\n"
        "- A subsequent backward+grad-population works (proves `set_to_none=True` semantics "
        "don't break autograd plumbing — autograd will allocate a fresh `.grad` tensor)."
    ),
    "stub_body": (
        "def cx22_eval_then_zero_grad(params, step, log_every, sample_fn, sink):\n"
        "    \"\"\"Maybe fire eval callback, then set every p.grad = None. Returns None.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: pre-populated grads are wiped to None.\n"
        "p1 = t.randn(3, requires_grad=True)\n"
        "p2 = t.randn(2, 4, requires_grad=True)\n"
        "p1.grad = t.ones_like(p1)\n"
        "p2.grad = t.full_like(p2, 7.0)\n"
        "sink_a = []\n"
        "ret = cx22_eval_then_zero_grad([p1, p2], step=4, log_every=2, sample_fn=lambda: 'x', sink=sink_a)\n"
        "assert ret is None\n"
        "assert p1.grad is None, f'p1.grad should be None after set_to_none semantics; got {p1.grad}'\n"
        "assert p2.grad is None, f'p2.grad should be None; got {p2.grad}'\n"
        "assert sink_a == [(4, 'x')], f'callback should have fired with (4, \"x\"); got {sink_a}'\n"
        "\n"
        "# Case B: callback does NOT fire when step % log_every != 0, but grad is STILL zeroed.\n"
        "p3 = t.tensor([1.0, 2.0], requires_grad=True)\n"
        "p3.grad = t.tensor([99.0, 99.0])\n"
        "sink_b = []\n"
        "calls_b = {'n': 0}\n"
        "def _sb():\n"
        "    calls_b['n'] += 1\n"
        "    return 'shouldnt-fire'\n"
        "cx22_eval_then_zero_grad([p3], step=5, log_every=4, sample_fn=_sb, sink=sink_b)\n"
        "assert p3.grad is None, 'grad must be wiped unconditionally — even when callback skipped'\n"
        "assert sink_b == [], 'callback should not fire when step % log_every != 0'\n"
        "assert calls_b['n'] == 0\n"
        "\n"
        "# Case C: subsequent backward populates fresh grad — autograd plumbing not broken.\n"
        "p4 = t.tensor([2.0, 3.0], requires_grad=True)\n"
        "p4.grad = t.ones(2)\n"
        "cx22_eval_then_zero_grad([p4], step=0, log_every=1, sample_fn=lambda: 'samp', sink=[])\n"
        "assert p4.grad is None\n"
        "# Now do a fresh backward and confirm grad gets populated correctly.\n"
        "(p4.sum()).backward()\n"
        "assert p4.grad is not None, 'after zero_grad(set_to_none=True) + new backward, grad should be populated'\n"
        "assert t.allclose(p4.grad, t.ones(2)), f'd(sum(p))/dp = 1; got {p4.grad.tolist()}'\n"
        "\n"
        "# Case D: params with grad=None to begin with stay None (no AttributeError).\n"
        "p5 = t.randn(4, requires_grad=True)\n"
        "assert p5.grad is None\n"
        "cx22_eval_then_zero_grad([p5], step=2, log_every=1, sample_fn=lambda: None, sink=[])\n"
        "assert p5.grad is None\n"
        "\n"
        "# Case E: empty param list runs fine.\n"
        "sink_e = []\n"
        "cx22_eval_then_zero_grad([], step=3, log_every=3, sample_fn=lambda: 'fire', sink=sink_e)\n"
        "assert sink_e == [(3, 'fire')], 'callback should still fire for empty params'\n"
        "\n"
        "# Case F: callback fires on step 0 too (0 % log_every == 0).\n"
        "p6 = t.randn(2, requires_grad=True)\n"
        "p6.grad = t.zeros(2)\n"
        "sink_f = []\n"
        "cx22_eval_then_zero_grad([p6], step=0, log_every=5, sample_fn=lambda: 'init', sink=sink_f)\n"
        "assert sink_f == [(0, 'init')], 'step 0 % 5 == 0, callback should fire'\n"
        "assert p6.grad is None"
    ),
    "solution_body": (
        "def cx22_eval_then_zero_grad(params, step, log_every, sample_fn, sink):\n"
        "    # Atom A (log-samples-eval-callback): conditional sample generation.\n"
        "    if step % log_every == 0:\n"
        "        sample = sample_fn()\n"
        "        sink.append((step, sample))\n"
        "    # Atom B (zero-grad-set-none): unconditional grad reset, set_to_none=True semantics.\n"
        "    for p in params:\n"
        "        p.grad = None"
    ),
    "solution_notes": (
        "The grad reset MUST be unconditional — even on steps where the eval callback doesn't fire, "
        "the trainer still finished a backward and needs a clean slate before the next one. Pairing "
        "the conditional eval with the unconditional zero_grad is the most common shape in real "
        "ARENA trainers. Note: setting `.grad = None` is preferred over `.grad.zero_()` because "
        "(a) it avoids a memset, and (b) it makes 'no backward happened yet' distinguishable from "
        "'a backward happened with all-zero gradients'."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "kcs": ["log-samples-eval-callback", "zero-grad-set-none"],
    "lo": (
        "Compose a step-gated eval/sample-logging callback with unconditional set_to_none zero_grad "
        "so a training step both produces a periodic sample sink and leaves grads cleared for the "
        "next backward."
    ),
}


# ===========================================================================
# cx23 — per-batch backward pass (dataloader + scalar backward)
# ===========================================================================
spec_23 = {
    "atom_ids": ["dataloader-batching", "backward-on-scalar-loss"],
    "subtopics": _subs(["dataloader-batching", "backward-on-scalar-loss"]),
    "primary_atom": "dataloader-batching",
    "part": "part5",
    "exercise_index": 23,
    "exercise_title": "DataLoader iterates batches, scalar-reduced loss backward per batch",
    "slug": "dataloader-then-per-batch-backward",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "The minimal real-data training step has two pieces:\n\n"
        "1. **DataLoader batching.** Wrap a `TensorDataset` in a `DataLoader(batch_size=B)` and "
        "iterate. Each iteration yields a tuple `(x_batch, y_batch)` of shape `(B, ...)`. The "
        "loader knows to slice the dataset into chunks of size `B` and shuffle if asked.\n"
        "2. **Scalar-reduced backward.** Compute a per-sample loss vector of shape `(B,)`, reduce "
        "to a scalar via `.mean()`, then `loss.backward()` to populate `param.grad`. Per-batch "
        "grads are *accumulated* if you don't zero between batches.\n\n"
        "**Anatomy.**\n"
        "```python\n"
        "loader = DataLoader(TensorDataset(x, y), batch_size=B)   # dataloader-batching.\n"
        "for xb, yb in loader:\n"
        "    per_sample = ((param * xb) - yb) ** 2                # shape (B,) ish.\n"
        "    loss = per_sample.mean()                             # backward-on-scalar-loss.\n"
        "    loss.backward()                                       # accumulates into param.grad.\n"
        "```"
    ),
    "prompt_body": (
        "Implement `cx23_dataloader_train(x_data, y_data, param, batch_size)`.\n\n"
        "Inputs:\n"
        "- `x_data` — `t.Tensor` of shape `(N,)`, the inputs.\n"
        "- `y_data` — `t.Tensor` of shape `(N,)`, the targets.\n"
        "- `param` — a scalar `t.Tensor` (shape `()`) with `requires_grad=True`. The 'model' is "
        "`y_hat = param * x`.\n"
        "- `batch_size` — int.\n\n"
        "Required behaviour:\n"
        "1. Build a `TensorDataset(x_data, y_data)`.\n"
        "2. Wrap in `DataLoader(ds, batch_size=batch_size, shuffle=False)` (atom: dataloader-batching).\n"
        "3. Initialise `losses = []`.\n"
        "4. For each `(xb, yb)` in the loader:\n"
        "   - Compute `per_sample = (param * xb - yb) ** 2` (shape `(B,)`).\n"
        "   - Reduce to a scalar via `.mean()` (atom: backward-on-scalar-loss).\n"
        "   - Call `.backward()`.\n"
        "   - Append the scalar loss as a Python float to `losses`.\n"
        "5. Return `losses`.\n\n"
        "Do **NOT** zero grads between batches — the test relies on grad accumulating across all "
        "batches (so `param.grad` at the end equals the SUM of per-batch grads, demonstrating "
        "accumulation semantics).\n\n"
        "Test checks:\n"
        "- `losses` has the right length (`ceil(N / batch_size)`).\n"
        "- Each loss is a Python float (proves you reduced to scalar before `.item()`).\n"
        "- `param.grad` accumulates across batches.\n"
        "- The total accumulated grad equals what you'd get from one big batch — `.mean()` per "
        "batch then summed across batches DOES NOT equal one big `.mean()` unless batches are "
        "equal-size, but the SUM of `.sum()` would. The test uses an N divisible by batch_size "
        "to keep the math clean."
    ),
    "stub_body": (
        "def cx23_dataloader_train(x_data, y_data, param, batch_size):\n"
        "    \"\"\"Iterate a DataLoader, scalar-reduce per-batch loss, backward. Returns list of batch losses.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: simple — N=8, batch_size=4 → 2 batches, 2 losses.\n"
        "t.manual_seed(0)\n"
        "x = t.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])\n"
        "y = 2.0 * x   # true param = 2.\n"
        "param = t.tensor(0.5, requires_grad=True)\n"
        "losses = cx23_dataloader_train(x, y, param, batch_size=4)\n"
        "assert isinstance(losses, list)\n"
        "assert len(losses) == 2, f'N=8, B=4 → 2 batches; got {len(losses)} losses'\n"
        "for L in losses:\n"
        "    assert isinstance(L, float), f'each loss should be a Python float (proves scalar reduction); got {type(L).__name__}'\n"
        "assert param.grad is not None, 'param.grad must be populated after backward'\n"
        "\n"
        "# Case B: grad accumulates — final grad equals sum of per-batch grads.\n"
        "# We can recompute the expected grad: each batch contributes mean of d/dparam (param*x - y)^2.\n"
        "# Total grad = sum_{batches} mean_b ( 2*(param*x - y)*x ).\n"
        "param2 = t.tensor(0.5, requires_grad=True)\n"
        "expected_grad = 0.0\n"
        "x_batches = [x[0:4], x[4:8]]\n"
        "y_batches = [y[0:4], y[4:8]]\n"
        "for xb, yb in zip(x_batches, y_batches):\n"
        "    g_batch = (2 * (param2.item() * xb - yb) * xb).mean().item()\n"
        "    expected_grad += g_batch\n"
        "losses2 = cx23_dataloader_train(x, y, param2, batch_size=4)\n"
        "assert abs(param2.grad.item() - expected_grad) < 1e-5, (\n"
        "    f'expected accumulated grad {expected_grad:.5f}; got {param2.grad.item():.5f}'\n"
        ")\n"
        "\n"
        "# Case C: batch_size=1 → N batches of size 1 each.\n"
        "x3 = t.tensor([1.0, 2.0, 3.0])\n"
        "y3 = t.tensor([2.0, 4.0, 6.0])\n"
        "param3 = t.tensor(0.0, requires_grad=True)\n"
        "losses3 = cx23_dataloader_train(x3, y3, param3, batch_size=1)\n"
        "assert len(losses3) == 3, f'B=1 should give N=3 batches; got {len(losses3)}'\n"
        "assert param3.grad is not None\n"
        "\n"
        "# Case D: batch_size > N → 1 batch with all data (DataLoader's last-batch behaviour).\n"
        "x4 = t.tensor([1.0, 2.0, 3.0])\n"
        "y4 = t.tensor([2.0, 4.0, 6.0])\n"
        "param4 = t.tensor(0.0, requires_grad=True)\n"
        "losses4 = cx23_dataloader_train(x4, y4, param4, batch_size=10)\n"
        "assert len(losses4) == 1, f'B=10 > N=3 should give exactly 1 batch; got {len(losses4)}'\n"
        "\n"
        "# Case E: loss values are roughly right magnitude.\n"
        "# For param=0, y=2x → per-sample loss = (0 - 2x)^2 = 4x^2.\n"
        "# Batch 1 (x=1,2,3): mean(4 + 16 + 36) = 56/3 ≈ 18.67.\n"
        "assert abs(losses4[0] - 56.0 / 3.0) < 1e-4, f'expected ~{56/3:.4f}; got {losses4[0]:.4f}'\n"
        "\n"
        "# Case F: backward populates grad even when batches have unequal sizes (last partial batch).\n"
        "x5 = t.arange(7.0)   # N=7.\n"
        "y5 = 3.0 * x5\n"
        "param5 = t.tensor(0.0, requires_grad=True)\n"
        "losses5 = cx23_dataloader_train(x5, y5, param5, batch_size=3)\n"
        "# N=7, B=3 → batches of sizes 3, 3, 1.\n"
        "assert len(losses5) == 3, f'N=7 B=3 → 3 batches (3+3+1); got {len(losses5)}'\n"
        "assert param5.grad is not None\n"
        "assert not t.isnan(param5.grad), 'grad must not be NaN for unequal batch sizes'"
    ),
    "solution_body": (
        "def cx23_dataloader_train(x_data, y_data, param, batch_size):\n"
        "    # Atom A (dataloader-batching): wrap tensors in TensorDataset → DataLoader.\n"
        "    ds = TensorDataset(x_data, y_data)\n"
        "    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)\n"
        "    losses = []\n"
        "    for xb, yb in loader:\n"
        "        # Per-sample squared error, shape (B,).\n"
        "        per_sample = (param * xb - yb) ** 2\n"
        "        # Atom B (backward-on-scalar-loss): reduce to scalar, then backward.\n"
        "        loss = per_sample.mean()\n"
        "        loss.backward()\n"
        "        losses.append(loss.item())\n"
        "    return losses"
    ),
    "solution_notes": (
        "`shuffle=False` is required for the test's grad-accumulation check to compare against a "
        "deterministic per-batch slicing of the data. In a real trainer you'd `shuffle=True` and "
        "call `optimizer.zero_grad()` BEFORE each batch's backward — that's the topic of cx22 / "
        "cx24. The composition isolated here is just 'data → batches → scalar backward'."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "kcs": ["dataloader-batching", "backward-on-scalar-loss"],
    "lo": (
        "Compose a TensorDataset+DataLoader batched iteration with per-batch scalar-reduced loss "
        "+ backward so the trainer accumulates gradients across all batches in a deterministic "
        "shuffle-off pass."
    ),
}


# ===========================================================================
# cx24 — batch loop with step counter (dataloader + step-counter-increment)
# ===========================================================================
spec_24 = {
    "atom_ids": ["dataloader-batching", "step-counter-increment"],
    "subtopics": _subs(["dataloader-batching", "step-counter-increment"]),
    "primary_atom": "dataloader-batching",
    "part": "part5",
    "exercise_index": 24,
    "exercise_title": "DataLoader batch loop with monotonically increasing step counter",
    "slug": "dataloader-batch-loop-step-counter",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "A real trainer needs to count **micro-batches across epochs**, not just iterations within "
        "one epoch. The composition:\n\n"
        "1. **DataLoader batching** — each epoch yields `ceil(N/B)` batches. The same `DataLoader` "
        "instance can be iterated multiple times (one outer `for epoch in range(E):`).\n"
        "2. **Step counter increment** — `step` is incremented ONCE per batch, across ALL epochs. "
        "After 3 epochs of 4 batches each, `step == 12`, not `step == 4`.\n\n"
        "**Why care.** Logging and LR schedules key off the global step, not the per-epoch step. "
        "If you reset `step` at the top of each epoch, your wandb x-axis goes backwards every "
        "epoch — the tell-tale visual symptom of this bug.\n\n"
        "**Anatomy.**\n"
        "```python\n"
        "loader = DataLoader(ds, batch_size=B)         # dataloader-batching.\n"
        "step = 0\n"
        "for epoch in range(n_epochs):\n"
        "    for xb, yb in loader:\n"
        "        step += 1                              # step-counter-increment, INSIDE batch loop.\n"
        "        # ... train ...\n"
        "```"
    ),
    "prompt_body": (
        "Implement `cx24_dataloader_step_counter(x_data, y_data, batch_size, n_epochs)`.\n\n"
        "Inputs:\n"
        "- `x_data`, `y_data` — `t.Tensor`s of shape `(N,)` each.\n"
        "- `batch_size` — int.\n"
        "- `n_epochs` — int.\n\n"
        "Required behaviour:\n"
        "1. Build a `TensorDataset(x_data, y_data)` and wrap in `DataLoader(ds, batch_size=batch_size, shuffle=False)` "
        "(atom: dataloader-batching).\n"
        "2. Initialise `step = 0` BEFORE the epoch loop (NOT inside).\n"
        "3. Initialise `step_history = []` — a list to which you append the GLOBAL step number "
        "after every batch.\n"
        "4. For each epoch in `range(n_epochs)`:\n"
        "   - For each `(xb, yb)` in the loader:\n"
        "     - Increment `step += 1` FIRST (atom: step-counter-increment).\n"
        "     - Append `step` to `step_history`.\n"
        "5. Return `(step, step_history)`.\n\n"
        "Test checks:\n"
        "- `step` is exactly `n_epochs * ceil(N / batch_size)`.\n"
        "- `step_history == [1, 2, 3, ...]` — monotonically increasing from 1, ONE PER BATCH, "
        "across all epochs.\n"
        "- A buggy 'reset step at top of epoch' would produce `[1, 2, 3, 4, 1, 2, 3, 4, ...]` "
        "which the test catches.\n"
        "- Zero epochs yields `step == 0` and `step_history == []`."
    ),
    "stub_body": (
        "def cx24_dataloader_step_counter(x_data, y_data, batch_size, n_epochs):\n"
        "    \"\"\"Iterate a DataLoader for n_epochs; return (final_step, step_history). Step counter is global.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "import math\n"
        "\n"
        "# Case A: 2 epochs * 4 batches/epoch = 8 total steps.\n"
        "x = t.arange(8.0)   # N=8.\n"
        "y = 2.0 * x\n"
        "final_step, hist = cx24_dataloader_step_counter(x, y, batch_size=2, n_epochs=2)\n"
        "# N=8, B=2 → 4 batches per epoch; 2 epochs → 8 total.\n"
        "assert final_step == 8, f'expected final step 8; got {final_step}'\n"
        "assert hist == [1, 2, 3, 4, 5, 6, 7, 8], f'step history must be monotonic 1..8; got {hist}'\n"
        "\n"
        "# Case B: zero epochs → step stays 0, history is empty.\n"
        "fs2, hist2 = cx24_dataloader_step_counter(x, y, batch_size=2, n_epochs=0)\n"
        "assert fs2 == 0\n"
        "assert hist2 == []\n"
        "\n"
        "# Case C: 3 epochs * 3 batches (N=6, B=2) = 9 total steps.\n"
        "x3 = t.arange(6.0)\n"
        "y3 = t.arange(6.0)\n"
        "fs3, hist3 = cx24_dataloader_step_counter(x3, y3, batch_size=2, n_epochs=3)\n"
        "assert fs3 == 9, f'3 epochs * 3 batches = 9 steps; got {fs3}'\n"
        "assert hist3 == list(range(1, 10)), f'expected [1..9]; got {hist3}'\n"
        "\n"
        "# Case D: partial last batch is still ONE batch step.\n"
        "x4 = t.arange(7.0)   # N=7, B=3 → batches of 3,3,1 = 3 batches.\n"
        "y4 = t.arange(7.0)\n"
        "fs4, hist4 = cx24_dataloader_step_counter(x4, y4, batch_size=3, n_epochs=2)\n"
        "# 2 epochs * 3 batches = 6 total.\n"
        "assert fs4 == 6, f'2 epochs * 3 batches (incl partial) = 6; got {fs4}'\n"
        "assert hist4 == [1, 2, 3, 4, 5, 6]\n"
        "\n"
        "# Case E: step counter does NOT reset between epochs — explicit guard.\n"
        "# A buggy impl that does `step = 0` inside the epoch loop would give [1,2,3,1,2,3].\n"
        "# Our correct impl gives [1,2,3,4,5,6]. We assert the correct sequence above. Also confirm\n"
        "# that hist[len(epoch1):] starts at len(epoch1)+1, NOT at 1.\n"
        "batches_per_epoch = math.ceil(7 / 3)   # = 3.\n"
        "first_step_of_epoch2 = hist4[batches_per_epoch]\n"
        "assert first_step_of_epoch2 == batches_per_epoch + 1, (\n"
        "    f'first step of epoch 2 should be {batches_per_epoch + 1} (no reset); '\n"
        "    f'got {first_step_of_epoch2} — looks like the step counter was reset at top of epoch'\n"
        ")"
    ),
    "solution_body": (
        "def cx24_dataloader_step_counter(x_data, y_data, batch_size, n_epochs):\n"
        "    # Atom A (dataloader-batching).\n"
        "    ds = TensorDataset(x_data, y_data)\n"
        "    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)\n"
        "    # Step counter lives OUTSIDE the epoch loop — global across all epochs.\n"
        "    step = 0\n"
        "    step_history = []\n"
        "    for _epoch in range(n_epochs):\n"
        "        for _xb, _yb in loader:\n"
        "            # Atom B (step-counter-increment): bump BEFORE recording.\n"
        "            step += 1\n"
        "            step_history.append(step)\n"
        "    return step, step_history"
    ),
    "solution_notes": (
        "The trap this exercise catches is putting `step = 0` *inside* the epoch loop. It looks "
        "innocuous because it 'works' within a single epoch, but the wandb dashboard, the LR "
        "scheduler, and the eval-every-K-steps callback all silently misbehave. The fix is just "
        "moving the assignment one indent level up — but it requires understanding the contract "
        "(global step) over the local pattern (per-epoch loop)."
    ),
    "extra_imports": NN_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 2,
    "kcs": ["dataloader-batching", "step-counter-increment"],
    "lo": (
        "Compose a DataLoader batch loop with a GLOBAL step counter so step indexes monotonically "
        "across epochs (never resetting at the top of each epoch)."
    ),
}


SPECS = [spec_19, spec_20, spec_21, spec_22, spec_23, spec_24]


def main() -> None:
    for spec in SPECS:
        out = emit_composite(spec)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
