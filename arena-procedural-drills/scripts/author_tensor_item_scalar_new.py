#!/usr/bin/env python3
"""Author Colab-native standalones (ex6-ex8) for atom `tensor-item-scalar`.

Each exercise exercises something flashcards cannot deliver: a multi-step
training loop, an early-stopping debug pipeline, or a stochastic simulation
visualization.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_standalone import emit_standalone

ATOM_ID = "tensor-item-scalar"
SUBTOPIC = "Numpy: Core array literacy"
TOPIC = "prereqs_numpy"

RECAP = (
    "## torch `.item()` — quick refresher\n"
    "\n"
    "`tensor.item()` extracts a Python scalar (`int`, `float`, or `bool`) from "
    "a 0-D or 1-element tensor. It detaches from the autograd graph and pulls "
    "the value back to CPU. This is the canonical bridge from tensor-world to "
    "Python-world: logging, control flow, plotting, and stop conditions all "
    "need scalars.\n"
    "\n"
    "**Calling `.item()` on a multi-element tensor raises.** Use `.tolist()` "
    "if you want every element as a Python list. Calling `.item()` inside "
    "a hot inner loop forces a CPU sync — fine for diagnostics, expensive in "
    "the training step itself."
)


SPECS = [
    # --------------------------------------------------------------- ex6
    {
        "atom_id": ATOM_ID,
        "subtopic": SUBTOPIC,
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP,
        "exercise_index": 6,
        "exercise_title": "training loop with .item() logging and loss curve",
        "slug": "training-loop-with-item-logging-and-loss-curve",
        "bloom_level": "Create",
        "difficulty_num": 4,
        "difficulty_dots": "🔴🔴🔴🔴⚪",
        "keywords": ["training-loop", "logging", "loss-curve", "visualization"],
        "kcs": ["item-zero-d-extract", "item-control-flow"],
        "lo": (
            "Combine a 3-step gradient-descent loop with `.item()` extraction "
            "to build a Python-side `losses` list and plot the loss curve."
        ),
        "prompt_body": (
            "Implement `ex6_train(x_init, target, lr, n_steps)`. A minimal "
            "gradient-descent loop that demonstrates the canonical use of "
            "`.item()` for logging:\n\n"
            "1. Start from `x = x_init.clone().detach().requires_grad_(True)`.\n"
            "2. For `n_steps` iterations:\n"
            "   a. Compute `loss = ((x - target) ** 2).sum()`. (Mean-squared "
            "error against the target.)\n"
            "   b. **Append `loss.item()` to a Python list** — this is the "
            "scalar extraction. Without `.item()` you'd accumulate a graph "
            "of tensors and leak memory.\n"
            "   c. Manually zero `x.grad` if it exists, call `loss.backward()`, "
            "and update `x.data -= lr * x.grad`.\n"
            "3. Return `(x.detach(), losses_list)`.\n\n"
            "Inputs:\n"
            "- `x_init`: starting 1-D float tensor.\n"
            "- `target`: same-shape target tensor.\n"
            "- `lr`: float learning rate.\n"
            "- `n_steps`: int.\n\n"
            "Output: tuple `(x_final, losses)`. `x_final` is a detached "
            "tensor; `losses` is a list of `n_steps` Python floats.\n\n"
            "The visualization plots the loss curve so you can see the "
            "convergence (or divergence) of your loop."
        ),
        "stub": (
            "def ex6_train(x_init: Tensor, target: Tensor, lr: float, n_steps: int) -> tuple:\n"
            '    """3-step GD with .item() logging. Returns (x_final, losses)."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "x_init = t.tensor([5.0, -3.0, 2.0])\n"
            "target = t.tensor([0.0, 0.0, 0.0])\n"
            "x_final, losses = ex6_train(x_init, target, lr=0.1, n_steps=5)\n"
            "# Type checks.\n"
            "assert isinstance(losses, list), f'losses must be a Python list, got {type(losses)}'\n"
            "assert len(losses) == 5, f'expected 5 losses, got {len(losses)}'\n"
            "for i, lv in enumerate(losses):\n"
            "    assert isinstance(lv, float), f'losses[{i}] must be a Python float, got {type(lv)}'\n"
            "assert isinstance(x_final, Tensor)\n"
            "assert not x_final.requires_grad, 'x_final should be detached'\n"
            "# Loss must decrease monotonically for this convex problem.\n"
            "for i in range(1, 5):\n"
            "    assert losses[i] < losses[i-1], (\n"
            "        f'loss not decreasing at step {i}: {losses[i-1]:.4f} → {losses[i]:.4f}'\n"
            "    )\n"
            "# First loss should equal sum((x_init - target)**2) = 25 + 9 + 4 = 38.\n"
            "assert abs(losses[0] - 38.0) < 1e-4, f'expected first loss 38.0, got {losses[0]}'\n"
            "# After 5 steps with lr=0.1, x should be much closer to target.\n"
            "assert (x_final.abs().max().item() < x_init.abs().max().item()), 'x should approach target'\n"
            "\n"
            "# --- Longer-horizon loss curve visualization ---\n"
            "x_big = t.tensor([10.0, -7.0, 3.0, 5.0, -2.0])\n"
            "target_big = t.zeros(5)\n"
            "_, big_losses = ex6_train(x_big, target_big, lr=0.05, n_steps=40)\n"
            "fig, ax = plt.subplots(figsize=(7, 3))\n"
            "ax.plot(big_losses, marker='o', markersize=3, color='darkblue')\n"
            "ax.set_xlabel('step')\n"
            "ax.set_ylabel('loss (sum of squares)')\n"
            "ax.set_title(f'ex6 training-loop loss curve (lr=0.05, n_steps=40)')\n"
            "ax.set_yscale('log')\n"
            "ax.grid(True, alpha=0.3)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        "solution_body": (
            "def ex6_train(x_init: Tensor, target: Tensor, lr: float, n_steps: int) -> tuple:\n"
            "    x = x_init.clone().detach().requires_grad_(True)\n"
            "    losses = []\n"
            "    for _ in range(n_steps):\n"
            "        loss = ((x - target) ** 2).sum()\n"
            "        losses.append(loss.item())  # scalar extract — no graph leak\n"
            "        if x.grad is not None:\n"
            "            x.grad.zero_()\n"
            "        loss.backward()\n"
            "        with t.no_grad():\n"
            "            x -= lr * x.grad\n"
            "    return x.detach(), losses"
        ),
        "solution_notes": (
            "**Why `.item()` and not just `losses.append(loss)`?** A bare "
            "`loss` is a 0-D tensor still wired into the autograd graph. "
            "Append 1000 of them and you're holding 1000 graphs in memory, "
            "which is exactly the leak that catches every PyTorch beginner. "
            "`.item()` cuts the graph and returns a plain Python float.\n\n"
            "**Why detach `x` at the end.** Returning a tensor with "
            "`requires_grad=True` invites the caller to accidentally chain "
            "more autograd onto an old graph. `.detach()` returns a fresh "
            "view with no graph history.\n\n"
            "**Why `with t.no_grad()` around the update.** The update is a "
            "Python operation that mutates `x.data`. Without `no_grad`, it'd "
            "be tracked as an op in the next backward — wrong. Inside the "
            "block, autograd ignores the write."
        ),
        "extra_imports": ["import matplotlib.pyplot as plt"],
    },
    # --------------------------------------------------------------- ex7
    {
        "atom_id": ATOM_ID,
        "subtopic": SUBTOPIC,
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP,
        "exercise_index": 7,
        "exercise_title": "early stopping via .item() threshold",
        "slug": "early-stopping-via-item-threshold",
        "bloom_level": "Apply",
        "difficulty_num": 3,
        "difficulty_dots": "🔴🔴🔴⚪⚪",
        "keywords": ["early-stopping", "control-flow", "threshold", "while-loop"],
        "kcs": ["item-zero-d-extract", "item-control-flow"],
        "lo": (
            "Use `.item()` to extract a scalar criterion each iteration and "
            "drive a Python `while` loop that stops when the criterion falls "
            "below a threshold."
        ),
        "prompt_body": (
            "Implement `ex7_iterate_until(x_init, decay, threshold, max_iter)`. "
            "The canonical use of `.item()` in control flow:\n\n"
            "1. Start from `x = x_init.clone()`.\n"
            "2. For up to `max_iter` iterations:\n"
            "   a. Compute the L2 norm: `crit = x.norm()` (a 0-D tensor).\n"
            "   b. Extract the scalar with `crit.item()`. **Print** "
            "`'iter {i}: norm={value:.4f}'` so you can see the descent.\n"
            "   c. If `crit.item() < threshold`, break out of the loop.\n"
            "   d. Otherwise, decay: `x = x * decay`.\n"
            "3. Return a dict: `{'x': x, 'iters_done': i + 1, 'final_norm': "
            "crit.item(), 'stopped_early': bool}`.\n\n"
            "Why `.item()` and not just `if crit < threshold`? In a `while` "
            "condition or `if` statement, Python implicitly calls `bool()` "
            "on the tensor. For a 0-D tensor this works but emits warnings "
            "in some torch versions; for multi-element tensors it raises. "
            "Calling `.item()` explicitly is the unambiguous, fast, and "
            "future-proof move.\n\n"
            "Inputs:\n"
            "- `x_init`: 1-D float tensor.\n"
            "- `decay`: float in (0, 1).\n"
            "- `threshold`: float > 0.\n"
            "- `max_iter`: int.\n\n"
            "The visualization plots `norm` vs iteration and marks the "
            "early-stop point if it triggered."
        ),
        "stub": (
            "def ex7_iterate_until(x_init: Tensor, decay: float, threshold: float, max_iter: int) -> dict:\n"
            '    """Decay x by `decay` each step; stop when norm < threshold via .item()."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "x_init = t.tensor([3.0, 4.0])  # norm = 5.0\n"
            "result = ex7_iterate_until(x_init, decay=0.5, threshold=1.0, max_iter=20)\n"
            "assert isinstance(result, dict), f'must return dict, got {type(result)}'\n"
            "assert set(result.keys()) >= {'x', 'iters_done', 'final_norm', 'stopped_early'}, (\n"
            "    f'missing required keys, got {result.keys()}'\n"
            "        )\n"
            "# 5.0 * 0.5^n < 1.0 → n > log2(5) ≈ 2.32 → n=3 → 4 iterations executed (0, 1, 2, 3).\n"
            "# But the check happens BEFORE decay, so iter 0 checks norm 5.0,\n"
            "# iter 1 checks 2.5, iter 2 checks 1.25, iter 3 checks 0.625 → break.\n"
            "assert result['stopped_early'] is True, 'must stop early before max_iter'\n"
            "assert result['iters_done'] == 4, f'expected 4 iters, got {result[\"iters_done\"]}'\n"
            "assert isinstance(result['final_norm'], float), f'final_norm must be a Python float, got {type(result[\"final_norm\"])}'\n"
            "assert result['final_norm'] < 1.0, f'final norm must be < threshold, got {result[\"final_norm\"]}'\n"
            "# x should reflect the decays.\n"
            "expected_x = x_init * (0.5 ** 3)  # 3 decays applied before the break check\n"
            "assert t.allclose(result['x'], expected_x, atol=1e-5), f'x mismatch: {result[\"x\"]} vs {expected_x}'\n"
            "\n"
            "# Edge case — already below threshold on iter 0.\n"
            "below = ex7_iterate_until(t.tensor([0.1, 0.1]), decay=0.5, threshold=1.0, max_iter=10)\n"
            "assert below['stopped_early'] is True\n"
            "assert below['iters_done'] == 1, f'instant stop should be 1 iter, got {below[\"iters_done\"]}'\n"
            "\n"
            "# Edge case — never reaches threshold within max_iter.\n"
            "stuck = ex7_iterate_until(t.tensor([100.0]), decay=0.99, threshold=0.01, max_iter=5)\n"
            "assert stuck['stopped_early'] is False, 'must not claim early stop if hit max_iter'\n"
            "assert stuck['iters_done'] == 5\n"
            "\n"
            "# --- Visualization with stop marker ---\n"
            "norms_track = []\n"
            "x_track = t.tensor([8.0, 6.0])\n"
            "thr = 0.3\n"
            "for i in range(30):\n"
            "    norms_track.append(x_track.norm().item())\n"
            "    if norms_track[-1] < thr:\n"
            "        break\n"
            "    x_track = x_track * 0.85\n"
            "fig, ax = plt.subplots(figsize=(7, 3))\n"
            "ax.plot(norms_track, marker='o', markersize=4, color='teal')\n"
            "ax.axhline(thr, color='red', linestyle='--', label=f'threshold={thr}')\n"
            "ax.axvline(len(norms_track) - 1, color='gray', linestyle=':', label=f'stop @ iter {len(norms_track) - 1}')\n"
            "ax.set_xlabel('iteration')\n"
            "ax.set_ylabel('norm (extracted via .item())')\n"
            "ax.set_title('ex7 early stopping — exits when norm < threshold')\n"
            "ax.set_yscale('log')\n"
            "ax.legend()\n"
            "ax.grid(True, alpha=0.3)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        "solution_body": (
            "def ex7_iterate_until(x_init: Tensor, decay: float, threshold: float, max_iter: int) -> dict:\n"
            "    x = x_init.clone()\n"
            "    stopped_early = False\n"
            "    final_norm = float('inf')\n"
            "    i = 0\n"
            "    for i in range(max_iter):\n"
            "        crit = x.norm()\n"
            "        val = crit.item()\n"
            "        print(f'  iter {i}: norm={val:.4f}')\n"
            "        final_norm = val\n"
            "        if val < threshold:\n"
            "            stopped_early = True\n"
            "            break\n"
            "        x = x * decay\n"
            "    return {\n"
            "        'x': x,\n"
            "        'iters_done': i + 1,\n"
            "        'final_norm': final_norm,\n"
            "        'stopped_early': stopped_early,\n"
            "    }"
        ),
        "solution_notes": (
            "**`.item()` is the explicit form of tensor → Python.** PyTorch "
            "will sometimes implicitly convert (e.g. `if x:` on a 0-D bool "
            "tensor) but the rules are quietly version-dependent. Calling "
            "`.item()` is unambiguous, fast (single CPU read), and works the "
            "same in every torch release.\n\n"
            "**CPU sync cost.** On GPU, `.item()` forces a host-device sync — "
            "the CPU has to wait for the GPU to finish the op that produced "
            "the scalar. In a hot training loop, calling `.item()` every "
            "step can dominate runtime. Reserve it for periodic logging and "
            "stop conditions, not for things you do per-batch.\n\n"
            "**Sentinel `iters_done`.** Returning `i + 1` works whether you "
            "broke early or completed all iterations because `i` survives "
            "after the `for` loop ends in Python (it doesn't have its own "
            "scope)."
        ),
        "extra_imports": ["import matplotlib.pyplot as plt"],
    },
    # --------------------------------------------------------------- ex8
    {
        "atom_id": ATOM_ID,
        "subtopic": SUBTOPIC,
        "topic_folder": TOPIC,
        "atom_recap_md": RECAP,
        "exercise_index": 8,
        "exercise_title": "random walk histogram via .item()",
        "slug": "random-walk-histogram-via-item",
        "bloom_level": "Create",
        "difficulty_num": 4,
        "difficulty_dots": "🔴🔴🔴🔴⚪",
        "keywords": ["random-walk", "simulation", "histogram", "visualization"],
        "kcs": ["item-zero-d-extract", "item-vs-tolist"],
        "lo": (
            "Run a stochastic simulation that accumulates scalar tensors via "
            "`.item()` into a Python list, then visualize the empirical "
            "distribution as a histogram."
        ),
        "prompt_body": (
            "Implement `ex8_random_walks(n_walks, n_steps, generator)`. A "
            "Monte-Carlo simulation that combines tensor ops with "
            "Python-scalar accumulation:\n\n"
            "1. Run `n_walks` independent 1-D random walks of `n_steps` each. "
            "Per walk: start at `0.0`, repeatedly add a sample from the "
            "standard normal distribution.\n"
            "2. Implementation: compute all increments at once with "
            "`t.randn(n_walks, n_steps, generator=generator)`, sum along the "
            "step axis to get a `(n_walks,)` tensor of final positions.\n"
            "3. **For each walk, call `.item()` on its final position** and "
            "append to a Python list `final_positions`. (Bulk `.tolist()` "
            "would also work — see solution notes — but the per-walk "
            "`.item()` loop is the version this exercise drills.)\n"
            "4. Return a dict: `{'final_positions': list_of_floats, "
            "'sample_walks': (5, n_steps) tensor of cumulative paths for "
            "the first 5 walks}`.\n\n"
            "Inputs:\n"
            "- `n_walks`: int.\n"
            "- `n_steps`: int.\n"
            "- `generator`: `torch.Generator` (for reproducibility — pass "
            "it through to `t.randn`).\n\n"
            "The visualization is a two-panel plot: trajectories of the 5 "
            "sample walks (left), histogram of `final_positions` overlaid "
            "with the theoretical N(0, √n_steps) curve (right)."
        ),
        "stub": (
            "def ex8_random_walks(n_walks: int, n_steps: int, generator: t.Generator) -> dict:\n"
            '    """Simulate random walks; extract per-walk final position via .item()."""\n'
            "    raise NotImplementedError()"
        ),
        "test_body": (
            "rng = t.Generator().manual_seed(2025)\n"
            "result = ex8_random_walks(n_walks=500, n_steps=100, generator=rng)\n"
            "assert isinstance(result, dict)\n"
            "assert set(result.keys()) >= {'final_positions', 'sample_walks'}, f'missing keys: {result.keys()}'\n"
            "fp = result['final_positions']\n"
            "assert isinstance(fp, list), f'final_positions must be a Python list, got {type(fp)}'\n"
            "assert len(fp) == 500, f'expected 500 entries, got {len(fp)}'\n"
            "for i, v in enumerate(fp[:10]):\n"
            "    assert isinstance(v, float), f'final_positions[{i}] must be a Python float (from .item()), got {type(v)}'\n"
            "# Statistical sanity — mean ≈ 0, std ≈ sqrt(n_steps) = 10.\n"
            "import statistics as _stats\n"
            "mean = _stats.mean(fp)\n"
            "stdev = _stats.stdev(fp)\n"
            "assert abs(mean) < 1.5, f'expected mean near 0, got {mean:.3f}'\n"
            "assert 8.0 < stdev < 12.0, f'expected stdev near 10, got {stdev:.3f}'\n"
            "# sample_walks shape check.\n"
            "sw = result['sample_walks']\n"
            "assert isinstance(sw, Tensor)\n"
            "assert sw.shape == (5, 100), f'expected (5, 100), got {tuple(sw.shape)}'\n"
            "# Cumulative paths must start near 0 (first step) and drift from there.\n"
            "# Each row's last value must match its final_position entry (within fp accuracy).\n"
            "for i in range(5):\n"
            "    assert abs(sw[i, -1].item() - fp[i]) < 1e-4, (\n"
            "        f'sample_walks[{i}, -1] = {sw[i, -1].item()} but final_positions[{i}] = {fp[i]}'\n"
            "    )\n"
            "\n"
            "# --- Two-panel visualization ---\n"
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))\n"
            "for i in range(5):\n"
            "    ax1.plot(sw[i].numpy(), alpha=0.7)\n"
            "ax1.set_title('ex8 — first 5 sample walks')\n"
            "ax1.set_xlabel('step')\n"
            "ax1.set_ylabel('position')\n"
            "ax1.grid(True, alpha=0.3)\n"
            "\n"
            "ax2.hist(fp, bins=30, density=True, color='lightcoral', edgecolor='black', label='empirical')\n"
            "# Theoretical N(0, sqrt(n_steps)) density.\n"
            "xs_th = np.linspace(min(fp), max(fp), 200)\n"
            "sigma = 10.0\n"
            "ys_th = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (xs_th / sigma) ** 2)\n"
            "ax2.plot(xs_th, ys_th, 'navy', linewidth=2, label='N(0, √100)')\n"
            "ax2.set_title('ex8 — final positions vs theory')\n"
            "ax2.set_xlabel('final position (extracted via .item())')\n"
            "ax2.set_ylabel('density')\n"
            "ax2.legend()\n"
            "ax2.grid(True, alpha=0.3)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        "solution_body": (
            "def ex8_random_walks(n_walks: int, n_steps: int, generator: t.Generator) -> dict:\n"
            "    increments = t.randn(n_walks, n_steps, generator=generator)\n"
            "    cumulative = increments.cumsum(dim=1)  # (n_walks, n_steps)\n"
            "    final_tensor = cumulative[:, -1]       # (n_walks,)\n"
            "    final_positions = []\n"
            "    for i in range(n_walks):\n"
            "        final_positions.append(final_tensor[i].item())\n"
            "    return {\n"
            "        'final_positions': final_positions,\n"
            "        'sample_walks': cumulative[:5].clone(),\n"
            "    }"
        ),
        "solution_notes": (
            "**`.item()` in a loop vs `.tolist()`.** For pulling many scalars "
            "back to Python, `final_tensor.tolist()` is faster (one CPU sync "
            "instead of `n_walks` of them). The per-walk `.item()` loop here "
            "is deliberately the slow version — it's the right shape for "
            "cases where each walk's scalar feeds an immediate Python-side "
            "decision (e.g. branching, accumulating into a non-tensor "
            "structure).\n\n"
            "**Cumulative-sum trick.** A random walk is just the cumulative "
            "sum of iid normal increments. Generating all the noise in one "
            "`t.randn(n_walks, n_steps)` call and doing one `.cumsum(dim=1)` "
            "is orders of magnitude faster than a Python `for step in range` "
            "loop, even though it produces the same trajectories.\n\n"
            "**Why the empirical std ≈ √n_steps.** A standard random walk's "
            "variance grows linearly with the number of steps, so its "
            "standard deviation grows as √n. With `n_steps=100` the "
            "theoretical std is exactly 10."
        ),
        "extra_imports": ["import matplotlib.pyplot as plt"],
    },
]


for spec in SPECS:
    path = emit_standalone(spec)
    rel = path.relative_to(path.parents[3])
    print(f"wrote {rel}")
