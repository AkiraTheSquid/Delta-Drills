"""Composite drills cx19..cx24 — batch-19 part3 (II-cell, ARENA Adam dual EMA).

Six composite procedural drills exercising 2-atom pairs from ARENA part 3 —
Adam optimizer internals: the dual `m`, `v` EMAs, the bias-correction divide,
and the `sqrt(...)+eps` denominator stabilization.

cx19  ema-first-moment + ema-second-moment             — dual m & v EMA buffers
cx20  ema-first-moment + bias-correction-divide        — m_hat = m / (1 - beta1**t)
cx21  ema-second-moment + bias-correction-divide       — v_hat = v / (1 - beta2**t)
cx22  ema-first-moment + sqrt-eps-stabilize            — track m alongside sqrt-eps denominator
cx23  ema-second-moment + sqrt-eps-stabilize           — sqrt(v_hat) + eps denominator
cx24  bias-correction-divide + sqrt-eps-stabilize      — m_hat / (sqrt(v_hat) + eps) ratio
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _emit_composite import emit_composite  # noqa: E402

INV = {a["atom_id"]: a for a in json.load(open("/tmp/drill_atoms.json"))}


def _subs(atom_ids):
    return [INV[a]["subtopic"] for a in atom_ids]


# These composites are pure tensor + Python math — no nn / F needed.
NO_EXTRA_IMPORTS: list[str] = []


# ===========================================================================
# cx19 — dual m & v EMA buffers from one gradient list
# ===========================================================================
spec_19 = {
    "atom_ids": ["ema-first-moment", "ema-second-moment"],
    "subtopics": _subs(["ema-first-moment", "ema-second-moment"]),
    "primary_atom": "ema-first-moment",
    "part": "part3",
    "exercise_index": 19,
    "exercise_title": "Adam dual m & v EMA buffers in one pass",
    "slug": "adam-dual-m-v-ema-update",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Adam tracks TWO running buffers per parameter:\n"
        "1. **First moment `m`** — an EMA of the raw gradient `g`. Acts as a momentum-like "
        "term: smoothes the descent direction.\n"
        "2. **Second moment `v`** — an EMA of the SQUARED gradient `g**2`. Acts as a "
        "per-coordinate adaptive scale: directions with large historical |g| get smaller "
        "effective steps.\n\n"
        "The two recurrences are INDEPENDENT (no cross-term):\n"
        "```python\n"
        "m = beta1 * m + (1 - beta1) * g\n"
        "v = beta2 * v + (1 - beta2) * g.pow(2)\n"
        "```\n"
        "Both share the same shape as the parameter (and the gradient). Real `torch.optim.Adam` "
        "runs both updates inside ONE per-parameter loop so the `(m, v, g)` triple is in cache "
        "together — that's the canonical composition.\n\n"
        "**Why both atoms together.** Either alone is just a moving average; together they ARE "
        "Adam's state. Bias correction and the denominator wiring come later — this drill is "
        "purely the two buffer recurrences.\n\n"
        "**`.copy_()` not assignment.** `m = beta1 * m + ...` rebinds the local name; the "
        "caller's list still points to the old tensor. `m.copy_(...)` mutates the storage in "
        "place so the caller sees the update."
    ),
    "prompt_body": (
        "Implement `cx19_update_m_and_v(m_list, v_list, grad_list, beta1, beta2)`.\n\n"
        "For each triple `(m, v, g)` from the three same-length lists, update IN PLACE:\n"
        "1. `m <- beta1 * m + (1 - beta1) * g`\n"
        "2. `v <- beta2 * v + (1 - beta2) * g.pow(2)`\n\n"
        "Use `m.copy_(...)` and `v.copy_(...)` so the original tensor storage (its `data_ptr()`) "
        "survives the call. Return `None`.\n\n"
        "The test checks: (a) both buffers update from a zero start by the correct closed-form; "
        "(b) `v` stays non-negative even with negative gradients (since we square); "
        "(c) the i-th `(m_i, v_i)` is independent of the j-th (no cross-talk between params); "
        "(d) after many steps of constant `g`, `m` converges to `g` and `v` converges to `g**2`; "
        "(e) tensor `data_ptr()` is preserved across many calls."
    ),
    "stub_body": (
        "def cx19_update_m_and_v(m_list, v_list, grad_list, beta1: float, beta2: float):\n"
        "    \"\"\"Update Adam's first- and second-moment buffers in place from one grad list.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: single param, one step from zero.\n"
        "m = t.zeros(3)\n"
        "v = t.zeros(3)\n"
        "g = t.tensor([1.0, -2.0, 3.0])\n"
        "ptr_m, ptr_v = m.data_ptr(), v.data_ptr()\n"
        "result = cx19_update_m_and_v([m], [v], [g], beta1=0.9, beta2=0.999)\n"
        "assert result is None, f'must return None; got {result}'\n"
        "# m = 0.9*0 + 0.1*g = 0.1*g\n"
        "assert t.allclose(m, 0.1 * g, atol=1e-6), f'm wrong: got {m}, expected {0.1*g}'\n"
        "# v = 0.999*0 + 0.001*g**2 = 0.001*g**2\n"
        "assert t.allclose(v, 0.001 * g.pow(2), atol=1e-6), f'v wrong: got {v}'\n"
        "assert m.data_ptr() == ptr_m, 'm must be mutated in place (copy_)'\n"
        "assert v.data_ptr() == ptr_v, 'v must be mutated in place (copy_)'\n"
        "\n"
        "# Case B: v stays non-negative even with negative gradient (g**2 is always >= 0).\n"
        "assert (v >= 0).all(), f'v entries must be non-negative; got {v}'\n"
        "\n"
        "# Case C: multi-param, mismatched shapes — no cross-talk.\n"
        "m1 = t.zeros(3)\n"
        "v1 = t.zeros(3)\n"
        "m2 = t.zeros(2, 4)\n"
        "v2 = t.zeros(2, 4)\n"
        "g1 = t.tensor([0.5, 0.5, 0.5])\n"
        "g2 = t.full((2, 4), 2.0)\n"
        "cx19_update_m_and_v([m1, m2], [v1, v2], [g1, g2], beta1=0.9, beta2=0.999)\n"
        "assert t.allclose(m1, t.full((3,), 0.05), atol=1e-6), 'm1 update wrong'\n"
        "assert t.allclose(m2, t.full((2, 4), 0.2), atol=1e-6), 'm2 update wrong'\n"
        "assert t.allclose(v1, t.full((3,), 0.00025), atol=1e-8), 'v1 update wrong'\n"
        "assert t.allclose(v2, t.full((2, 4), 0.004), atol=1e-8), 'v2 update wrong'\n"
        "\n"
        "# Case D: many steps of constant g => m -> g, v -> g**2.\n"
        "m = t.zeros(3)\n"
        "v = t.zeros(3)\n"
        "g = t.tensor([1.0, 2.0, -3.0])\n"
        "for _ in range(15000):\n"
        "    cx19_update_m_and_v([m], [v], [g], beta1=0.9, beta2=0.999)\n"
        "assert t.allclose(m, g, atol=1e-3), f'm should converge to g; got {m}'\n"
        "assert t.allclose(v, g.pow(2), atol=1e-2), f'v should converge to g**2; got {v}'\n"
        "\n"
        "# Case E: cross-check vs torch.optim.Adam's m & v buffers after one step.\n"
        "# Construct a one-parameter problem and step Adam once; compare exp_avg / exp_avg_sq.\n"
        "t.manual_seed(7)\n"
        "p = t.nn.Parameter(t.randn(5))\n"
        "g_ref = t.randn(5)\n"
        "opt = t.optim.Adam([p], lr=0.0, betas=(0.9, 0.999), eps=1e-8)  # lr=0 so p is irrelevant\n"
        "p.grad = g_ref.clone()\n"
        "opt.step()\n"
        "ref_state = opt.state[p]\n"
        "ref_m = ref_state['exp_avg']\n"
        "ref_v = ref_state['exp_avg_sq']\n"
        "# Replay with cx19 from zero.\n"
        "m_my = t.zeros(5)\n"
        "v_my = t.zeros(5)\n"
        "cx19_update_m_and_v([m_my], [v_my], [g_ref], beta1=0.9, beta2=0.999)\n"
        "assert t.allclose(m_my, ref_m, atol=1e-6), f'cx19 m disagrees with torch.optim.Adam exp_avg'\n"
        "assert t.allclose(v_my, ref_v, atol=1e-6), f'cx19 v disagrees with torch.optim.Adam exp_avg_sq'\n"
        "\n"
        "# Case F: data_ptr survives many calls.\n"
        "m = t.zeros(4)\n"
        "v = t.zeros(4)\n"
        "g = t.ones(4)\n"
        "ptr_m, ptr_v = m.data_ptr(), v.data_ptr()\n"
        "for _ in range(50):\n"
        "    cx19_update_m_and_v([m], [v], [g], 0.9, 0.999)\n"
        "assert m.data_ptr() == ptr_m, 'm must stay at the same storage across steps'\n"
        "assert v.data_ptr() == ptr_v, 'v must stay at the same storage across steps'"
    ),
    "solution_body": (
        "def cx19_update_m_and_v(m_list, v_list, grad_list, beta1, beta2):\n"
        "    for m, v, g in zip(m_list, v_list, grad_list):\n"
        "        # Atom A (ema-first-moment): EMA of g.\n"
        "        m.copy_(beta1 * m + (1.0 - beta1) * g)\n"
        "        # Atom B (ema-second-moment): EMA of g**2.\n"
        "        v.copy_(beta2 * v + (1.0 - beta2) * g.pow(2))"
    ),
    "solution_notes": (
        "**Why one combined loop.** Both updates are independent — `m` reads only m & g, `v` "
        "reads only v & g. Putting them in the same per-param loop is purely a cache / memory "
        "locality win: the `(m, v, g)` triple is touched once per param.\n\n"
        "**Order is free.** You could update `v` first then `m` — the result is the same. "
        "PyTorch's `torch.optim.Adam` updates `exp_avg` first then `exp_avg_sq` by convention."
    ),
    "extra_imports": NO_EXTRA_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["ema-first-moment", "ema-second-moment"],
    "lo": (
        "Compose Adam's first-moment EMA (m = beta1*m + (1-beta1)*g) with the second-moment EMA "
        "(v = beta2*v + (1-beta2)*g**2) in a single co-located in-place loop matching "
        "torch.optim.Adam's exp_avg / exp_avg_sq buffers."
    ),
}


# ===========================================================================
# cx20 — ema-first-moment + bias-correction-divide: m_hat = m / (1 - beta1**t)
# ===========================================================================
spec_20 = {
    "atom_ids": ["ema-first-moment", "bias-correction-divide"],
    "subtopics": _subs(["ema-first-moment", "bias-correction-divide"]),
    "primary_atom": "ema-first-moment",
    "part": "part3",
    "exercise_index": 20,
    "exercise_title": "m EMA then bias-correct: m_hat = m / (1 - beta1**t)",
    "slug": "ema-m-then-bias-correct",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Adam's first-moment EMA starts from zero. The recurrence "
        "`m = beta1 * m + (1 - beta1) * g` therefore underestimates the true mean of `g` for the "
        "first few steps — for constant `g`, after `t` steps you get `m_t = (1 - beta1**t) * g`.\n\n"
        "The fix is the **bias-correction divide**:\n"
        "```\n"
        "m_hat = m / (1 - beta1**t)\n"
        "```\n"
        "With constant `g`, this recovers `m_hat = g` EXACTLY at every step — the divide undoes "
        "the warmup shrinkage. As `t -> inf`, `beta1**t -> 0`, so the divisor `-> 1` and the "
        "correction fades to a no-op.\n\n"
        "**Composition.** This drill chains the two atoms: take a gradient `g`, drive one step of "
        "the `m` EMA from the current buffer, THEN apply the bias-correction divide. The combined "
        "function returns BOTH the updated `m` (caller stores it for the next step) and `m_hat` "
        "(used downstream in the parameter update).\n\n"
        "**Why `t` is 1-based.** At `t=0`, `beta1**0 = 1` and the divisor is 0 — division by zero. "
        "Adam's step counter starts at 1 and is incremented BEFORE the bias correction divide. "
        "Pass `t=0` and you get NaN.\n\n"
        "**Anatomy.**\n"
        "```python\n"
        "def step(m, g, beta1, t):\n"
        "    m_new = beta1 * m + (1 - beta1) * g    # atom A\n"
        "    m_hat = m_new / (1 - beta1 ** t)        # atom B\n"
        "    return m_new, m_hat\n"
        "```"
    ),
    "prompt_body": (
        "Implement `cx20_ema_then_bias_correct(m, g, beta1, t_step)`.\n\n"
        "Inputs:\n"
        "- `m`: current first-moment buffer (Tensor of any shape).\n"
        "- `g`: current gradient (Tensor, same shape as `m`).\n"
        "- `beta1`: float decay (e.g. 0.9).\n"
        "- `t_step`: int >= 1, the 1-based step counter.\n\n"
        "Returns a tuple `(m_new, m_hat)`:\n"
        "- `m_new = beta1 * m + (1 - beta1) * g` — the updated buffer (atom: ema-first-moment).\n"
        "- `m_hat = m_new / (1 - beta1 ** t_step)` — bias-corrected (atom: bias-correction-divide).\n\n"
        "Do NOT mutate the input `m` in place — return fresh tensors. (The caller is responsible for "
        "swapping `m` with `m_new` between steps.)\n\n"
        "Sanity properties the test verifies:\n"
        "- Step 1 with `m = 0`: `m_new = (1 - beta1) * g`, and `m_hat = g` EXACTLY (closed form).\n"
        "- Many steps of constant `g` starting from zero: `m_hat` stays equal to `g` at every step.\n"
        "- As `t_step -> inf`, the divisor -> 1, so `m_hat -> m_new`.\n"
        "- The input `m` is not mutated."
    ),
    "stub_body": (
        "def cx20_ema_then_bias_correct(m: Tensor, g: Tensor, beta1: float, t_step: int):\n"
        "    \"\"\"Return (m_new, m_hat) after one step of m-EMA + bias correction.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: step 1 with m = 0 should recover m_hat = g.\n"
        "m = t.zeros(4)\n"
        "g = t.tensor([1.0, -2.0, 3.0, 0.5])\n"
        "beta1 = 0.9\n"
        "m_new, m_hat = cx20_ema_then_bias_correct(m, g, beta1, t_step=1)\n"
        "assert t.allclose(m_new, 0.1 * g, atol=1e-6), f'm_new wrong: got {m_new}'\n"
        "assert t.allclose(m_hat, g, atol=1e-6), f'bias-corrected m_hat should equal g at step 1; got {m_hat}'\n"
        "\n"
        "# Case B: input m not mutated.\n"
        "m_in = t.tensor([0.5, 1.0])\n"
        "snapshot = m_in.clone()\n"
        "_ = cx20_ema_then_bias_correct(m_in, t.tensor([2.0, -1.0]), 0.9, t_step=3)\n"
        "assert t.equal(m_in, snapshot), f'input m was mutated; got {m_in}'\n"
        "\n"
        "# Case C: many steps of constant g — m_hat stays equal to g at EVERY step.\n"
        "m = t.zeros(3)\n"
        "g = t.tensor([1.0, 2.0, -3.0])\n"
        "beta1 = 0.9\n"
        "for step in range(1, 50):\n"
        "    m, m_hat = cx20_ema_then_bias_correct(m, g, beta1, t_step=step)\n"
        "    assert t.allclose(m_hat, g, atol=1e-5), (\n"
        "        f'step {step}: m_hat should equal g for constant g; got {m_hat}'\n"
        "    )\n"
        "\n"
        "# Case D: large t_step — m_hat approaches m_new (correction fades).\n"
        "m_big = t.tensor([0.7, -0.4])\n"
        "g_big = t.tensor([0.1, 0.1])\n"
        "m_new_big, m_hat_big = cx20_ema_then_bias_correct(m_big, g_big, 0.9, t_step=10000)\n"
        "assert t.allclose(m_hat_big, m_new_big, atol=1e-6), 'large t: m_hat should == m_new'\n"
        "\n"
        "# Case E: cross-check with torch.optim.Adam's first-moment + bias correction after step 1.\n"
        "t.manual_seed(11)\n"
        "p = t.nn.Parameter(t.randn(5))\n"
        "g_ref = t.randn(5)\n"
        "opt = t.optim.Adam([p], lr=0.0, betas=(0.9, 0.999), eps=1e-8)\n"
        "p.grad = g_ref.clone()\n"
        "opt.step()\n"
        "ref_m = opt.state[p]['exp_avg']\n"
        "# Adam's bias-corrected m_hat at step 1 = ref_m / (1 - 0.9**1)\n"
        "ref_m_hat = ref_m / (1 - 0.9 ** 1)\n"
        "m_zero = t.zeros(5)\n"
        "my_m_new, my_m_hat = cx20_ema_then_bias_correct(m_zero, g_ref, 0.9, t_step=1)\n"
        "assert t.allclose(my_m_new, ref_m, atol=1e-6), 'm_new disagrees with torch.optim.Adam exp_avg'\n"
        "assert t.allclose(my_m_hat, ref_m_hat, atol=1e-6), 'm_hat disagrees with bias-corrected exp_avg'\n"
        "\n"
        "# Case F: t_step=1 with constant nonzero m & g — bias correction is LOAD-BEARING.\n"
        "# Without dividing, m_new would be << g. With dividing, m_hat magnitude > m_new magnitude.\n"
        "m_pre = t.tensor([0.05, 0.05])\n"
        "g_now = t.tensor([1.0, 1.0])\n"
        "m_new_f, m_hat_f = cx20_ema_then_bias_correct(m_pre, g_now, 0.9, t_step=1)\n"
        "assert m_hat_f.abs().sum() > m_new_f.abs().sum() * 5, (\n"
        "    f'm_hat should be much larger than m_new at t=1 with beta1=0.9 (divisor=0.1); '\n"
        "    f'got m_new={m_new_f}, m_hat={m_hat_f} — did you skip the bias-correction divide?'\n"
        ")"
    ),
    "solution_body": (
        "def cx20_ema_then_bias_correct(m, g, beta1, t_step):\n"
        "    # Atom A (ema-first-moment): one EMA step on m.\n"
        "    m_new = beta1 * m + (1.0 - beta1) * g\n"
        "    # Atom B (bias-correction-divide): undo the zero-init bias by dividing by (1 - beta1^t).\n"
        "    m_hat = m_new / (1.0 - beta1 ** t_step)\n"
        "    return m_new, m_hat"
    ),
    "solution_notes": (
        "**Two outputs, not one.** The caller needs `m_new` to seed the next step's EMA AND "
        "`m_hat` for the current parameter update. Returning only `m_hat` would lose the "
        "uncorrected buffer state. Returning only `m_new` would skip bias correction — Adam's "
        "first few steps would crawl.\n\n"
        "**Don't mutate `m`.** Some impls write `m.mul_(beta1).add_(g, alpha=1-beta1)` to save "
        "an allocation. That's a valid optimization, but the caller-visible contract is identical: "
        "the returned tuple's first element IS the new buffer."
    ),
    "extra_imports": NO_EXTRA_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["ema-first-moment", "bias-correction-divide"],
    "lo": (
        "Compose the first-moment EMA recurrence (m <- beta1*m + (1-beta1)*g) with the "
        "bias-correction divide (m_hat = m / (1 - beta1**t)) so that one helper produces both "
        "the updated buffer and its de-biased view in a single Adam-style step."
    ),
}


# ===========================================================================
# cx21 — ema-second-moment + bias-correction-divide: v_hat = v / (1 - beta2**t)
# ===========================================================================
spec_21 = {
    "atom_ids": ["ema-second-moment", "bias-correction-divide"],
    "subtopics": _subs(["ema-second-moment", "bias-correction-divide"]),
    "primary_atom": "ema-second-moment",
    "part": "part3",
    "exercise_index": 21,
    "exercise_title": "v EMA then bias-correct: v_hat = v / (1 - beta2**t)",
    "slug": "ema-v-then-bias-correct",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "The second-moment EMA recurrence is structurally identical to the first-moment one, but "
        "EMAs the SQUARED gradient and uses a separate decay `beta2` (typically 0.999, much "
        "slower than `beta1 = 0.9`):\n"
        "```\n"
        "v = beta2 * v + (1 - beta2) * g.pow(2)\n"
        "```\n"
        "And the bias correction has the same shape, but the divisor uses `beta2`:\n"
        "```\n"
        "v_hat = v / (1 - beta2 ** t)\n"
        "```\n"
        "**Why a SEPARATE beta2.** `v` accumulates `g**2`, which is much noisier and skewed than "
        "`g` itself. A slower EMA (larger beta2) smooths it harder so the per-coordinate scale "
        "in the Adam update doesn't whip around batch-to-batch.\n\n"
        "**The classic bug.** Applying the WRONG bias-correction divisor (using `beta1` instead "
        "of `beta2`). Since `beta1 != beta2`, the resulting `v_hat` is wrong by the ratio "
        "`(1 - beta2**t) / (1 - beta1**t)` — a multiplicative scale error that gets squared into "
        "the denominator. Test cases here pin this down.\n\n"
        "**v_hat is always non-negative.** `v = EMA(g**2)` starts at 0 and adds only non-negative "
        "terms, so `v >= 0` elementwise. The divisor `1 - beta2**t` is positive for `t >= 1`, so "
        "`v_hat >= 0` too. Downstream code can safely take `sqrt(v_hat)`."
    ),
    "prompt_body": (
        "Implement `cx21_ema_v_then_bias_correct(v, g, beta2, t_step)`.\n\n"
        "Inputs:\n"
        "- `v`: current second-moment buffer (Tensor of any shape).\n"
        "- `g`: current gradient (Tensor, same shape as `v`).\n"
        "- `beta2`: float decay (typically 0.999).\n"
        "- `t_step`: int >= 1.\n\n"
        "Returns `(v_new, v_hat)`:\n"
        "- `v_new = beta2 * v + (1 - beta2) * g.pow(2)` (atom: ema-second-moment).\n"
        "- `v_hat = v_new / (1 - beta2 ** t_step)` (atom: bias-correction-divide).\n\n"
        "Do not mutate `v` in place.\n\n"
        "Tests verify:\n"
        "- Step 1 with `v = 0`: `v_new = (1 - beta2) * g**2`, `v_hat = g**2` (closed form).\n"
        "- `v_new` (and `v_hat`) is non-negative elementwise even when `g` is negative.\n"
        "- Many constant-g steps: `v_hat == g**2` at every step.\n"
        "- Cross-check vs `torch.optim.Adam`'s `exp_avg_sq` + the `1 - beta2**t` divisor.\n"
        "- Sabotage: using `beta1=0.9` as the divisor (the classic 'used wrong beta' bug) "
        "FAILS by a known factor — the test confirms you used `beta2`."
    ),
    "stub_body": (
        "def cx21_ema_v_then_bias_correct(v: Tensor, g: Tensor, beta2: float, t_step: int):\n"
        "    \"\"\"Return (v_new, v_hat) after one step of v-EMA + bias correction.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: step 1 with v = 0.\n"
        "v = t.zeros(4)\n"
        "g = t.tensor([1.0, -2.0, 3.0, 0.5])\n"
        "beta2 = 0.999\n"
        "v_new, v_hat = cx21_ema_v_then_bias_correct(v, g, beta2, t_step=1)\n"
        "assert t.allclose(v_new, (1 - beta2) * g.pow(2), atol=1e-6), f'v_new wrong: got {v_new}'\n"
        "assert t.allclose(v_hat, g.pow(2), atol=1e-5), (\n"
        "    f'bias-corrected v_hat should equal g**2 at step 1; got {v_hat}'\n"
        ")\n"
        "\n"
        "# Case B: non-negativity.\n"
        "assert (v_new >= 0).all(), 'v_new must be non-negative'\n"
        "assert (v_hat >= 0).all(), 'v_hat must be non-negative'\n"
        "\n"
        "# Case C: input v not mutated.\n"
        "v_in = t.tensor([0.5, 0.25])\n"
        "snap = v_in.clone()\n"
        "_ = cx21_ema_v_then_bias_correct(v_in, t.tensor([2.0, -1.0]), 0.999, t_step=3)\n"
        "assert t.equal(v_in, snap), 'input v was mutated'\n"
        "\n"
        "# Case D: many constant-g steps — v_hat stays equal to g**2.\n"
        "v = t.zeros(3)\n"
        "g = t.tensor([1.0, 2.0, -3.0])\n"
        "for step in range(1, 50):\n"
        "    v, v_hat = cx21_ema_v_then_bias_correct(v, g, 0.999, t_step=step)\n"
        "    assert t.allclose(v_hat, g.pow(2), atol=1e-3), (\n"
        "        f'step {step}: v_hat should equal g**2; got {v_hat}, expected {g.pow(2)}'\n"
        "    )\n"
        "\n"
        "# Case E: large t — v_hat -> v_new.\n"
        "v0 = t.tensor([0.3, 0.7])\n"
        "g0 = t.tensor([0.1, 0.1])\n"
        "v_new_big, v_hat_big = cx21_ema_v_then_bias_correct(v0, g0, 0.999, t_step=20000)\n"
        "assert t.allclose(v_hat_big, v_new_big, atol=1e-6), 'large t: v_hat should == v_new'\n"
        "\n"
        "# Case F: cross-check vs torch.optim.Adam.\n"
        "t.manual_seed(13)\n"
        "p = t.nn.Parameter(t.randn(5))\n"
        "g_ref = t.randn(5)\n"
        "opt = t.optim.Adam([p], lr=0.0, betas=(0.9, 0.999), eps=1e-8)\n"
        "p.grad = g_ref.clone()\n"
        "opt.step()\n"
        "ref_v = opt.state[p]['exp_avg_sq']\n"
        "ref_v_hat = ref_v / (1 - 0.999 ** 1)\n"
        "v_zero = t.zeros(5)\n"
        "my_v_new, my_v_hat = cx21_ema_v_then_bias_correct(v_zero, g_ref, 0.999, t_step=1)\n"
        "assert t.allclose(my_v_new, ref_v, atol=1e-6), 'v_new disagrees with torch.optim.Adam exp_avg_sq'\n"
        "assert t.allclose(my_v_hat, ref_v_hat, atol=1e-6), 'v_hat disagrees with bias-corrected exp_avg_sq'\n"
        "\n"
        "# Case G: sabotage — using beta1=0.9 as divisor would give v_hat ~ v_new/0.1 = 10*v_new.\n"
        "# With correct beta2=0.999 at step 1, v_hat = v_new/0.001 = 1000*v_new.\n"
        "v_san_in = t.zeros(2)\n"
        "g_san = t.tensor([2.0, 2.0])\n"
        "v_san, v_san_hat = cx21_ema_v_then_bias_correct(v_san_in, g_san, 0.999, t_step=1)\n"
        "# v_new = 0.001 * 4 = 0.004; v_hat = 0.004 / 0.001 = 4.0\n"
        "assert t.allclose(v_san_hat, t.tensor([4.0, 4.0]), atol=1e-4), (\n"
        "    f'sabotage: expected v_hat ~ [4.0, 4.0]; got {v_san_hat} — '\n"
        "    f'wrong divisor (did you use beta1 instead of beta2?)'\n"
        ")"
    ),
    "solution_body": (
        "def cx21_ema_v_then_bias_correct(v, g, beta2, t_step):\n"
        "    # Atom A (ema-second-moment): EMA on g**2 with decay beta2.\n"
        "    v_new = beta2 * v + (1.0 - beta2) * g.pow(2)\n"
        "    # Atom B (bias-correction-divide): use beta2 (NOT beta1) in the divisor.\n"
        "    v_hat = v_new / (1.0 - beta2 ** t_step)\n"
        "    return v_new, v_hat"
    ),
    "solution_notes": (
        "**Mirror of cx20, with two critical swaps:** `beta1 -> beta2`, `g -> g.pow(2)`. Same "
        "structural composition — the bias correction always uses the SAME beta as the EMA it "
        "corrects. Using `beta1` as the divisor for `v` is the textbook Adam-from-scratch bug.\n\n"
        "**Why `g.pow(2)` not `g * g`.** Both produce the same value. `pow(2)` is fused in "
        "PyTorch's elementwise CUDA kernel and skips one tensor allocation. For Adam's inner "
        "loop this matters at large model scale."
    ),
    "extra_imports": NO_EXTRA_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["ema-second-moment", "bias-correction-divide"],
    "lo": (
        "Compose the second-moment EMA (v <- beta2*v + (1-beta2)*g**2) with the "
        "bias-correction divide (v_hat = v / (1 - beta2**t)) using the CORRECT decay (beta2, not "
        "beta1), matching torch.optim.Adam's exp_avg_sq + bias-corrected denominator."
    ),
}


# ===========================================================================
# cx22 — ema-first-moment + sqrt-eps-stabilize: track m alongside sqrt-eps denominator
# ===========================================================================
spec_22 = {
    "atom_ids": ["ema-first-moment", "sqrt-eps-stabilize"],
    "subtopics": _subs(["ema-first-moment", "sqrt-eps-stabilize"]),
    "primary_atom": "ema-first-moment",
    "part": "part3",
    "exercise_index": 22,
    "exercise_title": "track m EMA alongside the sqrt(...)+eps denominator pattern",
    "slug": "track-m-with-sqrt-eps-denominator",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "This is a **rare-pair** drill: the two atoms don't share a single equation in Adam's "
        "step (the canonical denominator uses `v_hat`, not `m`), but they DO co-locate in "
        "Adam-from-scratch implementations — both live inside the per-parameter inner loop.\n\n"
        "**The pair.**\n"
        "- **ema-first-moment** — `m = beta1 * m + (1 - beta1) * g`. The numerator buffer.\n"
        "- **sqrt-eps-stabilize** — `denom = sqrt(positive_buffer) + eps` or equivalently "
        "`sqrt(positive_buffer + eps)`. The defensive stabilization.\n\n"
        "**Why pair them.** In ARENA's Adam writeup the inner loop interleaves: update m, update "
        "v, build the sqrt-eps denominator from v (or from an arbitrary non-negative buffer), "
        "form the step. The most common writing bug is to use the WRONG buffer in the sqrt-eps "
        "denominator (e.g. `sqrt(m**2) + eps` — a sign-discarding error) or to apply eps in the "
        "wrong place. This drill exercises the safe composition: m is updated, and a "
        "non-negative scratch buffer (passed in) gets the sqrt-eps treatment, returning both.\n\n"
        "**Where to put the eps.** ARENA / `torch.optim.Adam` use `sqrt(v_hat) + eps` (eps "
        "OUTSIDE the sqrt). Some BatchNorm/RMSNorm impls use `sqrt(var + eps)` (eps INSIDE). Both "
        "stabilize the divide; only the inside-sqrt form bounds the gradient near zero. Adam "
        "tolerates the outside-sqrt form because its `v_hat` is bounded away from zero by the "
        "warmup-step accumulation. For this drill, use the **Adam convention**: `sqrt(buf) + eps`."
    ),
    "prompt_body": (
        "Implement `cx22_m_step_and_denom(m, g, buf, beta1, eps)`.\n\n"
        "Inputs:\n"
        "- `m`: first-moment buffer (Tensor).\n"
        "- `g`: gradient (Tensor, same shape as `m`).\n"
        "- `buf`: a non-negative scratch buffer (Tensor) used to build the sqrt-eps denominator. "
        "Assume `buf >= 0` elementwise — the caller is responsible.\n"
        "- `beta1`: float decay.\n"
        "- `eps`: float stabilizer (typically 1e-8).\n\n"
        "Returns `(m_new, denom)`:\n"
        "- `m_new = beta1 * m + (1 - beta1) * g` (atom: ema-first-moment).\n"
        "- `denom = sqrt(buf) + eps` — Adam convention, eps OUTSIDE the sqrt (atom: sqrt-eps-stabilize).\n\n"
        "Do not mutate `m` or `buf`.\n\n"
        "Tests verify:\n"
        "- `m_new` follows the EMA closed form.\n"
        "- `denom > 0` strictly, even when `buf == 0`.\n"
        "- `denom` matches `sqrt(buf) + eps` to numerical precision.\n"
        "- The OUTSIDE-sqrt placement is detectable: when `buf` is all-zero, `denom == eps`, "
        "not `sqrt(eps) ~ 3.16e-4`.\n"
        "- The two atoms are INDEPENDENT — changing `buf` does not affect `m_new`, and changing "
        "`g` does not affect `denom`."
    ),
    "stub_body": (
        "def cx22_m_step_and_denom(m: Tensor, g: Tensor, buf: Tensor, beta1: float, eps: float):\n"
        "    \"\"\"Return (m_new, denom) — EMA step on m plus sqrt-eps denominator from buf.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: m EMA step matches the recurrence.\n"
        "m = t.zeros(4)\n"
        "g = t.tensor([1.0, -2.0, 3.0, 0.5])\n"
        "buf = t.tensor([1.0, 4.0, 9.0, 16.0])\n"
        "beta1 = 0.9\n"
        "eps = 1e-8\n"
        "m_new, denom = cx22_m_step_and_denom(m, g, buf, beta1, eps)\n"
        "assert t.allclose(m_new, 0.1 * g, atol=1e-6), f'm_new wrong: got {m_new}'\n"
        "\n"
        "# Case B: denom matches sqrt(buf) + eps.\n"
        "expected_denom = buf.sqrt() + eps\n"
        "assert t.allclose(denom, expected_denom, atol=1e-7), (\n"
        "    f'denom wrong: got {denom}, expected {expected_denom}'\n"
        ")\n"
        "\n"
        "# Case C: denom > 0 strictly, even on all-zero buf.\n"
        "buf_zero = t.zeros(3)\n"
        "m2 = t.zeros(3)\n"
        "g2 = t.ones(3)\n"
        "_, denom_zero = cx22_m_step_and_denom(m2, g2, buf_zero, 0.9, eps=1e-8)\n"
        "assert (denom_zero > 0).all(), f'denom must be strictly positive; got {denom_zero}'\n"
        "# Adam convention: denom == eps when buf == 0 (NOT sqrt(eps)).\n"
        "assert t.allclose(denom_zero, t.full((3,), 1e-8), atol=1e-12), (\n"
        "    f'on buf=0 with eps OUTSIDE sqrt, denom should be eps={1e-8}; got {denom_zero} — '\n"
        "    f'if you got ~3.16e-4 you put eps INSIDE the sqrt (sqrt(buf+eps)), use Adam convention.'\n"
        ")\n"
        "\n"
        "# Case D: inputs not mutated.\n"
        "m_in = t.tensor([0.5, 1.0])\n"
        "buf_in = t.tensor([1.0, 2.0])\n"
        "snap_m, snap_buf = m_in.clone(), buf_in.clone()\n"
        "_ = cx22_m_step_and_denom(m_in, t.zeros(2), buf_in, 0.9, eps=1e-8)\n"
        "assert t.equal(m_in, snap_m), 'm was mutated'\n"
        "assert t.equal(buf_in, snap_buf), 'buf was mutated'\n"
        "\n"
        "# Case E: independence — changing buf doesn't change m_new.\n"
        "m_a, _ = cx22_m_step_and_denom(t.zeros(3), t.ones(3), t.tensor([1.0, 2.0, 3.0]), 0.9, 1e-8)\n"
        "m_b, _ = cx22_m_step_and_denom(t.zeros(3), t.ones(3), t.tensor([99.0, 88.0, 77.0]), 0.9, 1e-8)\n"
        "assert t.allclose(m_a, m_b), 'm_new must not depend on buf'\n"
        "\n"
        "# Case F: independence — changing g doesn't change denom.\n"
        "_, d_a = cx22_m_step_and_denom(t.zeros(3), t.tensor([1.0, 2.0, 3.0]), t.ones(3), 0.9, 1e-8)\n"
        "_, d_b = cx22_m_step_and_denom(t.zeros(3), t.tensor([99.0, 88.0, 77.0]), t.ones(3), 0.9, 1e-8)\n"
        "assert t.allclose(d_a, d_b), 'denom must not depend on g'\n"
        "\n"
        "# Case G: many-step EMA of m converges to g (atom-A sanity).\n"
        "m = t.zeros(2)\n"
        "g = t.tensor([1.0, -2.0])\n"
        "for _ in range(500):\n"
        "    m, _ = cx22_m_step_and_denom(m, g, t.ones(2), 0.9, 1e-8)\n"
        "assert t.allclose(m, g, atol=1e-4), f'm should converge to g for constant g; got {m}'"
    ),
    "solution_body": (
        "def cx22_m_step_and_denom(m, g, buf, beta1, eps):\n"
        "    # Atom A (ema-first-moment): first-moment EMA step.\n"
        "    m_new = beta1 * m + (1.0 - beta1) * g\n"
        "    # Atom B (sqrt-eps-stabilize): Adam convention puts eps OUTSIDE the sqrt.\n"
        "    denom = buf.sqrt() + eps\n"
        "    return m_new, denom"
    ),
    "solution_notes": (
        "**Eps placement matters.** Adam uses `sqrt(v_hat) + eps` (eps outside) because `v_hat` "
        "stays bounded away from zero — bias correction at step 1 already inflates `v` by "
        "`1/(1-beta2)` ~ 1000x. BatchNorm uses `sqrt(var + eps)` (eps inside) because batch var "
        "CAN collapse to zero on a dead-ReLU channel.\n\n"
        "**Test discrimination.** Case C is the cleanest way to tell the two placements apart: "
        "with `buf == 0`, eps-outside gives `eps = 1e-8`, eps-inside gives `sqrt(1e-8) = 1e-4` — "
        "a 10,000x difference.\n\n"
        "**Why this rare pair is worth exercising.** ARENA's Adam-from-scratch lab interleaves "
        "all four atoms (m, v, bias correction, sqrt-eps denominator) inside one per-parameter "
        "loop. Pinning down the m + sqrt-eps composition independently helps catch wrong-buffer "
        "bugs (e.g. `sqrt(m_squared) + eps`) that pass the m-only and sqrt-eps-only tests."
    ),
    "extra_imports": NO_EXTRA_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 3,
    "kcs": ["ema-first-moment", "sqrt-eps-stabilize"],
    "lo": (
        "Co-locate Adam's first-moment EMA (m <- beta1*m + (1-beta1)*g) with the Adam-convention "
        "sqrt-eps stabilization (denom = sqrt(buf) + eps, eps OUTSIDE the sqrt) in a single "
        "helper that mirrors ARENA's Adam inner loop."
    ),
}


# ===========================================================================
# cx23 — ema-second-moment + sqrt-eps-stabilize: sqrt(v_hat) + eps denominator
# ===========================================================================
spec_23 = {
    "atom_ids": ["ema-second-moment", "sqrt-eps-stabilize"],
    "subtopics": _subs(["ema-second-moment", "sqrt-eps-stabilize"]),
    "primary_atom": "ema-second-moment",
    "part": "part3",
    "exercise_index": 23,
    "exercise_title": "v EMA into the sqrt(v_hat)+eps Adam denominator",
    "slug": "ema-v-into-sqrt-eps-denominator",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "This is the **load-bearing pair** at the heart of Adam's adaptive scale: take the second "
        "moment `v` (an EMA of `g**2`), bias-correct to `v_hat`, then form the denominator "
        "`sqrt(v_hat) + eps`. The full Adam parameter step divides `m_hat` by this denominator.\n\n"
        "**The two atoms.**\n"
        "- **ema-second-moment** — `v = beta2 * v + (1 - beta2) * g.pow(2)`. Already builds in "
        "non-negativity because `g**2 >= 0` and `v` starts at 0.\n"
        "- **sqrt-eps-stabilize** — `denom = sqrt(v_hat) + eps`. The eps is the guard against "
        "early-training `v_hat ~ 0` (e.g. when all gradient entries are zero).\n\n"
        "**Why this exact composition.** The whole point of `v` is to be a per-coordinate scale; "
        "dividing the parameter step by `sqrt(v_hat)` gives directions with large historical |g| "
        "smaller effective steps. Adding `eps` ensures the division is finite when `v_hat` is "
        "small.\n\n"
        "**Eps placement: OUTSIDE the sqrt (Adam convention).** "
        "```python\n"
        "denom = (v_hat).sqrt() + eps     # canonical Adam\n"
        "# NOT: denom = (v_hat + eps).sqrt()  # BatchNorm style — wrong for Adam\n"
        "```\n"
        "`torch.optim.Adam` uses outside-sqrt. AdamW uses outside-sqrt. The HuggingFace AdamW "
        "uses outside-sqrt. Anything else disagrees with the reference.\n\n"
        "**Anatomy.**\n"
        "```python\n"
        "def adam_denominator(v, g, beta2, t, eps):\n"
        "    v = beta2 * v + (1 - beta2) * g.pow(2)            # atom A\n"
        "    v_hat = v / (1 - beta2 ** t)                       # bias correction (assumed prior atom)\n"
        "    denom = v_hat.sqrt() + eps                          # atom B (outside-sqrt)\n"
        "    return v, denom\n"
        "```"
    ),
    "prompt_body": (
        "Implement `cx23_v_step_and_denom(v, g, beta2, t_step, eps)`.\n\n"
        "Inputs:\n"
        "- `v`: second-moment buffer (Tensor).\n"
        "- `g`: gradient (Tensor, same shape as `v`).\n"
        "- `beta2`: float decay (typically 0.999).\n"
        "- `t_step`: int >= 1.\n"
        "- `eps`: float stabilizer (typically 1e-8).\n\n"
        "Steps:\n"
        "1. Update `v_new = beta2 * v + (1 - beta2) * g.pow(2)` (atom: ema-second-moment).\n"
        "2. Bias-correct: `v_hat = v_new / (1 - beta2 ** t_step)`.\n"
        "3. Build `denom = sqrt(v_hat) + eps` — **eps OUTSIDE the sqrt** (atom: sqrt-eps-stabilize).\n\n"
        "Return `(v_new, denom)`. Do not mutate `v`.\n\n"
        "Tests verify:\n"
        "- The EMA recurrence is correct (atom A).\n"
        "- `denom > 0` strictly, even when `g == 0` and `v == 0`.\n"
        "- When `g == 0` and `v == 0`, `denom == eps` (proves eps is OUTSIDE the sqrt).\n"
        "- With constant `g` for many steps, `denom -> |g| + eps` (sqrt of `g**2` recovers `|g|`).\n"
        "- Cross-check vs `torch.optim.Adam`'s full denominator after one step."
    ),
    "stub_body": (
        "def cx23_v_step_and_denom(v: Tensor, g: Tensor, beta2: float, t_step: int, eps: float):\n"
        "    \"\"\"Return (v_new, denom) — v EMA step, bias-correct, then sqrt+eps denominator.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: step 1 with v = 0 — closed-form check.\n"
        "v = t.zeros(4)\n"
        "g = t.tensor([1.0, -2.0, 3.0, 0.5])\n"
        "beta2 = 0.999\n"
        "eps = 1e-8\n"
        "v_new, denom = cx23_v_step_and_denom(v, g, beta2, 1, eps)\n"
        "# v_new = (1 - 0.999) * g**2 = 0.001 * g**2\n"
        "assert t.allclose(v_new, 0.001 * g.pow(2), atol=1e-6), f'v_new wrong: got {v_new}'\n"
        "# v_hat at step 1 = v_new / (1 - 0.999) = g**2.  denom = sqrt(g**2) + eps = |g| + eps.\n"
        "expected_denom = g.abs() + eps\n"
        "assert t.allclose(denom, expected_denom, atol=1e-5), (\n"
        "    f'denom wrong: got {denom}, expected {expected_denom}'\n"
        ")\n"
        "\n"
        "# Case B: denom > 0 strictly even on zero buffer and zero gradient.\n"
        "v_z = t.zeros(3)\n"
        "g_z = t.zeros(3)\n"
        "_, denom_z = cx23_v_step_and_denom(v_z, g_z, 0.999, 1, eps=1e-8)\n"
        "assert (denom_z > 0).all(), f'denom must be strictly positive; got {denom_z}'\n"
        "# Eps placement: with v=0 and g=0, v_hat=0, so denom = sqrt(0) + eps = eps (NOT sqrt(eps)).\n"
        "assert t.allclose(denom_z, t.full((3,), 1e-8), atol=1e-12), (\n"
        "    f'eps must be OUTSIDE the sqrt: expected eps={1e-8}, got {denom_z[0].item()} — '\n"
        "    f'if ~3.16e-4 you wrote sqrt(v_hat + eps).'\n"
        ")\n"
        "\n"
        "# Case C: input v not mutated.\n"
        "v_in = t.tensor([0.5, 0.25])\n"
        "snap = v_in.clone()\n"
        "_ = cx23_v_step_and_denom(v_in, t.tensor([1.0, 1.0]), 0.999, 3, 1e-8)\n"
        "assert t.equal(v_in, snap), 'input v was mutated'\n"
        "\n"
        "# Case D: many constant-g steps — denom should approach |g| + eps.\n"
        "v = t.zeros(3)\n"
        "g = t.tensor([2.0, -3.0, 0.5])\n"
        "denom_last = None\n"
        "for step in range(1, 50):\n"
        "    v, denom_last = cx23_v_step_and_denom(v, g, 0.999, step, 1e-8)\n"
        "assert t.allclose(denom_last, g.abs() + 1e-8, atol=1e-3), (\n"
        "    f'after many constant-g steps, denom should be |g|+eps; got {denom_last}, expected {g.abs() + 1e-8}'\n"
        ")\n"
        "\n"
        "# Case E: cross-check vs torch.optim.Adam after one step.\n"
        "# Adam computes denom = (exp_avg_sq.sqrt() / sqrt(bias_correction2)) + eps  internally,\n"
        "# which equals sqrt(v_hat) + eps for v_hat = exp_avg_sq / (1 - beta2**t).\n"
        "t.manual_seed(17)\n"
        "p = t.nn.Parameter(t.randn(5))\n"
        "g_ref = t.randn(5)\n"
        "opt = t.optim.Adam([p], lr=0.0, betas=(0.9, 0.999), eps=1e-8)\n"
        "p.grad = g_ref.clone()\n"
        "opt.step()\n"
        "ref_v = opt.state[p]['exp_avg_sq']\n"
        "ref_v_hat = ref_v / (1 - 0.999 ** 1)\n"
        "ref_denom = ref_v_hat.sqrt() + 1e-8\n"
        "my_v_new, my_denom = cx23_v_step_and_denom(t.zeros(5), g_ref, 0.999, 1, 1e-8)\n"
        "assert t.allclose(my_v_new, ref_v, atol=1e-6), 'v_new disagrees with torch.optim.Adam'\n"
        "assert t.allclose(my_denom, ref_denom, atol=1e-6), (\n"
        "    f'denom disagrees with torch.optim.Adam: max err = {(my_denom - ref_denom).abs().max()}'\n"
        ")\n"
        "\n"
        "# Case F: shape preserved on multi-dim tensors.\n"
        "v_2d = t.rand(3, 4)\n"
        "g_2d = t.randn(3, 4)\n"
        "v_out, d_out = cx23_v_step_and_denom(v_2d, g_2d, 0.999, 5, 1e-8)\n"
        "assert tuple(v_out.shape) == (3, 4) and tuple(d_out.shape) == (3, 4)"
    ),
    "solution_body": (
        "def cx23_v_step_and_denom(v, g, beta2, t_step, eps):\n"
        "    # Atom A (ema-second-moment): EMA on g**2.\n"
        "    v_new = beta2 * v + (1.0 - beta2) * g.pow(2)\n"
        "    # Bias-correct (intermediate, not its own atom in this drill).\n"
        "    v_hat = v_new / (1.0 - beta2 ** t_step)\n"
        "    # Atom B (sqrt-eps-stabilize): Adam convention — eps OUTSIDE the sqrt.\n"
        "    denom = v_hat.sqrt() + eps\n"
        "    return v_new, denom"
    ),
    "solution_notes": (
        "**This pair is Adam's actual denominator.** The full step is "
        "`p <- p - lr * m_hat / denom`. Getting eps inside vs outside the sqrt is the most "
        "common Adam-from-scratch bug after the wrong-beta-divisor bug. Inside-sqrt "
        "(BatchNorm style) silently weakens the eps guard — `sqrt(eps) >> eps` when eps is "
        "small, so the divide is less aggressive in tiny-`v_hat` regimes.\n\n"
        "**The 'constant-g → |g| + eps' invariant** is the cleanest end-to-end check: after many "
        "steps, `v_hat = g**2` exactly (by the bias-correction round-trip from cx21), so "
        "`sqrt(v_hat) = |g|`, and the denominator carries no information about gradient sign — "
        "which is why Adam's per-coordinate scale only cares about magnitude."
    ),
    "extra_imports": NO_EXTRA_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "kcs": ["ema-second-moment", "sqrt-eps-stabilize"],
    "lo": (
        "Compose the second-moment EMA (v <- beta2*v + (1-beta2)*g**2) with the "
        "sqrt-eps-stabilize denominator (sqrt(v_hat) + eps with eps OUTSIDE the sqrt) — Adam's "
        "load-bearing adaptive-scale wiring, matching torch.optim.Adam's reference denominator."
    ),
}


# ===========================================================================
# cx24 — bias-correction-divide + sqrt-eps-stabilize: m_hat / (sqrt(v_hat) + eps)
# ===========================================================================
spec_24 = {
    "atom_ids": ["bias-correction-divide", "sqrt-eps-stabilize"],
    "subtopics": _subs(["bias-correction-divide", "sqrt-eps-stabilize"]),
    "primary_atom": "bias-correction-divide",
    "part": "part3",
    "exercise_index": 24,
    "exercise_title": "Adam ratio: m_hat / (sqrt(v_hat) + eps)",
    "slug": "adam-ratio-mhat-over-sqrt-vhat-plus-eps",
    "atom_recap_md": (
        "## How these two atoms compose\n\n"
        "Adam's full parameter update is `p <- p - lr * step` where the step is built from "
        "TWO bias-corrected moments and the sqrt-eps denominator:\n"
        "```\n"
        "step = m_hat / (sqrt(v_hat) + eps)\n"
        "     = ( m / (1 - beta1**t) )                         <- bias-correction-divide (atom A)\n"
        "       / ( sqrt( v / (1 - beta2**t) ) + eps )         <- sqrt-eps-stabilize (atom B)\n"
        "```\n"
        "**Both atoms are bias-correction-divide and sqrt-eps-stabilize.** Atom A is invoked "
        "TWICE in the equation (once for `m_hat`, once inside the sqrt for `v_hat`), and atom B "
        "wraps the result. The composition is the ratio that gives Adam its adaptive, "
        "scale-invariant per-coordinate step.\n\n"
        "**Why this exact form is scale-invariant.** Multiply `g` by 10. Then `m` scales by 10, "
        "and `v` scales by 100, so `sqrt(v_hat)` scales by 10 — the ratio is invariant. That's "
        "the whole reason Adam is so robust to gradient magnitude across layers.\n\n"
        "**Eps placement matters AT SCALE.** With eps-outside (Adam convention), the ratio at "
        "small `v_hat` becomes `m_hat / eps` — a finite-but-large number. With eps-inside "
        "(BatchNorm convention), the ratio is `m_hat / sqrt(eps)` — much smaller. Adam wants the "
        "MORE AGGRESSIVE step in tiny-`v` regimes, so eps-outside is correct.\n\n"
        "**Anatomy.**\n"
        "```python\n"
        "def adam_ratio(m, v, beta1, beta2, t, eps):\n"
        "    m_hat = m / (1 - beta1 ** t)\n"
        "    v_hat = v / (1 - beta2 ** t)\n"
        "    return m_hat / (v_hat.sqrt() + eps)\n"
        "```"
    ),
    "prompt_body": (
        "Implement `cx24_adam_ratio(m, v, beta1, beta2, t_step, eps)`.\n\n"
        "Inputs:\n"
        "- `m`: first-moment buffer (Tensor).\n"
        "- `v`: second-moment buffer (Tensor, non-negative elementwise — caller's responsibility).\n"
        "- `beta1`, `beta2`: floats.\n"
        "- `t_step`: int >= 1.\n"
        "- `eps`: float (typically 1e-8).\n\n"
        "Steps:\n"
        "1. `m_hat = m / (1 - beta1 ** t_step)` (atom: bias-correction-divide).\n"
        "2. `v_hat = v / (1 - beta2 ** t_step)` (atom: bias-correction-divide — second use).\n"
        "3. `denom = sqrt(v_hat) + eps` (atom: sqrt-eps-stabilize — eps OUTSIDE).\n"
        "4. Return `m_hat / denom` (a fresh tensor, same shape as `m`).\n\n"
        "Do not mutate `m` or `v`. Cross-check the full-step equation vs `torch.optim.Adam`'s "
        "reference, since this is the EXACT ratio Adam multiplies by `-lr`.\n\n"
        "Tests verify:\n"
        "- `m=0, v=0 -> ratio=0` (zero numerator, finite denominator from eps).\n"
        "- Sign of the ratio matches sign of `m` elementwise (denominator is always positive).\n"
        "- Scale-invariance: multiplying `(m, v)` by `(c, c**2)` leaves the ratio unchanged.\n"
        "- Cross-check: replicates Adam's update direction (`-step`) from `torch.optim.Adam`.\n"
        "- Eps placement: with `m=1, v=0`, ratio == `1 / eps` (eps outside), NOT `1 / sqrt(eps)`."
    ),
    "stub_body": (
        "def cx24_adam_ratio(m: Tensor, v: Tensor, beta1: float, beta2: float, t_step: int, eps: float) -> Tensor:\n"
        "    \"\"\"Return m_hat / (sqrt(v_hat) + eps), the Adam step direction.\"\"\"\n"
        "    raise NotImplementedError"
    ),
    "test_body": (
        "# Case A: m=0, v=0 — ratio should be 0 (numerator is 0, denominator is eps>0).\n"
        "m = t.zeros(3)\n"
        "v = t.zeros(3)\n"
        "r = cx24_adam_ratio(m, v, 0.9, 0.999, t_step=1, eps=1e-8)\n"
        "assert tuple(r.shape) == (3,)\n"
        "assert t.allclose(r, t.zeros(3), atol=1e-12), f'ratio with zero m and v must be 0; got {r}'\n"
        "\n"
        "# Case B: closed-form step 1. Suppose g was constant; then after 1 step:\n"
        "#   m = (1 - beta1) * g, v = (1 - beta2) * g**2\n"
        "#   m_hat = g, v_hat = g**2, sqrt(v_hat) = |g|.\n"
        "#   ratio = g / (|g| + eps) ~ sign(g) for |g| >> eps.\n"
        "beta1 = 0.9\n"
        "beta2 = 0.999\n"
        "g = t.tensor([1.0, -2.0, 0.5, -0.1])\n"
        "m = (1 - beta1) * g\n"
        "v = (1 - beta2) * g.pow(2)\n"
        "r = cx24_adam_ratio(m, v, beta1, beta2, t_step=1, eps=1e-8)\n"
        "expected = g / (g.abs() + 1e-8)\n"
        "assert t.allclose(r, expected, atol=1e-5), f'closed-form mismatch: got {r}, expected {expected}'\n"
        "# Sign sanity.\n"
        "assert t.equal(r.sign(), g.sign()), f'ratio sign should match m sign; got signs {r.sign()}'\n"
        "\n"
        "# Case C: scale-invariance — scaling (m, v) by (c, c**2) is a no-op on the ratio.\n"
        "m_a = t.tensor([0.3, -0.7, 0.2])\n"
        "v_a = t.tensor([0.5, 0.5, 0.5])\n"
        "c = 100.0\n"
        "r_a = cx24_adam_ratio(m_a, v_a, 0.9, 0.999, t_step=10, eps=1e-12)  # tiny eps to expose scale-invariance.\n"
        "r_b = cx24_adam_ratio(m_a * c, v_a * (c ** 2), 0.9, 0.999, t_step=10, eps=1e-12)\n"
        "assert t.allclose(r_a, r_b, atol=1e-5), (\n"
        "    f'Adam ratio must be scale-invariant under (m, v) -> (c*m, c**2*v); '\n"
        "    f'got r_a={r_a}, r_b={r_b}'\n"
        ")\n"
        "\n"
        "# Case D: eps placement. m=1, v=0 — ratio = 1 / (0 + eps) = 1/eps.\n"
        "m1 = t.tensor([1.0])\n"
        "v0 = t.tensor([0.0])\n"
        "r_eps = cx24_adam_ratio(m1, v0, 0.9, 0.999, t_step=10000, eps=1e-8)  # large t so m_hat ~ m.\n"
        "# m_hat ~ 1.0, v_hat ~ 0.0, denom = sqrt(0) + 1e-8 = 1e-8. ratio = 1e8.\n"
        "expected_huge = 1.0 / 1e-8\n"
        "assert abs(r_eps.item() - expected_huge) / expected_huge < 0.01, (\n"
        "    f'eps placement: expected ratio ~{expected_huge:.2e} (eps OUTSIDE sqrt); got {r_eps.item():.2e} — '\n"
        "    f'if you got ~1e4 you wrote sqrt(v_hat + eps) instead of sqrt(v_hat) + eps.'\n"
        ")\n"
        "\n"
        "# Case E: inputs not mutated.\n"
        "m_in = t.tensor([0.5, -0.5])\n"
        "v_in = t.tensor([1.0, 4.0])\n"
        "snap_m, snap_v = m_in.clone(), v_in.clone()\n"
        "_ = cx24_adam_ratio(m_in, v_in, 0.9, 0.999, t_step=3, eps=1e-8)\n"
        "assert t.equal(m_in, snap_m), 'm was mutated'\n"
        "assert t.equal(v_in, snap_v), 'v was mutated'\n"
        "\n"
        "# Case F: cross-check vs torch.optim.Adam — the ratio is the parameter step magnitude / lr.\n"
        "# We replicate Adam's first step by hand and compare with the actual delta on p.\n"
        "t.manual_seed(23)\n"
        "p_ref = t.nn.Parameter(t.randn(5))\n"
        "p_before = p_ref.detach().clone()\n"
        "g_ref = t.randn(5)\n"
        "lr = 0.05\n"
        "eps = 1e-8\n"
        "opt = t.optim.Adam([p_ref], lr=lr, betas=(0.9, 0.999), eps=eps)\n"
        "p_ref.grad = g_ref.clone()\n"
        "opt.step()\n"
        "# Adam: p <- p - lr * ratio, so ratio_observed = (p_before - p_after) / lr.\n"
        "ratio_observed = (p_before - p_ref.detach()) / lr\n"
        "# Reconstruct what cx24 returns at step 1 starting from m=0, v=0.\n"
        "m0 = (1 - 0.9) * g_ref\n"
        "v0 = (1 - 0.999) * g_ref.pow(2)\n"
        "ratio_ours = cx24_adam_ratio(m0, v0, 0.9, 0.999, 1, eps)\n"
        "assert t.allclose(ratio_ours, ratio_observed, atol=1e-5), (\n"
        "    f'cx24 ratio disagrees with torch.optim.Adam step direction; '\n"
        "    f'max err = {(ratio_ours - ratio_observed).abs().max().item():.2e}'\n"
        ")\n"
        "\n"
        "# Case G: shape preserved on multi-dim.\n"
        "m_2d = t.randn(3, 4)\n"
        "v_2d = t.rand(3, 4)\n"
        "r_2d = cx24_adam_ratio(m_2d, v_2d, 0.9, 0.999, 5, 1e-8)\n"
        "assert tuple(r_2d.shape) == (3, 4)"
    ),
    "solution_body": (
        "def cx24_adam_ratio(m, v, beta1, beta2, t_step, eps):\n"
        "    # Atom A (bias-correction-divide): applied to m...\n"
        "    m_hat = m / (1.0 - beta1 ** t_step)\n"
        "    # ...and to v (same atom, second invocation, using beta2).\n"
        "    v_hat = v / (1.0 - beta2 ** t_step)\n"
        "    # Atom B (sqrt-eps-stabilize): Adam convention — eps OUTSIDE the sqrt.\n"
        "    denom = v_hat.sqrt() + eps\n"
        "    return m_hat / denom"
    ),
    "solution_notes": (
        "**This is the load-bearing line of Adam.** Everything else (m/v EMAs, beta defaults, lr "
        "schedule, weight decay) is decoration around this ratio. Getting the ratio's structure "
        "wrong is what makes a from-scratch Adam train slower than `torch.optim.Adam`.\n\n"
        "**Two atoms, three operations.** The bias-correction-divide atom is invoked TWICE in "
        "the same expression — once on `m`, once on `v` — because both moments need de-biasing "
        "at the same step `t`. The sqrt-eps-stabilize atom wraps the v-side result. This is the "
        "most multiply-invoked atom composition in the entire Adam step.\n\n"
        "**Scale invariance is the diagnostic.** If your ratio depends on the absolute magnitude "
        "of `g` (not just its direction relative to history), you've broken the bias correction "
        "or moved eps inside the sqrt. Case C is the cleanest invariant: scaling `(m, v)` by "
        "`(c, c**2)` MUST leave the ratio invariant (for `eps << sqrt(v_hat)`)."
    ),
    "extra_imports": NO_EXTRA_IMPORTS,
    "bloom_level": "Apply",
    "difficulty_num": 4,
    "kcs": ["bias-correction-divide", "sqrt-eps-stabilize"],
    "lo": (
        "Compose the bias-correction divide (applied to BOTH m and v with their own betas) with "
        "the Adam-convention sqrt-eps stabilization (eps OUTSIDE the sqrt) to produce Adam's "
        "scale-invariant step direction m_hat / (sqrt(v_hat) + eps), matching torch.optim.Adam."
    ),
}


SPECS = [spec_19, spec_20, spec_21, spec_22, spec_23, spec_24]


def main() -> None:
    for spec in SPECS:
        out = emit_composite(spec)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
