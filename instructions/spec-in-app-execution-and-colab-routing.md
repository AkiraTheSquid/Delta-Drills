# Spec — in-app execution, expected-output display, and graph→Colab routing

Status: SUPERSEDED. Written 2026-07-31, superseded the same day.

**What happened.** Threads A and D were overtaken by a decision to revert the
practice frontend to `pre-selfreport-teardown-20260731`. Two premises this spec
rested on were wrong:

- §2 claimed the learner had no oracle. The pre-teardown `practice/ui.js:543`
  already rendered `expected:` / `got:` per failing case under a "Failed N of M
  test cases" header. The revert fixes problem 486 for the whole bank; the
  generated check cells in §A are unnecessary while problems are worked in-app.
- §D.2 claimed the deleted `runner.js` was Pyodide-only and had to be repointed
  at the backend. It already called `POST /api/practice/run-code` in backend
  mode (`runner.js:314` at that tag), with Pyodide as the guest/offline
  fallback only. The estimate of 3–5 days for item 4 was wrong; it was a
  `git restore`.
- The objection that reverting would discard the four fixes in `01d2b09` also
  fails: all four are artifacts of the self-report/Colab era
  (`questionHasColabRoute()`, a stale notebook href, `rel="noreferrer"`,
  anchor gating) and the revert removes each mechanism that caused them.

**Still live.** §0 (the chapter/GPU correction), §D.3–D.5 (Modal costing for
chapters 0–3, and the precompute-cache trap around the harness key), and
thread C (graph→Colab) survive as analysis and should be re-read before any
future re-implementation of the Colab split. The Colab infrastructure itself is
preserved on branch `colab-selfreport` and tag `colab-era-final`.

Covers three threads raised together, because they turn out to share one root
cause and one cheap fix.

---

## 0. Correction to the earlier cost model

An earlier analysis said chapter1_transformer_interp was expensive because 129
of its 210 notebooks call `from_pretrained`. That conflated **loads weights**
with **needs a GPU**. The actual model inventory across all of ARENA 5.0 in
`arena-book-colab/`:

| Chapter | Models loaded | Largest | CPU-viable |
|---|---|---|---|
| `ch-1-foundations` (448 drills) | none | — | yes |
| `chapter0_fundamentals` | none — builds nets from scratch | — | yes* |
| `chapter1_transformer_interp` | `gpt2-small` ×88 | 124M | **yes** |
| `chapter2_rl` | `pythia-14m` ×6, `distilbert-imdb` ×2, `Llama-3.2-1B` ×1 | 1B | yes* |
| `chapter3_llm_evals` | none — pure API | — | yes |
| `chapter4_alignment_science` | `gemma-2-27b-it` ×57, `Qwen2.5-14B` ×12, `Llama-3.1-8B` ×12, `DeepSeek-R1-Distill-{1.5B,8B,14B}` | **27B** | **no** |

Commands used:

```bash
grep -rhaoE "from_pretrained\(\s*\\\\?[\"'][^\"'\\\\]+" <chapter> | sed -E "s/.*[\"']//" | sort | uniq -c
```

`gpt2-small` is ~500 MB in fp32 and runs interp exercises on CPU comfortably.
It is not an A100 job. **The conclusion Seth pushed back with is correct:
chapters 0–3 are CPU-viable; only chapter 4 needs a GPU.**

\* Caveats, which are about *time*, not *weights*:

- `chapter0_fundamentals` asks the learner to train ResNet/CNN models. Most
  exercises test a layer against `torch`'s reference and are instant; the
  handful that actually train are slow on CPU.
- `chapter2_rl` trains PPO/DQN loops. CartPole is fine on CPU; Atari is not.

These are a minority of exercises and can be flagged "run this one in Colab"
rather than blocking the whole tier.

---

## 1. Root cause shared by threads A and C

`This-Directory-Only/questions_full.json` already stores, per question,
everything needed to tell a learner whether they were right:

```json
{
  "id": 486,
  "expected_output": "('torch.int64', 'torch.float32', False)",
  "test_cases": [
    {"setup_code": "import torch as t\nx = [[1, 2], [3, 4]]",
     "call": "solve(x)",
     "expected_expr": "('torch.int64', 'torch.float32', False)"},
    {"setup_code": "import torch as t\nx = [[1.0, 2.0]]",
     "call": "solve(x)",
     "expected_expr": "('torch.float32', 'torch.float32', True)"},
    {"setup_code": "import torch as t\nx = [[1, 2.5]]",
     "call": "solve(x)",
     "expected_expr": "('torch.float32', 'torch.float32', True)"},
    {"setup_code": "import torch as t\nx = [[0]]",
     "call": "solve(x)",
     "expected_expr": "('torch.int64', 'torch.float32', False)"}
  ]
}
```

`scripts/generate_colab_notebooks.py:170` `problem_cells()` emits exactly three
things — a header markdown cell carrying `question_text`, an optional hints
`<details>` cell, and a code cell carrying `starter_code`. It never reads
`expected_output` or `test_cases`.

So the notebook is a prompt and a blank editor with no oracle. That is the whole
of the "I had no idea whether I got it right" problem. Nothing is missing from
the data; one function drops it on the floor.

---

## 2. Thread A — make the drill notebooks self-checking

### A.1 Design constraint

A notebook's source is visible. Writing `expected_expr` into a check cell
hands the learner the answer for questions like 486 where the expected value
*is* the insight (`torch.int64` vs `torch.float32`).

Two mitigations, both cheap:

**Hash mode (default).** The check cell carries SHA-256 digests, not values.
The learner gets a per-case verdict and their own actual value, but not the
expected one until they ask.

```python
# === check ===
import hashlib, json

def _dd_digest(v):
    return hashlib.sha256(repr(v).encode()).hexdigest()[:16]

_dd_cases = [
    ("import torch as t\nx = [[1, 2], [3, 4]]", "solve(x)", "9f2c…"),
    ...
]
_dd_pass = 0
for _setup, _call, _want in _dd_cases:
    _ns = dict(globals())
    try:
        exec(_setup, _ns)
        _got = eval(_call, _ns)
        _ok = _dd_digest(_got) == _want
    except Exception as _e:
        _got, _ok = f"{type(_e).__name__}: {_e}", False
    print(("PASS  " if _ok else "FAIL  ") + _call + "  ->  " + repr(_got))
    _dd_pass += _ok
print(f"\n{_dd_pass}/{len(_dd_cases)} cases passed")
```

**Reveal mode.** A second collapsed `<details>` markdown cell holding the
plaintext expected values, so a stuck learner can open it deliberately. This is
the same affordance the existing hints cell already uses.

Hash mode preserves the one thing Colab otherwise cannot do — hide the oracle —
which was the single limitation named against staying on Colab.

### A.2 Beacon

The check cell is also the natural place to fire the completion beacon that
chapters 0–4 already have (`_dd_report_complete()` against
`POST /api/arena/complete`). When `_dd_pass == len(_dd_cases)`, report. This
closes the self-report gap on the 448 drills without any platform change.

### A.3 Changes

- `scripts/generate_colab_notebooks.py`
  - `problem_cells()` gains a `check` cell after the code cell, id
    `mint(f"dd-q{qid}-check")`.
  - New helper `check_cell(qid, cases, mode)`; digests computed at generate
    time in Python so the notebook never contains the plaintext in hash mode.
  - Skip when `test_cases` is empty (75 of 449 bank questions carry no tags;
    verify how many lack `test_cases` before assuming coverage).
- Regenerate: 9 notebooks, then `publish_colab_notebooks.sh`.
- `Local_Deployed_Shared/lessons/colab_notebooks.json` is regenerated in the
  same pass — do not hand-edit.

### A.4 Risk

Republishing rewrites the public `arena-book-colab` repo. Any learner with a
Colab tab open on an old revision keeps their copy; new opens get the new one.
Non-destructive, but it is an outward-facing publish and should be confirmed.

### A.5 Effort

**Half a day**, including regeneration and spot-checking q486.

---

## 3. Thread C — knowledge-graph node → Colab section

### C.1 What already exists

`Local_Deployed_Shared/practice/colab-route.js:221`:

```js
function urlForKc(kc) {
  var anchor = anchorForKc(kc);      // index.kps[kc] -> "dd-kp-<slug>"
  if (!anchor) return "";
  return notebookUrl(lessonForKc(kc), anchor);   // index.kcs[kc] -> lesson
}
function openKc(kc) { return openUrl(urlForKc(kc)); }
```

`lessons/colab_notebooks.json` currently maps **63 KCs** to both a lesson and a
`dd-kp-` anchor. The graph carries 64 KCs, so coverage is effectively complete.

Both `Local_Deployed_Shared/concept-graph/lesson-graph.js` and
`Local_Deployed_Shared/practice/colab-route.js` are loaded by the same
`Local_Deployed_Shared/index.html`, so `window.ColabRoute.openKc(kc)` is
directly callable from the graph with no plumbing. (Note the top-level
`concept-graph/` directory is unrelated — it holds the GraphML sources.)

**This feature is one button.**

### C.2 Changes

- `lesson-graph.js`: in the dock (`~:635`) and the maximize overlay
  (`openMaximize`, `~:1340`), add an "Open in Colab" control calling
  `window.ColabRoute.openKc(kc)`.
- Gate on `ColabRoute.urlForKc(kc)` returning non-empty — a node with no
  mapping must not render a dead button. Follow the precedent set in
  `graph-jump.js`, which deliberately shows an *explicitly untagged* state
  rather than hiding, so an unmapped concept is visible rather than silent.
- Reuse `TARGET = "delta-drills-colab"` so the graph steers the same single tab
  the practice queue does. This is free — `openKc` already goes through
  `openUrl`.
- Watch: `window.open` needs transient user activation. A click qualifies.
  Do not wire this to hover.

### C.3 Scope limit

`index.kps` only covers the 9 generated `ch-1-foundations` notebooks. Graph
nodes for ARENA chapters 0–4 have no `dd-kp-` anchor and will correctly render
no button. Extending to ARENA notebooks means anchoring upstream `.ipynb` files,
which are nbformat 4.2 with no stable cell ids — that is a separate project.

### C.4 Effort

**Half a day.**

---

## 4. Thread D — Tier 1 reconsidered

### D.1 The claim being vetted

"Tier 1 wouldn't be that hard, and it covers the first three chapters without
paying much of anything."

Split into two claims, because they have different answers.

### D.2 Claim 1 — the 448 drills need no Modal at all

**True, and stronger than stated.** The server-side grader was never torn down.
Still present and deployed on Fly:

- `backend/app/code_runner.py:36` — `preload_torch()` imports torch once at
  startup; submissions run in an `os.fork()` child seeing it via copy-on-write.
  Per-run cost is **milliseconds**. `torch.set_num_threads(1)` before any fork,
  because OMP pools do not survive `fork()` and deadlocked the child.
- `backend/app/practice/grading.py:86` — `grade_submission()`: function tests
  first (per-case value equality, fixture-aware seeding), then stdout match,
  then AI judge.
- `POST /api/practice/submit`, `POST /api/practice/run-code` — both live.
- The deleted frontend already called `/submit`
  (`api.js:180` at `pre-selfreport-teardown-20260731`).

`fly.toml` has `min_machines_running = 1`, so this capacity is **already paid
for**. Marginal cost per graded submission: **$0**.

What is missing is the editor UI: ~1,650 net lines across
`practice/{runner,notebook,visuals}.js` and
`styles/practice/{editor,notebook,ladder}.css`, recoverable from
`pre-selfreport-teardown-20260731`.

Note the old `runner.js` is **Pyodide**, i.e. client-side, and cannot run torch.
The restore must point the run button at `/api/practice/run-code` instead. That
is the real work in this item, not the CSS.

**Effort: 3–5 days. Cost: $0. No Modal.**

### D.3 Claim 2 — chapters 0–3 in-app on Modal

Also broadly true on cost, given §0. Sizing a CPU sandbox at 4 cores / 8 GiB:

```
4 × $0.0000131  = $0.0000524 /s
8 × $0.00000222 = $0.0000178 /s
                = $0.0000702 /s  ->  $0.25 /hour
```

- 30-minute persistent session: **$0.126**
- Stateless 30-second run: **$0.002**
- `gpt2-small` cached in a Modal Volume after first pull: ~2 s load, ~$0.0001

Twenty 30-minute sessions/month/learner = **$2.52**. Under Starter's $30/month
credit that is ~11 learners free, and at scale it is comfortably inside a $12
subscription.

**So the cost objection to chapters 0–3 does not hold.** The objection that
does hold is engineering scope, below.

### D.4 What chapters 0–3 in-app actually requires

The 448 drills are *stateless*: `solve(rows)` is self-contained, one call, one
verdict. ARENA exercises are *stateful* — cell 12 uses cell 3's variables, and
the exercise is embedded in a notebook with setup, prose, and plots.

Required, none of which the current backend has:

1. **Session-scoped interpreter.** One Modal Sandbox per active learner, keyed
   to their session, kept warm across submissions. Needs create/reuse/expire
   logic and a hard idle cap — Modal bills idle time, `scaledown_window`
   default 60 s, configurable 2 s–20 min.
2. **Environment image.** `transformer_lens`, `einops`, `jaxtyping`, `torch`,
   plus the ARENA exercise package. Baked into a Modal Image at build time, not
   pip-installed per session, or the cost model above is wrong by an order of
   magnitude.
3. **Weight cache.** `gpt2-small` in a Modal Volume, mounted read-only.
4. **Output transport.** stdout/stderr streaming, tracebacks, and matplotlib
   figures as images. `visuals.js` (466 lines, deleted) already renders arrays
   to canvas and is partly reusable; figures are new.
5. **Cell-oriented UI.** The deleted `notebook.js` (369 lines) is the closest
   thing, but its own header records that it faked cross-cell state by
   re-running every cell above, and called a persistent per-learner interpreter
   "a server-side session to build, expire and secure, which is not worth it."
   With a real session that compromise goes away — but building the session is
   precisely the work being estimated.
6. **Isolation.** Today `code_runner` forks inside the API process, on the box
   holding DB credentials. Acceptable for a small trusted user base; not for
   public multi-tenant. Modal Sandboxes are the stated fix, and this is the
   strongest non-cost argument for Modal.
7. **Content ingestion.** 385 ARENA `.ipynb` files would need rendering or
   conversion. Upstream notebooks are nbformat 4.2 with no cell ids.

**Effort: 4–6 weeks** for chapters 0–3 to a usable MVP, dominated by items 1,
4, 5 and 7. Item 7 alone could be scoped down by starting with one part of one
chapter.

### D.5 The caching idea, evaluated honestly

The proposal: if the learner submits code matching the canonical solution,
return a precomputed verdict instead of executing; otherwise execute normally.

**Where it does not help.** The submissions it would short-circuit are the 448
drills, which cost $0 today on the Fly fork runner (§D.2) and ~$0.002 on Modal.
Caching a free operation saves nothing. And the cost that *does* matter in a
Modal design is session idle time (§D.4 item 1), which no submission-level cache
touches.

**Where it does help, a lot.** `grading.py:31`:

```python
def run_and_get_expected_output(answer_code: str) -> str:
    result = run_code(answer_code, timeout=20)
    return result.stdout.strip()
```

This runs the **canonical answer** on the stdout-prediction path
(`grading.py:~137`) and on the AI-judge fallback — that is, the app executes
code twice per submission on those paths, once for the learner and once to
recompute an expected value that does not change between learners.
Precomputing it into the bank at build time removes half the execution on those
paths.

Better still, **that precomputed value is exactly what thread A needs to
display.** The two threads are the same work: compute expected output once,
store it, then both show it in the notebook and stop recomputing it at grade
time.

**The trap.** The code comments are emphatic about why the stored string is not
currently trusted:

> "never trust the stored CSV-era string" — 85 stored strings came from an
> UNSEEDED CSV-era capture, unreachable by any honest solution under the seeded
> harness.

So a precompute cache is only safe if its key includes the **harness** —
preamble plus RNG seed plus torch version — and is invalidated when any of
those change. A stale expected value does not fail loudly; it silently marks
correct solutions wrong and writes that into the mastery model. This is the one
place in this spec where getting it wrong is expensive, and it should go
through `/critic` before merge.

**A verdict cache keyed on normalized learner code** (strip comments and
whitespace, then hash) is a reasonable secondary optimization, but it must
store only *deterministic* verdicts — never AI-judge outputs, which are not
reproducible and would freeze one sampling of `gpt-4o-mini` into permanent
truth.

### D.6 Recommended sequence

| # | Item | Effort | Cost | Depends on |
|---|---|---|---|---|
| 1 | Precompute `expected_output` under the real harness | 1–2 d | $0 | — |
| 2 | Thread A — self-checking drill notebooks + beacon | 0.5 d | $0 | 1 |
| 3 | Thread C — graph node → Colab | 0.5 d | $0 | — |
| 4 | Restore in-app editor against `/api/practice/run-code` | 3–5 d | $0 | — |
| 5 | Chapters 0–3 in-app on Modal | 4–6 wk | ~$2.50/learner/mo | 4 |
| 6 | Chapter 4 on GPU | — | ~$20–25/learner/mo | 5 |

Items 1–4 are two weeks of work, cost nothing to run, and deliver: a learner
who can tell whether they were right, real hidden-test grading on all 448
drills, and a graph that navigates to the teaching material.

Item 5 is a real project whose cost objection has been withdrawn but whose
scope objection has not. Item 6 remains the only genuinely expensive tier, and
Colab already handles it via the learner's own subscription and API key.

---

## 5. Open questions

- ~~How many bank questions carry `test_cases`?~~ **Resolved: 499/499.** Every
  question is `submission_mode: "function"` and carries both `test_cases` and a
  non-empty `expected_output`, so thread A applies to the whole bank with no
  skip path needed:

  ```bash
  python3 -c "
  import json; qs=json.load(open('This-Directory-Only/questions_full.json'))
  print(len(qs), sum(1 for q in qs if q.get('test_cases')),
        sum(1 for q in qs if (q.get('expected_output') or '').strip()))"
  # -> 499 499 499
  ```

  Note this is 499 bank questions against 424 mapped into notebooks by
  `colab_notebooks.json` — the 75-question gap is the untagged CNN/PyTorch
  -Fundamentals rows described in `graph-jump.js`. They are gradeable but not
  currently placed in a notebook.
- Hash mode or reveal mode as the default for §A.1?
- Item 1 changes a grading input. `/critic` before it ships.
