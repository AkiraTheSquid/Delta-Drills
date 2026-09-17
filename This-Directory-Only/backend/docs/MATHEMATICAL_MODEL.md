# Delta Drills — Mathematical Model of the Mastery Engine

*Generated 2026-06-02 from the live code in `backend/app/`. Every formula below
is transcribed from `bkt_mastery.py`, `prioritization.py`, `concept_graph.py`,
and `adaptive.py` — not idealized. Where the code uses a v0 default constant
that is **not** literature-derived, it is flagged.*

---

## 0. The two objects and how they connect

Delta Drills runs on two coupled structures:

1. **The concept graph** $G = (\mathcal{A}, E)$ — a static, immutable-at-runtime
   directed acyclic graph. Nodes are *atoms* (drillable skills); edges are
   prerequisite relations, a subset of which are *encompassing*.
2. **The per-learner BKT state** — a time-stamped belief, one scalar per atom,
   that the learner knows that atom. This is the only thing that changes as a
   student practices.

The graph is the **fixed scaffolding**; the BKT state is the **moving belief**
that flows over it. The graph influences the belief in exactly two places:

- **Forward (gating):** prerequisite edges decide *what a learner is allowed to
  see next* (§4).
- **Backward (FIRe credit):** encompassing edges decide *how credit from one
  practiced atom trickles to the simpler atoms it subsumes* (§3.3).

Everything else — selection, difficulty, area scores — is a read of the BKT
state (§5).

---

## 1. The concept graph $G$

### 1.1 Atoms (nodes)

The atom set is
$$
\mathcal{A} = \{a_1, \dots, a_N\}, \qquad N = 207 \text{ (current drillable graph)}.
$$
Each atom $a$ (schema: `ConceptNode`) carries `id`, `title`, `topic`,
`subtopic`, `kind ∈ {knowledge, skill, strategy}`, `tier ∈ {core, supplemental}`.
Only `topic` enters the math (area scores, §5.3); the rest is metadata.

### 1.2 Prerequisite edges

An edge $e = (p \to d) \in E$ (schema: `PrerequisiteEdge`) states that atom
$p$ (the **prerequisite**, simpler) must precede atom $d$ (the **dependent**,
more advanced). Each edge carries:

| Field | Symbol | Meaning |
|---|---|---|
| `prerequisite_id` | $p$ | the simpler atom |
| `dependent_id` | $d$ | the advanced atom |
| `is_encompassing` | $\iota_e \in \{0,1\}$ | does mastering $d$ implicitly exercise $p$? |
| `propagation_weight` | $w_e \in [0,1]$ | fraction of credit that trickles $d \to p$ |
| `is_hard_gate` | — | reserved; gating currently uses all edges |
| `weight`, `confidence`, `rationale` | — | metadata / provenance |

**Consistency invariant** (enforced in `concept_graph._validate_graph`):
$$
\iota_e = 0 \iff w_e = 0, \qquad \iota_e = 1 \implies w_e > 0.
$$
So encompassing $\subseteq$ prerequisite, and the propagation weight is
meaningful **only** on encompassing edges.

### 1.3 The two induced relations

From $E$ the engine builds two single-hop indices (both `@lru_cache`d, since
$G$ is immutable at runtime):

**Gating prerequisites** of an atom (`_prereq_index`):
$$
\mathrm{Pre}(d) = \{\, p : (p \to d) \in E,\ p \notin \mathcal{A}_{\text{NG}} \,\}.
$$
$\mathcal{A}_{\text{NG}}$ is `NON_GATING_ATOMS` — a hand-maintained blocklist of
infrastructure/tooling/dataset atoms (e.g. `cuda-availability-check`,
`mnist-dataset`) that have no exercises and must never block learning.

**Encompassed atoms** of an atom (`_encompassing_index`):
$$
\mathrm{Enc}(d) = \{\, (p, w_e) : (p \to d) \in E,\ \iota_e = 1,\ w_e > 0 \,\}.
$$
Note the direction: $\mathrm{Enc}(d)$ returns the *simpler* atoms $p$ that get
credit when the *advanced* atom $d$ is practiced.

---

## 2. The learner state

For learner $u$ at wall-clock time $t$, the entire algorithmic state is two maps
over atoms (in `UserPracticeState`):
$$
L_a \in [0,1] \quad(\texttt{atom\_mastery}), \qquad
\tau_a \quad(\texttt{atom\_last\_ts}, \text{ ISO-8601}).
$$
$L_a$ is the BKT posterior $P(\text{learner knows atom } a)$ as last written;
$\tau_a$ is when it was last written. An atom never touched has no entry and is
treated as the prior $L_0$ (§3.1).

> The legacy `SubtopicState` fields (`baseline`, `p`, `n`, EWMA) still exist in
> the persisted state, but **are no longer algorithmic drivers** — since the BKT
> migration they are snapshots/readouts derived from the atom posteriors
> (`prioritization.get_subtopic_weights`). The single source of truth is
> $\{(L_a, \tau_a)\}$.

### 2.1 Global constants (all v0 defaults, not literature-derived)

$$
\begin{aligned}
L_0 &= 0.10 &&\text{(\texttt{P\_INIT}, prior } P(\text{known})) \\
T &= 0.30 &&\text{(\texttt{P\_TRANSIT}, learn rate per practice)} \\
g &= 0.20 &&\text{(\texttt{P\_GUESS}, } P(\text{correct} \mid \text{not known})) \\
s &= 0.10 &&\text{(\texttt{P\_SLIP}, } P(\text{wrong} \mid \text{known})) \\
\theta_{\text{m}} &= 0.95 &&\text{(\texttt{MASTERY\_THRESHOLD})} \\
\theta_{\text{u}} &= 0.85 &&\text{(\texttt{UNLOCK\_THRESHOLD})} \\
h &= 14 \text{ days} &&\text{(\texttt{HALF\_LIFE\_DAYS})}
\end{aligned}
$$

---

## 3. The BKT core (per atom)

Three operators act on a single atom's belief: **decay**, **observe**, and
**implicit-transit** (FIRe). They compose in `apply_attempt`.

### 3.1 Forgetting / decay

Vanilla BKT never forgets; Delta Drills regresses each belief toward the prior
$L_0$ by an exponential half-life since the atom was last touched. For a stored
belief $L$ last written at $\tau$, read at time $t$ (`decay`):
$$
\boxed{\;\mathrm{decay}(L, \tau, t) = L_0 + (L - L_0)\, \cdot\, 2^{-\Delta / h}\;}
\qquad \Delta = \max\!\Big(0,\ \tfrac{t - \tau}{86400}\Big)\ \text{days}.
$$
If $\tau$ is missing, decay is the identity. As $\Delta \to \infty$,
$L \to L_0$: stale knowledge slowly reverts to "unknown," which is what
resurfaces it for review (§5) without a separate staleness rule.

**The decay-adjusted belief read at time $t$** (`current_mastery`) is
$$
\hat{L}_a(t) = \mathrm{decay}\big(L_a, \tau_a, t\big),
$$
with $\hat{L}_a(t) = L_0$ for any never-practiced atom. Every downstream
decision uses $\hat{L}$, never the raw stored $L$.

### 3.2 Evidence update (a graded attempt)

A graded attempt on atom $a$ with outcome $x \in \{\text{correct}, \text{wrong}\}$
applies the Corbett–Anderson (1994) update: a Bayesian posterior given
guess/slip, followed by the learn-transit step (`observe`).

**Posterior given the observation:**
$$
P(\text{known} \mid \text{correct}) = \frac{L(1-s)}{L(1-s) + (1-L)\,g},
\qquad
P(\text{known} \mid \text{wrong}) = \frac{L\,s}{L\,s + (1-L)(1-g)}.
$$

**Learn-transit (the act of practicing teaches):** with posterior $L^{\ast}$,
$$
\boxed{\;\mathrm{observe}(L, x) = L^{\ast} + (1 - L^{\ast})\,T\;}
$$
This is the new belief *assuming the attempt definitely exercised $a$*.

### 3.3 Implicit-transit (one FIRe repetition)

An encompassed atom is not observed directly — it receives a *learning step of
size `gain` with no observation* (`implicit_transit`):
$$
\boxed{\;\mathrm{itransit}(L, \mathrm{gain}) = L + (1 - L)\cdot \mathrm{gain}\;}
\qquad \mathrm{gain}, L \text{ clamped to } [0,1].
$$
It has the same "move a fraction of the remaining distance to 1" shape as the
transit term, but driven by trickle-down credit instead of direct evidence.

---

## 4. Gating — the forward use of the graph

Gating decides what is *servable*. It never modifies belief.

**Atom readiness** (`atom_is_ready`): atom $a$ is ready to be *learned* iff every
gating prerequisite is already cleared:
$$
\mathrm{Ready}_u(a, t) \;=\; \bigwedge_{p \in \mathrm{Pre}(a)} \big[\, \hat{L}_p(t) \ge \theta_{\text{u}} \,\big].
$$
Root atoms ($\mathrm{Pre}(a) = \varnothing$) are always ready — the cold-start
entry points. Note the atom itself is **not** required (non-circular): readiness
gates *learning* $a$, not *having learned* it.

**Item unlock** (`item_is_unlocked`) — the single unified gate for any practice
item (bank question, drill, or composite ARENA exercise) tagged with required
atoms $R$:
$$
\mathrm{Unlocked}_u(R, t) \;=\; \bigwedge_{a \in R} \big[\, \hat{L}_a(t) \ge \theta_{\text{u}} \,\big].
$$
An item with $R = \varnothing$ is unlocked. A single-atom drill on $a$ effectively
becomes "is $a$ ready", since a bank question is gated on each tagged atom's
readiness (`question_is_unlocked`).

This one rule replaced the old per-surface ad-hoc gates (drills' flat
$0.50/0.70$, ARENA's hand-authored subtopic prereq list).

---

## 5. The attempt pipeline and credit flow — the backward use of the graph

This is where the two structures couple. When learner $u$ submits a graded
attempt at time $t$, the request handler (`feedback_router`) iterates the
question's **atom tags** — each a pair $(a, c)$ where $c \in [0,1]$ is the
**tag confidence** (how sure we are this question exercises atom $a$; an authored
drill is $\approx 1.0$, a noisy auto-tag less). For each tag it calls
`apply_attempt(a, \text{correct}, c)`.

### 5.1 The directly-practiced atom

Let $x$ be the outcome. First decay the prior to now, then apply evidence, then
**soft-apply by confidence** (the belief only moves a fraction $c$ of the way to
the full posterior):
$$
\begin{aligned}
L^{-}_a &= \mathrm{decay}(L_a, \tau_a, t) &&\text{(forgetting-adjusted prior)}\\
L^{\text{full}}_a &= \mathrm{observe}(L^{-}_a, x) &&\text{(if the tag were certain)}\\
L_a &\leftarrow L^{-}_a + \big(L^{\text{full}}_a - L^{-}_a\big)\cdot c &&\text{(confidence-scaled write)}\\
\tau_a &\leftarrow t.
\end{aligned}
$$
$c = 1 \Rightarrow$ standard BKT update; $c = 0 \Rightarrow$ no-op.

### 5.2 FIRe credit to encompassed atoms (correct attempts only)

If and only if the attempt was **correct**, define the implicit-rep magnitude
$$
\mathrm{gain} = T \cdot c,
$$
and for every encompassed pair $(p, w_e) \in \mathrm{Enc}(a)$ — i.e. every
simpler atom $p$ joined to $a$ by an encompassing edge of weight $w_e$ — apply a
single-hop implicit repetition against $p$'s own decayed prior:
$$
\begin{aligned}
L^{-}_p &= \mathrm{decay}(L_p, \tau_p, t)\\
L_p &\leftarrow \mathrm{itransit}\big(L^{-}_p,\ \mathrm{gain}\cdot w_e\big)
   \;=\; L^{-}_p + (1 - L^{-}_p)\cdot \mathrm{clamp}(T\, c\, w_e)\\
\tau_p &\leftarrow t.
\end{aligned}
$$

**Properties (by construction, asserted in `eg_validate.py`):**
- Credit flows **advanced $\to$ simpler** only (never the reverse).
- Credit fires **only on correct** attempts (a failed advanced attempt does not
  demonstrably exercise the sub-skill).
- It is **single-hop**: only atoms *directly* encompassed by $a$ are credited;
  there is no transitive cascade in one attempt. Deeper ancestors are reached
  only by separately practicing intermediate atoms.
- The trickle is **bounded**: $\mathrm{itransit}$ keeps $L_p \le 1$, and because
  $w_e, c \le 1$ the implicit gain $\le T$, so a single encompassing rep moves
  $p$ strictly less than a direct correct attempt would.
- **Encompassing edges are the only channel that shares evidence between atoms.**
  With $E_{\text{enc}} = \varnothing$ this reduces to independent per-atom BKT.

### 5.3 What persists

`apply_attempt` mutates $\{L_a\}$ and $\{\tau_a\}$ in place and returns the set
of changed atoms (direct + trickled) for logging / frontend sync. That mutated
state is the entire algorithmic memory; it is serialized to the learner's state
file (Fly volume in prod).

---

## 6. Selection & difficulty — reading the state

Selection consumes only $\hat{L}$; the graph re-enters solely through gating.

### 6.1 Subtopic mastery

A subtopic $S$ exercises an atom set $\mathcal{A}_S$ (from per-question atom
tags). Its mastery is the mean decayed posterior (`subtopic_mastery`):
$$
M_u(S, t) = \frac{1}{|\mathcal{A}_S|}\sum_{a \in \mathcal{A}_S} \hat{L}_a(t),
\qquad M_u(S,t) = L_0 \text{ if } \mathcal{A}_S = \varnothing.
$$

### 6.2 Weakest-first priority

With per-subtopic effective weight $\omega_S$ (uniform $1/|\mathcal{S}|$ unless
the learner set custom weights), the selection priority is (`select_next_subtopic`):
$$
\boxed{\;\mathrm{priority}_u(S) = \omega_S \cdot \big(1 - M_u(S, t)\big)\;}
$$
The next question is drawn from the **unlocked, not-yet-exhausted** subtopic of
highest priority (ties broken alphabetically for determinism). Because
un-practiced atoms sit at $L_0 = 0.10$, fresh subtopics read as weak and surface
first; decay regresses mastered ones over time, resurfacing stale material
through the *same* priority rule — this is Delta Drills' "spaced repetition"
(decay-driven resurfacing, **not** an SM-2/FSRS scheduler).

### 6.3 Target difficulty

Within the chosen subtopic, difficulty scales affinely with mastery
(`target_difficulty`), an empirical v0 map:
$$
\mathrm{diff}_u(S) = \mathrm{clip}\big(20 + 80\cdot M_u(S,t),\ 10,\ 100\big).
$$
Low mastery $\Rightarrow$ easy items; near-mastery $\Rightarrow$ the hardest.

### 6.4 Area score (learner-facing readout)

Per `topic` $T$, the displayed "area score" is the mean decayed posterior over
practiced atoms in that topic (`area_scores`):
$$
\mathrm{area}_u(T, t) = \frac{1}{|\mathcal{A}_T^{\text{seen}}|}
\sum_{a \in \mathcal{A}_T^{\text{seen}}} \hat{L}_a(t).
$$
This is a pure readout — it drives nothing.

---

## 7. End-to-end summary

```
                    ┌─────────────────────── concept graph G (static) ──────────────────────┐
                    │  atoms A          prereq edges E        encompassing E_enc ⊆ E          │
                    │                   ─ Pre(a) (gating)     ─ Enc(a) (FIRe credit)          │
                    └───────────┬───────────────────────────────────────┬───────────────────┘
                       forward  │ gating                        backward │ credit
                                ▼                                        ▼
   learner picks    select_next_subtopic ──► priority = ω·(1 − M)    apply_attempt(a, correct, c)
   a question  ◄────  filtered by item_is_unlocked (Pre)             ├─ decay → observe → soft-write  L_a
                                ▲                                     └─ if correct: ∀(p,w)∈Enc(a):
                                │                                          itransit(L_p, T·c·w)
                                │                                                  │
                                └──────────── reads ĥL (decayed) ◄─────────────────┘
                                              state: {L_a, τ_a}  (the only thing that changes)
```

**One sentence:** the concept graph is a fixed DAG whose prerequisite edges gate
*what you may practice* and whose encompassing sub-edges route *fractional credit
backward* from advanced atoms to the simpler ones they subsume, while a decaying
per-atom Bayesian Knowledge Tracing posterior is the only mutable state — updated
by direct graded evidence and by single-hop FIRe trickle, and read back to pick
the weakest unlocked atom to drill next.

---

## 8. Calibration caveat

Every numeric constant in §2.1 (and the affine maps in §6) is a **v0 engineering
default**, explicitly *not* literature-derived (see the `bkt_mastery.py` module
docstring and the 2026-05-24 source audit). They should be re-fit once real
per-atom attempt data accumulates. The *structure* above is the contract; the
*numbers* are placeholders.
