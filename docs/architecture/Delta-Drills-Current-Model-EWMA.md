# Delta Drills — Current Mastery & Selection Model (EWMA stack with prerequisite gating)

> Snapshot of the live engine as of 2026-05-26. Combines the original
> per-subtopic Exponentially Weighted Moving Average (EWMA) update rule
> with the prerequisite-gating layer that has been added since the
> original math document was written. The model is implemented across
> `backend/app/adaptive.py`, `backend/app/prioritization.py`,
> `backend/app/practice/arena_rating_router.py`, and the frontend
> `practice/arena-unlock.js` + `practice/drills-catalog.js`.

---

## 0. Notation

Let

- $\mathcal{S}$ — the finite set of subtopics covered by the question bank (e.g. *Einops: Rearrange*, *CNN: Conv output shape*).
- $\mathcal{T}$ — the finite set of topics. Every subtopic $s$ belongs to exactly one topic $T(s)\in\mathcal{T}$.
- $\mathcal{D}$ — the catalog of procedural drills (Colab notebooks). Each drill $d\in\mathcal{D}$ carries
  - $\mathrm{subs}(d) \subseteq \mathcal{S}$ — the subtopics it exercises (one for single-atom drills, multiple for composites),
  - a flag $\mathrm{comp}(d)\in\{0,1\}$ — whether $d$ is a composite drill,
  - an index $i(d)\in\mathbb{N}$ giving its position in the catalog list.
- $\mathcal{E}$ — the set of ARENA exercises. Each $e\in\mathcal{E}$ carries problem–concept links $L_e=\{(s_i,\omega_i)\}_{i=1}^{m_e}$ where $\omega_i\ge 0$ are the per-link skill weights.

For each subtopic $s$ and student, the per-subtopic adaptive state is the triple

$$
\sigma_s \;=\; \bigl(n_s,\; b_s,\; p_s\bigr) \;\in\; \mathbb{N}\times[0,100]\times[0,1],
$$

where $n_s$ is the number of recorded attempts, $b_s$ is the running baseline score, and $p_s$ is the running correctness rate. A timestamp $\tau_s$ of the last update is also kept for time decay (see §1.6).

---

## 1. Per-subtopic adaptive engine

This is the layer that turns each individual attempt into an update of $\sigma_s$ and a target difficulty for the next question on that subtopic.

### 1.1 Cold start ($n_s \le 3$)

The first three attempts on each subtopic are scripted to sample three difficulty bands:

$$
\mathrm{target}(n_s) \;=\;
\begin{cases}
25 & n_s = 0 \\
50 & n_s = 1 \\
75 & n_s = 2
\end{cases}
\quad\quad (\text{cold-start ladder})
$$

The state $\sigma_s$ still accumulates during cold start (using the same EWMA blends as below), but the target difficulty is fixed by the ladder rather than computed from $\sigma_s$.

### 1.2 Indicator and feedback recency factor

After every attempt, the student self-reports a grade $\mathrm{grade}(n) \in [0,100]$ and selects a feedback level. The grade-binarizing indicator is

$$
\mathbf{1}\bigl(\mathrm{grade}(n)\bigr) \;=\;
\begin{cases}
1 & \mathrm{grade}(n) > 85 \\
0 & \mathrm{grade}(n) \le 85
\end{cases}
$$

The feedback level is mapped to a per-attempt recency factor $\alpha(n)$:

$$
\alpha(n) \;=\;
\begin{cases}
0.30 & \text{"Not much"} \\
0.60 & \text{"Somewhat"} \\
0.85 & \text{"A lot"}
\end{cases}
$$

A higher $\alpha(n)$ means the EWMA absorbs the current attempt more aggressively.

### 1.3 Correctness rate update

The running correctness rate evolves as a fixed-recency EWMA on the binarized indicator:

$$
p(n) \;=\; \alpha(n)\cdot \mathbf{1}\bigl(\mathrm{grade}(n)\bigr) \;+\; \bigl(1-\alpha(n)\bigr)\, p(n-1),
\quad p(0)=0.5.
$$

### 1.4 Score and baseline

Each attempt has a difficulty $\mathrm{diff}(n)\in[0,100]$ assigned by the recommender. The raw score is

$$
\mathrm{score}(n) \;=\; \frac{\mathrm{grade}(n)\cdot\mathrm{diff}(n)}{100}.
$$

The baseline (a running estimate of "how high a difficulty the student is consistently scoring at") is then

$$
b(n) \;=\; \alpha(n)\cdot\mathrm{score}(n) \;+\; \bigl(1-\alpha(n)\bigr)\, b(n-1),
\quad b(0)=0.
$$

### 1.5 Difficulty multiplier and target difficulty

Define the piecewise multiplier

$$
\mu\bigl(p\bigr) \;=\;
\begin{cases}
0.5 + 0.5\left(\dfrac{p}{0.85}\right)^{1.8} & p \le 0.85 \\[1em]
\min\!\left(2.5,\; 1 + \left(\dfrac{p-0.85}{0.15}\right)^{2.5}\right) & p > 0.85
\end{cases}
$$

After the third attempt ($n_s > 3$), the target difficulty for the next problem on $s$ is

$$
\mathrm{target}(n) \;=\; \mathrm{clip}_{[0,100]}\Bigl(\, b(n)\cdot\mu\bigl(p(n)\bigr)\Bigr).
$$

### 1.6 Time decay between attempts

Let $\Delta t$ be the number of days elapsed since the last update timestamp $\tau_s$, and let $\tau_{1/2}=14$ days be the half-life. Just before applying the EWMA update of §1.3–1.5 on the next attempt, both $b$ and $p$ are shrunk toward their priors $B_0=0$ and $P_0=0.5$:

$$
b \;\leftarrow\; B_0 + \bigl(b - B_0\bigr)\cdot 2^{-\Delta t/\tau_{1/2}},
$$

$$
p \;\leftarrow\; P_0 + \bigl(p - P_0\bigr)\cdot 2^{-\Delta t/\tau_{1/2}}.
$$

This makes prolonged inactivity regress the mastery estimate back toward "no information" rather than letting it stay at the last observed value indefinitely.

---

## 2. Subtopic selection (next-question routing)

This layer decides *which subtopic to pull the next question from* whenever the student requests a new question.

### 2.1 User-defined effective weights

Through the Statistics page, the student assigns a topic percentage $P_T\in[0,\infty)$ to each topic $T\in\mathcal{T}$ and, within each topic, a subtopic share $P_s\in[0,\infty)$ to each $s$ with $T(s)=T$. The effective weight of subtopic $s$ is

$$
w_s \;=\; \frac{P_{T(s)}}{100}\cdot\frac{P_s}{100}.
$$

Weights need not sum to 1; only their relative magnitudes affect selection. If no custom weights are set, the system falls back to uniform weights

$$
w_s \;=\; \frac{1}{\lvert\mathcal{S}\rvert}.
$$

### 2.2 Learning-rate estimate via EWMA on baseline deltas

Let the recorded baseline sequence for subtopic $s$ be $b_{s,1}, b_{s,2}, \dots, b_{s,n_s}$. Define the step deltas

$$
\delta_{s,t} \;=\; b_{s,t} - b_{s,t-1}, \quad t=2,\dots,n_s.
$$

The learning-rate estimate $\hat{r}_s$ is an EWMA over these deltas with smoothing constant

$$
\alpha_{\mathrm{lr}} \;=\; 1 - e^{-\lambda}, \qquad \lambda = 0.3,
$$

(note: $\alpha_{\mathrm{lr}}$ here is the *learning-rate EWMA smoothing*, distinct from $\alpha(n)$ in §1.2). For $n_s < 2$ there are no deltas yet, so we use a cold-start prior:

$$
\hat{r}_s \;=\; 0.5 \quad\text{when}\quad n_s < 2.
$$

Otherwise recursively define

$$
\begin{aligned}
s_1 &= \delta_{s,2} \\
s_k &= \alpha_{\mathrm{lr}}\,\delta_{s,k+1} + (1-\alpha_{\mathrm{lr}})\,s_{k-1}, \quad k=2,\dots,n_s-1
\end{aligned}
$$

and set

$$
\hat{r}_s \;=\; s_{n_s-1}.
$$

### 2.3 Gradient, masking, and selection

Let $\mathcal{S}_{\mathrm{open}}\subseteq\mathcal{S}$ be the set of subtopics with at least one unserved question. The subtopic priority (gradient) is

$$
g_s \;=\; w_s \cdot \hat{r}_s.
$$

The next subtopic is chosen by

$$
s^{*} \;=\; \arg\max_{s\,\in\,\mathcal{S}_{\mathrm{open}}}\; g_s.
$$

If $\mathcal{S}_{\mathrm{open}}=\emptyset$ (i.e. every question has been served), the served-set is reset and the selection is recomputed. Topics are not selected directly: the induced topic is simply

$$
T^{*} \;=\; T(s^{*}).
$$

The actual question is then drawn from the bank for subtopic $s^{*}$ at a difficulty close to $\mathrm{target}(n_{s^{*}})$ from §1.5.

---

## 3. Prerequisite-gated procedural-drill recommender

This is the layer that decides, *after* a question/feedback cycle finishes, whether to fire a procedural Colab drill — and if so, which one.

### 3.1 Per-subtopic mastery readout

The frontend's gating logic reads a scalar mastery score per subtopic from the backend state. With $p_s$ the running correctness rate of §1.3, the readout is

$$
\rho_s \;=\; 100\cdot p_s \;\in\; [0,100].
$$

(For sake of completeness: if $p_s$ is missing but a positive baseline exists, the readout falls back to $b_s$. If neither is set, $\rho_s$ is undefined and treated as "no information.")

### 3.2 Drill unlock predicate

Each drill $d\in\mathcal{D}$ unlocks once *every* one of its constituent subtopics has crossed the unlock floor $\theta_{\mathrm{unlock}}=50$:

$$
U(d) \;\equiv\; \bigl(\forall s\in\mathrm{subs}(d):\; \rho_s\;\text{is defined and}\;\rho_s \ge \theta_{\mathrm{unlock}}\bigr).
$$

For single-atom drills this is one inequality; for composites it is a conjunction over 2–3 atoms.

### 3.3 Composite promote predicate

To prevent composite drills from never firing (they sit at the back of the catalog list, behind hundreds of single-atom drills), composites are *promoted* ahead of further single-atom drills once their atoms are strongly mastered. Let $\theta_{\mathrm{promote}}=70 > \theta_{\mathrm{unlock}}$. Define

$$
P(d) \;\equiv\; \mathrm{comp}(d) \;\wedge\; \bigl(\forall s\in\mathrm{subs}(d):\; \rho_s\;\text{is defined and}\;\rho_s \ge \theta_{\mathrm{promote}}\bigr).
$$

### 3.4 Two-pass selection

Let $\Sigma\subseteq\mathcal{D}$ be the set of drills already shown to the student (tracked in `localStorage`). The next drill is chosen in two passes:

**Pass 1 (composite promotion).**
$$
d^{*}_{1} \;=\; \arg\min_{d}\;\bigl\{\, i(d)\;:\; d\notin\Sigma\;\wedge\; P(d)\,\bigr\}.
$$

**Pass 2 (in-order unlocked fallback).** If Pass 1 returns no drill,
$$
d^{*}_{2} \;=\; \arg\min_{d}\;\bigl\{\, i(d)\;:\; d\notin\Sigma\;\wedge\; U(d)\,\bigr\}.
$$

The fired drill is $d^{*} = d^{*}_{1}$ if defined, else $d^{*}_{2}$; if both are empty the recommender returns "no drill available."

### 3.5 ARENA exercise readiness

ARENA exercises live on a separate surface (the Courses tab). Each exercise $e\in\mathcal{E}$ has skill-weight links $L_e=\{(s_i,\omega_i)\}$, and its readiness score is the weight-normalized average of constituent atom masteries:

$$
R(e) \;=\;
\begin{cases}
\dfrac{\displaystyle\sum_{(s_i,\omega_i)\in L_e} \omega_i\cdot\min(\rho_{s_i},100)}
       {\displaystyle\sum_{(s_i,\omega_i)\in L_e} \omega_i}
& \text{if } L_e\neq\emptyset \\[1em]
R_{0}(e) & \text{otherwise (static fallback)}
\end{cases}
$$

$R(e)$ governs the readiness bar shown on ARENA cards and is consumed by `targeted-practice` for ordering search results.

---

## 4. Composite drill rating (multi-subtopic credit)

When the student rates a procedural drill, the system fires *one update per constituent subtopic*. This is the "even-skill averaging" credit-assignment scheme.

### 4.1 Self-rating mapping

Drill self-rating $r$ is one of four buttons:

| Button | Correct $c(r)$ | Feedback $f(r)$ |
|---|---|---|
| ✓ Solved in target time, no help | $1$ | "A lot" |
| ✓ Solved in target time, with a hint | $1$ | "Somewhat" |
| ✓ Solved correctly, over target time | $1$ | "Not much" |
| ✗ Looked up the solution | $0$ | "A lot" |

The feedback level $f(r)$ maps to $\alpha(n)\in\{0.3,0.6,0.85\}$ exactly as in §1.2.

### 4.2 Parallel apply-feedback

For composite drill $d$ with $\mathrm{subs}(d)=\{s_1,\dots,s_k\}$, the rating event triggers $k$ synthetic attempts in parallel. For each $s_i$:

$$
\mathrm{diff}_{s_i} \;=\; \bigl\lfloor\mathrm{target}(n_{s_i})\bigr\rfloor
\quad\text{(fallback to } 50 \text{ if undefined)},
$$

$$
\mathrm{grade}_{s_i} \;=\; 100\cdot c(r),
$$

$$
\mathrm{score}_{s_i} \;=\; \frac{\mathrm{grade}_{s_i}\cdot\mathrm{diff}_{s_i}}{100}.
$$

Then the §1.3–1.5 updates fire independently on each $s_i$ with the same $\alpha(n)=\alpha(f(r))$. After the parallel updates,

$$
n_{s_i} \;\leftarrow\; n_{s_i} + 1 \quad\forall\, i,
$$

so each constituent atom advances one cold-start step. Each subtopic accumulates its own evidence; there is no shared per-composite latent state.

This is mathematically equivalent to "equal-credit" multi-tag PFA (cf. Maier 2021): each tagged subtopic receives the same correctness signal weighted by its own current target difficulty.

---

## 5. Constants summary

| Symbol | Value | Source / meaning |
|---|---|---|
| $\alpha(n)$ ("Not much"/"Somewhat"/"A lot") | $0.30$ / $0.60$ / $0.85$ | Per-attempt EWMA recency factor (§1.2) |
| $\alpha_{\mathrm{lr}}$ | $1 - e^{-0.3} \approx 0.259$ | Learning-rate EWMA smoothing (§2.2) |
| $\tau_{1/2}$ | $14$ days | Mastery half-life (§1.6) |
| $B_0,\,P_0$ | $0,\,0.5$ | Priors for baseline and correctness rate |
| Cold-start ladder | $\{25, 50, 75\}$ | Target difficulties for $n_s\in\{0,1,2\}$ |
| Grade threshold for "correct" | $85$ | Indicator $\mathbf{1}(\mathrm{grade}>85)$ |
| $\theta_{\mathrm{unlock}}$ | $50$ | Drill unlock floor (§3.2) |
| $\theta_{\mathrm{promote}}$ | $70$ | Composite promotion floor (§3.3) |
| $\mu(p)$ cap | $2.5$ | Maximum difficulty multiplier (§1.5) |
| $\mu(p)$ knee | $p = 0.85$ | Piecewise breakpoint (§1.5) |

---

## 6. Composition diagram

The four layers compose top-to-bottom on every interaction:

1. **Subtopic selection (§2)** chooses $s^{*}$ using $g_s = w_s\hat{r}_s$.
2. **Per-subtopic engine (§1)** serves a question at difficulty $\approx\mathrm{target}(n_{s^{*}})$, then on feedback updates $\sigma_{s^{*}}=(n,b,p)$ with EWMA + time decay.
3. **Prereq-gated drill recommender (§3)** checks every drill against $\rho_s$ readouts; fires a composite if any is promotable, otherwise the next in-order unlocked single-atom drill.
4. **Composite drill rating (§4)** propagates one rating into $k$ parallel per-subtopic updates, each running the same §1 machinery.

There is no global "student ability" scalar; mastery lives per subtopic, and harder content unlocks through prerequisite conjunctions in §3, not through a global difficulty axis.

---

*Engine code: `This-Directory-Only/backend/app/{adaptive.py, prioritization.py, practice/arena_rating_router.py}` (Python) and `Local_Deployed_Shared/{practice/arena-unlock.js, practice/drills-catalog.js, stats/predicted-prereqs-temp.js}` (frontend).*
