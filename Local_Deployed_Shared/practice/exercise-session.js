/* ================================================================
   EXERCISE PRACTICE SESSIONS — "Practice make_rays_1d" on an ARENA page

   Seth, 2026-09-06: "you just click on the problem you want to practice,
   with its own time limit, that you can set manually for the amount of
   answer time and the amount of review time ... when you press the button
   to practice the problem, it asks you: 1. answer time 2. review time 3.
   how many problems ... show the maximum amount of time. 4. when you press
   start, it starts the session, but you can still pause ... next time when
   you press the button it resumes that session."

   WHAT THIS FILE OWNS
     • the button under each exercise heading the notebook renders
       (arena-notebook.js announces `arena-notebook:rendered`; the heading →
       KC map is lessons/arena_exercise_kcs.json)
     • the setup dialog: PRACTICE UNTIL READY (2026-09-23) and the two
       per-problem clocks (the same presets as the idle clock,
       practice/session-clock.js). There is no count any more.
     • starting the block: kc-practice.js `startRoute(kc)` sets up the route
       (practice/ready-route.js → backend/app/ready_route.py), timer.js
       `configure({answer, review, exercise})` installs the clocks, then the
       normal `PracticeSession.start()`
     • resuming: a paused block for this notebook and exercise turns
       the button into "Resume", and pressing it is timer.js's own resume

   🪦 "Related drills" — a read-only list of every drill on the KC and its
   prerequisites, folded under each block — was REMOVED 2026-09-11. Seth: "I
   never told the ai to add that." The block is two buttons, nothing else.

   WHAT IT DOES NOT OWN — and must not grow into
     Grading, mastery, the rung estimate, the clock itself, and the route.
     What comes next inside the block — the exercise, a probe of what it rests
     on, a drill on the weak spot — is the backend's call, read through
     kc-practice.js::nextQuestion; this file only shows the learner the note
     each grade produces (`onNote`).
   ================================================================ */

(() => {
  "use strict";

  const MAP_URL = "lessons/arena_exercise_kcs.json";
  const DEFAULTS = { answer: "5m", review: "2m" };
  /* `### Exercise - implement `make_rays_1d`` and the variants ARENA uses. */
  const HEADING_RE = /Exercise\s*[-–—:]\s*(?:implement|write|fill in|complete)?\s*`?([A-Za-z_]\w*)`?/i;

  let map = null;
  let mapPromise = null;
  let modal = null;
  let pending = null; // the exercise the open dialog is about

  /* 🔴 `PracticeSession` is a top-level `const` in timer.js — script-global,
     NOT a `window` property. `window.PracticeSession` is undefined and a
     lookup through it silently disables every button in this file. */
  const _session = () =>
    (typeof PracticeSession !== "undefined" ? PracticeSession : window.PracticeSession) || null;
  // These belong to exercise blocks; the ordinary question clock has no picker.
  const _options = () => [
    { id: "1m", secs: 60, label: "1:00" },
    { id: "2m", secs: 120, label: "2:00" },
    { id: "5m", secs: 300, label: "5:00" },
    { id: "10m", secs: 600, label: "10:00" },
    { id: "20m", secs: 1200, label: "20:00" },
    { id: "off", secs: null, label: "No limit" },
  ];
  const _option = (id) => _options().find((o) => o.id === id) || null;

  const _loadMap = () => {
    if (map) return Promise.resolve(map);
    if (!mapPromise) {
      mapPromise = fetch(MAP_URL, { cache: "no-cache" })
        .then((r) => (r.ok ? r.json() : {}))
        .catch(() => ({}))
        .then((data) => {
          map = data && typeof data === "object" ? data : {};
          return map;
        });
    }
    return mapPromise;
  };

  /* ── the learner's last setup, per account ────────────────────── */
  const _cfgKey = () => {
    try {
      return `${getPracticeStorageKey()}_exercise_cfg`;
    } catch (_) {
      return "practice_progress_guest_exercise_cfg";
    }
  };
  const _readCfg = () => {
    try {
      const raw = JSON.parse(localStorage.getItem(_cfgKey()) || "null");
      const answer = raw && _option(raw.answer) ? raw.answer : DEFAULTS.answer;
      const review = raw && _option(raw.review) ? raw.review : DEFAULTS.review;
      // Neither `attemptFirst` (the dialog's fourth question until 2026-09-09)
      // nor `quota` (its count until 2026-09-23) is read back: a stored value
      // must not resurrect a control the dialog no longer has.
      return { answer, review };
    } catch (_) {
      return { ...DEFAULTS };
    }
  };
  const _writeCfg = (cfg) => {
    try {
      localStorage.setItem(_cfgKey(), JSON.stringify(cfg));
    } catch (_) {}
  };

  /* ── clock text ────────────────────────────────────────────────── */
  const _mmss = (secs) => {
    const m = Math.floor(secs / 60);
    const s = secs % 60;
    return `${m}:${String(s).padStart(2, "0")}`;
  };

  /* 🪦 THE COUNT, AND THE TWO TIMES OVER IT, WENT 2026-09-23. The dialog
     asked "how many problems" (a slider, 1–40) and printed what that many
     would cost — expected (the learner's median pace) and maximum (every
     clock run out). Seth: "I want one of the options to be to practice
     backward in the reverse order of the algorithm until ready as opposed to
     practicing for a certain amount of time … I'm thinking of just completely
     replacing the other one with this one." A block that ends when the model
     says ready has no count to price, so both readouts went with it. */

  /* ── which paused block, if any, is this exercise's ────────────── */
  const _pausedFor = (exercise) => {
    const s = _session();
    if (!s || !s.hasPausedSession?.()) return null;
    const cfg = s.pausedConfig?.();
    const ex = typeof exercise === "string" ? { kc: exercise } : exercise;
    if (!cfg?.exercise || !ex || cfg.exercise.kc !== ex.kc) return null;
    if (ex.fn && (cfg.exercise.fn !== ex.fn || cfg.exercise.nb !== ex.nb)) return null;
    return { config: cfg, served: s.pausedServed?.() || 0 };
  };

  /* ── the dialog ────────────────────────────────────────────────── */
  const _esc = (v) =>
    String(v == null ? "" : v).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

  const _radios = (name, chosen) =>
    _options()
      .map(
        (o) =>
          `<label class="dd-ex-opt${o.id === chosen ? " is-on" : ""}">` +
          `<input type="radio" name="${name}" value="${_esc(o.id)}"${o.id === chosen ? " checked" : ""}>` +
          `<span>${_esc(o.label)}</span></label>`,
      )
      .join("");

  const _ensureModal = () => {
    if (modal) return modal;
    modal = document.createElement("div");
    modal.className = "dd-ex-modal hidden";
    modal.setAttribute("role", "dialog");
    modal.setAttribute("aria-modal", "true");
    modal.innerHTML =
      '<div class="dd-ex-card">' +
      '<h3 class="dd-ex-title"></h3>' +
      '<p class="dd-ex-sub"></p>' +
      '<div class="dd-ex-resume hidden">' +
      '<p class="dd-ex-resume-text"></p>' +
      '<div class="dd-ex-actions">' +
      '<button type="button" class="primary dd-ex-resume-btn">Resume session</button>' +
      '<button type="button" class="ghost dd-ex-discard-btn">Discard and set up a new one</button>' +
      "</div></div>" +
      /* PRACTICE UNTIL READY — the dialog's one mode (2026-09-23). No count:
         the route (backend/app/ready_route.py) opens on the exercise itself,
         works backward through what it rests on after a miss, drills the weak
         spot, comes back up, and ends when the model says ready. The list
         below says that in the learner's words; the two clocks stay one
         `<details>` down. */
      '<form class="dd-ex-form">' +
      '<div class="dd-ex-field dd-ex-route">' +
      '<ol class="dd-ex-steps">' +
      '<li>First, a version of <b class="dd-ex-fn"></b> itself.</li>' +
      "<li>Miss it, and it works backward through what it rests on to find the weak spot.</li>" +
      "<li>It drills that spot, then brings you back to the problem.</li>" +
      "<li>It stops when you're ready — no fixed number of questions.</li>" +
      "</ol></div>" +
      '<details class="dd-ex-advanced"><summary>Per-problem clocks</summary>' +
      '<fieldset class="dd-ex-field"><legend>Answer time per problem</legend><div class="dd-ex-opts" data-for="answer"></div></fieldset>' +
      '<fieldset class="dd-ex-field"><legend>Review time per problem</legend><div class="dd-ex-opts" data-for="review"></div></fieldset>' +
      "</details>" +
      '<p class="dd-ex-error hidden" role="alert"></p>' +
      '<div class="dd-ex-actions">' +
      '<button type="submit" class="primary dd-ex-start-btn">Start</button>' +
      '<button type="button" class="ghost dd-ex-cancel-btn">Cancel</button>' +
      "</div></form></div>";
    document.body.appendChild(modal);

    const form = modal.querySelector(".dd-ex-form");
    const paint = () => {
      modal.querySelectorAll(".dd-ex-opt").forEach((l) =>
        l.classList.toggle("is-on", l.querySelector("input").checked));
    };
    form.addEventListener("change", paint);
    form.addEventListener("submit", (e) => {
      e.preventDefault();
      _start();
    });
    modal.querySelector(".dd-ex-cancel-btn").onclick = _close;
    modal.querySelector(".dd-ex-resume-btn").onclick = _resume;
    modal.querySelector(".dd-ex-discard-btn").onclick = () => {
      _session()?.discard?.();
      _open(pending);
    };
    modal.addEventListener("click", (e) => {
      if (e.target === modal) _close();
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && !modal.classList.contains("hidden")) _close();
    });
    modal._paint = paint;
    return modal;
  };

  const _formCfg = () => {
    const form = modal.querySelector(".dd-ex-form");
    const pick = (name) => form.querySelector(`input[name="${name}"]:checked`)?.value;
    const answer = _option(pick("answer")) ? pick("answer") : DEFAULTS.answer;
    const review = _option(pick("review")) ? pick("review") : DEFAULTS.review;
    return { answer, review };
  };

  const _open = (ex) => {
    pending = ex;
    const m = _ensureModal();
    m.querySelector(".dd-ex-title").textContent = `Practice ${ex.title} until ready`;
    m.querySelector(".dd-ex-sub").textContent = ex.kcTitle ? `Concept: ${ex.kcTitle}` : "";
    m.querySelector(".dd-ex-fn").textContent = ex.title;
    const paused = _pausedFor(ex);
    const resumeBox = m.querySelector(".dd-ex-resume");
    const form = m.querySelector(".dd-ex-form");
    resumeBox.classList.toggle("hidden", !paused);
    form.classList.toggle("hidden", !!paused);
    if (paused) {
      const c = paused.config;
      const a = c && "answer" in c ? (c.answer === null ? "no limit" : _mmss(c.answer)) : "picker";
      const r = c && "review" in c ? (c.review === null ? "no limit" : _mmss(c.review)) : "picker";
      m.querySelector(".dd-ex-resume-text").textContent =
        `Paused at question ${paused.served}${c.quota ? ` of ${c.quota}` : ""} · ${a} answer · ${r} review. Your code, clock and place in the route are saved.`;
    } else {
      const cfg = _readCfg();
      form.querySelector('[data-for="answer"]').innerHTML = _radios("answer", cfg.answer);
      form.querySelector('[data-for="review"]').innerHTML = _radios("review", cfg.review);
      m.querySelector(".dd-ex-error").classList.add("hidden");
      m._paint();
    }
    m.classList.remove("hidden");
    (paused ? m.querySelector(".dd-ex-resume-btn") : m.querySelector(".dd-ex-start-btn"))?.focus();
  };

  const _close = () => {
    if (modal) modal.classList.add("hidden");
    pending = null;
  };

  const _goPractice = () => {
    if (typeof switchTab === "function") switchTab("practice");
    else if (typeof window.switchTab === "function") window.switchTab("practice");
  };

  const _fail = (text) => {
    const err = modal.querySelector(".dd-ex-error");
    err.textContent = text;
    err.classList.remove("hidden");
  };

  const _start = async () => {
    const ex = pending;
    const s = _session();
    if (!ex) return;
    if (!s) {
      _fail("The practice engine did not load — reload the page.");
      return;
    }
    if (s.isActive?.()) {
      _fail("A block is already running on the Practice tab — pause it first.");
      return;
    }
    const cfg = _formCfg();
    _writeCfg(cfg);
    const btn = modal.querySelector(".dd-ex-start-btn");
    btn.disabled = true;
    /* Anything installed below and not started is torn down on the way out,
       so a failed start leaves no ladder or config for the idle button (or
       the next attempt) to trip over. */
    const rollback = () => {
      window.KcPractice?.stop?.();
      s.configure?.(null);
    };
    try {
      /* A paused block of ANOTHER exercise (or a plain one) is dropped here:
         one snapshot per account, and the learner just chose this one.
         🔴 BEFORE `startRoute`, never after: `discard` calls
         `KcPractice.stop()`, which would clear the ladder just built.
         Codex, 2026-09-06. */
      if (s.hasPausedSession?.()) s.discard?.();
      const kp = window.KcPractice;
      /* 🔴 `startRoute` (2026-09-23) — practice until ready. It replaced
         `startScoped` + a count: the ordinary ladder narrowed to one concept,
         with a miss pulling that concept's prerequisites in front of the
         queue. The route does the same backward move, but asks the model
         which prerequisite is the weak one instead of queueing all of them,
         and stops on "ready" instead of on a count.

         Without a backend there is nothing to route with, and `startRoute`
         leaves the plain scoped ladder in place — the old behaviour, minus
         the count. practice/exercise-planner.js (the greedy budgeted plan)
         is still reached through `KcPractice.startPlanned`, and its
         `masteryFromEstimate` / `READY` are what the timer's drill
         recommendation reads. */
      const exerciseIds = [ex.original, ...(ex.variants || [])].filter(Number.isFinite);
      const ok = await kp?.startRoute?.(ex.kc, { exerciseIds, title: ex.title });
      if (!ok) {
        rollback();
        _fail("No drills are attached to this exercise's concept yet.");
        return;
      }
      s.configure({
        answer: _option(cfg.answer).secs,
        review: _option(cfg.review).secs,
        exercise: { kc: ex.kc, fn: ex.fn, title: ex.title, nb: ex.nb, cell: ex.cell, mode: "ready" },
      });
      _close();
      document.getElementById("page-practice")?.classList.add("dd-exercise-session");
      /* 🔴 START BEFORE SWITCHING TABS. app.js::switchTab("practice") re-fetches
         a question unless a session is active or paused, and `start()` makes
         the session active synchronously — the other order served (and
         counted) two questions for one. */
      s.start();
      _goPractice();
    } catch (err) {
      console.warn("[exercise-session] could not start:", err);
      rollback();
      _fail("Could not start — " + (err?.message || err));
    } finally {
      btn.disabled = false;
    }
  };

  const _resume = () => {
    const s = _session();
    if (!s) return;
    _close();
    _goPractice();
    document.getElementById("page-practice")?.classList.add("dd-exercise-session");
    s.resume();
  };

  /* ── the button on the notebook page ───────────────────────────── */
  /* Three spellings of "this cell is an exercise", in the order ARENA uses
     them (0.1 first, then the two 0.0 shapes):
       • `### Exercise - implement `make_rays_1d``  → key "make_rays_1d"
       • `#### (1) Column-stacking` (0.0 image ops, no def) → key "(1)"
       • a CODE cell holding `def rearrange_1(` / the five `def einsum_*(`
         (0.0 sections A–I and the einsum block) → one key per def
       • a CODE cell holding `class ReLU(nn.Module):` (0.2) → one key per class
     A markdown cell yields at most one key; a code cell may yield several
     — the 0.0 einsum exercises share ONE cell — and each gets its own block.
     🔴 A CELL ID OUTRANKS EVERY NAME (Z, 2026-09-11): 0.2 has two different
     exercises both called `train`, and a stride-size answer cell with no def
     at all. A map entry keyed `exercise:<cell id>` (the compiled cell's
     `data-cell-id`, e.g. `exercise:0-2-c047`) claims that cell outright and
     its name keys are not consulted, so one cell never grows two blocks. */
  const CELL_KEY = (cell) => (cell.dataset.cellId ? `exercise:${cell.dataset.cellId}` : null);
  const TAG_RE = /^\((\w{1,3})\)\s/;
  /* The run button (▶) is glued to the first line of a code cell's
     textContent, so a def may follow it instead of a newline. `class` too
     (Z, 2026-09-11): 0.2's exercises are `class ReLU(nn.Module)`, `class
     Linear`, … — one key per TOP-LEVEL def or class in the cell.
     🔴 Column 0 only after a newline: `\s*` there also matched the indented
     `def forward(` inside every class, so one cell of four modules yielded
     four `forward` keys (codex, 2026-09-11). Whitespace is allowed only after
     the glued ▶. */
  const DEF_RE = /(?:^|\n|▶\s*)(?:def|class)\s+([A-Za-z_]\w*)\s*[(:]/g;
  const _srcOf = (cell) =>
    cell.querySelector("textarea")?.value ?? cell.querySelector("pre, code")?.textContent ?? cell.textContent ?? "";
  const _cellKeys = (cell) => {
    if (cell.classList.contains("nbv-md")) {
      const h = cell.querySelector("h1, h2, h3, h4");
      const text = (h ? h.textContent : cell.textContent || "").trim();
      const m = HEADING_RE.exec(text);
      if (m) return [m[1]];
      const tag = TAG_RE.exec(text);
      return tag ? [`(${tag[1]})`] : [];
    }
    if (cell.classList.contains("nbv-code")) {
      return Array.from(_srcOf(cell).matchAll(DEF_RE), (m) => m[1]);
    }
    return [];
  };

  /* ── WHERE THE TWO BUTTONS GO ───────────────────────────────────
     🔴 ABOVE THE CELL THE LEARNER TYPES IN, NOT UNDER WHATEVER CELL
     HAPPENED TO CARRY THE EXERCISE'S NAME. Seth, 2026-09-22: "it should show
     up below the problem statement and question, rather than below any of
     your code cells ... sometimes it's just not consistent."

     The inconsistency was structural. A block was inserted `afterend` of its
     ANCHOR, and the anchor is whichever cell the name was found in — which
     is a different kind of cell from one notebook to the next:

       0.1 / 0.2  the name is in the heading (`### Exercise - implement
                  make_rays_1d`), so the buttons landed between the heading
                  and the rest of the prose that explains the problem — above
                  half the question.
       0.0 A–I    the name is only in the CODE cell (`def rearrange_1(`), so
                  the buttons landed UNDER the answer cell, below the box the
                  learner is about to type in.
       0.0 (1)–(8) the name is a `(N)` tag on the heading, and the heading is
                  followed by `display_soln_array_as_img(N)` — the picture the
                  learner has to reproduce. That cell IS the question (Seth:
                  "it essentially has a code block that has the solution
                  image, and that code block would go above the buttons"), so
                  the buttons belong after it, not before it.

     One rule covers all three: walk forward from the anchor over everything
     that is still the PROBLEM — more prose, the target-image cell — and stop
     immediately above the first cell the learner answers in. The question
     ends where their cursor starts.

     A question written INSIDE the answer cell as a comment (`# Your code here
     - define arr1`) is the one case where the buttons sit above part of the
     question, and Seth called that out as fine: "that's an edge case where
     it's okay that the button is above or whatever."

     When nothing below the anchor looks like an answer cell the old
     behaviour stands — the buttons go straight under the anchor, which for a
     prose anchor is still under the statement. */
  /* What a cell the learner is meant to fill in says about itself. ARENA
     marks them three ways and uses all three within one chapter. */
  const ANSWER_RE = /your code here|raise\s+NotImplementedError|#\s*(?:TODO|EXERCISE)\b/i;
  /* 🔴 A FUSE, NOT THE RULE. This was 6 siblings, and codex (2026-09-22) was
     right that a budget is the wrong bound: it counts the disclosures the walk
     SKIPS and the blocks the walk itself injects, so a statement with four
     hints and two blocks above it exhausts the budget and falls back to the
     anchor — reinstating, for exactly the longest questions, the bug this rule
     exists to remove. What actually ends a question is a BOUNDARY: a heading,
     the Solution, or the next exercise's own cell. Those are what the walk
     stops on. This number only keeps a malformed DOM from walking the whole
     notebook, and nothing should ever reach it. */
  const PLACE_FUSE = 60;

  /* A disclosure BEFORE the answer is part of the offer — ARENA's "Help - …",
     "Hint 2", "Question - why …", "Aside - …" are all things the learner reads
     while deciding how to attack the problem, and 0.2 puts several of them
     between the exercise heading and the stub. The one that is NOT is the
     solution, which upstream titles exactly "Solution"; reaching it means this
     exercise has no answer cell of its own and the walk has to stop rather
     than run into the next section. */
  const SOLUTION_RE = /^\s*solutions?\b/i;

  const _isAnswerCell = (cell, table) => {
    if (!cell.classList?.contains("nbv-code")) return false;
    // A `%pip install` setup cell is nobody's answer.
    if (cell.dataset.role === "magic") return false;
    if (ANSWER_RE.test(_srcOf(cell))) return true;
    // 0.0's `def rearrange_1(` / 0.2's `class ReLU(`: the stub IS the answer.
    return _cellKeys(cell).some((key) => table[key]?.kc) || !!table[CELL_KEY(cell) || ""]?.kc;
  };

  /* The tracked exercises a cell is the stub FOR — empty for a cell that only
     looks like an answer (`# Your code here`, a bare `raise`). */
  const _ownersOf = (cell, table) =>
    _cellKeys(cell).concat(CELL_KEY(cell) || []).filter((key) => table[key]?.kc);

  /** Where this exercise's block belongs, as an `insertAdjacentElement` pair.
      `wanted` = the exercise keys this anchor is about to mint blocks for. */
  const _placeFor = (anchor, table, wanted) => {
    if (_isAnswerCell(anchor, table)) return { ref: anchor, where: "beforebegin" };
    let node = anchor.nextElementSibling;
    for (let i = 0; node && i < PLACE_FUSE; i += 1) {
      // Our own injected blocks are not content and cost the walk nothing.
      if (node.classList.contains("dd-ex-block")) { node = node.nextElementSibling; continue; }
      if (node.dataset.role === "details") {
        if (SOLUTION_RE.test(node.querySelector("summary")?.textContent || "")) break;
        node = node.nextElementSibling;
        continue;
      }
      /* 🔴 ANOTHER EXERCISE'S CELL IS A BOUNDARY, NOT PART OF THIS QUESTION.
         Every cell says which tracked exercises it belongs to — a stub by its
         `def`/`class`, a heading or `(N)` tag by its name — so a cell naming
         only OTHER exercises is where this question ended. Stop there rather
         than hoist the block above someone else's box or read their statement
         as more of this one. A cell that claims nobody (`# Your code here -
         define arr3`, a paragraph of prose) is still fair game, which is what
         keeps 0.0's untitled answer cells and target images working. Found by
         codex, 2026-09-22. */
      const owners = _ownersOf(node, table);
      const mine = !owners.length || !wanted?.length || owners.some((key) => wanted.includes(key));
      if (_isAnswerCell(node, table)) {
        if (!mine) break;
        return { ref: node, where: "beforebegin" };
      }
      if (!mine) break;
      // A new heading is a new section — stop rather than jump the boundary.
      if (node.classList.contains("nbv-md") && node.querySelector("h1, h2, h3, h4")) break;
      node = node.nextElementSibling;
    }
    return { ref: anchor, where: "afterend" };
  };

  const _syncButton = (block) => {
    const btn = block.querySelector(".dd-ex-btn");
    const timerBtn = block.querySelector(".dd-ex-timer-btn");
    const ex = block._exercise;
    const paused = _pausedFor(ex);
    /* Two buttons of EQUAL weight (Seth, 2026-09-11: "give equal importance to
       the drill or start problem option"). Neither is `.primary`; both are the
       same box (styles/practice/exercise-session.css). */
    btn.textContent = paused
      ? `Resume practice · question ${paused.served}${paused.config.quota ? ` of ${paused.config.quota}` : ""}`
      : "Practice until ready";
    /* The button says the budget out loud, because the budget is the notebook's
       and the learner has no other way to know what they are agreeing to.
       ExerciseTimer reads it off the exercise's own "You should spend up to …"
       line; when the notebook does not say, it says its fallback rather than
       nothing. A clock the learner paused on this exercise is offered back
       with what is left on it. */
    const T = window.ExerciseTimer;
    if (timerBtn) {
      const left = T?.pausedFor?.(ex);
      if (left) {
        timerBtn.textContent = `Resume timer · ${Math.floor(left / 60)}:${String(left % 60).padStart(2, "0")} left`;
      } else {
        const secs = T ? T.budgetSecs(block) : null;
        const mins = Number.isFinite(secs) ? Math.round(secs / 60) : null;
        timerBtn.textContent = mins ? `Start ${mins} minute question timer` : "Start question timer";
      }
      // Another exercise is on the clock. Starting a second one would silently
      // abandon the first, so say why the button is off rather than doing it.
      const busy = !!T?.activeExercise?.() && !T.isRunning(ex);
      timerBtn.disabled = !T || busy;
      timerBtn.title = busy
        ? `${T.activeExercise().title || T.activeExercise().fn} is on the clock — pause or finish it first.`
        : "";
    }
  };

  /* Every block's buttons describe one shared clock, so all of them change when
     it starts or stops — not just the one that was pressed. */
  document.addEventListener("dd-exercise-timer:change", () => {
    document.querySelectorAll(".dd-ex-block").forEach(_syncButton);
  });

  const _decorate = async (nbId, host) => {
    const m = await _loadMap();
    const table = { ...(m["*"] || {}), ...(m[nbId] || {}) };
    if (!Object.keys(table).length) return;
    const seen = new Set(Array.from(host.querySelectorAll(".dd-ex-block"),
      (block) => block._exercise?.fn).filter(Boolean));
    const titles = [];
    host.querySelectorAll(".nbv-cell.nbv-md, .nbv-cell.nbv-code").forEach((cell) => {
      const byId = CELL_KEY(cell);
      const keys = byId && table[byId]?.kc ? [byId] : _cellKeys(cell);
      const wanted = keys.filter((fn) => table[fn]?.kc && !seen.has(fn));
      if (!wanted.length) return;
      /* 🔴 THE ORDER DEPENDS ON THE DIRECTION. Several defs share one cell
         (0.0's five `einsum_*`), and every block is inserted against the SAME
         reference node — so `afterend` has to run backwards to read top-down
         and `beforebegin` has to run forwards. Reversing unconditionally, as
         this did when there was only one direction, stacks the einsum blocks
         bottom-up above their cell. */
      const at = _placeFor(cell, table, wanted);
      const ordered = at.where === "beforebegin" ? wanted : wanted.slice().reverse();
      ordered.forEach((fn) => {
      seen.add(fn);
      const entry = table[fn];
      const ex = {
        kc: entry.kc, fn, title: entry.title || fn, kcTitle: null, nb: nbId, cell: cell.dataset.cellId,
        // The attempt pool: variants of THIS exercise at its own difficulty
        // (practice/exercise-planner.js). Empty ⇒ the plain scoped ladder.
        variants: Array.isArray(entry.variants) ? entry.variants.filter(Number.isFinite) : [],
        // The exercise's OWN bank question, so a timed attempt that runs out
        // has something honest to be recorded against (practice/exercise-timer.js).
        original: Number.isFinite(entry.original) ? entry.original : null,
      };
      const block = document.createElement("div");
      block.className = "dd-ex-block";
      block._exercise = ex;
      block._sourceCell = cell;
      /* TWO BUTTONS, ONE JOB EACH (Seth, 2026-09-09). This used to be a single
         "Practice <fn>" button that opened a dialog and took the learner to the
         Practice tab — so the only thing you could do with a notebook exercise
         was leave the notebook. The problem is on the page; the first button
         times an attempt at it where it is, and the second is the trip to the
         drills, taken on purpose.

         While the clock runs, practice/exercise-timer.js puts `is-live` on
         this block and its own `.dd-ex-live` row after `.dd-ex-row`; the CSS
         swaps the two. The verdict slot is last so it reads under either. */
      block.innerHTML =
        '<div class="dd-ex-row">' +
        '<button type="button" class="dd-ex-timer-btn"></button>' +
        '<button type="button" class="dd-ex-btn"></button></div>' +
        '<div class="dd-ex-verdict hidden"></div>';
      block.querySelector(".dd-ex-btn").onclick = () => _open(ex);
      block.querySelector(".dd-ex-timer-btn").onclick = () => {
        const T = window.ExerciseTimer;
        if (!T || T.isRunning(ex)) return;
        if (T.pausedFor(ex)) T.resume(ex, block);
        else T.start(ex, block);
      };
      at.ref.insertAdjacentElement(at.where, block);
      _syncButton(block);
      // A clock the learner started before a reload comes back here, because
      // this is the first moment the exercise and its block exist together.
      window.ExerciseTimer?.restore?.(ex, block);
      const named = window.LessonGate?.getKpEntry?.(ex.kc).then((entry) => {
        if (entry && entry.kp && entry.kp.title) ex.kcTitle = entry.kp.title;
      }).catch(() => {});
      titles.push(named);
      });
    });
    /* For practice/arena-notebook-focus.js, which tints each section's heading
       and the topbar pill with the concept's readiness and needs the concept's
       NAME to do it — so this waits for the title lookups above rather than
       announcing blocks whose concept is still an id. The blocks themselves are
       usable before this fires. */
    await Promise.all(titles);
    document.dispatchEvent(new CustomEvent("dd-exercise-blocks:decorated", { detail: { id: nbId, host } }));
  };

  /* ── hooks timer.js calls ──────────────────────────────────────── */
  const _summaryEl = () => document.getElementById("session-summary");

  const _backLink = (config) => {
    const ex = config && config.exercise;
    if (!ex || !ex.nb) return "";
    const href = `?arena=${encodeURIComponent(ex.nb)}${ex.cell ? `#arena-${encodeURIComponent(ex.cell)}` : ""}`;
    return ` <a class="dd-ex-back" href="${_esc(href)}">Back to ${_esc(ex.title || "the notebook")} ↩</a>`;
  };

  const onEnd = (reason, served, config, outcome) => {
    document.getElementById("page-practice")?.classList.remove("dd-exercise-session");
    document.querySelectorAll(".dd-ex-block").forEach(_syncButton);
    const el = _summaryEl();
    if (!el || !config?.exercise) return;
    if (reason === "complete" && outcome?.text) {
      el.innerHTML = `<b>${_esc(config.exercise.title || "Exercise")}</b> — ${_esc(outcome.text)}` + _backLink(config);
    } else if (reason === "complete" && outcome?.solved) {
      el.innerHTML =
        `<b>${_esc(config.exercise.title || "Exercise")}</b> — solved on attempt ${outcome.attempts}, ${served} of ${config.quota} questions used. Recorded answers are kept.` +
        _backLink(config);
    } else if (reason === "complete") {
      const tried = outcome && outcome.attempts ? ` ${outcome.attempts} attempt${outcome.attempts === 1 ? "" : "s"} at a variant, none solved yet.` : "";
      el.innerHTML =
        `<b>${_esc(config.exercise.title || "Exercise")}</b> — done, ${served} of ${config.quota} questions.${tried} Recorded answers are kept.` +
        _backLink(config);
    } else if (reason === "discarded") {
      el.textContent = `Saved ${config.exercise.title || "exercise"} session discarded.`;
    } else {
      el.innerHTML = _esc(el.textContent) + _backLink(config);
    }
    el.classList.remove("hidden");
  };

  const onResume = (config) => {
    if (config?.exercise) document.getElementById("page-practice")?.classList.add("dd-exercise-session");
    document.querySelectorAll(".dd-ex-block").forEach(_syncButton);
  };

  /* A one-line note from the ladder after a grade ("3 prerequisite drills
     queued", "Miss — re-planning; torch.slice-assignment looks weakest"),
     shown on the phase label for a moment. */
  const onNote = (note) => {
    const el = document.getElementById("session-phase");
    if (!el || !note) return;
    const label = el.textContent;
    el.textContent = `Reviewing · ${note}`;
    setTimeout(() => {
      if (el.textContent.startsWith("Reviewing ·")) el.textContent = label;
    }, 4000);
  };

  document.addEventListener("arena-notebook:rendered", (e) => {
    const { id, host } = e.detail || {};
    if (id && host) _decorate(id, host).catch((err) => console.warn("[exercise-session]", err));
  });
  /* The notebook may have rendered before this file evaluated (deep link). */
  const late = document.querySelector("#page-arena-notebook .nbv-cells");
  if (late && window.ArenaNotebook?.current?.id) {
    _decorate(window.ArenaNotebook.current.id, late.parentElement).catch(() => {});
  }

  window.ExerciseSession = { open: _open, onEnd, onResume, onNote, pausedFor: _pausedFor };
})();
