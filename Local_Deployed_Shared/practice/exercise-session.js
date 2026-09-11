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
     • the setup dialog: two pickers (the same presets as the idle clock,
       practice/session-clock.js), a count, and the maximum-time line
     • starting the block: kc-practice.js `startScoped(kc)` builds the
       ladder, timer.js `configure({answer, review, quota, exercise})`
       installs the numbers, then the normal `PracticeSession.start()`
     • resuming: a paused block for this notebook and exercise turns
       the button into "Resume", and pressing it is timer.js's own resume

   🪦 "Related drills" — a read-only list of every drill on the KC and its
   prerequisites, folded under each block — was REMOVED 2026-09-11. Seth: "I
   never told the ai to add that." The block is two buttons, nothing else.

   WHAT IT DOES NOT OWN — and must not grow into
     Grading, mastery, the rung estimate, the clock itself. A miss inside the
     block pulls the concept's prerequisites in front of the queue, but that
     is kc-practice.js::onMiss called from timer.js::recordReviewResult; this
     file only shows the learner that it happened (`onPrereqsQueued`).
   ================================================================ */

(() => {
  "use strict";

  const MAP_URL = "lessons/arena_exercise_kcs.json";
  const DEFAULTS = { answer: "5m", review: "2m", quota: 8 };
  const QUOTA_MIN = 1;
  const QUOTA_MAX = 40;
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
      const quota = raw && Number.isFinite(raw.quota)
        ? Math.min(QUOTA_MAX, Math.max(QUOTA_MIN, Math.round(raw.quota)))
        : DEFAULTS.quota;
      // `attemptFirst` is deliberately NOT read back. It was the dialog's
      // fourth question until 2026-09-09 and a stored `true` from before then
      // must not resurrect a control the dialog no longer has — the notebook's
      // own timed attempt replaced it (practice/exercise-timer.js).
      return { answer, review, quota };
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
  const _long = (secs) => {
    const h = Math.floor(secs / 3600);
    const m = Math.round((secs % 3600) / 60);
    if (h && m) return `${h} h ${m} min`;
    if (h) return `${h} h`;
    return `${m} min`;
  };
  const _maxTimeText = (cfg) => {
    const a = _option(cfg.answer);
    const r = _option(cfg.review);
    if (!a || !r) return "";
    if (a.secs === null || r.secs === null) {
      const per = a.secs === null && r.secs === null
        ? "no time limit"
        : a.secs === null
          ? `no answer limit · ${_mmss(r.secs)} review`
          : `${_mmss(a.secs)} answer · no review limit`;
      return `open-ended (${per})`;
    }
    const per = a.secs + r.secs;
    return _long(per * cfg.quota);
  };

  /* ── the two numbers over the slider ───────────────────────────────
     Seth, 2026-09-09: "it should display both how long it expects that it will
     take you and the maximum amount of time that it would take you if you used
     up all the time."

     They are two different KINDS of number and the labels have to keep them
     apart, because a learner who reads the left one as a promise and then sits
     through the right one will not trust either again:

       EXPECTED is a description of THIS learner — the median seconds they
       actually spend answering a problem (practice/answer-history.js), times
       the count. It covers ANSWERING and says so. It is clamped to the answer
       cap per problem, because a learner slower than the cap does not get to
       be slower than the cap: the clock submits for them.

       MAXIMUM is arithmetic on the caps they chose — every problem running its
       answer clock and its review clock all the way down. Nothing about the
       learner is in it.

     🔴 NO ESTIMATE UNTIL THERE IS ONE. With fewer than a handful of answered
     problems on record there is no median worth showing, and filling the gap
     with a fraction of the cap would be inventing the learner's pace and
     printing it as if it were measured. The line says what is missing and what
     would fix it instead. */
  const _expectedText = (cfg) => {
    const H = window.AnswerHistory;
    const a = _option(cfg.answer);
    const per = H && typeof H.secondsPerProblem === "function" ? H.secondsPerProblem() : null;
    if (!Number.isFinite(per)) {
      const need = (H && H.MIN_SAMPLES) || 3;
      const have = H && typeof H.samples === "function" ? H.samples() : 0;
      return {
        text: "no pace on record yet",
        hint: `answer ${Math.max(1, need - have)} more problem${need - have === 1 ? "" : "s"} and this becomes your own median`,
      };
    }
    const capped = a && a.secs !== null ? Math.min(per, a.secs) : per;
    return {
      text: _long(capped * cfg.quota),
      hint: `${_mmss(Math.round(capped))} per problem — your median over your last ${H.samples()} answers`,
    };
  };

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
      /* ONE QUESTION: how long do you want to work on this?

         Seth, 2026-09-09: "instead of being greedy, it just asks you how long
         do you want to work on the problem. You should have a slider … as you
         increase the number of problems, it should automatically display above
         it the expected amount of time to the left and the maximum amount of
         time to the right."

         The slider's unit is PROBLEMS, not minutes, and that is deliberate:
         the block is a number of drills and always was — the two times over it
         are what that number costs. Sliding by minutes would round to a count
         behind the learner's back and then disagree with the count it ran.

         The two clocks are still here, one `<details>` down, because they are
         what the maximum is made of and a learner who wants 2:00 answers
         should not have to accept 5:00 to use the slider. They are no longer
         the first thing the dialog asks. */
      '<form class="dd-ex-form">' +
      '<div class="dd-ex-field dd-ex-count">' +
      '<div class="dd-ex-readout">' +
      '<span class="dd-ex-expected"><b class="dd-ex-expected-val"></b>' +
      '<span class="dd-ex-readout-label">expected, answering</span>' +
      '<span class="dd-ex-readout-hint"></span></span>' +
      '<span class="dd-ex-max"><b class="dd-ex-max-val"></b>' +
      '<span class="dd-ex-readout-label">maximum, if every clock runs out</span>' +
      '<span class="dd-ex-readout-hint"></span></span>' +
      "</div>" +
      `<label class="dd-ex-slider-label" for="dd-ex-quota"><span class="dd-ex-quota-text"></span></label>` +
      `<input type="range" id="dd-ex-quota" name="quota" min="${QUOTA_MIN}" max="${QUOTA_MAX}" step="1">` +
      '<span class="dd-ex-hint">A maximum, not a target — the block ends early if you run out of drills on this concept. ' +
      "Ordinary practice, narrowed to this exercise's concept: a miss pulls that concept's prerequisites in front of the queue.</span></div>" +
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
      const cfg = _formCfg();
      const a = _option(cfg.answer);
      const r = _option(cfg.review);
      const expected = _expectedText(cfg);
      modal.querySelector(".dd-ex-expected-val").textContent = expected.text;
      modal.querySelector(".dd-ex-expected .dd-ex-readout-hint").textContent = expected.hint;
      modal.querySelector(".dd-ex-max-val").textContent = _maxTimeText(cfg);
      modal.querySelector(".dd-ex-max .dd-ex-readout-hint").textContent =
        a && r && a.secs !== null && r.secs !== null
          ? `${cfg.quota} × (${_mmss(a.secs)} answer + ${_mmss(r.secs)} review)`
          : "one of the two clocks is off";
      modal.querySelector(".dd-ex-quota-text").textContent =
        `${cfg.quota} drill${cfg.quota === 1 ? "" : "s"} on this concept`;
      modal.querySelectorAll(".dd-ex-opt").forEach((l) =>
        l.classList.toggle("is-on", l.querySelector("input").checked));
    };
    form.addEventListener("change", paint);
    form.addEventListener("input", paint);
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
    const raw = Number(form.querySelector('input[name="quota"]').value);
    const quota = Number.isFinite(raw) ? Math.min(QUOTA_MAX, Math.max(QUOTA_MIN, Math.round(raw))) : DEFAULTS.quota;
    return { answer, review, quota };
  };

  const _open = (ex) => {
    pending = ex;
    const m = _ensureModal();
    m.querySelector(".dd-ex-title").textContent = `Drill ${ex.kcTitle || ex.title}`;
    m.querySelector(".dd-ex-sub").textContent =
      `Ordinary practice, narrowed to the concept behind ${ex.title}. Get one wrong and its prerequisite drills come next, on the same clock.`;
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
        `Paused at question ${paused.served}${c.quota ? ` of ${c.quota}` : ""} · ${a} answer · ${r} review. Your code, clock and place in the ladder are saved.`;
    } else {
      const cfg = _readCfg();
      form.querySelector('[data-for="answer"]').innerHTML = _radios("answer", cfg.answer);
      form.querySelector('[data-for="review"]').innerHTML = _radios("review", cfg.review);
      form.querySelector('input[name="quota"]').value = String(cfg.quota);
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
         🔴 BEFORE `startScoped`, never after: `discard` calls
         `KcPractice.stop()`, which would clear the ladder just built.
         Codex, 2026-09-06. */
      if (s.hasPausedSession?.()) s.discard?.();
      const kp = window.KcPractice;
      /* 🔴 `startScoped`, NOT `startPlanned` — the greedy planner is off this
         path as of 2026-09-09. Seth: "instead of being greedy … I think I just
         want it to do the normal practice, except that it's just practicing
         the skills for that specific problem."

         The planner existed to spend a small budget on whichever of {prep, an
         attempt at a variant of the exercise} most raised the chance of
         solving the exercise inside that budget. The attempt half of that
         question now has a better answer than a variant on the Practice tab:
         the exercise itself, in the notebook, on its own clock
         (practice/exercise-timer.js). What is left for this button is prep —
         which is the ordinary ladder, narrowed to one concept, and that is
         exactly what `startScoped` builds.

         practice/exercise-planner.js is NOT deleted and NOT dead: it is still
         reached through `KcPractice.startPlanned`, and its `masteryFromEstimate`
         / `READY` are what the timer's drill recommendation reads. Removing it
         is a separate decision from removing it from this button. */
      const ok = await kp?.startScoped?.(ex.kc);
      if (!ok) {
        rollback();
        _fail("No drills are attached to this exercise's concept yet.");
        return;
      }
      s.configure({
        answer: _option(cfg.answer).secs,
        review: _option(cfg.review).secs,
        quota: cfg.quota,
        exercise: { kc: ex.kc, fn: ex.fn, title: ex.title, nb: ex.nb, cell: ex.cell },
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
     — the 0.0 einsum exercises share ONE cell — and each gets its own block. */
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
      const src = cell.querySelector("textarea")?.value ?? cell.querySelector("pre, code")?.textContent ?? cell.textContent ?? "";
      return Array.from(src.matchAll(DEF_RE), (m) => m[1]);
    }
    return [];
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
      ? `Resume drills · question ${paused.served}${paused.config.quota ? ` of ${paused.config.quota}` : ""}`
      : "Drill prerequisite concepts";
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
    host.querySelectorAll(".nbv-cell.nbv-md, .nbv-cell.nbv-code").forEach((cell) => {
      if (cell.nextElementSibling?.classList?.contains("dd-ex-block")) return;
      // Blocks go after the cell in reverse so several defs in one cell read top-down.
      _cellKeys(cell).filter((fn) => table[fn]?.kc && !seen.has(fn)).reverse().forEach((fn) => {
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
      cell.insertAdjacentElement("afterend", block);
      _syncButton(block);
      // A clock the learner started before a reload comes back here, because
      // this is the first moment the exercise and its block exist together.
      window.ExerciseTimer?.restore?.(ex, block);
      window.LessonGate?.getKpEntry?.(ex.kc).then((entry) => {
        if (entry && entry.kp && entry.kp.title) ex.kcTitle = entry.kp.title;
      }).catch(() => {});
      });
    });
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
    if (reason === "complete" && outcome?.solved) {
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
