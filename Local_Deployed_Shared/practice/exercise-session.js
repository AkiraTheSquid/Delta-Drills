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
     • the "Related drills" list: every drill on the KC and its
       prerequisites, grouped by concept and rung — read-only

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
  const _options = () => (window.SessionClock && window.SessionClock.OPTIONS) || [
    { id: "2m", secs: 120, label: "2:00" },
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
      return { answer, review, quota, attemptFirst: raw?.attemptFirst !== false };
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
      return `Maximum time: open-ended (${per})`;
    }
    const per = a.secs + r.secs;
    return `Maximum time: ${_long(per * cfg.quota)} — ${cfg.quota} × (${_mmss(a.secs)} + ${_mmss(r.secs)})`;
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
      '<form class="dd-ex-form">' +
      '<fieldset class="dd-ex-field"><legend>1. Answer time per problem</legend><div class="dd-ex-opts" data-for="answer"></div></fieldset>' +
      '<fieldset class="dd-ex-field"><legend>2. Review time per problem</legend><div class="dd-ex-opts" data-for="review"></div></fieldset>' +
      '<div class="dd-ex-field dd-ex-count"><label>3. How many problems ' +
      `<input type="number" name="quota" min="${QUOTA_MIN}" max="${QUOTA_MAX}" step="1" inputmode="numeric"></label>` +
      '<span class="dd-ex-hint">A maximum, not a target: each slot goes to prep or to a variant of this problem, whichever most raises the chance of solving one within what is left — fewer questions, greedier. The block ends when you solve one; a miss re-plans. The maximum time holds.</span></div>' +
      '<fieldset class="dd-ex-field dd-ex-start-choice"><legend>4. How do you want to start?</legend>' +
      '<div class="dd-ex-opts"><label class="dd-ex-opt"><input type="radio" name="startMode" value="attempt" checked>' +
      '<span>Try the problem first</span></label>' +
      '<label class="dd-ex-opt"><input type="radio" name="startMode" value="adaptive">' +
      '<span>Let the app choose prep or an attempt</span></label></div>' +
      '<p class="dd-ex-hint">Test what you already know before any prep. This uses one question from your maximum; after a miss, the app plans the remaining questions.</p></fieldset>' +
      '<p class="dd-ex-max" aria-live="polite"></p>' +
      '<p class="dd-ex-error hidden" role="alert"></p>' +
      '<div class="dd-ex-actions">' +
      '<button type="submit" class="primary dd-ex-start-btn">Start</button>' +
      '<button type="button" class="ghost dd-ex-cancel-btn">Cancel</button>' +
      "</div></form></div>";
    document.body.appendChild(modal);

    const form = modal.querySelector(".dd-ex-form");
    const paint = () => {
      const cfg = _formCfg();
      modal.querySelector(".dd-ex-max").textContent = _maxTimeText(cfg);
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
    return { answer, review, quota, attemptFirst: pick("startMode") === "attempt" };
  };

  const _open = (ex) => {
    pending = ex;
    const m = _ensureModal();
    m.querySelector(".dd-ex-title").textContent = `Practice ${ex.title}`;
    m.querySelector(".dd-ex-sub").textContent =
      `Drills on ${ex.kcTitle || ex.kc}. Get one wrong and its prerequisite drills come next, on the same clock.`;
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
      const canAttempt = Array.isArray(ex.variants) && ex.variants.length > 0;
      form.querySelector(".dd-ex-start-choice").hidden = !canAttempt;
      form.querySelector(`input[name="startMode"][value="${canAttempt && cfg.attemptFirst !== false ? "attempt" : "adaptive"}"]`).checked = true;
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
      const ok = kp && typeof kp.startPlanned === "function"
        ? await kp.startPlanned(ex.kc, { quota: cfg.quota, variants: ex.variants, attemptFirst: cfg.attemptFirst })
        : await kp?.startScoped?.(ex.kc);
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
     A markdown cell yields at most one key; a code cell may yield several
     — the 0.0 einsum exercises share ONE cell — and each gets its own block. */
  const TAG_RE = /^\((\w{1,3})\)\s/;
  /* The run button (▶) is glued to the first line of a code cell's
     textContent, so a def may follow it instead of a newline. */
  const DEF_RE = /(?:^|[\n▶])\s*def\s+([A-Za-z_]\w*)\s*\(/g;
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
    const note = block.querySelector(".dd-ex-note");
    const ex = block._exercise;
    const paused = _pausedFor(ex);
    btn.textContent = paused ? `Resume practice · ${ex.title}` : `Practice ${ex.title}`;
    note.textContent = paused
      ? `Paused at question ${paused.served}${paused.config.quota ? ` of ${paused.config.quota}` : ""}.`
      : "Timed drills on this exercise's concept. A miss pulls in its prerequisites.";
  };

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
      };
      const block = document.createElement("div");
      block.className = "dd-ex-block";
      block._exercise = ex;
      block.innerHTML =
        '<div class="dd-ex-row"><button type="button" class="primary dd-ex-btn"></button>' +
        '<span class="dd-ex-note"></span></div>' +
        '<details class="dd-ex-related"><summary>Related drills</summary><div class="dd-ex-related-body">Loading…</div></details>';
      block.querySelector(".dd-ex-btn").onclick = () => _open(ex);
      block.querySelector(".dd-ex-related").addEventListener("toggle", (e) => {
        if (e.target.open && !block._relatedLoaded) {
          block._relatedLoaded = true;
          _renderRelated(ex, block.querySelector(".dd-ex-related-body"));
        }
      }, { once: false });
      cell.insertAdjacentElement("afterend", block);
      _syncButton(block);
      window.LessonGate?.getKpEntry?.(ex.kc).then((entry) => {
        if (entry && entry.kp && entry.kp.title) ex.kcTitle = entry.kp.title;
      }).catch(() => {});
      });
    });
  };

  /* ── related drills: the KC and its prerequisites, by rung ─────── */
  const RUNGS = [
    ["faded_items", "Faded", (it) => it && it.question_id],
    ["guided_items", "Guided", (it) => it && it.question_id],
    ["independent_items", "Solo", (id) => id],
    ["integrated_items", "Integrated", (it) => it && it.question_id],
  ];

  const _firstLine = (text) => {
    const line = String(text || "").split("\n").map((l) => l.trim()).find(Boolean) || "";
    return line.replace(/[`*_#]/g, "").slice(0, 110);
  };

  const _renderRelated = async (ex, host) => {
    try {
      const gate = window.LessonGate;
      if (!gate?.getKpEntry) throw new Error("lessons not loaded");
      if (typeof loadQuestionsBank === "function") await loadQuestionsBank();
      const root = await gate.getKpEntry(ex.kc);
      if (!root || !root.kp) throw new Error("no lesson for " + ex.kc);
      const kcs = [ex.kc, ...(root.kp.supporting_kcs || [])];
      const sections = [];
      let total = 0;
      for (const kc of kcs) {
        const entry = kc === ex.kc ? root : await gate.getKpEntry(kc);
        if (!entry || !entry.kp) continue;
        const rows = [];
        for (const [field, label, pick] of RUNGS) {
          const ids = (entry.kp[field] || []).map(pick).filter((id) => Number.isFinite(id));
          if (!ids.length) continue;
          const items = ids.map((id) => {
            const q = typeof getQuestionFromBank === "function" ? getQuestionFromBank(id) : null;
            const text = q ? _firstLine(q.question_text || q.prompt) : `question ${id}`;
            return `<li><span class="dd-ex-qid">#${id}</span> ${_esc(text)}</li>`;
          });
          total += items.length;
          rows.push(`<div class="dd-ex-rung"><span class="dd-ex-rung-label">${label} · ${items.length}</span><ul>${items.join("")}</ul></div>`);
        }
        if (!rows.length) continue;
        sections.push(
          `<section class="dd-ex-kc${kc === ex.kc ? " is-target" : ""}">` +
          `<h4>${_esc(entry.kp.title || kc)}${kc === ex.kc ? "" : ' <span class="dd-ex-prereq">prerequisite</span>'}</h4>` +
          rows.join("") + "</section>",
        );
      }
      host.innerHTML = sections.length
        ? `<p class="dd-ex-related-count">${total} drills across ${sections.length} concept${sections.length === 1 ? "" : "s"}. A miss on the top concept queues the prerequisite rows.</p>` + sections.join("")
        : "<p>No drills attached yet.</p>";
    } catch (err) {
      host.textContent = "Could not list the drills — " + (err?.message || err);
    }
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
