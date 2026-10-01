/* ================================================================
   OR CHOOSE A CONCEPT — the Learner Home's candidate list.

   Seth, 2026-09-28: "for the learner home, there is a list of candidate
   concepts to work on, along with displaying your ability for each area
   using the same bar that is usually at the top ... this would give the
   learner some choice". And: "there should be a button that says 'practice
   with AI' as well as the candidate list (and one of the items is the one
   that the ai would have recommended). after they learn the concept, it
   brings them back to the original page or if they fail too many times, it
   shows a screen saying that they were not ready for that concept, and need
   to practice other concepts first."

   The list and every decision in it are the backend's
   (GET /api/practice/concept-candidates, app/concept_choice.py): five
   concepts the algorithm would serve, its own pick first, each with the
   XP one solved problem on it would earn and the learner's knowledge K on
   a slim copy of the topbar concept pill's meter (styles/concept-choice.css).
   This file draws them and starts a block.

   XP PER PROBLEM (Seth, 2026-09-29: "not all concepts give the same amount
   of exp right? ... on the far right, make it such that it displays the
   amount of exp you would get from solving that sort of problem using the
   same layout, but with different numbers"): the server's `xp_per_problem`
   (app/learning_xp.py::solve_xp) — the model's expected gain from one
   solved problem, priced at the concept's worth, to the nearest 5. The
   figure sits where "57 / 80 XP" did; the meter under the row is still the
   concept's knowledge toward READY.

   A CHOSEN CONCEPT runs the exercise dialog's route with `kind: "choice"`
   (practice/ready-route.js → POST /api/practice/concept-route): the concept
   alone, until the model says learned or not ready. Started exactly the way
   practice/exercise-session.js starts an exercise block — discard any paused
   block FIRST, then `startRoute`, `configure`, `start` — so pause, resume
   and the end of the block are timer.js's own. The end lands back here
   (the session goes idle) and is announced by exercise-session.js::onEnd
   as `dd-exercise-end`: learned prints one line in the summary; not ready
   opens the dialog below, one message and one button back to the list.

   "Practice with AI" is the button above the list (practice/session-idle.js).
   ================================================================ */
(function () {
  "use strict";

  /* 🔴 `PracticeSession` and `apiFetch` are top-level consts of classic
     scripts: NOT on window. Resolve through the lexical scope. */
  const _session = () =>
    typeof PracticeSession !== "undefined" ? PracticeSession : window.PracticeSession;
  const _fetch = () => (typeof apiFetch === "function" ? apiFetch : window.apiFetch);
  const _backend = () => typeof practiceMode !== "undefined" && practiceMode === "backend";
  const _signedIn = () => window.DDIdentity?.isSignedIn?.() === true;

  const root = () => document.getElementById("concept-choice");
  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  };

  let data = null; // {items, ready_at}
  let busy = false;
  let seq = 0;
  /* A failed read tries again on its own: a read during a backend restart
     (every Fly deploy) used to leave this blank until the page was left and
     re-entered (Seth, 2026-10-01). 2 s, 5 s, 10 s, 20 s, 40 s, then a
     minute apart, eight tries in all; any fresh read resets it. */
  const RETRY_S = [2, 5, 10, 20, 40, 60, 60, 60];
  const AGAIN = {}; // load(AGAIN) = a retry; any other call is a fresh read
  let retryTimer = null;
  let tries = 0;
  const retryLater = () => {
    if (tries >= RETRY_S.length) return;
    retryTimer = setTimeout(() => load(AGAIN), RETRY_S[tries] * 1000);
    tries += 1;
  };

  async function load(why) {
    const mine = ++seq;
    clearTimeout(retryTimer);
    if (why !== AGAIN) tries = 0;
    const f = _fetch();
    if (!_signedIn() || !_backend() || typeof f !== "function") {
      data = null;
      paint();
      return;
    }
    try {
      // A paused chosen block is listed even once it leaves the five.
      const keep = pausedKc();
      const res = await f(`/api/practice/concept-candidates${keep ? `?keep=${encodeURIComponent(keep)}` : ""}`);
      if (!res?.ok) throw new Error(`concept-candidates ${res?.status}`);
      const got = await res.json();
      if (mine !== seq) return;
      data = got && Array.isArray(got.items) ? got : null;
      tries = 0;
    } catch (err) {
      // A failed read keeps what was drawn, and tries again shortly.
      console.warn("[concept-choice]", err);
      if (mine === seq) retryLater();
      return;
    }
    paint();
  }

  // ── the list ───────────────────────────────────────────────────
  /** The concept of a paused chosen-concept block, if one is waiting. */
  const pausedKc = () => {
    const s = _session();
    if (!s?.hasPausedSession?.()) return null;
    const ex = s.pausedConfig?.()?.exercise;
    return ex?.mode === "choice" ? ex.kc : null;
  };

  /* A row is a ledger line (Seth, 2026-09-28, on v1 — the full topbar pill
     per row, label written on the fill: "this just doesn't look that good").
     The name gets its own line and is never cut; the XP one problem earns
     is a figure in the home's numeral face; the pill stays as the meter
     under them — its track, its fill, its colours — slimmed to a rule, so no
     word straddles a seam. */
  function row(it, readyAt, paused, i) {
    const xp = Math.max(0, Math.round(Number(it.xp_per_problem) || 0));
    const known = Math.max(0, Math.min(1, (Number(it.k) || 0) / readyAt));
    const b = el("button", `cc-row${it.recommended ? " is-ai" : ""}${paused ? " is-paused" : ""}`);
    b.type = "button";
    b.style.setProperty("--cc-i", i);
    b.setAttribute(
      "aria-label",
      `${paused ? "Continue" : "Practice"} ${it.title} — ${xp} XP a problem, ${Math.round(known * 100)}% learned` +
        `${it.recommended ? ", the AI's pick" : ""}${it.answers ? "" : ", not tried yet"}`,
    );
    // What the row is, before its name: the AI's pick, the paused block.
    const flags = [];
    if (paused) flags.push(el("span", "cc-flag cc-flag--continue", "Paused · continue"));
    if (it.recommended) flags.push(el("span", "cc-flag cc-flag--ai", "AI pick"));
    if (flags.length) {
      const eyebrow = el("span", "cc-eyebrow");
      eyebrow.append(...flags);
      b.appendChild(eyebrow);
    }
    const sub = [it.lesson_title, it.answers ? "" : "not tried yet"].filter(Boolean).join(" · ");
    const figure = el("span", "cc-figure");
    figure.append(el("span", "cc-xp", `+${xp}`), el("span", "cc-of", "XP / problem"));
    const bar = el("span", "cc-bar");
    bar.setAttribute("aria-hidden", "true");
    bar.style.setProperty("--cc-pct", `${known * 100}%`);
    bar.appendChild(el("span", "cc-bar-fill"));
    b.append(el("span", "cc-title", it.title), figure, el("span", "cc-sub", sub), bar);
    b.addEventListener("click", () => choose(it, paused));
    const li = el("li", "cc-item");
    li.appendChild(b);
    return li;
  }

  function paint() {
    const r = root();
    if (!r) return;
    const items = data?.items || [];
    r.hidden = !items.length;
    const readyAt = Number(data?.ready_at) || 0.8;
    const paused = pausedKc();
    r.querySelector(".cc-list")?.replaceChildren(...items.map((it, i) => row(it, readyAt, it.kc === paused, i)));
  }

  const fail = (text) => {
    const e = root()?.querySelector(".cc-error");
    if (!e) return;
    e.textContent = text;
    e.classList.toggle("hidden", !text);
  };

  const setBusy = (on) => {
    busy = on;
    root()?.querySelectorAll(".cc-row").forEach((b) => { b.disabled = on; });
  };

  // ── starting a block on the chosen concept ─────────────────────
  async function choose(it, paused) {
    const s = _session();
    if (busy) return;
    fail("");
    if (!s) return fail("The practice engine did not load — reload the page.");
    if (s.isActive?.()) return fail("A block is already running — pause it first.");
    // Its own paused block: timer.js's resume, unless the saved question was refused.
    const resumeBtn = document.getElementById("session-resume-btn");
    if (paused && resumeBtn && !resumeBtn.disabled) {
      resumeBtn.click();
      return;
    }
    // Already learned (only the AI's pick is listed at READY): nothing to
    // route — the algorithm's own queue is what practising it means.
    if (Number(it.k) >= (Number(data?.ready_at) || 0.8)) {
      document.getElementById("session-continue-btn")?.click();
      return;
    }
    setBusy(true);
    const rollback = () => {
      window.KcPractice?.stop?.();
      s.configure?.(null);
    };
    try {
      /* 🔴 BEFORE `startRoute`, never after: `discard` calls
         `KcPractice.stop()`, which would clear the route just built. */
      if (s.hasPausedSession?.()) s.discard?.();
      const ok = await window.KcPractice?.startRoute?.(it.kc, { title: it.title, kind: "choice" });
      if (!ok) {
        rollback();
        fail(`No drills for ${it.title} yet.`);
        return;
      }
      s.configure({ exercise: { kc: it.kc, title: it.title, mode: "choice" } });
      s.start();
    } catch (err) {
      console.warn("[concept-choice] could not start:", err);
      rollback();
      fail("Could not start — " + (err?.message || err));
    } finally {
      setBusy(false);
    }
  }

  // ── not ready: one message, one way back ──────────────────────
  let dialog = null;
  function notReady(title) {
    if (!dialog) {
      dialog = el("dialog", "cc-not-ready");
      dialog.setAttribute("aria-labelledby", "cc-not-ready-head");
      const head = el("h2", "cc-not-ready-head", "Not ready for this one yet");
      head.id = "cc-not-ready-head";
      const btn = el("button", "primary cc-not-ready-btn", "Back to the list");
      btn.type = "button";
      btn.addEventListener("click", () => dialog.close());
      dialog.append(head, el("p", "cc-not-ready-text"), btn);
      dialog.addEventListener("close", () => root()?.querySelector(".cc-row")?.focus());
      document.body.appendChild(dialog);
    }
    dialog.querySelector(".cc-not-ready-text").textContent =
      `Your answers on ${title} say it is out of reach for now. Practice other concepts first — ` +
      "Practice with AI picks them for you. Your answers are kept.";
    if (typeof dialog.showModal === "function" && !dialog.open) dialog.showModal();
  }

  document.addEventListener("dd-exercise-end", (e) => {
    const { reason, config, outcome } = e.detail || {};
    if (config?.exercise?.mode !== "choice") return;
    if (reason === "complete" && outcome?.notReady) notReady(config.exercise.title || "This concept");
    load();
  });

  // ── when to read the list again ────────────────────────────────
  /* Every time the Learner Home comes back into view — the tab shown, or a
     block paused or ended (the page goes `session-idle`): answers change
     the knowledge the list is ordered and drawn by. */
  const watchArrival = () => {
    const page = document.getElementById("page-practice");
    if (!page || typeof MutationObserver !== "function") return;
    const idle = () => !page.classList.contains("hidden") && page.classList.contains("session-idle");
    let was = idle();
    new MutationObserver(() => {
      const now = idle();
      if (now && !was) load();
      was = now;
    }).observe(page, { attributes: true, attributeFilter: ["class"] });
  };
  window.addEventListener("delta:auth-state-changed", () => {
    data = null;
    paint();
    load();
  });
  // practice/init.js announces the backend; a read before it answers nothing.
  window.addEventListener("delta:practice-mode-ready", load);

  const boot = () => {
    watchArrival();
    load();
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();

  window.DDConceptChoice = { reload: load };
})();
