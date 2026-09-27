/* ================================================================
   SURVEY.JS — "What have you done before?" on entering Practice.

   Seth, 2026-09-26: no placement test as a separate mode; keep
   explore/exploit and the novice / intermediate / expert question, and when
   the learner enters the Practice tab without having answered, ask a few
   simple, fast questions — PyTorch before? a linear algebra course? — as a
   jumpstart on the prior for explore/exploit, per area rather than one
   prior across the board.

   The questions come from GET /api/practice/survey (app/area_survey.py
   owns them, and which areas each one speaks for). Each is answered
   Never / A little / A lot. The level row shows only while the backend
   still takes it (`level_open`: nothing answered yet). Skip (or a question
   left blank) is stored as skipped, so it is not asked again; a course
   enabled later brings only its own question, on the next visit.

   While the card is up, the rest of the idle Learner Home steps aside
   (`#page-practice.is-surveying`, styles/survey.css): it is a question,
   not a banner. It never shows over a question on screen — only on the
   idle surface (`.session-idle`).
   ================================================================ */
(function () {
  "use strict";

  const LABELS = { never: "Never", some: "A little", lot: "A lot" };
  const LEVEL_LABELS = { novice: "Novice", intermediate: "Intermediate", expert: "Expert" };

  const page = () => document.getElementById("page-practice");
  const host = () => document.getElementById("learner-survey");
  const fetcher = () => (typeof apiFetch === "function" ? apiFetch : window.apiFetch);
  const hasSession = () => typeof authToken !== "undefined" && !!authToken;
  // Only on the idle Learner Home: never over a question on screen (a GET
  // that resolves after the learner pressed Continue must not paint).
  const idle = () => {
    const p = page();
    return !!p && !p.classList.contains("hidden") && p.classList.contains("session-idle");
  };

  let data = null; // the last GET body
  let seq = 0;

  const el = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  };

  const hide = () => {
    const h = host();
    if (h) {
      h.hidden = true;
      h.replaceChildren();
    }
    page()?.classList.remove("is-surveying");
  };

  /** A row of three toggle buttons; `pick(value)` on press. */
  const choices = (label, options, current, pick) => {
    const group = el("div", "survey-choices");
    group.setAttribute("role", "group");
    group.setAttribute("aria-label", label);
    options.forEach(([value, text]) => {
      const b = el("button", "survey-choice", text);
      b.type = "button";
      b.setAttribute("aria-pressed", String(value === current));
      b.addEventListener("click", () => {
        group.querySelectorAll(".survey-choice").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
        pick(value);
      });
      group.appendChild(b);
    });
    return group;
  };

  const render = () => {
    const h = host();
    if (!h || !data || data.answered || !idle()) return hide();
    const answers = {};
    let level = data.level_open ? data.level : null;

    const card = el("div", "survey-card");
    card.appendChild(el("p", "survey-eyebrow", "Before you start"));
    card.appendChild(el("h2", "survey-title", "What have you done before?"));
    card.appendChild(el("p", "survey-lede",
      "A few taps so the first questions start in the right place. Your answers only decide where to look; the questions you answer decide the rest."));

    if (data.level_open) {
      const row = el("div", "survey-row survey-row-level");
      row.appendChild(el("p", "survey-q", "How much of this have you done?"));
      const levels = (data.levels || Object.keys(LEVEL_LABELS)).map((v) => [v, LEVEL_LABELS[v] || v]);
      row.appendChild(choices("Your level", levels, level, (v) => { level = v; }));
      card.appendChild(row);
    }

    const opts = (data.choices || Object.keys(LABELS)).map((c) => [c, LABELS[c] || c]);
    (data.questions || []).forEach((q) => {
      const row = el("div", "survey-row");
      row.appendChild(el("p", "survey-q", q.text));
      row.appendChild(choices(q.text, opts, null, (v) => { answers[q.id] = v; }));
      card.appendChild(row);
    });

    const status = el("p", "survey-status");
    status.setAttribute("role", "status");
    const actions = el("div", "survey-actions");
    const go = el("button", "primary survey-go", "Start practising");
    go.type = "button";
    const skip = el("button", "survey-skip", "Skip");
    skip.type = "button";
    const send = async (body) => {
      go.disabled = skip.disabled = true;
      status.textContent = "";
      try {
        const _fetch = fetcher();
        const res = await _fetch("/api/practice/survey", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        const next = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(next?.detail || "Your answers could not be saved.");
        data = next;
        seq += 1; // an older read must not bring the card back
        hide();
        // The queue's explore gate reads the survey: let practice re-plan.
        window.dispatchEvent(new CustomEvent("delta:practice-state-changed"));
      } catch (err) {
        status.textContent = err.message || "Your answers could not be saved.";
        go.disabled = skip.disabled = false;
      }
    };
    go.addEventListener("click", async () => {
      await send({ answers, level });
      // Answered: the button says "Start practising", so it starts.
      if (data?.answered) document.getElementById("session-continue-btn")?.click();
    });
    skip.addEventListener("click", () => send({ answers: {} }));
    actions.append(go, skip);
    card.append(actions, status);

    h.replaceChildren(card);
    h.hidden = false;
    page()?.classList.add("is-surveying");
  };

  const load = async () => {
    const mine = ++seq;
    if (!hasSession() || typeof fetcher() !== "function") {
      data = null;
      return hide();
    }
    try {
      const res = await fetcher()("/api/practice/survey");
      if (!res?.ok) throw new Error(String(res?.status));
      const body = await res.json();
      if (mine !== seq) return;
      data = body;
      render();
    } catch (_) {
      // A failed read never blocks practice: no card.
      if (mine === seq) hide();
    }
  };

  // The Practice page has no "shown" event of its own; switchTab toggles
  // its `hidden` class and the session its `session-idle` class, so those
  // are what is watched. Every entry re-reads: a course enabled since the
  // last visit may have brought a question.
  const watch = () => {
    const p = page();
    if (!p) return;
    let wasHidden = p.classList.contains("hidden");
    let wasIdle = idle();
    new MutationObserver(() => {
      const nowHidden = p.classList.contains("hidden");
      const nowIdle = idle();
      if (wasHidden && !nowHidden) load();
      else if (nowIdle !== wasIdle) render(); // leaving idle hides; returning shows a pending card
      wasHidden = nowHidden;
      wasIdle = nowIdle;
    }).observe(p, { attributes: true, attributeFilter: ["class"] });
    if (!wasHidden) load();
  };

  window.addEventListener("delta:auth-state-changed", () => {
    data = null;
    hide();
    if (!page()?.classList.contains("hidden")) load();
  });

  window.addEventListener("delta:practice-mode-ready", () => {
    if (!page()?.classList.contains("hidden") && !data?.answered) load();
  });

  window.DeltaSurvey = { reload: load };

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", watch);
  else watch();
})();
