/* ================================================================
   CONCEPT TOPBAR — one persistent header for the whole practice page

   What it replaces, and why. The lesson screen used to build its own
   `.lesson-topbar` inside the question HTML. Two problems followed from that:
   it lived inside `.practice-left`, so it stopped at the panel divider instead
   of spanning the page; and it was part of an innerHTML that gets thrown away
   on every render, so nothing above the fold could survive between pages.

   This module owns that strip instead. It sits above `.practice-split` in the
   markup, so it spans both panels, and it is updated rather than rebuilt. It is
   the one place on the page that answers three questions the learner should
   never have to hunt for:

     * WHICH concept is this?          — named, and clickable into the graph
     * WHERE am I in the sequence?     — four dots, all four always visible
     * HOW WELL am I doing on it?      — the interval, not the point estimate

   THE FOUR RUNGS

   The scaffold sequence is `lesson -> worked -> faded -> solo`: read the
   teaching page, then solve beside a visible example, then finish a partly
   written solution, then work unaided. The backend still speaks the older
   vocabulary (`worked, faded, partial, solo`) where its `worked` means the
   lesson page; `STAGE_ALIASES` maps one onto the other so the display can be
   correct before the state migration lands. That mapping is the only place the
   two vocabularies meet — do not scatter it.

   All four dots are always drawn, including the ones already passed and the
   ones not yet reached. A single label ("Faded") tells a learner what they are
   doing; four dots tell them how much support they have already given up and
   how much is left, which is the thing that makes the ladder legible as a
   ladder rather than as an arbitrary change of question format.
   ================================================================ */

const ConceptTopbar = (() => {
  "use strict";

  // Display order. Left to right is decreasing support.
  const STAGES = [
    { id: "lesson", label: "Lesson", blurb: "Read the explanation and run the examples." },
    { id: "worked", label: "Worked", blurb: "Solve beside a worked example you can still see." },
    { id: "faded", label: "Faded", blurb: "Most of the solution is written — supply the rest." },
    { id: "solo", label: "Solo", blurb: "No scaffold. You have earned it." },
  ];

  /* Backend stage -> displayed rung.

     The server's `worked` is the lesson screen (it is the rung at which
     `LessonGate` takes over and no drill is served), and its `faded` and
     `partial` are the two supported drill rungs. Renaming those in the backend
     rewrites every learner's stored `kc_ladder[kc].attempts[].stage`, so the
     display is corrected first and the state migration follows separately. */
  const STAGE_ALIASES = {
    worked: "lesson",
    faded: "worked",
    partial: "faded",
    solo: "solo",
    independent: "solo",
    // Already-new vocabulary passes through untouched, so this file needs no
    // edit on the day the backend switches over.
    lesson: "lesson",
  };

  const _index = (stage) => STAGES.findIndex((s) => s.id === stage);

  const esc = (value) =>
    String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");

  const _el = (id) => document.getElementById(id);

  /* The concept currently on screen, so `update()` can refresh the estimate
     after a submit without the caller having to re-supply everything. */
  let current = { kc: null, stage: null };

  const normalizeStage = (stage) => STAGE_ALIASES[stage] || null;

  const _openGraph = (kc) => {
    if (!kc) return;
    if (typeof switchTab === "function") switchTab("knowledge-graph");
    requestAnimationFrame(() => {
      if (typeof window.deltaFocusConceptGraphKc === "function") {
        window.deltaFocusConceptGraphKc(kc);
      }
    });
  };

  /* The estimate, rendered as an interval.

     Deliberately not the point estimate: 2-for-2 reads as 100% and tells the
     learner nothing about how little that rests on. The bar is drawn from the
     interval too — a wide bar IS the message that the number is not yet worth
     much, and it is the same quantity the ladder promotes on, so what the
     learner sees is what the system is actually deciding with. */
  const _estHtml = (est) => {
    if (!est || !est.n) {
      return (
        '<span class="concept-topbar-est-new" ' +
        'title="No graded attempts at this concept yet — nothing to estimate from.">' +
        "no attempts yet</span>"
      );
    }
    const lo = Math.max(0, Math.min(1, est.ci?.[0] ?? 0));
    const hi = Math.max(0, Math.min(1, est.ci?.[1] ?? 1));
    const loPct = Math.round(lo * 100);
    const hiPct = Math.round(hi * 100);
    const left = (lo * 100).toFixed(1);
    const width = Math.max(1, (hi - lo) * 100).toFixed(1);
    return (
      `<span class="concept-topbar-est-count">${est.correct}/${est.n}</span>` +
      '<span class="concept-topbar-est-bar" aria-hidden="true">' +
      `<span class="concept-topbar-est-fill" style="left:${left}%;width:${width}%"></span>` +
      "</span>" +
      `<span class="concept-topbar-est-range">${loPct}–${hiPct}%</span>`
    );
  };

  const _stagesHtml = (stage) => {
    const active = _index(stage);
    return STAGES.map((s, i) => {
      const state = i < active ? "is-done" : i === active ? "is-active" : "is-todo";
      return (
        `<li class="stage-dot ${state}" title="${esc(s.label)} — ${esc(s.blurb)}"` +
        (i === active ? ' aria-current="step"' : "") +
        ">" +
        '<span class="stage-dot-mark" aria-hidden="true"></span>' +
        `<span class="stage-dot-label">${esc(s.label)}</span>` +
        "</li>"
      );
    }).join("");
  };

  /* Show the bar for one concept.

     `stage` accepts either vocabulary. An unrecognised stage hides the dots
     rather than guessing a position — showing the learner the wrong rung is
     worse than showing them none. */
  const show = ({ kc, title, eyebrow, stage, estimate } = {}) => {
    const host = _el("concept-topbar");
    if (!host) return;
    const normalized = normalizeStage(stage);
    current = { kc: kc || null, stage: normalized };

    const eyebrowEl = _el("concept-topbar-eyebrow");
    if (eyebrowEl) eyebrowEl.textContent = eyebrow || "Concept";

    const kcBtn = _el("concept-topbar-kc");
    if (kcBtn) {
      kcBtn.textContent = title || kc || "";
      kcBtn.dataset.kc = kc || "";
      kcBtn.hidden = !kc;
      kcBtn.title = kc ? `Open “${kc}” in the knowledge graph` : "";
      kcBtn.onclick = () => _openGraph(kcBtn.dataset.kc);
    }

    const stagesEl = _el("concept-topbar-stages");
    if (stagesEl) {
      stagesEl.innerHTML = normalized ? _stagesHtml(normalized) : "";
      stagesEl.hidden = !normalized;
    }

    setEstimate(estimate);
    host.classList.remove("hidden");
  };

  /* Refresh only the estimate — what a submit changes. Kept separate so a
     grade does not have to re-render the concept name and dots, which would
     flash the one part of the page that is supposed to be stable. */
  const setEstimate = (estimate) => {
    const estEl = _el("concept-topbar-est");
    if (estEl) estEl.innerHTML = _estHtml(estimate);
  };

  /* Move the dots without touching anything else — used when a rung is earned
     mid-screen (reading the worked example promotes immediately). */
  const setStage = (stage) => {
    const normalized = normalizeStage(stage);
    if (!normalized) return;
    current.stage = normalized;
    const stagesEl = _el("concept-topbar-stages");
    if (stagesEl) {
      stagesEl.innerHTML = _stagesHtml(normalized);
      stagesEl.hidden = false;
    }
  };

  const hide = () => {
    const host = _el("concept-topbar");
    if (host) host.classList.add("hidden");
    current = { kc: null, stage: null };
  };

  const activeKc = () => current.kc;

  return { show, hide, setEstimate, setStage, activeKc, normalizeStage, STAGES };
})();

window.ConceptTopbar = ConceptTopbar;
