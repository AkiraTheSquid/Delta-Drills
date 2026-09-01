/* ================================================================
   ACTIVITY CHART — problems answered per day, Monday to Sunday

   The Learner Home's "how much did I actually work this week" readout
   (Seth, 2026-09-01: "Below the information regarding your strength in
   all the different areas is the amount of problems that you've done
   across the different days as like a bar chart"). Seven bars, current
   week only, drawn into #activity-week-chart under #learner-areas.

   WHERE THE NUMBERS COME FROM
     /api/practice/activity-week — the backend counts ANSWERED attempt
     rows in the learner's own attempt log. The client sends
     `new Date().getTimezoneOffset()` so the backend buckets rows into
     the LEARNER'S days; without it an evening session here lands on
     tomorrow's bar (the log stamps UTC).

   WHEN IT REDRAWS
     Once when practice mode resolves (`delta:practice-mode-ready`, the
     first moment auth is settled), and again after every graded attempt
     via `delta:practice-state-changed` — that event fires from api.js on
     each state move, so today's bar grows as the learner works. The
     refetch is debounced: a submit fires the event more than once and
     one fetch per answer is already more than the chart needs.

   🔴 SIGNED OUT (or any fetch failure) HIDES THE WHOLE SECTION via
   `.is-empty` — same contract as #learner-areas: an empty bordered box
   that says nothing is worse than no box. A guest has no attempt log to
   read and the endpoint answers 401.
   ================================================================ */

(function () {
  const DAY_LETTERS = ["M", "T", "W", "T", "F", "S", "S"];
  let debounceTimer = null;
  // Monotonic fetch ticket. Refreshes overlap — mode-ready and the direct
  // boot call race, and a slow response from before an attempt could land
  // AFTER the refetch that attempt triggered, painting yesterday's count
  // over today's. Only the newest request may render.
  let fetchSeq = 0;

  const host = () => document.getElementById("learner-activity");
  const chart = () => document.getElementById("activity-week-chart");

  function render(payload) {
    const section = host();
    const el = chart();
    if (!section || !el) return;
    const days = Array.isArray(payload?.days) ? payload.days : [];
    if (days.length !== 7) {
      section.classList.add("is-empty");
      return;
    }

    // Bars scale against the week's own best day, not a fixed ceiling —
    // the chart answers "which days did I work", so relative height is
    // the honest scale and a 3-problem Monday still shows next to a
    // 40-problem Tuesday. max(…, 1) keeps an all-zero week from
    // dividing by zero.
    const max = Math.max(1, ...days.map((d) => Number(d.count) || 0));

    el.textContent = "";
    days.forEach((day, i) => {
      const count = Number(day.count) || 0;
      const col = document.createElement("div");
      col.className = "activity-day";
      if (day.date === payload.today) col.classList.add("is-today");

      const label = document.createElement("span");
      label.className = "activity-day-count";
      // A zero is noise repeated seven times on a quiet week; the empty
      // label keeps the column's height so the bars still align.
      label.textContent = count > 0 ? String(count) : "";

      const track = document.createElement("div");
      track.className = "activity-day-track";
      const bar = document.createElement("i");
      bar.className = "activity-day-bar";
      bar.style.height = count > 0 ? `${Math.max(6, (count / max) * 100)}%` : "0";
      track.appendChild(bar);

      const letter = document.createElement("span");
      letter.className = "activity-day-letter";
      letter.textContent = DAY_LETTERS[i];

      col.append(label, track, letter);
      el.appendChild(col);
    });

    const total = Number(payload.total) || 0;
    el.setAttribute(
      "aria-label",
      `Problems answered this week, Monday to Sunday: ${days
        .map((d, i) => `${DAY_LETTERS[i]} ${Number(d.count) || 0}`)
        .join(", ")}. Total ${total}.`
    );
    section.classList.remove("is-empty");
  }

  async function refresh() {
    const section = host();
    if (!section) return;
    // apiFetch is app.js's top-level const (a global lexical binding, not
    // a window property in every build) — same guarded read the other
    // practice modules use.
    const _fetch = typeof apiFetch === "function" ? apiFetch : window.apiFetch;
    if (typeof _fetch !== "function") {
      section.classList.add("is-empty");
      return;
    }
    const seq = ++fetchSeq;
    try {
      const tz = new Date().getTimezoneOffset();
      const res = await _fetch(`/api/practice/activity-week?tz_offset=${tz}`);
      if (!res?.ok) throw new Error(`activity-week ${res?.status}`);
      const data = await res.json();
      if (seq !== fetchSeq) return; // a newer refresh owns the chart now
      render(data);
    } catch {
      if (seq !== fetchSeq) return;
      section.classList.add("is-empty");
    }
  }

  function refreshSoon() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(refresh, 900);
  }

  window.addEventListener("delta:practice-mode-ready", refresh);
  window.addEventListener("delta:practice-state-changed", refreshSoon);
  // mode-ready may already have fired before this script attached its
  // listener (script order is not a contract here); one direct call
  // covers that, and the signed-out guard inside makes it harmless.
  refresh();

  window.ActivityChart = { refresh };
})();
