/* ================================================================
   XP TARGET — the learner's daily XP target, set two ways, and today's
   progress toward it.

   Seth, 2026-09-29: "you can actually interact with the graph by changing
   your daily threshold goal … a button on the far right of the yellow line
   where you can drag the yellow line up and down and as you change it it's
   automatically updating the date at which you complete the course.
   Additionally … change the date at which you complete the course by
   clicking on it … the day of the month … swap in between the different
   months. And whenever you click on it, it automatically adjusts your daily
   target." And: "in between the graph and the percentage completion … the
   bar that displays the completion towards your learning target".

     goal(s)        the block between the course's % and the bars: today's
                    XP toward the target as a bar, and the finish date as a
                    button that opens the browser's date picker (months to
                    page through, a day to click). Picking a day saves a DATE
                    target; the server spreads what is left over the days
                    through it (learning_xp.daily_target), so the line moves.
     wire(box, s)   the knob at the right end of the chart's yellow line
                    (./xp-charts.js `drag: true`). Dragging moves the line and
                    re-reads the finish date as it goes; letting go saves a
                    DAILY target. Keyboard: ↑/↓ 1 XP, PageUp/PageDown 10.

   One conversion both ways, the server's own: a date D is `need` XP a day
   with need = ceil(remaining_open / days through D), so a daily D finishes
   ceil(remaining_open / D) days from today, today counted.

   No target yet: the line is drawn faint at a suggestion (your pace, or 20)
   and nothing is saved until you move it or pick a date.

   Problems a day beside the XP (2026-10-01): a learner read "20 XP a day"
   next to "+15 XP / problem" as two problems a day, but XP per problem runs
   from ~0 to ~20 as a concept fills. The server counts the problems the
   remainder takes by the same model (`problems_remaining`,
   app/learning_pace.py); a daily D is ≈ D × problems_remaining / remaining
   problems a day — both read NOW (not remaining_open), so today's learning
   shrinks them together.

   And the time (2026-10-02): a LeetCode problem takes far longer than an
   ARENA drill, so the same problems a day is a different day. The server's
   `minutes_remaining` (assumed minutes per problem by course and difficulty)
   prices it the same way. Both divide `remaining_in_reach`: concepts whose
   bank has no problem near the learner are in neither count.
   ================================================================ */
(function () {
  "use strict";

  const { el, fmt, longDate, addDays, daysBetween } = window.DDXpCharts.util;
  const MAX_DAILY = 2000; // learning_xp.set_target's bound

  const remaining = (s) => Math.max(0, s.remaining_open ?? s.remaining ?? 0);
  /** The last day of a finish at `daily` XP a day. */
  const finishFor = (s, daily) => {
    const rem = remaining(s);
    if (!(daily > 0) || rem <= 0) return null;
    return addDays(s.today.date, Math.ceil(rem / daily) - 1);
  };
  /** Problems and minutes a day that `daily` XP a day stands for, or null. */
  const problemsFor = (s, daily) => {
    const rem = Math.max(0, s.remaining_in_reach ?? s.remaining ?? 0);
    if (!(daily > 0) || rem <= 0 || !(s.problems_remaining > 0)) return null;
    const n = Math.max(1, Math.round(Math.min(1, daily / rem) * s.problems_remaining));
    // Time = the shown count × the remainder's mean minutes a problem, so the
    // two never disagree (codex: "≈ 1 problem (~5 min)" on a 25-minute course).
    return { n, min: s.minutes_remaining > 0 ? (n * s.minutes_remaining) / s.problems_remaining : null };
  };
  const timeText = (m) => (m == null ? "" : m < 60 ? `${Math.max(5, Math.round(m / 5) * 5)} min`
    : `${Math.round(m / 30) / 2} h`);
  const problemsText = (p) => {
    if (!p) return "";
    const t = timeText(p.min);
    return ` · ≈ ${p.n} problem${p.n === 1 ? "" : "s"}${t ? ` (~${t})` : ""} a day`;
  };
  /** XP a day to finish by `date` (learning_xp.daily_target). */
  const dailyFor = (s, date) => Math.ceil(remaining(s) / Math.max(1, daysBetween(s.today.date, date) + 1));

  /** What the home draws: {daily, finish, set}. `set` false = a suggestion. */
  function current(s) {
    const t = s.target;
    if (t?.mode === "daily" && t.daily > 0) return { daily: t.daily, finish: finishFor(s, t.daily), set: true };
    if (t?.mode === "date" && s.today.target > 0 && t.date >= s.today.date) {
      return { daily: s.today.target, finish: t.date, set: true };
    }
    const daily = s.pace > 0 ? Math.max(5, Math.round(s.pace / 5) * 5) : 20;
    return { daily, finish: finishFor(s, daily), set: false };
  }

  // The goal block on the page, so a drag on the chart can rewrite it.
  let goalEl = null;

  function paintGoal(s, t) {
    if (!goalEl) return;
    const done = s.today.xp || 0;
    const frac = t.set && t.daily > 0 ? Math.min(1, done / t.daily) : 0;
    goalEl.classList.toggle("is-unset", !t.set);
    goalEl.classList.toggle("is-met", t.set && done >= t.daily);
    goalEl.querySelector(".xp-goal-fill").style.width = `${(frac * 100).toFixed(1)}%`;
    goalEl.querySelector(".xp-goal-bar").setAttribute("aria-valuenow", String(Math.round(frac * 100)));
    goalEl.querySelector(".xp-goal-nums").textContent = t.set
      ? `${fmt(done)} of ${fmt(t.daily)} XP today${problemsText(problemsFor(s, t.daily))}`
      : `${fmt(done)} XP today · no daily target yet`;
    goalEl.querySelector(".xp-goal-pct").textContent = t.set ? `${Math.round(frac * 100)}%` : "";
    // A finished course has no date line (goal() below).
    const btn = goalEl.querySelector(".xp-goal-date");
    if (!btn) return;
    btn.textContent = t.finish ? longDate(t.finish) : "Pick a date";
    goalEl.querySelector(".xp-goal-lead").textContent = t.set ? "Finish the course by " : "Pick a finish date: ";
    const input = goalEl.querySelector("input[type=date]");
    if (t.finish) input.value = t.finish;
  }

  const say = (text, error = false) => {
    const note = goalEl?.querySelector(".xp-goal-note");
    if (!note) return;
    note.textContent = text;
    note.classList.toggle("is-error", error);
  };
  const HINT = "Drag the dot on the yellow line to change your daily target.";

  // One save at a time, in the order they were made: a quick drag and then a
  // date pick must land in that order, or the older reply repaints last.
  let queue = Promise.resolve();
  async function save(body) {
    say("Saving…");
    const mine = queue.then(() => window.DeltaXP.setTarget(body)); // repaints the home from the reply
    queue = mine.catch(() => {});
    try {
      await mine;
    } catch (err) {
      say(err.message || "The target could not be saved.", true);
      return false;
    }
    return true;
  }

  /** The block between the course's % and the bars. */
  function goal(s) {
    const t = current(s);
    const box = el("section", "xp-goal");
    box.setAttribute("aria-label", "Today's target");

    const head = el("div", "xp-goal-head");
    head.append(el("span", "xp-goal-nums"), el("span", "xp-goal-pct"));
    const bar = el("div", "xp-goal-bar");
    bar.setAttribute("role", "progressbar");
    bar.setAttribute("aria-valuemin", "0");
    bar.setAttribute("aria-valuemax", "100");
    bar.setAttribute("aria-label", "Today's XP toward your daily target");
    bar.appendChild(el("div", "xp-goal-fill"));

    // The finish date: a button over a hidden native date input, whose
    // picker (months, then a day) is the one Seth asked for.
    const finish = el("p", "xp-goal-finish");
    const pick = el("span", "xp-goal-pick");
    const btn = el("button", "xp-goal-date");
    btn.type = "button";
    btn.title = "Pick the day you want to finish; your daily target is worked out from it";
    const input = el("input", "xp-goal-input");
    input.type = "date";
    input.min = s.today.date;
    input.tabIndex = -1;
    input.setAttribute("aria-hidden", "true");
    btn.addEventListener("click", () => {
      try {
        if (typeof input.showPicker === "function") input.showPicker();
        else input.click();
      } catch (_) {
        input.focus();
      }
    });
    input.addEventListener("change", async () => {
      const date = input.value;
      if (!date || date < s.today.date) return;
      const was = current(s);
      if (was.set && date === was.finish && s.target?.mode === "date") return;
      // Show the new numbers at once; the reply repaints them for real.
      paintGoal(s, { daily: dailyFor(s, date), finish: date, set: true });
      if (!(await save({ mode: "date", date }))) paintGoal(s, was);
    });
    pick.append(btn, input);
    finish.append(el("span", "xp-goal-lead"), pick);

    const note = el("p", "xp-goal-note", HINT);
    note.setAttribute("aria-live", "polite");
    box.append(head, bar, finish, note);
    goalEl = box;
    paintGoal(s, t);
    if (remaining(s) <= 0) {
      finish.replaceChildren(el("span", "xp-goal-lead", "Course complete."));
      note.textContent = "";
    }
    return box;
  }

  /** Make the chart's yellow line draggable by its knob. */
  function wire(box, s) {
    const p = box.xpTarget;
    if (!p || !window.DeltaXP?.setTarget || remaining(s) <= 0) return box;
    const start = current(s);
    let value = start.daily;
    const { knob, line, label } = p;
    knob.setAttribute("tabindex", "0");
    knob.setAttribute("role", "slider");
    knob.setAttribute("aria-label", "Daily XP target");
    knob.setAttribute("aria-valuemin", "1");
    knob.setAttribute("aria-valuemax", String(Math.min(MAX_DAILY, Math.floor(p.top))));

    const place = (v) => {
      value = Math.max(1, Math.min(MAX_DAILY, Math.floor(p.top), Math.round(v)));
      const yy = p.y(value);
      line.setAttribute("y1", yy);
      line.setAttribute("y2", yy);
      label.setAttribute("y", yy - 6);
      label.textContent = `target ${fmt(value)}`;
      knob.setAttribute("transform", `translate(${p.R - 4} ${yy})`);
      knob.setAttribute("aria-valuenow", String(value));
      const f = finishFor(s, value);
      const n = problemsFor(s, value);
      knob.setAttribute("aria-valuetext",
        `${value} XP a day${n ? `, about ${n.n} problems${n.min ? `, about ${timeText(n.min)}` : ""}` : ""}${f ? `, finish ${longDate(f)}` : ""}`);
      p.svg.classList.remove("is-target-unset");
      paintGoal(s, { daily: value, finish: f, set: true });
    };
    // Seat the knob, then put back what the goal block said: a date target's
    // own date, which the daily number re-read forward can land a day short of.
    place(value);
    p.svg.classList.toggle("is-target-unset", !start.set);
    paintGoal(s, start);

    // Only a moved line saves: a click on the knob leaves a suggestion
    // unsaved and a date target a date target.
    let moved = false;
    const commit = async () => {
      if (!moved) return;
      moved = false;
      if (start.set && s.target?.mode === "daily" && value === start.daily) return;
      if (!(await save({ mode: "daily", daily: value }))) {
        place(start.daily);
        p.svg.classList.toggle("is-target-unset", !start.set);
        paintGoal(s, start);
      }
    };

    /** A pointer's y in the SVG's own units (it is drawn at its real width,
        but a phone may still scale it). */
    const svgY = (clientY) => {
      const r = p.svg.getBoundingClientRect();
      return ((clientY - r.top) / Math.max(1, r.height)) * p.svg.viewBox.baseVal.height;
    };
    let dragging = false;
    knob.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      e.preventDefault();
      dragging = true;
      knob.setPointerCapture(e.pointerId);
      p.svg.classList.add("is-dragging");
    });
    knob.addEventListener("pointermove", (e) => {
      if (!dragging) return;
      const before = value;
      place(p.valueAt(svgY(e.clientY)));
      if (value !== before) moved = true;
    });
    const end = (e) => {
      if (!dragging) return;
      dragging = false;
      p.svg.classList.remove("is-dragging");
      try { knob.releasePointerCapture(e.pointerId); } catch (_) {}
      commit();
    };
    knob.addEventListener("pointerup", end);
    knob.addEventListener("pointercancel", end);

    let keyTimer = null;
    knob.addEventListener("keydown", (e) => {
      const step = { ArrowUp: 1, ArrowRight: 1, ArrowDown: -1, ArrowLeft: -1, PageUp: 10, PageDown: -10 }[e.key];
      if (!step) return;
      e.preventDefault();
      const before = value;
      place(value + step);
      if (value !== before) moved = true;
      clearTimeout(keyTimer);
      keyTimer = setTimeout(commit, 700);
    });
    return box;
  }

  window.DDXpTarget = { goal, wire, current, finishFor, dailyFor, problemsFor };
})();
