/* ================================================================
   XP TARGET DRAG — set the finish date on "Toward the course" directly.

   Seth, 2026-09-27: drag the yellow dot to change the target date, or click
   anywhere on the graph to set the target there; the daily goal is re-planned
   for that date. A press anywhere on the plot picks the day under the
   pointer, and holding it drags. While it moves, the graph redraws with the
   target there and the readout says what it asks for a day; letting go saves
   it (a date target, replacing an XP-per-day one) and the summary that comes
   back repaints the Learner Home.

   Keyboard: focus the graph, ← / → move a day, PageUp / PageDown a week,
   Enter saves, Escape puts it back.

   The x range is pinned for the whole gesture: in All the axis ends at the
   target, and letting it follow would slide the date away from the pointer.
   ================================================================ */
(function () {
  "use strict";

  const { fmt, longDate, daysBetween, addDays } = window.DDXpCharts.util;

  /** XP a day to finish by `date` — backend learning_xp.daily_target: what
      was left at today's open, spread over the days through `date`. */
  const needFor = (s, date) =>
    Math.ceil(Math.max(0, s.remaining_open ?? s.remaining) / Math.max(1, daysBetween(s.today.date, date) + 1));

  const asTarget = (s, date) => ({
    ...s,
    target: { mode: "date", date },
    today: { ...s.today, target: needFor(s, date) },
  });

  /** Wire the chart `box` returned by DDXpCharts.trajectory(s, opts). */
  function attach(box, s, opts) {
    const plot = box.xpPlot;
    if (!plot || !window.DeltaXP?.setTarget) return box;
    const readout = box.querySelector(".xp-readout");
    const today = s.today.date;
    const first = addDays(today, 1);
    const window_ = { start: plot.start, end: plot.end };
    const saved = s.target?.mode === "date" ? s.target.date : null;
    let picked = null; // the date being previewed, null when showing the saved target
    let dragging = false;
    let busy = false;

    const rest = () => {
      if (saved && saved >= today) {
        return `Finish by ${longDate(saved)} · ${fmt(s.today.target)} XP a day. Drag the dot or click the graph to move it.`;
      }
      return "Click the graph on the day you want to finish; the daily goal is planned for it.";
    };
    const tell = (date, verb) =>
      `${verb} ${longDate(date)} · ${fmt(needFor(s, date))} XP a day for ${fmt(daysBetween(today, date) + 1)} days` +
      (s.target?.mode === "daily" ? " (replaces your XP-per-day target)" : "");

    /** A date the target can take: not before tomorrow, not past the axis. */
    const clamp = (date) => (date < first ? first : date > plot.end ? plot.end : date);
    /** The readout; a keyboard change is announced, a mouse hover is not. */
    const say = (text, { error = false, announce = false } = {}) => {
      readout.setAttribute("aria-live", announce || error ? "polite" : "off");
      readout.classList.toggle("is-error", error);
      readout.textContent = text;
    };

    /** The day under a pointer: the nearest day on the axis. */
    const dateAt = (clientX) => {
      const rect = plot.svg.getBoundingClientRect();
      const sx = ((clientX - rect.left) / Math.max(1, rect.width)) * plot.W;
      const frac = Math.min(1, Math.max(0, (sx - plot.L) / (plot.R - plot.L)));
      return clamp(addDays(plot.start, Math.round(frac * plot.span)));
    };

    /** Redraw into the same <svg>: replacing the element would drop the pointer capture. */
    const draw = (summary) => {
      const next = window.DDXpCharts.trajectory(summary, { ...opts, window: window_ });
      const svg = next.xpPlot?.svg;
      if (!svg) return;
      plot.svg.replaceChildren(...svg.childNodes);
      const legend = next.querySelector(".xp-legend");
      if (legend) box.querySelector(".xp-legend")?.replaceWith(legend);
    };

    const preview = (date, announce = false) => {
      if (date === picked) return;
      picked = date;
      box.classList.add("is-picking");
      draw(asTarget(s, date));
      say(tell(date, "Finish by") + (announce ? ". Enter sets it." : ""), { announce });
    };
    const cancel = () => {
      dragging = false;
      if (picked === null) return;
      picked = null;
      box.classList.remove("is-picking");
      draw(s);
      say(rest());
    };
    const commit = async () => {
      dragging = false;
      const date = picked;
      if (date === null || busy) return;
      if (date === saved) {
        cancel();
        return;
      }
      busy = true;
      say(`Saving ${longDate(date)}…`, { announce: true });
      try {
        // The summary it returns repaints the whole Learner Home, this graph included.
        await window.DeltaXP.setTarget({ mode: "date", date });
      } catch (err) {
        busy = false;
        cancel();
        say(String(err?.message || err), { error: true });
      }
    };

    const svg = plot.svg;
    svg.setAttribute("tabindex", "0");
    svg.setAttribute("aria-label", `${svg.getAttribute("aria-label")}. ` +
      "Click a day, or use the arrow keys and Enter, to set the date you want to finish by.");
    svg.style.touchAction = "pan-y"; // a vertical swipe still scrolls the page
    say(rest());

    svg.addEventListener("pointerdown", (e) => {
      if (busy || (e.pointerType === "mouse" && e.button !== 0)) return;
      dragging = true;
      svg.setPointerCapture?.(e.pointerId);
      preview(dateAt(e.clientX));
      e.preventDefault();
    });
    svg.addEventListener("pointermove", (e) => {
      if (busy) return;
      if (dragging) preview(dateAt(e.clientX));
      else if (e.pointerType === "mouse") say(tell(dateAt(e.clientX), "Click to finish by"));
    });
    svg.addEventListener("pointerup", () => { if (dragging) commit(); });
    svg.addEventListener("pointercancel", cancel);
    svg.addEventListener("pointerleave", () => { if (!dragging && !busy && picked === null) say(rest()); });

    svg.addEventListener("keydown", (e) => {
      if (busy) return;
      const step = { ArrowRight: 1, ArrowLeft: -1, PageUp: 7, PageDown: -7 }[e.key];
      if (step) {
        const from = picked ?? (saved && saved >= first ? saved : first);
        preview(clamp(addDays(from, step)), true);
        e.preventDefault();
      } else if (e.key === "Enter" && picked !== null) {
        commit();
        e.preventDefault();
      } else if (e.key === "Escape") cancel();
    });
    svg.addEventListener("blur", () => { if (!dragging && !busy) cancel(); });
    return box;
  }

  window.DDXpTargetDrag = { attach, needFor };
})();
