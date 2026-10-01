/* ================================================================
   XP GROUP VIEW — the Learner Home's bars, and this week's leaderboard
   under them.

   Seth, 2026-09-29: "instead of displaying multiple people, it just
   displays your graph only … instead there is a leaderboard … per week
   with the ranking of the different people and their progress for that
   week … the person that's done the most … has a filled-up bar all the way,
   and then everyone else is scaled against that … the number one person is
   in first place". Seth, 2026-09-30: "the leaderboard is across the board
   for everyone, rather than being scoped to groups". So:

     your bars           always `ctx.solo()` (./xp-panel.js, with the
                         draggable target of ./xp-target.js)
     This week · Everyone  one row per learner with XP this week (and
                         you, even at 0), most XP first: rank, name, a bar
                         scaled to the leader's XP, the number. Ties share
                         a rank (1, 1, 3).

   The week is the calendar week, Monday through today, cut at the VIEWER's
   midnight (app/global_xp.py). A name shows only where you share a study
   group with that learner, else "Learner"; the server never sends an
   email or id.

   The board (GET /api/practice/leaderboard/xp) is read on boot and again
   each time the Learner Home is shown. Your own row reads the live summary,
   never the board's copy, so it moves the moment you answer.
   🪦 The group-scoped board (GET /groups/xp, 2026-09-27..29) is unread here.
   ================================================================ */
(function () {
  "use strict";

  const { el, fmt, addDays, shortDate, dateOf } = window.DDXpCharts.util;
  let roster; // undefined = not read yet, null = none (signed out / failed first read), else {week_start, rows}

  const fetcher = () => (typeof apiFetch === "function" ? apiFetch : window.apiFetch);
  const signedIn = () => window.DDIdentity?.isSignedIn?.() === true;
  const repaint = () => window.DDXpPanel?.paintGraphs?.();

  let seq = 0;
  async function load() {
    const mine = ++seq;
    const _fetch = fetcher();
    if (!signedIn() || typeof _fetch !== "function") {
      const had = !!roster;
      roster = null;
      if (had) repaint();
      return;
    }
    try {
      const res = await _fetch(`/api/practice/leaderboard/xp?${window.DeltaXP.tzQuery()}`);
      if (!res?.ok) throw new Error(`leaderboard/xp ${res?.status}`);
      const data = await res.json();
      if (mine !== seq) return;
      roster = Array.isArray(data?.rows) ? data : null;
    } catch (err) {
      // A failed read keeps whatever was drawn; the next arrival tries again.
      console.warn("[xp-group-view]", err);
      return;
    }
    repaint();
  }

  // ── the leaderboard ────────────────────────────────────────────
  /** Monday of the week `iso` falls in. */
  const monday = (iso) => addDays(iso, -((dateOf(iso).getDay() + 6) % 7));
  const weekXp = (xp, from) =>
    xp && Array.isArray(xp.days) ? xp.days.reduce((a, d) => a + (d.date >= from ? d.xp || 0 : 0), 0) : null;

  /** Everyone's week, most XP first. `you` is the live summary. */
  function board(you) {
    const today = you.today.date;
    const from = monday(today);
    // A board read last week (the page left open over Monday) shows nobody's
    // old numbers as this week's: only your live row until the next read.
    const fresh = roster.week_start === from;
    const rows = roster.rows
      .filter((r) => fresh || r.is_you)
      .map((r) => ({ ...r, week: r.is_you ? weekXp(you, from) : r.xp }));
    rows.sort((a, b) => (b.week ?? -1) - (a.week ?? -1) || Number(b.is_you) - Number(a.is_you));
    const lead = Math.max(0, ...rows.map((r) => r.week || 0));

    const box = el("section", "xp-board");
    box.setAttribute("aria-label", "This week's leaderboard, everyone");
    const head = el("div", "xp-board-head");
    head.append(
      el("h3", "xp-board-title", "This week · Everyone"),
      el("p", "xp-board-note", from === today ? shortDate(today) : `${shortDate(from)} – today`),
    );
    const list = el("ol", "xp-board-list");
    let rank = 0;
    let prev = null;
    rows.forEach((r, i) => {
      // Competition ranking: equal XP, equal place.
      if (r.week !== prev) rank = i + 1;
      prev = r.week;
      const li = el("li", `xp-board-row${r.is_you ? " is-you" : ""}${rank === 1 && r.week > 0 ? " is-first" : ""}`);
      const name = el("span", "xp-board-name", r.display_name);
      if (r.is_you) name.appendChild(el("span", "xp-you-tag", "you"));
      const bar = el("span", "xp-board-bar");
      const fill = el("span", "xp-board-fill");
      fill.style.width = `${lead > 0 && r.week > 0 ? Math.max(1.5, (r.week / lead) * 100).toFixed(1) : 0}%`;
      bar.appendChild(fill);
      li.append(
        el("span", "xp-board-rank", r.week == null ? "–" : String(rank)),
        name,
        bar,
        el("span", "xp-board-xp", r.week == null ? "—" : `${fmt(r.week)} XP`),
      );
      if (r.week == null) li.title = "Could not be read just now.";
      list.appendChild(li);
    });
    box.append(head, list);
    return box;
  }

  /** Draw the column into `right`: your bars, and the week's leaderboard
      under them once it has been read. */
  function paint(right, summary, ctx) {
    const mine = ctx.solo();
    if (!roster) right.replaceChildren(...mine);
    else right.replaceChildren(...mine, board(summary));
    return true;
  }

  // Read the roster again whenever the Learner Home comes into view.
  const watchArrival = () => {
    const page = document.getElementById("page-practice");
    if (!page || typeof MutationObserver !== "function") return;
    let shown = !page.classList.contains("hidden");
    new MutationObserver(() => {
      const now = !page.classList.contains("hidden");
      if (now && !shown) load();
      shown = now;
    }).observe(page, { attributes: true, attributeFilter: ["class"] });
  };
  // A different account (or none) must not see the last one's board — its
  // "you" row would be wrong — even for the length of a request (codex,
  // 2026-09-27): drop it and repaint first.
  window.addEventListener("delta:auth-state-changed", () => {
    const had = !!roster;
    roster = undefined;
    if (had) repaint();
    load();
  });

  const boot = () => {
    watchArrival();
    load();
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();

  window.DDXpGroupView = { paint, reload: load };
})();
