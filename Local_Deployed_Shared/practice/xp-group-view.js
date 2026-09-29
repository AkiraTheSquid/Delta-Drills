/* ================================================================
   XP GROUP VIEW — the Learner Home's bars, in a group or not.

   Seth, 2026-09-27: joining a group changes the Learner Home — one row per
   member, you first, each chart titled with their name (2026-09-28: "rows
   instead of the grid with one row for each user"), every row on one scale
   so the cards compare by eye.

   Seth, 2026-09-29: "it doesn't have all the fancy drop downs: it just has
   the bar graph with your learning every day". 🪦 So the Time / Graph /
   Measure selects are gone, and with them "Just me", the course line and
   the problems-solved measure: in a group, a week of daily XP bars per
   member under the group's name; out of one (or before the roster has come
   back), `ctx.solo()` — your own bars, drawn by ./xp-panel.js.

   The roster (GET /api/practice/groups/xp, app/group_xp.py) is read on
   boot and again each time the Learner Home is shown: joining and leaving
   happen on the Groups tab, so arriving back here is when it changes.
   Your own card draws from the live summary, never the roster's copy.
   ================================================================ */
(function () {
  "use strict";

  const { el } = window.DDXpCharts.util;
  const PAD = 14; // a row's inner padding (styles/xp-group.css)

  let roster; // undefined = not read yet, null = in no group, else {group, members}

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
      const res = await _fetch(`/api/practice/groups/xp?${window.DeltaXP.tzQuery()}`);
      if (!res?.ok) throw new Error(`groups/xp ${res?.status}`);
      const data = await res.json();
      if (mine !== seq) return;
      roster = data?.group ? data : null;
    } catch (err) {
      // A failed read keeps whatever was drawn; the next arrival tries again.
      console.warn("[xp-group-view]", err);
      return;
    }
    repaint();
  }

  // ── the rows ───────────────────────────────────────────────────
  /** One full-width row per member, one scale. `you` is the live summary. */
  function cards(ctx, you) {
    const people = roster.members.map((m) => ({ ...m, xp: m.is_you ? you : m.xp }));
    const drawable = people.filter((m) => m.xp && Array.isArray(m.xp.days));
    const width = ctx.width - 2 * PAD;
    // The tallest bar anywhere in the week sets everybody's scale.
    const peak = Math.max(10, ...drawable.map((m) => Math.max(...m.xp.days.slice(-ctx.days).map((d) => d.xp || 0))));

    const rows = el("div", "xp-group-rows");
    people.forEach((m) => {
      const card = el("div", `xp-group-card${m.is_you ? " is-you" : ""}`);
      if (!m.xp || !Array.isArray(m.xp.days)) {
        card.append(el("h3", "xp-chart-title", m.display_name), el("p", "xp-chart-note", "Could not be read just now."));
      } else {
        const box = window.DDXpCharts.bars(m.xp, { width, count: ctx.days, peak, target: false });
        const title = box.querySelector(".xp-chart-title");
        if (title) {
          title.textContent = m.display_name;
          if (m.is_you) title.appendChild(el("span", "xp-you-tag", "you"));
        }
        card.appendChild(box);
      }
      rows.appendChild(card);
    });
    return rows;
  }

  /** Draw the column into `right`: the group's rows under its name, or out
      of a group your own bars (`ctx.solo()`). */
  function paint(right, summary, ctx) {
    if (!roster) {
      right.replaceChildren(...ctx.solo());
      return true;
    }
    const n = roster.members.length;
    right.replaceChildren(
      el("p", "xp-group-name", `${roster.group.name} · ${n} member${n === 1 ? "" : "s"}`),
      cards(ctx, summary),
    );
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
  // A different account (or none) must not see the last one's group, even
  // for the length of a request (codex, 2026-09-27): drop it and repaint first.
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
