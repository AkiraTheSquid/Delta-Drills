/* ================================================================
   PLACEMENT RESULTS — what the test measured, and where it left you.

   The results card used to say "Placement results and ARENA curriculum
   progress will appear here." It said that to someone who had ALREADY
   finished the test, which made the whole surface look broken — and it was
   a lie by omission, because /api/practice/diagnostic/status has returned
   the real numbers the entire time:

       areas: [{ topic, theta, sd, probes }]   theta = 100 × P(known), folded
                                               from the per-concept rows
       kcs: [{ kc, title, lesson, topic, p, state, probes, arena_probes,
               arena_linked }]                 one row per enabled concept
       fast_track: [kc]                        settled with one probe each
       edge_violations: [{ prereq, dependent, p_prereq, p_dependent }]
       atoms_seeded: <int>                     concepts seeded at finish()
       plan: { hours, budget_secs, spent_secs, … }
       completed_at, self_reported_level

   Nothing rendered them. This file does, and it is the ONLY writer of the
   card's body — diagnostic-page.js decides WHETHER the card shows, this
   decides what is in it.

   🔴 THE HEADLINE FIGURE IS NOT COMPUTED HERE ANY MORE
     It used to be the mean of `readiness(theta)` over the nine areas — a
     second, independent answer to the same question the Practice tab's idle
     dial answers, and the two never matched. `diagnostic.py` seeds a finished
     placement at SEED_MASTERY_CAP = 0.92, under the 0.95 the dial counted as
     mastered, so this card said "45% ready for ARENA" and the next screen the
     learner saw said 0%. Seth, 2026-08-23: "it needs to be the same".

     So the figure comes from practice/readiness.js — the same call, the same
     concepts, the same words underneath — and it is handed the placement's
     `completed_at` as its stamp so the server's per-concept report is
     re-fetched once after the seeding runs, rather than being read from a
     cache filled before the test finished.

     The theta→readiness map below did NOT go away with it. It is still the
     only thing that can score the nine AREAS, which are what this test
     actually estimates and the one thing the shared figure cannot show.

   THE HONESTY RULE THIS FILE EXISTS TO KEEP
     A placement built from two probes knows almost nothing about seven of
     its nine areas, and the backend still returns a theta for every one of
     them (the untouched ones are the prior, propagated). Printing those as
     bare percentages would invent confidence the test never earned. So an
     area with `probes === 0` is labelled "not probed" and dimmed, every
     area carries its own ± band from `sd`, and the summary says how many
     of the areas were actually probed. A number the learner can't trust is
     worse than no number, because they will plan around it.

   SHORT ON PURPOSE
     Seth, 2026-08-23: get rid of "the information overload of text for the
     placement diagnostic". Every sentence that used to explain the figure is
     now either a chip (three or four words, scannable) or gone. What survived
     is the part that changes what the learner does: which areas are weakest,
     and which of them the test never actually looked at.
   ================================================================ */

const PlacementResults = (() => {
  const byId = (id) => document.getElementById(id);

  /* Mirrors of app/diagnostic.py. finish() seeds per-atom BKT mastery through
     exactly this map (`_mastery_from_theta`), so these four numbers are the
     difference between reporting the learner's placement and reporting a
     figure nothing else in the system agrees with. practice/watch.py parses
     BOTH files and fails if they ever drift apart. */
  const SEED_MASTERY_FLOOR = 0.02; // diagnostic.py SEED_MASTERY_FLOOR
  const SEED_MASTERY_CAP = 0.92;   // diagnostic.py SEED_MASTERY_CAP
  /* 2026-09-07: `theta` is 100 × P(known) — the topic-θ difficulty scale and
     its _DIFF_FLOOR/_DIFF_SPAN map retired with the per-concept model, and
     the seeding is `_mastery_from_p`, a clip of P to the range above. ALEKS's
     cut-offs (placement_model.py IN_STATE / OUT_OF_STATE) sort the rows. */
  const KNOWN_P = 0.8;
  const UNKNOWN_P = 0.2;

  const readiness = (theta) => {
    const m = Number(theta) / 100;
    if (!Number.isFinite(m)) return SEED_MASTERY_FLOOR;
    return Math.max(SEED_MASTERY_FLOOR, Math.min(SEED_MASTERY_CAP, m));
  };
  const pct = (r) => Math.round(r * 100);

  /* `sd` is on the same 0-100 scale as theta, so it converts to readiness
     points by the same span. Capped at 50 because a band wider than that is
     "we do not know" and a ±78 reads as precision about the uncertainty. */
  const band = (sd) => {
    const b = Math.round(Number(sd));
    return Number.isFinite(b) ? Math.max(1, Math.min(50, b)) : null;
  };

  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };

  /* Day and month, and the year only when it is not this one — a chip is two
     or three words and "2026" is a word that says nothing 51 weeks of the
     year. It says everything on the 52nd, when the card is reporting a
     placement the learner took last winter. */
  const completedOn = (iso) => {
    if (!iso) return null;
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return null;
    const opts = { day: "numeric", month: "short" };
    if (d.getFullYear() !== new Date().getFullYear()) opts.year = "numeric";
    return d.toLocaleDateString(undefined, opts);
  };

  /* ---- the three blocks -------------------------------------------- */

  /* Was a sentence of four clauses joined by "·". Now chips: the same four
     facts, each two or three words, read at a glance instead of parsed. */
  const renderMeta = (status, areas, probedCount) => {
    const host = byId("placement-results-meta");
    if (!host) return;
    host.textContent = "";
    const chip = (text) => host.appendChild(el("span", "placement-chip", text));
    const on = completedOn(status.completed_at);
    if (on) chip(on);
    chip(`${Number(status.probes_done) || 0} problems`);
    const spent = Number(status.plan?.spent_secs);
    if (Number.isFinite(spent) && spent > 0) {
      const h = Math.floor(spent / 3600);
      const m = Math.floor((spent % 3600) / 60);
      chip(h > 0 ? `${h}h ${m}m` : m > 0 ? `${m} min` : "under a minute");
    }
    const kcs = Array.isArray(status.kcs) ? status.kcs : [];
    if (kcs.length) {
      chip(`${kcs.filter((k) => (Number(k.probes) || 0) > 0).length}/${kcs.length} concepts probed`);
    } else if (areas.length) {
      chip(`${probedCount}/${areas.length} areas probed`);
    }
    const seeded = Number(status.atoms_seeded);
    if (Number.isFinite(seeded)) chip(`${seeded} concepts seeded`);
  };

  /* The headline. Same dial, same number and same caption as the Practice
     tab's idle screen (practice/session-idle.js) — the point is that a
     learner reading this card and then opening Practice sees one figure, not
     two. Drawn here rather than shared as markup because the two live on
     different pages; the class names are the shared part, and they are styled
     once in styles/practice/readiness.css. */
  const renderOverall = (status, areas, probedCount) => {
    const host = byId("placement-readiness");
    if (!host) return;
    host.textContent = "";

    const figure = el("div", "placement-overall");
    const dial = el("div", "readiness-dial readiness-dial--sm readiness-dial--unknown");
    dial.setAttribute("role", "img");
    /* Labelled before it has a number, or a screen reader meets an image with
       no name at all for as long as the read takes. Replaced with the figure
       itself once there is one. */
    dial.setAttribute("aria-label", "Readiness for the ARENA curriculum");
    const value = el("span", "readiness-pct", "—");
    dial.appendChild(value);

    const say = el("div", "placement-overall-say");
    const caption = el("p", "placement-overall-caption", "ready for the ARENA curriculum");
    const detail = el("p", "placement-overall-detail", "reading your concept map…");
    say.appendChild(caption);
    say.appendChild(detail);

    /* The only caveat left on the card, and it is the one that changes a
       decision: an area the test never probed is the prior wearing a
       percentage. One clause, not the paragraph that was here. */
    const unprobed = areas.length - probedCount;
    if (unprobed > 0) {
      say.appendChild(
        el(
          "p",
          "placement-overall-caveat",
          `${unprobed} of ${areas.length} areas carried from your other answers, not probed directly.`,
        ),
      );
    }

    figure.appendChild(dial);
    figure.appendChild(say);
    host.appendChild(figure);

    /* 🔴 ONE READER, and it is asynchronous — the concept map is 63 lookups
       behind a registry fetch and a lattice refresh. The card paints its shape
       first and fills the number in, rather than holding the whole results
       screen back on a network call.

       `stamp` is the placement's completion time: it forces exactly one fresh
       lattice fetch after finish() seeded the concepts, so this card cannot
       report the level the learner had BEFORE the test they just took. Every
       later re-render (the status refresh fires on each practice state change)
       reuses that same fetch. */
    window.PracticeReadiness?.read({ stamp: status.completed_at })
      .then((info) => {
        if (!info) return; // unknown, not zero — the "—" already says so
        dial.classList.remove("readiness-dial--unknown");
        dial.style.setProperty("--dd-ready-pct", String(info.pct));
        dial.setAttribute("aria-label", `${info.pct} percent ready for the ARENA curriculum`);
        value.textContent = `${info.pct}%`;
        detail.textContent = window.PracticeReadiness.detail(info);
      })
      .catch(() => {
        detail.textContent = "readiness unavailable right now";
      });
  };

  /* 🔴 CALLED ON EVERY STATUS READ, not only on a finished placement. The block
     it fills moved out of #diagnostic-results and onto the Learner Home's idle
     surface on 2026-08-24 — same element, same id, same writer — because the
     card it used to live in shows only after the test is complete, and the
     screen a learner opens every day was naming nothing at all. Seth: "it should
     display the information about einops, numpy, and einsum to be learned".

     `#learner-areas` is the section around it; `.is-empty` takes the whole
     bordered box off the screen when there is nothing honest to draw, which is
     not the same as drawing an empty box. A signed-out visitor and a failed
     status call both land there. */
  const renderAreas = (areas) => {
    const host = byId("placement-areas");
    const section = byId("learner-areas");
    if (!host) return;
    host.textContent = "";
    section?.classList.toggle("is-empty", !areas.length);
    if (!areas.length) return;

    /* Deliberately NOT "this is the order practice will work through".
       prioritization.py is weakest-first, but over SUBTOPICS, weighted, and
       scaled by how reachable the easiest available question is — so the real
       order will not match this list row for row. Promising that it would is
       a claim the app then breaks in front of the learner. Two words now: the
       sentence that used to hedge all of that was longer than the list. */
    host.appendChild(el("div", "placement-areas-head", "Weakest first"));

    // Weakest first: the card answers "what do I still need", so the thing
    // the learner needs most has to be the first row, not an alphabetical
    // accident. Ties break toward the area with more evidence behind it.
    const rows = areas
      .slice()
      .sort((a, b) => readiness(a.theta) - readiness(b.theta) || (b.probes || 0) - (a.probes || 0));

    rows.forEach((area) => {
      const probes = Number(area.probes) || 0;
      const r = readiness(area.theta);
      const row = el("div", `placement-area${probes ? "" : " placement-area--unprobed"}`);
      row.appendChild(el("span", "placement-area-name", String(area.topic || "Other")));

      const bar = el("span", "placement-area-bar");
      const fill = el("i");
      fill.style.width = `${pct(r)}%`;
      bar.appendChild(fill);
      row.appendChild(bar);

      row.appendChild(el("span", "placement-area-pct", `${pct(r)}%`));
      const b = band(area.sd);
      row.appendChild(el("span", "placement-area-conf", b === null ? "" : `±${b}`));
      row.appendChild(
        el("span", "placement-area-probes", probes ? `${probes} probed` : "not probed"),
      );
      host.appendChild(row);
    });
  };

  /* ---- per-concept report ------------------------------------------- */

  /* The graph-wide result, one row per enabled concept, grouped by what the
     test concluded: known / uncertain / unknown (ALEKS's own three states).
     Weakest first inside each group. The honesty rule applies row by row: a
     concept with `probes === 0` was never asked and is marked so, and the
     ARENA rows say when the evidence was ARENA's own exercise.

     `fast_track` (settled in one problem) and `edge_violations` (a dependent
     the learner knows better than its prerequisite — the sign an edge in the
     graph is wrong, which the 6-hour plan exists to find) each get a short
     line; both are empty for most runs and then draw nothing.

     Anchored INSIDE the results card, after #placement-readiness. The host is
     minted once here because this file is the card body's only writer; the
     card itself (#diagnostic-results) stays static in index.html. */
  const STATE_LABEL = { known: "Known", uncertain: "Uncertain", unknown: "To learn" };
  const STATE_ORDER = ["unknown", "uncertain", "known"];
  const stateOf = (k) =>
    k.state || (Number(k.p) >= KNOWN_P ? "known" : Number(k.p) <= UNKNOWN_P ? "unknown" : "uncertain");

  const _kcsHost = () => {
    let host = byId("placement-kcs");
    if (host) return host;
    const after = byId("placement-readiness");
    if (!after?.parentNode) return null;
    host = el("div", "placement-kcs");
    host.id = "placement-kcs";
    after.parentNode.insertBefore(host, after.nextSibling);
    return host;
  };

  const renderKcs = (status) => {
    const host = _kcsHost();
    if (!host) return;
    host.textContent = "";  // `renderKcs(null)` = clear: no learner, no report
    const kcs = Array.isArray(status?.kcs) ? status.kcs : [];
    if (!kcs.length) return;

    STATE_ORDER.forEach((state) => {
      const rows = kcs
        .filter((k) => stateOf(k) === state)
        .sort((a, b) => Number(a.p) - Number(b.p) || (Number(b.probes) || 0) - (Number(a.probes) || 0));
      if (!rows.length) return;
      const group = el("div", `placement-kc-group placement-kc-group--${state}`);
      group.appendChild(el("div", "placement-areas-head", `${STATE_LABEL[state]} · ${rows.length}`));
      rows.forEach((k) => {
        const probes = Number(k.probes) || 0;
        const arena = Number(k.arena_probes) || 0;
        const r = readiness(Number(k.p) * 100);
        const row = el("div", `placement-area placement-kc${probes ? "" : " placement-area--unprobed"}`);
        const name = el("span", "placement-area-name", String(k.title || k.kc));
        if (k.lesson) name.title = String(k.lesson);
        row.appendChild(name);
        const bar = el("span", "placement-area-bar");
        const fill = el("i");
        fill.style.width = `${pct(r)}%`;
        bar.appendChild(fill);
        row.appendChild(bar);
        row.appendChild(el("span", "placement-area-pct", `${pct(r)}%`));
        row.appendChild(el("span", "placement-area-conf", k.arena_linked ? (arena ? "ARENA ✓" : "ARENA") : ""));
        row.appendChild(el("span", "placement-area-probes",
          probes ? `${probes} probed${arena ? ` · ${arena} ARENA` : ""}` : "not probed"));
        group.appendChild(row);
      });
      host.appendChild(group);
    });

    const byKc = Object.fromEntries(kcs.map((k) => [k.kc, k]));
    const title = (id) => String(byKc[id]?.title || id);
    /* ALEKS's fast track: the concepts the test left UNCERTAIN go first in
       practice. They are the "Uncertain" rows above, so one line with the
       count — not 41 names again. */
    const fast = Array.isArray(status.fast_track) ? status.fast_track : [];
    if (fast.length) {
      host.appendChild(el("p", "placement-kc-note",
        `${fast.length} uncertain concept${fast.length === 1 ? "" : "s"} go first in practice.`));
    }
    const edges = Array.isArray(status.edge_violations) ? status.edge_violations : [];
    if (edges.length) {
      const note = el("p", "placement-kc-note placement-kc-note--edges");
      note.appendChild(el("strong", null, "Graph edges this run disagrees with: "));
      note.appendChild(document.createTextNode(
        edges.map((e) => `${title(e.dependent)} (${pct(Number(e.p_dependent))}%) known better than its prerequisite ${title(e.prereq)} (${pct(Number(e.p_prereq))}%)`).join("; "),
      ));
      host.appendChild(note);
    }
  };

  /* ---- entry point -------------------------------------------------- */

  const render = (status) => {
    const areas = Array.isArray(status?.areas) ? status.areas : [];
    const probedCount = areas.filter((a) => (Number(a.probes) || 0) > 0).length;
    renderMeta(status || {}, areas, probedCount);
    renderOverall(status || {}, areas, probedCount);
    renderKcs(status || {});
    renderAreas(areas);

    const empty = byId("placement-results-empty");
    // A finished placement with no areas at all means the bank changed under
    // the account. Say so rather than render three empty containers.
    const kcCount = Array.isArray(status?.kcs) ? status.kcs.length : 0;
    empty?.classList.toggle("hidden", areas.length > 0 || kcCount > 0);
  };

  /* `renderAreas` is public now: diagnostic-page.js calls it on every status,
     while `render` (the meta chips and the overall figure, which ARE about one
     completed test) stays gated on `completed_at`. */
  return { render, renderAreas, renderKcs, readiness, band };
})();
window.PlacementResults = PlacementResults;
